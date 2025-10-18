// SSB Query 3.2 with L32 Compression - RANDOM ACCESS VERSION
// Query: SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
//        FROM customer, lineorder, supplier, date
//        WHERE lo_custkey = c_custkey
//              AND lo_suppkey = s_suppkey
//              AND lo_orderdate = d_datekey
//              AND c_nation = 'UNITED STATES' (encoded as 24)
//              AND s_nation = 'UNITED STATES' (encoded as 24)
//              AND d_year >= 1992 AND d_year <= 1997
//        GROUP BY c_city, s_city, d_year
//        ORDER BY d_year ASC, revenue DESC


#include "l3_codec.hpp"
#include "ssb_l3_utils.cuh"

#include "ssb_utils.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace std;

struct CustomerEntry {
    uint32_t c_city;
    uint32_t c_nation;
    uint32_t c_region;
};

struct SupplierEntry {
    uint32_t s_city;
    uint32_t s_nation;
    uint32_t s_region;
};

struct DateEntry {
    uint32_t d_year;
};

struct Candidate {
    int global_idx;
    uint32_t c_city;
    uint32_t s_city;
};

struct GroupKey {
    uint32_t c_city;
    uint32_t s_city;
    uint32_t d_year;

    bool operator<(const GroupKey& other) const {
        if (d_year != other.d_year) return d_year < other.d_year;
        if (c_city != other.c_city) return c_city < other.c_city;
        return s_city < other.s_city;
    }
};

// Random access decompression function
template<typename T>
__device__ T randomAccessDecompress(
    const CompressedDataL3<T>* compressed_data,
    int global_idx) {

    // Binary search to find partition
    int left = 0;
    int right = compressed_data->num_partitions - 1;
    int partition_idx = 0;

    while (left <= right) {
        int mid = (left + right) / 2;
        int start = compressed_data->d_start_indices[mid];
        int end = compressed_data->d_end_indices[mid];

        if (global_idx >= start && global_idx < end) {
            partition_idx = mid;
            break;
        } else if (global_idx < start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    // Load partition metadata
    int start_idx = compressed_data->d_start_indices[partition_idx];
    int local_idx = global_idx - start_idx;

    double theta0 = compressed_data->d_model_params[partition_idx * 4];
    double theta1 = compressed_data->d_model_params[partition_idx * 4 + 1];
    int delta_bits = compressed_data->d_delta_bits[partition_idx];
    int64_t bit_offset_base = compressed_data->d_delta_array_bit_offsets[partition_idx];

    // Extract delta
    long long delta = 0;
    if (compressed_data->d_plain_deltas != nullptr) {
        delta = compressed_data->d_plain_deltas[global_idx];
    } else if (delta_bits > 0 && compressed_data->delta_array != nullptr) {
        int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
        delta = extractDelta_Optimized<T>(compressed_data->delta_array, bit_offset, delta_bits);
    }

    // Decompress
    double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
    T predicted_T = static_cast<T>(__double2int_rn(predicted));
    return applyDelta(predicted_T, delta);
}

// Stage 1: Scan lo_custkey and lo_suppkey to generate candidates
template<typename T>
__global__ void ssb_q32_stage1_scan(
    const CompressedDataL3<T>* c_custkey,
    const CompressedDataL3<T>* c_suppkey,
    const CustomerEntry* d_customer_ht,
    const SupplierEntry* d_supplier_ht,
    int num_entries,
    Candidate* d_candidates,
    int* d_candidate_count) {

    __shared__ PartitionMetaOpt s_meta_custkey;
    __shared__ PartitionMetaOpt s_meta_suppkey;
    __shared__ Candidate s_local_candidates[256];
    __shared__ int s_local_count;

    int partition_idx = blockIdx.x;
    if (partition_idx >= c_custkey->num_partitions) return;

    if (threadIdx.x == 0) {
        s_local_count = 0;

        s_meta_custkey.start_idx = c_custkey->d_start_indices[partition_idx];
        s_meta_custkey.delta_bits = c_custkey->d_delta_bits[partition_idx];
        s_meta_custkey.bit_offset_base = c_custkey->d_delta_array_bit_offsets[partition_idx];
        s_meta_custkey.theta0 = c_custkey->d_model_params[partition_idx * 4];
        s_meta_custkey.theta1 = c_custkey->d_model_params[partition_idx * 4 + 1];
        s_meta_custkey.partition_len = c_custkey->d_end_indices[partition_idx] - s_meta_custkey.start_idx;

        s_meta_suppkey.start_idx = c_suppkey->d_start_indices[partition_idx];
        s_meta_suppkey.delta_bits = c_suppkey->d_delta_bits[partition_idx];
        s_meta_suppkey.bit_offset_base = c_suppkey->d_delta_array_bit_offsets[partition_idx];
        s_meta_suppkey.theta0 = c_suppkey->d_model_params[partition_idx * 4];
        s_meta_suppkey.theta1 = c_suppkey->d_model_params[partition_idx * 4 + 1];
        s_meta_suppkey.partition_len = c_suppkey->d_end_indices[partition_idx] - s_meta_suppkey.start_idx;
    }

    __syncthreads();

    for (int local_idx = threadIdx.x; local_idx < s_meta_custkey.partition_len; local_idx += blockDim.x) {
        int global_idx = s_meta_custkey.start_idx + local_idx;
        if (global_idx >= num_entries) continue;

        // Decompress lo_custkey
        long long delta = 0;
        if (c_custkey->d_plain_deltas != nullptr) {
            delta = c_custkey->d_plain_deltas[global_idx];
        } else if (s_meta_custkey.delta_bits > 0 && c_custkey->delta_array != nullptr) {
            int64_t bit_offset = s_meta_custkey.bit_offset_base + (int64_t)local_idx * s_meta_custkey.delta_bits;
            delta = extractDelta_Optimized<T>(c_custkey->delta_array, bit_offset, s_meta_custkey.delta_bits);
        }
        double predicted = fma(s_meta_custkey.theta1, static_cast<double>(local_idx), s_meta_custkey.theta0);
        T custkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter by c_nation
        if (custkey == 0 || custkey > 300000) continue;
        CustomerEntry cust = d_customer_ht[custkey];
        if (cust.c_nation != 24) continue; // c_nation = 'UNITED STATES'

        // Decompress lo_suppkey
        delta = 0;
        if (c_suppkey->d_plain_deltas != nullptr) {
            delta = c_suppkey->d_plain_deltas[global_idx];
        } else if (s_meta_suppkey.delta_bits > 0 && c_suppkey->delta_array != nullptr) {
            int64_t bit_offset = s_meta_suppkey.bit_offset_base + (int64_t)local_idx * s_meta_suppkey.delta_bits;
            delta = extractDelta_Optimized<T>(c_suppkey->delta_array, bit_offset, s_meta_suppkey.delta_bits);
        }
        predicted = fma(s_meta_suppkey.theta1, static_cast<double>(local_idx), s_meta_suppkey.theta0);
        T suppkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter by s_nation
        if (suppkey == 0 || suppkey > 20000) continue;
        SupplierEntry supp = d_supplier_ht[suppkey];
        if (supp.s_nation != 24) continue; // s_nation = 'UNITED STATES'

        // Add to local candidate buffer
        int pos = atomicAdd(&s_local_count, 1);
        if (pos < 256) {
            s_local_candidates[pos] = {global_idx, cust.c_city, supp.s_city};
        } else {
            // Flush to global memory
            __syncthreads();
            if (threadIdx.x == 0) {
                int global_pos = atomicAdd(d_candidate_count, s_local_count - 1);
                for (int i = 0; i < 256; i++) {
                    d_candidates[global_pos + i] = s_local_candidates[i];
                }
                s_local_candidates[0] = {global_idx, cust.c_city, supp.s_city};
                s_local_count = 1;
            }
            __syncthreads();
        }
    }

    __syncthreads();

    // Flush remaining candidates
    if (threadIdx.x == 0 && s_local_count > 0) {
        int global_pos = atomicAdd(d_candidate_count, s_local_count);
        for (int i = 0; i < s_local_count; i++) {
            d_candidates[global_pos + i] = s_local_candidates[i];
        }
    }
}

// Stage 2: Random access lo_orderdate and lo_revenue for candidates
template<typename T>
__global__ void ssb_q32_stage2_random_access(
    const CompressedDataL3<T>* c_orderdate,
    const CompressedDataL3<T>* c_revenue,
    const DateEntry* d_date_ht,
    const Candidate* d_candidates,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_ccities,
    uint32_t* d_group_scities,
    uint32_t* d_group_years,
    uint64_t* d_group_keys,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    Candidate cand = d_candidates[idx];
    int global_idx = cand.global_idx;

    // Random access lo_orderdate
    T orderdate = randomAccessDecompress<T>(c_orderdate, global_idx);

    // Filter by d_year
    if (orderdate >= 20000000 || d_date_ht[orderdate].d_year == 0) return;
    uint32_t d_year = d_date_ht[orderdate].d_year;
    if (d_year < 1992 || d_year > 1997) return;

    // Random access lo_revenue
    T revenue = randomAccessDecompress<T>(c_revenue, global_idx);

    // Hash-based GROUP BY
    uint64_t combined_key = ((uint64_t)cand.c_city << 32) | ((uint64_t)cand.s_city << 16) | d_year;
    uint32_t hash = (uint32_t)(combined_key % max_groups);

    int group_idx = -1;
    for (int probe = 0; probe < 64; probe++) {
        int gidx = (hash + probe * probe) % max_groups;

        uint64_t existing_key = d_group_keys[gidx];
        if (existing_key == 0) {
            uint64_t old = atomicCAS((unsigned long long*)&d_group_keys[gidx], 0ULL, combined_key);
            if (old == 0 || old == combined_key) {
                d_group_ccities[gidx] = cand.c_city;
                d_group_scities[gidx] = cand.s_city;
                d_group_years[gidx] = d_year;
                group_idx = gidx;
                break;
            }
        } else if (existing_key == combined_key) {
            group_idx = gidx;
            break;
        }
    }

    if (group_idx >= 0) {
        atomicAdd(&d_group_results[group_idx], (unsigned long long)revenue);
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 3.2 with L32 Compression - RANDOM ACCESS" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue" << endl;
    cout << "       FROM customer, lineorder, supplier, date" << endl;
    cout << "       WHERE lo_custkey = c_custkey" << endl;
    cout << "             AND lo_suppkey = s_suppkey" << endl;
    cout << "             AND lo_orderdate = d_datekey" << endl;
    cout << "             AND c_nation = 'UNITED STATES'" << endl;
    cout << "             AND s_nation = 'UNITED STATES'" << endl;
    cout << "             AND d_year >= 1992 AND d_year <= 1997" << endl;
    cout << "       GROUP BY c_city, s_city, d_year" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    cout << "Loading LINEORDER columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_custkey = loadColumn<uint32_t>("lo_custkey", LO_LEN);
    vector<uint32_t> lo_suppkey = loadColumn<uint32_t>("lo_suppkey", LO_LEN);
    vector<uint32_t> lo_revenue = loadColumn<uint32_t>("lo_revenue", LO_LEN);

    cout << "Loading dimension tables..." << endl;
    vector<uint32_t> c_custkey = loadColumn<uint32_t>("c_custkey", 300000);
    vector<uint32_t> c_city = loadColumn<uint32_t>("c_city", 300000);
    vector<uint32_t> c_nation = loadColumn<uint32_t>("c_nation", 300000);
    vector<uint32_t> c_region = loadColumn<uint32_t>("c_region", 300000);

    vector<uint32_t> s_suppkey = loadColumn<uint32_t>("s_suppkey", 20000);
    vector<uint32_t> s_city = loadColumn<uint32_t>("s_city", 20000);
    vector<uint32_t> s_nation = loadColumn<uint32_t>("s_nation", 20000);
    vector<uint32_t> s_region = loadColumn<uint32_t>("s_region", 20000);

    vector<uint32_t> d_datekey = loadColumn<uint32_t>("d_datekey", 2556);
    vector<uint32_t> d_year = loadColumn<uint32_t>("d_year", 2556);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded all columns in " << load_time << " seconds" << endl;
    cout << endl;

    cout << "Building hash tables..." << endl;
    vector<CustomerEntry> customer_ht(300001);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        customer_ht[c_custkey[i]].c_city = c_city[i];
        customer_ht[c_custkey[i]].c_nation = c_nation[i];
        customer_ht[c_custkey[i]].c_region = c_region[i];
    }

    vector<SupplierEntry> supplier_ht(20001);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        supplier_ht[s_suppkey[i]].s_city = s_city[i];
        supplier_ht[s_suppkey[i]].s_nation = s_nation[i];
        supplier_ht[s_suppkey[i]].s_region = s_region[i];
    }

    vector<DateEntry> date_ht(20000000);
    for (size_t i = 0; i < d_datekey.size(); i++) {
        date_ht[d_datekey[i]].d_year = d_year[i];
    }

    cout << "✓ Hash tables built" << endl;
    cout << endl;

    cout << "Compressing LINEORDER columns with L32..." << endl;
    
    
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataL3<uint32_t>* c_orderdate = compressData(lo_orderdate, 4096);
    CompressedDataL3<uint32_t>* c_custkey_c = compressData(lo_custkey, 4096);
    CompressedDataL3<uint32_t>* c_suppkey_c = compressData(lo_suppkey, 4096);
    CompressedDataL3<uint32_t>* c_revenue = compressData(lo_revenue, 4096);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << endl;

    CustomerEntry* d_customer_ht;
    SupplierEntry* d_supplier_ht;
    DateEntry* d_date_ht;

    CUDA_CHECK(cudaMalloc(&d_customer_ht, customer_ht.size() * sizeof(CustomerEntry)));
    CUDA_CHECK(cudaMalloc(&d_supplier_ht, supplier_ht.size() * sizeof(SupplierEntry)));
    CUDA_CHECK(cudaMalloc(&d_date_ht, date_ht.size() * sizeof(DateEntry)));

    CUDA_CHECK(cudaMemcpy(d_customer_ht, customer_ht.data(), customer_ht.size() * sizeof(CustomerEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_supplier_ht, supplier_ht.data(), supplier_ht.size() * sizeof(SupplierEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_date_ht, date_ht.data(), date_ht.size() * sizeof(DateEntry), cudaMemcpyHostToDevice));

    const int MAX_CANDIDATES = 10000000;
    Candidate* d_candidates;
    int* d_candidate_count;
    CUDA_CHECK(cudaMalloc(&d_candidates, MAX_CANDIDATES * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));

    const int MAX_GROUPS = 50000;
    unsigned long long* d_group_results;
    uint32_t* d_group_ccities;
    uint32_t* d_group_scities;
    uint32_t* d_group_years;
    uint64_t* d_group_keys;

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_ccities, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_scities, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_keys, MAX_GROUPS * sizeof(uint64_t)));

    cout << "** Running Random Access Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_ccities, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_scities, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_keys, 0, MAX_GROUPS * sizeof(uint64_t)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Scan and generate candidates
        ssb_q32_stage1_scan<uint32_t><<<c_custkey_c->num_partitions, 256>>>(
            c_custkey_c->d_self, c_suppkey_c->d_self,
            d_customer_ht, d_supplier_ht, LO_LEN,
            d_candidates, d_candidate_count
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        int h_candidate_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: Random access for candidates
        if (h_candidate_count > 0) {
            int blocks = (h_candidate_count + 255) / 256;
            ssb_q32_stage2_random_access<uint32_t><<<blocks, 256>>>(
                c_orderdate->d_self, c_revenue->d_self,
                d_date_ht, d_candidates, h_candidate_count,
                d_group_results, d_group_ccities, d_group_scities, d_group_years, d_group_keys, MAX_GROUPS
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        vector<unsigned long long> h_group_results(MAX_GROUPS);
        vector<uint32_t> h_group_ccities(MAX_GROUPS);
        vector<uint32_t> h_group_scities(MAX_GROUPS);
        vector<uint32_t> h_group_years(MAX_GROUPS);

        CUDA_CHECK(cudaMemcpy(h_group_results.data(), d_group_results, MAX_GROUPS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_ccities.data(), d_group_ccities, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_scities.data(), d_group_scities, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_years.data(), d_group_years, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        map<GroupKey, unsigned long long> results_map;
        for (int i = 0; i < MAX_GROUPS; i++) {
            if (h_group_ccities[i] != 0) {
                GroupKey key{h_group_ccities[i], h_group_scities[i], h_group_years[i]};
                results_map[key] += h_group_results[i];
            }
        }

        double selectivity = (double)h_candidate_count / LO_LEN * 100.0;
        cout << "{\"query\":32,\"time_query\":" << query_time
             << ",\"num_groups\":" << results_map.size()
             << ",\"candidates\":" << h_candidate_count
             << ",\"selectivity\":" << selectivity << "}" << endl;

        if (t == 0) {
            cout << "Sample results (c_city, s_city, d_year, revenue):" << endl;
            int count = 0;
            for (const auto& [key, revenue] : results_map) {
                cout << "  " << key.c_city << ", " << key.s_city << ", " << key.d_year << ": " << revenue << endl;
                if (++count >= 10) break;
            }
        }
    }

    cout << endl;

    CUDA_CHECK(cudaFree(d_customer_ht));
    CUDA_CHECK(cudaFree(d_supplier_ht));
    CUDA_CHECK(cudaFree(d_date_ht));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_group_results));
    CUDA_CHECK(cudaFree(d_group_ccities));
    CUDA_CHECK(cudaFree(d_group_scities));
    CUDA_CHECK(cudaFree(d_group_years));
    CUDA_CHECK(cudaFree(d_group_keys));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_custkey_c);
    freeCompressedData(c_suppkey_c);
    freeCompressedData(c_revenue);

    cout << "========================================================================" << endl;

    return 0;
}
