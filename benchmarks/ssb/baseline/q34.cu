// SSB Query 3.4 with L32 Compression - Random Access Optimized
// Query: SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
//        FROM customer, lineorder, supplier, date
//        WHERE lo_custkey = c_custkey
//              AND lo_suppkey = s_suppkey
//              AND lo_orderdate = d_datekey
//              AND c_city IN ('UNITED KI1', 'UNITED KI5') (101, 105)
//              AND s_city IN ('UNITED KI1', 'UNITED KI5') (101, 105)
//              AND d_yearmonth = 'Dec1997' (19971201, 19971202)
//        GROUP BY c_city, s_city, d_year


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

struct Candidate {
    int global_idx;
    uint32_t d_year;
};

// Random access decompression helper
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
    
    // Load metadata
    int start_idx = compressed_data->d_start_indices[partition_idx];
    int local_idx = global_idx - start_idx;
    
    double theta0 = compressed_data->d_model_params[partition_idx * 4];
    double theta1 = compressed_data->d_model_params[partition_idx * 4 + 1];
    
    // Extract delta
    long long delta = 0;
    if (compressed_data->d_plain_deltas != nullptr) {
        delta = compressed_data->d_plain_deltas[global_idx];
    } else {
        int delta_bits = compressed_data->d_delta_bits[partition_idx];
        if (delta_bits > 0 && compressed_data->delta_array != nullptr) {
            int64_t bit_offset = compressed_data->d_delta_array_bit_offsets[partition_idx] + 
                                (int64_t)local_idx * delta_bits;
            delta = extractDelta_Optimized<T>(compressed_data->delta_array, bit_offset, delta_bits);
        }
    }
    
    // Predict and apply delta
    double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
    T predicted_T = static_cast<T>(__double2int_rn(predicted));
    return applyDelta(predicted_T, delta);
}

// Stage 1: Scan orderdate with d_yearmonth filter
template<typename T>
__global__ void ssb_q34_stage1_scan(
    const CompressedDataL3<T>* c_orderdate,
    const DateEntry* d_date_ht,
    int num_entries,
    Candidate* d_candidates,
    int* d_candidate_count) {

    __shared__ PartitionMetaOpt s_meta_orderdate;
    __shared__ Candidate s_candidates[256];
    __shared__ int s_candidate_count;

    int partition_idx = blockIdx.x;
    if (partition_idx >= c_orderdate->num_partitions) return;

    if (threadIdx.x == 0) {
        s_candidate_count = 0;

        s_meta_orderdate.start_idx = c_orderdate->d_start_indices[partition_idx];
        s_meta_orderdate.delta_bits = c_orderdate->d_delta_bits[partition_idx];
        s_meta_orderdate.bit_offset_base = c_orderdate->d_delta_array_bit_offsets[partition_idx];
        s_meta_orderdate.theta0 = c_orderdate->d_model_params[partition_idx * 4];
        s_meta_orderdate.theta1 = c_orderdate->d_model_params[partition_idx * 4 + 1];
        s_meta_orderdate.partition_len = c_orderdate->d_end_indices[partition_idx] - s_meta_orderdate.start_idx;
    }

    __syncthreads();

    for (int local_idx = threadIdx.x; local_idx < s_meta_orderdate.partition_len; local_idx += blockDim.x) {
        int global_idx = s_meta_orderdate.start_idx + local_idx;
        if (global_idx >= num_entries) continue;

        long long delta;
        double predicted;

        // Decompress orderdate
        delta = 0;
        if (c_orderdate->d_plain_deltas != nullptr) {
            delta = c_orderdate->d_plain_deltas[global_idx];
        } else if (s_meta_orderdate.delta_bits > 0 && c_orderdate->delta_array != nullptr) {
            int64_t bit_offset = s_meta_orderdate.bit_offset_base + (int64_t)local_idx * s_meta_orderdate.delta_bits;
            delta = extractDelta_Optimized<T>(c_orderdate->delta_array, bit_offset, s_meta_orderdate.delta_bits);
        }
        predicted = fma(s_meta_orderdate.theta1, static_cast<double>(local_idx), s_meta_orderdate.theta0);
        T orderdate = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter: d_yearmonth = 'Dec1997' (19971201 to 19971231)
        if (orderdate < 19971201 || orderdate > 19971231) continue;
        // Extract year directly from orderdate (YYYYMMDD format)
        uint32_t d_year = orderdate / 10000;

        // Add to candidate list
        int local_pos = atomicAdd(&s_candidate_count, 1);
        if (local_pos < 256) {
            s_candidates[local_pos].global_idx = global_idx;
            s_candidates[local_pos].d_year = d_year;
        } else {
            // Flush to global memory
            if (local_pos == 256) {
                int global_offset = atomicAdd(d_candidate_count, 256);
                for (int i = 0; i < 256; i++) {
                    d_candidates[global_offset + i] = s_candidates[i];
                }
                s_candidate_count = 1;
                s_candidates[0].global_idx = global_idx;
                s_candidates[0].d_year = d_year;
            }
        }
    }

    __syncthreads();

    // Final flush
    if (threadIdx.x == 0 && s_candidate_count > 0 && s_candidate_count < 256) {
        int global_offset = atomicAdd(d_candidate_count, s_candidate_count);
        for (int i = 0; i < s_candidate_count; i++) {
            d_candidates[global_offset + i] = s_candidates[i];
        }
    }
}

// Stage 2: Random access custkey, suppkey, revenue for candidates
template<typename T>
__global__ void ssb_q34_stage2_random_access(
    const CompressedDataL3<T>* c_custkey,
    const CompressedDataL3<T>* c_suppkey,
    const CompressedDataL3<T>* c_revenue,
    const CustomerEntry* d_customer_ht,
    const SupplierEntry* d_supplier_ht,
    const Candidate* d_candidates,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_ccities,
    uint32_t* d_group_scities,
    uint32_t* d_group_years,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    Candidate cand = d_candidates[idx];
    int global_idx = cand.global_idx;
    uint32_t d_year = cand.d_year;

    // Random access custkey
    T custkey = randomAccessDecompress(c_custkey, global_idx);
    if (custkey == 0 || custkey > 300000) return;
    uint32_t c_city = d_customer_ht[custkey].c_city;
    if (c_city != 101 && c_city != 105) return;  // Early exit

    // Random access suppkey
    T suppkey = randomAccessDecompress(c_suppkey, global_idx);
    if (suppkey == 0 || suppkey > 20000) return;
    uint32_t s_city = d_supplier_ht[suppkey].s_city;
    if (s_city != 101 && s_city != 105) return;  // Early exit

    // Random access revenue
    T revenue = randomAccessDecompress(c_revenue, global_idx);

    // GROUP BY (c_city, s_city, d_year) using linear search
    int group_idx = -1;
    for (int i = 0; i < max_groups; i++) {
        if (d_group_ccities[i] == c_city &&
            d_group_scities[i] == s_city &&
            d_group_years[i] == d_year) {
            group_idx = i;
            break;
        }
        if (d_group_ccities[i] == 0) {
            unsigned int old = atomicCAS(&d_group_ccities[i], 0, c_city);
            if (old == 0 || old == c_city) {
                atomicCAS(&d_group_scities[i], 0, s_city);
                atomicCAS(&d_group_years[i], 0, d_year);
                if (d_group_scities[i] == s_city && d_group_years[i] == d_year) {
                    group_idx = i;
                    break;
                }
            }
        }
    }

    if (group_idx >= 0) {
        atomicAdd(&d_group_results[group_idx], (unsigned long long)revenue);
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 3.4 with L32 Compression - Random Access Optimized" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue" << endl;
    cout << "       FROM customer, lineorder, supplier, date" << endl;
    cout << "       WHERE lo_custkey = c_custkey" << endl;
    cout << "             AND lo_suppkey = s_suppkey" << endl;
    cout << "             AND lo_orderdate = d_datekey" << endl;
    cout << "             AND c_city IN ('UNITED KI1', 'UNITED KI5')" << endl;
    cout << "             AND s_city IN ('UNITED KI1', 'UNITED KI5')" << endl;
    cout << "             AND d_yearmonth = 'Dec1997'" << endl;
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

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_ccities, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_scities, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));

    cout << "** Running Random Access Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_ccities, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_scities, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Scan orderdate with d_yearmonth filter
        ssb_q34_stage1_scan<uint32_t><<<c_orderdate->num_partitions, 256>>>(
            c_orderdate->d_self, d_date_ht, LO_LEN,
            d_candidates, d_candidate_count
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int num_candidates;
        CUDA_CHECK(cudaMemcpy(&num_candidates, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: Random access custkey, suppkey, revenue for candidates
        if (num_candidates > 0) {
            int blocks = (num_candidates + 255) / 256;
            ssb_q34_stage2_random_access<uint32_t><<<blocks, 256>>>(
                c_custkey_c->d_self, c_suppkey_c->d_self, c_revenue->d_self,
                d_customer_ht, d_supplier_ht,
                d_candidates, num_candidates,
                d_group_results, d_group_ccities, d_group_scities, d_group_years, MAX_GROUPS
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

        double selectivity = (num_candidates * 100.0) / LO_LEN;
        cout << "{\"query\":34,\"time_query\":" << query_time 
             << ",\"num_candidates\":" << num_candidates 
             << ",\"selectivity\":" << selectivity
             << ",\"num_groups\":" << results_map.size() << "}" << endl;

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

    freeCompressedData(c_orderdate);
    freeCompressedData(c_custkey_c);
    freeCompressedData(c_suppkey_c);
    freeCompressedData(c_revenue);

    cout << "========================================================================" << endl;

    return 0;
}
