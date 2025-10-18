// SSB Query 2.1 with L32 Compression - RANDOM ACCESS VERSION
// Query: SELECT SUM(lo_revenue), d_year, p_brand1
//        FROM lineorder, date, part, supplier
//        WHERE lo_orderdate = d_datekey
//              AND lo_partkey = p_partkey
//              AND lo_suppkey = s_suppkey
//              AND p_category = 'MFGR#12'  (encoded as 12)
//              AND s_region = 'AMERICA'    (encoded as 1)
//        GROUP BY d_year, p_brand1
//
// Random Access Strategy:
// Stage 1: Scan lo_partkey, filter by p_category=12
// Stage 2: Random access lo_suppkey, filter by s_region=1
// Stage 3: Random access lo_orderdate and lo_revenue for final candidates


#include "l3_codec.hpp"
#include "ssb_l3_utils.cuh"

#include "ssb_utils.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace std;

// Hash table entry for PART table
struct PartEntry {
    uint32_t p_brand1;
    uint32_t p_category;
};

// Hash table entry for SUPPLIER table
struct SupplierEntry {
    uint32_t s_region;
};

// Hash table entry for DATE table
struct DateEntry {
    uint32_t d_year;
};

// Candidate structure for Stage 1
struct Candidate {
    int global_idx;
    uint32_t p_brand1;
};

// Group key for aggregation
struct GroupKey {
    uint32_t d_year;
    uint32_t p_brand1;

    bool operator<(const GroupKey& other) const {
        if (d_year != other.d_year) return d_year < other.d_year;
        return p_brand1 < other.p_brand1;
    }
};

// Device function for random access decompression
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
    int delta_bits = compressed_data->d_delta_bits[partition_idx];
    int64_t bit_offset_base = compressed_data->d_delta_array_bit_offsets[partition_idx];
    double theta0 = compressed_data->d_model_params[partition_idx * 4];
    double theta1 = compressed_data->d_model_params[partition_idx * 4 + 1];

    int local_idx = global_idx - start_idx;

    // Decompress value
    long long delta = 0;
    if (compressed_data->d_plain_deltas != nullptr) {
        delta = compressed_data->d_plain_deltas[global_idx];
    } else if (delta_bits > 0 && compressed_data->delta_array != nullptr) {
        int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
        delta = extractDelta_Optimized<T>(compressed_data->delta_array, bit_offset, delta_bits);
    }

    double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
    T predicted_T = static_cast<T>(__double2int_rn(predicted));
    return applyDelta(predicted_T, delta);
}

// Stage 1: Scan lo_partkey and filter by p_category
template<typename T>
__global__ void ssb_q21_stage1_scan(
    const CompressedDataL3<T>* c_partkey,
    const PartEntry* d_part_ht,
    int num_entries,
    Candidate* d_candidates,
    int* d_candidate_count) {

    __shared__ PartitionMetaOpt s_meta;
    __shared__ Candidate s_local_candidates[256];
    __shared__ int s_local_count;

    int partition_idx = blockIdx.x;
    if (partition_idx >= c_partkey->num_partitions) return;

    // Load metadata
    if (threadIdx.x == 0) {
        s_meta.start_idx = c_partkey->d_start_indices[partition_idx];
        s_meta.delta_bits = c_partkey->d_delta_bits[partition_idx];
        s_meta.bit_offset_base = c_partkey->d_delta_array_bit_offsets[partition_idx];
        s_meta.theta0 = c_partkey->d_model_params[partition_idx * 4];
        s_meta.theta1 = c_partkey->d_model_params[partition_idx * 4 + 1];
        s_meta.partition_len = c_partkey->d_end_indices[partition_idx] - s_meta.start_idx;
        s_local_count = 0;
    }
    __syncthreads();

    // Scan partition
    for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
        int global_idx = s_meta.start_idx + local_idx;
        if (global_idx >= num_entries) continue;

        // Decompress partkey
        long long delta = 0;
        if (c_partkey->d_plain_deltas != nullptr) {
            delta = c_partkey->d_plain_deltas[global_idx];
        } else if (s_meta.delta_bits > 0 && c_partkey->delta_array != nullptr) {
            int64_t bit_offset = s_meta.bit_offset_base + (int64_t)local_idx * s_meta.delta_bits;
            delta = extractDelta_Optimized<T>(c_partkey->delta_array, bit_offset, s_meta.delta_bits);
        }

        double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
        T partkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter by p_category
        if (partkey > 0 && partkey <= 800000) {
            if (d_part_ht[partkey].p_category == 12) {
                // Add to candidates
                int pos = atomicAdd(&s_local_count, 1);
                if (pos < 256) {
                    s_local_candidates[pos].global_idx = global_idx;
                    s_local_candidates[pos].p_brand1 = d_part_ht[partkey].p_brand1;
                }
            }
        }
    }
    __syncthreads();

    // Write to global memory
    if (threadIdx.x == 0) {
        int base_offset = atomicAdd(d_candidate_count, s_local_count);
        for (int i = 0; i < s_local_count && i < 256; i++) {
            d_candidates[base_offset + i] = s_local_candidates[i];
        }
    }
}

// Stage 2: Random access lo_suppkey and filter by s_region, then access orderdate/revenue
template<typename T>
__global__ void ssb_q21_stage2_random_access(
    const CompressedDataL3<T>* c_suppkey,
    const CompressedDataL3<T>* c_orderdate,
    const CompressedDataL3<T>* c_revenue,
    const SupplierEntry* d_supplier_ht,
    const DateEntry* d_date_ht,
    const Candidate* d_candidates,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_years,
    uint32_t* d_group_brands,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    Candidate cand = d_candidates[idx];
    int global_idx = cand.global_idx;
    uint32_t p_brand1 = cand.p_brand1;

    // Random access suppkey
    T suppkey = randomAccessDecompress(c_suppkey, global_idx);
    if (suppkey == 0 || suppkey > 20000) return;
    if (d_supplier_ht[suppkey].s_region != 1) return;  // s_region = 'AMERICA'

    // Random access orderdate
    T orderdate = randomAccessDecompress(c_orderdate, global_idx);
    if (orderdate >= 20000000) return;
    if (d_date_ht[orderdate].d_year == 0) return;
    uint32_t d_year = d_date_ht[orderdate].d_year;

    // Random access revenue
    T revenue = randomAccessDecompress(c_revenue, global_idx);

    // Hash-based GROUP BY
    uint32_t hash = ((d_year * 31u) + p_brand1) % max_groups;

    for (int probe = 0; probe < 32; probe++) {
        int group_idx = (hash + probe * probe) % max_groups;

        if (d_group_years[group_idx] == d_year && d_group_brands[group_idx] == p_brand1) {
            atomicAdd(&d_group_results[group_idx], (unsigned long long)revenue);
            return;
        }

        if (d_group_years[group_idx] == 0) {
            unsigned int old = atomicCAS(&d_group_years[group_idx], 0, d_year);
            if (old == 0) {
                d_group_brands[group_idx] = p_brand1;
                atomicAdd(&d_group_results[group_idx], (unsigned long long)revenue);
                return;
            } else if (old == d_year && d_group_brands[group_idx] == p_brand1) {
                atomicAdd(&d_group_results[group_idx], (unsigned long long)revenue);
                return;
            }
        }
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 2.1 with L32 Compression - RANDOM ACCESS VERSION" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT SUM(lo_revenue), d_year, p_brand1" << endl;
    cout << "       FROM lineorder, date, part, supplier" << endl;
    cout << "       WHERE lo_orderdate = d_datekey" << endl;
    cout << "             AND lo_partkey = p_partkey" << endl;
    cout << "             AND lo_suppkey = s_suppkey" << endl;
    cout << "             AND p_category = 'MFGR#12'" << endl;
    cout << "             AND s_region = 'AMERICA'" << endl;
    cout << "       GROUP BY d_year, p_brand1" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    // Load LINEORDER columns
    cout << "Loading LINEORDER columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_partkey = loadColumn<uint32_t>("lo_partkey", LO_LEN);
    vector<uint32_t> lo_suppkey = loadColumn<uint32_t>("lo_suppkey", LO_LEN);
    vector<uint32_t> lo_revenue = loadColumn<uint32_t>("lo_revenue", LO_LEN);

    // Load dimension tables
    cout << "Loading dimension tables..." << endl;
    vector<uint32_t> p_partkey = loadColumn<uint32_t>("p_partkey", 800000);
    vector<uint32_t> p_category = loadColumn<uint32_t>("p_category", 800000);
    vector<uint32_t> p_brand1 = loadColumn<uint32_t>("p_brand1", 800000);

    vector<uint32_t> s_suppkey = loadColumn<uint32_t>("s_suppkey", 20000);
    vector<uint32_t> s_region = loadColumn<uint32_t>("s_region", 20000);

    vector<uint32_t> d_datekey = loadColumn<uint32_t>("d_datekey", 2556);
    vector<uint32_t> d_year = loadColumn<uint32_t>("d_year", 2556);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded all columns in " << load_time << " seconds" << endl;
    cout << endl;

    // Build hash tables on CPU
    cout << "Building hash tables..." << endl;
    vector<PartEntry> part_ht(800001);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        part_ht[p_partkey[i]].p_brand1 = p_brand1[i];
        part_ht[p_partkey[i]].p_category = p_category[i];
    }

    vector<SupplierEntry> supplier_ht(20001);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        supplier_ht[s_suppkey[i]].s_region = s_region[i];
    }

    vector<DateEntry> date_ht(20000000);
    for (size_t i = 0; i < d_datekey.size(); i++) {
        date_ht[d_datekey[i]].d_year = d_year[i];
    }

    cout << "✓ Hash tables built" << endl;
    cout << endl;

    // Compress LINEORDER columns
    cout << "Compressing LINEORDER columns with L32..." << endl;
    
    
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataL3<uint32_t>* c_orderdate = compressData(lo_orderdate, 4096);
    CompressedDataL3<uint32_t>* c_partkey = compressData(lo_partkey, 4096);
    CompressedDataL3<uint32_t>* c_suppkey = compressData(lo_suppkey, 4096);
    CompressedDataL3<uint32_t>* c_revenue = compressData(lo_revenue, 4096);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << endl;

    // Copy hash tables to GPU
    PartEntry* d_part_ht;
    SupplierEntry* d_supplier_ht;
    DateEntry* d_date_ht;

    CUDA_CHECK(cudaMalloc(&d_part_ht, part_ht.size() * sizeof(PartEntry)));
    CUDA_CHECK(cudaMalloc(&d_supplier_ht, supplier_ht.size() * sizeof(SupplierEntry)));
    CUDA_CHECK(cudaMalloc(&d_date_ht, date_ht.size() * sizeof(DateEntry)));

    CUDA_CHECK(cudaMemcpy(d_part_ht, part_ht.data(), part_ht.size() * sizeof(PartEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_supplier_ht, supplier_ht.data(), supplier_ht.size() * sizeof(SupplierEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_date_ht, date_ht.data(), date_ht.size() * sizeof(DateEntry), cudaMemcpyHostToDevice));

    // Allocate candidates buffer
    Candidate* d_candidates;
    int* d_candidate_count;
    CUDA_CHECK(cudaMalloc(&d_candidates, LO_LEN * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));

    // Allocate group aggregation arrays
    const int MAX_GROUPS = 10000;
    unsigned long long* d_group_results;
    uint32_t* d_group_years;
    uint32_t* d_group_brands;

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_brands, MAX_GROUPS * sizeof(uint32_t)));

    cout << "** Running Random Access Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_brands, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Scan lo_partkey
        int grid_size = c_partkey->num_partitions;
        int block_size = 256;

        ssb_q21_stage1_scan<uint32_t><<<grid_size, block_size>>>(
            c_partkey->d_self,
            d_part_ht,
            LO_LEN,
            d_candidates,
            d_candidate_count
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        // Get candidate count
        int h_candidate_count;
        CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: Random access for candidates
        if (h_candidate_count > 0) {
            grid_size = (h_candidate_count + 255) / 256;
            block_size = 256;

            ssb_q21_stage2_random_access<uint32_t><<<grid_size, block_size>>>(
                c_suppkey->d_self,
                c_orderdate->d_self,
                c_revenue->d_self,
                d_supplier_ht,
                d_date_ht,
                d_candidates,
                h_candidate_count,
                d_group_results,
                d_group_years,
                d_group_brands,
                MAX_GROUPS
            );

            CUDA_CHECK(cudaDeviceSynchronize());
        }

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        // Copy results back
        vector<unsigned long long> h_group_results(MAX_GROUPS);
        vector<uint32_t> h_group_years(MAX_GROUPS);
        vector<uint32_t> h_group_brands(MAX_GROUPS);

        CUDA_CHECK(cudaMemcpy(h_group_results.data(), d_group_results, MAX_GROUPS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_years.data(), d_group_years, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_brands.data(), d_group_brands, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Sort and display results
        map<GroupKey, unsigned long long> results_map;
        for (int i = 0; i < MAX_GROUPS; i++) {
            if (h_group_years[i] != 0) {
                GroupKey key{h_group_years[i], h_group_brands[i]};
                results_map[key] += h_group_results[i];
            }
        }

        double selectivity = 100.0 * h_candidate_count / LO_LEN;
        cout << "{\"query\":21,\"time_query\":" << query_time
             << ",\"num_groups\":" << results_map.size()
             << ",\"candidates\":" << h_candidate_count
             << ",\"selectivity\":" << selectivity << "}" << endl;

        if (t == 0) {
            cout << "Sample results (d_year, p_brand1, revenue):" << endl;
            int count = 0;
            for (const auto& [key, revenue] : results_map) {
                cout << "  " << key.d_year << ", " << key.p_brand1 << ": " << revenue << endl;
                if (++count >= 10) break;
            }
        }
    }

    cout << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_part_ht));
    CUDA_CHECK(cudaFree(d_supplier_ht));
    CUDA_CHECK(cudaFree(d_date_ht));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_group_results));
    CUDA_CHECK(cudaFree(d_group_years));
    CUDA_CHECK(cudaFree(d_group_brands));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_partkey);
    freeCompressedData(c_suppkey);
    freeCompressedData(c_revenue);

    cout << "========================================================================" << endl;

    return 0;
}
