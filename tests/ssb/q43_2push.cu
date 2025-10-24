// SSB Query 4.3 with GLECO2 Compression - Random Access Optimized (_2push version)
// Query: SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit
//        FROM date, customer, supplier, part, lineorder
//        WHERE lo_custkey = c_custkey
//              AND lo_suppkey = s_suppkey
//              AND lo_partkey = p_partkey
//              AND lo_orderdate = d_datekey
//              AND s_nation = 'UNITED STATES' (24)
//              AND (d_year = 1997 OR d_year = 1998)
//              AND p_category = 'MFGR#14' (14)
//        GROUP BY d_year, s_city, p_brand1


#include "L3_codec.hpp"
#include "L3_predicate_pushdown.cuh"
#include "ssb_L3_utils.cuh"

#include "ssb_utils.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace std;

struct CustomerEntry {
    uint32_t c_nation;
    uint32_t c_region;
};

struct SupplierEntry {
    uint32_t s_nation;
    uint32_t s_city;
};

struct PartEntry {
    uint32_t p_brand1;
    uint32_t p_category;
};

struct DateEntry {
    uint32_t d_year;
};

struct GroupKey {
    uint32_t d_year;
    uint32_t s_city;
    uint32_t p_brand1;

    bool operator<(const GroupKey& other) const {
        if (d_year != other.d_year) return d_year < other.d_year;
        if (s_city != other.s_city) return s_city < other.s_city;
        return p_brand1 < other.p_brand1;
    }
};

struct Candidate {
    int global_idx;
    uint32_t d_year;
    uint32_t s_city;
    uint32_t p_brand1;
};

// Stage 1: Scan orderdate, partkey, suppkey with filters
template<typename T>
__global__ void ssb_q43_stage1_scan(
    const CompressedDataGLECO<T>* c_orderdate,
    const CompressedDataGLECO<T>* c_partkey,
    const CompressedDataGLECO<T>* c_suppkey,
    const DateEntry* d_date_ht,
    const PartEntry* d_part_ht,
    const SupplierEntry* d_supplier_ht,
    int num_entries,
    Candidate* d_candidates,
    int* d_candidate_count) {

    __shared__ PartitionMetaOpt s_meta_orderdate;
    __shared__ PartitionMetaOpt s_meta_partkey;
    __shared__ PartitionMetaOpt s_meta_suppkey;
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

        s_meta_partkey.start_idx = c_partkey->d_start_indices[partition_idx];
        s_meta_partkey.delta_bits = c_partkey->d_delta_bits[partition_idx];
        s_meta_partkey.bit_offset_base = c_partkey->d_delta_array_bit_offsets[partition_idx];
        s_meta_partkey.theta0 = c_partkey->d_model_params[partition_idx * 4];
        s_meta_partkey.theta1 = c_partkey->d_model_params[partition_idx * 4 + 1];
        s_meta_partkey.partition_len = c_partkey->d_end_indices[partition_idx] - s_meta_partkey.start_idx;

        s_meta_suppkey.start_idx = c_suppkey->d_start_indices[partition_idx];
        s_meta_suppkey.delta_bits = c_suppkey->d_delta_bits[partition_idx];
        s_meta_suppkey.bit_offset_base = c_suppkey->d_delta_array_bit_offsets[partition_idx];
        s_meta_suppkey.theta0 = c_suppkey->d_model_params[partition_idx * 4];
        s_meta_suppkey.theta1 = c_suppkey->d_model_params[partition_idx * 4 + 1];
        s_meta_suppkey.partition_len = c_suppkey->d_end_indices[partition_idx] - s_meta_suppkey.start_idx;
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

        // Filter: d_year IN (1997, 1998)
        if (orderdate >= 20000000 || d_date_ht[orderdate].d_year == 0) continue;
        uint32_t d_year = d_date_ht[orderdate].d_year;
        if (d_year != 1997 && d_year != 1998) continue;

        // Decompress partkey
        delta = 0;
        if (c_partkey->d_plain_deltas != nullptr) {
            delta = c_partkey->d_plain_deltas[global_idx];
        } else if (s_meta_partkey.delta_bits > 0 && c_partkey->delta_array != nullptr) {
            int64_t bit_offset = s_meta_partkey.bit_offset_base + (int64_t)local_idx * s_meta_partkey.delta_bits;
            delta = extractDelta_Optimized<T>(c_partkey->delta_array, bit_offset, s_meta_partkey.delta_bits);
        }
        predicted = fma(s_meta_partkey.theta1, static_cast<double>(local_idx), s_meta_partkey.theta0);
        T partkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter: p_category = 14
        if (partkey == 0 || partkey > 800000) continue;
        if (d_part_ht[partkey].p_category != 14) continue;
        uint32_t p_brand1 = d_part_ht[partkey].p_brand1;

        // Decompress suppkey
        delta = 0;
        if (c_suppkey->d_plain_deltas != nullptr) {
            delta = c_suppkey->d_plain_deltas[global_idx];
        } else if (s_meta_suppkey.delta_bits > 0 && c_suppkey->delta_array != nullptr) {
            int64_t bit_offset = s_meta_suppkey.bit_offset_base + (int64_t)local_idx * s_meta_suppkey.delta_bits;
            delta = extractDelta_Optimized<T>(c_suppkey->delta_array, bit_offset, s_meta_suppkey.delta_bits);
        }
        predicted = fma(s_meta_suppkey.theta1, static_cast<double>(local_idx), s_meta_suppkey.theta0);
        T suppkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        // Filter: s_nation = 24 (UNITED STATES)
        if (suppkey == 0 || suppkey > 20000) continue;
        if (d_supplier_ht[suppkey].s_nation != 24) continue;
        uint32_t s_city = d_supplier_ht[suppkey].s_city;

        // Add to candidate list
        int local_pos = atomicAdd(&s_candidate_count, 1);
        if (local_pos < 256) {
            s_candidates[local_pos].global_idx = global_idx;
            s_candidates[local_pos].d_year = d_year;
            s_candidates[local_pos].s_city = s_city;
            s_candidates[local_pos].p_brand1 = p_brand1;
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
                s_candidates[0].s_city = s_city;
                s_candidates[0].p_brand1 = p_brand1;
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

// Stage 2: Random access custkey, revenue, supplycost for candidates
template<typename T>
__global__ void ssb_q43_stage2_random_access(
    const CompressedDataGLECO<T>* c_custkey,
    const CompressedDataGLECO<T>* c_revenue,
    const CompressedDataGLECO<T>* c_supplycost,
    const Candidate* d_candidates,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_years,
    uint32_t* d_group_scities,
    uint32_t* d_group_brands,
    uint64_t* d_group_keys,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    Candidate cand = d_candidates[idx];
    int global_idx = cand.global_idx;

    // Random access custkey (validity check only)
    T custkey = randomAccessDecompress(c_custkey, global_idx);
    if (custkey == 0 || custkey > 300000) return;

    // Random access revenue
    T revenue = randomAccessDecompress(c_revenue, global_idx);

    // Random access supplycost
    T supplycost = randomAccessDecompress(c_supplycost, global_idx);

    // Calculate profit
    unsigned long long profit = (revenue > supplycost) ? (revenue - supplycost) : 0;

    // Hash-based GROUP BY (d_year, s_city, p_brand1)
    uint64_t combined_key = ((uint64_t)cand.d_year << 32) | ((uint64_t)cand.s_city << 16) | cand.p_brand1;
    uint32_t hash = (uint32_t)(combined_key % max_groups);

    int group_idx = -1;
    for (int probe = 0; probe < 64; probe++) {
        int group_pos = (hash + probe * probe) % max_groups;

        uint64_t existing_key = d_group_keys[group_pos];
        if (existing_key == 0) {
            uint64_t old = atomicCAS((unsigned long long*)&d_group_keys[group_pos], 0ULL, combined_key);
            if (old == 0 || old == combined_key) {
                d_group_years[group_pos] = cand.d_year;
                d_group_scities[group_pos] = cand.s_city;
                d_group_brands[group_pos] = cand.p_brand1;
                group_idx = group_pos;
                break;
            }
        } else if (existing_key == combined_key) {
            group_idx = group_pos;
            break;
        }
    }

    if (group_idx >= 0) {
        atomicAdd(&d_group_results[group_idx], profit);
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 4.3 with GLECO2 Compression - Random Access Optimized" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit" << endl;
    cout << "       FROM date, customer, supplier, part, lineorder" << endl;
    cout << "       WHERE lo_custkey = c_custkey" << endl;
    cout << "             AND lo_suppkey = s_suppkey" << endl;
    cout << "             AND lo_partkey = p_partkey" << endl;
    cout << "             AND lo_orderdate = d_datekey" << endl;
    cout << "             AND s_nation = 'UNITED STATES'" << endl;
    cout << "             AND (d_year = 1997 OR d_year = 1998)" << endl;
    cout << "             AND p_category = 'MFGR#14'" << endl;
    cout << "       GROUP BY d_year, s_city, p_brand1" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    cout << "Loading LINEORDER columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_custkey = loadColumn<uint32_t>("lo_custkey", LO_LEN);
    vector<uint32_t> lo_suppkey = loadColumn<uint32_t>("lo_suppkey", LO_LEN);
    vector<uint32_t> lo_partkey = loadColumn<uint32_t>("lo_partkey", LO_LEN);
    vector<uint32_t> lo_revenue = loadColumn<uint32_t>("lo_revenue", LO_LEN);
    vector<uint32_t> lo_supplycost = loadColumn<uint32_t>("lo_supplycost", LO_LEN);

    cout << "Loading dimension tables..." << endl;
    vector<uint32_t> c_custkey = loadColumn<uint32_t>("c_custkey", 300000);

    vector<uint32_t> s_suppkey = loadColumn<uint32_t>("s_suppkey", 20000);
    vector<uint32_t> s_nation = loadColumn<uint32_t>("s_nation", 20000);
    vector<uint32_t> s_city = loadColumn<uint32_t>("s_city", 20000);

    vector<uint32_t> p_partkey = loadColumn<uint32_t>("p_partkey", 800000);
    vector<uint32_t> p_brand1 = loadColumn<uint32_t>("p_brand1", 800000);
    vector<uint32_t> p_category = loadColumn<uint32_t>("p_category", 800000);

    vector<uint32_t> d_datekey = loadColumn<uint32_t>("d_datekey", 2556);
    vector<uint32_t> d_year = loadColumn<uint32_t>("d_year", 2556);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded all columns in " << load_time << " seconds" << endl;
    cout << endl;

    cout << "Building hash tables..." << endl;
    vector<SupplierEntry> supplier_ht(20001);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        supplier_ht[s_suppkey[i]].s_nation = s_nation[i];
        supplier_ht[s_suppkey[i]].s_city = s_city[i];
    }

    vector<PartEntry> part_ht(800001);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        part_ht[p_partkey[i]].p_brand1 = p_brand1[i];
        part_ht[p_partkey[i]].p_category = p_category[i];
    }

    vector<DateEntry> date_ht(20000000);
    for (size_t i = 0; i < d_datekey.size(); i++) {
        date_ht[d_datekey[i]].d_year = d_year[i];
    }

    cout << "✓ Hash tables built" << endl;
    cout << endl;

    cout << "Compressing LINEORDER columns with GLECO2..." << endl;
    
    
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataGLECO<uint32_t>* c_orderdate = compressData(lo_orderdate, 1024);
    CompressedDataGLECO<uint32_t>* c_custkey_c = compressData(lo_custkey, 1024);
    CompressedDataGLECO<uint32_t>* c_suppkey_c = compressData(lo_suppkey, 1024);
    CompressedDataGLECO<uint32_t>* c_partkey_c = compressData(lo_partkey, 1024);
    CompressedDataGLECO<uint32_t>* c_revenue = compressData(lo_revenue, 1024);
    CompressedDataGLECO<uint32_t>* c_supplycost_c = compressData(lo_supplycost, 1024);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << endl;

    SupplierEntry* d_supplier_ht;
    PartEntry* d_part_ht;
    DateEntry* d_date_ht;

    CUDA_CHECK(cudaMalloc(&d_supplier_ht, supplier_ht.size() * sizeof(SupplierEntry)));
    CUDA_CHECK(cudaMalloc(&d_part_ht, part_ht.size() * sizeof(PartEntry)));
    CUDA_CHECK(cudaMalloc(&d_date_ht, date_ht.size() * sizeof(DateEntry)));

    CUDA_CHECK(cudaMemcpy(d_supplier_ht, supplier_ht.data(), supplier_ht.size() * sizeof(SupplierEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_part_ht, part_ht.data(), part_ht.size() * sizeof(PartEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_date_ht, date_ht.data(), date_ht.size() * sizeof(DateEntry), cudaMemcpyHostToDevice));

    const int MAX_CANDIDATES = 10000000;
    Candidate* d_candidates;
    int* d_candidate_count;
    CUDA_CHECK(cudaMalloc(&d_candidates, MAX_CANDIDATES * sizeof(Candidate)));
    CUDA_CHECK(cudaMalloc(&d_candidate_count, sizeof(int)));

    const int MAX_GROUPS = 50000;
    unsigned long long* d_group_results;
    uint32_t* d_group_years;
    uint32_t* d_group_scities;
    uint32_t* d_group_brands;
    uint64_t* d_group_keys;

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_scities, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_brands, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_keys, MAX_GROUPS * sizeof(uint64_t)));

    cout << "** Running Random Access Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_scities, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_brands, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_keys, 0, MAX_GROUPS * sizeof(uint64_t)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Scan orderdate, partkey, suppkey with filters
        ssb_q43_stage1_scan<uint32_t><<<c_orderdate->num_partitions, 256>>>(
            c_orderdate->d_self, c_partkey_c->d_self, c_suppkey_c->d_self,
            d_date_ht, d_part_ht, d_supplier_ht, LO_LEN,
            d_candidates, d_candidate_count
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int num_candidates;
        CUDA_CHECK(cudaMemcpy(&num_candidates, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: Random access custkey, revenue, supplycost for candidates
        if (num_candidates > 0) {
            int blocks = (num_candidates + 255) / 256;
            ssb_q43_stage2_random_access<uint32_t><<<blocks, 256>>>(
                c_custkey_c->d_self, c_revenue->d_self, c_supplycost_c->d_self,
                d_candidates, num_candidates,
                d_group_results, d_group_years, d_group_scities, d_group_brands, d_group_keys, MAX_GROUPS
            );
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        vector<unsigned long long> h_group_results(MAX_GROUPS);
        vector<uint32_t> h_group_years(MAX_GROUPS);
        vector<uint32_t> h_group_scities(MAX_GROUPS);
        vector<uint32_t> h_group_brands(MAX_GROUPS);

        CUDA_CHECK(cudaMemcpy(h_group_results.data(), d_group_results, MAX_GROUPS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_years.data(), d_group_years, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_scities.data(), d_group_scities, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_brands.data(), d_group_brands, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        map<GroupKey, unsigned long long> results_map;
        for (int i = 0; i < MAX_GROUPS; i++) {
            if (h_group_years[i] != 0) {
                GroupKey key{h_group_years[i], h_group_scities[i], h_group_brands[i]};
                results_map[key] += h_group_results[i];
            }
        }

        double selectivity = (num_candidates * 100.0) / LO_LEN;
        cout << "{\"query\":43,\"time_query\":" << query_time
             << ",\"num_candidates\":" << num_candidates
             << ",\"partitions_pruned\":0"
             << ",\"selectivity\":" << selectivity
             << ",\"num_groups\":" << results_map.size() << "}" << endl;

        if (t == 0) {
            cout << "Sample results (d_year, s_city, p_brand1, profit):" << endl;
            int count = 0;
            for (const auto& [key, profit] : results_map) {
                cout << "  " << key.d_year << ", " << key.s_city << ", " << key.p_brand1 << ": " << profit << endl;
                if (++count >= 10) break;
            }
        }
    }

    cout << endl;

    CUDA_CHECK(cudaFree(d_supplier_ht));
    CUDA_CHECK(cudaFree(d_part_ht));
    CUDA_CHECK(cudaFree(d_date_ht));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_group_results));
    CUDA_CHECK(cudaFree(d_group_years));
    CUDA_CHECK(cudaFree(d_group_scities));
    CUDA_CHECK(cudaFree(d_group_brands));
    CUDA_CHECK(cudaFree(d_group_keys));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_custkey_c);
    freeCompressedData(c_suppkey_c);
    freeCompressedData(c_partkey_c);
    freeCompressedData(c_revenue);
    freeCompressedData(c_supplycost_c);

    cout << "========================================================================" << endl;

    return 0;
}
