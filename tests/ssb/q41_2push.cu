// SSB Query 4.1 with GLECO2 Compression - RANDOM ACCESS VERSION   (_2push version)
// Stage 1: Scan partkey, custkey, suppkey (3 cols)
// Stage 2: Random access orderdate, revenue, supplycost (3 cols)

// 
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
    uint32_t s_region;
};

struct PartEntry {
    uint32_t p_mfgr;
};

struct DateEntry {
    uint32_t d_year;
};

struct Candidate {
    int global_idx;
    uint32_t c_nation;
};

struct GroupKey {
    uint32_t d_year;
    uint32_t c_nation;

    bool operator<(const GroupKey& other) const {
        if (d_year != other.d_year) return d_year < other.d_year;
        return c_nation < other.c_nation;
    }
};

// Stage 1: Scan partkey, custkey, suppkey
template<typename T>
__global__ void ssb_q41_stage1_scan(
    const CompressedDataGLECO<T>* c_partkey,
    const CompressedDataGLECO<T>* c_custkey,
    const CompressedDataGLECO<T>* c_suppkey,
    const PartEntry* d_part_ht,
    const CustomerEntry* d_customer_ht,
    const SupplierEntry* d_supplier_ht,
    int num_entries,
    Candidate* d_candidates,
    int* d_candidate_count) {

    __shared__ PartitionMetaOpt s_meta[3];
    __shared__ Candidate s_local_candidates[256];
    __shared__ int s_local_count;

    int partition_idx = blockIdx.x;
    if (partition_idx >= c_partkey->num_partitions) return;

    if (threadIdx.x == 0) {
        s_local_count = 0;
        for (int i = 0; i < 3; i++) {
            const CompressedDataGLECO<T>* data = (i == 0) ? c_partkey : (i == 1) ? c_custkey : c_suppkey;
            s_meta[i].start_idx = data->d_start_indices[partition_idx];
            s_meta[i].delta_bits = data->d_delta_bits[partition_idx];
            s_meta[i].bit_offset_base = data->d_delta_array_bit_offsets[partition_idx];
            s_meta[i].theta0 = data->d_model_params[partition_idx * 4];
            s_meta[i].theta1 = data->d_model_params[partition_idx * 4 + 1];
            s_meta[i].partition_len = data->d_end_indices[partition_idx] - s_meta[i].start_idx;
        }
    }

    __syncthreads();

    for (int local_idx = threadIdx.x; local_idx < s_meta[0].partition_len; local_idx += blockDim.x) {
        int global_idx = s_meta[0].start_idx + local_idx;
        if (global_idx >= num_entries) continue;

        // Decompress partkey
        long long delta = 0;
        if (c_partkey->d_plain_deltas != nullptr) {
            delta = c_partkey->d_plain_deltas[global_idx];
        } else if (s_meta[0].delta_bits > 0) {
            int64_t bit_offset = s_meta[0].bit_offset_base + (int64_t)local_idx * s_meta[0].delta_bits;
            delta = extractDelta_Optimized<T>(c_partkey->delta_array, bit_offset, s_meta[0].delta_bits);
        }
        double predicted = fma(s_meta[0].theta1, static_cast<double>(local_idx), s_meta[0].theta0);
        T partkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        if (partkey == 0 || partkey > 800000) continue;
        uint32_t p_mfgr = d_part_ht[partkey].p_mfgr;
        if (p_mfgr != 0 && p_mfgr != 1) continue;

        // Decompress custkey
        delta = 0;
        if (c_custkey->d_plain_deltas != nullptr) {
            delta = c_custkey->d_plain_deltas[global_idx];
        } else if (s_meta[1].delta_bits > 0) {
            int64_t bit_offset = s_meta[1].bit_offset_base + (int64_t)local_idx * s_meta[1].delta_bits;
            delta = extractDelta_Optimized<T>(c_custkey->delta_array, bit_offset, s_meta[1].delta_bits);
        }
        predicted = fma(s_meta[1].theta1, static_cast<double>(local_idx), s_meta[1].theta0);
        T custkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        if (custkey == 0 || custkey > 300000) continue;
        if (d_customer_ht[custkey].c_region != 1) continue;
        uint32_t c_nation = d_customer_ht[custkey].c_nation;

        // Decompress suppkey
        delta = 0;
        if (c_suppkey->d_plain_deltas != nullptr) {
            delta = c_suppkey->d_plain_deltas[global_idx];
        } else if (s_meta[2].delta_bits > 0) {
            int64_t bit_offset = s_meta[2].bit_offset_base + (int64_t)local_idx * s_meta[2].delta_bits;
            delta = extractDelta_Optimized<T>(c_suppkey->delta_array, bit_offset, s_meta[2].delta_bits);
        }
        predicted = fma(s_meta[2].theta1, static_cast<double>(local_idx), s_meta[2].theta0);
        T suppkey = applyDelta(static_cast<T>(__double2int_rn(predicted)), delta);

        if (suppkey == 0 || suppkey > 20000) continue;
        if (d_supplier_ht[suppkey].s_region != 1) continue;

        int pos = atomicAdd(&s_local_count, 1);
        if (pos < 256) {
            s_local_candidates[pos] = {global_idx, c_nation};
        } else {
            __syncthreads();
            if (threadIdx.x == 0) {
                int global_pos = atomicAdd(d_candidate_count, s_local_count - 1);
                for (int i = 0; i < 256; i++) {
                    d_candidates[global_pos + i] = s_local_candidates[i];
                }
                s_local_candidates[0] = {global_idx, c_nation};
                s_local_count = 1;
            }
            __syncthreads();
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && s_local_count > 0) {
        int global_pos = atomicAdd(d_candidate_count, s_local_count);
        for (int i = 0; i < s_local_count; i++) {
            d_candidates[global_pos + i] = s_local_candidates[i];
        }
    }
}

// Stage 2: Random access orderdate, revenue, supplycost
template<typename T>
__global__ void ssb_q41_stage2_random_access(
    const CompressedDataGLECO<T>* c_orderdate,
    const CompressedDataGLECO<T>* c_revenue,
    const CompressedDataGLECO<T>* c_supplycost,
    const DateEntry* d_date_ht,
    const Candidate* d_candidates,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_years,
    uint32_t* d_group_nations,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    Candidate cand = d_candidates[idx];
    int global_idx = cand.global_idx;

    T orderdate = randomAccessDecompress<T>(c_orderdate, global_idx);

    if (orderdate >= 20000000 || d_date_ht[orderdate].d_year == 0) return;
    uint32_t d_year = d_date_ht[orderdate].d_year;

    T revenue = randomAccessDecompress<T>(c_revenue, global_idx);
    T supplycost = randomAccessDecompress<T>(c_supplycost, global_idx);

    unsigned long long profit = (revenue > supplycost) ? (revenue - supplycost) : 0;

    uint64_t combined_key = ((uint64_t)d_year << 32) | cand.c_nation;
    uint32_t hash = (uint32_t)(combined_key % max_groups);

    int group_idx = -1;
    for (int probe = 0; probe < 64; probe++) {
        int gidx = (hash + probe * probe) % max_groups;

        if (d_group_years[gidx] == d_year && d_group_nations[gidx] == cand.c_nation) {
            group_idx = gidx;
            break;
        }

        if (d_group_years[gidx] == 0) {
            unsigned int old = atomicCAS(&d_group_years[gidx], 0, d_year);
            if (old == 0) {
                d_group_nations[gidx] = cand.c_nation;
                group_idx = gidx;
                break;
            } else if (old == d_year && d_group_nations[gidx] == cand.c_nation) {
                group_idx = gidx;
                break;
            }
        }
    }

    if (group_idx >= 0) {
        atomicAdd(&d_group_results[group_idx], profit);
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 4.1 with GLECO2 Compression - RANDOM ACCESS" << endl;
    cout << "========================================================================" << endl;
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
    vector<uint32_t> c_nation = loadColumn<uint32_t>("c_nation", 300000);
    vector<uint32_t> c_region = loadColumn<uint32_t>("c_region", 300000);

    vector<uint32_t> s_suppkey = loadColumn<uint32_t>("s_suppkey", 20000);
    vector<uint32_t> s_region = loadColumn<uint32_t>("s_region", 20000);

    vector<uint32_t> p_partkey = loadColumn<uint32_t>("p_partkey", 800000);
    vector<uint32_t> p_mfgr = loadColumn<uint32_t>("p_mfgr", 800000);

    vector<uint32_t> d_datekey = loadColumn<uint32_t>("d_datekey", 2556);
    vector<uint32_t> d_year = loadColumn<uint32_t>("d_year", 2556);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded all columns in " << load_time << " seconds" << endl;
    cout << endl;

    cout << "Building hash tables..." << endl;
    vector<CustomerEntry> customer_ht(300001);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        customer_ht[c_custkey[i]].c_nation = c_nation[i];
        customer_ht[c_custkey[i]].c_region = c_region[i];
    }

    vector<SupplierEntry> supplier_ht(20001);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        supplier_ht[s_suppkey[i]].s_region = s_region[i];
    }

    vector<PartEntry> part_ht(800001);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        part_ht[p_partkey[i]].p_mfgr = p_mfgr[i];
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

    CustomerEntry* d_customer_ht;
    SupplierEntry* d_supplier_ht;
    PartEntry* d_part_ht;
    DateEntry* d_date_ht;

    CUDA_CHECK(cudaMalloc(&d_customer_ht, customer_ht.size() * sizeof(CustomerEntry)));
    CUDA_CHECK(cudaMalloc(&d_supplier_ht, supplier_ht.size() * sizeof(SupplierEntry)));
    CUDA_CHECK(cudaMalloc(&d_part_ht, part_ht.size() * sizeof(PartEntry)));
    CUDA_CHECK(cudaMalloc(&d_date_ht, date_ht.size() * sizeof(DateEntry)));

    CUDA_CHECK(cudaMemcpy(d_customer_ht, customer_ht.data(), customer_ht.size() * sizeof(CustomerEntry), cudaMemcpyHostToDevice));
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
    uint32_t* d_group_nations;

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_nations, MAX_GROUPS * sizeof(uint32_t)));

    cout << "** Running Random Access Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_candidate_count, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_nations, 0, MAX_GROUPS * sizeof(uint32_t)));

        auto query_start = chrono::high_resolution_clock::now();

        ssb_q41_stage1_scan<uint32_t><<<c_partkey_c->num_partitions, 256>>>(
            c_partkey_c->d_self, c_custkey_c->d_self, c_suppkey_c->d_self,
            d_part_ht, d_customer_ht, d_supplier_ht, LO_LEN,
            d_candidates, d_candidate_count
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        int h_candidate_count = 0;
        CUDA_CHECK(cudaMemcpy(&h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_candidate_count > 0) {
            int blocks = (h_candidate_count + 255) / 256;
            ssb_q41_stage2_random_access<uint32_t><<<blocks, 256>>>(
                c_orderdate->d_self, c_revenue->d_self, c_supplycost_c->d_self,
                d_date_ht, d_candidates, h_candidate_count,
                d_group_results, d_group_years, d_group_nations, MAX_GROUPS
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        vector<unsigned long long> h_group_results(MAX_GROUPS);
        vector<uint32_t> h_group_years(MAX_GROUPS);
        vector<uint32_t> h_group_nations(MAX_GROUPS);

        CUDA_CHECK(cudaMemcpy(h_group_results.data(), d_group_results, MAX_GROUPS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_years.data(), d_group_years, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_nations.data(), d_group_nations, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        map<GroupKey, unsigned long long> results_map;
        for (int i = 0; i < MAX_GROUPS; i++) {
            if (h_group_years[i] != 0) {
                GroupKey key{h_group_years[i], h_group_nations[i]};
                results_map[key] += h_group_results[i];
            }
        }

        double selectivity = (double)h_candidate_count / LO_LEN * 100.0;
        cout << "{\"query\":41,\"time_query\":" << query_time
             << ",\"num_groups\":" << results_map.size()
             << ",\"candidates\":" << h_candidate_count
             << ",\"partitions_pruned\":0"
             << ",\"selectivity\":" << selectivity << "}" << endl;

        if (t == 0) {
            cout << "Sample results (d_year, c_nation, profit):" << endl;
            int count = 0;
            for (const auto& [key, profit] : results_map) {
                cout << "  " << key.d_year << ", " << key.c_nation << ": " << profit << endl;
                if (++count >= 10) break;
            }
        }
    }

    cout << endl;

    CUDA_CHECK(cudaFree(d_customer_ht));
    CUDA_CHECK(cudaFree(d_supplier_ht));
    CUDA_CHECK(cudaFree(d_part_ht));
    CUDA_CHECK(cudaFree(d_date_ht));
    CUDA_CHECK(cudaFree(d_candidates));
    CUDA_CHECK(cudaFree(d_candidate_count));
    CUDA_CHECK(cudaFree(d_group_results));
    CUDA_CHECK(cudaFree(d_group_years));
    CUDA_CHECK(cudaFree(d_group_nations));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_custkey_c);
    freeCompressedData(c_suppkey_c);
    freeCompressedData(c_partkey_c);
    freeCompressedData(c_revenue);
    freeCompressedData(c_supplycost_c);

    cout << "========================================================================" << endl;

    return 0;
}
