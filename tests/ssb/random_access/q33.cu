/**
 * @file q33.cu
 * @brief SSB Q3.3 Implementation - Staged Random Access Optimization
 *
 * Query:
 *   SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey AND lo_orderdate = d_datekey
 *     AND (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
 *     AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
 *     AND d_year >= 1992 AND d_year <= 1997
 *   GROUP BY c_city, s_city, d_year
 *
 * Staged Decomposition Strategy:
 *   Stage 1: Decompress suppkey, probe supplier city filter → ~0.8% (very selective)
 *   Stage 2: Random access custkey for passing rows, probe customer filter
 *   Stage 3: Random access orderdate + revenue, aggregate
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "ssb_filter_kernels.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t CITY_UK1 = 231;
constexpr uint32_t CITY_UK5 = 235;
constexpr int NUM_YEARS = 6;
constexpr int NUM_CITIES = 250;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_CITIES;

/**
 * @brief Stage 1: Probe supplier city filter (s_city IN ('UNITED KI1', 'UNITED KI5'))
 */
__global__ void probeSupplierFilterQ33Kernel(
    const uint32_t* __restrict__ suppkeys,
    int num_rows,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    int* __restrict__ passing_indices,
    uint32_t* __restrict__ passing_s_cities,
    int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    bool found = false;
    uint32_t s_city = 0;

    if (tid < num_rows) {
        uint32_t sk = suppkeys[tid];
        int slot = hash_murmur3(sk, ht_s_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_s_size;
            if (ht_s_keys[s] == sk) {
                s_city = ht_s_values[s];
                found = true;
                break;
            }
            if (ht_s_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
        passing_s_cities[global_pos] = s_city;
    }
}

/**
 * @brief Stage 2: Probe customer city filter with input indices
 */
__global__ void probeCustomerWithIndicesQ33Kernel(
    const uint32_t* __restrict__ custkeys,
    const int* __restrict__ input_indices,
    const uint32_t* __restrict__ input_s_cities,
    int num_rows,
    const uint32_t* __restrict__ ht_c_keys,
    const uint32_t* __restrict__ ht_c_values,
    int ht_c_size,
    int* __restrict__ output_indices,
    uint32_t* __restrict__ output_c_cities,
    uint32_t* __restrict__ output_s_cities,
    int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    bool found = false;
    uint32_t c_city = 0;
    int original_idx = -1;
    uint32_t s_city = 0;

    if (tid < num_rows) {
        uint32_t ck = custkeys[tid];
        original_idx = input_indices[tid];
        s_city = input_s_cities[tid];

        int slot = hash_murmur3(ck, ht_c_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_c_size;
            if (ht_c_keys[s] == ck) {
                c_city = ht_c_values[s];
                found = true;
                break;
            }
            if (ht_c_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        output_indices[global_pos] = original_idx;
        output_c_cities[global_pos] = c_city;
        output_s_cities[global_pos] = s_city;
    }
}

__global__ void aggregateQ33Kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ passing_c_cities,
    const uint32_t* __restrict__ passing_s_cities,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    unsigned long long* __restrict__ agg_revenue)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t od = lo_orderdate[i];
        uint32_t rev = lo_revenue[i];
        uint32_t c_city = passing_c_cities[i];
        uint32_t s_city = passing_s_cities[i];

        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool found = false;
        for (int probe = 0; probe < 32; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; found = true; break; }
            if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }

        if (found && d_year >= 1992 && d_year <= 1997) {
            int year_idx = d_year - 1992;
            if (c_city < NUM_CITIES && s_city < NUM_CITIES) {
                int agg_idx = year_idx * NUM_CITIES * NUM_CITIES + c_city * NUM_CITIES + s_city;
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(rev));
            }
        }
    }
}

__global__ void build_date_ht_q33(const uint32_t* d_datekey, const uint32_t* d_year, int n, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t key = d_datekey[idx], val = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_customer_city_filter_ht_q33(const uint32_t* c_custkey, const uint32_t* c_city, int n,
                                                   uint32_t city1, uint32_t city2, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t city = c_city[idx];
    if (city != city1 && city != city2) return;
    uint32_t key = c_custkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = city; return; }
    }
}

__global__ void build_supplier_city_filter_ht_q33(const uint32_t* s_suppkey, const uint32_t* s_city, int n,
                                                   uint32_t city1, uint32_t city2, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t city = s_city[idx];
    if (city != city1 && city != city2) return;
    uint32_t key = s_suppkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = city; return; }
    }
}

void runQ33RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_custkey.getTotalElements();

    // Build hash tables
    timer.start();
    int ht_d_size = D_LEN * 2, ht_c_size = C_LEN * 2, ht_s_size = S_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_c_keys, *ht_c_values, *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t)); cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_date_ht_q33<<<(D_LEN+255)/256, 256>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_customer_city_filter_ht_q33<<<(C_LEN+255)/256, 256>>>(data.d_c_custkey, data.d_c_city, C_LEN, CITY_UK1, CITY_UK5, ht_c_keys, ht_c_values, ht_c_size);
    build_supplier_city_filter_ht_q33<<<(S_LEN+255)/256, 256>>>(data.d_s_suppkey, data.d_s_city, S_LEN, CITY_UK1, CITY_UK5, ht_s_keys, ht_s_values, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== STAGED DECOMPOSITION ==========

    // === STAGE 1: Decompress suppkey + probe supplier city filter ===
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, total_rows * sizeof(uint32_t));

    timer.start();
    acc_suppkey.decompressAll(d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_suppkey_ms = timer.elapsed_ms();

    int* d_stage1_indices;
    uint32_t* d_stage1_s_cities;
    int* d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_stage1_s_cities, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probeSupplierFilterQ33Kernel<<<grid_stage1, block_size>>>(
        d_suppkey, total_rows,
        ht_s_keys, ht_s_values, ht_s_size,
        d_stage1_indices, d_stage1_s_cities, d_num_stage1);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage1_ms = timer.elapsed_ms();

    int h_num_stage1;
    cudaMemcpy(&h_num_stage1, d_num_stage1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    cudaFree(d_num_stage1);

    float stage1_selectivity = 100.0f * h_num_stage1 / total_rows;
    std::cout << "Stage 1 (supplier city filter): " << stage1_selectivity << "% (" << h_num_stage1 << " rows)" << std::endl;

    if (h_num_stage1 == 0) {
        std::cout << "\n=== Q3.3 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage1_indices); cudaFree(d_stage1_s_cities);
        cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
        return;
    }

    // === STAGE 2: Random access custkey + probe customer city filter ===
    uint32_t* d_custkey;
    cudaMalloc(&d_custkey, h_num_stage1 * sizeof(uint32_t));

    timer.start();
    acc_custkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_custkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_custkey_ms = timer.elapsed_ms();

    int* d_stage2_indices;
    uint32_t* d_stage2_c_cities;
    uint32_t* d_stage2_s_cities;
    int* d_num_stage2;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_stage2_c_cities, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_stage2_s_cities, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probeCustomerWithIndicesQ33Kernel<<<grid_stage2, block_size>>>(
        d_custkey, d_stage1_indices, d_stage1_s_cities, h_num_stage1,
        ht_c_keys, ht_c_values, ht_c_size,
        d_stage2_indices, d_stage2_c_cities, d_stage2_s_cities, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_custkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_stage1_s_cities);
    cudaFree(d_num_stage2);

    float stage2_selectivity = 100.0f * h_num_stage2 / total_rows;
    std::cout << "Stage 2 (customer city filter): " << stage2_selectivity << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q3.3 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage2_indices); cudaFree(d_stage2_c_cities); cudaFree(d_stage2_s_cities);
        cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
        return;
    }

    // === STAGE 3: Random access orderdate + revenue, aggregate ===
    uint32_t *d_orderdate, *d_revenue;
    cudaMalloc(&d_orderdate, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_stage2 * sizeof(uint32_t));

    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_orderdate, stream);
    acc_revenue.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_final_ms = timer.elapsed_ms();

    unsigned long long* d_agg;
    cudaMalloc(&d_agg, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    int grid_agg = min((h_num_stage2 + block_size - 1) / block_size, 256);
    aggregateQ33Kernel<<<grid_agg, block_size>>>(
        d_orderdate, d_revenue, d_stage2_c_cities, d_stage2_s_cities, h_num_stage2,
        ht_d_keys, ht_d_values, ht_d_size, d_agg);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    // Collect results
    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    unsigned long long total = 0; int groups = 0;
    for (size_t i = 0; i < AGG_SIZE; ++i) if (h_agg[i] > 0) { total += h_agg[i]; groups++; }

    float data_load_total = decompress_suppkey_ms + ra_custkey_ms + ra_final_ms;
    float kernel_total = probe_stage1_ms + probe_stage2_ms + aggregate_ms;

    timing.data_load_ms = data_load_total;
    timing.kernel_ms = kernel_total;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q3.3 Results (Staged Random Access) ===" << std::endl;
    std::cout << "Groups: " << groups << ", Total: " << total << std::endl;
    std::cout << "\nTiming breakdown (STAGED):" << std::endl;
    std::cout << "  Hash build:           " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Stage 1 decomp supp:  " << decompress_suppkey_ms << " ms" << std::endl;
    std::cout << "  Stage 1 probe supp:   " << probe_stage1_ms << " ms (120M rows)" << std::endl;
    std::cout << "  Stage 2 RA custkey:   " << ra_custkey_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 2 probe cust:   " << probe_stage2_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 3 RA final:     " << ra_final_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q3.3");

    // Cleanup
    cudaFree(d_stage2_indices); cudaFree(d_stage2_c_cities); cudaFree(d_stage2_s_cities);
    cudaFree(d_orderdate); cudaFree(d_revenue); cudaFree(d_agg);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];
    std::cout << "=== SSB Q3.3 - Staged Random Access ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    QueryTiming timing;
    runQ33RandomAccess(data, timing);
    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ33RandomAccess(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
