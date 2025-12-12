/**
 * @file q43.cu
 * @brief SSB Q4.3 Implementation - Staged Random Access Optimization
 *
 * Query:
 *   SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM date, customer, supplier, part, lineorder
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND s_nation = 'UNITED STATES'
 *     AND (d_year = 1997 OR d_year = 1998)
 *     AND p_category = 'MFGR#14'
 *   GROUP BY d_year, s_city, p_brand1 ORDER BY d_year, s_city, p_brand1;
 *
 * Staged Decomposition Strategy:
 *   Stage 1: Decompress partkey, probe part filter → ~4%, get p_brand
 *   Stage 2: Random access suppkey, probe supplier filter → get s_city
 *   Stage 3: Random access orderdate, probe date filter (d_year=1997/1998) → get d_year
 *   Stage 4: Random access revenue + supplycost, aggregate
 *
 * Note: Q4.3 does NOT have customer filter, only supplier nation filter
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

constexpr uint32_t S_NATION_US = 24;  // UNITED STATES
constexpr uint32_t P_CATEGORY_MFGR14 = 14;  // MFGR#14
constexpr int NUM_YEARS = 2;  // 1997, 1998
constexpr int NUM_CITIES = 250;
constexpr int NUM_BRANDS = 1000;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_BRANDS;

/**
 * @brief Stage 1: Probe part hash table, get p_brand
 */
__global__ void probePartFilterQ43Kernel(
    const uint32_t* __restrict__ partkeys,
    int num_rows,
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    int* __restrict__ passing_indices,
    uint32_t* __restrict__ passing_p_brands,
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
    uint32_t p_brand = 0;

    if (tid < num_rows) {
        uint32_t pk = partkeys[tid];
        int slot = hash_murmur3(pk, ht_p_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_p_size;
            if (ht_p_keys[s] == pk) {
                p_brand = ht_p_values[s];
                found = true;
                break;
            }
            if (ht_p_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
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
        passing_p_brands[global_pos] = p_brand;
    }
}

/**
 * @brief Stage 2: Probe supplier hash table with input indices, get s_city
 */
__global__ void probeSupplierWithIndicesQ43Kernel(
    const uint32_t* __restrict__ suppkeys,
    const int* __restrict__ input_indices,
    const uint32_t* __restrict__ input_p_brands,
    int num_rows,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    int* __restrict__ output_indices,
    uint32_t* __restrict__ output_p_brands,
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
    int original_idx = -1;
    uint32_t p_brand = 0;
    uint32_t s_city = 0;

    if (tid < num_rows) {
        uint32_t sk = suppkeys[tid];
        original_idx = input_indices[tid];
        p_brand = input_p_brands[tid];

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
        output_indices[global_pos] = original_idx;
        output_p_brands[global_pos] = p_brand;
        output_s_cities[global_pos] = s_city;
    }
}

/**
 * @brief Stage 3: Probe date hash table with input indices, get d_year
 */
__global__ void probeDateWithIndicesQ43Kernel(
    const uint32_t* __restrict__ orderdates,
    const int* __restrict__ input_indices,
    const uint32_t* __restrict__ input_p_brands,
    const uint32_t* __restrict__ input_s_cities,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    int* __restrict__ output_indices,
    uint32_t* __restrict__ output_p_brands,
    uint32_t* __restrict__ output_s_cities,
    uint32_t* __restrict__ output_d_years,
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
    int original_idx = -1;
    uint32_t p_brand = 0;
    uint32_t s_city = 0;
    uint32_t d_year = 0;

    if (tid < num_rows) {
        uint32_t od = orderdates[tid];
        original_idx = input_indices[tid];
        p_brand = input_p_brands[tid];
        s_city = input_s_cities[tid];

        int slot = hash_murmur3(od, ht_d_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_d_size;
            if (ht_d_keys[s] == od) {
                d_year = ht_d_values[s];
                found = true;
                break;
            }
            if (ht_d_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
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
        output_p_brands[global_pos] = p_brand;
        output_s_cities[global_pos] = s_city;
        output_d_years[global_pos] = d_year;
    }
}

/**
 * @brief Stage 4: Final aggregation
 */
__global__ void aggregateQ43StagedKernel(
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ lo_supplycost,
    const uint32_t* __restrict__ passing_d_years,
    const uint32_t* __restrict__ passing_s_cities,
    const uint32_t* __restrict__ passing_p_brands,
    int num_rows,
    long long* __restrict__ agg_profit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t rev = lo_revenue[i];
        uint32_t cost = lo_supplycost[i];
        uint32_t d_year = passing_d_years[i];
        uint32_t s_city = passing_s_cities[i];
        uint32_t p_brand = passing_p_brands[i];

        int year_idx = d_year - 1997;  // 0 for 1997, 1 for 1998
        if (year_idx >= 0 && year_idx < NUM_YEARS && s_city < NUM_CITIES && p_brand < NUM_BRANDS) {
            int agg_idx = year_idx * NUM_CITIES * NUM_BRANDS + s_city * NUM_BRANDS + p_brand;
            long long profit = static_cast<long long>(rev) - static_cast<long long>(cost);
            atomicAdd(reinterpret_cast<unsigned long long*>(&agg_profit[agg_idx]),
                     static_cast<unsigned long long>(profit));
        }
    }
}

// Hash table build kernels
__global__ void build_date_year_filter_ht_q43(const uint32_t* d_datekey, const uint32_t* d_year, int n,
                                               uint32_t year1, uint32_t year2, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t year = d_year[idx];
    if (year != year1 && year != year2) return;
    uint32_t key = d_datekey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = year; return; }
    }
}

__global__ void build_supplier_nation_city_ht_q43(const uint32_t* s_suppkey, const uint32_t* s_nation, const uint32_t* s_city,
                                                   int n, uint32_t filter_nation, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (s_nation[idx] != filter_nation) return;
    uint32_t key = s_suppkey[idx], val = s_city[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_part_category_brand_ht_q43(const uint32_t* p_partkey, const uint32_t* p_category, const uint32_t* p_brand1,
                                                  int n, uint32_t filter_category, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (p_category[idx] != filter_category) return;
    uint32_t key = p_partkey[idx], val = p_brand1[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

void runQ43RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);
    CompressedColumnAccessorVertical<uint32_t> acc_supplycost(&data.lo_supplycost);

    int total_rows = acc_suppkey.getTotalElements();

    // Build dimension hash tables
    timer.start();

    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));

    build_date_year_filter_ht_q43<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, 1997, 1998, ht_d_keys, ht_d_values, ht_d_size);

    int ht_s_size = S_LEN * 2;
    uint32_t *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_supplier_nation_city_ht_q43<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN, S_NATION_US,
        ht_s_keys, ht_s_values, ht_s_size);

    int ht_p_size = P_LEN * 2;
    uint32_t *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    build_part_category_brand_ht_q43<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN, P_CATEGORY_MFGR14,
        ht_p_keys, ht_p_values, ht_p_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== STAGED DECOMPOSITION ==========

    // === STAGE 1: Decompress partkey + probe part filter, get p_brand ===
    uint32_t* d_partkey;
    cudaMalloc(&d_partkey, total_rows * sizeof(uint32_t));

    timer.start();
    acc_partkey.decompressAll(d_partkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_partkey_ms = timer.elapsed_ms();

    int* d_stage1_indices;
    uint32_t* d_stage1_p_brands;
    int* d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_stage1_p_brands, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probePartFilterQ43Kernel<<<grid_stage1, block_size>>>(
        d_partkey, total_rows,
        ht_p_keys, ht_p_values, ht_p_size,
        d_stage1_indices, d_stage1_p_brands, d_num_stage1);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage1_ms = timer.elapsed_ms();

    int h_num_stage1;
    cudaMemcpy(&h_num_stage1, d_num_stage1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_partkey);
    cudaFree(d_num_stage1);

    float stage1_selectivity = 100.0f * h_num_stage1 / total_rows;
    std::cout << "Stage 1 (part filter): " << stage1_selectivity << "% (" << h_num_stage1 << " rows)" << std::endl;

    if (h_num_stage1 == 0) {
        std::cout << "\n=== Q4.3 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage1_indices); cudaFree(d_stage1_p_brands);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_s_keys); cudaFree(ht_s_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        return;
    }

    // === STAGE 2: Random access suppkey + probe supplier filter, get s_city ===
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, h_num_stage1 * sizeof(uint32_t));

    timer.start();
    acc_suppkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_suppkey_ms = timer.elapsed_ms();

    int* d_stage2_indices;
    uint32_t* d_stage2_p_brands;
    uint32_t* d_stage2_s_cities;
    int* d_num_stage2;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_stage2_p_brands, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_stage2_s_cities, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probeSupplierWithIndicesQ43Kernel<<<grid_stage2, block_size>>>(
        d_suppkey, d_stage1_indices, d_stage1_p_brands, h_num_stage1,
        ht_s_keys, ht_s_values, ht_s_size,
        d_stage2_indices, d_stage2_p_brands, d_stage2_s_cities, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_stage1_p_brands);
    cudaFree(d_num_stage2);

    float stage2_selectivity = 100.0f * h_num_stage2 / total_rows;
    std::cout << "Stage 2 (supplier filter): " << stage2_selectivity << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q4.3 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage2_indices); cudaFree(d_stage2_p_brands); cudaFree(d_stage2_s_cities);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_s_keys); cudaFree(ht_s_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        return;
    }

    // === STAGE 3: Random access orderdate + probe date filter, get d_year ===
    uint32_t* d_orderdate;
    cudaMalloc(&d_orderdate, h_num_stage2 * sizeof(uint32_t));

    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_orderdate, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_orderdate_ms = timer.elapsed_ms();

    int* d_stage3_indices;
    uint32_t* d_stage3_p_brands;
    uint32_t* d_stage3_s_cities;
    uint32_t* d_stage3_d_years;
    int* d_num_stage3;
    cudaMalloc(&d_stage3_indices, h_num_stage2 * sizeof(int));
    cudaMalloc(&d_stage3_p_brands, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_stage3_s_cities, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_stage3_d_years, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage3, sizeof(int));
    cudaMemset(d_num_stage3, 0, sizeof(int));

    timer.start();
    int grid_stage3 = (h_num_stage2 + block_size - 1) / block_size;
    probeDateWithIndicesQ43Kernel<<<grid_stage3, block_size>>>(
        d_orderdate, d_stage2_indices, d_stage2_p_brands, d_stage2_s_cities, h_num_stage2,
        ht_d_keys, ht_d_values, ht_d_size,
        d_stage3_indices, d_stage3_p_brands, d_stage3_s_cities, d_stage3_d_years, d_num_stage3);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage3_ms = timer.elapsed_ms();

    int h_num_stage3;
    cudaMemcpy(&h_num_stage3, d_num_stage3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_orderdate);
    cudaFree(d_stage2_indices);
    cudaFree(d_stage2_p_brands);
    cudaFree(d_stage2_s_cities);
    cudaFree(d_num_stage3);

    float stage3_selectivity = 100.0f * h_num_stage3 / total_rows;
    std::cout << "Stage 3 (date filter): " << stage3_selectivity << "% (" << h_num_stage3 << " rows)" << std::endl;

    if (h_num_stage3 == 0) {
        std::cout << "\n=== Q4.3 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage3_indices); cudaFree(d_stage3_p_brands);
        cudaFree(d_stage3_s_cities); cudaFree(d_stage3_d_years);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_s_keys); cudaFree(ht_s_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        return;
    }

    // === STAGE 4: Random access revenue + supplycost, aggregate ===
    uint32_t *d_revenue, *d_supplycost;
    cudaMalloc(&d_revenue, h_num_stage3 * sizeof(uint32_t));
    cudaMalloc(&d_supplycost, h_num_stage3 * sizeof(uint32_t));

    timer.start();
    acc_revenue.randomAccessBatchIndices(d_stage3_indices, h_num_stage3, d_revenue, stream);
    acc_supplycost.randomAccessBatchIndices(d_stage3_indices, h_num_stage3, d_supplycost, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_final_ms = timer.elapsed_ms();

    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    timer.start();
    int grid_agg = min((h_num_stage3 + block_size - 1) / block_size, 256);
    aggregateQ43StagedKernel<<<grid_agg, block_size>>>(
        d_revenue, d_supplycost, d_stage3_d_years, d_stage3_s_cities, d_stage3_p_brands,
        h_num_stage3, d_agg_profit);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    // Collect results
    std::vector<long long> h_agg_profit(AGG_SIZE);
    cudaMemcpy(h_agg_profit.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    long long total_profit = 0;
    int num_groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) {
        if (h_agg_profit[i] != 0) {
            total_profit += h_agg_profit[i];
            num_groups++;
        }
    }

    float data_load_total = decompress_partkey_ms + ra_suppkey_ms + ra_orderdate_ms + ra_final_ms;
    float kernel_total = probe_stage1_ms + probe_stage2_ms + probe_stage3_ms + aggregate_ms;

    timing.data_load_ms = data_load_total;
    timing.kernel_ms = kernel_total;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q4.3 Results (Staged Random Access) ===" << std::endl;
    std::cout << "Groups: " << num_groups << ", Total profit: " << total_profit << std::endl;
    std::cout << "\nTiming breakdown (STAGED):" << std::endl;
    std::cout << "  Hash build:           " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Stage 1 decomp part:  " << decompress_partkey_ms << " ms" << std::endl;
    std::cout << "  Stage 1 probe part:   " << probe_stage1_ms << " ms (120M rows)" << std::endl;
    std::cout << "  Stage 2 RA suppkey:   " << ra_suppkey_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 2 probe supp:   " << probe_stage2_ms << " ms" << std::endl;
    std::cout << "  Stage 3 RA orderdate: " << ra_orderdate_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Stage 3 probe date:   " << probe_stage3_ms << " ms" << std::endl;
    std::cout << "  Stage 4 RA final:     " << ra_final_ms << " ms (" << h_num_stage3 << " rows)" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q4.3");

    // Cleanup
    cudaFree(d_stage3_indices); cudaFree(d_stage3_p_brands);
    cudaFree(d_stage3_s_cities); cudaFree(d_stage3_d_years);
    cudaFree(d_revenue); cudaFree(d_supplycost);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(d_agg_profit);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q4.3 - Staged Random Access ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ43RandomAccess(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ43RandomAccess(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
