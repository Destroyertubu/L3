/**
 * @file q41.cu
 * @brief SSB Q4.1 Implementation - Staged Random Access Optimization
 *
 * Query:
 *   SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM date, customer, supplier, part, lineorder
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND c_region = 'AMERICA' AND s_region = 'AMERICA'
 *     AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 *   GROUP BY d_year, c_nation ORDER BY d_year, c_nation;
 *
 * Staged Decomposition Strategy:
 *   Stage 1: Decompress partkey, probe part filter → ~40%
 *   Stage 2: Random access suppkey for passing rows, probe supplier filter → ~8%
 *   Stage 3: Random access custkey for passing rows, probe customer filter + get c_nation → ~1.6%
 *   Stage 4: Random access orderdate, revenue, supplycost, aggregate with year lookup
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

constexpr uint32_t C_REGION_AMERICA = 1;
constexpr uint32_t S_REGION_AMERICA = 1;
constexpr uint32_t P_MFGR_1 = 1;
constexpr uint32_t P_MFGR_2 = 2;
constexpr int NUM_YEARS = 7;
constexpr int NUM_NATIONS = 25;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS;

/**
 * @brief Stage 1: Probe part hash table (p_mfgr IN ('MFGR#1', 'MFGR#2'))
 */
__global__ void probePartFilterQ41Kernel(
    const uint32_t* __restrict__ partkeys,
    int num_rows,
    const uint32_t* __restrict__ ht_p_keys,
    int ht_p_size,
    int* __restrict__ passing_indices,
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

    if (tid < num_rows) {
        uint32_t pk = partkeys[tid];
        int slot = hash_murmur3(pk, ht_p_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_p_size;
            if (ht_p_keys[s] == pk) {
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
    }
}

/**
 * @brief Stage 2: Probe supplier hash table with input indices
 */
__global__ void probeSupplierWithIndicesQ41Kernel(
    const uint32_t* __restrict__ suppkeys,
    const int* __restrict__ input_indices,
    int num_rows,
    const uint32_t* __restrict__ ht_s_keys,
    int ht_s_size,
    int* __restrict__ output_indices,
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

    if (tid < num_rows) {
        uint32_t sk = suppkeys[tid];
        original_idx = input_indices[tid];

        int slot = hash_murmur3(sk, ht_s_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_s_size;
            if (ht_s_keys[s] == sk) {
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
    }
}

/**
 * @brief Stage 3: Probe customer hash table with input indices, get c_nation
 */
__global__ void probeCustomerWithIndicesQ41Kernel(
    const uint32_t* __restrict__ custkeys,
    const int* __restrict__ input_indices,
    int num_rows,
    const uint32_t* __restrict__ ht_c_keys,
    const uint32_t* __restrict__ ht_c_values,
    int ht_c_size,
    int* __restrict__ output_indices,
    uint32_t* __restrict__ output_c_nations,
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
    uint32_t c_nation = 0;

    if (tid < num_rows) {
        uint32_t ck = custkeys[tid];
        original_idx = input_indices[tid];

        int slot = hash_murmur3(ck, ht_c_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_c_size;
            if (ht_c_keys[s] == ck) {
                c_nation = ht_c_values[s];
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
        output_c_nations[global_pos] = c_nation;
    }
}

/**
 * @brief Stage 4: Final aggregation with date hash lookup
 */
__global__ void aggregateQ41StagedKernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ lo_supplycost,
    const uint32_t* __restrict__ passing_c_nations,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    long long* __restrict__ agg_profit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t od = lo_orderdate[i];
        uint32_t rev = lo_revenue[i];
        uint32_t cost = lo_supplycost[i];
        uint32_t c_nation = passing_c_nations[i];

        // Probe date hash table to get year
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool found = false;
        for (int probe = 0; probe < 32; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) {
                d_year = ht_d_values[slot];
                found = true;
                break;
            }
            if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }

        if (found) {
            int year_idx = d_year - 1992;
            if (year_idx >= 0 && year_idx < NUM_YEARS && c_nation < NUM_NATIONS) {
                int agg_idx = year_idx * NUM_NATIONS + c_nation;
                long long profit = static_cast<long long>(rev) - static_cast<long long>(cost);
                atomicAdd(reinterpret_cast<unsigned long long*>(&agg_profit[agg_idx]),
                         static_cast<unsigned long long>(profit));
            }
        }
    }
}

// Hash table build kernels
__global__ void build_date_ht_q41(const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t key = d_datekey[idx];
    uint32_t value = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

__global__ void build_customer_region_ht_q41(
    const uint32_t* c_custkey, const uint32_t* c_region, const uint32_t* c_nation,
    int num_rows, uint32_t filter_region,
    uint32_t* ht_keys, uint32_t* ht_values, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (c_region[idx] != filter_region) return;
    uint32_t key = c_custkey[idx];
    uint32_t value = c_nation[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

__global__ void build_supplier_region_ht_q41(
    const uint32_t* s_suppkey, const uint32_t* s_region,
    int num_rows, uint32_t filter_region,
    uint32_t* ht_keys, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

__global__ void build_part_mfgr_ht_q41(
    const uint32_t* p_partkey, const uint32_t* p_mfgr,
    int num_rows, uint32_t mfgr1, uint32_t mfgr2,
    uint32_t* ht_keys, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;
    uint32_t key = p_partkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

void runQ41RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);
    CompressedColumnAccessorVertical<uint32_t> acc_supplycost(&data.lo_supplycost);

    int total_rows = acc_custkey.getTotalElements();

    // Build dimension hash tables
    timer.start();

    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));

    build_date_ht_q41<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_c_size = C_LEN * 2;
    uint32_t *ht_c_keys, *ht_c_values;
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));

    build_customer_region_ht_q41<<<(C_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, C_REGION_AMERICA,
        ht_c_keys, ht_c_values, ht_c_size);

    int ht_s_size = S_LEN * 2;
    uint32_t* ht_s_keys;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_supplier_region_ht_q41<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_size);

    int ht_p_size = P_LEN * 2;
    uint32_t* ht_p_keys;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    build_part_mfgr_ht_q41<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_mfgr, P_LEN, P_MFGR_1, P_MFGR_2, ht_p_keys, ht_p_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== STAGED DECOMPOSITION ==========

    // === STAGE 1: Decompress partkey + probe part filter ===
    uint32_t* d_partkey;
    cudaMalloc(&d_partkey, total_rows * sizeof(uint32_t));

    timer.start();
    acc_partkey.decompressAll(d_partkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_partkey_ms = timer.elapsed_ms();

    int* d_stage1_indices;
    int* d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probePartFilterQ41Kernel<<<grid_stage1, block_size>>>(
        d_partkey, total_rows,
        ht_p_keys, ht_p_size,
        d_stage1_indices, d_num_stage1);
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
        std::cout << "\n=== Q4.1 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage1_indices);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_c_keys); cudaFree(ht_c_values);
        cudaFree(ht_s_keys); cudaFree(ht_p_keys);
        return;
    }

    // === STAGE 2: Random access suppkey + probe supplier filter ===
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, h_num_stage1 * sizeof(uint32_t));

    timer.start();
    acc_suppkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_suppkey_ms = timer.elapsed_ms();

    int* d_stage2_indices;
    int* d_num_stage2;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probeSupplierWithIndicesQ41Kernel<<<grid_stage2, block_size>>>(
        d_suppkey, d_stage1_indices, h_num_stage1,
        ht_s_keys, ht_s_size,
        d_stage2_indices, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_num_stage2);

    float stage2_selectivity = 100.0f * h_num_stage2 / total_rows;
    std::cout << "Stage 2 (supplier filter): " << stage2_selectivity << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q4.1 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage2_indices);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_c_keys); cudaFree(ht_c_values);
        cudaFree(ht_s_keys); cudaFree(ht_p_keys);
        return;
    }

    // === STAGE 3: Random access custkey + probe customer filter ===
    uint32_t* d_custkey;
    cudaMalloc(&d_custkey, h_num_stage2 * sizeof(uint32_t));

    timer.start();
    acc_custkey.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_custkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_custkey_ms = timer.elapsed_ms();

    int* d_stage3_indices;
    uint32_t* d_stage3_c_nations;
    int* d_num_stage3;
    cudaMalloc(&d_stage3_indices, h_num_stage2 * sizeof(int));
    cudaMalloc(&d_stage3_c_nations, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage3, sizeof(int));
    cudaMemset(d_num_stage3, 0, sizeof(int));

    timer.start();
    int grid_stage3 = (h_num_stage2 + block_size - 1) / block_size;
    probeCustomerWithIndicesQ41Kernel<<<grid_stage3, block_size>>>(
        d_custkey, d_stage2_indices, h_num_stage2,
        ht_c_keys, ht_c_values, ht_c_size,
        d_stage3_indices, d_stage3_c_nations, d_num_stage3);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage3_ms = timer.elapsed_ms();

    int h_num_stage3;
    cudaMemcpy(&h_num_stage3, d_num_stage3, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_custkey);
    cudaFree(d_stage2_indices);
    cudaFree(d_num_stage3);

    float stage3_selectivity = 100.0f * h_num_stage3 / total_rows;
    std::cout << "Stage 3 (customer filter): " << stage3_selectivity << "% (" << h_num_stage3 << " rows)" << std::endl;

    if (h_num_stage3 == 0) {
        std::cout << "\n=== Q4.1 Results (Staged Random Access) ===\nTotal: 0\n";
        cudaFree(d_stage3_indices); cudaFree(d_stage3_c_nations);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_c_keys); cudaFree(ht_c_values);
        cudaFree(ht_s_keys); cudaFree(ht_p_keys);
        return;
    }

    // === STAGE 4: Random access orderdate, revenue, supplycost + aggregate ===
    uint32_t *d_orderdate, *d_revenue, *d_supplycost;
    cudaMalloc(&d_orderdate, h_num_stage3 * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_stage3 * sizeof(uint32_t));
    cudaMalloc(&d_supplycost, h_num_stage3 * sizeof(uint32_t));

    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_stage3_indices, h_num_stage3, d_orderdate, stream);
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
    aggregateQ41StagedKernel<<<grid_agg, block_size>>>(
        d_orderdate, d_revenue, d_supplycost, d_stage3_c_nations, h_num_stage3,
        ht_d_keys, ht_d_values, ht_d_size, d_agg_profit);
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

    float data_load_total = decompress_partkey_ms + ra_suppkey_ms + ra_custkey_ms + ra_final_ms;
    float kernel_total = probe_stage1_ms + probe_stage2_ms + probe_stage3_ms + aggregate_ms;

    timing.data_load_ms = data_load_total;
    timing.kernel_ms = kernel_total;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q4.1 Results (Staged Random Access) ===" << std::endl;
    std::cout << "Groups: " << num_groups << ", Total profit: " << total_profit << std::endl;
    std::cout << "\nTiming breakdown (STAGED):" << std::endl;
    std::cout << "  Hash build:           " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Stage 1 decomp part:  " << decompress_partkey_ms << " ms" << std::endl;
    std::cout << "  Stage 1 probe part:   " << probe_stage1_ms << " ms (120M rows)" << std::endl;
    std::cout << "  Stage 2 RA suppkey:   " << ra_suppkey_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 2 probe supp:   " << probe_stage2_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 3 RA custkey:   " << ra_custkey_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Stage 3 probe cust:   " << probe_stage3_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Stage 4 RA final:     " << ra_final_ms << " ms (" << h_num_stage3 << " rows)" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q4.1");

    // Cleanup
    cudaFree(d_stage3_indices); cudaFree(d_stage3_c_nations);
    cudaFree(d_orderdate); cudaFree(d_revenue); cudaFree(d_supplycost);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_c_keys); cudaFree(ht_c_values);
    cudaFree(ht_s_keys); cudaFree(ht_p_keys);
    cudaFree(d_agg_profit);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q4.1 - Staged Random Access ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ41RandomAccess(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ41RandomAccess(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
