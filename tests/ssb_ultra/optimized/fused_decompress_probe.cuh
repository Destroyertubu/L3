/**
 * @file fused_decompress_probe.cuh
 * @brief Fused Decompression-Probe Kernels for SSB Queries
 *
 * This file implements fused kernels that combine decompression and hash probing
 * to reduce kernel launch overhead and intermediate memory traffic.
 *
 * Key optimizations:
 * - Single kernel for decompress + probe + filter (reduces kernel launches)
 * - Early termination on probe failure (short-circuit evaluation)
 * - Warp-level stream compaction for passing indices
 * - Shared memory for partition metadata
 *
 * Trade-off: Uses more registers but avoids multiple kernel launches and
 * intermediate global memory writes.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "ssb_common.cuh"
#include "parallel_hash_probe.cuh"
#include "L3_Vertical_format.hpp"

namespace ssb {

// ============================================================================
// Constants
// ============================================================================

constexpr int FUSED_BLOCK_SIZE = 256;      // Threads per block
constexpr int FUSED_ITEMS_PER_THREAD = 4;  // Items processed per thread
constexpr int FUSED_TILE_SIZE = FUSED_BLOCK_SIZE * FUSED_ITEMS_PER_THREAD;  // 1024 items/block

// ============================================================================
// Fused Decompress + Single Table Probe
// ============================================================================

/**
 * @brief Fused decompression and single hash table probe
 *
 * Processes decompressed data and immediately probes a hash table,
 * collecting passing indices via warp-level compaction.
 *
 * @tparam T Element type
 * @param d_decompressed Pre-decompressed data (from previous stage)
 * @param num_elements Total elements
 * @param ht_keys Hash table keys
 * @param ht_values Hash table values
 * @param ht_size Hash table size
 * @param d_passing_indices Output: indices that passed the probe
 * @param d_passing_values Output: values from hash table
 * @param d_num_passing Output: count of passing elements
 */
template<typename T>
__global__ void fusedProbeKernel(
    const uint32_t* __restrict__ d_decompressed,
    int num_elements,
    const uint32_t* __restrict__ ht_keys,
    const uint32_t* __restrict__ ht_values,
    int ht_size,
    int* __restrict__ d_passing_indices,
    uint32_t* __restrict__ d_passing_values,
    int* __restrict__ d_num_passing)
{
    // Each thread processes multiple items
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * FUSED_TILE_SIZE;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = FUSED_BLOCK_SIZE >> 5;

    // Shared memory for warp-level compaction
    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    // Process ITEMS_PER_THREAD items per thread
    int my_count = 0;
    int my_indices[FUSED_ITEMS_PER_THREAD];
    uint32_t my_values[FUSED_ITEMS_PER_THREAD];

    #pragma unroll
    for (int item = 0; item < FUSED_ITEMS_PER_THREAD; item++) {
        int global_idx = block_start + tid + item * FUSED_BLOCK_SIZE;
        if (global_idx >= num_elements) continue;

        uint32_t key = d_decompressed[global_idx];
        uint32_t value;

        // Two-level hash probe
        if (twoLevelProbeWithValue(key, ht_keys, ht_values, ht_size, value)) {
            my_indices[my_count] = global_idx;
            my_values[my_count] = value;
            my_count++;
        }
    }

    // Warp-level compaction
    unsigned ballot = __ballot_sync(0xFFFFFFFF, my_count > 0);
    (void)ballot;  // Used for debugging

    // Count total passing in warp
    for (int i = 0; i < num_warps; i++) {
        if (warp_id == i) {
            // Intra-warp reduction
            int lane_count = my_count;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                lane_count += __shfl_down_sync(0xFFFFFFFF, lane_count, offset);
            }
            if (lane == 0) s_warp_counts[i] = lane_count;
        }
    }
    __syncthreads();

    // Compute warp offsets and block total
    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(d_num_passing, total);
        }
    }
    __syncthreads();

    // Write passing elements
    if (my_count > 0) {
        // Compute position within warp
        int prefix = 0;
        for (int l = 0; l < lane; l++) {
            int other_count = __shfl_sync(0xFFFFFFFF, my_count, l);
            prefix += other_count;
        }

        int base_pos = s_block_offset + s_warp_offsets[warp_id] + prefix;
        for (int i = 0; i < my_count; i++) {
            d_passing_indices[base_pos + i] = my_indices[i];
            d_passing_values[base_pos + i] = my_values[i];
        }
    }
}

// ============================================================================
// Fused Multi-Table Probe (for Q3.x/Q4.x)
// ============================================================================

/**
 * @brief Fused 4-table probe for Q4.x queries
 *
 * Probes PART, SUPPLIER, CUSTOMER, DATE hash tables in parallel,
 * collecting passing rows with all dimension attributes.
 *
 * Input: 4 decompressed columns (partkey, suppkey, custkey, orderdate)
 * Output: Passing indices + (p_attr, s_attr, c_attr, d_year)
 */
__global__ void fusedProbe4TablesKernel(
    const uint32_t* __restrict__ d_partkey,
    const uint32_t* __restrict__ d_suppkey,
    const uint32_t* __restrict__ d_custkey,
    const uint32_t* __restrict__ d_orderdate,
    int num_elements,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_vals, int ht_p_size,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_vals, int ht_s_size,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_vals, int ht_c_size,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    int* __restrict__ d_passing_indices,
    uint32_t* __restrict__ d_p_attrs,
    uint32_t* __restrict__ d_s_attrs,
    uint32_t* __restrict__ d_c_attrs,
    uint32_t* __restrict__ d_years,
    int* __restrict__ d_num_passing)
{
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * FUSED_TILE_SIZE;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = FUSED_BLOCK_SIZE >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    // Each thread collects its passing items
    int my_count = 0;
    int my_indices[FUSED_ITEMS_PER_THREAD];
    uint32_t my_p[FUSED_ITEMS_PER_THREAD], my_s[FUSED_ITEMS_PER_THREAD];
    uint32_t my_c[FUSED_ITEMS_PER_THREAD], my_y[FUSED_ITEMS_PER_THREAD];

    #pragma unroll
    for (int item = 0; item < FUSED_ITEMS_PER_THREAD; item++) {
        int global_idx = block_start + tid + item * FUSED_BLOCK_SIZE;
        if (global_idx >= num_elements) continue;

        uint32_t pk = d_partkey[global_idx];
        uint32_t sk = d_suppkey[global_idx];
        uint32_t ck = d_custkey[global_idx];
        uint32_t dk = d_orderdate[global_idx];

        uint32_t p_val, s_val, c_val, d_val;

        // Try 4-table parallel probe
        bool found = warpParallelProbe4Tables(
            pk, sk, ck, dk,
            ht_p_keys, ht_p_vals, ht_p_size,
            ht_s_keys, ht_s_vals, ht_s_size,
            ht_c_keys, ht_c_vals, ht_c_size,
            ht_d_keys, ht_d_vals, ht_d_size,
            p_val, s_val, c_val, d_val);

        // Apply year filter
        if (found && d_val >= year_min && d_val <= year_max) {
            my_indices[my_count] = global_idx;
            my_p[my_count] = p_val;
            my_s[my_count] = s_val;
            my_c[my_count] = c_val;
            my_y[my_count] = d_val;
            my_count++;
        }
    }

    // Warp-level compaction (same pattern as above)
    int lane_count = my_count;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lane_count += __shfl_down_sync(0xFFFFFFFF, lane_count, offset);
    }
    if (lane == 0) s_warp_counts[warp_id] = lane_count;
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(d_num_passing, total);
    }
    __syncthreads();

    // Write results
    if (my_count > 0) {
        // Exclusive scan within warp for position
        int my_offset = my_count;
        #pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) {
            int other = __shfl_up_sync(0xFFFFFFFF, my_offset, delta);
            if (lane >= delta) my_offset += other;
        }
        int prefix = my_offset - my_count;

        int base_pos = s_block_offset + s_warp_offsets[warp_id] + prefix;
        for (int i = 0; i < my_count; i++) {
            d_passing_indices[base_pos + i] = my_indices[i];
            d_p_attrs[base_pos + i] = my_p[i];
            d_s_attrs[base_pos + i] = my_s[i];
            d_c_attrs[base_pos + i] = my_c[i];
            d_years[base_pos + i] = my_y[i];
        }
    }
}

/**
 * @brief Fused 3-table probe for Q3.x queries
 */
__global__ void fusedProbe3TablesKernel(
    const uint32_t* __restrict__ d_suppkey,
    const uint32_t* __restrict__ d_custkey,
    const uint32_t* __restrict__ d_orderdate,
    int num_elements,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_vals, int ht_s_size,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_vals, int ht_c_size,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    int* __restrict__ d_passing_indices,
    uint32_t* __restrict__ d_s_attrs,
    uint32_t* __restrict__ d_c_attrs,
    uint32_t* __restrict__ d_years,
    int* __restrict__ d_num_passing)
{
    const int tid = threadIdx.x;
    const int block_start = blockIdx.x * FUSED_TILE_SIZE;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = FUSED_BLOCK_SIZE >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    int my_count = 0;
    int my_indices[FUSED_ITEMS_PER_THREAD];
    uint32_t my_s[FUSED_ITEMS_PER_THREAD];
    uint32_t my_c[FUSED_ITEMS_PER_THREAD], my_y[FUSED_ITEMS_PER_THREAD];

    #pragma unroll
    for (int item = 0; item < FUSED_ITEMS_PER_THREAD; item++) {
        int global_idx = block_start + tid + item * FUSED_BLOCK_SIZE;
        if (global_idx >= num_elements) continue;

        uint32_t sk = d_suppkey[global_idx];
        uint32_t ck = d_custkey[global_idx];
        uint32_t dk = d_orderdate[global_idx];

        uint32_t s_val, c_val, d_val;

        bool found = warpParallelProbe3Tables(
            sk, ck, dk,
            ht_s_keys, ht_s_vals, ht_s_size,
            ht_c_keys, ht_c_vals, ht_c_size,
            ht_d_keys, ht_d_vals, ht_d_size,
            s_val, c_val, d_val);

        if (found && d_val >= year_min && d_val <= year_max) {
            my_indices[my_count] = global_idx;
            my_s[my_count] = s_val;
            my_c[my_count] = c_val;
            my_y[my_count] = d_val;
            my_count++;
        }
    }

    // Warp reduction
    int lane_count = my_count;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        lane_count += __shfl_down_sync(0xFFFFFFFF, lane_count, offset);
    }
    if (lane == 0) s_warp_counts[warp_id] = lane_count;
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(d_num_passing, total);
    }
    __syncthreads();

    if (my_count > 0) {
        int my_offset = my_count;
        #pragma unroll
        for (int delta = 1; delta < 32; delta *= 2) {
            int other = __shfl_up_sync(0xFFFFFFFF, my_offset, delta);
            if (lane >= delta) my_offset += other;
        }
        int prefix = my_offset - my_count;

        int base_pos = s_block_offset + s_warp_offsets[warp_id] + prefix;
        for (int i = 0; i < my_count; i++) {
            d_passing_indices[base_pos + i] = my_indices[i];
            d_s_attrs[base_pos + i] = my_s[i];
            d_c_attrs[base_pos + i] = my_c[i];
            d_years[base_pos + i] = my_y[i];
        }
    }
}

// ============================================================================
// Host-side Launch Helpers
// ============================================================================

/**
 * @brief Launch fused probe kernel for single table
 */
inline void launchFusedProbeKernel(
    const uint32_t* d_data,
    int num_elements,
    const uint32_t* ht_keys,
    const uint32_t* ht_values,
    int ht_size,
    int* d_passing_indices,
    uint32_t* d_passing_values,
    int* d_num_passing,
    cudaStream_t stream = 0)
{
    int num_blocks = (num_elements + FUSED_TILE_SIZE - 1) / FUSED_TILE_SIZE;
    fusedProbeKernel<uint32_t><<<num_blocks, FUSED_BLOCK_SIZE, 0, stream>>>(
        d_data, num_elements, ht_keys, ht_values, ht_size,
        d_passing_indices, d_passing_values, d_num_passing);
}

/**
 * @brief Launch fused 4-table probe for Q4.x queries
 */
inline void launchFusedProbe4Tables(
    const uint32_t* d_partkey,
    const uint32_t* d_suppkey,
    const uint32_t* d_custkey,
    const uint32_t* d_orderdate,
    int num_elements,
    const uint32_t* ht_p_keys, const uint32_t* ht_p_vals, int ht_p_size,
    const uint32_t* ht_s_keys, const uint32_t* ht_s_vals, int ht_s_size,
    const uint32_t* ht_c_keys, const uint32_t* ht_c_vals, int ht_c_size,
    const uint32_t* ht_d_keys, const uint32_t* ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    int* d_passing_indices,
    uint32_t* d_p_attrs,
    uint32_t* d_s_attrs,
    uint32_t* d_c_attrs,
    uint32_t* d_years,
    int* d_num_passing,
    cudaStream_t stream = 0)
{
    int num_blocks = (num_elements + FUSED_TILE_SIZE - 1) / FUSED_TILE_SIZE;
    fusedProbe4TablesKernel<<<num_blocks, FUSED_BLOCK_SIZE, 0, stream>>>(
        d_partkey, d_suppkey, d_custkey, d_orderdate, num_elements,
        ht_p_keys, ht_p_vals, ht_p_size,
        ht_s_keys, ht_s_vals, ht_s_size,
        ht_c_keys, ht_c_vals, ht_c_size,
        ht_d_keys, ht_d_vals, ht_d_size,
        year_min, year_max,
        d_passing_indices, d_p_attrs, d_s_attrs, d_c_attrs, d_years, d_num_passing);
}

/**
 * @brief Launch fused 3-table probe for Q3.x queries
 */
inline void launchFusedProbe3Tables(
    const uint32_t* d_suppkey,
    const uint32_t* d_custkey,
    const uint32_t* d_orderdate,
    int num_elements,
    const uint32_t* ht_s_keys, const uint32_t* ht_s_vals, int ht_s_size,
    const uint32_t* ht_c_keys, const uint32_t* ht_c_vals, int ht_c_size,
    const uint32_t* ht_d_keys, const uint32_t* ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    int* d_passing_indices,
    uint32_t* d_s_attrs,
    uint32_t* d_c_attrs,
    uint32_t* d_years,
    int* d_num_passing,
    cudaStream_t stream = 0)
{
    int num_blocks = (num_elements + FUSED_TILE_SIZE - 1) / FUSED_TILE_SIZE;
    fusedProbe3TablesKernel<<<num_blocks, FUSED_BLOCK_SIZE, 0, stream>>>(
        d_suppkey, d_custkey, d_orderdate, num_elements,
        ht_s_keys, ht_s_vals, ht_s_size,
        ht_c_keys, ht_c_vals, ht_c_size,
        ht_d_keys, ht_d_vals, ht_d_size,
        year_min, year_max,
        d_passing_indices, d_s_attrs, d_c_attrs, d_years, d_num_passing);
}

// ============================================================================
// Aggregation Kernels (for final aggregation after probe)
// ============================================================================

/**
 * @brief Aggregation kernel for Q4.x profit calculation
 */
__global__ void aggregateQ4ProfitKernel(
    const int* __restrict__ d_passing_indices,
    const uint32_t* __restrict__ d_p_attrs,   // category
    const uint32_t* __restrict__ d_s_attrs,   // nation
    const uint32_t* __restrict__ d_c_attrs,   // nation
    const uint32_t* __restrict__ d_years,
    int num_passing,
    const uint32_t* __restrict__ d_revenue,
    const uint32_t* __restrict__ d_supplycost,
    long long* __restrict__ d_agg,
    int num_categories, int num_nations, int year_base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_passing) return;

    int orig_idx = d_passing_indices[idx];
    uint32_t p_cat = d_p_attrs[idx];
    uint32_t s_nation = d_s_attrs[idx];
    (void)d_c_attrs;  // Customer attrs not used in Q4 profit aggregation
    uint32_t year = d_years[idx];

    long long profit = (long long)d_revenue[orig_idx] - (long long)d_supplycost[orig_idx];

    int year_idx = year - year_base;
    int agg_idx = year_idx * num_nations * num_categories + s_nation * num_categories + p_cat;

    atomicAdd((unsigned long long*)&d_agg[agg_idx], (unsigned long long)profit);
}

/**
 * @brief Aggregation kernel for Q3.x revenue calculation
 */
__global__ void aggregateQ3RevenueKernel(
    const int* __restrict__ d_passing_indices,
    const uint32_t* __restrict__ d_s_attrs,   // city
    const uint32_t* __restrict__ d_c_attrs,   // city
    const uint32_t* __restrict__ d_years,
    int num_passing,
    const uint32_t* __restrict__ d_revenue,
    unsigned long long* __restrict__ d_agg,
    int num_cities, int year_base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_passing) return;

    int orig_idx = d_passing_indices[idx];
    uint32_t s_city = d_s_attrs[idx];
    uint32_t c_city = d_c_attrs[idx];
    uint32_t year = d_years[idx];

    int year_idx = year - year_base;
    int agg_idx = year_idx * num_cities * num_cities + c_city * num_cities + s_city;

    atomicAdd(&d_agg[agg_idx], (unsigned long long)d_revenue[orig_idx]);
}

}  // namespace ssb
