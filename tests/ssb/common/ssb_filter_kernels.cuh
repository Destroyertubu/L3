/**
 * @file ssb_filter_kernels.cuh
 * @brief Stream Compaction Filter Kernels for SSB Predicate Pushdown
 *
 * Provides GPU kernels for filtering data and generating compacted index arrays.
 * Used for "filter column full decompress + non-filter column random access" optimization.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace ssb {

// ============================================================================
// Constants
// ============================================================================

constexpr int FILTER_BLOCK_SIZE = 256;
constexpr int FILTER_WARP_SIZE = 32;

// ============================================================================
// Warp-level Primitives
// ============================================================================

/**
 * @brief Count bits set in a warp using ballot
 */
__device__ __forceinline__ int warpPopcount(int predicate) {
    unsigned int ballot = __ballot_sync(0xffffffff, predicate);
    return __popc(ballot);
}

/**
 * @brief Compute exclusive prefix sum within a warp
 */
__device__ __forceinline__ int warpExclusiveScan(int value) {
    unsigned int ballot = __ballot_sync(0xffffffff, value);
    int lane = threadIdx.x & 31;
    // Mask out bits at and above current lane
    unsigned int mask = (1u << lane) - 1;
    return __popc(ballot & mask);
}

// ============================================================================
// Stream Compaction Filter Kernels
// ============================================================================

/**
 * @brief Single-predicate date range filter with stream compaction
 *
 * Filters rows where date is in [date_min, date_max] and outputs
 * compacted indices of passing rows.
 *
 * Uses warp-level ballot + block-level atomic for efficient compaction.
 *
 * @param dates Input date array
 * @param num_rows Number of rows
 * @param date_min Minimum date (inclusive)
 * @param date_max Maximum date (inclusive)
 * @param passing_indices Output: indices of rows passing filter
 * @param num_passing Output: count of passing rows (atomic counter)
 */
__global__ void filterDateRangeCompactKernel(
    const uint32_t* __restrict__ dates,
    int num_rows,
    uint32_t date_min,
    uint32_t date_max,
    int* __restrict__ passing_indices,
    int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // Shared memory for warp-level counts and block offset
    __shared__ int s_warp_counts[8];  // Max 8 warps per block
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    // Step 1: Each thread evaluates predicate
    int passes = 0;
    if (tid < num_rows) {
        uint32_t date = dates[tid];
        passes = (date >= date_min && date <= date_max) ? 1 : 0;
    }

    // Step 2: Warp-level ballot to count passing threads
    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    // Lane 0 of each warp stores count
    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    // Step 3: Thread 0 computes warp offsets and reserves block space
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        // Atomically reserve space in global output
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    // Step 4: Each passing thread writes its index
    if (passes && tid < num_rows) {
        // Compute position within warp
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
    }
}

/**
 * @brief Multi-predicate filter for Q1.1 (date + discount + quantity)
 *
 * Applies all three predicates and outputs compacted indices.
 * This is useful when all filter columns are already decompressed.
 *
 * Predicates:
 * - date IN [date_min, date_max]
 * - discount IN [discount_min, discount_max]
 * - quantity < quantity_max
 *
 * @param lo_orderdate Date column
 * @param lo_discount Discount column
 * @param lo_quantity Quantity column
 * @param num_rows Number of rows
 * @param date_min, date_max Date range
 * @param discount_min, discount_max Discount range
 * @param quantity_max Quantity threshold
 * @param passing_indices Output: indices of rows passing ALL predicates
 * @param num_passing Output: count of passing rows
 */
__global__ void filterQ11CompactKernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    int num_rows,
    uint32_t date_min, uint32_t date_max,
    uint32_t discount_min, uint32_t discount_max,
    uint32_t quantity_max,
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

    // Evaluate all predicates
    int passes = 0;
    if (tid < num_rows) {
        uint32_t date = lo_orderdate[tid];
        uint32_t discount = lo_discount[tid];
        uint32_t quantity = lo_quantity[tid];

        passes = (date >= date_min && date <= date_max &&
                  discount >= discount_min && discount <= discount_max &&
                  quantity < quantity_max) ? 1 : 0;
    }

    // Warp-level ballot
    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    // Compute offsets and reserve space
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    // Write passing indices
    if (passes && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
    }
}

/**
 * @brief Two-stage filter: First filter by date, output indices
 *
 * Stage 1 of optimized pipeline:
 * - Full decompress lo_orderdate
 * - Filter by date range
 * - Output date_passing_indices
 *
 * Then use random access to get lo_discount, lo_quantity for secondary filtering.
 */
__global__ void filterDateOnlyCompactKernel(
    const uint32_t* __restrict__ lo_orderdate,
    int num_rows,
    uint32_t date_min, uint32_t date_max,
    int* __restrict__ passing_indices,
    int* __restrict__ num_passing)
{
    // This is identical to filterDateRangeCompactKernel
    // Keeping separate for clarity and potential future optimizations

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    int passes = 0;
    if (tid < num_rows) {
        uint32_t date = lo_orderdate[tid];
        passes = (date >= date_min && date <= date_max) ? 1 : 0;
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    if (passes && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
    }
}

/**
 * @brief Secondary filter on already-filtered data
 *
 * Stage 2: Apply discount/quantity predicates on random-accessed data
 * Input arrays are already filtered (only date-passing rows)
 *
 * @param lo_discount Discount values for date-passing rows
 * @param lo_quantity Quantity values for date-passing rows
 * @param input_indices Original global indices of date-passing rows
 * @param num_rows Number of date-passing rows
 * @param discount_min, discount_max Discount range
 * @param quantity_max Quantity threshold
 * @param passing_indices Output: global indices passing ALL predicates
 * @param num_passing Output: count of final passing rows
 */
__global__ void filterSecondaryCompactKernel(
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    const int* __restrict__ input_indices,
    int num_rows,
    uint32_t discount_min, uint32_t discount_max,
    uint32_t quantity_max,
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

    int passes = 0;
    int original_idx = -1;
    if (tid < num_rows) {
        uint32_t discount = lo_discount[tid];
        uint32_t quantity = lo_quantity[tid];
        original_idx = input_indices[tid];

        passes = (discount >= discount_min && discount <= discount_max &&
                  quantity < quantity_max) ? 1 : 0;
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    if (passes && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = original_idx;  // Store ORIGINAL global index
    }
}

/**
 * @brief Simple aggregation kernel for already-filtered data
 *
 * Computes SUM(extendedprice * discount) for rows that passed all filters.
 * No additional filtering needed - all inputs are pre-filtered.
 *
 * @param lo_extendedprice Extended price for passing rows
 * @param lo_discount Discount for passing rows
 * @param num_rows Number of passing rows
 * @param revenue Output: sum of revenue
 */
__global__ void aggregateRevenueKernel(
    const uint32_t* __restrict__ lo_extendedprice,
    const uint32_t* __restrict__ lo_discount,
    int num_rows,
    unsigned long long* __restrict__ revenue)
{
    __shared__ unsigned long long s_sum[8];  // One per warp

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    // Each thread accumulates its portion
    unsigned long long local_sum = 0;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        local_sum += static_cast<unsigned long long>(lo_extendedprice[i]) *
                     static_cast<unsigned long long>(lo_discount[i]);
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Lane 0 stores to shared memory
    if (lane == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (warp_id == 0) {
        local_sum = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane == 0 && local_sum > 0) {
            atomicAdd(revenue, local_sum);
        }
    }
}

/**
 * @brief Aggregation with secondary filtering (discount + quantity check)
 *
 * For cases where lo_extendedprice is random-accessed but discount/quantity
 * haven't been fully filtered yet.
 */
__global__ void aggregateRevenueWithFilterKernel(
    const uint32_t* __restrict__ lo_extendedprice,
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    int num_rows,
    uint32_t discount_min, uint32_t discount_max,
    uint32_t quantity_max,
    unsigned long long* __restrict__ revenue)
{
    __shared__ unsigned long long s_sum[8];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    unsigned long long local_sum = 0;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t discount = lo_discount[i];
        uint32_t quantity = lo_quantity[i];

        if (discount >= discount_min && discount <= discount_max &&
            quantity < quantity_max) {
            local_sum += static_cast<unsigned long long>(lo_extendedprice[i]) *
                         static_cast<unsigned long long>(discount);
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        local_sum = (lane < (blockDim.x >> 5)) ? s_sum[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (lane == 0 && local_sum > 0) {
            atomicAdd(revenue, local_sum);
        }
    }
}

// ============================================================================
// Host Helper Functions
// ============================================================================

/**
 * @brief Launch date range filter with stream compaction
 */
inline void launchDateRangeFilter(
    const uint32_t* d_dates,
    int num_rows,
    uint32_t date_min,
    uint32_t date_max,
    int* d_passing_indices,
    int* d_num_passing,
    cudaStream_t stream = 0)
{
    int grid_size = (num_rows + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
    filterDateRangeCompactKernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_dates, num_rows, date_min, date_max, d_passing_indices, d_num_passing);
}

/**
 * @brief Launch Q1.1 multi-predicate filter
 */
inline void launchQ11Filter(
    const uint32_t* d_orderdate,
    const uint32_t* d_discount,
    const uint32_t* d_quantity,
    int num_rows,
    uint32_t date_min, uint32_t date_max,
    uint32_t discount_min, uint32_t discount_max,
    uint32_t quantity_max,
    int* d_passing_indices,
    int* d_num_passing,
    cudaStream_t stream = 0)
{
    int grid_size = (num_rows + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
    filterQ11CompactKernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_orderdate, d_discount, d_quantity, num_rows,
        date_min, date_max, discount_min, discount_max, quantity_max,
        d_passing_indices, d_num_passing);
}

/**
 * @brief Launch revenue aggregation on pre-filtered data
 */
inline void launchRevenueAggregation(
    const uint32_t* d_extendedprice,
    const uint32_t* d_discount,
    int num_rows,
    unsigned long long* d_revenue,
    cudaStream_t stream = 0)
{
    int grid_size = min((num_rows + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 256);
    aggregateRevenueKernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_extendedprice, d_discount, num_rows, d_revenue);
}

}  // namespace ssb
