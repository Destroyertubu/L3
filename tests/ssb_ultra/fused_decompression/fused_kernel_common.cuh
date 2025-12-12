/**
 * @file fused_kernel_common.cuh
 * @brief Common Infrastructure for Optimized Fused Decompression Kernels
 *
 * This header provides:
 * - Configuration constants (thread coarsening)
 * - Shared memory metadata cache structures
 * - Warp shuffle reduction primitives
 * - Selection flag initialization and predicates
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_fused {

// ============================================================================
// Configuration Constants (Thread Coarsening)
// ============================================================================

constexpr int FUSED_BLOCK_SIZE = 128;
constexpr int FUSED_ITEMS_PER_THREAD = 4;
constexpr int FUSED_TILE_SIZE = FUSED_BLOCK_SIZE * FUSED_ITEMS_PER_THREAD;  // 512 items

// Maximum columns in fused query (Q4.x has 6: orderdate, custkey, suppkey, partkey, revenue, supplycost)
constexpr int MAX_FUSED_COLUMNS = 6;

// Maximum warps per block
constexpr int MAX_WARPS_PER_BLOCK = FUSED_BLOCK_SIZE / 32;  // 4 warps

// ============================================================================
// Partition Metadata Cache (Shared Memory)
// ============================================================================

// Guard to prevent redefinition in l3_decompress_device.cuh
#define L3_FUSED_PARTITION_METADATA_CACHE_DEFINED

/**
 * @brief Per-column partition metadata cached in shared memory
 *
 * Loaded once per block (by thread 0), then shared across all threads.
 * Avoids repeated global memory reads of the same metadata.
 */
struct PartitionMetadataCache {
    int32_t model_type;        // Polynomial model type
    double model_params[4];    // Polynomial coefficients (32 bytes)
    int32_t delta_bits;        // Bit width for deltas
    int64_t bit_offset_base;   // Starting bit offset in delta array
};

/**
 * @brief Full shared memory structure for fused kernels
 */
struct FusedKernelSharedMem {
    PartitionMetadataCache columns[MAX_FUSED_COLUMNS];  // ~300 bytes
    int32_t partition_size;                              // Elements in partition
    int32_t start_idx;                                   // Global start index

    // Warp reduction scratch space
    unsigned long long warp_sums[MAX_WARPS_PER_BLOCK];   // 32 bytes
    long long warp_sums_signed[MAX_WARPS_PER_BLOCK];     // 32 bytes (for profit)
};

// ============================================================================
// Warp Shuffle Reduction Primitives
// ============================================================================

/**
 * @brief Warp-level reduction using shuffle (unsigned long long)
 *
 * Reduces values within a warp using __shfl_down_sync.
 * Result is in lane 0 after completion.
 */
__device__ __forceinline__
unsigned long long warpReduceSum(unsigned long long val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level reduction using shuffle (signed long long for profit)
 */
__device__ __forceinline__
long long warpReduceSumSigned(long long val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Block-level reduction using warp shuffles
 *
 * More efficient than shared memory tree reduction:
 * 1. Intra-warp reduction (shuffle, no sync needed)
 * 2. Store warp results to shared memory
 * 3. Single warp does final reduction
 *
 * @param val       Thread's local value
 * @param smem_sums Shared memory for warp sums (size = num_warps)
 * @return          Reduced sum (valid only in thread 0)
 */
__device__ __forceinline__
unsigned long long blockReduceSum(unsigned long long val, unsigned long long* smem_sums) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // Intra-warp reduction
    val = warpReduceSum(val);

    // Lane 0 of each warp stores to shared memory
    if (lane_id == 0) {
        smem_sums[warp_id] = val;
    }
    __syncthreads();

    // First warp does final reduction
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem_sums[lane_id] : 0ULL;
        val = warpReduceSum(val);
    }

    return val;  // Only thread 0 has final result
}

/**
 * @brief Block-level reduction for signed values (profit aggregation)
 */
__device__ __forceinline__
long long blockReduceSumSigned(long long val, long long* smem_sums) {
    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    val = warpReduceSumSigned(val);

    if (lane_id == 0) {
        smem_sums[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? smem_sums[lane_id] : 0LL;
        val = warpReduceSumSigned(val);
    }

    return val;
}

// ============================================================================
// Early Termination Check
// ============================================================================

/**
 * @brief Check if all selection flags are zero (early termination)
 *
 * Uses __syncthreads_or to do block-level check efficiently.
 * Returns true if ALL threads have count=0, meaning no rows pass filters.
 */
template<int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
bool IsTerm(const int (&selection_flags)[IPT]) {
    int count = 0;
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        count += selection_flags[i];
    }
    // Block-level OR: returns 0 only if ALL threads have count=0
    return __syncthreads_or(count) == 0;
}

// ============================================================================
// Selection Flag Primitives
// ============================================================================

/**
 * @brief Initialize selection flags for a tile
 *
 * Sets flags to 1 for valid items, 0 for out-of-bounds items.
 * Uses strided access pattern for coalescing.
 */
template<int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void InitFlags(int (&selection_flags)[IPT], int num_items) {
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int idx = tid + i * FUSED_BLOCK_SIZE;
        selection_flags[i] = (idx < num_items) ? 1 : 0;
    }
}

/**
 * @brief Predicate: value >= lo AND value <= hi (inclusive range)
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockPredAndBetween(
    const T (&values)[IPT],
    T lo, T hi,
    int (&selection_flags)[IPT])
{
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            selection_flags[i] = (values[i] >= lo && values[i] <= hi) ? 1 : 0;
        }
    }
}

/**
 * @brief Predicate: value < threshold
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockPredAndLT(
    const T (&values)[IPT],
    T threshold,
    int (&selection_flags)[IPT])
{
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            selection_flags[i] = (values[i] < threshold) ? 1 : 0;
        }
    }
}

/**
 * @brief Predicate: value == target
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockPredAndEQ(
    const T (&values)[IPT],
    T target,
    int (&selection_flags)[IPT])
{
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            selection_flags[i] = (values[i] == target) ? 1 : 0;
        }
    }
}

/**
 * @brief Predicate: value >= threshold
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockPredAndGE(
    const T (&values)[IPT],
    T threshold,
    int (&selection_flags)[IPT])
{
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            selection_flags[i] = (values[i] >= threshold) ? 1 : 0;
        }
    }
}

/**
 * @brief Predicate: value <= threshold
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockPredAndLE(
    const T (&values)[IPT],
    T threshold,
    int (&selection_flags)[IPT])
{
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            selection_flags[i] = (values[i] <= threshold) ? 1 : 0;
        }
    }
}

// ============================================================================
// Metadata Loading
// ============================================================================

/**
 * @brief Load partition metadata to shared memory (single-column)
 *
 * Called by thread 0 only. Other threads wait at __syncthreads().
 */
__device__ __forceinline__
void loadColumnMetadata(
    PartitionMetadataCache& cache,
    int partition_id,
    const int32_t* model_types,
    const double* model_params,
    const int32_t* delta_bits,
    const int64_t* bit_offsets)
{
    cache.model_type = model_types[partition_id];
    cache.delta_bits = delta_bits[partition_id];
    cache.bit_offset_base = bit_offsets[partition_id];

    // Load 4 params (could use vector load for better perf)
    int param_base = partition_id * 4;
    cache.model_params[0] = model_params[param_base];
    cache.model_params[1] = model_params[param_base + 1];
    cache.model_params[2] = model_params[param_base + 2];
    cache.model_params[3] = model_params[param_base + 3];
}

// ============================================================================
// Custom Atomics
// ============================================================================

/**
 * @brief AtomicAdd for signed 64-bit (profit aggregation)
 */
__device__ __forceinline__
long long atomicAdd_int64(long long* address, long long val) {
    return (long long)atomicAdd((unsigned long long*)address, (unsigned long long)val);
}

} // namespace l3_fused
