/**
 * @file hierarchical_agg_v8.cuh
 * @brief Hierarchical aggregation helpers using warp shuffle + shared memory
 *
 * V8 Key Optimization: Reduce global atomics by 32x using local aggregation
 *
 * Strategy:
 * 1. Warp shuffle reduction to combine 32 threads
 * 2. Shared memory for block-local aggregation (hash-based buckets)
 * 3. Single flush to global at block exit
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_v8 {

// ============================================================================
// Configuration
// ============================================================================

// Aggregation dimensions for Q2.1
constexpr int AGG_NUM_YEARS_V8 = 7;      // 1992-1998
constexpr int AGG_NUM_BRANDS_V8 = 1000;  // Brand codes 0-999

// Local aggregation hash table size (power of 2 for efficient modulo)
constexpr int LOCAL_AGG_SLOTS = 256;  // Shared memory slots per block

// ============================================================================
// Warp-Level Reduction Helpers
// ============================================================================

/**
 * @brief Warp shuffle reduction for summing revenues
 *
 * Reduces 32 partial sums to single sum in lane 0.
 * Each thread contributes its local sum for a specific (year, brand_bucket) key.
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
 * @brief Warp shuffle to broadcast value from lane 0 to all lanes
 */
__device__ __forceinline__
unsigned long long warpBroadcast(unsigned long long val, int srcLane = 0) {
    return __shfl_sync(0xFFFFFFFF, val, srcLane);
}

// ============================================================================
// Hash-Based Local Aggregation Structure
// ============================================================================

/**
 * @brief Local aggregation slot in shared memory
 *
 * Uses open addressing with linear probing.
 * Key = (year << 16) | brand for compact encoding.
 */
struct LocalAggSlot {
    uint32_t key;           // Packed (year, brand) - 0 means empty
    unsigned long long sum;  // Running sum of revenues
};

/**
 * @brief Compute hash for (year, brand) pair
 */
__device__ __forceinline__
int aggHash(int year, int brand) {
    // Simple multiplicative hash
    uint32_t key = (year << 16) | brand;
    return (key * 2654435769u) >> (32 - 8);  // Use top 8 bits for 256 slots
}

/**
 * @brief Pack (year, brand) into single key
 */
__device__ __forceinline__
uint32_t packKey(int year, int brand) {
    return (static_cast<uint32_t>(year) << 16) | static_cast<uint32_t>(brand);
}

/**
 * @brief Unpack year from key
 */
__device__ __forceinline__
int unpackYear(uint32_t key) {
    return static_cast<int>(key >> 16);
}

/**
 * @brief Unpack brand from key
 */
__device__ __forceinline__
int unpackBrand(uint32_t key) {
    return static_cast<int>(key & 0xFFFF);
}

// ============================================================================
// Local Aggregation Operations
// ============================================================================

/**
 * @brief Initialize shared memory aggregation slots
 *
 * Call at block start with all threads participating.
 */
__device__ __forceinline__
void initLocalAgg(LocalAggSlot* smem_agg, int tid, int block_size) {
    for (int i = tid; i < LOCAL_AGG_SLOTS; i += block_size) {
        smem_agg[i].key = 0;
        smem_agg[i].sum = 0;
    }
}

/**
 * @brief Add revenue to local aggregation (shared memory)
 *
 * Uses atomic operations on shared memory (much faster than global).
 * Falls back to direct global atomic on collision overflow.
 */
__device__ __forceinline__
void localAggregate(
    LocalAggSlot* smem_agg,
    int year, int brand,
    unsigned long long revenue,
    unsigned long long* global_agg)  // Fallback for overflow
{
    if (revenue == 0) return;

    uint32_t key = packKey(year, brand);
    int slot = aggHash(year, brand);

    // Linear probing with limited attempts
    #pragma unroll
    for (int probe = 0; probe < 4; probe++) {
        int idx = (slot + probe) & (LOCAL_AGG_SLOTS - 1);

        // Try to claim or match slot
        uint32_t old_key = atomicCAS(&smem_agg[idx].key, 0, key);

        if (old_key == 0 || old_key == key) {
            // Slot is ours - add revenue
            atomicAdd(&smem_agg[idx].sum, revenue);
            return;
        }
    }

    // Overflow: fall back to global atomic
    int global_idx = year * AGG_NUM_BRANDS_V8 + brand;
    atomicAdd(&global_agg[global_idx], revenue);
}

/**
 * @brief Flush local aggregation to global result
 *
 * Call at block end with syncthreads before and after.
 */
__device__ __forceinline__
void flushLocalAgg(
    LocalAggSlot* smem_agg,
    int tid, int block_size,
    unsigned long long* global_agg)
{
    for (int i = tid; i < LOCAL_AGG_SLOTS; i += block_size) {
        uint32_t key = smem_agg[i].key;
        unsigned long long sum = smem_agg[i].sum;

        if (key != 0 && sum > 0) {
            int year = unpackYear(key);
            int brand = unpackBrand(key);
            int global_idx = year * AGG_NUM_BRANDS_V8 + brand;
            atomicAdd(&global_agg[global_idx], sum);
        }
    }
}

// ============================================================================
// Simplified Direct Local Aggregation (Alternative)
// ============================================================================

/**
 * @brief Direct warp-level aggregation without shared memory hash table
 *
 * For cases where most threads have unique (year, brand) keys,
 * use warp shuffle to find duplicates and combine them.
 */
__device__ __forceinline__
void warpAggregateSimple(
    int year, int brand, unsigned long long revenue,
    unsigned long long* global_agg,
    bool valid)
{
    if (!valid || revenue == 0) return;

    // Each thread has its own (year, brand, revenue)
    // Use warp shuffle to find threads with same key and combine

    int lane_id = threadIdx.x & 31;
    uint32_t key = packKey(year, brand);

    // Find lowest lane with same key
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, key);
    int leader = __ffs(match_mask) - 1;

    if (leader == lane_id) {
        // I'm the leader for this key - sum all matching lanes
        unsigned long long total = 0;
        unsigned int mask = match_mask;
        while (mask) {
            int src = __ffs(mask) - 1;
            total += __shfl_sync(match_mask, revenue, src);
            mask &= (mask - 1);  // Clear lowest bit
        }

        // Single atomic per unique key in warp
        int global_idx = year * AGG_NUM_BRANDS_V8 + brand;
        atomicAdd(&global_agg[global_idx], total);
    }
}

}  // namespace l3_v8
