/**
 * @file prefetch_helpers_v5.cuh
 * @brief Shared Prefetch Helpers for L3 V5 Fused Decompression
 *
 * Key optimizations (from L3 standalone decoder):
 * 1. Prefetch all needed words to registers BEFORE extraction
 * 2. Extract from registers (zero latency)
 * 3. One warp per tile, each thread handles 8 values per mini-vector
 *
 * L3 Interleaved Format (256-value mini-vectors):
 *   Lane L has values at indices: L, L+32, L+64, L+96, L+128, L+160, L+192, L+224
 *   Bits are stored: Lane 0's 8 values, Lane 1's 8 values, ..., Lane 31's 8 values
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_v5 {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SZ = 32;
constexpr int VALUES_PER_LANE = 8;  // L3 interleaved: 8 values per lane in mini-vector
constexpr int MINI_VEC_SIZE = 256;
// Note: TILE_SIZE is defined in tile_metadata.cuh (l3_fused namespace)

// ============================================================================
// Prefetch-based Delta Extraction (L3 Native Format)
// ============================================================================

/**
 * Prefetch all words needed for this lane's 8 values from a mini-vector,
 * then extract from registers. This matches L3 standalone decoder's strategy.
 *
 * @param data        L3 interleaved delta array
 * @param mv_bit_base Bit offset to start of this mini-vector in partition
 * @param lane_id     Thread's lane ID (0-31)
 * @param delta_bits  Bit width for deltas
 * @param base        FOR base value
 * @param output      Output array (8 values)
 */
__device__ __forceinline__
void prefetchExtract8FOR(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane_id,
    int delta_bits,
    uint32_t base,
    uint32_t (&output)[8])
{
    if (delta_bits == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            output[i] = base;
        }
        return;
    }

    // Lane's bit position within mini-vector
    // Bits layout: Lane0[8*bw], Lane1[8*bw], ..., Lane31[8*bw]
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_LANE * delta_bits;

    // Calculate words needed (same as L3 standalone decoder)
    int64_t lane_word_start = lane_bit_start >> 5;
    int bits_per_lane = VALUES_PER_LANE * delta_bits;
    int words_needed = (bits_per_lane + 31 + 32) / 32;  // +32 for alignment padding
    words_needed = min(words_needed, 20);

    // Prefetch to registers (KEY OPTIMIZATION!)
    uint32_t lane_words[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        lane_words[i] = (i < words_needed) ? __ldg(&data[lane_word_start + i]) : 0;
    }

    // Extract from registers (zero latency)
    int local_bit = lane_bit_start & 31;
    uint64_t mask = (1ULL << delta_bits) - 1;

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint32_t delta = (combined >> bit_in_word) & mask;

        output[v] = base + delta;
        local_bit += delta_bits;
    }
}

/**
 * Conditional prefetch - only extract values where flags are set.
 */
__device__ __forceinline__
void prefetchExtract8FORConditional(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane_id,
    int delta_bits,
    uint32_t base,
    const int (&flags)[8],
    uint32_t (&output)[8])
{
    // Check if any flag is set (branchless OR)
    int any_set = flags[0] | flags[1] | flags[2] | flags[3] |
                  flags[4] | flags[5] | flags[6] | flags[7];
    if (!any_set) return;

    if (delta_bits == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (flags[i]) output[i] = base;
        }
        return;
    }

    // Lane's bit position within mini-vector
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_LANE * delta_bits;

    // Prefetch to registers (always do for uniform execution)
    int64_t lane_word_start = lane_bit_start >> 5;
    int bits_per_lane = VALUES_PER_LANE * delta_bits;
    int words_needed = (bits_per_lane + 31 + 32) / 32;
    words_needed = min(words_needed, 20);

    uint32_t lane_words[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        lane_words[i] = (i < words_needed) ? __ldg(&data[lane_word_start + i]) : 0;
    }

    // Extract only where needed
    int local_bit = lane_bit_start & 31;
    uint64_t mask = (1ULL << delta_bits) - 1;

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            output[v] = base + ((combined >> bit_in_word) & mask);
        }
        local_bit += delta_bits;
    }
}

// ============================================================================
// Warp Reduction Helper
// ============================================================================

__device__ __forceinline__
unsigned long long warpReduceSum(unsigned long long val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__
long long warpReduceSumSigned(long long val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ============================================================================
// Mini-Vector Bit Offset Helper
// ============================================================================

/**
 * Compute bit offset for a mini-vector within a partition
 */
__device__ __forceinline__
int64_t computeMVBitBase(int64_t partition_bit_base, int local_mv_idx, int delta_bits) {
    return partition_bit_base + static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * delta_bits;
}

} // namespace l3_v5
