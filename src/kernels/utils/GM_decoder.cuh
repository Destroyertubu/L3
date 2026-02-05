/**
 * GM Decoder - True Interleaved (Word-Interleaved) Decoding
 *
 * GM = Generalized Memory-coalesced format
 *
 * This header provides high-level decoding functions for GM (Word-Interleaved) layout.
 * Based on bitpack_utils_Transposed.cuh with a simplified API.
 *
 * Key Feature: COALESCED MEMORY ACCESS
 *   - 32 threads read 32 consecutive words in each iteration
 *   - Single 128-byte cache line transaction per warp read
 *
 * Memory Layout:
 *   GM:       [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *   Vertical: [L0_W0][L0_W1]...[L0_WN][L1_W0][L1_W1]...[L31_WN]
 *
 * Date: 2025-01-26
 */

#ifndef GM_DECODER_CUH
#define GM_DECODER_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "bitpack_utils_Transposed.cuh"

namespace GM_decoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int VALUES_PER_THREAD = 8;
constexpr int MINI_VECTOR_SIZE = 256;

// ============================================================================
// Compile-Time Bit Width Decoding (Optimal Performance)
// ============================================================================

/**
 * Decode mini-vector values with compile-time bit width (FOR_BITPACK model)
 *
 * This is the core decoding function for GM layout.
 * Uses coalesced memory access for optimal GPU performance.
 *
 * @tparam BIT_WIDTH    Compile-time bit width
 * @param data          Compressed data array (GM layout)
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Thread's lane ID (0-31)
 * @param base_value    Base value for FOR_BITPACK model
 * @param out           Output array (8 values)
 */
template<int BIT_WIDTH>
__device__ __forceinline__
void decode(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    // Calculate words per lane
    constexpr int BITS_PER_LANE = VALUES_PER_THREAD * BIT_WIDTH;
    constexpr int WORDS_PER_LANE = (BITS_PER_LANE + 31) / 32;

    // Mask for extracting values
    constexpr uint32_t MASK = (BIT_WIDTH == 32) ? ~0U : ((1U << BIT_WIDTH) - 1U);

    // ========== COALESCED LOAD ==========
    // All 32 threads read consecutive addresses!
    // Thread L reads: mv_word_base + W * 32 + L
    uint32_t lane_words[WORDS_PER_LANE + 1];

    #pragma unroll
    for (int w = 0; w <= WORDS_PER_LANE; w++) {
        // GM Layout: word[w * 32 + lane_id]
        int64_t addr = mv_word_base + static_cast<int64_t>(w) * WARP_SIZE + lane_id;
        lane_words[w] = __ldg(&data[addr]);
    }

    // ========== EXTRACT VALUES ==========
    int local_bit = 0;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        int word_idx = local_bit >> 5;       // local_bit / 32
        int bit_in_word = local_bit & 31;    // local_bit % 32

        // Combine two words and extract
        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint32_t delta = static_cast<uint32_t>((combined >> bit_in_word) & MASK);

        out[v] = base_value + delta;
        local_bit += BIT_WIDTH;
    }
}

/**
 * Decode mini-vector values with compile-time bit width (64-bit output)
 */
template<int BIT_WIDTH>
__device__ __forceinline__
void decode64(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint64_t base_value,
    uint64_t (&out)[VALUES_PER_THREAD])
{
    constexpr int BITS_PER_LANE = VALUES_PER_THREAD * BIT_WIDTH;
    constexpr int WORDS_PER_LANE = (BITS_PER_LANE + 31) / 32;
    constexpr uint64_t MASK = (BIT_WIDTH == 64) ? ~0ULL : ((1ULL << BIT_WIDTH) - 1ULL);

    uint32_t lane_words[WORDS_PER_LANE + 2];

    #pragma unroll
    for (int w = 0; w <= WORDS_PER_LANE + 1; w++) {
        if (w <= WORDS_PER_LANE) {
            int64_t addr = mv_word_base + static_cast<int64_t>(w) * WARP_SIZE + lane_id;
            lane_words[w] = __ldg(&data[addr]);
        } else {
            lane_words[w] = 0;
        }
    }

    int local_bit = 0;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint64_t extracted = (combined >> bit_in_word);

        // Handle 3-word extraction for BIT_WIDTH > 32
        if constexpr (BIT_WIDTH > 32) {
            if (bit_in_word > 0 && (64 - bit_in_word) < BIT_WIDTH) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
            }
        }

        out[v] = base_value + (extracted & MASK);
        local_bit += BIT_WIDTH;
    }
}

// ============================================================================
// Runtime Bit Width Decoding (Flexible)
// ============================================================================

/**
 * Decode mini-vector values with runtime bit width
 *
 * @param data          Compressed data array (GM layout)
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Thread's lane ID (0-31)
 * @param base_value    Base value for FOR_BITPACK model
 * @param bit_width     Runtime bit width
 * @param out           Output array (8 values)
 */
__device__ __forceinline__
void decodeRuntime(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    int bit_width,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    // Calculate words per lane (runtime)
    int bits_per_lane = VALUES_PER_THREAD * bit_width;
    int words_per_lane = (bits_per_lane + 31) / 32;

    // Mask for extracting values
    uint32_t mask = (bit_width == 32) ? ~0U : ((1U << bit_width) - 1U);

    // Use Transposed coalesced loading
    constexpr int MAX_WORDS = 18;  // Max for 64-bit, 8 values
    uint32_t lane_words[MAX_WORDS];

    Transposed::loadCoalescedWordsDynamic(
        data,
        mv_word_base,
        lane_id,
        words_per_lane + 1,
        lane_words,
        MAX_WORDS
    );

    // Extract values
    int local_bit = 0;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint32_t delta = static_cast<uint32_t>((combined >> bit_in_word) & mask);

        out[v] = base_value + delta;
        local_bit += bit_width;
    }
}

// ============================================================================
// Tail Value Decoding (Sequential)
// ============================================================================

/**
 * Decode a single tail value (sequential packing)
 *
 * Tail values (partition_size % 256) use sequential packing
 * since there's no warp-level pattern to exploit.
 *
 * @param data          Compressed data array
 * @param tail_bit_base Bit offset where tail data starts
 * @param tail_idx      Index within tail (0 to tail_size-1)
 * @param base_value    Base value for FOR_BITPACK model
 * @param bit_width     Bit width
 * @return Decoded value
 */
template<typename T = uint32_t>
__device__ __forceinline__
T decodeTail(
    const uint32_t* __restrict__ data,
    int64_t tail_bit_base,
    int tail_idx,
    T base_value,
    int bit_width)
{
    if (bit_width == 0) {
        return base_value;
    }

    int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_idx) * bit_width;
    uint64_t extracted = Transposed::extractFromBuffer(
        data + (bit_offset >> 5),
        static_cast<int>(bit_offset & 31),
        bit_width
    );

    return base_value + static_cast<T>(extracted);
}

// ============================================================================
// Specialized Decoders for Common Bit Widths
// ============================================================================

/**
 * Decode with BIT_WIDTH = 16 (common for SSB queries)
 */
__device__ __forceinline__
void decode16(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    decode<16>(data, mv_word_base, lane_id, base_value, out);
}

/**
 * Decode with BIT_WIDTH = 6 (quantity column)
 */
__device__ __forceinline__
void decode6(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    decode<6>(data, mv_word_base, lane_id, base_value, out);
}

/**
 * Decode with BIT_WIDTH = 4 (discount column)
 */
__device__ __forceinline__
void decode4(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    decode<4>(data, mv_word_base, lane_id, base_value, out);
}

/**
 * Decode with BIT_WIDTH = 20 (partkey/custkey columns)
 */
__device__ __forceinline__
void decode20(
    const uint32_t* __restrict__ data,
    int64_t mv_word_base,
    int lane_id,
    uint32_t base_value,
    uint32_t (&out)[VALUES_PER_THREAD])
{
    decode<20>(data, mv_word_base, lane_id, base_value, out);
}

} // namespace GM_decoder

#endif // GM_DECODER_CUH
