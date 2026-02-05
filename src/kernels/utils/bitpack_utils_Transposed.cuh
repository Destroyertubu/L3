#ifndef BITPACK_UTILS_TRANSPOSED_CUH
#define BITPACK_UTILS_TRANSPOSED_CUH

/**
 * Transposed (Word-Interleaved) Bit-Packing Utilities
 *
 * This header provides utilities for the Transposed memory layout.
 * Most bit manipulation functions are reused from bitpack_utils_Vertical.cuh.
 *
 * KEY DIFFERENCE from Vertical:
 *   - Vertical loads: Thread L reads words at stride = words_per_lane
 *   - Transposed loads: Thread L reads words at consecutive addresses (coalesced!)
 *
 * Memory Layout Comparison:
 *   Vertical:   [L0_W0][L0_W1]...[L0_WN][L1_W0][L1_W1]...[L31_WN]
 *   Transposed: [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *
 * Date: 2025-12-16
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "bitpack_utils_Vertical.cuh"  // Reuse mask and extraction utilities
#include "L3_Transposed_format.hpp"

namespace Transposed {

// Reuse constants and utilities from Vertical namespace
using Vertical::WARP_SIZE;
using Vertical::CACHE_LINE_BYTES;
using Vertical::CACHE_LINE_WORDS;
using Vertical::mask64;
using Vertical::mask32;
using Vertical::mask64_rt;
using Vertical::mask32_rt;
using Vertical::sign_extend_32;
using Vertical::sign_extend_64;
using Vertical::extract_branchless_64;
using Vertical::extract_branchless_64_rt;
using Vertical::extract_branchless_32;

// ============================================================================
// TRANSPOSED COALESCED LOADING
// ============================================================================

/**
 * Load words in coalesced pattern for one mini-vector
 *
 * This is the KEY optimization of Transposed layout:
 * - All 32 threads read consecutive memory addresses
 * - Single 128-byte cache line transaction per iteration
 *
 * Memory access pattern:
 *   Iteration w=0: Thread 0 reads addr[0], Thread 1 reads addr[1], ..., Thread 31 reads addr[31]
 *   Iteration w=1: Thread 0 reads addr[32], Thread 1 reads addr[33], ..., Thread 31 reads addr[63]
 *   ...
 *
 * Compare to Vertical (strided):
 *   Thread 0 reads addr[0, 1, 2, ...]
 *   Thread 1 reads addr[W, W+1, W+2, ...] where W = words_per_lane (STRIDED!)
 *
 * @param transposed_array  Base pointer to transposed data
 * @param mv_word_base      Word offset where mini-vector starts
 * @param lane_id           Thread's lane ID (0-31)
 * @param words_per_lane    Number of words each lane has
 * @param lane_words        Output: array to store lane's words (in register)
 * @param max_words         Maximum words to load
 */
template<int MAX_WORDS = 8>
__device__ __forceinline__
void loadCoalescedWords(
    const uint32_t* __restrict__ transposed_array,
    int64_t mv_word_base,
    int lane_id,
    int words_per_lane,
    uint32_t* lane_words)
{
    // Load words in coalesced pattern
    // Word W for lane L is at: mv_word_base + W * 32 + L
    #pragma unroll
    for (int w = 0; w < MAX_WORDS; w++) {
        if (w < words_per_lane) {
            // COALESCED: All 32 threads access consecutive addresses!
            // Thread 0 reads mv_word_base + w*32 + 0
            // Thread 1 reads mv_word_base + w*32 + 1
            // Thread 2 reads mv_word_base + w*32 + 2
            // ... etc
            int64_t addr = mv_word_base + static_cast<int64_t>(w) * WARP_SIZE + lane_id;
            lane_words[w] = __ldg(&transposed_array[addr]);
        } else {
            lane_words[w] = 0;
        }
    }
}

/**
 * Load words with dynamic count (runtime words_per_lane)
 */
__device__ __forceinline__
void loadCoalescedWordsDynamic(
    const uint32_t* __restrict__ transposed_array,
    int64_t mv_word_base,
    int lane_id,
    int words_per_lane,
    uint32_t* lane_words,
    int max_words)
{
    for (int w = 0; w < max_words; w++) {
        if (w < words_per_lane) {
            int64_t addr = mv_word_base + static_cast<int64_t>(w) * WARP_SIZE + lane_id;
            lane_words[w] = __ldg(&transposed_array[addr]);
        } else {
            lane_words[w] = 0;
        }
    }
}

// ============================================================================
// TRANSPOSED BIT EXTRACTION FROM REGISTER BUFFER
// ============================================================================

/**
 * Extract value from pre-loaded lane words buffer
 *
 * This extracts a value at a given bit position from the lane_words buffer
 * that was loaded using loadCoalescedWords.
 *
 * @param lane_words   Array of words belonging to this lane (in registers)
 * @param bit_in_lane  Starting bit position within lane's data
 * @param bit_width    Width of value to extract
 * @return Extracted value
 */
__device__ __forceinline__
uint64_t extractFromBuffer(
    const uint32_t* lane_words,
    int bit_in_lane,
    int bit_width)
{
    int word_idx = bit_in_lane >> 5;      // bit_in_lane / 32
    int bit_in_word = bit_in_lane & 31;   // bit_in_lane % 32

    // Combine two consecutive words for crossing extraction
    uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                       lane_words[word_idx];
    uint64_t extracted = (combined >> bit_in_word);

    // Handle 3-word extraction for delta_bits > 32 with bit_in_word > 0
    if (bit_width > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < bit_width) {
        int bits_from_first_two = 64 - bit_in_word;
        extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
    }

    return extracted & mask64_rt(bit_width);
}

/**
 * Extract with compile-time bit width for common cases
 */
template<int BITS>
__device__ __forceinline__
uint64_t extractFromBufferCT(
    const uint32_t* lane_words,
    int bit_in_lane)
{
    static_assert(BITS >= 0 && BITS <= 64, "BITS out of range");

    if constexpr (BITS == 0) {
        return 0ULL;
    }

    int word_idx = bit_in_lane >> 5;
    int bit_in_word = bit_in_lane & 31;

    uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                       lane_words[word_idx];
    uint64_t extracted = (combined >> bit_in_word);

    if constexpr (BITS > 32) {
        // May need third word
        if (bit_in_word > 0 && (32 - bit_in_word + 32) < BITS) {
            int bits_from_first_two = 64 - bit_in_word;
            extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
        }
    }

    return extracted & mask64<BITS>();
}

// ============================================================================
// TRANSPOSED BIT PACKING (FOR ENCODER)
// ============================================================================

/**
 * Pack a value into the transposed array at given position
 *
 * This packs a value into the word-interleaved layout.
 * Uses atomicOr for thread-safe writes.
 *
 * @param transposed_array  Target array
 * @param mv_word_base      Word offset where mini-vector starts
 * @param lane_id           Lane ID (0-31)
 * @param bit_in_lane       Bit position within lane's data
 * @param value             Value to pack
 * @param bit_width         Width of value
 */
__device__ __forceinline__
void packToTransposed(
    uint32_t* transposed_array,
    int64_t mv_word_base,
    int lane_id,
    int bit_in_lane,
    uint64_t value,
    int bit_width)
{
    int word_in_lane = bit_in_lane / 32;
    int bit_in_word = bit_in_lane % 32;

    // Calculate transposed word address
    int64_t word_addr = mv_word_base + static_cast<int64_t>(word_in_lane) * WARP_SIZE + lane_id;

    // First word
    int bits_in_first = min(bit_width, 32 - bit_in_word);
    uint32_t mask_first = (bits_in_first == 32) ? ~0U : ((1U << bits_in_first) - 1U);
    uint32_t val_first = static_cast<uint32_t>(value & mask_first) << bit_in_word;
    atomicOr(&transposed_array[word_addr], val_first);

    int bits_remaining = bit_width - bits_in_first;
    if (bits_remaining > 0) {
        // Second word (at next word_in_lane, same transposed pattern)
        int64_t word_addr_2 = mv_word_base + static_cast<int64_t>(word_in_lane + 1) * WARP_SIZE + lane_id;
        uint64_t shifted = value >> bits_in_first;
        int bits_in_second = min(bits_remaining, 32);
        uint32_t mask_second = (bits_in_second == 32) ? ~0U : ((1U << bits_in_second) - 1U);
        uint32_t val_second = static_cast<uint32_t>(shifted & mask_second);
        atomicOr(&transposed_array[word_addr_2], val_second);

        bits_remaining -= bits_in_second;
        if (bits_remaining > 0) {
            // Third word (for delta_bits > 32 crossing 2 word boundaries)
            int64_t word_addr_3 = mv_word_base + static_cast<int64_t>(word_in_lane + 2) * WARP_SIZE + lane_id;
            uint32_t val_third = static_cast<uint32_t>(shifted >> 32);
            atomicOr(&transposed_array[word_addr_3], val_third);
        }
    }
}

} // namespace Transposed

#endif // BITPACK_UTILS_TRANSPOSED_CUH
