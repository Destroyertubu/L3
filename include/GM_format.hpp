/**
 * GM Format - True Interleaved (Word-Interleaved) Layout
 *
 * GM = Generalized Memory-coalesced format
 *
 * This header re-exports the Transposed format with GM naming convention.
 * The Transposed format implements the true FastLanes-style interleaved layout:
 *
 * Memory Layout (Word-Interleaved / Row-Major):
 *   [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *
 * Memory Access Pattern:
 *   Thread 0 reads addr[base + W*32 + 0]
 *   Thread 1 reads addr[base + W*32 + 1]
 *   ...
 *   Thread 31 reads addr[base + W*32 + 31]
 *
 *   --> 32 threads read 32 consecutive addresses --> Perfect 128-byte coalesced!
 *
 * Date: 2025-01-26
 */

#ifndef GM_FORMAT_HPP
#define GM_FORMAT_HPP

#include "L3_Transposed_format.hpp"

// ============================================================================
// GM Type Aliases
// ============================================================================

/**
 * GM Compressed Data Container
 *
 * Alias for CompressedDataTransposed, using Word-Interleaved layout
 * for optimal GPU memory coalescing during decompression.
 */
template<typename T>
using CompressedDataGM = CompressedDataTransposed<T>;

/**
 * GM Configuration
 *
 * Alias for TransposedConfig with all the same options.
 */
using GMConfig = TransposedConfig;

/**
 * GM Memory Layout Constant
 */
constexpr MemoryLayout GM_LAYOUT = MemoryLayout::WORD_INTERLEAVED;

// ============================================================================
// GM Constants (inherited from Transposed/Vertical)
// ============================================================================

// Mini-Vector Configuration:
//   MINI_VECTOR_SIZE = 256 (values per mini-vector)
//   VALUES_PER_THREAD = 8 (values per lane/thread)
//   LANES_PER_MINI_VECTOR = 32 (warp size)
//   MAX_BIT_WIDTH = 64

// ============================================================================
// GM Layout Utility Functions
// ============================================================================

namespace GM {

/**
 * Calculate number of words per lane for given bit_width
 *
 * @param bit_width  Bits per delta value
 * @return Number of 32-bit words needed per lane
 */
__host__ __device__ __forceinline__
int wordsPerLane(int bit_width) {
    return calcWordsPerLane(bit_width);
}

/**
 * Calculate total words for one mini-vector in GM layout
 *
 * @param bit_width  Bits per delta value
 * @return Total words for one mini-vector
 */
__host__ __device__ __forceinline__
int64_t wordsPerMiniVector(int bit_width) {
    return calcTransposedMiniVectorWords(bit_width);
}

/**
 * Calculate word address in GM (Word-Interleaved) layout
 *
 * Key formula:
 *   word_addr = mv_word_base + word_in_lane * 32 + lane_id
 *
 * This ensures all 32 threads' word[W] are stored consecutively,
 * enabling coalesced memory access!
 *
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Lane index (0-31, maps to warp thread)
 * @param word_in_lane  Which word within this lane's data (0 to words_per_lane-1)
 * @return Absolute word address in GM array
 */
__device__ __forceinline__
int64_t wordAddress(int64_t mv_word_base, int lane_id, int word_in_lane) {
    return calcTransposedWordAddr(mv_word_base, lane_id, word_in_lane);
}

/**
 * Calculate bit offset in GM (Word-Interleaved) layout
 *
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Lane index (0-31)
 * @param value_idx     Value index within lane (0-7)
 * @param bit_width     Bits per delta value
 * @return Absolute bit offset in GM array
 */
__device__ __forceinline__
int64_t bitOffset(int64_t mv_word_base, int lane_id, int value_idx, int bit_width) {
    return calcTransposedBitOffset(mv_word_base, lane_id, value_idx, bit_width);
}

} // namespace GM

#endif // GM_FORMAT_HPP
