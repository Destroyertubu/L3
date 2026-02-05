#ifndef L3_TRANSPOSED_FORMAT_HPP
#define L3_TRANSPOSED_FORMAT_HPP

/**
 * L3 Transposed (Word-Interleaved) Format Specification
 *
 * This header defines data structures and constants for the Transposed version
 * of L3 compression. Key difference from Vertical (Lane-Major) format:
 *
 * VERTICAL (Lane-Major) - Current:
 *   Memory: [Lane0_W0][Lane0_W1]...[Lane0_WN][Lane1_W0][Lane1_W1]...[Lane31_WN]
 *   Access: Thread 0 reads addr 0, Thread 1 reads addr N+1 (STRIDED!)
 *
 * TRANSPOSED (Word-Interleaved) - New:
 *   Memory: [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *   Access: Thread 0 reads addr 0, Thread 1 reads addr 1 (COALESCED!)
 *
 * Benefits of Transposed layout:
 *   - Perfect memory coalescing: 32 threads read 32 consecutive words
 *   - Single 128-byte cache line transaction per warp read
 *   - 4x better L1 cache efficiency
 *
 * FORMAT_VERSION: 4.0 (Transposed/Word-Interleaved)
 * Date: 2025-12-16
 */

#include "L3_format.hpp"
#include "L3_opt.h"
#include "L3_codec.hpp"
#include "L3_Vertical_format.hpp"  // Reuse constants (MINI_VECTOR_SIZE, VALUES_PER_THREAD, etc.)
#include <cstdint>
#include <stdexcept>

// Format version for Transposed format
constexpr uint32_t L3_TRANSPOSED_FORMAT_VERSION = 0x00040000;  // v4.0.0
constexpr uint32_t L3_TRANSPOSED_MAGIC = 0x4C335458;  // "L3TX" in ASCII

// Mini-Vector Constants are inherited from L3_Vertical_format.hpp:
//   MINI_VECTOR_SIZE = 256
//   VALUES_PER_THREAD = 8
//   LANES_PER_MINI_VECTOR = 32
//   MAX_BIT_WIDTH = 64
//   REGISTER_BUFFER_WORDS = 4

// ============================================================================
// Memory Layout Enumeration
// ============================================================================

/**
 * Memory layout type for mini-vector storage
 */
enum class MemoryLayout : int32_t {
    LANE_MAJOR = 0,       // Vertical: [Lane0 all words][Lane1 all words]...
    WORD_INTERLEAVED = 1  // Transposed: [All lanes word0][All lanes word1]...
};

// ============================================================================
// Transposed Layout Structure
// ============================================================================

/**
 * Mini-Vector Transposed Layout (256 values)
 *
 * Word-Interleaved Layout:
 *   For 256 values, 32 lanes (warp threads), 8 values per lane:
 *
 *   Value-to-Lane assignment (same as Vertical):
 *     Lane 0:  v[0], v[32], v[64], v[96], v[128], v[160], v[192], v[224]
 *     Lane 1:  v[1], v[33], v[65], v[97], v[129], v[161], v[193], v[225]
 *     ...
 *     Lane 31: v[31], v[63], v[95], v[127], v[159], v[191], v[223], v[255]
 *
 *   Memory Storage (DIFFERENT from Vertical):
 *     Word 0:  Lane 0's word 0 (bits 0-31)
 *     Word 1:  Lane 1's word 0 (bits 0-31)
 *     ...
 *     Word 31: Lane 31's word 0 (bits 0-31)
 *     Word 32: Lane 0's word 1 (bits 32-63)
 *     Word 33: Lane 1's word 1 (bits 32-63)
 *     ...
 *
 *   Access Pattern for warp read of word W:
 *     Thread 0 reads word[W * 32 + 0]
 *     Thread 1 reads word[W * 32 + 1]
 *     Thread 2 reads word[W * 32 + 2]
 *     ...
 *     Thread 31 reads word[W * 32 + 31]
 *     → 32 CONSECUTIVE addresses = 128-byte coalesced transaction!
 */

/**
 * Transposed Compressed Data Container
 *
 * Uses Word-Interleaved layout for optimal GPU memory coalescing.
 */
template<typename T>
struct CompressedDataTransposed {
    // ========== Partition Info ==========
    int32_t num_partitions;
    int32_t total_values;

    // Partition metadata (device pointers)
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double*  d_model_params;           // [num_partitions * 4]
    int32_t* d_delta_bits;
    int64_t* d_error_bounds;

    // Predicate pushdown bounds (optional)
    T* d_partition_min_values;
    T* d_partition_max_values;

    // ========== Transposed Delta Storage ==========

    // Compressed deltas in word-interleaved format
    uint32_t* d_transposed_deltas;     // Bit-packed, word-interleaved layout
    int64_t   transposed_delta_words;  // Total words in transposed array

    // Per-partition transposed metadata
    int32_t* d_num_mini_vectors;       // [num_partitions] mini-vectors per partition
    int32_t* d_tail_sizes;             // [num_partitions] tail values per partition
    int64_t* d_transposed_offsets;     // [num_partitions] word offset into transposed array

    // Layout identifier
    MemoryLayout layout;

    // Kernel timing (ms)
    float kernel_time_ms;

    // Self-reference for device-side use
    CompressedDataTransposed<T>* d_self;

    // Constructor
    CompressedDataTransposed()
        : num_partitions(0), total_values(0),
          d_start_indices(nullptr), d_end_indices(nullptr),
          d_model_types(nullptr), d_model_params(nullptr),
          d_delta_bits(nullptr), d_error_bounds(nullptr),
          d_partition_min_values(nullptr), d_partition_max_values(nullptr),
          d_transposed_deltas(nullptr), transposed_delta_words(0),
          d_num_mini_vectors(nullptr), d_tail_sizes(nullptr),
          d_transposed_offsets(nullptr),
          layout(MemoryLayout::WORD_INTERLEAVED),
          kernel_time_ms(0.0f),
          d_self(nullptr) {}
};

// ============================================================================
// Encoding Configuration
// ============================================================================

/**
 * Transposed Compression Configuration
 */
struct TransposedConfig {
    // Base L3 config
    int partition_size_hint;
    int max_delta_bits;
    bool use_variable_partition;
    double error_bound_factor;

    // Transposed-specific options
    bool enable_branchless_unpack;
    int register_buffer_size;
    bool enable_adaptive_selection;
    int fixed_model_type;
    bool skip_metadata_recompute;

    // Partitioning strategy
    PartitioningStrategy partitioning_strategy;

    // Cost-optimal partitioning parameters
    int cost_analysis_block_size;
    int cost_min_partition_size;        // Initial partition size, merge from here
    int cost_max_partition_size;        // Maximum partition size after merging
    int cost_breakpoint_threshold;
    float cost_merge_benefit_threshold;
    int cost_max_merge_rounds;
    bool cost_enable_merging;

    // Default config
    TransposedConfig()
        : partition_size_hint(4096),
          max_delta_bits(64),
          use_variable_partition(true),
          error_bound_factor(1.0),
          enable_branchless_unpack(true),
          register_buffer_size(REGISTER_BUFFER_WORDS),
          enable_adaptive_selection(true),
          fixed_model_type(MODEL_LINEAR),
          skip_metadata_recompute(false),
          partitioning_strategy(PartitioningStrategy::FIXED),
          cost_analysis_block_size(2048),
          cost_min_partition_size(256),
          cost_max_partition_size(8192),
          cost_breakpoint_threshold(2),
          cost_merge_benefit_threshold(0.05f),
          cost_max_merge_rounds(4),
          cost_enable_merging(true) {}

    static TransposedConfig defaultConfig() {
        return TransposedConfig();
    }

    static TransposedConfig costOptimal() {
        TransposedConfig config;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        return config;
    }
};

// ============================================================================
// Utility Functions for Transposed Layout
// ============================================================================

/**
 * Calculate number of words per lane for given bit_width
 */
__host__ __device__ __forceinline__
int calcWordsPerLane(int bit_width) {
    int bits_per_lane = VALUES_PER_THREAD * bit_width;  // 8 * bit_width
    return (bits_per_lane + 31) / 32;
}

/**
 * Calculate total words for one mini-vector in transposed layout
 *
 * In transposed layout, words are interleaved:
 * Total words = words_per_lane * 32 (same total as Vertical, different arrangement)
 */
__host__ __device__ __forceinline__
int64_t calcTransposedMiniVectorWords(int bit_width) {
    return static_cast<int64_t>(calcWordsPerLane(bit_width)) * LANES_PER_MINI_VECTOR;
}

/**
 * Calculate words needed for transposed storage (mini-vectors + tail)
 */
__host__ __device__ __forceinline__
int64_t calcTransposedWords(int num_values, int bit_width) {
    int num_mini_vectors = num_values / MINI_VECTOR_SIZE;
    int tail_size = num_values % MINI_VECTOR_SIZE;

    // Mini-vectors use transposed layout
    int64_t mini_vector_words = static_cast<int64_t>(num_mini_vectors) *
                                 calcTransposedMiniVectorWords(bit_width);

    // Tail uses sequential packing (same as Vertical)
    int64_t tail_bits = static_cast<int64_t>(tail_size) * bit_width;
    int64_t tail_words = (tail_bits + 31) / 32;

    return mini_vector_words + tail_words;
}

/**
 * Calculate word address for TRANSPOSED (Word-Interleaved) layout
 *
 * Key difference from Vertical:
 *   Vertical:   lane_word_addr = mv_base + lane_id * words_per_lane + word_in_lane
 *   Transposed: lane_word_addr = mv_base + word_in_lane * 32 + lane_id
 *
 * This means:
 *   - Word W of all 32 lanes are stored consecutively
 *   - 32 threads reading word W access addresses [base + W*32 + 0..31]
 *   - Perfect 128-byte coalesced memory access!
 *
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Lane index (0-31, maps to warp thread)
 * @param word_in_lane  Which word within this lane's data (0 to words_per_lane-1)
 * @return Absolute word address in transposed array
 */
__device__ __forceinline__
int64_t calcTransposedWordAddr(int64_t mv_word_base, int lane_id, int word_in_lane) {
    // Word W of all 32 lanes are stored together
    // Word address for lane L, word W = base + W * 32 + L
    return mv_word_base + static_cast<int64_t>(word_in_lane) * LANES_PER_MINI_VECTOR + lane_id;
}

/**
 * Calculate bit offset for TRANSPOSED (Word-Interleaved) layout
 *
 * For value V in lane L:
 *   1. Calculate which word within lane's data: word_in_lane = (V * bit_width) / 32
 *   2. Calculate bit position within that word: bit_in_word = (V * bit_width) % 32
 *   3. Calculate transposed word address: base + word_in_lane * 32 + lane_id
 *   4. Final bit offset: word_addr * 32 + bit_in_word
 *
 * @param mv_word_base  Word offset where mini-vector starts
 * @param lane_id       Lane index (0-31)
 * @param value_idx     Value index within lane (0-7)
 * @param bit_width     Bits per delta value
 * @return Absolute bit offset in transposed array
 */
__device__ __forceinline__
int64_t calcTransposedBitOffset(int64_t mv_word_base, int lane_id, int value_idx, int bit_width) {
    // Calculate position within lane's bit stream
    int bit_in_lane = value_idx * bit_width;
    int word_in_lane = bit_in_lane / 32;
    int bit_in_word = bit_in_lane % 32;

    // Calculate transposed word address
    int64_t word_addr = calcTransposedWordAddr(mv_word_base, lane_id, word_in_lane);

    // Final bit offset
    return word_addr * 32 + bit_in_word;
}

/**
 * Calculate word address for tail values (sequential, after mini-vectors)
 *
 * Tail values use simple sequential packing (same as Vertical)
 */
__device__ __forceinline__
int64_t calcTransposedTailBitOffset(int64_t tail_word_base, int tail_idx, int bit_width) {
    return tail_word_base * 32 + static_cast<int64_t>(tail_idx) * bit_width;
}

/**
 * Map global index to mini-vector coordinates (same as Vertical)
 */
__device__ __forceinline__
void globalToMiniVector(int global_idx, int& mini_vector_idx, int& lane_id, int& value_idx) {
    mini_vector_idx = global_idx / MINI_VECTOR_SIZE;
    int local_idx = global_idx % MINI_VECTOR_SIZE;

    // Value at local_idx maps to lane (local_idx % 32) and value index (local_idx / 32)
    lane_id = local_idx % LANES_PER_MINI_VECTOR;
    value_idx = local_idx / LANES_PER_MINI_VECTOR;
}

/**
 * Map mini-vector coordinates to global index (same as Vertical)
 */
__device__ __forceinline__
int miniVectorToGlobal(int mini_vector_idx, int lane_id, int value_idx) {
    return mini_vector_idx * MINI_VECTOR_SIZE + value_idx * LANES_PER_MINI_VECTOR + lane_id;
}

#endif // L3_TRANSPOSED_FORMAT_HPP
