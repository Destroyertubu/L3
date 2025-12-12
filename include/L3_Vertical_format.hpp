#ifndef L3_Vertical_FORMAT_HPP
#define L3_Vertical_FORMAT_HPP

/**
 * L3 Vertical-Optimized Format Specification
 *
 * This header defines data structures and constants for the Vertical-optimized
 * version of L3 compression. Key optimizations:
 *
 * 1. Branchless unpacking - Eliminates conditional branches in bit extraction
 * 2. Register buffering - Reduces memory transactions via prefetching
 * 3. Mini-vector interleaved encoding (256 values) - SIMD-friendly layout
 * 4. INTERLEAVED-ONLY storage - Single format for both scan and random access
 *
 * FORMAT_VERSION: 3.0 (Interleaved-only)
 * Date: 2025-12-07
 *
 * STORAGE FORMAT: Interleaved deltas contain:
 *   - Mini-vectors (256 values each, interleaved layout)
 *   - Tail values (partition_size % 256, sequential layout after mini-vectors)
 */

#include "L3_format.hpp"
#include "L3.h"
#include "L3_codec.hpp"  // For PartitioningStrategy
#include <cstdint>
#include <stdexcept>

// Format version for Vertical-optimized format
constexpr uint32_t L3_Vertical_FORMAT_VERSION = 0x00030000;  // v3.0.0
constexpr uint32_t L3_Vertical_MAGIC = 0x4C334653;  // "L3FS" in ASCII

// ============================================================================
// Mini-Vector Constants (Inspired by Vertical GPU paper)
// ============================================================================

// Mini-vector size: 256 values (8 values per thread * 32 threads/warp)
constexpr int MINI_VECTOR_SIZE = 256;
constexpr int VALUES_PER_THREAD = 8;
constexpr int LANES_PER_MINI_VECTOR = 32;  // Warp size

// Threshold: minimum partition size for interleaved encoding (legacy, not used)
constexpr int INTERLEAVED_THRESHOLD = 512;

// Maximum supported bit width
constexpr int MAX_BIT_WIDTH = 64;

// Register buffer size for prefetching
constexpr int REGISTER_BUFFER_WORDS = 4;

// ============================================================================
// Interleaved Layout Structure
// ============================================================================

/**
 * Mini-Vector Interleaved Layout (256 values)
 *
 * Interleaved Layout (Vertical-inspired):
 *   For 256 values, 32 lanes (warp threads), 8 values per lane:
 *
 *   Lane 0:  v[0], v[32], v[64], v[96], v[128], v[160], v[192], v[224]
 *   Lane 1:  v[1], v[33], v[65], v[97], v[129], v[161], v[193], v[225]
 *   ...
 *   Lane 31: v[31], v[63], v[95], v[127], v[159], v[191], v[223], v[255]
 *
 *   Benefits:
 *   - Coalesced memory access (32 threads read consecutive addresses)
 *   - No bank conflicts in shared memory
 *   - Warp-level parallelism in unpacking
 *
 * Bit-level layout within each lane:
 *   For bit_width=w, lane L contains:
 *   Bits [0, w*8): 8 packed deltas for indices L, L+32, L+64, ...
 *
 *   Example (w=5, lane 0):
 *   Bits 0-4:   delta[0]
 *   Bits 5-9:   delta[32]
 *   Bits 10-14: delta[64]
 *   ...
 *   Bits 35-39: delta[224]
 *   Total bits per lane: 5 * 8 = 40 bits
 *
 * PARTITION STORAGE FORMAT (v3.0):
 *   [Mini-Vector 0][Mini-Vector 1]...[Mini-Vector N-1][Tail Values]
 *
 *   Where:
 *   - N = partition_size / 256 (number of complete mini-vectors)
 *   - Tail size = partition_size % 256 (sequential packed after mini-vectors)
 */

/**
 * Interleaved Partition Data
 *
 * Stores a partition's deltas in interleaved format (mini-vectors + tail).
 */
struct InterleavedPartitionData {
    // Interleaved format (mini-vectors + tail)
    uint32_t* interleaved_deltas;     // Bit-packed, interleaved layout
    int64_t   interleaved_word_offset;// Word offset into global interleaved array
    int32_t   num_mini_vectors;       // Number of 256-value mini-vectors
    int32_t   tail_size;              // Remaining values (< 256)

    // Common metadata
    int32_t   partition_size;         // Total values in partition
    int32_t   bit_width;              // Bits per delta
};

/**
 * Vertical-Optimized Compressed Data Container (v3.0 - Interleaved Only)
 *
 * Single-format storage: all deltas in interleaved array (mini-vectors + tail).
 * Supports both batch scan (interleaved path) and random access (via coordinate mapping).
 */
template<typename T>
struct CompressedDataVertical {
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

    // ========== Interleaved Delta Storage (ONLY FORMAT) ==========

    // Compressed deltas (mini-vectors + tail for all partitions)
    uint32_t* d_interleaved_deltas;    // Bit-packed interleaved format
    int64_t   interleaved_delta_words; // Total words in interleaved array

    // Per-partition interleaved metadata
    int32_t* d_num_mini_vectors;       // [num_partitions] mini-vectors per partition
    int32_t* d_tail_sizes;             // [num_partitions] tail values per partition
    int64_t* d_interleaved_offsets;    // [num_partitions] word offset into interleaved array

    // Statistics
    int64_t  total_interleaved_partitions;  // Always equals num_partitions now

    // Kernel timing (ms) - only measures actual GPU kernel execution
    float kernel_time_ms;  // Total kernel execution time

    // Self-reference for device-side use
    CompressedDataVertical<T>* d_self;

    // ========== LEGACY FIELDS (for compatibility during transition) ==========
    // These will be removed in future versions
    uint32_t* d_sequential_deltas;     // DEPRECATED: set to nullptr
    int64_t   sequential_delta_words;  // DEPRECATED: set to 0
    int64_t*  d_delta_array_bit_offsets; // DEPRECATED: set to nullptr
    bool*     d_has_interleaved;       // DEPRECATED: set to nullptr

    // Constructor
    CompressedDataVertical()
        : num_partitions(0), total_values(0),
          d_start_indices(nullptr), d_end_indices(nullptr),
          d_model_types(nullptr), d_model_params(nullptr),
          d_delta_bits(nullptr), d_error_bounds(nullptr),
          d_partition_min_values(nullptr), d_partition_max_values(nullptr),
          d_interleaved_deltas(nullptr), interleaved_delta_words(0),
          d_num_mini_vectors(nullptr), d_tail_sizes(nullptr),
          d_interleaved_offsets(nullptr),
          total_interleaved_partitions(0),
          kernel_time_ms(0.0f),
          d_self(nullptr),
          // Legacy fields
          d_sequential_deltas(nullptr), sequential_delta_words(0),
          d_delta_array_bit_offsets(nullptr), d_has_interleaved(nullptr) {}

    /**
     * Convert from base L3 format
     *
     * NOTE: This creates a PARTIAL conversion - the interleaved data is NOT populated.
     * Only use this for BRANCHLESS decoder mode which uses d_delta_array_bit_offsets.
     *
     * For full interleaved support, use Vertical encoder directly.
     */
    static CompressedDataVertical<T> fromBase(const CompressedDataL3<T>& base) {
        CompressedDataVertical<T> result;
        result.num_partitions = base.num_partitions;
        result.total_values = base.total_values;
        result.d_start_indices = base.d_start_indices;
        result.d_end_indices = base.d_end_indices;
        result.d_model_types = base.d_model_types;
        result.d_model_params = base.d_model_params;
        result.d_delta_bits = base.d_delta_bits;
        result.d_error_bounds = base.d_error_bounds;
        result.d_partition_min_values = base.d_partition_min_values;
        result.d_partition_max_values = base.d_partition_max_values;

        // Legacy fields for BRANCHLESS mode compatibility
        result.d_sequential_deltas = base.delta_array;
        result.sequential_delta_words = base.delta_array_words;
        result.d_delta_array_bit_offsets = base.d_delta_array_bit_offsets;

        // Interleaved fields NOT populated - decoder must use BRANCHLESS mode
        result.d_interleaved_deltas = nullptr;
        result.interleaved_delta_words = 0;
        result.d_num_mini_vectors = nullptr;
        result.d_tail_sizes = nullptr;
        result.d_interleaved_offsets = nullptr;
        result.total_interleaved_partitions = 0;

        return result;
    }

    /**
     * Convert to CompressedDataOpt format for L3 decoder compatibility
     *
     * This enables Vertical-encoded data to be decompressed by any L3 decoder:
     * - STANDARD (decompression_kernels.cu)
     * - OPTIMIZED (decoder_warp_opt.cu)
     * - SPECIALIZED (decoder_specialized.cu)
     * - PHASE2 (decompression_kernels_phase2.cu)
     * - PHASE2_BUCKET (decompression_kernels_phase2_bucket.cu)
     * - KERNELS_OPT (decompression_kernels_opt.cu)
     *
     * NOTE: For v3.0 format, this requires computing sequential bit offsets
     *       from interleaved metadata. Currently only works with legacy fields.
     *
     * Note: This is a zero-copy conversion - pointers are shared, not copied.
     */
    /**
     * Check if sequential format is available for L3 decoder compatibility
     */
    bool hasSequentialFormat() const {
        return d_sequential_deltas != nullptr && d_delta_array_bit_offsets != nullptr;
    }

    CompressedDataOpt<T> toL3() const {
        // Validate that sequential format is available
        // v3.0 interleaved-only format is NOT compatible with L3 decoders
        if (!hasSequentialFormat()) {
            throw std::runtime_error(
                "toL3() requires sequential format (d_sequential_deltas and d_delta_array_bit_offsets). "
                "v3.0 interleaved-only format is incompatible with L3 decoders. "
                "Use Vertical decoder (--decoder Vertical) instead.");
        }

        CompressedDataOpt<T> result;
        result.d_start_indices = this->d_start_indices;
        result.d_end_indices = this->d_end_indices;
        result.d_model_types = this->d_model_types;
        result.d_model_params = this->d_model_params;
        result.d_delta_bits = this->d_delta_bits;
        result.d_delta_array_bit_offsets = this->d_delta_array_bit_offsets;
        result.delta_array = this->d_sequential_deltas;
        result.d_plain_deltas = nullptr;  // Not available in Vertical format
        result.num_partitions = this->num_partitions;
        result.total_elements = this->total_values;
        return result;
    }
};

// ============================================================================
// Encoding Configuration
// ============================================================================

/**
 * Vertical Compression Configuration (v3.0)
 *
 * NOTE: v3.0 uses interleaved-only format. Sequential format has been removed.
 */
struct VerticalConfig {
    // Base L3 config
    int partition_size_hint;
    int max_delta_bits;
    bool use_variable_partition;
    double error_bound_factor;

    // Vertical-specific options (v3.0)
    bool enable_interleaved;            // Always true in v3.0
    bool enable_branchless_unpack;      // Use branchless bit extraction
    int register_buffer_size;           // Words to prefetch into registers
    bool enable_adaptive_selection;     // Use adaptive model selection (all models)
    int fixed_model_type;               // Fixed model type when enable_adaptive_selection=false
                                        // Values: MODEL_LINEAR(1), MODEL_POLYNOMIAL2(2),
                                        //         MODEL_POLYNOMIAL3(3), MODEL_FOR_BITPACK(4)
    bool skip_metadata_recompute;       // Skip model parameter recomputation, use values from PartitionInfo directly

    // Partitioning strategy (NEW in v3.1)
    PartitioningStrategy partitioning_strategy;  // FIXED, VARIANCE_ADAPTIVE, COST_OPTIMAL

    // Cost-optimal partitioning parameters (used when partitioning_strategy == COST_OPTIMAL)
    int cost_analysis_block_size;       // Size of blocks for delta-bits analysis
    int cost_min_partition_size;        // Minimum partition size (warp-aligned)
    int cost_max_partition_size;        // Maximum partition size
    int cost_target_partition_size;     // Preferred partition size
    int cost_breakpoint_threshold;      // Delta-bits change to trigger breakpoint
    float cost_merge_benefit_threshold; // Minimum benefit (5%) to merge
    int cost_max_merge_rounds;          // Maximum merge iterations
    bool cost_enable_merging;           // Enable cost-based merging

    // Default config
    VerticalConfig()
        : partition_size_hint(4096),
          max_delta_bits(64),
          use_variable_partition(true),
          error_bound_factor(1.0),
          enable_interleaved(true),
          enable_branchless_unpack(true),
          register_buffer_size(REGISTER_BUFFER_WORDS),
          enable_adaptive_selection(true),  // Enable adaptive selection by default
          fixed_model_type(MODEL_LINEAR),   // Default to LINEAR when not adaptive
          skip_metadata_recompute(false),   // Default: recompute metadata (original behavior)
          // Partitioning defaults (v3.1)
          partitioning_strategy(PartitioningStrategy::FIXED),  // Keep FIXED as default for compatibility
          cost_analysis_block_size(2048),
          cost_min_partition_size(256),
          cost_max_partition_size(8192),
          cost_target_partition_size(2048),
          cost_breakpoint_threshold(2),
          cost_merge_benefit_threshold(0.05f),
          cost_max_merge_rounds(4),
          cost_enable_merging(true) {}

    // Default mode (fixed partitions, interleaved-only)
    static VerticalConfig defaultConfig() {
        return VerticalConfig();
    }

    // Cost-optimal mode (better compression ratio)
    static VerticalConfig costOptimal() {
        VerticalConfig config;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        return config;
    }

    // Scan-optimized mode (same as default in v3.0)
    static VerticalConfig scanOptimized() {
        return VerticalConfig();
    }
};

// ============================================================================
// Decompression Mode Selection
// ============================================================================

/**
 * Decompression path selection
 */
enum class DecompressMode : int32_t {
    AUTO = 0,           // Auto-select based on access pattern
    SEQUENTIAL = 1,     // Force sequential unpacking (random access)
    INTERLEAVED = 2,    // Force interleaved unpacking (batch scan)
    BRANCHLESS = 3      // Branchless sequential (hybrid)
};

/**
 * Query hint for path selection
 */
struct DecompressHint {
    DecompressMode mode;
    int64_t start_idx;          // Start of access range
    int64_t end_idx;            // End of access range (exclusive)
    bool is_point_query;        // True if accessing single value
    bool is_predicate_scan;     // True if scanning with predicate

    DecompressHint()
        : mode(DecompressMode::AUTO),
          start_idx(0), end_idx(-1),
          is_point_query(false), is_predicate_scan(false) {}

    static DecompressHint pointQuery(int64_t idx) {
        DecompressHint hint;
        hint.mode = DecompressMode::SEQUENTIAL;
        hint.start_idx = idx;
        hint.end_idx = idx + 1;
        hint.is_point_query = true;
        return hint;
    }

    static DecompressHint rangeQuery(int64_t start, int64_t end) {
        DecompressHint hint;
        hint.mode = DecompressMode::AUTO;
        hint.start_idx = start;
        hint.end_idx = end;
        return hint;
    }

    static DecompressHint fullScan() {
        DecompressHint hint;
        hint.mode = DecompressMode::INTERLEAVED;
        return hint;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate words needed for mini-vector interleaved storage
 */
__host__ __device__ __forceinline__
int64_t calcInterleavedWords(int num_values, int bit_width) {
    int num_mini_vectors = num_values / MINI_VECTOR_SIZE;
    int tail_size = num_values % MINI_VECTOR_SIZE;

    // Each mini-vector: 32 lanes * (8 * bit_width) bits = 256 * bit_width bits
    int64_t mini_vector_bits = static_cast<int64_t>(num_mini_vectors) * MINI_VECTOR_SIZE * bit_width;

    // Tail uses sequential packing
    int64_t tail_bits = static_cast<int64_t>(tail_size) * bit_width;

    return (mini_vector_bits + tail_bits + 31) / 32;
}

/**
 * Calculate bit offset for interleaved access
 *
 * For mini-vector V, lane L, value index I within lane (0-7):
 *   - Mini-vector base: V * 256 * bit_width bits
 *   - Lane offset: L * 8 * bit_width bits
 *   - Value offset: I * bit_width bits
 */
__device__ __forceinline__
int64_t calcInterleavedBitOffset(int mini_vector_idx, int lane_id, int value_idx, int bit_width) {
    int64_t mv_base = static_cast<int64_t>(mini_vector_idx) * MINI_VECTOR_SIZE * bit_width;
    int64_t lane_offset = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * bit_width;
    int64_t value_offset = static_cast<int64_t>(value_idx) * bit_width;
    return mv_base + lane_offset + value_offset;
}

/**
 * Map global index to mini-vector coordinates
 */
__device__ __forceinline__
void globalToInterleaved(int global_idx, int& mini_vector_idx, int& lane_id, int& value_idx) {
    mini_vector_idx = global_idx / MINI_VECTOR_SIZE;
    int local_idx = global_idx % MINI_VECTOR_SIZE;

    // Interleaved mapping: value at local_idx maps to lane (local_idx % 32)
    // and value index (local_idx / 32)
    lane_id = local_idx % LANES_PER_MINI_VECTOR;
    value_idx = local_idx / LANES_PER_MINI_VECTOR;
}

/**
 * Map mini-vector coordinates to global index
 */
__device__ __forceinline__
int interleavedToGlobal(int mini_vector_idx, int lane_id, int value_idx) {
    return mini_vector_idx * MINI_VECTOR_SIZE + value_idx * LANES_PER_MINI_VECTOR + lane_id;
}

#endif // L3_Vertical_FORMAT_HPP
