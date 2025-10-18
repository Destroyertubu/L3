#ifndef L3_INTERNAL_FORMAT_HPP
#define L3_INTERNAL_FORMAT_HPP

#include <cstdint>

/**
 * L3 (Greedy LEarned COmpression) Format Specification
 *
 * This header defines the invariant bitstream layout and data structures
 * that MUST remain consistent between encoder and decoder.
 *
 * FORMAT_VERSION: 1.0
 * Last Updated: 2025-10-18
 *
 * PRINCIPLE: Model-based compression with residual encoding
 * - Data is split into partitions
 * - Each partition fits a model (linear, polynomial, or direct copy)
 * - Residuals (deltas) are bit-packed with partition-specific width
 * - Decompression: prediction + delta
 */

namespace l3 {

// Format version for compatibility checking
constexpr uint32_t L3_FORMAT_VERSION = 0x00010000;  // v1.0.0
constexpr uint32_t L3_MAGIC = 0x474C4543;  // "GLEC" in ASCII

// Model types (MUST match between encoder and decoder)
enum ModelType : int32_t {
    MODEL_CONSTANT = 0,      // f(x) = θ₀
    MODEL_LINEAR = 1,        // f(x) = θ₀ + θ₁·x
    MODEL_POLYNOMIAL2 = 2,   // f(x) = θ₀ + θ₁·x + θ₂·x²
    MODEL_POLYNOMIAL3 = 3,   // f(x) = θ₀ + θ₁·x + θ₂·x² + θ₃·x³
    MODEL_DIRECT_COPY = 4    // No prediction, store raw values
};

/**
 * Partition Metadata (SoA - Struct of Arrays layout)
 *
 * INVARIANT: All arrays have num_partitions elements
 * INVARIANT: Partitions are non-overlapping and cover [0, total_values)
 * INVARIANT: start_indices[i] < end_indices[i]
 * INVARIANT: end_indices[i] <= start_indices[i+1]
 */
struct PartitionMetadata {
    int32_t* d_start_indices;           // [num_partitions] Start index (inclusive)
    int32_t* d_end_indices;             // [num_partitions] End index (exclusive)
    int32_t* d_model_types;             // [num_partitions] Model type per partition
    double*  d_model_params;            // [num_partitions * 4] θ₀, θ₁, θ₂, θ₃ per partition
    int32_t* d_delta_bits;              // [num_partitions] Bits per delta in this partition
    int64_t* d_delta_array_bit_offsets; // [num_partitions] Bit offset into delta_array
    int64_t* d_error_bounds;            // [num_partitions] Max |residual| (for validation)
};

/**
 * Compressed Data Container
 *
 * LAYOUT:
 * 1. Metadata (partition info) - SoA format
 * 2. Delta array - bit-packed residuals
 * 3. Optional: Pre-unpacked deltas (for high-throughput mode)
 *
 * BITSTREAM STRUCTURE:
 * For each partition p:
 *   - Model parameters: θ₀, θ₁, θ₂, θ₃ (doubles)
 *   - Delta bit width: w bits
 *   - Deltas: [delta₀, delta₁, ..., delta_n] packed at offset[p]
 *
 * Delta packing:
 *   - Deltas are signed integers, stored in w-bit two's complement
 *   - Bit offset calculation: bit_offset = offset[p] + i * w
 *   - Word index: word_idx = bit_offset / 32
 *   - In-word offset: bit_in_word = bit_offset % 32
 *   - Deltas may span word boundaries (require 2-word read)
 */
template<typename T>
struct CompressedDataL3 {
    // Metadata
    int32_t num_partitions;      // Number of partitions
    int32_t total_values;        // Total elements in original data

    // Partition metadata (device pointers)
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double*  d_model_params;     // [num_partitions * 4]
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;
    int64_t* d_error_bounds;

    // Compressed delta array (bit-packed)
    uint32_t* delta_array;       // Bit-packed deltas
    int64_t   delta_array_words; // Number of uint32 words allocated

    // Optional: Pre-unpacked deltas (high-throughput mode)
    int64_t* d_plain_deltas;     // [total_values] or nullptr

    // Predicate Pushdown Optimization: Value bounds per partition
    T* d_partition_min_values;   // [num_partitions] Minimum actual value in partition
    T* d_partition_max_values;   // [num_partitions] Maximum actual value in partition

    // Self-reference for device-side use
    CompressedDataL3<T>* d_self;

    // Constructor
    CompressedDataL3()
        : num_partitions(0), total_values(0),
          d_start_indices(nullptr), d_end_indices(nullptr),
          d_model_types(nullptr), d_model_params(nullptr),
          d_delta_bits(nullptr), d_delta_array_bit_offsets(nullptr),
          d_error_bounds(nullptr), delta_array(nullptr),
          delta_array_words(0), d_plain_deltas(nullptr),
          d_partition_min_values(nullptr), d_partition_max_values(nullptr),
          d_self(nullptr) {}
};

/**
 * Bit Packing/Unpacking Invariants
 *
 * ENCODER invariant:
 *   packed_value = (actual_value - predicted_value) & ((1 << bit_width) - 1)
 *
 * DECODER invariant:
 *   extracted = (delta_array[word] >> bit_offset) & ((1 << bit_width) - 1)
 *   signed_delta = sign_extend(extracted, bit_width)
 *   decompressed = predicted_value + signed_delta
 *
 * Sign extension:
 *   If bit_width < 32:
 *     sign_bit = extracted >> (bit_width - 1)
 *     if sign_bit == 1:
 *       extracted |= ~((1 << bit_width) - 1)  // Set upper bits
 *
 * CRITICAL: Encoder and decoder MUST use identical:
 *   - Model parameters (θ)
 *   - Bit width per partition
 *   - Bit offset calculation
 *   - Sign extension logic
 */

/**
 * Compression Parameters
 */
struct CompressionConfig {
    int partition_size_hint;     // Target partition size (adaptive)
    int max_delta_bits;          // Maximum bits per delta (default: 32)
    bool use_variable_partition; // Adaptive vs fixed-size partitions
    double error_bound_factor;   // Acceptable error bound multiplier

    // Default config
    CompressionConfig()
        : partition_size_hint(4096),
          max_delta_bits(32),
          use_variable_partition(true),
          error_bound_factor(1.0) {}
};

// Serialization header (for file storage)
struct L3SerializedHeader {
    uint32_t magic;           // L3_MAGIC
    uint32_t format_version;  // L3_FORMAT_VERSION
    uint32_t element_size;    // sizeof(T)
    uint32_t num_partitions;
    int32_t  total_values;
    int64_t  delta_array_words;
    uint64_t checksum;        // CRC32 or similar (optional)
};

} // namespace l3

#endif // L3_INTERNAL_FORMAT_HPP
