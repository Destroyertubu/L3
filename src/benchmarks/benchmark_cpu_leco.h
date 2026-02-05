/**
 * LeCo (Learned Compression) CPU Implementation
 * Based on SIGMOD'24 paper - Cost-Optimal Encoder
 *
 * Implements variable-length partitioning with greedy merge algorithm
 * for 32-bit and 64-bit integers.
 */

#ifndef BENCHMARK_CPU_LECO_H_
#define BENCHMARK_CPU_LECO_H_

#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <limits>
#include <chrono>

namespace leco {

// ============================================================================
// Configuration
// ============================================================================

struct LeCoConfig {
    int32_t overhead = 13;              // Segment metadata size in bytes
    int32_t block_size = 65536;         // Block size for encoding
    double cost_decline_threshold = 0.0001;  // Stop merge when <0.01% improvement
};

// ============================================================================
// Segment Structures
// ============================================================================

/**
 * Compressed segment metadata
 */
template<typename T>
struct LeCoSegment {
    int32_t start_idx;      // Start index (inclusive)
    int32_t end_idx;        // End index (exclusive)
    long double theta0;     // Intercept (centered) - long double for uint64 precision
    long double theta1;     // Slope - long double for uint64 precision
    int32_t delta_bits;     // Bits per delta (includes sign bit)
    int64_t bit_offset;     // Bit offset into packed delta array
};

/**
 * Compressed data block
 */
template<typename T>
struct LeCoCompressedBlock {
    int32_t total_values;
    int32_t num_segments;
    std::vector<LeCoSegment<T>> segments;
    std::vector<uint8_t> packed_data;   // Bit-packed residuals
    int64_t total_bits;

    // Statistics
    double compression_ratio;
    int64_t original_bytes;
    int64_t compressed_bytes;
};

/**
 * Segment node for merge algorithm (doubly-linked list)
 */
struct MergeSegment {
    int start_index;          // Segment start (inclusive)
    int end_index;            // Segment end (inclusive)
    int64_t max_delta;        // Maximum first-order delta in segment
    int64_t min_delta;        // Minimum first-order delta in segment
    int64_t next_delta;       // Delta at boundary with next segment
    int double_delta_next;    // Bit requirement for second-order delta
    MergeSegment* prev;       // Doubly-linked list pointers
    MergeSegment* next;

    MergeSegment(int start, int end, int64_t max_d, int64_t min_d,
                 int64_t next_d, int bit_next)
        : start_index(start), end_index(end), max_delta(max_d),
          min_delta(min_d), next_delta(next_d), double_delta_next(bit_next),
          prev(nullptr), next(nullptr) {}
};

// ============================================================================
// Compression Statistics
// ============================================================================

struct LeCoStats {
    double compression_ratio;
    double encode_time_ms;
    double decode_time_ms;
    int64_t original_bytes;
    int64_t compressed_bytes;
    int32_t num_segments;
    double avg_segment_length;
    double avg_delta_bits;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Calculate bits needed to represent an unsigned integer value
 */
template<typename T>
inline uint32_t bitsRequired(T v) {
    if (v == 0) return 0;
#if defined(__clang__) || defined(__GNUC__)
    if constexpr (sizeof(T) <= 4) {
        return 32 - __builtin_clz(static_cast<uint32_t>(v));
    } else {
        return 64 - __builtin_clzll(static_cast<uint64_t>(v));
    }
#else
    uint32_t bits = 0;
    while (v) {
        bits++;
        v >>= 1;
    }
    return bits;
#endif
}

/**
 * Calculate bits needed for a signed value range
 * Used for second-order delta calculation
 */
template<typename T>
inline uint32_t calcBitsForRange(int64_t min_val, int64_t max_val) {
    int64_t range = static_cast<int64_t>(std::ceil(std::abs(max_val - min_val) / 2.0));
    if (range == 0) return 0;
    return bitsRequired(static_cast<uint64_t>(range)) + 1;  // +1 for sign bit
}

// ============================================================================
// Core API Functions
// ============================================================================

/**
 * Encode data using LeCo cost-optimal encoder
 *
 * @param data Input data array
 * @param length Number of elements
 * @param config Compression configuration
 * @return Compressed block
 */
template<typename T>
LeCoCompressedBlock<T> lecoEncode(const T* data, size_t length,
                                   const LeCoConfig& config = LeCoConfig());

/**
 * Decode entire compressed block
 *
 * @param compressed Compressed block
 * @return Decompressed data
 */
template<typename T>
std::vector<T> lecoDecode(const LeCoCompressedBlock<T>& compressed);

/**
 * Random access decode - get single value at index
 *
 * @param compressed Compressed block
 * @param index Index to decode
 * @return Decoded value
 */
template<typename T>
T lecoDecodeAt(const LeCoCompressedBlock<T>& compressed, int32_t index);

/**
 * Convenience function for encoding vector
 */
template<typename T>
LeCoCompressedBlock<T> lecoEncode(const std::vector<T>& data,
                                   const LeCoConfig& config = LeCoConfig());

// ============================================================================
// Internal Functions (exposed for testing)
// ============================================================================

namespace internal {

/**
 * Fit linear regression model with theta0 centering
 *
 * @param data Input data pointer
 * @param start Start index
 * @param length Number of elements
 * @param theta0 Output: centered intercept
 * @param theta1 Output: slope
 * @param delta_bits Output: bits required for deltas
 * @param max_delta Output: maximum absolute delta
 */
template<typename T>
void fitLinearModel(const T* data, int32_t start, int32_t length,
                    long double& theta0, long double& theta1, int32_t& delta_bits,
                    T& max_delta);

/**
 * Calculate compressed size for a segment
 *
 * @param data Input data pointer
 * @param start Start index (inclusive)
 * @param end End index (inclusive)
 * @return Size in bytes
 */
template<typename T>
uint64_t calculateSegmentSize(const T* data, int32_t start, int32_t end);

/**
 * Pack a single signed delta value
 *
 * @param packed_data Output buffer
 * @param bit_offset Bit offset in buffer
 * @param delta Signed delta value
 * @param delta_bits Bits per delta
 */
void packDelta(std::vector<uint8_t>& packed_data, int64_t& bit_offset,
               int64_t delta, int32_t delta_bits);

/**
 * Unpack a single signed delta value
 *
 * @param packed_data Input buffer
 * @param bit_offset Bit offset in buffer
 * @param delta_bits Bits per delta
 * @return Signed delta value
 */
int64_t unpackDelta(const std::vector<uint8_t>& packed_data, int64_t bit_offset,
                    int32_t delta_bits);

/**
 * Cost-optimal partitioning using greedy merge algorithm
 *
 * @param data Input data pointer
 * @param length Number of elements
 * @param config Compression configuration
 * @return Vector of segment start indices
 */
template<typename T>
std::vector<int32_t> costOptimalPartition(const T* data, size_t length,
                                           const LeCoConfig& config);

}  // namespace internal

// ============================================================================
// Explicit Template Instantiation Declarations
// ============================================================================

extern template LeCoCompressedBlock<uint32_t> lecoEncode(const uint32_t*, size_t, const LeCoConfig&);
extern template LeCoCompressedBlock<uint64_t> lecoEncode(const uint64_t*, size_t, const LeCoConfig&);
extern template std::vector<uint32_t> lecoDecode(const LeCoCompressedBlock<uint32_t>&);
extern template std::vector<uint64_t> lecoDecode(const LeCoCompressedBlock<uint64_t>&);
extern template uint32_t lecoDecodeAt(const LeCoCompressedBlock<uint32_t>&, int32_t);
extern template uint64_t lecoDecodeAt(const LeCoCompressedBlock<uint64_t>&, int32_t);

}  // namespace leco

#endif  // BENCHMARK_CPU_LECO_H_
