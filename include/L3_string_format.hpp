/**
 * L3 String Compression Format Specification
 *
 * Based on LeCo (Learned Compression) string support:
 * - Order-preserving string-to-integer mapping
 * - Common prefix extraction
 * - Character set reduction (M-base encoding)
 * - 2's power base optimization for fast GPU decoding
 * - Adaptive padding strategy
 *
 * FORMAT_VERSION: 1.0
 */

#ifndef L3_STRING_FORMAT_HPP
#define L3_STRING_FORMAT_HPP

#include <cstdint>
#include <string>
#include <vector>

// Handle CUDA/non-CUDA compilation
#ifdef __CUDACC__
#define L3_HOST_DEVICE __host__ __device__
#else
#define L3_HOST_DEVICE
#endif

// Maximum supported string length after prefix removal
// With 2^5 (32) base per character, max 12 chars = 60 bits (fits in uint64_t)
// With 2^5 base, max 25 chars = 125 bits (fits in uint128_t)
// With 2^5 base, max 51 chars = 255 bits (fits in uint256_t)
constexpr int MAX_STRING_LENGTH_64 = 12;   // For uint64_t encoding
constexpr int MAX_STRING_LENGTH_128 = 25;  // For uint128_t encoding
constexpr int MAX_STRING_LENGTH_256 = 51;  // For uint256_t encoding

// Encoding precision types
enum StringEncodingType : int32_t {
    STRING_ENCODING_64 = 0,    // Use uint64_t (up to 12 chars @ 5-bit)
    STRING_ENCODING_128 = 1,   // Use uint128_gpu (up to 25 chars @ 5-bit)
    STRING_ENCODING_256 = 2    // Use uint256_gpu (up to 51 chars @ 5-bit)
};

// String encoding modes
enum StringEncodingMode : int32_t {
    STRING_MODE_ASCII256 = 0,      // Full ASCII (256-base), 8 bits per char
    STRING_MODE_SUBSET = 1,        // Character subset (M-base)
    STRING_MODE_SUBSET_SHIFT = 2,  // 2's power base for fast decoding
    STRING_MODE_DIRECT = 3         // Direct storage (no encoding)
};

/**
 * 128-bit unsigned integer for GPU
 * Needed for strings longer than 12 characters with subset encoding
 */
struct alignas(16) uint128_gpu {
    uint64_t low;
    uint64_t high;

    L3_HOST_DEVICE uint128_gpu() : low(0), high(0) {}
    L3_HOST_DEVICE uint128_gpu(uint64_t l) : low(l), high(0) {}
    L3_HOST_DEVICE uint128_gpu(uint64_t h, uint64_t l) : low(l), high(h) {}

    // Addition
    L3_HOST_DEVICE uint128_gpu operator+(const uint128_gpu& other) const {
        uint128_gpu result;
        result.low = low + other.low;
        result.high = high + other.high + (result.low < low ? 1 : 0);
        return result;
    }

    // Subtraction
    L3_HOST_DEVICE uint128_gpu operator-(const uint128_gpu& other) const {
        uint128_gpu result;
        result.low = low - other.low;
        result.high = high - other.high - (low < other.low ? 1 : 0);
        return result;
    }

    // Left shift
    L3_HOST_DEVICE uint128_gpu operator<<(int shift) const {
        uint128_gpu result;
        if (shift >= 128) {
            result.low = 0;
            result.high = 0;
        } else if (shift >= 64) {
            result.low = 0;
            result.high = low << (shift - 64);
        } else if (shift > 0) {
            result.high = (high << shift) | (low >> (64 - shift));
            result.low = low << shift;
        } else {
            result = *this;
        }
        return result;
    }

    // Right shift
    L3_HOST_DEVICE uint128_gpu operator>>(int shift) const {
        uint128_gpu result;
        if (shift >= 128) {
            result.low = 0;
            result.high = 0;
        } else if (shift >= 64) {
            result.high = 0;
            result.low = high >> (shift - 64);
        } else if (shift > 0) {
            result.low = (low >> shift) | (high << (64 - shift));
            result.high = high >> shift;
        } else {
            result = *this;
        }
        return result;
    }

    // Bitwise AND
    L3_HOST_DEVICE uint128_gpu operator&(const uint128_gpu& other) const {
        return uint128_gpu(high & other.high, low & other.low);
    }

    // Bitwise OR
    L3_HOST_DEVICE uint128_gpu operator|(const uint128_gpu& other) const {
        return uint128_gpu(high | other.high, low | other.low);
    }

    // Comparison
    L3_HOST_DEVICE bool operator<(const uint128_gpu& other) const {
        return (high < other.high) || (high == other.high && low < other.low);
    }

    L3_HOST_DEVICE bool operator>(const uint128_gpu& other) const {
        return (high > other.high) || (high == other.high && low > other.low);
    }

    L3_HOST_DEVICE bool operator==(const uint128_gpu& other) const {
        return high == other.high && low == other.low;
    }

    // Multiplication by small integer
    L3_HOST_DEVICE uint128_gpu operator*(uint32_t mult) const {
        uint64_t low_low = (low & 0xFFFFFFFFULL) * mult;
        uint64_t low_high = (low >> 32) * mult;
        uint64_t high_low = (high & 0xFFFFFFFFULL) * mult;
        uint64_t high_high = (high >> 32) * mult;

        uint128_gpu result;
        result.low = low_low + (low_high << 32);
        uint64_t carry = (result.low < low_low) ? 1 : 0;
        carry += (low_high >> 32);
        result.high = high_low + (high_high << 32) + carry;
        return result;
    }

    // Get lower 64 bits as uint64_t
    L3_HOST_DEVICE uint64_t to_uint64() const {
        return low;
    }

    // Check if fits in 64 bits
    L3_HOST_DEVICE bool fits_uint64() const {
        return high == 0;
    }
};

/**
 * 256-bit unsigned integer for GPU
 * Needed for strings longer than 25 characters with subset encoding
 * Layout: words[0] = lowest 64 bits, words[3] = highest 64 bits
 */
struct alignas(32) uint256_gpu {
    uint64_t words[4];  // words[0]=lowest, words[3]=highest

    L3_HOST_DEVICE uint256_gpu() {
        words[0] = words[1] = words[2] = words[3] = 0;
    }

    L3_HOST_DEVICE uint256_gpu(uint64_t val) {
        words[0] = val;
        words[1] = words[2] = words[3] = 0;
    }

    L3_HOST_DEVICE uint256_gpu(uint64_t w3, uint64_t w2, uint64_t w1, uint64_t w0) {
        words[0] = w0;
        words[1] = w1;
        words[2] = w2;
        words[3] = w3;
    }

    // Addition with carry propagation
    L3_HOST_DEVICE uint256_gpu operator+(const uint256_gpu& other) const {
        uint256_gpu result;
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t sum = words[i] + other.words[i] + carry;
            carry = (sum < words[i]) || (carry && sum == words[i]) ? 1 : 0;
            result.words[i] = sum;
        }
        return result;
    }

    // Subtraction with borrow propagation
    L3_HOST_DEVICE uint256_gpu operator-(const uint256_gpu& other) const {
        uint256_gpu result;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t diff = words[i] - other.words[i] - borrow;
            borrow = (words[i] < other.words[i]) || (borrow && words[i] == other.words[i]) ? 1 : 0;
            result.words[i] = diff;
        }
        return result;
    }

    // Left shift (supports shifts 0-255)
    L3_HOST_DEVICE uint256_gpu operator<<(int shift) const {
        uint256_gpu result;
        if (shift >= 256) {
            // All bits shifted out
            return result;  // Already zeroed
        } else if (shift >= 192) {
            result.words[3] = words[0] << (shift - 192);
        } else if (shift >= 128) {
            int s = shift - 128;
            result.words[3] = (words[1] << s) | (s > 0 ? (words[0] >> (64 - s)) : 0);
            result.words[2] = words[0] << s;
        } else if (shift >= 64) {
            int s = shift - 64;
            result.words[3] = (words[2] << s) | (s > 0 ? (words[1] >> (64 - s)) : 0);
            result.words[2] = (words[1] << s) | (s > 0 ? (words[0] >> (64 - s)) : 0);
            result.words[1] = words[0] << s;
        } else if (shift > 0) {
            result.words[3] = (words[3] << shift) | (words[2] >> (64 - shift));
            result.words[2] = (words[2] << shift) | (words[1] >> (64 - shift));
            result.words[1] = (words[1] << shift) | (words[0] >> (64 - shift));
            result.words[0] = words[0] << shift;
        } else {
            result = *this;
        }
        return result;
    }

    // Right shift (supports shifts 0-255)
    L3_HOST_DEVICE uint256_gpu operator>>(int shift) const {
        uint256_gpu result;
        if (shift >= 256) {
            // All bits shifted out
            return result;  // Already zeroed
        } else if (shift >= 192) {
            result.words[0] = words[3] >> (shift - 192);
        } else if (shift >= 128) {
            int s = shift - 128;
            result.words[0] = (words[2] >> s) | (s > 0 ? (words[3] << (64 - s)) : 0);
            result.words[1] = words[3] >> s;
        } else if (shift >= 64) {
            int s = shift - 64;
            result.words[0] = (words[1] >> s) | (s > 0 ? (words[2] << (64 - s)) : 0);
            result.words[1] = (words[2] >> s) | (s > 0 ? (words[3] << (64 - s)) : 0);
            result.words[2] = words[3] >> s;
        } else if (shift > 0) {
            result.words[0] = (words[0] >> shift) | (words[1] << (64 - shift));
            result.words[1] = (words[1] >> shift) | (words[2] << (64 - shift));
            result.words[2] = (words[2] >> shift) | (words[3] << (64 - shift));
            result.words[3] = words[3] >> shift;
        } else {
            result = *this;
        }
        return result;
    }

    // Bitwise AND
    L3_HOST_DEVICE uint256_gpu operator&(const uint256_gpu& other) const {
        uint256_gpu result;
        for (int i = 0; i < 4; i++) {
            result.words[i] = words[i] & other.words[i];
        }
        return result;
    }

    // Bitwise OR
    L3_HOST_DEVICE uint256_gpu operator|(const uint256_gpu& other) const {
        uint256_gpu result;
        for (int i = 0; i < 4; i++) {
            result.words[i] = words[i] | other.words[i];
        }
        return result;
    }

    // Comparison: less than
    L3_HOST_DEVICE bool operator<(const uint256_gpu& other) const {
        for (int i = 3; i >= 0; i--) {
            if (words[i] < other.words[i]) return true;
            if (words[i] > other.words[i]) return false;
        }
        return false;  // Equal
    }

    // Comparison: greater than
    L3_HOST_DEVICE bool operator>(const uint256_gpu& other) const {
        for (int i = 3; i >= 0; i--) {
            if (words[i] > other.words[i]) return true;
            if (words[i] < other.words[i]) return false;
        }
        return false;  // Equal
    }

    // Comparison: equality
    L3_HOST_DEVICE bool operator==(const uint256_gpu& other) const {
        return words[0] == other.words[0] && words[1] == other.words[1] &&
               words[2] == other.words[2] && words[3] == other.words[3];
    }

    // Multiplication by small integer (32-bit)
    L3_HOST_DEVICE uint256_gpu operator*(uint32_t mult) const {
        uint256_gpu result;
        uint64_t carry = 0;
        for (int i = 0; i < 4; i++) {
            // Split 64-bit word into two 32-bit halves
            uint64_t lo = (words[i] & 0xFFFFFFFFULL) * mult;
            uint64_t hi = (words[i] >> 32) * mult;
            // Combine with carry
            uint64_t sum = lo + carry;
            carry = (sum < lo) ? 1 : 0;
            sum += (hi << 32);
            if (sum < (hi << 32)) carry++;
            carry += (hi >> 32);
            result.words[i] = sum;
        }
        return result;
    }

    // Get lower 64 bits
    L3_HOST_DEVICE uint64_t to_uint64() const {
        return words[0];
    }

    // Check if fits in 64 bits
    L3_HOST_DEVICE bool fits_uint64() const {
        return words[1] == 0 && words[2] == 0 && words[3] == 0;
    }

    // Check if fits in 128 bits
    L3_HOST_DEVICE bool fits_uint128() const {
        return words[2] == 0 && words[3] == 0;
    }

    // Convert to uint128_gpu (truncating)
    L3_HOST_DEVICE uint128_gpu to_uint128() const {
        return uint128_gpu(words[1], words[0]);
    }

    // Create from uint128_gpu
    L3_HOST_DEVICE static uint256_gpu from_uint128(const uint128_gpu& val) {
        uint256_gpu result;
        result.words[0] = val.low;
        result.words[1] = val.high;
        return result;
    }
};

/**
 * String Compression Configuration
 */
struct StringCompressionConfig {
    StringEncodingMode mode;       // Encoding mode
    int32_t min_char;              // Minimum character in charset (e.g., 'a' = 97)
    int32_t max_char;              // Maximum character in charset (e.g., 'z' = 122)
    int32_t shift_bits;            // Bits per character for shift mode (ceil(log2(charset_size)))
    int32_t max_string_length;     // Maximum string length after prefix removal
    bool use_common_prefix;        // Extract common prefix
    std::string common_prefix;     // Extracted common prefix
    int32_t partition_size;        // Target partition size

    StringCompressionConfig()
        : mode(STRING_MODE_SUBSET_SHIFT),
          min_char(0), max_char(255),
          shift_bits(8), max_string_length(12),
          use_common_prefix(true),
          partition_size(4096) {}

    // Calculate charset size
    int32_t charset_size() const {
        return max_char - min_char + 1;
    }

    // Calculate bits needed per character (2's power ceiling)
    int32_t bits_per_char() const {
        int32_t size = charset_size();
        int32_t bits = 0;
        while ((1 << bits) < size) bits++;
        return bits;
    }
};

/**
 * String Partition Metadata
 * Extended from numeric partition metadata
 */
struct StringPartitionMetadata {
    int32_t* d_start_indices;           // [num_partitions] Start index (inclusive)
    int32_t* d_end_indices;             // [num_partitions] End index (exclusive)
    int32_t* d_model_types;             // [num_partitions] Model type

    // For 64-bit encoded values
    double* d_model_params_64;          // [num_partitions * 4] θ₀, θ₁, θ₂, θ₃
    int32_t* d_delta_bits_64;           // [num_partitions] Bits per delta
    int64_t* d_delta_array_bit_offsets_64; // [num_partitions] Bit offset

    // For 128-bit encoded values (longer strings)
    uint128_gpu* d_model_params_128;    // [num_partitions * 4] θ₀, θ₁, θ₂, θ₃
    int32_t* d_delta_bits_128;          // [num_partitions] Bits per delta
    int64_t* d_delta_array_bit_offsets_128; // [num_partitions] Bit offset

    // For 256-bit encoded values (very long strings)
    uint256_gpu* d_model_params_256;    // [num_partitions * 4] θ₀, θ₁, θ₂, θ₃
    int32_t* d_delta_bits_256;          // [num_partitions] Bits per delta
    int64_t* d_delta_array_bit_offsets_256; // [num_partitions] Bit offset

    // Original string lengths (needed for reconstruction)
    int8_t* d_original_lengths;         // [total_strings] Original string length
};

/**
 * Compressed String Data Container
 */
struct CompressedStringData {
    // Basic info
    int32_t num_partitions;
    int32_t total_strings;

    // Encoding configuration (device copy)
    StringEncodingMode mode;
    StringEncodingType encoding_type;  // 64/128/256-bit precision
    int32_t min_char;
    int32_t max_char;
    int32_t shift_bits;
    int32_t max_encoded_length;      // Max chars after prefix removal
    int32_t common_prefix_length;
    char* d_common_prefix;           // [common_prefix_length] Common prefix on device

    // Partition metadata (device pointers)
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double* d_model_params;          // [num_partitions * 4]
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;
    int64_t* d_error_bounds;

    // Original string lengths (for variable-length strings)
    int8_t* d_original_lengths;      // [total_strings]

    // Bit-packed delta array
    uint32_t* delta_array;
    int64_t delta_array_words;

    // Encoded integer values (optional, for debugging)
    uint64_t* d_encoded_values_64;     // [total_strings] if fits in 64 bits
    uint128_gpu* d_encoded_values_128; // [total_strings] if needs 128 bits
    uint256_gpu* d_encoded_values_256; // [total_strings] if needs 256 bits

    // Self-reference for device-side use
    CompressedStringData* d_self;

    // Statistics
    double compression_ratio;
    float compression_time_ms;
    float decompression_time_ms;

    CompressedStringData()
        : num_partitions(0), total_strings(0),
          mode(STRING_MODE_SUBSET_SHIFT),
          encoding_type(STRING_ENCODING_64),
          min_char(0), max_char(255), shift_bits(8),
          max_encoded_length(12), common_prefix_length(0),
          d_common_prefix(nullptr),
          d_start_indices(nullptr), d_end_indices(nullptr),
          d_model_types(nullptr), d_model_params(nullptr),
          d_delta_bits(nullptr), d_delta_array_bit_offsets(nullptr),
          d_error_bounds(nullptr), d_original_lengths(nullptr),
          delta_array(nullptr), delta_array_words(0),
          d_encoded_values_64(nullptr), d_encoded_values_128(nullptr),
          d_encoded_values_256(nullptr),
          d_self(nullptr),
          compression_ratio(0.0), compression_time_ms(0.0),
          decompression_time_ms(0.0) {}
};

/**
 * String Compression Statistics
 */
struct StringCompressionStats {
    int64_t original_bytes;          // Total bytes of original strings
    int64_t compressed_bytes;        // Total compressed size
    double compression_ratio;        // original / compressed
    float compression_time_ms;
    float compression_throughput_gbps;
    int32_t num_partitions;
    int32_t total_strings;
    int32_t common_prefix_length;
    int32_t avg_delta_bits;          // Weighted average delta bits
    int64_t total_bits_used;
};

/**
 * String Decompression Statistics
 */
struct StringDecompressionStats {
    int64_t decompressed_bytes;
    int64_t compressed_bytes;
    float decompression_time_ms;
    float decompression_throughput_gbps;
    int32_t total_strings;
};

#endif // L3_STRING_FORMAT_HPP
