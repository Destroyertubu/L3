/**
 * L3 String Compression Utilities
 *
 * Provides CPU and GPU utilities for:
 * - String-to-integer encoding (order-preserving)
 * - Integer-to-string decoding
 * - Common prefix extraction
 * - Character set analysis
 *
 * Based on LeCo string support with GPU optimizations.
 */

#ifndef L3_STRING_UTILS_HPP
#define L3_STRING_UTILS_HPP

#include "L3_string_format.hpp"
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

// Handle CUDA/non-CUDA compilation
#ifdef __CUDACC__
#include <cuda_runtime.h>
#define L3_DEVICE __device__
#define L3_FORCEINLINE __forceinline__
#define L3_MAX(a, b) max(a, b)
#define L3_MIN(a, b) min(a, b)
#else
#define L3_DEVICE
#define L3_FORCEINLINE inline
#define L3_MAX(a, b) std::max(a, b)
#define L3_MIN(a, b) std::min(a, b)
#endif

// ============================================================================
// CPU Utilities - Character Set Analysis
// ============================================================================

/**
 * Analyze character set in a collection of strings
 * Returns min and max character codes
 */
inline void analyzeCharacterSet(
    const std::vector<std::string>& strings,
    int32_t& min_char,
    int32_t& max_char)
{
    min_char = 255;
    max_char = 0;

    for (const auto& s : strings) {
        for (char c : s) {
            int32_t code = static_cast<uint8_t>(c);
            min_char = std::min(min_char, code);
            max_char = std::max(max_char, code);
        }
    }

    // Ensure valid range
    if (min_char > max_char) {
        min_char = 0;
        max_char = 255;
    }
}

/**
 * Find common prefix among all strings
 */
inline std::string findCommonPrefix(const std::vector<std::string>& strings)
{
    if (strings.empty()) return "";
    if (strings.size() == 1) return strings[0];

    std::string prefix = strings[0];

    for (size_t i = 1; i < strings.size() && !prefix.empty(); ++i) {
        const std::string& s = strings[i];
        size_t j = 0;
        while (j < prefix.size() && j < s.size() && prefix[j] == s[j]) {
            ++j;
        }
        prefix = prefix.substr(0, j);
    }

    return prefix;
}

/**
 * Calculate bits needed for character set (2's power ceiling)
 */
inline int32_t calculateShiftBits(int32_t charset_size)
{
    if (charset_size <= 1) return 1;
    int32_t bits = 0;
    int32_t val = charset_size - 1;
    while (val > 0) {
        bits++;
        val >>= 1;
    }
    return bits;
}

// ============================================================================
// CPU String-to-Integer Encoding
// ============================================================================

/**
 * Encode string to uint64_t using full ASCII (256-base)
 * Maximum 8 characters
 */
inline uint64_t stringToUint64_ASCII256(const std::string& str)
{
    uint64_t result = 0;
    size_t len = std::min(str.size(), size_t(8));

    for (size_t i = 0; i < len; ++i) {
        result = (result << 8) | static_cast<uint8_t>(str[i]);
    }

    return result;
}

/**
 * Encode string to uint64_t using subset encoding (M-base)
 * Supports longer strings with reduced character set
 */
inline uint64_t stringToUint64_Subset(
    const std::string& str,
    int32_t min_char,
    int32_t max_char)
{
    int32_t charset_size = max_char - min_char + 1;
    uint64_t result = 0;

    for (char c : str) {
        result *= charset_size;
        int32_t code = static_cast<uint8_t>(c) - min_char;
        code = std::max(0, std::min(code, charset_size - 1));
        result += code;
    }

    return result;
}

/**
 * Encode string to uint64_t using 2's power base (bit shifts)
 * Fastest for GPU decoding
 */
inline uint64_t stringToUint64_Shift(
    const std::string& str,
    int32_t min_char,
    int32_t shift_bits)
{
    uint64_t result = 0;
    uint32_t mask = (1U << shift_bits) - 1;

    for (char c : str) {
        result <<= shift_bits;
        int32_t code = static_cast<uint8_t>(c) - min_char;
        code = std::max(0, std::min(code, static_cast<int32_t>(mask)));
        result |= code;
    }

    return result;
}

/**
 * Encode string to uint128_gpu using 2's power base
 * For longer strings (13-25 characters with 5-bit encoding)
 */
inline uint128_gpu stringToUint128_Shift(
    const std::string& str,
    int32_t min_char,
    int32_t shift_bits)
{
    uint128_gpu result;
    uint32_t mask = (1U << shift_bits) - 1;

    for (char c : str) {
        result = result << shift_bits;
        int32_t code = static_cast<uint8_t>(c) - min_char;
        code = std::max(0, std::min(code, static_cast<int32_t>(mask)));
        result = result | uint128_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

// ============================================================================
// CPU Integer-to-String Decoding
// ============================================================================

/**
 * Decode uint64_t to string using full ASCII (256-base)
 */
inline std::string uint64ToString_ASCII256(uint64_t value, int32_t length)
{
    std::string result(length, '\0');
    for (int32_t i = length - 1; i >= 0; --i) {
        result[i] = static_cast<char>(value & 0xFF);
        value >>= 8;
    }
    return result;
}

/**
 * Decode uint64_t to string using 2's power base (bit shifts)
 */
inline std::string uint64ToString_Shift(
    uint64_t value,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    std::string result(length, '\0');
    uint32_t mask = (1U << shift_bits) - 1;

    for (int32_t i = length - 1; i >= 0; --i) {
        result[i] = static_cast<char>((value & mask) + min_char);
        value >>= shift_bits;
    }

    return result;
}

/**
 * Decode uint128_gpu to string using 2's power base
 */
inline std::string uint128ToString_Shift(
    uint128_gpu value,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    std::string result(length, '\0');
    uint128_gpu mask(static_cast<uint64_t>((1U << shift_bits) - 1));

    for (int32_t i = length - 1; i >= 0; --i) {
        uint64_t code = (value & mask).to_uint64();
        result[i] = static_cast<char>(code + min_char);
        value = value >> shift_bits;
    }

    return result;
}

/**
 * Encode string to uint256_gpu using 2's power base
 * For very long strings (26-51 characters with 5-bit encoding)
 */
inline uint256_gpu stringToUint256_Shift(
    const std::string& str,
    int32_t min_char,
    int32_t shift_bits)
{
    uint256_gpu result;
    uint32_t mask = (1U << shift_bits) - 1;

    for (char c : str) {
        result = result << shift_bits;
        int32_t code = static_cast<uint8_t>(c) - min_char;
        code = std::max(0, std::min(code, static_cast<int32_t>(mask)));
        result = result | uint256_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

/**
 * Decode uint256_gpu to string using 2's power base
 */
inline std::string uint256ToString_Shift(
    uint256_gpu value,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    std::string result(length, '\0');
    uint256_gpu mask(static_cast<uint64_t>((1U << shift_bits) - 1));

    for (int32_t i = length - 1; i >= 0; --i) {
        uint64_t code = (value & mask).to_uint64();
        result[i] = static_cast<char>(code + min_char);
        value = value >> shift_bits;
    }

    return result;
}

// ============================================================================
// Adaptive Padding Utilities
// ============================================================================

/**
 * Calculate min and max possible encoded values for a string
 * with different padding lengths.
 *
 * This implements LeCo's adaptive padding strategy where the padding
 * can vary from padding_min to padding_max to minimize delta values.
 */
inline void calculatePaddingBounds(
    const std::string& str,
    int32_t min_char,
    int32_t max_char,
    int32_t shift_bits,
    int32_t padding_min,
    int32_t padding_max,
    uint64_t& min_value,
    uint64_t& max_value)
{
    // Minimum: pad with min_char up to padding_min
    std::string min_str = str;
    while (static_cast<int32_t>(min_str.size()) < padding_min) {
        min_str += static_cast<char>(min_char);
    }
    min_value = stringToUint64_Shift(min_str, min_char, shift_bits);

    // Maximum: pad with max_char up to padding_max
    std::string max_str = str;
    while (static_cast<int32_t>(max_str.size()) < padding_max) {
        max_str += static_cast<char>(max_char);
    }
    max_value = stringToUint64_Shift(max_str, min_char, shift_bits);
}

// ============================================================================
// Batch Encoding (CPU)
// ============================================================================

/**
 * Encode a batch of strings to uint64_t values
 */
inline void encodeStringBatch(
    const std::vector<std::string>& strings,
    const StringCompressionConfig& config,
    std::vector<uint64_t>& encoded_values,
    std::vector<int8_t>& original_lengths)
{
    size_t n = strings.size();
    encoded_values.resize(n);
    original_lengths.resize(n);

    int prefix_len = config.common_prefix.size();

    for (size_t i = 0; i < n; ++i) {
        // Remove common prefix
        std::string suffix = strings[i].substr(prefix_len);
        original_lengths[i] = static_cast<int8_t>(suffix.size());

        // Pad or truncate to max length
        if (suffix.size() < static_cast<size_t>(config.max_string_length)) {
            suffix.append(config.max_string_length - suffix.size(),
                          static_cast<char>(config.min_char));
        } else if (suffix.size() > static_cast<size_t>(config.max_string_length)) {
            suffix = suffix.substr(0, config.max_string_length);
        }

        // Encode based on mode
        switch (config.mode) {
            case STRING_MODE_ASCII256:
                encoded_values[i] = stringToUint64_ASCII256(suffix);
                break;
            case STRING_MODE_SUBSET:
            case STRING_MODE_SUBSET_SHIFT:
                encoded_values[i] = stringToUint64_Shift(
                    suffix, config.min_char, config.shift_bits);
                break;
            default:
                encoded_values[i] = 0;
                break;
        }
    }
}

/**
 * Decode a batch of uint64_t values to strings
 */
inline void decodeStringBatch(
    const std::vector<uint64_t>& encoded_values,
    const std::vector<int8_t>& original_lengths,
    const StringCompressionConfig& config,
    std::vector<std::string>& strings)
{
    size_t n = encoded_values.size();
    strings.resize(n);

    for (size_t i = 0; i < n; ++i) {
        std::string decoded;

        switch (config.mode) {
            case STRING_MODE_ASCII256:
                decoded = uint64ToString_ASCII256(
                    encoded_values[i], config.max_string_length);
                break;
            case STRING_MODE_SUBSET:
            case STRING_MODE_SUBSET_SHIFT:
                decoded = uint64ToString_Shift(
                    encoded_values[i], config.max_string_length,
                    config.min_char, config.shift_bits);
                break;
            default:
                decoded = std::string(config.max_string_length, '\0');
                break;
        }

        // Truncate to original length and add prefix
        decoded = decoded.substr(0, original_lengths[i]);
        strings[i] = config.common_prefix + decoded;
    }
}

// ============================================================================
// GPU Device Functions
// ============================================================================

#ifdef __CUDACC__

/**
 * GPU device function: Encode string to uint64_t using shift mode
 * Input string is stored as char array with known length
 */
L3_DEVICE L3_FORCEINLINE uint64_t encodeString_GPU(
    const char* str,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint64_t result = 0;
    uint32_t mask = (1U << shift_bits) - 1;

    for (int32_t i = 0; i < length; ++i) {
        result <<= shift_bits;
        int32_t code = static_cast<uint8_t>(str[i]) - min_char;
        code = L3_MAX(0, L3_MIN(code, static_cast<int32_t>(mask)));
        result |= code;
    }

    return result;
}

/**
 * GPU device function: Decode uint64_t to string using shift mode
 */
L3_DEVICE L3_FORCEINLINE void decodeString_GPU(
    uint64_t value,
    char* output,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint32_t mask = (1U << shift_bits) - 1;

    for (int32_t i = length - 1; i >= 0; --i) {
        output[i] = static_cast<char>((value & mask) + min_char);
        value >>= shift_bits;
    }
}

/**
 * GPU device function: Encode string to uint128_gpu using shift mode
 */
L3_DEVICE L3_FORCEINLINE uint128_gpu encodeString128_GPU(
    const char* str,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint128_gpu result;
    uint32_t mask = (1U << shift_bits) - 1;

    for (int32_t i = 0; i < length; ++i) {
        result = result << shift_bits;
        int32_t code = static_cast<uint8_t>(str[i]) - min_char;
        code = L3_MAX(0, L3_MIN(code, static_cast<int32_t>(mask)));
        result = result | uint128_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

/**
 * GPU device function: Decode uint128_gpu to string using shift mode
 */
L3_DEVICE L3_FORCEINLINE void decodeString128_GPU(
    uint128_gpu value,
    char* output,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint128_gpu mask(static_cast<uint64_t>((1U << shift_bits) - 1));

    for (int32_t i = length - 1; i >= 0; --i) {
        uint64_t code = (value & mask).to_uint64();
        output[i] = static_cast<char>(code + min_char);
        value = value >> shift_bits;
    }
}

/**
 * GPU device function: Encode string to uint256_gpu using shift mode
 */
L3_DEVICE L3_FORCEINLINE uint256_gpu encodeString256_GPU(
    const char* str,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint256_gpu result;
    uint32_t mask = (1U << shift_bits) - 1;

    for (int32_t i = 0; i < length; ++i) {
        result = result << shift_bits;
        int32_t code = static_cast<uint8_t>(str[i]) - min_char;
        code = L3_MAX(0, L3_MIN(code, static_cast<int32_t>(mask)));
        result = result | uint256_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

/**
 * GPU device function: Decode uint256_gpu to string using shift mode
 */
L3_DEVICE L3_FORCEINLINE void decodeString256_GPU(
    uint256_gpu value,
    char* output,
    int32_t length,
    int32_t min_char,
    int32_t shift_bits)
{
    uint256_gpu mask(static_cast<uint64_t>((1U << shift_bits) - 1));

    for (int32_t i = length - 1; i >= 0; --i) {
        uint64_t code = (value & mask).to_uint64();
        output[i] = static_cast<char>(code + min_char);
        value = value >> shift_bits;
    }
}

#endif // __CUDACC__

// ============================================================================
// Configuration Helper
// ============================================================================

/**
 * Determine the optimal encoding type based on max string length and shift bits
 *
 * This implements LeCo's automatic type selection:
 * - 64-bit:  strings up to ~12 chars @ 5-bit, ~8 chars @ 8-bit
 * - 128-bit: strings up to ~25 chars @ 5-bit, ~16 chars @ 8-bit
 * - 256-bit: strings up to ~51 chars @ 5-bit, ~32 chars @ 8-bit
 *
 * @param max_string_length Maximum string length after prefix removal
 * @param shift_bits Bits per character
 * @return StringEncodingType (STRING_ENCODING_64, STRING_ENCODING_128, or STRING_ENCODING_256)
 */
inline StringEncodingType selectEncodingType(int32_t max_string_length, int32_t shift_bits)
{
    // Calculate maximum length for each encoding type
    int32_t max_len_64bit = 64 / shift_bits;    // ~12 chars @ 5-bit
    int32_t max_len_128bit = 128 / shift_bits;  // ~25 chars @ 5-bit
    int32_t max_len_256bit = 256 / shift_bits;  // ~51 chars @ 5-bit

    if (max_string_length <= max_len_64bit) {
        return STRING_ENCODING_64;
    } else if (max_string_length <= max_len_128bit) {
        return STRING_ENCODING_128;
    } else if (max_string_length <= max_len_256bit) {
        return STRING_ENCODING_256;
    } else {
        // Fallback to 256-bit (strings will be truncated)
        return STRING_ENCODING_256;
    }
}

/**
 * Get maximum supported string length for a given encoding type and shift bits
 */
inline int32_t getMaxStringLength(StringEncodingType encoding_type, int32_t shift_bits)
{
    switch (encoding_type) {
        case STRING_ENCODING_64:
            return 64 / shift_bits;
        case STRING_ENCODING_128:
            return 128 / shift_bits;
        case STRING_ENCODING_256:
            return 256 / shift_bits;
        default:
            return 64 / shift_bits;
    }
}

/**
 * Get encoding type name for debugging/logging
 */
inline const char* getEncodingTypeName(StringEncodingType encoding_type)
{
    switch (encoding_type) {
        case STRING_ENCODING_64:  return "64-bit";
        case STRING_ENCODING_128: return "128-bit";
        case STRING_ENCODING_256: return "256-bit";
        default: return "unknown";
    }
}

/**
 * Auto-configure string compression based on input strings
 *
 * Automatically selects 64/128/256-bit encoding based on max string length:
 * - 64-bit:  up to 12 chars with 5-bit encoding, 8 chars with 8-bit
 * - 128-bit: up to 25 chars with 5-bit encoding, 16 chars with 8-bit
 * - 256-bit: up to 51 chars with 5-bit encoding, 32 chars with 8-bit
 */
inline StringCompressionConfig autoConfigureStringCompression(
    const std::vector<std::string>& strings,
    bool extract_prefix = true,
    int32_t target_partition_size = 4096)
{
    StringCompressionConfig config;
    config.partition_size = target_partition_size;

    if (strings.empty()) {
        return config;
    }

    // Extract common prefix if requested
    if (extract_prefix) {
        config.common_prefix = findCommonPrefix(strings);
        config.use_common_prefix = !config.common_prefix.empty();
    }

    // Build strings without prefix for analysis
    std::vector<std::string> suffixes;
    suffixes.reserve(strings.size());
    size_t prefix_len = config.common_prefix.size();

    for (const auto& s : strings) {
        suffixes.push_back(s.substr(prefix_len));
    }

    // Analyze character set
    analyzeCharacterSet(suffixes, config.min_char, config.max_char);

    // Calculate shift bits
    config.shift_bits = calculateShiftBits(config.max_char - config.min_char + 1);

    // Find max string length after prefix removal
    size_t max_len = 0;
    for (const auto& s : suffixes) {
        max_len = std::max(max_len, s.size());
    }

    // Set max string length (actual max, not limited)
    config.max_string_length = static_cast<int32_t>(max_len);

    // Use subset shift mode for efficient GPU decoding
    config.mode = STRING_MODE_SUBSET_SHIFT;

    return config;
}

/**
 * Extended auto-configuration that also returns the selected encoding type
 * This is useful for determining which GPU kernels to use
 */
inline std::pair<StringCompressionConfig, StringEncodingType> autoConfigureWithEncodingType(
    const std::vector<std::string>& strings,
    bool extract_prefix = true,
    int32_t target_partition_size = 4096)
{
    StringCompressionConfig config = autoConfigureStringCompression(
        strings, extract_prefix, target_partition_size);

    // Select encoding type based on max string length
    StringEncodingType encoding_type = selectEncodingType(
        config.max_string_length, config.shift_bits);

    // Limit max_string_length to what the selected encoding type supports
    int32_t max_supported = getMaxStringLength(encoding_type, config.shift_bits);
    if (config.max_string_length > max_supported) {
        config.max_string_length = max_supported;
    }

    return std::make_pair(config, encoding_type);
}

#endif // L3_STRING_UTILS_HPP
