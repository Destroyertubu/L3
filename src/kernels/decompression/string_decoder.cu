/**
 * L3 String Decoder GPU Kernels
 *
 * GPU-optimized string decompression using:
 * - Warp-level cooperation for efficient bit unpacking
 * - Coalesced memory writes
 * - 2's power base decoding with bit shifts
 * - Shared memory for model parameters
 *
 * Based on LeCo string decompression with GPU optimizations.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "L3_string_format.hpp"
#include "L3_format.hpp"

#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

// ============================================================================
// Bit Unpacking Device Functions
// ============================================================================

/**
 * Extract delta from bit-packed array
 * Returns signed delta value
 */
__device__ __forceinline__ int64_t extractDelta(
    const uint32_t* __restrict__ delta_array,
    int64_t bit_offset,
    int32_t delta_bits)
{
    if (delta_bits <= 0) return 0;

    int word_idx = bit_offset / 32;
    int offset_in_word = bit_offset % 32;

    uint64_t raw_value = 0;

    if (delta_bits <= 32) {
        // Single or double word read
        uint64_t word0 = delta_array[word_idx];
        raw_value = word0 >> offset_in_word;

        if (offset_in_word + delta_bits > 32) {
            uint64_t word1 = delta_array[word_idx + 1];
            raw_value |= word1 << (32 - offset_in_word);
        }

        // Mask to delta_bits
        raw_value &= (1ULL << delta_bits) - 1;
    } else {
        // Multi-word read for > 32 bits
        int bits_read = 0;
        int current_word = word_idx;
        int current_offset = offset_in_word;

        while (bits_read < delta_bits) {
            int bits_in_word = min(delta_bits - bits_read, 32 - current_offset);
            uint64_t word = delta_array[current_word];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> current_offset) & mask;
            raw_value |= part << bits_read;

            bits_read += bits_in_word;
            current_word++;
            current_offset = 0;
        }
    }

    // Sign extension
    if (delta_bits < 64) {
        uint64_t sign_bit = 1ULL << (delta_bits - 1);
        if (raw_value & sign_bit) {
            raw_value |= ~((1ULL << delta_bits) - 1);
        }
    }

    return static_cast<int64_t>(raw_value);
}

// ============================================================================
// String Decoding Device Functions
// ============================================================================

/**
 * Decode uint64_t to string using 2's power base (bit shifts)
 * Output written to char array
 */
__device__ __forceinline__ void decodeUint64ToString(
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
 * Decode uint128_gpu to string using 2's power base
 */
__device__ __forceinline__ void decodeUint128ToString(
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
 * Decode uint256_gpu to string using 2's power base
 */
__device__ __forceinline__ void decodeUint256ToString(
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

// ============================================================================
// 128-bit Delta Extraction
// ============================================================================

/**
 * Extract 128-bit delta from bit-packed array
 * Returns signed delta as uint128_gpu (two's complement)
 */
__device__ __forceinline__ uint128_gpu extractDelta128(
    const uint32_t* __restrict__ delta_array,
    int64_t bit_offset,
    int32_t delta_bits)
{
    if (delta_bits <= 0) return uint128_gpu(0);

    uint128_gpu result;
    int word_idx = bit_offset / 32;
    int offset_in_word = bit_offset % 32;
    int bits_read = 0;

    // Read low 64 bits
    uint64_t low = 0;
    int bits_for_low = min(delta_bits, 64);
    while (bits_read < bits_for_low) {
        int bits_in_word = min(bits_for_low - bits_read, 32 - offset_in_word);
        uint64_t word = delta_array[word_idx];
        uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
        uint64_t part = (word >> offset_in_word) & mask;
        low |= part << bits_read;

        bits_read += bits_in_word;
        word_idx++;
        offset_in_word = 0;
    }
    result.low = low;

    // Read high 64 bits if needed
    if (delta_bits > 64) {
        uint64_t high = 0;
        int bits_for_high = delta_bits - 64;
        int high_bits_read = 0;
        while (high_bits_read < bits_for_high) {
            int bits_in_word = min(bits_for_high - high_bits_read, 32 - offset_in_word);
            uint64_t word = delta_array[word_idx];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> offset_in_word) & mask;
            high |= part << high_bits_read;

            high_bits_read += bits_in_word;
            word_idx++;
            offset_in_word = 0;
        }
        result.high = high;
    }

    return result;
}

// ============================================================================
// 256-bit Delta Extraction
// ============================================================================

/**
 * Extract 256-bit delta from bit-packed array
 * Returns signed delta as uint256_gpu (two's complement)
 */
__device__ __forceinline__ uint256_gpu extractDelta256(
    const uint32_t* __restrict__ delta_array,
    int64_t bit_offset,
    int32_t delta_bits)
{
    if (delta_bits <= 0) return uint256_gpu(0);

    uint256_gpu result;
    int word_idx = bit_offset / 32;
    int offset_in_word = bit_offset % 32;
    int bits_read = 0;

    // Read up to 4 words (256 bits)
    for (int w = 0; w < 4 && bits_read < delta_bits; w++) {
        uint64_t word_val = 0;
        int bits_for_word = min(delta_bits - bits_read, 64);
        int word_bits_read = 0;

        while (word_bits_read < bits_for_word) {
            int bits_in_word = min(bits_for_word - word_bits_read, 32 - offset_in_word);
            uint64_t word = delta_array[word_idx];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> offset_in_word) & mask;
            word_val |= part << word_bits_read;

            word_bits_read += bits_in_word;
            bits_read += bits_in_word;
            word_idx++;
            offset_in_word = 0;
        }
        result.words[w] = word_val;
    }

    return result;
}

// ============================================================================
// String Decompression Kernel (to encoded values)
// ============================================================================

/**
 * Decompress to uint64_t encoded values
 * This is the first step - produces encoded integers that can then be
 * converted back to strings.
 */
__global__ void decompressToEncodedValuesKernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    int num_partitions,
    uint64_t* __restrict__ d_encoded_values)
{
    // Each block handles one partition
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    // Load model parameters to shared memory
    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    // Process elements in parallel
    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;

        // Calculate prediction
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        int64_t pred_int = static_cast<int64_t>(round(predicted));

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        int64_t delta = extractDelta(delta_array, bit_offset, delta_bits);

        // Reconstruct value
        int64_t value = pred_int + delta;
        d_encoded_values[global_idx] = static_cast<uint64_t>(value);
    }
}

// ============================================================================
// String Reconstruction Kernel
// ============================================================================

/**
 * Convert encoded uint64_t values back to strings
 * Handles prefix prepending and length restoration
 *
 * Output format: flattened char array with fixed-size slots
 */
__global__ void reconstructStringsKernel(
    const uint64_t* __restrict__ d_encoded_values,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,  // Fixed size per output string
    char* __restrict__ d_output_strings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        char* output = d_output_strings + idx * output_string_stride;

        // Copy common prefix
        for (int i = 0; i < common_prefix_length; ++i) {
            output[i] = d_common_prefix[i];
        }

        // Decode suffix from encoded value
        uint64_t encoded = d_encoded_values[idx];
        int8_t orig_len = d_original_lengths[idx];

        // The encoded value has padding - we need to shift to remove padding
        // Padding is at the least significant bits
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 64) {
            encoded >>= pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint64ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate (if there's space)
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

// ============================================================================
// Warp-Optimized Decompression Kernel
// ============================================================================

/**
 * Optimized decompression using warp cooperation
 * Each warp processes one partition for better load balancing
 */
__global__ void decompressWarpOptKernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    int num_partitions,
    uint64_t* __restrict__ d_encoded_values)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (warp_id >= num_partitions) return;

    int start_idx = d_start_indices[warp_id];
    int end_idx = d_end_indices[warp_id];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    // Load parameters (all lanes read, but coalesced)
    double theta0 = d_model_params[warp_id * 4];
    double theta1 = d_model_params[warp_id * 4 + 1];
    int32_t delta_bits = d_delta_bits[warp_id];
    int64_t bit_offset_base = d_bit_offsets[warp_id];

    // Each lane processes multiple elements
    for (int i = lane_id; i < segment_len; i += WARP_SIZE) {
        int global_idx = start_idx + i;

        // Calculate prediction
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        int64_t pred_int = static_cast<int64_t>(round(predicted));

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        int64_t delta = extractDelta(delta_array, bit_offset, delta_bits);

        // Reconstruct value
        d_encoded_values[global_idx] = static_cast<uint64_t>(pred_int + delta);
    }
}

// ============================================================================
// Combined Decompression + Reconstruction Kernel
// ============================================================================

/**
 * Single-pass decompression directly to strings
 * Combines bit unpacking, prediction, and string decoding
 */
__global__ void decompressToStringsKernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* __restrict__ d_output_strings)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    // Shared memory for model parameters
    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    // Process elements
    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;
        char* output = d_output_strings + global_idx * output_string_stride;

        // Copy common prefix
        for (int j = 0; j < common_prefix_length; ++j) {
            output[j] = d_common_prefix[j];
        }

        // Calculate prediction
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        int64_t pred_int = static_cast<int64_t>(round(predicted));

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        int64_t delta = extractDelta(delta_array, bit_offset, delta_bits);

        // Reconstruct encoded value
        uint64_t encoded = static_cast<uint64_t>(pred_int + delta);

        // Get original length and decode
        int8_t orig_len = d_original_lengths[global_idx];

        // Remove padding from encoded value
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 64) {
            encoded >>= pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint64ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

// ============================================================================
// 128-bit String Decompression Kernels
// ============================================================================

/**
 * Decompress to uint128_gpu encoded values
 */
__global__ void decompressToEncodedValues128Kernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    int num_partitions,
    uint128_gpu* __restrict__ d_encoded_values)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;

        // Calculate prediction using double approximation
        double predicted = fma(theta1, static_cast<double>(i), theta0);

        // Convert to 128-bit
        uint128_gpu pred_128;
        if (predicted >= 0) {
            pred_128.high = static_cast<uint64_t>(predicted / 18446744073709551616.0);
            pred_128.low = static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0));
        }

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        uint128_gpu delta = extractDelta128(delta_array, bit_offset, delta_bits);

        // Reconstruct value
        d_encoded_values[global_idx] = pred_128 + delta;
    }
}

/**
 * Convert 128-bit encoded values back to strings
 */
__global__ void reconstructStrings128Kernel(
    const uint128_gpu* __restrict__ d_encoded_values,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* __restrict__ d_output_strings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        char* output = d_output_strings + idx * output_string_stride;

        // Copy common prefix
        for (int i = 0; i < common_prefix_length; ++i) {
            output[i] = d_common_prefix[i];
        }

        // Decode suffix from encoded value
        uint128_gpu encoded = d_encoded_values[idx];
        int8_t orig_len = d_original_lengths[idx];

        // Remove padding
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 128) {
            encoded = encoded >> pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint128ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

/**
 * Combined 128-bit decompression directly to strings
 */
__global__ void decompressToStrings128Kernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* __restrict__ d_output_strings)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;
        char* output = d_output_strings + global_idx * output_string_stride;

        // Copy common prefix
        for (int j = 0; j < common_prefix_length; ++j) {
            output[j] = d_common_prefix[j];
        }

        // Calculate prediction
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        uint128_gpu pred_128;
        if (predicted >= 0) {
            pred_128.high = static_cast<uint64_t>(predicted / 18446744073709551616.0);
            pred_128.low = static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0));
        }

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        uint128_gpu delta = extractDelta128(delta_array, bit_offset, delta_bits);

        // Reconstruct encoded value
        uint128_gpu encoded = pred_128 + delta;

        // Get original length and remove padding
        int8_t orig_len = d_original_lengths[global_idx];
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 128) {
            encoded = encoded >> pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint128ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

// ============================================================================
// 256-bit String Decompression Kernels
// ============================================================================

/**
 * Decompress to uint256_gpu encoded values
 */
__global__ void decompressToEncodedValues256Kernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    int num_partitions,
    uint256_gpu* __restrict__ d_encoded_values)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;

        // Calculate prediction (approximation)
        double predicted = fma(theta1, static_cast<double>(i), theta0);

        // Convert to 256-bit (approximation for practical values)
        uint256_gpu pred_256;
        if (predicted >= 0 && predicted < 1e77) {
            pred_256 = uint256_gpu(static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0)));
        }

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        uint256_gpu delta = extractDelta256(delta_array, bit_offset, delta_bits);

        // Reconstruct value
        d_encoded_values[global_idx] = pred_256 + delta;
    }
}

/**
 * Convert 256-bit encoded values back to strings
 */
__global__ void reconstructStrings256Kernel(
    const uint256_gpu* __restrict__ d_encoded_values,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* __restrict__ d_output_strings)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        char* output = d_output_strings + idx * output_string_stride;

        // Copy common prefix
        for (int i = 0; i < common_prefix_length; ++i) {
            output[i] = d_common_prefix[i];
        }

        // Decode suffix from encoded value
        uint256_gpu encoded = d_encoded_values[idx];
        int8_t orig_len = d_original_lengths[idx];

        // Remove padding
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 256) {
            encoded = encoded >> pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint256ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

/**
 * Combined 256-bit decompression directly to strings
 */
__global__ void decompressToStrings256Kernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_bit_offsets,
    const int8_t* __restrict__ d_original_lengths,
    const char* __restrict__ d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* __restrict__ d_output_strings)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    __shared__ double theta0, theta1;
    __shared__ int32_t delta_bits;
    __shared__ int64_t bit_offset_base;

    if (threadIdx.x == 0) {
        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];
        delta_bits = d_delta_bits[partition_idx];
        bit_offset_base = d_bit_offsets[partition_idx];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < segment_len; i += blockDim.x) {
        int global_idx = start_idx + i;
        char* output = d_output_strings + global_idx * output_string_stride;

        // Copy common prefix
        for (int j = 0; j < common_prefix_length; ++j) {
            output[j] = d_common_prefix[j];
        }

        // Calculate prediction (approximation)
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        uint256_gpu pred_256;
        if (predicted >= 0 && predicted < 1e77) {
            pred_256 = uint256_gpu(static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0)));
        }

        // Extract delta
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(i) * delta_bits;
        uint256_gpu delta = extractDelta256(delta_array, bit_offset, delta_bits);

        // Reconstruct encoded value
        uint256_gpu encoded = pred_256 + delta;

        // Get original length and remove padding
        int8_t orig_len = d_original_lengths[global_idx];
        int pad_chars = max_encoded_length - orig_len;
        int pad_bits = pad_chars * shift_bits;
        if (pad_bits > 0 && pad_bits < 256) {
            encoded = encoded >> pad_bits;
        }

        // Decode to string
        char* suffix_start = output + common_prefix_length;
        decodeUint256ToString(encoded, suffix_start, orig_len, min_char, shift_bits);

        // Null terminate
        int total_len = common_prefix_length + orig_len;
        if (total_len < output_string_stride) {
            output[total_len] = '\0';
        }
    }
}

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

void launchDecompressToEncodedValues(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    uint64_t* d_encoded_values,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToEncodedValuesKernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, d_encoded_values);
}

void launchReconstructStrings(
    const uint64_t* d_encoded_values,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    reconstructStringsKernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values, d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_strings, output_string_stride,
        d_output_strings);
}

void launchDecompressWarpOpt(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    uint64_t* d_encoded_values,
    cudaStream_t stream)
{
    // Each warp handles one partition
    int warps_needed = num_partitions;
    int threads = 256;
    int blocks = (warps_needed * WARP_SIZE + threads - 1) / threads;

    decompressWarpOptKernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_params, d_delta_bits, d_bit_offsets,
        num_partitions, d_encoded_values);
}

void launchDecompressToStrings(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToStringsKernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_params, d_delta_bits, d_bit_offsets,
        d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_partitions, output_string_stride,
        d_output_strings);
}

// ============================================================================
// 128-bit Kernel Launch Wrappers
// ============================================================================

void launchDecompressToEncodedValues128(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    uint128_gpu* d_encoded_values,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToEncodedValues128Kernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, d_encoded_values);
}

void launchReconstructStrings128(
    const uint128_gpu* d_encoded_values,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    reconstructStrings128Kernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values, d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_strings, output_string_stride,
        d_output_strings);
}

void launchDecompressToStrings128(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToStrings128Kernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_params, d_delta_bits, d_bit_offsets,
        d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_partitions, output_string_stride,
        d_output_strings);
}

// ============================================================================
// 256-bit Kernel Launch Wrappers
// ============================================================================

void launchDecompressToEncodedValues256(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    uint256_gpu* d_encoded_values,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToEncodedValues256Kernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, d_encoded_values);
}

void launchReconstructStrings256(
    const uint256_gpu* d_encoded_values,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    reconstructStrings256Kernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values, d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_strings, output_string_stride,
        d_output_strings);
}

void launchDecompressToStrings256(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;

    decompressToStrings256Kernel<<<blocks, threads, 0, stream>>>(
        delta_array,
        d_start_indices, d_end_indices,
        d_model_params, d_delta_bits, d_bit_offsets,
        d_original_lengths,
        d_common_prefix, common_prefix_length,
        max_encoded_length, min_char, shift_bits,
        num_partitions, output_string_stride,
        d_output_strings);
}
