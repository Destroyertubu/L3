/**
 * L3 String Encoder GPU Kernels
 *
 * GPU-optimized string compression using:
 * - Warp-level primitives for reduction
 * - Coalesced memory access
 * - Shared memory buffering
 * - 2's power base encoding for fast bit shifts
 *
 * Based on LeCo string compression with GPU optimizations.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cooperative_groups.h>
#include "L3_string_format.hpp"
#include "L3_format.hpp"

namespace cg = cooperative_groups;

#define WARP_SIZE 32
#define MAX_DELTA_BITS 64
#define FULL_MASK 0xFFFFFFFF

// ============================================================================
// Warp-level Primitives (same as numeric encoder)
// ============================================================================

template<typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ int64_t warpReduceMax(int64_t val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        int64_t other = __shfl_down_sync(mask, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

__device__ __forceinline__ bool warpAny(bool val) {
    unsigned mask = __activemask();
    return __any_sync(mask, val);
}

// ============================================================================
// String Encoding Device Functions
// ============================================================================

/**
 * Encode string to uint64_t using 2's power base (bit shifts)
 * Input: char array in global memory
 */
__device__ __forceinline__ uint64_t encodeStringToUint64(
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
        code = max(0, min(code, static_cast<int32_t>(mask)));
        result |= code;
    }

    return result;
}

/**
 * Encode string to uint128_gpu using 2's power base
 */
__device__ __forceinline__ uint128_gpu encodeStringToUint128(
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
        code = max(0, min(code, static_cast<int32_t>(mask)));
        result = result | uint128_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

/**
 * Encode string to uint256_gpu using 2's power base
 * For strings up to 51 characters with 5-bit encoding
 */
__device__ __forceinline__ uint256_gpu encodeStringToUint256(
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
        code = max(0, min(code, static_cast<int32_t>(mask)));
        result = result | uint256_gpu(static_cast<uint64_t>(code));
    }

    return result;
}

// ============================================================================
// String Batch Encoding Kernel
// ============================================================================

/**
 * Kernel to encode strings to uint64_t values
 *
 * Input:
 *   - d_strings: Flattened string data (all strings concatenated)
 *   - d_string_offsets: Start offset of each string in d_strings
 *   - d_string_lengths: Length of each string (after prefix removal)
 *   - num_strings: Number of strings to encode
 *
 * Output:
 *   - d_encoded_values: Encoded uint64_t values
 *   - d_original_lengths: Original string lengths (for decoding)
 */
__global__ void encodeStringsToUint64Kernel(
    const char* __restrict__ d_strings,
    const int32_t* __restrict__ d_string_offsets,
    const int8_t* __restrict__ d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint64_t* __restrict__ d_encoded_values,
    int8_t* __restrict__ d_original_lengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        int32_t offset = d_string_offsets[idx];
        int8_t length = d_string_lengths[idx];

        // Encode string
        uint64_t encoded = encodeStringToUint64(
            d_strings + offset, length, min_char, shift_bits);

        // Pad to max_length by shifting left
        int32_t pad_bits = (max_length - length) * shift_bits;
        if (pad_bits > 0) {
            encoded <<= pad_bits;
        }

        d_encoded_values[idx] = encoded;
        d_original_lengths[idx] = length;
    }
}

/**
 * Kernel to encode strings to uint128_gpu values
 * For strings with 13-25 characters (5-bit encoding)
 */
__global__ void encodeStringsToUint128Kernel(
    const char* __restrict__ d_strings,
    const int32_t* __restrict__ d_string_offsets,
    const int8_t* __restrict__ d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint128_gpu* __restrict__ d_encoded_values,
    int8_t* __restrict__ d_original_lengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        int32_t offset = d_string_offsets[idx];
        int8_t length = d_string_lengths[idx];

        // Encode string
        uint128_gpu encoded = encodeStringToUint128(
            d_strings + offset, length, min_char, shift_bits);

        // Pad to max_length by shifting left
        int32_t pad_bits = (max_length - length) * shift_bits;
        if (pad_bits > 0) {
            encoded = encoded << pad_bits;
        }

        d_encoded_values[idx] = encoded;
        d_original_lengths[idx] = length;
    }
}

/**
 * Kernel to encode strings to uint256_gpu values
 * For strings with 26-51 characters (5-bit encoding)
 */
__global__ void encodeStringsToUint256Kernel(
    const char* __restrict__ d_strings,
    const int32_t* __restrict__ d_string_offsets,
    const int8_t* __restrict__ d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint256_gpu* __restrict__ d_encoded_values,
    int8_t* __restrict__ d_original_lengths)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_strings) {
        int32_t offset = d_string_offsets[idx];
        int8_t length = d_string_lengths[idx];

        // Encode string
        uint256_gpu encoded = encodeStringToUint256(
            d_strings + offset, length, min_char, shift_bits);

        // Pad to max_length by shifting left
        int32_t pad_bits = (max_length - length) * shift_bits;
        if (pad_bits > 0) {
            encoded = encoded << pad_bits;
        }

        d_encoded_values[idx] = encoded;
        d_original_lengths[idx] = length;
    }
}

// ============================================================================
// String Model Fitting Kernel (Linear Regression on Encoded Values)
// ============================================================================

/**
 * Kernel to fit linear model to encoded string values per partition
 * Uses warp-level reductions for efficiency
 *
 * This kernel processes encoded uint64_t values the same way as numeric compression,
 * but is specialized for the uint64_t type used in string encoding.
 */
__global__ void fitStringModelKernel(
    const uint64_t* __restrict__ d_encoded_values,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_error_bounds,
    int num_partitions,
    int64_t* __restrict__ d_total_bits)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    // Shared memory for warp-level reductions
    extern __shared__ char shared_mem[];
    double* warp_sums = reinterpret_cast<double*>(shared_mem);
    int64_t* warp_max_error = reinterpret_cast<int64_t*>(warp_sums + num_warps * 4);

    // Phase 1: Fit linear model using warp-level reductions
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(d_encoded_values[start_idx + i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    // Warp-level reduction
    sum_x = warpReduceSum(sum_x);
    sum_y = warpReduceSum(sum_y);
    sum_xx = warpReduceSum(sum_xx);
    sum_xy = warpReduceSum(sum_xy);

    // Store warp results
    if (lane_id == 0) {
        warp_sums[warp_id * 4] = sum_x;
        warp_sums[warp_id * 4 + 1] = sum_y;
        warp_sums[warp_id * 4 + 2] = sum_xx;
        warp_sums[warp_id * 4 + 3] = sum_xy;
    }
    __syncthreads();

    // Final reduction in first warp
    double theta0, theta1;
    if (warp_id == 0 && lane_id < num_warps) {
        sum_x = warp_sums[lane_id * 4];
        sum_y = warp_sums[lane_id * 4 + 1];
        sum_xx = warp_sums[lane_id * 4 + 2];
        sum_xy = warp_sums[lane_id * 4 + 3];

        sum_x = warpReduceSum(sum_x);
        sum_y = warpReduceSum(sum_y);
        sum_xx = warpReduceSum(sum_xx);
        sum_xy = warpReduceSum(sum_xy);

        if (lane_id == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * sum_xx - sum_x * sum_x;

            if (fabs(determinant) > 1e-10) {
                theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
                theta0 = (sum_y - theta1 * sum_x) / n;
            } else {
                theta1 = 0.0;
                theta0 = sum_y / n;
            }

            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;

            // Store in shared memory for other threads
            warp_sums[0] = theta0;
            warp_sums[1] = theta1;
        }
    }
    __syncthreads();

    theta0 = warp_sums[0];
    theta1 = warp_sums[1];

    // Phase 2: Calculate maximum error
    int64_t max_error = 0;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        double predicted = theta0 + theta1 * i;
        uint64_t actual = d_encoded_values[start_idx + i];
        int64_t pred_int = static_cast<int64_t>(round(predicted));
        int64_t actual_int = static_cast<int64_t>(actual);
        int64_t delta = actual_int - pred_int;
        int64_t abs_error = (delta < 0) ? -delta : delta;
        max_error = (abs_error > max_error) ? abs_error : max_error;
    }

    // Warp-level max reduction
    max_error = warpReduceMax(max_error);

    if (lane_id == 0) {
        warp_max_error[warp_id] = max_error;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0 && lane_id < num_warps) {
        max_error = warp_max_error[lane_id];
        max_error = warpReduceMax(max_error);

        if (lane_id == 0) {
            d_error_bounds[partition_idx] = max_error;

            // Calculate delta bits
            int delta_bits = 0;
            if (max_error > 0) {
                unsigned long long temp = static_cast<unsigned long long>(max_error);
                delta_bits = 64 - __clzll(temp) + 1;  // +1 for sign bit
                delta_bits = min(delta_bits, MAX_DELTA_BITS);
                delta_bits = max(delta_bits, 0);
            }
            d_delta_bits[partition_idx] = delta_bits;

            // Update total bits counter
            int64_t partition_bits = static_cast<int64_t>(segment_len) * delta_bits;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_bits),
                      static_cast<unsigned long long>(partition_bits));
        }
    }
}

// ============================================================================
// String Delta Packing Kernel
// ============================================================================

/**
 * Kernel to calculate and pack delta values for encoded strings
 */
__global__ void packStringDeltasKernel(
    const uint64_t* __restrict__ d_encoded_values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* __restrict__ delta_array)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions == 0) return;

    int max_idx = d_end_indices[num_partitions - 1];

    for (int current_idx = g_idx; current_idx < max_idx; current_idx += g_stride) {
        // Find partition for this index (binary search)
        int p_left = 0, p_right = num_partitions - 1;
        int partition_idx = -1;

        // Fast path: estimate partition
        int estimated = current_idx * num_partitions / max_idx;
        if (estimated < num_partitions) {
            if (current_idx >= d_start_indices[estimated] &&
                current_idx < d_end_indices[estimated]) {
                partition_idx = estimated;
            }
        }

        // Binary search fallback
        if (partition_idx == -1) {
            while (p_left <= p_right) {
                int mid = p_left + (p_right - p_left) / 2;
                if (current_idx >= d_start_indices[mid] &&
                    current_idx < d_end_indices[mid]) {
                    partition_idx = mid;
                    break;
                } else if (current_idx < d_start_indices[mid]) {
                    p_right = mid - 1;
                } else {
                    p_left = mid + 1;
                }
            }
        }

        if (partition_idx == -1) continue;

        // Get partition parameters
        int32_t delta_bits = d_delta_bits[partition_idx];
        if (delta_bits <= 0) continue;

        int64_t bit_offset_base = d_delta_array_bit_offsets[partition_idx];
        int32_t start_idx = d_start_indices[partition_idx];
        int32_t local_idx = current_idx - start_idx;

        // Calculate prediction
        double theta0 = d_model_params[partition_idx * 4];
        double theta1 = d_model_params[partition_idx * 4 + 1];
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);

        // Calculate delta
        uint64_t actual = d_encoded_values[current_idx];
        int64_t pred_int = static_cast<int64_t>(round(predicted));
        int64_t delta = static_cast<int64_t>(actual) - pred_int;

        // Pack delta into bit array
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;

        if (delta_bits <= 32) {
            uint32_t packed = static_cast<uint32_t>(delta & ((1ULL << delta_bits) - 1ULL));
            int word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;

            if (delta_bits + offset_in_word <= 32) {
                atomicOr(&delta_array[word_idx], packed << offset_in_word);
            } else {
                atomicOr(&delta_array[word_idx], packed << offset_in_word);
                atomicOr(&delta_array[word_idx + 1], packed >> (32 - offset_in_word));
            }
        } else {
            // For deltas > 32 bits
            uint64_t packed = delta & ((delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL));
            int word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;
            int bits_remaining = delta_bits;

            while (bits_remaining > 0) {
                int bits_in_word = min(bits_remaining, 32 - offset_in_word);
                uint32_t mask = (bits_in_word == 32) ? ~0U : ((1U << bits_in_word) - 1U);
                uint32_t value = static_cast<uint32_t>(packed & mask) << offset_in_word;
                atomicOr(&delta_array[word_idx], value);

                packed >>= bits_in_word;
                bits_remaining -= bits_in_word;
                word_idx++;
                offset_in_word = 0;
            }
        }
    }
}

// ============================================================================
// 128-bit String Model Fitting and Delta Packing
// ============================================================================

/**
 * Warp reduction for uint128_gpu max
 */
__device__ __forceinline__ uint128_gpu warpReduceMax128(uint128_gpu val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        uint128_gpu other;
        other.low = __shfl_down_sync(mask, val.low, offset);
        other.high = __shfl_down_sync(mask, val.high, offset);
        if (other > val) val = other;
    }
    return val;
}

/**
 * Kernel to fit linear model to 128-bit encoded string values per partition
 * Uses simplified double-precision fitting (works for most practical string encodings)
 */
__global__ void fitStringModel128Kernel(
    const uint128_gpu* __restrict__ d_encoded_values,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    uint128_gpu* __restrict__ d_error_bounds,
    int num_partitions,
    int64_t* __restrict__ d_total_bits)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ char shared_mem[];
    double* warp_sums = reinterpret_cast<double*>(shared_mem);
    uint128_gpu* warp_max_error = reinterpret_cast<uint128_gpu*>(warp_sums + num_warps * 4);

    // Phase 1: Fit linear model using least squares on lower 64 bits
    // For 128-bit values, we use the high bits for slope estimation
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        double x = static_cast<double>(i);
        uint128_gpu val = d_encoded_values[start_idx + i];
        // Use full precision for small values, approximate for large
        double y = static_cast<double>(val.high) * 18446744073709551616.0 +
                   static_cast<double>(val.low);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    sum_x = warpReduceSum(sum_x);
    sum_y = warpReduceSum(sum_y);
    sum_xx = warpReduceSum(sum_xx);
    sum_xy = warpReduceSum(sum_xy);

    if (lane_id == 0) {
        warp_sums[warp_id * 4] = sum_x;
        warp_sums[warp_id * 4 + 1] = sum_y;
        warp_sums[warp_id * 4 + 2] = sum_xx;
        warp_sums[warp_id * 4 + 3] = sum_xy;
    }
    __syncthreads();

    double theta0, theta1;
    if (warp_id == 0 && lane_id < num_warps) {
        sum_x = warp_sums[lane_id * 4];
        sum_y = warp_sums[lane_id * 4 + 1];
        sum_xx = warp_sums[lane_id * 4 + 2];
        sum_xy = warp_sums[lane_id * 4 + 3];

        sum_x = warpReduceSum(sum_x);
        sum_y = warpReduceSum(sum_y);
        sum_xx = warpReduceSum(sum_xx);
        sum_xy = warpReduceSum(sum_xy);

        if (lane_id == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * sum_xx - sum_x * sum_x;

            if (fabs(determinant) > 1e-10) {
                theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
                theta0 = (sum_y - theta1 * sum_x) / n;
            } else {
                theta1 = 0.0;
                theta0 = sum_y / n;
            }

            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;

            warp_sums[0] = theta0;
            warp_sums[1] = theta1;
        }
    }
    __syncthreads();

    theta0 = warp_sums[0];
    theta1 = warp_sums[1];

    // Phase 2: Calculate maximum error using 128-bit arithmetic
    uint128_gpu max_error;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        double predicted = theta0 + theta1 * i;
        uint128_gpu actual = d_encoded_values[start_idx + i];

        // Convert prediction to 128-bit
        uint128_gpu pred_128;
        if (predicted >= 0) {
            pred_128.high = static_cast<uint64_t>(predicted / 18446744073709551616.0);
            pred_128.low = static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0));
        } else {
            pred_128 = uint128_gpu(0);
        }

        // Calculate absolute error
        uint128_gpu error;
        if (actual > pred_128) {
            error = actual - pred_128;
        } else {
            error = pred_128 - actual;
        }

        if (error > max_error) max_error = error;
    }

    max_error = warpReduceMax128(max_error);

    if (lane_id == 0) {
        warp_max_error[warp_id] = max_error;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        max_error = warp_max_error[lane_id];
        max_error = warpReduceMax128(max_error);

        if (lane_id == 0) {
            d_error_bounds[partition_idx] = max_error;

            // Calculate delta bits from max_error
            int delta_bits = 0;
            if (max_error.high > 0) {
                delta_bits = 128 - __clzll(max_error.high) + 1;
            } else if (max_error.low > 0) {
                delta_bits = 64 - __clzll(max_error.low) + 1;
            }
            delta_bits = min(delta_bits, 128);
            d_delta_bits[partition_idx] = delta_bits;

            int64_t partition_bits = static_cast<int64_t>(segment_len) * delta_bits;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_bits),
                      static_cast<unsigned long long>(partition_bits));
        }
    }
}

/**
 * Kernel to pack 128-bit delta values
 */
__global__ void packStringDeltas128Kernel(
    const uint128_gpu* __restrict__ d_encoded_values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* __restrict__ delta_array)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions == 0) return;

    int max_idx = d_end_indices[num_partitions - 1];

    for (int current_idx = g_idx; current_idx < max_idx; current_idx += g_stride) {
        // Binary search for partition
        int p_left = 0, p_right = num_partitions - 1;
        int partition_idx = -1;

        int estimated = current_idx * num_partitions / max_idx;
        if (estimated < num_partitions) {
            if (current_idx >= d_start_indices[estimated] &&
                current_idx < d_end_indices[estimated]) {
                partition_idx = estimated;
            }
        }

        if (partition_idx == -1) {
            while (p_left <= p_right) {
                int mid = p_left + (p_right - p_left) / 2;
                if (current_idx >= d_start_indices[mid] &&
                    current_idx < d_end_indices[mid]) {
                    partition_idx = mid;
                    break;
                } else if (current_idx < d_start_indices[mid]) {
                    p_right = mid - 1;
                } else {
                    p_left = mid + 1;
                }
            }
        }

        if (partition_idx == -1) continue;

        int32_t delta_bits = d_delta_bits[partition_idx];
        if (delta_bits <= 0) continue;

        int64_t bit_offset_base = d_delta_array_bit_offsets[partition_idx];
        int32_t start_idx = d_start_indices[partition_idx];
        int32_t local_idx = current_idx - start_idx;

        double theta0 = d_model_params[partition_idx * 4];
        double theta1 = d_model_params[partition_idx * 4 + 1];
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);

        uint128_gpu actual = d_encoded_values[current_idx];

        // Convert prediction to 128-bit
        uint128_gpu pred_128;
        if (predicted >= 0) {
            pred_128.high = static_cast<uint64_t>(predicted / 18446744073709551616.0);
            pred_128.low = static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0));
        } else {
            pred_128 = uint128_gpu(0);
        }

        // Calculate signed delta (we use two's complement representation)
        uint128_gpu delta = actual - pred_128;

        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;

        // Pack delta into bit array (up to 128 bits)
        int word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = delta_bits;

        // Pack low 64 bits
        uint64_t packed = delta.low;
        int bits_to_pack = min(bits_remaining, 64);

        while (bits_to_pack > 0) {
            int bits_in_word = min(bits_to_pack, 32 - offset_in_word);
            uint32_t mask = (bits_in_word == 32) ? ~0U : ((1U << bits_in_word) - 1U);
            uint32_t value = static_cast<uint32_t>(packed & mask) << offset_in_word;
            atomicOr(&delta_array[word_idx], value);

            packed >>= bits_in_word;
            bits_to_pack -= bits_in_word;
            bits_remaining -= bits_in_word;
            word_idx++;
            offset_in_word = 0;
        }

        // Pack high 64 bits if needed
        if (bits_remaining > 0 && delta_bits > 64) {
            packed = delta.high;
            bits_to_pack = min(bits_remaining, 64);

            while (bits_to_pack > 0) {
                int bits_in_word = min(bits_to_pack, 32 - offset_in_word);
                uint32_t mask = (bits_in_word == 32) ? ~0U : ((1U << bits_in_word) - 1U);
                uint32_t value = static_cast<uint32_t>(packed & mask) << offset_in_word;
                atomicOr(&delta_array[word_idx], value);

                packed >>= bits_in_word;
                bits_to_pack -= bits_in_word;
                word_idx++;
                offset_in_word = 0;
            }
        }
    }
}

// ============================================================================
// 256-bit String Model Fitting and Delta Packing
// ============================================================================

/**
 * Warp reduction for uint256_gpu max
 */
__device__ __forceinline__ uint256_gpu warpReduceMax256(uint256_gpu val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        uint256_gpu other;
        other.words[0] = __shfl_down_sync(mask, val.words[0], offset);
        other.words[1] = __shfl_down_sync(mask, val.words[1], offset);
        other.words[2] = __shfl_down_sync(mask, val.words[2], offset);
        other.words[3] = __shfl_down_sync(mask, val.words[3], offset);
        if (other > val) val = other;
    }
    return val;
}

/**
 * Kernel to fit linear model to 256-bit encoded string values per partition
 */
__global__ void fitStringModel256Kernel(
    const uint256_gpu* __restrict__ d_encoded_values,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    uint256_gpu* __restrict__ d_error_bounds,
    int num_partitions,
    int64_t* __restrict__ d_total_bits)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    extern __shared__ char shared_mem[];
    double* warp_sums = reinterpret_cast<double*>(shared_mem);
    uint256_gpu* warp_max_error = reinterpret_cast<uint256_gpu*>(warp_sums + num_warps * 4);

    // Phase 1: Fit linear model (approximation using most significant bits)
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        double x = static_cast<double>(i);
        uint256_gpu val = d_encoded_values[start_idx + i];
        // Approximate: use highest non-zero word for scaling
        double y;
        if (val.words[3] > 0) {
            y = static_cast<double>(val.words[3]) * 3.4028236692093846e+38 * 3.4028236692093846e+38;
        } else if (val.words[2] > 0) {
            y = static_cast<double>(val.words[2]) * 3.4028236692093846e+38;
        } else if (val.words[1] > 0) {
            y = static_cast<double>(val.words[1]) * 18446744073709551616.0;
        } else {
            y = static_cast<double>(val.words[0]);
        }
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    sum_x = warpReduceSum(sum_x);
    sum_y = warpReduceSum(sum_y);
    sum_xx = warpReduceSum(sum_xx);
    sum_xy = warpReduceSum(sum_xy);

    if (lane_id == 0) {
        warp_sums[warp_id * 4] = sum_x;
        warp_sums[warp_id * 4 + 1] = sum_y;
        warp_sums[warp_id * 4 + 2] = sum_xx;
        warp_sums[warp_id * 4 + 3] = sum_xy;
    }
    __syncthreads();

    double theta0, theta1;
    if (warp_id == 0 && lane_id < num_warps) {
        sum_x = warp_sums[lane_id * 4];
        sum_y = warp_sums[lane_id * 4 + 1];
        sum_xx = warp_sums[lane_id * 4 + 2];
        sum_xy = warp_sums[lane_id * 4 + 3];

        sum_x = warpReduceSum(sum_x);
        sum_y = warpReduceSum(sum_y);
        sum_xx = warpReduceSum(sum_xx);
        sum_xy = warpReduceSum(sum_xy);

        if (lane_id == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * sum_xx - sum_x * sum_x;

            if (fabs(determinant) > 1e-10) {
                theta1 = (n * sum_xy - sum_x * sum_y) / determinant;
                theta0 = (sum_y - theta1 * sum_x) / n;
            } else {
                theta1 = 0.0;
                theta0 = sum_y / n;
            }

            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;

            warp_sums[0] = theta0;
            warp_sums[1] = theta1;
        }
    }
    __syncthreads();

    theta0 = warp_sums[0];
    theta1 = warp_sums[1];

    // Phase 2: Calculate maximum error using 256-bit arithmetic
    uint256_gpu max_error;

    for (int i = tid; i < segment_len; i += blockDim.x) {
        uint256_gpu actual = d_encoded_values[start_idx + i];

        // Simple prediction conversion (for practical string values)
        double predicted = theta0 + theta1 * i;
        uint256_gpu pred_256;
        if (predicted >= 0 && predicted < 1e77) {
            // Approximate conversion
            pred_256 = uint256_gpu(static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0)));
        }

        uint256_gpu error;
        if (actual > pred_256) {
            error = actual - pred_256;
        } else {
            error = pred_256 - actual;
        }

        if (error > max_error) max_error = error;
    }

    max_error = warpReduceMax256(max_error);

    if (lane_id == 0) {
        warp_max_error[warp_id] = max_error;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < num_warps) {
        max_error = warp_max_error[lane_id];
        max_error = warpReduceMax256(max_error);

        if (lane_id == 0) {
            d_error_bounds[partition_idx] = max_error;

            // Calculate delta bits
            int delta_bits = 0;
            if (max_error.words[3] > 0) {
                delta_bits = 192 + 64 - __clzll(max_error.words[3]) + 1;
            } else if (max_error.words[2] > 0) {
                delta_bits = 128 + 64 - __clzll(max_error.words[2]) + 1;
            } else if (max_error.words[1] > 0) {
                delta_bits = 64 + 64 - __clzll(max_error.words[1]) + 1;
            } else if (max_error.words[0] > 0) {
                delta_bits = 64 - __clzll(max_error.words[0]) + 1;
            }
            delta_bits = min(delta_bits, 256);
            d_delta_bits[partition_idx] = delta_bits;

            int64_t partition_bits = static_cast<int64_t>(segment_len) * delta_bits;
            atomicAdd(reinterpret_cast<unsigned long long*>(d_total_bits),
                      static_cast<unsigned long long>(partition_bits));
        }
    }
}

/**
 * Kernel to pack 256-bit delta values
 */
__global__ void packStringDeltas256Kernel(
    const uint256_gpu* __restrict__ d_encoded_values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* __restrict__ delta_array)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions == 0) return;

    int max_idx = d_end_indices[num_partitions - 1];

    for (int current_idx = g_idx; current_idx < max_idx; current_idx += g_stride) {
        int p_left = 0, p_right = num_partitions - 1;
        int partition_idx = -1;

        int estimated = current_idx * num_partitions / max_idx;
        if (estimated < num_partitions) {
            if (current_idx >= d_start_indices[estimated] &&
                current_idx < d_end_indices[estimated]) {
                partition_idx = estimated;
            }
        }

        if (partition_idx == -1) {
            while (p_left <= p_right) {
                int mid = p_left + (p_right - p_left) / 2;
                if (current_idx >= d_start_indices[mid] &&
                    current_idx < d_end_indices[mid]) {
                    partition_idx = mid;
                    break;
                } else if (current_idx < d_start_indices[mid]) {
                    p_right = mid - 1;
                } else {
                    p_left = mid + 1;
                }
            }
        }

        if (partition_idx == -1) continue;

        int32_t delta_bits = d_delta_bits[partition_idx];
        if (delta_bits <= 0) continue;

        int64_t bit_offset_base = d_delta_array_bit_offsets[partition_idx];
        int32_t start_idx = d_start_indices[partition_idx];
        int32_t local_idx = current_idx - start_idx;

        double theta0 = d_model_params[partition_idx * 4];
        double theta1 = d_model_params[partition_idx * 4 + 1];
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);

        uint256_gpu actual = d_encoded_values[current_idx];

        // Convert prediction
        uint256_gpu pred_256;
        if (predicted >= 0 && predicted < 1e77) {
            pred_256 = uint256_gpu(static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0)));
        }

        uint256_gpu delta = actual - pred_256;

        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;

        // Pack delta (up to 256 bits)
        int word_idx = bit_offset / 32;
        int offset_in_word = bit_offset % 32;
        int bits_remaining = delta_bits;

        // Pack each 64-bit word
        for (int w = 0; w < 4 && bits_remaining > 0; w++) {
            uint64_t packed = delta.words[w];
            int bits_to_pack = min(bits_remaining, 64);

            while (bits_to_pack > 0) {
                int bits_in_word = min(bits_to_pack, 32 - offset_in_word);
                uint32_t mask = (bits_in_word == 32) ? ~0U : ((1U << bits_in_word) - 1U);
                uint32_t value = static_cast<uint32_t>(packed & mask) << offset_in_word;
                atomicOr(&delta_array[word_idx], value);

                packed >>= bits_in_word;
                bits_to_pack -= bits_in_word;
                bits_remaining -= bits_in_word;
                word_idx++;
                offset_in_word = 0;
            }
        }
    }
}

// ============================================================================
// Bit Offset Calculation Kernel
// ============================================================================

__global__ void setStringBitOffsetsKernel(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_bit_offsets,
    int num_partitions)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_partitions) {
        // Calculate bit offset using sequential prefix sum
        int64_t offset = 0;
        for (int i = 0; i < tid; i++) {
            int seg_len = d_end_indices[i] - d_start_indices[i];
            offset += static_cast<int64_t>(seg_len) * d_delta_bits[i];
        }
        d_bit_offsets[tid] = offset;
    }
}

// ============================================================================
// Kernel Launch Wrappers
// ============================================================================

void launchEncodeStringsToUint64(
    const char* d_strings,
    const int32_t* d_string_offsets,
    const int8_t* d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint64_t* d_encoded_values,
    int8_t* d_original_lengths,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    encodeStringsToUint64Kernel<<<blocks, threads, 0, stream>>>(
        d_strings, d_string_offsets, d_string_lengths,
        num_strings, min_char, shift_bits, max_length,
        d_encoded_values, d_original_lengths);
}

void launchFitStringModel(
    const uint64_t* d_encoded_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;
    int num_warps = threads / WARP_SIZE;

    size_t shared_mem_size = num_warps * 4 * sizeof(double) +
                             num_warps * sizeof(int64_t);

    fitStringModelKernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_error_bounds,
        num_partitions, d_total_bits);
}

void launchSetStringBitOffsets(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_delta_bits,
    int64_t* d_bit_offsets,
    int num_partitions,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_partitions + threads - 1) / threads;

    setStringBitOffsetsKernel<<<blocks, threads, 0, stream>>>(
        d_start_indices, d_end_indices, d_delta_bits,
        d_bit_offsets, num_partitions);
}

void launchPackStringDeltas(
    const uint64_t* d_encoded_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    int total_elements,
    uint32_t* delta_array,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
    blocks = min(blocks, 65535);

    packStringDeltasKernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, delta_array);
}

// ============================================================================
// 128-bit Kernel Launch Wrappers
// ============================================================================

void launchEncodeStringsToUint128(
    const char* d_strings,
    const int32_t* d_string_offsets,
    const int8_t* d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint128_gpu* d_encoded_values,
    int8_t* d_original_lengths,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    encodeStringsToUint128Kernel<<<blocks, threads, 0, stream>>>(
        d_strings, d_string_offsets, d_string_lengths,
        num_strings, min_char, shift_bits, max_length,
        d_encoded_values, d_original_lengths);
}

void launchFitStringModel128(
    const uint128_gpu* d_encoded_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    uint128_gpu* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;
    int num_warps = threads / WARP_SIZE;

    size_t shared_mem_size = num_warps * 4 * sizeof(double) +
                             num_warps * sizeof(uint128_gpu);

    fitStringModel128Kernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_error_bounds,
        num_partitions, d_total_bits);
}

void launchPackStringDeltas128(
    const uint128_gpu* d_encoded_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    int total_elements,
    uint32_t* delta_array,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
    blocks = min(blocks, 65535);

    packStringDeltas128Kernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, delta_array);
}

// ============================================================================
// 256-bit Kernel Launch Wrappers
// ============================================================================

void launchEncodeStringsToUint256(
    const char* d_strings,
    const int32_t* d_string_offsets,
    const int8_t* d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint256_gpu* d_encoded_values,
    int8_t* d_original_lengths,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_strings + threads - 1) / threads;

    encodeStringsToUint256Kernel<<<blocks, threads, 0, stream>>>(
        d_strings, d_string_offsets, d_string_lengths,
        num_strings, min_char, shift_bits, max_length,
        d_encoded_values, d_original_lengths);
}

void launchFitStringModel256(
    const uint256_gpu* d_encoded_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    uint256_gpu* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = num_partitions;
    int num_warps = threads / WARP_SIZE;

    size_t shared_mem_size = num_warps * 4 * sizeof(double) +
                             num_warps * sizeof(uint256_gpu);

    fitStringModel256Kernel<<<blocks, threads, shared_mem_size, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_error_bounds,
        num_partitions, d_total_bits);
}

void launchPackStringDeltas256(
    const uint256_gpu* d_encoded_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    int total_elements,
    uint32_t* delta_array,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads * 4 - 1) / (threads * 4);
    blocks = min(blocks, 65535);

    packStringDeltas256Kernel<<<blocks, threads, 0, stream>>>(
        d_encoded_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_bit_offsets,
        num_partitions, delta_array);
}
