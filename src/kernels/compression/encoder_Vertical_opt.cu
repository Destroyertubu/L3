/**
 * L3 Vertical-Optimized Encoder
 *
 * Extends the original L3 encoder with Vertical-inspired optimizations:
 *
 * 1. DUAL-FORMAT ENCODING:
 *    - Sequential format preserved for random access
 *    - Interleaved mini-vector format added for batch scan
 *
 * 2. MINI-VECTOR INTERLEAVED LAYOUT (256 values):
 *    - 32 lanes (warp threads) × 8 values per lane
 *    - Coalesced memory access pattern
 *    - Warp-level parallelism in decompression
 *
 * 3. OPTIMIZED PACKING:
 *    - Branchless bit packing for sequential format
 *    - Parallel reordering for interleaved format
 *
 * COMPATIBILITY:
 * - Extends CompressedDataL3 without breaking existing decoders
 * - Sequential format remains identical for backward compatibility
 *
 * Platform: SM 8.0+ (Ampere and later)
 * Date: 2025-12-04
 */

#ifndef ENCODER_Vertical_OPT_CU
#define ENCODER_Vertical_OPT_CU

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <cstring>
#include <type_traits>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../utils/bitpack_utils_Vertical.cuh"
#include "../utils/finite_diff_shared.cuh"  // Shared finite difference functions
#include "adaptive_selector.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"  // GPU cost-optimal partitioner
#include "encoder_cost_optimal_gpu_merge_v3.cuh"

namespace Vertical_encoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr double MODEL_OVERHEAD_BYTES = 64.0;

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * Compute prediction using polynomial model (Horner's method) - matches decoder exactly
 * CRITICAL: Use __dadd_rn/__dmul_rn to match decoder's FP64 computation (no FMA)
 */
template<typename T>
__device__ __forceinline__
T computePredictionPoly(int32_t model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:  // 0
            predicted = params[0];
            break;
        case MODEL_LINEAR:    // 1
            predicted = __dadd_rn(params[0], __dmul_rn(params[1], x));
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = __dadd_rn(params[0], __dmul_rn(x, __dadd_rn(params[1], __dmul_rn(x, params[2]))));
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = __dadd_rn(params[0], __dmul_rn(x, __dadd_rn(params[1], __dmul_rn(x, __dadd_rn(params[2], __dmul_rn(x, params[3]))))));
            break;
        default:
            // Fallback to linear for unknown types
            predicted = __dadd_rn(params[0], __dmul_rn(params[1], x));
            break;
    }

    // CRITICAL: Use __double2ull_rn for uint64_t to avoid overflow when predicted > INT64_MAX
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(predicted));
    } else {
        return static_cast<T>(__double2ll_rn(predicted));
    }
}

// =============================================================================
// Finite Difference Prediction Functions (FP64 accumulation for precision)
// These MUST match decoder exactly to avoid rounding errors
// =============================================================================

// INTEGER version - fast, uses int64_t accumulation
// Initial value and step computed from FP64, then pure integer arithmetic
// CRITICAL: step must be computed as difference of rounded values, not rounded difference
template<typename T>
__device__ __forceinline__
void computeLinearFiniteDiff_INT(
    const double* params,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction (output, INT64)
    int64_t& step)   // step per iteration (output, INT64)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0);
    // CRITICAL: step = y1 - y0, not __double2ll_rn(b * s)
    int64_t y1 = __double2ll_rn(a + b * (x0 + s));
    step = y1 - y;
}

template<typename T>
__device__ __forceinline__
void computePoly2FiniteDiff_INT(
    const double* params,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction
    int64_t& d1,     // first difference
    int64_t& d2)     // second difference (constant)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0 + c * x0 * x0);
    d1 = __double2ll_rn(b * s + c * s * (2.0 * x0 + s));
    d2 = __double2ll_rn(2.0 * c * s * s);
}

template<typename T>
__device__ __forceinline__
void computePoly3FiniteDiff_INT(
    const double* params,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction
    int64_t& d1,     // first difference
    int64_t& d2,     // second difference
    int64_t& d3)     // third difference (constant)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double d = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0 + c * x0 * x0 + d * x0 * x0 * x0);

    double x1 = x0 + s;
    double y1_fp = a + b * x1 + c * x1 * x1 + d * x1 * x1 * x1;
    d1 = __double2ll_rn(y1_fp) - y;

    double x2 = x0 + 2.0 * s;
    double y2_fp = a + b * x2 + c * x2 * x2 + d * x2 * x2 * x2;
    int64_t d1_next = __double2ll_rn(y2_fp) - __double2ll_rn(y1_fp);
    d2 = d1_next - d1;

    d3 = __double2ll_rn(6.0 * d * s * s * s);
}

// FP64 version - uses __dadd_rn/__dmul_rn to match decoder exactly
template<typename T>
__device__ __forceinline__
void computeLinearFiniteDiff_FP64(
    const double* params,
    int start_idx,
    int stride,
    double& y,       // initial prediction (output, FP64)
    double& step)    // step per iteration (output, FP64)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);
    y = __dadd_rn(a, __dmul_rn(b, x0));
    step = __dmul_rn(b, s);
}

template<typename T>
__device__ __forceinline__
void computePoly2FiniteDiff_FP64(
    const double* params,
    int start_idx,
    int stride,
    double& y,       // initial prediction
    double& d1,      // first difference
    double& d2)      // second difference (constant)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    double x0_sq = __dmul_rn(x0, x0);
    y = __dadd_rn(__dadd_rn(a, __dmul_rn(b, x0)), __dmul_rn(c, x0_sq));

    double two_x0_plus_s = __dadd_rn(__dmul_rn(2.0, x0), s);
    double c_s = __dmul_rn(c, s);
    d1 = __dadd_rn(__dmul_rn(b, s), __dmul_rn(c_s, two_x0_plus_s));

    d2 = __dmul_rn(2.0, __dmul_rn(c, __dmul_rn(s, s)));
}

template<typename T>
__device__ __forceinline__
void computePoly3FiniteDiff_FP64(
    const double* params,
    int start_idx,
    int stride,
    double& y,       // initial prediction
    double& d1,      // first difference
    double& d2,      // second difference
    double& d3)      // third difference (constant)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double d = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = __dmul_rn(x0, x0);
    double x0_cu = __dmul_rn(x0_sq, x0);
    y = __dadd_rn(__dadd_rn(__dadd_rn(a, __dmul_rn(b, x0)), __dmul_rn(c, x0_sq)), __dmul_rn(d, x0_cu));

    // Compute y1
    double x1 = __dadd_rn(x0, s);
    double x1_sq = __dmul_rn(x1, x1);
    double x1_cu = __dmul_rn(x1_sq, x1);
    double y1 = __dadd_rn(__dadd_rn(__dadd_rn(a, __dmul_rn(b, x1)), __dmul_rn(c, x1_sq)), __dmul_rn(d, x1_cu));
    d1 = __dadd_rn(y1, -y);

    // Compute y2
    double x2 = __dadd_rn(x0, __dmul_rn(2.0, s));
    double x2_sq = __dmul_rn(x2, x2);
    double x2_cu = __dmul_rn(x2_sq, x2);
    double y2 = __dadd_rn(__dadd_rn(__dadd_rn(a, __dmul_rn(b, x2)), __dmul_rn(c, x2_sq)), __dmul_rn(d, x2_cu));
    double d1_next = __dadd_rn(y2, -y1);
    d2 = __dadd_rn(d1_next, -d1);

    // Third difference is constant
    d3 = __dmul_rn(6.0, __dmul_rn(d, __dmul_rn(s, __dmul_rn(s, s))));
}

template<typename T>
__device__ __forceinline__
T fp64_to_int(double val) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(val));
    } else {
        return static_cast<T>(__double2ll_rn(val));
    }
}

__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ int64_t warpReduceMax(int64_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__device__ __forceinline__ int64_t blockReduceMax(int64_t val) {
    __shared__ int64_t shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : LLONG_MIN;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ __forceinline__ int computeBitsForValue(uint64_t val) {
    if (val == 0) return 0;
    return 64 - __clzll(val);
}

template<typename T>
__device__ __forceinline__ int64_t calculateDelta(T actual, T predicted) {
    if (actual >= predicted) {
        return static_cast<int64_t>(actual - predicted);
    } else {
        return -static_cast<int64_t>(predicted - actual);
    }
}

// ============================================================================
// Sequential Delta Packing Kernel (Branchless Optimized)
// ============================================================================

/**
 * Branchless delta packing - eliminates conditional branches for word boundaries
 *
 * KEY OPTIMIZATION: Always writes to two words, but the second write
 * is harmless (writes zeros) when value doesn't cross boundary.
 */
template<typename T>
__global__ void packDeltasSequentialBranchless(
    const T* __restrict__ values,
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

    for (int idx = g_idx; idx < max_idx; idx += g_stride) {
        // Binary search for partition
        int left = 0, right = num_partitions - 1;
        int pid = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (idx >= d_start_indices[mid] && idx < d_end_indices[mid]) {
                pid = mid;
                break;
            } else if (idx < d_start_indices[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (pid < 0) continue;

        int32_t model_type = d_model_types[pid];

        // Skip CONSTANT/RLE partitions - they are encoded in packTailValuesKernel
        if (model_type == MODEL_CONSTANT) continue;

        int32_t delta_bits = d_delta_bits[pid];
        int64_t bit_offset_base = d_delta_array_bit_offsets[pid];
        int32_t start_idx = d_start_indices[pid];
        int local_idx = idx - start_idx;

        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;

        uint64_t packed_value;

        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: delta = val - base (base stored in theta0)
            // Use bit-pattern copy to restore exact integer for 64-bit types (values > 2^53)
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(d_model_params[pid * 4]));
            } else {
                base = static_cast<T>(__double2ll_rn(d_model_params[pid * 4]));
            }
            packed_value = static_cast<uint64_t>(values[idx] - base);
        } else {
            // Compute polynomial prediction using Horner's method
            // CRITICAL: Use __dadd_rn/__dmul_rn to match decoder's FP64 accumulation
            double theta0 = d_model_params[pid * 4];
            double theta1 = d_model_params[pid * 4 + 1];
            double theta2 = d_model_params[pid * 4 + 2];
            double theta3 = d_model_params[pid * 4 + 3];
            double x = static_cast<double>(local_idx);
            double predicted;

            switch (model_type) {
                case MODEL_LINEAR:
                    predicted = __dadd_rn(theta0, __dmul_rn(theta1, x));
                    break;
                case MODEL_POLYNOMIAL2:
                    // Horner: a0 + x*(a1 + x*a2)
                    predicted = __dadd_rn(theta0, __dmul_rn(x, __dadd_rn(theta1, __dmul_rn(x, theta2))));
                    break;
                case MODEL_POLYNOMIAL3:
                    // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                    predicted = __dadd_rn(theta0, __dmul_rn(x, __dadd_rn(theta1, __dmul_rn(x, __dadd_rn(theta2, __dmul_rn(x, theta3))))));
                    break;
                default:
                    predicted = __dadd_rn(theta0, __dmul_rn(theta1, x));
                    break;
            }

            // Use __double2ll_rn for signed types, __double2ull_rn for unsigned (banker's rounding)
            T pred_val;
            if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                pred_val = static_cast<T>(__double2ull_rn(predicted));
            } else {
                pred_val = static_cast<T>(__double2ll_rn(predicted));
            }
            int64_t delta = calculateDelta(values[idx], pred_val);

            // Mask to bit width (two's complement preserved)
            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;
        }

        if (delta_bits == 0) continue;

        // BRANCHLESS PACKING
        // Always compute both word indices and shifts
        int word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        int bits_to_second = bit_in_word + delta_bits - 32;

        // First word: always write (bits in first word)
        int bits_in_first = min(delta_bits, 32 - bit_in_word);
        uint32_t mask_first = (bits_in_first == 32) ? ~0U : ((1U << bits_in_first) - 1U);
        uint32_t val_first = static_cast<uint32_t>(packed_value & mask_first) << bit_in_word;
        atomicOr(&delta_array[word_idx], val_first);

        // Second word: branchless - writes zeros if no overflow
        // (bits_to_second > 0) determines if we actually have overflow bits
        uint32_t val_second = static_cast<uint32_t>(packed_value >> bits_in_first);
        uint32_t write_mask = (bits_to_second > 0) ? val_second : 0U;
        atomicOr(&delta_array[word_idx + 1], write_mask);

        // Third word (for > 64 - 32 = 32 bit overflow, i.e., bit_width > 32)
        if (delta_bits > 32 && bit_in_word > 0) {
            int bits_to_third = bits_to_second - 32;
            if (bits_to_third > 0) {
                uint32_t val_third = static_cast<uint32_t>(packed_value >> (bits_in_first + 32));
                atomicOr(&delta_array[word_idx + 2], val_third);
            }
        }
    }
}

// ============================================================================
// Interleaved Packing Kernel
// ============================================================================

/**
 * Convert sequential deltas to interleaved mini-vector format
 *
 * Each block processes one mini-vector (256 values).
 * Each thread handles 8 values in its lane.
 *
 * Input layout (sequential):  v0, v1, v2, ..., v255
 * Output layout (interleaved): Lane0: v0,v32,v64,...  Lane1: v1,v33,v65,...
 */
template<typename T>
__global__ void convertToInterleavedKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,  // nullptr = fixed layout (pid * 4)
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int32_t* __restrict__ d_mv_prefix_sum,  // NEW: prefix sum for binary search
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint32_t* __restrict__ interleaved_array)
{
    // Grid: one block per mini-vector
    // We need to map blockIdx to (partition, mini_vector_within_partition)

    // Use shared memory to find partition for this block
    __shared__ int s_pid;
    __shared__ int s_mv_in_partition;
    __shared__ int s_partition_start;
    __shared__ int s_delta_bits;
    __shared__ double s_params[4];  // All 4 model params
    __shared__ int64_t s_interleaved_base;
    __shared__ int s_model_type;
    __shared__ bool s_valid;  // Flag to indicate if this block has a valid mini-vector


    // Thread 0 finds the partition using binary search (O(log n) instead of O(n))
    if (threadIdx.x == 0) {
        int mv_global = blockIdx.x;
        s_valid = false;  // Initialize to invalid

        // Binary search: find partition p where d_mv_prefix_sum[p] <= mv_global < d_mv_prefix_sum[p+1]
        int left = 0, right = num_partitions - 1;
        int found_p = -1;

        while (left <= right) {
            int mid = (left + right) / 2;
            int prefix_mid = d_mv_prefix_sum[mid];
            int prefix_next = (mid + 1 < num_partitions) ? d_mv_prefix_sum[mid + 1] : (prefix_mid + d_num_mini_vectors[mid]);

            if (mv_global < prefix_mid) {
                right = mid - 1;
            } else if (mv_global >= prefix_next) {
                left = mid + 1;
            } else {
                // Found: prefix_mid <= mv_global < prefix_next
                found_p = mid;
                break;
            }
        }

        if (found_p >= 0) {
            int p = found_p;
            s_pid = p;
            s_mv_in_partition = mv_global - d_mv_prefix_sum[p];
            s_partition_start = d_start_indices[p];
            s_delta_bits = d_delta_bits[p];
            s_model_type = d_model_types[p];

            // Variable-length params: only load what the model needs.
            // Fixed layout: d_param_offsets==nullptr, safe to read pid*4..pid*4+3.
            const int64_t param_base =
                (d_param_offsets == nullptr) ? (static_cast<int64_t>(p) * 4) : d_param_offsets[p];
            s_params[0] = s_params[1] = s_params[2] = s_params[3] = 0.0;
            if (s_model_type == MODEL_FOR_BITPACK) {
                s_params[0] = d_model_params[param_base + 0];
            } else if (s_model_type == MODEL_LINEAR) {
                s_params[0] = d_model_params[param_base + 0];
                s_params[1] = d_model_params[param_base + 1];
            } else if (s_model_type == MODEL_POLYNOMIAL2) {
                s_params[0] = d_model_params[param_base + 0];
                s_params[1] = d_model_params[param_base + 1];
                s_params[2] = d_model_params[param_base + 2];
            } else if (s_model_type == MODEL_POLYNOMIAL3) {
                s_params[0] = d_model_params[param_base + 0];
                s_params[1] = d_model_params[param_base + 1];
                s_params[2] = d_model_params[param_base + 2];
                s_params[3] = d_model_params[param_base + 3];
            } else {
                // Conservative fallback: load first param (covers DIRECT_COPY/rare types)
                // For fixed layout, this still works; for variable layout, unknown types
                // should not appear in the mini-vector path.
                if (d_param_offsets == nullptr) {
                    s_params[0] = d_model_params[param_base + 0];
                    s_params[1] = d_model_params[param_base + 1];
                    s_params[2] = d_model_params[param_base + 2];
                    s_params[3] = d_model_params[param_base + 3];
                }
            }
            s_interleaved_base = d_interleaved_offsets[p];
            s_valid = true;
        }
    }
    __syncthreads();

    // Early exit if this block has no valid mini-vector to process
    if (!s_valid) return;

    int pid = s_pid;
    int mv_idx = s_mv_in_partition;
    int partition_start = s_partition_start;
    int delta_bits = s_delta_bits;
    double params[4];
    params[0] = s_params[0];
    params[1] = s_params[1];
    params[2] = s_params[2];
    params[3] = s_params[3];
    int64_t interleaved_base = s_interleaved_base;
    int model_type = s_model_type;

    // Each thread is a lane (0 to LANES_PER_MINI_VECTOR-1), processes VALUES_PER_THREAD values
    int lane_id = threadIdx.x;
    if (lane_id >= LANES_PER_MINI_VECTOR) return;  // Only need LANES_PER_MINI_VECTOR threads

    // Global indices this lane processes
    int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

    // Calculate bit position for this lane's data
    // Lane L's data in mini-vector starts at: mv_word_base + L * VALUES_PER_THREAD * bit_width bits
    int64_t mv_word_base = interleaved_base + static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits / 32;
    int64_t lane_bit_start = (mv_word_base << 5) + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;

    // =========================================================================
    // FP64 Accumulation for polynomial models
    // CRITICAL: Must match decoder exactly for bit-exact consistency
    // FP64 accumulation eliminates drift error that INT accumulation would cause.
    // =========================================================================

    // Initialize FP64 accumulation state based on model type
    double y_fp = 0.0, d1_fp = 0.0, d2_fp = 0.0, d3_fp = 0.0;
    const int start_idx = mv_idx * MINI_VECTOR_SIZE + lane_id;
    const int stride = LANES_PER_MINI_VECTOR;  // stride = number of lanes

    if (model_type == MODEL_LINEAR) {
        double step_fp;
        FiniteDiff::computeLinearFP64Accum<T>(params, start_idx, stride, y_fp, step_fp);
        d1_fp = step_fp;  // constant step for linear
    } else if (model_type == MODEL_POLYNOMIAL2) {
        FiniteDiff::computePoly2FP64Accum<T>(params, start_idx, stride, y_fp, d1_fp, d2_fp);
    } else if (model_type == MODEL_POLYNOMIAL3) {
        FiniteDiff::computePoly3FP64Accum<T>(params, start_idx, stride, y_fp, d1_fp, d2_fp, d3_fp);
    }

    // Process VALUES_PER_THREAD values for this lane
    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        // Global index: lane_id + v * LANES_PER_MINI_VECTOR within mini-vector
        int local_idx_in_mv = lane_id + v * LANES_PER_MINI_VECTOR;
        int global_idx = mv_start_global + local_idx_in_mv;

        uint64_t packed_value;

        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: delta = val - base (base stored in params[0])
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }
            packed_value = static_cast<uint64_t>(values[global_idx] - base);
        } else if (model_type == MODEL_LINEAR) {
            // LINEAR: FP64 accumulation (matches partitioner/decoder exactly)
            T pred_val = static_cast<T>(__double2ll_rn(y_fp));
            int64_t delta = calculateDelta(values[global_idx], pred_val);
            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;

            y_fp = FiniteDiff::d_add(y_fp, d1_fp);  // FP64 accumulation
        } else if (model_type == MODEL_POLYNOMIAL2) {
            // POLY2: FP64 accumulation (matches partitioner/decoder exactly)
            T pred_val = static_cast<T>(__double2ll_rn(y_fp));
            int64_t delta = calculateDelta(values[global_idx], pred_val);
            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;

            y_fp = FiniteDiff::d_add(y_fp, d1_fp);
            d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);  // FP64 accumulation
        } else if (model_type == MODEL_POLYNOMIAL3) {
            // POLY3: FP64 accumulation (matches partitioner/decoder exactly)
            T pred_val = static_cast<T>(__double2ll_rn(y_fp));
            int64_t delta = calculateDelta(values[global_idx], pred_val);
            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;

            y_fp = FiniteDiff::d_add(y_fp, d1_fp);
            d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
            d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);  // FP64 accumulation
        } else {
            // CONSTANT model: prediction is just params[0]
            T pred_val;
            if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                pred_val = static_cast<T>(__double2ull_rn(params[0]));
            } else {
                pred_val = static_cast<T>(__double2ll_rn(params[0]));
            }
            int64_t delta = calculateDelta(values[global_idx], pred_val);

            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;
        }

        // Pack into interleaved array - handle up to 3 words for delta_bits > 32
        int64_t bit_offset = lane_bit_start + static_cast<int64_t>(v) * delta_bits;
        int word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;

        // First word
        int bits_in_first = min(delta_bits, 32 - bit_in_word);
        uint32_t mask_first = (bits_in_first == 32) ? ~0U : ((1U << bits_in_first) - 1U);
        uint32_t val_first = static_cast<uint32_t>(packed_value & mask_first) << bit_in_word;
        atomicOr(&interleaved_array[word_idx], val_first);

        int bits_remaining = delta_bits - bits_in_first;
        if (bits_remaining > 0) {
            // Second word
            uint64_t shifted = packed_value >> bits_in_first;
            int bits_in_second = min(bits_remaining, 32);
            uint32_t mask_second = (bits_in_second == 32) ? ~0U : ((1U << bits_in_second) - 1U);
            uint32_t val_second = static_cast<uint32_t>(shifted & mask_second);
            atomicOr(&interleaved_array[word_idx + 1], val_second);

            bits_remaining -= bits_in_second;
            if (bits_remaining > 0) {
                // Third word (for delta_bits > 32 crossing 2 word boundaries)
                uint32_t val_third = static_cast<uint32_t>(shifted >> 32);
                atomicOr(&interleaved_array[word_idx + 2], val_third);
            }
        }
    }
}

// ============================================================================
// Tail Values Packing Kernel
// ============================================================================

/**
 * Pack tail values (partition_size % 256) into interleaved array
 *
 * Each block processes one partition's tail data.
 * Tail data is placed immediately after the mini-vectors in the interleaved array.
 */
template<typename T>
__global__ void packTailValuesKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,  // nullptr = fixed layout (pid * 4)
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int32_t* __restrict__ d_tail_sizes,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint32_t* __restrict__ interleaved_array)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int model_type = d_model_types[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    if (interleaved_base < 0) return;

    // ========== CONSTANT encoding ==========
    // Traditional CONSTANT (delta_bits=0): no data to store, all values = params[0]
    // RLE CONSTANT (delta_bits>0): store (value-base, count) pairs
    if (model_type == MODEL_CONSTANT) {
        int count_bits = d_delta_bits[pid];

        // Traditional CONSTANT: no delta data needed
        if (count_bits == 0) {
            return;  // All values are params[0], nothing to encode
        }

        // RLE mode: encode using LINEAR model for values and counts sequences
        // LINEAR params stored in theta array, residuals stored in interleaved
        // theta0 = num_runs, theta1 = value_intercept, theta2 = value_slope, theta3 = count_slope
        // value_residual_bits and count_residual_bits encoded in first word of interleaved
        int start = d_start_indices[pid];
        int end = d_end_indices[pid];
        int psize = end - start;
        if (psize <= 0) return;

        const int64_t param_base =
            (d_param_offsets == nullptr) ? (static_cast<int64_t>(pid) * 4) : d_param_offsets[pid];

        // RLE parameters from model_params (unified CONSTANT/RLE layout)
        int expected_num_runs = static_cast<int>(d_model_params[param_base + 0]);  // num_runs (informational)
        double base_d = d_model_params[param_base + 1];                             // base_value (bit-pattern for 64-bit)

        T base_value;
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(base_d));
        } else {
            base_value = static_cast<T>(__double2ll_rn(base_d));
        }

        int64_t base_bit_offset = interleaved_base << 5;  // words to bits

        // Thread 0 collects runs, fits LINEAR models, and encodes
        if (threadIdx.x == 0) {
            // Collect all runs into local arrays (max 512 runs supported)
            int64_t run_values[512];
            int run_counts[512];
            int run_idx = 0;
            int idx = start;

            while (idx < end && run_idx < 512) {
                T current_val = values[idx];
                int count = 1;
                while (idx + count < end && values[idx + count] == current_val) {
                    count++;
                }
                if constexpr (std::is_signed<T>::value) {
                    run_values[run_idx] = static_cast<int64_t>(current_val) - static_cast<int64_t>(base_value);
                } else {
                    run_values[run_idx] = static_cast<int64_t>(current_val - base_value);
                }
                run_counts[run_idx] = count;
                idx += count;
                run_idx++;
            }
            // Use the actual number of runs we found (should match expected_num_runs)
            int num_runs = run_idx;

            // LINEAR fit for values: y = intercept + slope * x
            double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
            for (int i = 0; i < num_runs; i++) {
                double x = static_cast<double>(i);
                double y = static_cast<double>(run_values[i]);
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            double n_d = static_cast<double>(num_runs);
            double denom = n_d * sum_x2 - sum_x * sum_x;
            double value_slope_d = (denom != 0.0) ? (n_d * sum_xy - sum_x * sum_y) / denom : 0.0;
            double value_intercept_d = (sum_y - value_slope_d * sum_x) / n_d;
            // Convert to float for storage - use same precision as decoder
            float value_intercept_f = static_cast<float>(value_intercept_d);
            float value_slope_f = static_cast<float>(value_slope_d);

            // Compute value residuals using float precision (matches decoder)
            int64_t value_residuals[512];
            int64_t max_value_residual = 0;
            for (int i = 0; i < num_runs; i++) {
                double predicted = static_cast<double>(value_intercept_f) + static_cast<double>(value_slope_f) * static_cast<double>(i);
                int64_t residual = run_values[i] - static_cast<int64_t>(llrint(predicted));
                value_residuals[i] = residual;
                int64_t abs_res = (residual < 0) ? -residual : residual;
                if (abs_res > max_value_residual) max_value_residual = abs_res;
            }
            int value_residual_bits = (max_value_residual == 0) ? 0 : (64 - __clzll(static_cast<unsigned long long>(max_value_residual))) + 1;

            // LINEAR fit for counts
            sum_x = sum_y = sum_xy = sum_x2 = 0;
            for (int i = 0; i < num_runs; i++) {
                double x = static_cast<double>(i);
                double y = static_cast<double>(run_counts[i]);
                sum_x += x;
                sum_y += y;
                sum_xy += x * y;
                sum_x2 += x * x;
            }
            double count_slope_d = (denom != 0.0) ? (n_d * sum_xy - sum_x * sum_y) / denom : 0.0;
            double count_intercept_d = (sum_y - count_slope_d * sum_x) / n_d;
            // Convert to float for storage - use same precision as decoder
            float count_intercept_f = static_cast<float>(count_intercept_d);
            float count_slope_f = static_cast<float>(count_slope_d);

            // Compute count residuals using float precision (matches decoder)
            int64_t count_residuals[512];
            int64_t max_count_residual = 0;
            for (int i = 0; i < num_runs; i++) {
                double predicted = static_cast<double>(count_intercept_f) + static_cast<double>(count_slope_f) * static_cast<double>(i);
                int64_t residual = run_counts[i] - static_cast<int64_t>(llrint(predicted));
                count_residuals[i] = residual;
                int64_t abs_res = (residual < 0) ? -residual : residual;
                if (abs_res > max_count_residual) max_count_residual = abs_res;
            }
            int count_residual_bits = (max_count_residual == 0) ? 0 : (64 - __clzll(static_cast<unsigned long long>(max_count_residual))) + 1;

            // Store header: [value_intercept(f), value_slope(f), count_intercept(f), count_slope(f), bits_info, num_runs]
            // Using float to save space: 4 floats + 2 words = 6 words = 24 bytes
            int header_word_base = interleaved_base;
            float* fptr = reinterpret_cast<float*>(&interleaved_array[header_word_base]);
            fptr[0] = value_intercept_f;
            fptr[1] = value_slope_f;
            fptr[2] = count_intercept_f;
            fptr[3] = count_slope_f;
            // Pack bits info: value_residual_bits(8) | count_residual_bits(8)
            interleaved_array[header_word_base + 4] = (value_residual_bits & 0xFF) | ((count_residual_bits & 0xFF) << 8);
            // Store actual num_runs
            interleaved_array[header_word_base + 5] = static_cast<uint32_t>(num_runs);

            // Data starts after header (6 words = 192 bits)
            int64_t data_bit_base = base_bit_offset + 192;
            int64_t values_bit_base = data_bit_base;
            int64_t counts_bit_base = data_bit_base + static_cast<int64_t>(num_runs) * value_residual_bits;

            // Write value residuals
            for (int i = 0; i < num_runs; i++) {
                if (value_residual_bits == 0) break;
                uint64_t encoded = static_cast<uint64_t>(value_residuals[i]);
                int64_t bit_offset = values_bit_base + static_cast<int64_t>(i) * value_residual_bits;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;
                int bits_remaining = value_residual_bits;
                int bits_written = 0;

                while (bits_remaining > 0) {
                    int bits_this_word = min(bits_remaining, 32 - bit_in_word);
                    uint32_t mask = (bits_this_word == 32) ? ~0U : ((1U << bits_this_word) - 1U);
                    uint32_t val = static_cast<uint32_t>((encoded >> bits_written) & mask) << bit_in_word;
                    atomicOr(&interleaved_array[word_idx], val);
                    bits_written += bits_this_word;
                    bits_remaining -= bits_this_word;
                    word_idx++;
                    bit_in_word = 0;
                }
            }

            // Write count residuals
            for (int i = 0; i < num_runs; i++) {
                if (count_residual_bits == 0) break;
                uint64_t encoded = static_cast<uint64_t>(count_residuals[i]);
                int64_t bit_offset = counts_bit_base + static_cast<int64_t>(i) * count_residual_bits;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;
                int bits_remaining = count_residual_bits;
                int bits_written = 0;

                while (bits_remaining > 0) {
                    int bits_this_word = min(bits_remaining, 32 - bit_in_word);
                    uint32_t mask = (bits_this_word == 32) ? ~0U : ((1U << bits_this_word) - 1U);
                    uint32_t val = static_cast<uint32_t>((encoded >> bits_written) & mask) << bit_in_word;
                    atomicOr(&interleaved_array[word_idx], val);
                    bits_written += bits_this_word;
                    bits_remaining -= bits_this_word;
                    word_idx++;
                    bit_in_word = 0;
                }
            }
        }
        return;  // Done with CONSTANT/RLE partition
    }

    // ========== Non-CONSTANT: tail encoding ==========
    int tail_size = d_tail_sizes[pid];
    if (tail_size == 0) return;  // No tail to pack

    int num_mv = d_num_mini_vectors[pid];
    int delta_bits = d_delta_bits[pid];
    int partition_start = d_start_indices[pid];

    // Load model parameters into registers (variable-length aware)
    double params[4] = {0.0, 0.0, 0.0, 0.0};
    const int64_t param_base =
        (d_param_offsets == nullptr) ? (static_cast<int64_t>(pid) * 4) : d_param_offsets[pid];
    if (model_type == MODEL_FOR_BITPACK) {
        params[0] = d_model_params[param_base + 0];
    } else if (model_type == MODEL_LINEAR) {
        params[0] = d_model_params[param_base + 0];
        params[1] = d_model_params[param_base + 1];
    } else if (model_type == MODEL_POLYNOMIAL2) {
        params[0] = d_model_params[param_base + 0];
        params[1] = d_model_params[param_base + 1];
        params[2] = d_model_params[param_base + 2];
    } else if (model_type == MODEL_POLYNOMIAL3) {
        params[0] = d_model_params[param_base + 0];
        params[1] = d_model_params[param_base + 1];
        params[2] = d_model_params[param_base + 2];
        params[3] = d_model_params[param_base + 3];
    } else {
        // Conservative fallback for fixed layout / rare types
        if (d_param_offsets == nullptr) {
            params[0] = d_model_params[param_base + 0];
            params[1] = d_model_params[param_base + 1];
            params[2] = d_model_params[param_base + 2];
            params[3] = d_model_params[param_base + 3];
        }
    }

    // Tail starts after all mini-vectors
    int tail_start_local = num_mv * MINI_VECTOR_SIZE;
    int tail_start_global = partition_start + tail_start_local;

    // Bit offset where tail data begins in interleaved array
    // Tail is placed after all mini-vector bits
    int64_t mv_bits = static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
    int64_t tail_bit_base = (interleaved_base << 5) + mv_bits;

    // Each thread handles multiple tail values
    for (int i = threadIdx.x; i < tail_size; i += blockDim.x) {
        int global_idx = tail_start_global + i;
        int local_idx_in_partition = tail_start_local + i;

        uint64_t packed_value;

        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: delta = val - base
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }
            packed_value = static_cast<uint64_t>(values[global_idx] - base);
        } else {
            // Compute prediction using polynomial model
            T pred_val = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
            int64_t delta = calculateDelta(values[global_idx], pred_val);

            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;
        }

        if (delta_bits == 0) continue;

        // Pack into interleaved array
        int64_t bit_offset = tail_bit_base + static_cast<int64_t>(i) * delta_bits;
        int word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;

        // First word
        int bits_in_first = min(delta_bits, 32 - bit_in_word);
        uint32_t mask_first = (bits_in_first == 32) ? ~0U : ((1U << bits_in_first) - 1U);
        uint32_t val_first = static_cast<uint32_t>(packed_value & mask_first) << bit_in_word;
        atomicOr(&interleaved_array[word_idx], val_first);

        int bits_remaining = delta_bits - bits_in_first;
        if (bits_remaining > 0) {
            // Second word
            uint64_t shifted = packed_value >> bits_in_first;
            int bits_in_second = min(bits_remaining, 32);
            uint32_t mask_second = (bits_in_second == 32) ? ~0U : ((1U << bits_in_second) - 1U);
            uint32_t val_second = static_cast<uint32_t>(shifted & mask_second);
            atomicOr(&interleaved_array[word_idx + 1], val_second);

            bits_remaining -= bits_in_second;
            if (bits_remaining > 0) {
                // Third word (for delta_bits > 32 crossing 2 word boundaries)
                uint32_t val_third = static_cast<uint32_t>(shifted >> 32);
                atomicOr(&interleaved_array[word_idx + 2], val_third);
            }
        }
    }
}

// ============================================================================
// Host Encoder Functions
// ============================================================================

/**
 * Compute partition metadata (model parameters, delta bits)
 * Reuses logic from original encoder
 */
template<typename T>
void computePartitionMetadata(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions)
{
    for (auto& part : partitions) {
        int start = part.start_idx;
        int end = part.end_idx;
        int n = end - start;

        if (n <= 0) continue;

        // Linear regression
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
        for (int i = 0; i < n; i++) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(data[start + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;

        double theta0, theta1;
        if (std::fabs(det) > 1e-10) {
            theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            theta0 = (sum_y - theta1 * sum_x) / dn;
        } else {
            theta1 = 0.0;
            theta0 = sum_y / dn;
        }

        part.model_type = MODEL_LINEAR;
        part.model_params[0] = theta0;
        part.model_params[1] = theta1;
        part.model_params[2] = 0.0;
        part.model_params[3] = 0.0;

        // Compute max error
        // Use llrint for consistency with decoder's __double2ll_rn (banker's rounding)
        long long max_error = 0;
        for (int i = 0; i < n; i++) {
            double predicted = theta0 + theta1 * i;
            T pred_val = static_cast<T>(std::llrint(predicted));
            long long delta;
            if (data[start + i] >= pred_val) {
                delta = static_cast<long long>(data[start + i] - pred_val);
            } else {
                delta = -static_cast<long long>(pred_val - data[start + i]);
            }
            long long abs_delta = (delta < 0) ? -delta : delta;
            max_error = (abs_delta > max_error) ? abs_delta : max_error;
        }

        part.error_bound = max_error;
        int bits = 0;
        if (max_error > 0) {
            bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
        }
        part.delta_bits = bits;
    }
}

/**
 * Compute partition metadata using adaptive model selection
 *
 * Automatically chooses between LINEAR and FOR+BitPacking based on cost analysis.
 * Uses the adaptive_selector kernel for GPU-accelerated decision making.
 */
template<typename T>
void computePartitionMetadataAdaptive(
    const T* d_data,
    std::vector<PartitionInfo>& partitions,
    cudaStream_t stream = 0)
{
    int num_partitions = partitions.size();
    if (num_partitions == 0) return;

    // Allocate device memory for indices and decisions
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    adaptive_selector::ModelDecision<T>* d_decisions;

    cudaMalloc(&d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&d_end_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

    // Copy indices to device
    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        h_start[i] = partitions[i].start_idx;
        h_end[i] = partitions[i].end_idx;
    }

    cudaMemcpyAsync(d_start_indices, h_start.data(),
                    num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_end_indices, h_end.data(),
                    num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // Launch adaptive selector kernel
    adaptive_selector::launchAdaptiveSelector<T>(
        d_data, d_start_indices, d_end_indices, num_partitions, d_decisions, stream);

    // Copy decisions back to host
    std::vector<adaptive_selector::ModelDecision<T>> h_decisions(num_partitions);
    cudaMemcpyAsync(h_decisions.data(), d_decisions,
                    num_partitions * sizeof(adaptive_selector::ModelDecision<T>),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Update partitions with decisions
    for (int i = 0; i < num_partitions; i++) {
        const auto& decision = h_decisions[i];

        partitions[i].model_type = decision.model_type;
        partitions[i].delta_bits = decision.delta_bits;

        // Copy all 4 model parameters (works for all model types)
        partitions[i].model_params[0] = decision.params[0];
        partitions[i].model_params[1] = decision.params[1];
        partitions[i].model_params[2] = decision.params[2];
        partitions[i].model_params[3] = decision.params[3];

        // Store error bound (range)
        partitions[i].error_bound = decision.max_val - decision.min_val;
    }

    // Cleanup
    cudaFree(d_start_indices);
    cudaFree(d_end_indices);
    cudaFree(d_decisions);
}

/**
 * CPU version of adaptive partition metadata computation
 *
 * Supports all polynomial models (LINEAR, POLYNOMIAL2, POLYNOMIAL3) and FOR+BitPacking.
 * Model parameters are stored as:
 *   - LINEAR:    params[0]=θ₀ (intercept), params[1]=θ₁ (slope)
 *   - POLY2:     params[0]=θ₀, params[1]=θ₁, params[2]=θ₂
 *   - POLY3:     params[0]=θ₀, params[1]=θ₁, params[2]=θ₂, params[3]=θ₃
 *   - FOR+BP:    params[0]=base (min value)
 */
template<typename T>
void computePartitionMetadataAdaptiveCPU(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions)
{
    for (auto& part : partitions) {
        auto decision = adaptive_selector::computeDecisionCPU<T>(
            data.data(), part.start_idx, part.end_idx);

        part.model_type = decision.model_type;
        part.delta_bits = decision.delta_bits;

        // Copy all 4 model parameters (works for all model types)
        part.model_params[0] = decision.params[0];
        part.model_params[1] = decision.params[1];
        part.model_params[2] = decision.params[2];
        part.model_params[3] = decision.params[3];

        part.error_bound = decision.max_val - decision.min_val;
    }
}

/**
 * CPU version of fixed model partition metadata computation
 *
 * Forces a specific model type for all partitions.
 */
template<typename T>
void computePartitionMetadataFixedCPU(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions,
    int fixed_model_type)
{
    for (auto& part : partitions) {
        auto decision = adaptive_selector::computeFixedModelCPU<T>(
            data.data(), part.start_idx, part.end_idx, fixed_model_type);

        part.model_type = decision.model_type;
        part.delta_bits = decision.delta_bits;

        // Copy all 4 model parameters
        part.model_params[0] = decision.params[0];
        part.model_params[1] = decision.params[1];
        part.model_params[2] = decision.params[2];
        part.model_params[3] = decision.params[3];

        part.error_bound = decision.max_val - decision.min_val;
    }
}

/**
 * Compute interleaved metadata for partitions
 * Now ALL partitions use interleaved format (no sequential-only fallback)
 */
void computeInterleavedMetadata(
    std::vector<PartitionInfo>& partitions,
    std::vector<int32_t>& num_mini_vectors,
    std::vector<int32_t>& tail_sizes,
    std::vector<int64_t>& interleaved_offsets,
    const VerticalConfig& config)
{
    num_mini_vectors.resize(partitions.size());
    tail_sizes.resize(partitions.size());
    interleaved_offsets.resize(partitions.size());

    int64_t current_word_offset = 0;

    for (size_t p = 0; p < partitions.size(); p++) {
        int partition_size = partitions[p].end_idx - partitions[p].start_idx;
        int bit_width = partitions[p].delta_bits;
        int model_type = partitions[p].model_type;

        interleaved_offsets[p] = current_word_offset;

        // Check if this is a CONSTANT/RLE partition
        if (model_type == MODEL_CONSTANT) {
            int num_runs = static_cast<int>(partitions[p].model_params[0]);

            if (num_runs == 1) {
                // Traditional CONSTANT: no interleaved data needed
                num_mini_vectors[p] = 0;
                tail_sizes[p] = 0;
                // No space needed
            } else {
                // RLE CONSTANT: needs header + residuals
                // Header: 6 words (24 bytes) for LINEAR params
                // Residuals: num_runs * (value_residual_bits + count_residual_bits) bits
                // Conservative estimate: use value_bits and count_bits from model_params
                int value_bits = static_cast<int>(partitions[p].model_params[2]);
                int count_bits = static_cast<int>(partitions[p].model_params[3]);
                // Residual bits could be larger due to float precision, add safety margin
                int max_residual_bits = std::max(value_bits + 2, count_bits + 2);
                int64_t residual_bits = static_cast<int64_t>(num_runs) * 2 * max_residual_bits;
                int64_t total_bits = 192 + residual_bits;  // 6 words = 192 bits header
                int64_t words = (total_bits + 31) / 32;

                num_mini_vectors[p] = 0;
                tail_sizes[p] = 0;
                current_word_offset += words;
            }
        } else {
            // Non-CONSTANT: use mini-vector + tail format
            num_mini_vectors[p] = partition_size / MINI_VECTOR_SIZE;
            tail_sizes[p] = partition_size % MINI_VECTOR_SIZE;

            // Calculate words for this partition's interleaved data (mini-vectors + tail)
            int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[p]) * MINI_VECTOR_SIZE * bit_width;
            int64_t tail_bits = static_cast<int64_t>(tail_sizes[p]) * bit_width;
            int64_t total_bits = mv_bits + tail_bits;
            int64_t words = (total_bits + 31) / 32;

            current_word_offset += words;
        }
    }
}

/**
 * Main encoder function - creates Vertical-optimized compressed data
 */
template<typename T>
CompressedDataVertical<T> encodeVertical(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataVertical<T> result;
    result.num_partitions = partitions.size();
    result.total_values = data.size();

    if (partitions.empty() || data.empty()) {
        return result;
    }

    // ========== Step 1: Compute partition metadata ==========
    if (!config.skip_metadata_recompute) {
        // Normal behavior: compute model parameters
        if (config.enable_adaptive_selection) {
            // Use adaptive model selection (all models: LINEAR/POLY2/POLY3/FOR)
            computePartitionMetadataAdaptiveCPU(data, partitions);
        } else {
            // Use fixed model type specified in config
            computePartitionMetadataFixedCPU(data, partitions, config.fixed_model_type);
        }
    }
    // If skip_metadata_recompute=true, use the values already in partitions

    // ========== Step 2: Compute interleaved metadata ==========
    std::vector<int32_t> num_mini_vectors, tail_sizes;
    std::vector<int64_t> interleaved_offsets;
    computeInterleavedMetadata(partitions, num_mini_vectors, tail_sizes,
                               interleaved_offsets, config);

    // Calculate total interleaved words (now stores ALL data - mini-vectors + tails + RLE data)
    int64_t max_interleaved_offset = 0;
    int total_mini_vectors = 0;
    for (size_t p = 0; p < partitions.size(); p++) {
        int model_type = partitions[p].model_type;
        int64_t end_word;

        if (model_type == MODEL_CONSTANT) {
            int num_runs = static_cast<int>(partitions[p].model_params[0]);
            if (num_runs == 1) {
                // Traditional CONSTANT: no data
                end_word = interleaved_offsets[p];
            } else {
                // RLE: calculate same way as computeInterleavedMetadata
                int value_bits = static_cast<int>(partitions[p].model_params[2]);
                int count_bits = static_cast<int>(partitions[p].model_params[3]);
                int max_residual_bits = std::max(value_bits + 2, count_bits + 2);
                int64_t residual_bits = static_cast<int64_t>(num_runs) * 2 * max_residual_bits;
                int64_t total_bits = 192 + residual_bits;
                int64_t words = (total_bits + 31) / 32;
                end_word = interleaved_offsets[p] + words;
            }
        } else {
            // Non-CONSTANT: mini-vectors + tail
            int bit_width = partitions[p].delta_bits;
            int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[p]) * MINI_VECTOR_SIZE * bit_width;
            int64_t tail_bits = static_cast<int64_t>(tail_sizes[p]) * bit_width;
            end_word = interleaved_offsets[p] + (mv_bits + tail_bits + 31) / 32;
        }
        max_interleaved_offset = std::max(max_interleaved_offset, end_word);
        total_mini_vectors += num_mini_vectors[p];
        result.total_interleaved_partitions++;
    }
    result.interleaved_delta_words = max_interleaved_offset + 4;

    // ========== Step 3: Allocate device memory ==========
    int np = result.num_partitions;

    // Partition metadata arrays
    cudaMalloc(&result.d_start_indices, np * sizeof(int32_t));
    cudaMalloc(&result.d_end_indices, np * sizeof(int32_t));
    cudaMalloc(&result.d_model_types, np * sizeof(int32_t));
    cudaMalloc(&result.d_delta_bits, np * sizeof(int32_t));
    cudaMalloc(&result.d_delta_array_bit_offsets, np * sizeof(int64_t));
    cudaMalloc(&result.d_error_bounds, np * sizeof(int64_t));

    // Variable parameter storage: compute offsets and allocate
    std::vector<int64_t> h_param_offsets(np);
    int64_t running_offset = 0;
    for (int p = 0; p < np; p++) {
        h_param_offsets[p] = running_offset;
        running_offset += getParamCount(partitions[p].model_type, partitions[p].delta_bits);
    }
    result.total_param_count = running_offset;
    result.use_variable_params = true;

    // Allocate variable-length params array and offsets
    if (result.total_param_count > 0) {
        cudaMalloc(&result.d_model_params, result.total_param_count * sizeof(double));
    } else {
        result.d_model_params = nullptr;
    }
    cudaMalloc(&result.d_param_offsets, np * sizeof(int64_t));

    // Interleaved metadata and delta data (stores ALL data - mini-vectors + tails)
    cudaMalloc(&result.d_num_mini_vectors, np * sizeof(int32_t));
    cudaMalloc(&result.d_tail_sizes, np * sizeof(int32_t));
    cudaMalloc(&result.d_interleaved_offsets, np * sizeof(int64_t));

    if (result.interleaved_delta_words > 0) {
        cudaMalloc(&result.d_interleaved_deltas,
                  result.interleaved_delta_words * sizeof(uint32_t));
        cudaMemsetAsync(result.d_interleaved_deltas, 0,
                       result.interleaved_delta_words * sizeof(uint32_t), stream);
    }

    // ========== Step 4: Copy metadata to device ==========
    std::vector<int32_t> h_start_indices(np), h_end_indices(np), h_model_types(np), h_delta_bits(np);
    std::vector<double> h_model_params(result.total_param_count);  // Variable-length
    std::vector<int64_t> h_error_bounds(np);

    for (int p = 0; p < np; p++) {
        h_start_indices[p] = partitions[p].start_idx;
        h_end_indices[p] = partitions[p].end_idx;
        h_model_types[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_error_bounds[p] = partitions[p].error_bound;

        // Pack variable-length parameters
        int64_t offset = h_param_offsets[p];
        int param_count = getParamCount(partitions[p].model_type, partitions[p].delta_bits);
        for (int j = 0; j < param_count; j++) {
            h_model_params[offset + j] = partitions[p].model_params[j];
        }
    }

    cudaMemcpyAsync(result.d_start_indices, h_start_indices.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_end_indices, h_end_indices.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_model_types, h_model_types.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    if (result.total_param_count > 0) {
        cudaMemcpyAsync(result.d_model_params, h_model_params.data(),
                        result.total_param_count * sizeof(double), cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(result.d_param_offsets, h_param_offsets.data(),
                    np * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_delta_bits, h_delta_bits.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_error_bounds, h_error_bounds.data(),
                    np * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Interleaved metadata
    cudaMemcpyAsync(result.d_num_mini_vectors, num_mini_vectors.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_tail_sizes, tail_sizes.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_interleaved_offsets, interleaved_offsets.data(),
                    np * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    // Compute mini-vector prefix sum on host and copy to device
    std::vector<int32_t> mv_prefix_sum(np);
    mv_prefix_sum[0] = 0;
    for (int p = 1; p < np; p++) {
        mv_prefix_sum[p] = mv_prefix_sum[p-1] + num_mini_vectors[p-1];
    }
    int32_t* d_mv_prefix_sum;
    cudaMalloc(&d_mv_prefix_sum, np * sizeof(int32_t));
    cudaMemcpyAsync(d_mv_prefix_sum, mv_prefix_sum.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // ========== Step 5: Copy data to device ==========
    T* d_values;
    cudaMalloc(&d_values, data.size() * sizeof(T));
    cudaMemcpyAsync(d_values, data.data(), data.size() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);

    // ========== Step 6: Pack interleaved deltas (mini-vectors) ==========
    if (total_mini_vectors > 0) {
        // One block per mini-vector, LANES_PER_MINI_VECTOR threads per block (one per lane)
	        convertToInterleavedKernel<T><<<total_mini_vectors, LANES_PER_MINI_VECTOR, 0, stream>>>(
	            d_values,
	            result.d_start_indices,
	            result.d_end_indices,
	            result.d_model_types,
	            result.d_model_params,
	            result.d_param_offsets,
	            result.d_delta_bits,
	            result.d_num_mini_vectors,
	            d_mv_prefix_sum,  // NEW: prefix sum for binary search
	            result.d_interleaved_offsets,
	            np,
            result.d_interleaved_deltas
        );
    }

    cudaFree(d_mv_prefix_sum);  // Free after kernel completes

    // ========== Step 7: Pack tail values ==========
    // One block per partition, BLOCK_SIZE threads per block
	    packTailValuesKernel<T><<<np, BLOCK_SIZE, 0, stream>>>(
	        d_values,
	        result.d_start_indices,
	        result.d_end_indices,
	        result.d_model_types,
	        result.d_model_params,
	        result.d_param_offsets,
	        result.d_delta_bits,
	        result.d_num_mini_vectors,
	        result.d_tail_sizes,
	        result.d_interleaved_offsets,
	        np,
        result.d_interleaved_deltas
    );

    cudaStreamSynchronize(stream);
    cudaFree(d_values);

    return result;
}

/**
 * Simple partitioning (fixed size) for testing
 */
template<typename T>
std::vector<PartitionInfo> createFixedPartitions(
    int data_size,
    int partition_size)
{
    std::vector<PartitionInfo> partitions;
    int num_parts = (data_size + partition_size - 1) / partition_size;

    for (int p = 0; p < num_parts; p++) {
        PartitionInfo info;
        info.start_idx = p * partition_size;
        info.end_idx = std::min((p + 1) * partition_size, data_size);
        info.model_type = MODEL_LINEAR;
        info.delta_bits = 0;
        info.delta_array_bit_offset = 0;
        info.error_bound = 0;
        for (int j = 0; j < 4; j++) info.model_params[j] = 0.0;
        partitions.push_back(info);
    }

    return partitions;
}

/**
 * Free compressed data
 */
template<typename T>
void freeCompressedData(CompressedDataVertical<T>& data)
{
    if (data.d_start_indices) cudaFree(data.d_start_indices);
    if (data.d_end_indices) cudaFree(data.d_end_indices);
    if (data.d_model_types) cudaFree(data.d_model_types);
    if (data.d_model_params) cudaFree(data.d_model_params);
    if (data.d_param_offsets) cudaFree(data.d_param_offsets);
    if (data.d_delta_bits) cudaFree(data.d_delta_bits);
    if (data.d_delta_array_bit_offsets) cudaFree(data.d_delta_array_bit_offsets);
    if (data.d_error_bounds) cudaFree(data.d_error_bounds);
    if (data.d_sequential_deltas) cudaFree(data.d_sequential_deltas);
    if (data.d_partition_min_values) cudaFree(data.d_partition_min_values);
    if (data.d_partition_max_values) cudaFree(data.d_partition_max_values);
    if (data.d_interleaved_deltas) cudaFree(data.d_interleaved_deltas);
    if (data.d_num_mini_vectors) cudaFree(data.d_num_mini_vectors);
    if (data.d_tail_sizes) cudaFree(data.d_tail_sizes);
    if (data.d_interleaved_offsets) cudaFree(data.d_interleaved_offsets);
    if (data.d_has_interleaved) cudaFree(data.d_has_interleaved);
    if (data.d_self) cudaFree(data.d_self);
    // V5 metadata
    if (data.d_metadata_v5) cudaFree(data.d_metadata_v5);

    data = CompressedDataVertical<T>();  // Reset to empty
}

// ============================================================================
// Full GPU Pipeline Encoder - GPU Kernels for Metadata Computation
// ============================================================================

/**
 * Kernel: Generate fixed-size partition indices directly on GPU
 * Eliminates H→D transfer for partition indices
 */
__global__ void generateFixedPartitionsKernel(
    int32_t* __restrict__ start_indices,
    int32_t* __restrict__ end_indices,
    int partition_size,
    int total_elements,
    int num_partitions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_partitions) return;

    start_indices[i] = i * partition_size;
    end_indices[i] = min((i + 1) * partition_size, total_elements);
}

/**
 * Kernel: Unpack ModelDecision structures into separate metadata arrays
 * Eliminates D→H transfer and CPU unpacking loop
 */
template<typename T>
__global__ void unpackDecisionsKernel(
    const adaptive_selector::ModelDecision<T>* __restrict__ decisions,
    int32_t* __restrict__ model_types,
    double* __restrict__ model_params,      // size: num_partitions * 4
    int32_t* __restrict__ delta_bits,
    int64_t* __restrict__ error_bounds,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    const auto& dec = decisions[pid];
    model_types[pid] = dec.model_type;
    delta_bits[pid] = dec.delta_bits;
    error_bounds[pid] = dec.max_val - dec.min_val;

    // Copy all 4 model parameters
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        model_params[pid * 4 + j] = dec.params[j];
    }
}

/**
 * Kernel: Compute bit counts for each partition (partition_size * delta_bits)
 * Used for parallel prefix sum to compute bit offsets
 */
__global__ void computePartitionBitCountsKernel(
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    const int32_t* __restrict__ delta_bits,
    int64_t* __restrict__ bit_counts,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    int psize = end_indices[pid] - start_indices[pid];
    bit_counts[pid] = static_cast<int64_t>(psize) * delta_bits[pid];
}

/**
 * Kernel: Compute interleaved metadata for each partition
 * Now ALL partitions use interleaved format
 * - num_mini_vectors: number of complete 256-value blocks
 * - tail_sizes: remaining values after mini-vectors
 * - word_counts: number of words for this partition's interleaved data
 */
__global__ void computeInterleavedMetadataKernel(
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    const int32_t* __restrict__ model_types,
    const int32_t* __restrict__ delta_bits,
    const double* __restrict__ model_params,
    int32_t* __restrict__ num_mini_vectors,
    int32_t* __restrict__ tail_sizes,
    int64_t* __restrict__ word_counts,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    int psize = end_indices[pid] - start_indices[pid];
    int bit_width = delta_bits[pid];
    int model_type = model_types[pid];

    // Check if this is a CONSTANT partition
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(model_params[pid * 4]);  // theta0 = num_runs

        if (num_runs == 1) {
            // Traditional CONSTANT: no data needed
            num_mini_vectors[pid] = 0;
            tail_sizes[pid] = 0;
            word_counts[pid] = 0;
        } else {
            // RLE partition with LINEAR model: 6-word header + residuals
            int value_bits = static_cast<int>(model_params[pid * 4 + 2]);  // theta2 = value_bits
            int count_bits = static_cast<int>(model_params[pid * 4 + 3]);  // theta3 = count_bits
            // Residual bits could be larger due to float precision, add safety margin
            int max_residual_bits = max(value_bits + 2, count_bits + 2);
            int64_t residual_bits = static_cast<int64_t>(num_runs) * 2 * max_residual_bits;
            int64_t total_bits = 192 + residual_bits;  // 6 words header = 192 bits

            num_mini_vectors[pid] = 0;
            tail_sizes[pid] = 0;
            word_counts[pid] = (total_bits + 31) / 32;
        }
    } else {
        // Non-CONSTANT: use interleaved format (mini-vectors + tail)
        num_mini_vectors[pid] = psize / MINI_VECTOR_SIZE;
        tail_sizes[pid] = psize % MINI_VECTOR_SIZE;

        int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[pid]) * MINI_VECTOR_SIZE * bit_width;
        int64_t tail_bits = static_cast<int64_t>(tail_sizes[pid]) * bit_width;
        word_counts[pid] = (mv_bits + tail_bits + 31) / 32;
    }
}

/**
 * Kernel: Set interleaved offsets from prefix sum results
 * Now all partitions use interleaved format
 */
__global__ void setInterleavedOffsetsKernel(
    const int64_t* __restrict__ prefix_sums,
    int64_t* __restrict__ interleaved_offsets,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    interleaved_offsets[pid] = prefix_sums[pid];
}

/**
 * Kernel: Count total mini-vectors across all partitions
 * Uses parallel reduction
 */
__global__ void countTotalMiniVectorsKernel(
    const int32_t* __restrict__ num_mini_vectors,
    int num_partitions,
    int* __restrict__ total_count)
{
    __shared__ int shared_sum[256];

    int tid = threadIdx.x;
    int local_sum = 0;

    for (int i = tid; i < num_partitions; i += blockDim.x) {
        local_sum += num_mini_vectors[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total_count = shared_sum[0];
    }
}

/**
 * Full GPU Pipeline Encoder
 *
 * Performs entire compression pipeline on GPU with minimal CPU involvement.
 * Only uses interleaved format (no redundant sequential storage).
 *
 * Steps:
 * 1. Data upload (unavoidable)
 * 2. Partition index generation (GPU)
 * 3. Model selection (GPU)
 * 4. Decision unpacking (GPU)
 * 5. Interleaved metadata computation (GPU)
 * 6. Mini-vector delta packing (GPU)
 * 7. Tail delta packing (GPU)
 */
/**
 * Helper: Convert VerticalConfig to CostOptimalConfig
 */
inline CostOptimalConfig VerticalToCostOptimalConfig(const VerticalConfig& fl_config, int partition_size) {
    CostOptimalConfig cost_config;
    cost_config.analysis_block_size = fl_config.cost_analysis_block_size;
    cost_config.min_partition_size = fl_config.cost_min_partition_size;
    cost_config.max_partition_size = fl_config.cost_max_partition_size;
    cost_config.breakpoint_threshold = fl_config.cost_breakpoint_threshold;
    cost_config.merge_benefit_threshold = fl_config.cost_merge_benefit_threshold;
    cost_config.max_merge_rounds = fl_config.cost_max_merge_rounds;
    cost_config.enable_merging = fl_config.cost_enable_merging;
    cost_config.enable_rle = fl_config.enable_rle;
    // NOTE: enable_polynomial_models controls POLY2/POLY3 model selection.
    // Now respects user's enable_adaptive_selection flag (consistent with Transposed encoder).
    cost_config.enable_polynomial_models = fl_config.enable_adaptive_selection;
    return cost_config;
}

/**
 * Helper: Copy PartitionInfo vector to device arrays
 */
template<typename T>
void copyPartitionsToDevice(
    const std::vector<PartitionInfo>& partitions,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    cudaStream_t stream)
{
    int np = partitions.size();
    std::vector<int32_t> h_start(np), h_end(np), h_model_types(np), h_delta_bits(np);
    std::vector<double> h_model_params(np * 4);
    std::vector<int64_t> h_error_bounds(np);

    for (int i = 0; i < np; i++) {
        h_start[i] = partitions[i].start_idx;
        h_end[i] = partitions[i].end_idx;
        h_model_types[i] = partitions[i].model_type;
        h_delta_bits[i] = partitions[i].delta_bits;
        h_error_bounds[i] = partitions[i].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[i * 4 + j] = partitions[i].model_params[j];
        }
    }

    cudaMemcpyAsync(d_start_indices, h_start.data(), np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_end_indices, h_end.data(), np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_model_types, h_model_types.data(), np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_model_params, h_model_params.data(), np * 4 * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_delta_bits, h_delta_bits.data(), np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_error_bounds, h_error_bounds.data(), np * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
}

/**
 * GPU Kernel: Merge separate theta arrays into interleaved model_params array
 * Input: theta0[np], theta1[np], theta2[np], theta3[np]
 * Output: model_params[np*4] where model_params[i*4+j] = thetaJ[i]
 */
__global__ void mergeModelParamsKernel(
    const double* __restrict__ theta0,
    const double* __restrict__ theta1,
    const double* __restrict__ theta2,
    const double* __restrict__ theta3,
    double* __restrict__ model_params,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    model_params[pid * 4 + 0] = theta0[pid];
    model_params[pid * 4 + 1] = theta1[pid];
    model_params[pid * 4 + 2] = theta2[pid];
    model_params[pid * 4 + 3] = theta3[pid];
}

/**
 * Helper: Copy GPU partition result to device arrays (GPU-to-GPU, no CPU roundtrip!)
 */
template<typename T, typename GPUPartitionResultT>
void copyGPUPartitionResultToDevice(
    const GPUPartitionResultT& gpu_result,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    cudaStream_t stream)
{
    int np = gpu_result.num_partitions;

    // Device-to-device copies (fast!)
    cudaMemcpyAsync(d_start_indices, gpu_result.d_starts, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_end_indices, gpu_result.d_ends, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_model_types, gpu_result.d_model_types, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_delta_bits, gpu_result.d_delta_bits, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_error_bounds, gpu_result.d_max_errors, np * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);

    // Merge theta0/1/2/3 into model_params using GPU kernel
    int threads = 256;
    int blocks = (np + threads - 1) / threads;
    mergeModelParamsKernel<<<blocks, threads, 0, stream>>>(
        gpu_result.d_theta0,
        gpu_result.d_theta1,
        gpu_result.d_theta2,
        gpu_result.d_theta3,
        d_model_params,
        np);
}

// ============================================================================
// Variable Parameter Storage GPU Kernels
// ============================================================================

/**
 * Kernel: Compute parameter count for each partition based on model type and delta_bits.
 * Used for variable-length parameter storage.
 */
__global__ void computeParamCountsKernel(
    const int32_t* __restrict__ d_model_types,
    const int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_param_counts,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;
    d_param_counts[pid] = getParamCount(d_model_types[pid], d_delta_bits[pid]);
}

/**
 * Kernel: Pack parameters from fixed layout to variable-length layout.
 * Reads from fixed_params[pid * 4 + j] and writes to variable_params[offset + j].
 */
__global__ void packVariableParamsFromFixedKernel(
    const double* __restrict__ fixed_params,
    const int32_t* __restrict__ d_model_types,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_param_offsets,
    double* __restrict__ variable_params,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    int64_t offset = d_param_offsets[pid];
    int param_count = getParamCount(d_model_types[pid], d_delta_bits[pid]);

    if (param_count >= 1) variable_params[offset + 0] = fixed_params[pid * 4 + 0];
    if (param_count >= 2) variable_params[offset + 1] = fixed_params[pid * 4 + 1];
    if (param_count >= 3) variable_params[offset + 2] = fixed_params[pid * 4 + 2];
    if (param_count >= 4) variable_params[offset + 3] = fixed_params[pid * 4 + 3];
}

/**
 * Helper: Convert fixed-layout model_params to variable-length layout.
 * Call this after encoding is complete to reduce metadata storage.
 *
 * @param d_fixed_params     Input: fixed layout [num_partitions * 4]
 * @param d_model_types      Model types for each partition
 * @param d_delta_bits       Delta bits for each partition
 * @param num_partitions     Number of partitions
 * @param result             CompressedDataVertical to update
 * @param stream             CUDA stream
 */
template<typename T>
void convertToVariableParams(
    double* d_fixed_params,
    const int32_t* d_model_types,
    const int32_t* d_delta_bits,
    int num_partitions,
    CompressedDataVertical<T>& result,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (num_partitions + threads - 1) / threads;

    // 1. Allocate temporary param_counts array
    int64_t* d_param_counts;
    cudaMalloc(&d_param_counts, num_partitions * sizeof(int64_t));

    // 2. Compute parameter count for each partition
    computeParamCountsKernel<<<blocks, threads, 0, stream>>>(
        d_model_types,
        d_delta_bits,
        d_param_counts,
        num_partitions);

    // 3. Allocate param_offsets and compute prefix sum
    cudaMalloc(&result.d_param_offsets, num_partitions * sizeof(int64_t));

    thrust::device_ptr<int64_t> counts_ptr(d_param_counts);
    thrust::device_ptr<int64_t> offsets_ptr(result.d_param_offsets);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        counts_ptr, counts_ptr + num_partitions,
        offsets_ptr);

    // 4. Get total parameter count
    int64_t h_last_count, h_last_offset;
    cudaMemcpyAsync(&h_last_count, d_param_counts + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_offset, result.d_param_offsets + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    result.total_param_count = h_last_offset + h_last_count;

    // 5. Allocate and pack variable-length parameters
    if (result.total_param_count > 0) {
        double* d_variable_params;
        cudaMalloc(&d_variable_params, result.total_param_count * sizeof(double));

        packVariableParamsFromFixedKernel<<<blocks, threads, 0, stream>>>(
            d_fixed_params,
            d_model_types,
            d_delta_bits,
            result.d_param_offsets,
            d_variable_params,
            num_partitions);

        // Replace fixed params with variable params
        cudaFree(d_fixed_params);
        result.d_model_params = d_variable_params;
    } else {
        // No parameters needed (all RLE partitions)
        cudaFree(d_fixed_params);
        result.d_model_params = nullptr;
    }

    result.use_variable_params = true;
    cudaFree(d_param_counts);
}

/**
 * Kernel to compute partition min/max values for predicate pushdown
 * One block per partition, parallel reduction within each block
 */
template<typename T>
__global__ void computePartitionMinMaxKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    int num_partitions,
    T* __restrict__ d_partition_min,
    T* __restrict__ d_partition_max)
{
    extern __shared__ char shared_mem[];
    T* s_min = reinterpret_cast<T*>(shared_mem);
    T* s_max = reinterpret_cast<T*>(shared_mem + blockDim.x * sizeof(T));

    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = d_start_indices[pid];
    int end = d_end_indices[pid];
    int size = end - start;

    if (size == 0) {
        if (threadIdx.x == 0) {
            d_partition_min[pid] = T(0);
            d_partition_max[pid] = T(0);
        }
        return;
    }

    // Each thread scans its portion of the data
    T local_min = values[start];
    T local_max = values[start];

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        T val = values[start + i];
        local_min = (val < local_min) ? val : local_min;
        local_max = (val > local_max) ? val : local_max;
    }

    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    // Parallel reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            T other_min = s_min[threadIdx.x + stride];
            T other_max = s_max[threadIdx.x + stride];
            s_min[threadIdx.x] = (other_min < s_min[threadIdx.x]) ? other_min : s_min[threadIdx.x];
            s_max[threadIdx.x] = (other_max > s_max[threadIdx.x]) ? other_max : s_max[threadIdx.x];
        }
        __syncthreads();
    }

    // Write result
    if (threadIdx.x == 0) {
        d_partition_min[pid] = s_min[0];
        d_partition_max[pid] = s_max[0];
    }
}

/**
 * Kernel to generate V5 consolidated metadata
 *
 * Combines all separate metadata arrays into a single PartitionMetadataV5 array.
 * This reduces L1 cache misses during decompression from 6+ to 1-2.
 *
 * One thread per partition.
 */
__global__ void generateV5MetadataKernel(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double*  __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int32_t* __restrict__ d_tail_sizes,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    PartitionMetadataV5* __restrict__ d_metadata_v5)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    // Load from separate arrays
    PartitionMetadataV5 meta;
    meta.start_idx = d_start_indices[pid];
    meta.end_idx = d_end_indices[pid];
    meta.model_type = d_model_types[pid];
    meta.delta_bits = d_delta_bits[pid];
    meta.num_mini_vectors = d_num_mini_vectors[pid];
    meta.tail_size = d_tail_sizes[pid];
    meta.interleaved_offset = d_interleaved_offsets[pid];

    // Copy model parameters (always load 4 for simplicity)
    meta.params[0] = d_model_params[pid * 4 + 0];
    meta.params[1] = d_model_params[pid * 4 + 1];
    meta.params[2] = d_model_params[pid * 4 + 2];
    meta.params[3] = d_model_params[pid * 4 + 3];

    // Write consolidated structure
    d_metadata_v5[pid] = meta;
}

template<typename T, typename PartitionerT>
CompressedDataVertical<T> encodeVerticalGPU_Internal(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataVertical<T> result;

    // For COST_OPTIMAL, partition_size is ignored (uses config.cost_min/max_partition_size)
    bool use_cost_optimal = (config.partitioning_strategy == PartitioningStrategy::COST_OPTIMAL);
    if (data.empty() || (!use_cost_optimal && partition_size <= 0)) {
        return result;
    }

    size_t n = data.size();
    int num_partitions;

    // ========== Step 1: Upload data to GPU ==========
    T* d_data;
    cudaMalloc(&d_data, n * sizeof(T));
    cudaMemcpyAsync(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream);

    // ========== CUDA Events for kernel timing (includes partitioning + encoding) ==========
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    // Temporary arrays for partition indices (used differently based on strategy)
    int32_t* d_start_indices_temp = nullptr;
    int32_t* d_end_indices_temp = nullptr;
    adaptive_selector::ModelDecision<T>* d_decisions = nullptr;

    // Storage for cost-optimal partitions
    std::vector<PartitionInfo> cost_optimal_partitions;

    if (use_cost_optimal) {
        // ========== COST_OPTIMAL Path: Use GPU cost-optimal partitioner ==========
        CostOptimalConfig cost_config = VerticalToCostOptimalConfig(config, partition_size);

        // Wait for data upload to complete, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        // Run GPU-based cost-optimal partitioning (PURE GPU - no CPU roundtrip!)
        PartitionerT partitioner(data, cost_config, stream);
        auto gpu_partitions = partitioner.partitionGPU();  // Pure GPU path
        num_partitions = gpu_partitions.num_partitions;

        if (num_partitions == 0) {
            // Fallback: treat as empty result
            cudaFree(d_data);
            return result;
        }

        // Allocate result arrays
        result.num_partitions = num_partitions;
        result.total_values = n;
        result.total_interleaved_partitions = num_partitions;

        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_interleaved_offsets, num_partitions * sizeof(int64_t));

        // Variable params: GPU path uses fixed layout for now
        result.d_param_offsets = nullptr;
        result.total_param_count = num_partitions * 4;
        result.use_variable_params = false;

        // Allocate min/max arrays for predicate pushdown
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        // Copy partition info using GPU-to-GPU transfer (no CPU roundtrip!)
        copyGPUPartitionResultToDevice<T>(
            gpu_partitions,
            result.d_start_indices, result.d_end_indices,
            result.d_model_types, result.d_model_params,
            result.d_delta_bits, result.d_error_bounds,
            stream);

        // Override model selection with adaptive selector if enabled (V2 only).
        if (config.enable_adaptive_selection) {
            if constexpr (!std::is_same<PartitionerT, GPUCostOptimalPartitionerV3<T>>::value) {
                adaptive_selector::ModelDecision<T>* d_decisions_temp;
                cudaMalloc(&d_decisions_temp, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

                adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
                    d_data, result.d_start_indices, result.d_end_indices,
                    num_partitions, d_decisions_temp, config.encoder_selector_block_size, stream,
                    config.enable_rle);

                // Unpack decisions to override model types and parameters
                int threads_unpack = BLOCK_SIZE;
                int blocks_unpack = (num_partitions + threads_unpack - 1) / threads_unpack;
                unpackDecisionsKernel<T><<<blocks_unpack, threads_unpack, 0, stream>>>(
                    d_decisions_temp,
                    result.d_model_types,
                    result.d_model_params,
                    result.d_delta_bits,
                    result.d_error_bounds,
                    num_partitions
                );

                cudaFree(d_decisions_temp);
            }
        }

    } else {
        // ========== FIXED Path: Use original fixed partitioning ==========
        // Wait for data upload, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        num_partitions = (n + partition_size - 1) / partition_size;

        result.num_partitions = num_partitions;
        result.total_values = n;
        result.total_interleaved_partitions = num_partitions;

        int threads = BLOCK_SIZE;
        int blocks_np = (num_partitions + threads - 1) / threads;

        // Generate partition indices on GPU
        cudaMalloc(&d_start_indices_temp, num_partitions * sizeof(int32_t));
        cudaMalloc(&d_end_indices_temp, num_partitions * sizeof(int32_t));

        generateFixedPartitionsKernel<<<blocks_np, threads, 0, stream>>>(
            d_start_indices_temp,
            d_end_indices_temp,
            partition_size,
            static_cast<int>(n),
            num_partitions
        );

        // Run GPU model selector
        cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

        if (config.enable_adaptive_selection) {
            adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions, d_decisions,
                config.encoder_selector_block_size, stream, config.enable_rle);
        } else {
            adaptive_selector::launchFixedModelSelector<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions,
                config.fixed_model_type, d_decisions, config.encoder_selector_block_size, stream);
        }

        // Allocate result metadata arrays
        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_interleaved_offsets, num_partitions * sizeof(int64_t));

        // Variable params: GPU path uses fixed layout for now
        result.d_param_offsets = nullptr;
        result.total_param_count = num_partitions * 4;
        result.use_variable_params = false;

        // Allocate min/max arrays for predicate pushdown
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        // Copy partition indices from temp to result
        cudaMemcpyAsync(result.d_start_indices, d_start_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result.d_end_indices, d_end_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        // Unpack decisions on GPU
        int threads_unpack = BLOCK_SIZE;
        int blocks_unpack = (num_partitions + threads_unpack - 1) / threads_unpack;
        unpackDecisionsKernel<T><<<blocks_unpack, threads_unpack, 0, stream>>>(
            d_decisions,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_error_bounds,
            num_partitions
        );
    }

    // ========== Common Path: Compute interleaved metadata and pack deltas ==========
    int threads = BLOCK_SIZE;
    int blocks_np = (num_partitions + threads - 1) / threads;

    // ========== Step 6: Compute interleaved metadata on GPU ==========
    int64_t* d_word_counts;
    cudaMalloc(&d_word_counts, num_partitions * sizeof(int64_t));

    computeInterleavedMetadataKernel<<<blocks_np, threads, 0, stream>>>(
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_delta_bits,
        result.d_model_params,
        result.d_num_mini_vectors,
        result.d_tail_sizes,
        d_word_counts,
        num_partitions
    );

    // GPU prefix sum for interleaved word offsets
    int64_t* d_word_prefix;
    cudaMalloc(&d_word_prefix, num_partitions * sizeof(int64_t));

    thrust::device_ptr<int64_t> wcount_ptr(d_word_counts);
    thrust::device_ptr<int64_t> wprefix_ptr(d_word_prefix);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        wcount_ptr, wcount_ptr + num_partitions,
        wprefix_ptr
    );

    // Set interleaved offsets
    setInterleavedOffsetsKernel<<<blocks_np, threads, 0, stream>>>(
        d_word_prefix,
        result.d_interleaved_offsets,
        num_partitions
    );

    // Get total interleaved words and mini-vector count
    int64_t h_last_wcount, h_last_wprefix;
    cudaMemcpyAsync(&h_last_wcount, d_word_counts + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_wprefix, d_word_prefix + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);

    // Count total mini-vectors on GPU and compute prefix sum for binary search
    int total_mini_vectors = 0;
    int* d_total_mv;
    int32_t* d_mv_prefix_sum;
    cudaMalloc(&d_total_mv, sizeof(int));
    cudaMalloc(&d_mv_prefix_sum, num_partitions * sizeof(int32_t));

    // Compute prefix sum of mini-vector counts for O(log n) partition lookup
    thrust::device_ptr<int32_t> mv_ptr(result.d_num_mini_vectors);
    thrust::device_ptr<int32_t> mv_prefix_ptr(d_mv_prefix_sum);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        mv_ptr, mv_ptr + num_partitions,
        mv_prefix_ptr
    );

    countTotalMiniVectorsKernel<<<1, 256, 0, stream>>>(
        result.d_num_mini_vectors,
        num_partitions,
        d_total_mv
    );
    cudaMemcpyAsync(&total_mini_vectors, d_total_mv, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    result.interleaved_delta_words = h_last_wprefix + h_last_wcount + 4;

    // Allocate interleaved deltas
    if (result.interleaved_delta_words > 0) {
        cudaMalloc(&result.d_interleaved_deltas,
                  result.interleaved_delta_words * sizeof(uint32_t));
        cudaMemsetAsync(result.d_interleaved_deltas, 0,
                       result.interleaved_delta_words * sizeof(uint32_t), stream);
    }

    cudaFree(d_word_counts);
    cudaFree(d_word_prefix);
    cudaFree(d_total_mv);

    // ========== Step 7: Pack mini-vector deltas (GPU) ==========
    if (total_mini_vectors > 0) {
	        convertToInterleavedKernel<T><<<total_mini_vectors, LANES_PER_MINI_VECTOR, 0, stream>>>(
	            d_data,
	            result.d_start_indices,
	            result.d_end_indices,
	            result.d_model_types,
	            result.d_model_params,
	            result.d_param_offsets,
	            result.d_delta_bits,
	            result.d_num_mini_vectors,
	            d_mv_prefix_sum,  // NEW: prefix sum for binary search
	            result.d_interleaved_offsets,
	            num_partitions,
            result.d_interleaved_deltas
        );
    }
    {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
        }
    }

    cudaFree(d_mv_prefix_sum);  // Free after kernel completes

    // ========== Step 8: Pack tail values (GPU) ==========
	    packTailValuesKernel<T><<<num_partitions, BLOCK_SIZE, 0, stream>>>(
	        d_data,
	        result.d_start_indices,
	        result.d_end_indices,
	        result.d_model_types,
	        result.d_model_params,
	        result.d_param_offsets,
	        result.d_delta_bits,
	        result.d_num_mini_vectors,
	        result.d_tail_sizes,
	        result.d_interleaved_offsets,
	        num_partitions,
        result.d_interleaved_deltas
    );

    // ========== Step 9: Compute partition min/max for predicate pushdown ==========
    {
        int minmax_block_size = 256;
        size_t minmax_shared_mem = 2 * minmax_block_size * sizeof(T);
        computePartitionMinMaxKernel<T><<<num_partitions, minmax_block_size, minmax_shared_mem, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            num_partitions,
            result.d_partition_min_values,
            result.d_partition_max_values
        );
    }

    // ========== Record kernel end time ==========
    cudaEventRecord(kernel_end, stream);

    // ========== Cleanup ==========
    cudaStreamSynchronize(stream);

    // Calculate kernel time
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end);
    result.kernel_time_ms = kernel_ms;
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);

    cudaFree(d_data);
    // Free temporary arrays only if they were allocated (FIXED path only)
    if (d_start_indices_temp) cudaFree(d_start_indices_temp);
    if (d_end_indices_temp) cudaFree(d_end_indices_temp);
    if (d_decisions) cudaFree(d_decisions);

    // ========== Generate V5 Consolidated Metadata ==========
    // IMPORTANT: Must be done BEFORE convertToVariableParams since V5 uses fixed layout (pid * 4)
    // Allocate V5 metadata array (64 bytes per partition, cache-line aligned)
    cudaMalloc(&result.d_metadata_v5, num_partitions * sizeof(PartitionMetadataV5));

    // Launch kernel to populate V5 metadata (uses fixed layout d_model_params[pid * 4])
    {
        int block_size = 256;
        int num_blocks = (num_partitions + block_size - 1) / block_size;
        generateV5MetadataKernel<<<num_blocks, block_size, 0, stream>>>(
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,  // Fixed layout: pid * 4
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_tail_sizes,
            result.d_interleaved_offsets,
            num_partitions,
            result.d_metadata_v5
        );
    }
    result.use_v5_metadata = true;

#if 1  // Set to 1 to enable variable params
    // Convert fixed-layout params to variable-length layout
    // NOTE: This must be AFTER V5 metadata generation since V5 uses fixed layout
    convertToVariableParams<T>(
        result.d_model_params,
        result.d_model_types,
        result.d_delta_bits,
        num_partitions,
        result,
        stream);
#endif

    return result;
}

template<typename T>
CompressedDataVertical<T> encodeVerticalGPU(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    return encodeVerticalGPU_Internal<T, GPUCostOptimalPartitionerV2<T>>(
        data, partition_size, config, stream);
}

template<typename T>
CompressedDataVertical<T> encodeVerticalGPU_PolyCost(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    return encodeVerticalGPU_Internal<T, GPUCostOptimalPartitionerV3<T>>(
        data, partition_size, config, stream);
}

/**
 * Full GPU Pipeline Encoder - Zero-Sync Version
 *
 * Uses pre-allocation to eliminate all intermediate synchronization points.
 * Only the final cudaStreamSynchronize remains.
 *
 * Trade-off: Uses more memory (worst-case 64-bit per value) but achieves
 * zero mid-pipeline CPU-GPU synchronization.
 *
 * Storage: INTERLEAVED-ONLY format (mini-vectors + tail)
 */
template<typename T, typename PartitionerT>
CompressedDataVertical<T> encodeVerticalGPU_ZeroSync_Internal(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataVertical<T> result;

    // For COST_OPTIMAL, partition_size is ignored (uses config.cost_min/max_partition_size)
    bool use_cost_optimal = (config.partitioning_strategy == PartitioningStrategy::COST_OPTIMAL);
    if (data.empty() || (!use_cost_optimal && partition_size <= 0)) {
        return result;
    }

    size_t n = data.size();
    int num_partitions;

    // ========== Step 1: Upload data to GPU ==========
    T* d_data;
    cudaMalloc(&d_data, n * sizeof(T));
    cudaMemcpyAsync(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream);

    // ========== CUDA Events for kernel timing (includes partitioning + encoding) ==========
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    // Temporary arrays for partition indices (used differently based on strategy)
    int32_t* d_start_indices_temp = nullptr;
    int32_t* d_end_indices_temp = nullptr;
    adaptive_selector::ModelDecision<T>* d_decisions = nullptr;

    // Storage for cost-optimal partitions
    std::vector<PartitionInfo> cost_optimal_partitions;

    if (use_cost_optimal) {
        // ========== COST_OPTIMAL Path: Use GPU cost-optimal partitioner ==========
        CostOptimalConfig cost_config = VerticalToCostOptimalConfig(config, partition_size);

        // Wait for data upload to complete, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        // Run GPU-based cost-optimal partitioning
        PartitionerT partitioner(data, cost_config, stream);
        cost_optimal_partitions = partitioner.partition();
        num_partitions = cost_optimal_partitions.size();

        if (num_partitions == 0) {
            cudaFree(d_data);
            return result;
        }

        // Allocate result arrays
        result.num_partitions = num_partitions;
        result.total_values = n;
        result.total_interleaved_partitions = num_partitions;

        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_interleaved_offsets, num_partitions * sizeof(int64_t));

        // Variable params: CPU path uses fixed layout via copyPartitionsToDevice
        result.d_param_offsets = nullptr;
        result.total_param_count = num_partitions * 4;
        result.use_variable_params = false;

        // Allocate min/max arrays for predicate pushdown
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        // Copy partition info from cost-optimal partitioner to device
        copyPartitionsToDevice<T>(
            cost_optimal_partitions,
            result.d_start_indices, result.d_end_indices,
            result.d_model_types, result.d_model_params,
            result.d_delta_bits, result.d_error_bounds,
            stream);

        // Override model selection with adaptive selector if enabled (V2 only).
        if (config.enable_adaptive_selection) {
            if constexpr (!std::is_same<PartitionerT, GPUCostOptimalPartitionerV3<T>>::value) {
                adaptive_selector::ModelDecision<T>* d_decisions_temp;
                cudaMalloc(&d_decisions_temp, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

                adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
                    d_data, result.d_start_indices, result.d_end_indices,
                    num_partitions, d_decisions_temp, config.encoder_selector_block_size, stream,
                    config.enable_rle);

                // Unpack decisions to override model types and parameters
                int threads_unpack = BLOCK_SIZE;
                int blocks_unpack = (num_partitions + threads_unpack - 1) / threads_unpack;
                unpackDecisionsKernel<T><<<blocks_unpack, threads_unpack, 0, stream>>>(
                    d_decisions_temp,
                    result.d_model_types,
                    result.d_model_params,
                    result.d_delta_bits,
                    result.d_error_bounds,
                    num_partitions
                );

                cudaFree(d_decisions_temp);
            }
        }

    } else {
        // ========== FIXED Path: Use original fixed partitioning ==========
        // Wait for data upload, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        num_partitions = (n + partition_size - 1) / partition_size;

        result.num_partitions = num_partitions;
        result.total_values = n;
        result.total_interleaved_partitions = num_partitions;

        int threads = BLOCK_SIZE;
        int blocks_np = (num_partitions + threads - 1) / threads;

        // Generate partition indices on GPU
        cudaMalloc(&d_start_indices_temp, num_partitions * sizeof(int32_t));
        cudaMalloc(&d_end_indices_temp, num_partitions * sizeof(int32_t));

        generateFixedPartitionsKernel<<<blocks_np, threads, 0, stream>>>(
            d_start_indices_temp,
            d_end_indices_temp,
            partition_size,
            static_cast<int>(n),
            num_partitions
        );

        // Run GPU model selector
        cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

        if (config.enable_adaptive_selection) {
            adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions, d_decisions,
                config.encoder_selector_block_size, stream, config.enable_rle);
        } else {
            adaptive_selector::launchFixedModelSelector<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions,
                config.fixed_model_type, d_decisions, config.encoder_selector_block_size, stream);
        }

        // Allocate result metadata arrays
        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_interleaved_offsets, num_partitions * sizeof(int64_t));

        // Variable params: GPU path uses fixed layout for now
        result.d_param_offsets = nullptr;
        result.total_param_count = num_partitions * 4;
        result.use_variable_params = false;

        // Allocate min/max arrays for predicate pushdown
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        // Copy partition indices from temp to result
        cudaMemcpyAsync(result.d_start_indices, d_start_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result.d_end_indices, d_end_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        // Unpack decisions on GPU
        int threads_unpack = BLOCK_SIZE;
        int blocks_unpack = (num_partitions + threads_unpack - 1) / threads_unpack;
        unpackDecisionsKernel<T><<<blocks_unpack, threads_unpack, 0, stream>>>(
            d_decisions,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_error_bounds,
            num_partitions
        );
    }

    // ========== Common Path: Compute interleaved metadata and pack deltas ==========
    int threads = BLOCK_SIZE;
    int blocks_np = (num_partitions + threads - 1) / threads;

    // Compute interleaved metadata on GPU
    int64_t* d_word_counts;
    cudaMalloc(&d_word_counts, num_partitions * sizeof(int64_t));

    computeInterleavedMetadataKernel<<<blocks_np, threads, 0, stream>>>(
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_delta_bits,
        result.d_model_params,
        result.d_num_mini_vectors,
        result.d_tail_sizes,
        d_word_counts,
        num_partitions
    );

    // GPU prefix sum for interleaved word offsets
    int64_t* d_word_prefix;
    cudaMalloc(&d_word_prefix, num_partitions * sizeof(int64_t));

    thrust::device_ptr<int64_t> wcount_ptr(d_word_counts);
    thrust::device_ptr<int64_t> wprefix_ptr(d_word_prefix);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        wcount_ptr, wcount_ptr + num_partitions,
        wprefix_ptr
    );

    // Set interleaved offsets from prefix sums
    setInterleavedOffsetsKernel<<<blocks_np, threads, 0, stream>>>(
        d_word_prefix,
        result.d_interleaved_offsets,
        num_partitions
    );

    // Compute mini-vector prefix sum for O(log n) partition lookup
    int32_t* d_mv_prefix_sum;
    cudaMalloc(&d_mv_prefix_sum, num_partitions * sizeof(int32_t));
    thrust::device_ptr<int32_t> mv_ptr(result.d_num_mini_vectors);
    thrust::device_ptr<int32_t> mv_prefix_ptr(d_mv_prefix_sum);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        mv_ptr, mv_ptr + num_partitions,
        mv_prefix_ptr
    );

    // ========== ZERO-SYNC: Pre-allocate interleaved buffer with max size ==========
    // Maximum bits = n * 64 (worst case: each value needs 64 bits)
    int64_t max_interleaved_bits = static_cast<int64_t>(n) * 64;
    int64_t max_interleaved_words = (max_interleaved_bits + 31) / 32 + 4;

    cudaMalloc(&result.d_interleaved_deltas,
              max_interleaved_words * sizeof(uint32_t));
    cudaMemsetAsync(result.d_interleaved_deltas, 0,
                   max_interleaved_words * sizeof(uint32_t), stream);

    // NOTE: Keep d_word_counts and d_word_prefix alive for accurate size calculation after sync

    // Pack mini-vector deltas (GPU)
    // Pre-calculate max mini-vectors (no sync needed)
    int max_mini_vectors = (n + MINI_VECTOR_SIZE - 1) / MINI_VECTOR_SIZE;

    if (max_mini_vectors > 0) {
	        convertToInterleavedKernel<T><<<max_mini_vectors, LANES_PER_MINI_VECTOR, 0, stream>>>(
	            d_data,
	            result.d_start_indices,
	            result.d_end_indices,
	            result.d_model_types,
	            result.d_model_params,
	            result.d_param_offsets,
	            result.d_delta_bits,
	            result.d_num_mini_vectors,
	            d_mv_prefix_sum,  // NEW: prefix sum for binary search
	            result.d_interleaved_offsets,
	            num_partitions,
            result.d_interleaved_deltas
        );
    }

    cudaFree(d_mv_prefix_sum);  // Free after kernel completes

    // Pack tail values (GPU)
	packTailValuesKernel<T><<<num_partitions, BLOCK_SIZE, 0, stream>>>(
	    d_data,
	    result.d_start_indices,
	    result.d_end_indices,
	    result.d_model_types,
	    result.d_model_params,
	    result.d_param_offsets,
	    result.d_delta_bits,
	    result.d_num_mini_vectors,
	    result.d_tail_sizes,
	    result.d_interleaved_offsets,
	    num_partitions,
        result.d_interleaved_deltas
    );

    // Compute partition min/max for predicate pushdown
    {
        int minmax_block_size = 256;
        size_t minmax_shared_mem = 2 * minmax_block_size * sizeof(T);
        computePartitionMinMaxKernel<T><<<num_partitions, minmax_block_size, minmax_shared_mem, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            num_partitions,
            result.d_partition_min_values,
            result.d_partition_max_values
        );
    }

    // ========== Record kernel end time ==========
    cudaEventRecord(kernel_end, stream);

    // ========== ONLY synchronization point ==========
    cudaStreamSynchronize(stream);

    // ========== ZERO-SYNC: Now read accurate word count (after sync) ==========
    int64_t h_last_wcount, h_last_wprefix;
    cudaMemcpy(&h_last_wcount, d_word_counts + num_partitions - 1,
               sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_wprefix, d_word_prefix + num_partitions - 1,
               sizeof(int64_t), cudaMemcpyDeviceToHost);
    result.interleaved_delta_words = h_last_wprefix + h_last_wcount + 4;

    // Now free the temporary arrays
    cudaFree(d_word_counts);
    cudaFree(d_word_prefix);

    // Calculate kernel time
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end);
    result.kernel_time_ms = kernel_ms;
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);

    // Cleanup
    cudaFree(d_data);
    // Free temporary arrays only if they were allocated (FIXED path only)
    if (d_start_indices_temp) cudaFree(d_start_indices_temp);
    if (d_end_indices_temp) cudaFree(d_end_indices_temp);
    if (d_decisions) cudaFree(d_decisions);

    // ========== Generate V5 Consolidated Metadata ==========
    // IMPORTANT: Must be done BEFORE convertToVariableParams since V5 uses fixed layout (pid * 4)
    // Allocate V5 metadata array (64 bytes per partition, cache-line aligned)
    cudaMalloc(&result.d_metadata_v5, num_partitions * sizeof(PartitionMetadataV5));

    // Launch kernel to populate V5 metadata (uses fixed layout d_model_params[pid * 4])
    {
        int block_size = 256;
        int num_blocks = (num_partitions + block_size - 1) / block_size;
        generateV5MetadataKernel<<<num_blocks, block_size, 0, stream>>>(
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,  // Fixed layout: pid * 4
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_tail_sizes,
            result.d_interleaved_offsets,
            num_partitions,
            result.d_metadata_v5
        );
    }
    result.use_v5_metadata = true;

#if 1  // Set to 1 to enable variable params
    // Convert fixed-layout params to variable-length layout
    // NOTE: This must be AFTER V5 metadata generation since V5 uses fixed layout
    convertToVariableParams<T>(
        result.d_model_params,
        result.d_model_types,
        result.d_delta_bits,
        num_partitions,
        result,
        stream);
#endif

    return result;
}

template<typename T>
CompressedDataVertical<T> encodeVerticalGPU_ZeroSync(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    return encodeVerticalGPU_ZeroSync_Internal<T, GPUCostOptimalPartitionerV2<T>>(
        data, partition_size, config, stream);
}

template<typename T>
CompressedDataVertical<T> encodeVerticalGPU_ZeroSync_PolyCost(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    return encodeVerticalGPU_ZeroSync_Internal<T, GPUCostOptimalPartitionerV3<T>>(
        data, partition_size, config, stream);
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template std::vector<PartitionInfo> createFixedPartitions<uint32_t>(int, int);
template std::vector<PartitionInfo> createFixedPartitions<uint64_t>(int, int);
template std::vector<PartitionInfo> createFixedPartitions<int32_t>(int, int);
template std::vector<PartitionInfo> createFixedPartitions<int64_t>(int, int);

template CompressedDataVertical<uint32_t> encodeVertical<uint32_t>(
    const std::vector<uint32_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<uint64_t> encodeVertical<uint64_t>(
    const std::vector<uint64_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int32_t> encodeVertical<int32_t>(
    const std::vector<int32_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int64_t> encodeVertical<int64_t>(
    const std::vector<int64_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);

template void freeCompressedData<uint32_t>(CompressedDataVertical<uint32_t>&);
template void freeCompressedData<uint64_t>(CompressedDataVertical<uint64_t>&);
template void freeCompressedData<int32_t>(CompressedDataVertical<int32_t>&);
template void freeCompressedData<int64_t>(CompressedDataVertical<int64_t>&);

template CompressedDataVertical<uint32_t> encodeVerticalGPU<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<uint64_t> encodeVerticalGPU<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int32_t> encodeVerticalGPU<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int64_t> encodeVerticalGPU<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

template CompressedDataVertical<uint32_t> encodeVerticalGPU_ZeroSync<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<uint64_t> encodeVerticalGPU_ZeroSync<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int32_t> encodeVerticalGPU_ZeroSync<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int64_t> encodeVerticalGPU_ZeroSync<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

template CompressedDataVertical<uint32_t> encodeVerticalGPU_PolyCost<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<uint64_t> encodeVerticalGPU_PolyCost<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int32_t> encodeVerticalGPU_PolyCost<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int64_t> encodeVerticalGPU_PolyCost<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

template CompressedDataVertical<uint32_t> encodeVerticalGPU_ZeroSync_PolyCost<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<uint64_t> encodeVerticalGPU_ZeroSync_PolyCost<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int32_t> encodeVerticalGPU_ZeroSync_PolyCost<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
template CompressedDataVertical<int64_t> encodeVerticalGPU_ZeroSync_PolyCost<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

}  // namespace Vertical_encoder

#endif // ENCODER_Vertical_OPT_CU
