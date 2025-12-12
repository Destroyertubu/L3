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

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../utils/bitpack_utils_Vertical.cuh"
#include "adaptive_selector.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"  // GPU cost-optimal partitioner

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
            predicted = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        default:
            // Fallback to linear for unknown types
            predicted = params[0] + params[1] * x;
            break;
    }

    // CRITICAL: Use __double2ull_rn for uint64_t to avoid overflow when predicted > INT64_MAX
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(predicted));
    } else {
        return static_cast<T>(__double2ll_rn(predicted));
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
            double theta0 = d_model_params[pid * 4];
            double theta1 = d_model_params[pid * 4 + 1];
            double theta2 = d_model_params[pid * 4 + 2];
            double theta3 = d_model_params[pid * 4 + 3];
            double x = static_cast<double>(local_idx);
            double predicted;

            switch (model_type) {
                case MODEL_CONSTANT:
                    predicted = theta0;
                    break;
                case MODEL_LINEAR:
                    predicted = theta0 + theta1 * x;
                    break;
                case MODEL_POLYNOMIAL2:
                    // Horner: a0 + x*(a1 + x*a2)
                    predicted = theta0 + x * (theta1 + x * theta2);
                    break;
                case MODEL_POLYNOMIAL3:
                    // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                    predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
                    break;
                default:
                    predicted = theta0 + theta1 * x;
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
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
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

    // Thread 0 finds the partition
    if (threadIdx.x == 0) {
        int mv_global = blockIdx.x;
        int cumulative_mv = 0;
        s_valid = false;  // Initialize to invalid

        for (int p = 0; p < num_partitions; p++) {
            int num_mv = d_num_mini_vectors[p];
            if (mv_global < cumulative_mv + num_mv) {
                s_pid = p;
                s_mv_in_partition = mv_global - cumulative_mv;
                s_partition_start = d_start_indices[p];
                s_delta_bits = d_delta_bits[p];
                s_params[0] = d_model_params[p * 4];
                s_params[1] = d_model_params[p * 4 + 1];
                s_params[2] = d_model_params[p * 4 + 2];
                s_params[3] = d_model_params[p * 4 + 3];
                s_interleaved_base = d_interleaved_offsets[p];
                s_model_type = d_model_types[p];
                s_valid = true;  // Mark as valid
                break;
            }
            cumulative_mv += num_mv;
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

    // Each thread is a lane (0-31), processes 8 values
    int lane_id = threadIdx.x;
    if (lane_id >= WARP_SIZE) return;  // Only need 32 threads

    // Global indices this lane processes
    int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

    // Calculate bit position for this lane's data
    // Lane L's data in mini-vector starts at: mv_word_base + L * 8 * bit_width bits
    int64_t mv_word_base = interleaved_base + static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits / 32;
    int64_t lane_bit_start = (mv_word_base << 5) + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;

    // Process 8 values for this lane
    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        // Global index: lane_id + v * 32 within mini-vector
        int local_idx_in_mv = lane_id + v * WARP_SIZE;
        int global_idx = mv_start_global + local_idx_in_mv;
        int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + local_idx_in_mv;

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
        } else {
            // Compute prediction using polynomial model - matches decoder exactly
            T pred_val = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
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
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int32_t* __restrict__ d_tail_sizes,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint32_t* __restrict__ interleaved_array)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int tail_size = d_tail_sizes[pid];
    if (tail_size == 0) return;  // No tail to pack

    int64_t interleaved_base = d_interleaved_offsets[pid];
    if (interleaved_base < 0) return;  // Not using interleaved format

    int num_mv = d_num_mini_vectors[pid];
    int delta_bits = d_delta_bits[pid];
    int model_type = d_model_types[pid];
    int partition_start = d_start_indices[pid];

    // Load model parameters into registers
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

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

        // All partitions use interleaved format now
        num_mini_vectors[p] = partition_size / MINI_VECTOR_SIZE;
        tail_sizes[p] = partition_size % MINI_VECTOR_SIZE;
        interleaved_offsets[p] = current_word_offset;

        // Calculate words for this partition's interleaved data (mini-vectors + tail)
        int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[p]) * MINI_VECTOR_SIZE * bit_width;
        int64_t tail_bits = static_cast<int64_t>(tail_sizes[p]) * bit_width;
        int64_t total_bits = mv_bits + tail_bits;
        int64_t words = (total_bits + 31) / 32;

        current_word_offset += words;
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

    // Calculate total interleaved words (now stores ALL data - mini-vectors + tails)
    int64_t max_interleaved_offset = 0;
    int total_mini_vectors = 0;
    for (size_t p = 0; p < partitions.size(); p++) {
        int bit_width = partitions[p].delta_bits;
        int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[p]) * MINI_VECTOR_SIZE * bit_width;
        int64_t tail_bits = static_cast<int64_t>(tail_sizes[p]) * bit_width;
        int64_t end_word = interleaved_offsets[p] + (mv_bits + tail_bits + 31) / 32;
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
    cudaMalloc(&result.d_model_params, np * 4 * sizeof(double));
    cudaMalloc(&result.d_delta_bits, np * sizeof(int32_t));
    cudaMalloc(&result.d_delta_array_bit_offsets, np * sizeof(int64_t));
    cudaMalloc(&result.d_error_bounds, np * sizeof(int64_t));

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
    std::vector<double> h_model_params(np * 4);
    std::vector<int64_t> h_error_bounds(np);

    for (int p = 0; p < np; p++) {
        h_start_indices[p] = partitions[p].start_idx;
        h_end_indices[p] = partitions[p].end_idx;
        h_model_types[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
    }

    cudaMemcpyAsync(result.d_start_indices, h_start_indices.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_end_indices, h_end_indices.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_model_types, h_model_types.data(),
                    np * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.d_model_params, h_model_params.data(),
                    np * 4 * sizeof(double), cudaMemcpyHostToDevice, stream);
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

    // ========== Step 5: Copy data to device ==========
    T* d_values;
    cudaMalloc(&d_values, data.size() * sizeof(T));
    cudaMemcpyAsync(d_values, data.data(), data.size() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);

    // ========== Step 6: Pack interleaved deltas (mini-vectors) ==========
    if (total_mini_vectors > 0) {
        // One block per mini-vector, 32 threads per block (one per lane)
        convertToInterleavedKernel<T><<<total_mini_vectors, WARP_SIZE, 0, stream>>>(
            d_values,
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_interleaved_offsets,
            np,
            result.d_interleaved_deltas
        );
    }

    // ========== Step 7: Pack tail values ==========
    // One block per partition, BLOCK_SIZE threads per block
    packTailValuesKernel<T><<<np, BLOCK_SIZE, 0, stream>>>(
        d_values,
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_model_params,
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
    const int32_t* __restrict__ delta_bits,
    int32_t* __restrict__ num_mini_vectors,
    int32_t* __restrict__ tail_sizes,
    int64_t* __restrict__ word_counts,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    int psize = end_indices[pid] - start_indices[pid];
    int bit_width = delta_bits[pid];

    // All partitions use interleaved format now
    num_mini_vectors[pid] = psize / MINI_VECTOR_SIZE;
    tail_sizes[pid] = psize % MINI_VECTOR_SIZE;

    int64_t mv_bits = static_cast<int64_t>(num_mini_vectors[pid]) * MINI_VECTOR_SIZE * bit_width;
    int64_t tail_bits = static_cast<int64_t>(tail_sizes[pid]) * bit_width;
    word_counts[pid] = (mv_bits + tail_bits + 31) / 32;
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
    cost_config.target_partition_size = (fl_config.cost_target_partition_size > 0) ?
                                         fl_config.cost_target_partition_size : partition_size;
    cost_config.breakpoint_threshold = fl_config.cost_breakpoint_threshold;
    cost_config.merge_benefit_threshold = fl_config.cost_merge_benefit_threshold;
    cost_config.max_merge_rounds = fl_config.cost_max_merge_rounds;
    cost_config.enable_merging = fl_config.cost_enable_merging;
    // NOTE: enable_polynomial_models controls POLY2/POLY3 model selection, which is
    // different from enable_adaptive_selection (LINEAR vs FOR). The polynomial
    // refit modifies costs with different overhead values (16 vs 64 bytes),
    // which breaks merge cost calculations. Disable by default.
    cost_config.enable_polynomial_models = false;  // Keep disabled to avoid cost mismatches
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
 * Helper: Copy GPUPartitionResult to device arrays (GPU-to-GPU, no CPU roundtrip!)
 */
template<typename T>
void copyGPUPartitionResultToDevice(
    const GPUPartitionResult<T>& gpu_result,
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

template<typename T>
CompressedDataVertical<T> encodeVerticalGPU(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataVertical<T> result;

    if (data.empty() || partition_size <= 0) {
        return result;
    }

    size_t n = data.size();
    int num_partitions;
    bool use_cost_optimal = (config.partitioning_strategy == PartitioningStrategy::COST_OPTIMAL);

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
        // ========== COST_OPTIMAL Path: Use GPUCostOptimalPartitionerV2 ==========
        CostOptimalConfig cost_config = VerticalToCostOptimalConfig(config, partition_size);

        // Wait for data upload to complete, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        // Run GPU-based cost-optimal partitioning (PURE GPU - no CPU roundtrip!)
        GPUCostOptimalPartitionerV2<T> partitioner(data, cost_config, stream);
        GPUPartitionResult<T> gpu_partitions = partitioner.partitionGPU();  // Pure GPU path
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
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions, d_decisions, stream);
        } else {
            adaptive_selector::launchFixedModelSelector<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions,
                config.fixed_model_type, d_decisions, stream);
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
        result.d_delta_bits,
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

    // Count total mini-vectors on GPU
    int total_mini_vectors = 0;
    int* d_total_mv;
    cudaMalloc(&d_total_mv, sizeof(int));
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
        convertToInterleavedKernel<T><<<total_mini_vectors, WARP_SIZE, 0, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_interleaved_offsets,
            num_partitions,
            result.d_interleaved_deltas
        );
    }

    // ========== Step 8: Pack tail values (GPU) ==========
    packTailValuesKernel<T><<<num_partitions, BLOCK_SIZE, 0, stream>>>(
        d_data,
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_model_params,
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

    return result;
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
template<typename T>
CompressedDataVertical<T> encodeVerticalGPU_ZeroSync(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataVertical<T> result;

    if (data.empty() || partition_size <= 0) {
        return result;
    }

    size_t n = data.size();
    int num_partitions;
    bool use_cost_optimal = (config.partitioning_strategy == PartitioningStrategy::COST_OPTIMAL);

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
        // ========== COST_OPTIMAL Path: Use GPUCostOptimalPartitionerV2 ==========
        CostOptimalConfig cost_config = VerticalToCostOptimalConfig(config, partition_size);

        // Wait for data upload to complete, then start kernel timing
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        // Run GPU-based cost-optimal partitioning
        GPUCostOptimalPartitionerV2<T> partitioner(data, cost_config, stream);
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
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions, d_decisions, stream);
        } else {
            adaptive_selector::launchFixedModelSelector<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions,
                config.fixed_model_type, d_decisions, stream);
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
        result.d_delta_bits,
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

    // ========== ZERO-SYNC: Pre-allocate interleaved buffer with max size ==========
    // Maximum bits = n * 64 (worst case: each value needs 64 bits)
    int64_t max_interleaved_bits = static_cast<int64_t>(n) * 64;
    result.interleaved_delta_words = (max_interleaved_bits + 31) / 32 + 4;

    cudaMalloc(&result.d_interleaved_deltas,
              result.interleaved_delta_words * sizeof(uint32_t));
    cudaMemsetAsync(result.d_interleaved_deltas, 0,
                   result.interleaved_delta_words * sizeof(uint32_t), stream);

    cudaFree(d_word_counts);
    cudaFree(d_word_prefix);

    // Pack mini-vector deltas (GPU)
    // Pre-calculate max mini-vectors (no sync needed)
    int max_mini_vectors = (n + MINI_VECTOR_SIZE - 1) / MINI_VECTOR_SIZE;

    if (max_mini_vectors > 0) {
        convertToInterleavedKernel<T><<<max_mini_vectors, WARP_SIZE, 0, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_interleaved_offsets,
            num_partitions,
            result.d_interleaved_deltas
        );
    }

    // Pack tail values (GPU)
    packTailValuesKernel<T><<<num_partitions, BLOCK_SIZE, 0, stream>>>(
        d_data,
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_model_params,
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

    return result;
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

}  // namespace Vertical_encoder

#endif // ENCODER_Vertical_OPT_CU
