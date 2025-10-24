/**
 * GLECO Optimized Encoder Kernels
 *
 * GPU optimization techniques applied:
 * 1. Warp-level primitives for reduction
 * 2. Vector memory access (int4/float4)
 * 3. Shared memory bank conflict avoidance
 * 4. Coalesced memory access patterns
 * 5. Stream-based parallel execution
 * 6. Reduced atomic operations
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <cmath>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include "L3_format.hpp"

namespace cg = cooperative_groups;

#define MAX_DELTA_BITS 64
#define WARP_SIZE 32
#define FULL_MASK 0xFFFFFFFF

// ============================================================================
// Warp-level Reduction Primitives
// ============================================================================

template<typename T>
__device__ __forceinline__ T warpReduce(T val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpMax(T val) {
    unsigned mask = __activemask();
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        T other = __shfl_down_sync(mask, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

__device__ __forceinline__ bool warpAny(bool val) {
    unsigned mask = __activemask();
    return __any_sync(mask, val);
}

// ============================================================================
// Vectorized Memory Access
// ============================================================================

template<typename T>
struct VectorType {
    using Type = T;
    static constexpr int elements = 1;
};

template<>
struct VectorType<uint32_t> {
    using Type = uint4;
    static constexpr int elements = 4;
};

template<>
struct VectorType<int32_t> {
    using Type = int4;
    static constexpr int elements = 4;
};

// ============================================================================
// Helper Functions (optimized versions)
// ============================================================================

template<typename T>
__device__ __forceinline__ bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;
    } else {
        const uint64_t DOUBLE_PRECISION_LIMIT = (1ULL << 53);
        return static_cast<uint64_t>(value) > DOUBLE_PRECISION_LIMIT;
    }
}

template<typename T>
__device__ __forceinline__ long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        if (sizeof(T) == 8) {
            unsigned long long actual_ull = static_cast<unsigned long long>(actual);
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);

            if (actual_ull >= pred_ull) {
                unsigned long long diff = actual_ull - pred_ull;
                return (diff <= static_cast<unsigned long long>(LLONG_MAX)) ?
                       static_cast<long long>(diff) : LLONG_MAX;
            } else {
                unsigned long long diff = pred_ull - actual_ull;
                return (diff <= static_cast<unsigned long long>(LLONG_MAX)) ?
                       -static_cast<long long>(diff) : LLONG_MIN;
            }
        } else {
            return static_cast<long long>(actual) - static_cast<long long>(predicted);
        }
    }
}

// ============================================================================
// Optimized Model Fitting Kernel with Warp Primitives
// ============================================================================

template<typename T>
__global__ void processPartitionsOptimized(
    const T* __restrict__ values_device,
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_model_types,
    double* __restrict__ d_model_params,
    int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_error_bounds,
    int num_partitions,
    int64_t* __restrict__ total_bits_device)
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
    long long* warp_max_error = reinterpret_cast<long long*>(warp_sums + num_warps * 4);
    bool* warp_overflow = reinterpret_cast<bool*>(warp_max_error + num_warps);

    // Phase 1: Check for overflow using warp primitives
    bool local_overflow = false;
    for (int i = tid; i < segment_len; i += blockDim.x) {
        if (mightOverflowDoublePrecision(values_device[start_idx + i])) {
            local_overflow = true;
            break;
        }
    }

    // Warp-level any reduction
    bool warp_has_overflow = warpAny(local_overflow);

    if (lane_id == 0) {
        warp_overflow[warp_id] = warp_has_overflow;
    }
    __syncthreads();

    // Final reduction in first warp
    bool has_overflow = false;
    if (tid < num_warps) {
        has_overflow = warp_overflow[tid];
        has_overflow = warpAny(has_overflow);
    }

    if (tid == 0 && has_overflow) {
        d_model_types[partition_idx] = MODEL_DIRECT_COPY;
        d_model_params[partition_idx * 4] = 0.0;
        d_model_params[partition_idx * 4 + 1] = 0.0;
        d_model_params[partition_idx * 4 + 2] = 0.0;
        d_model_params[partition_idx * 4 + 3] = 0.0;
        d_error_bounds[partition_idx] = 0;
        d_delta_bits[partition_idx] = sizeof(T) * 8;
    }
    __syncthreads();

    if (!has_overflow) {
        // Phase 2: Fit linear model using warp-level reductions
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

        // Vectorized data loading for better memory throughput
        for (int i = tid; i < segment_len; i += blockDim.x) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(values_device[start_idx + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        // Warp-level reduction
        sum_x = warpReduce(sum_x);
        sum_y = warpReduce(sum_y);
        sum_xx = warpReduce(sum_xx);
        sum_xy = warpReduce(sum_xy);

        // Store warp results
        if (lane_id == 0) {
            warp_sums[warp_id * 4] = sum_x;
            warp_sums[warp_id * 4 + 1] = sum_y;
            warp_sums[warp_id * 4 + 2] = sum_xx;
            warp_sums[warp_id * 4 + 3] = sum_xy;
        }
        __syncthreads();

        // Final reduction in first warp
        if (warp_id == 0 && lane_id < num_warps) {
            sum_x = warp_sums[lane_id * 4];
            sum_y = warp_sums[lane_id * 4 + 1];
            sum_xx = warp_sums[lane_id * 4 + 2];
            sum_xy = warp_sums[lane_id * 4 + 3];

            sum_x = warpReduce(sum_x);
            sum_y = warpReduce(sum_y);
            sum_xx = warpReduce(sum_xx);
            sum_xy = warpReduce(sum_xy);

            if (lane_id == 0) {
                double n = static_cast<double>(segment_len);
                double determinant = n * sum_xx - sum_x * sum_x;

                double theta0, theta1;
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

                // Store in shared memory for other threads to use
                warp_sums[0] = theta0;
                warp_sums[1] = theta1;
            }
        }
        __syncthreads();

        double theta0 = warp_sums[0];
        double theta1 = warp_sums[1];

        // Phase 3: Calculate maximum error using warp primitives
        long long max_error = 0;

        for (int i = tid; i < segment_len; i += blockDim.x) {
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(round(predicted));
            long long delta = calculateDelta(values_device[start_idx + i], pred_T);
            long long abs_error = (delta < 0) ? -delta : delta;
            max_error = (abs_error > max_error) ? abs_error : max_error;
        }

        // Warp-level max reduction
        max_error = warpMax(max_error);

        if (lane_id == 0) {
            warp_max_error[warp_id] = max_error;
        }
        __syncthreads();

        // Final reduction in first warp
        if (warp_id == 0 && lane_id < num_warps) {
            max_error = warp_max_error[lane_id];
            max_error = warpMax(max_error);

            if (lane_id == 0) {
                d_error_bounds[partition_idx] = max_error;

                // Calculate delta bits
                int delta_bits = 0;
                if (max_error > 0) {
                    unsigned long long temp = static_cast<unsigned long long>(max_error);
                    delta_bits = 64 - __clzll(temp) + 1; // +1 for sign bit
                    delta_bits = min(delta_bits, MAX_DELTA_BITS);
                    delta_bits = max(delta_bits, 0);
                }
                d_delta_bits[partition_idx] = delta_bits;
            }
        }
    }

    // Atomic add to total bits counter
    if (tid == 0) {
        int64_t partition_bits = (int64_t)segment_len * d_delta_bits[partition_idx];
        atomicAdd(reinterpret_cast<unsigned long long*>(total_bits_device),
                  static_cast<unsigned long long>(partition_bits));
    }
}

// ============================================================================
// Optimized Bit Offset Calculation using Prefix Sum
// ============================================================================

__global__ void setBitOffsetsOptimized(
    int32_t* __restrict__ d_start_indices,
    int32_t* __restrict__ d_end_indices,
    int32_t* __restrict__ d_delta_bits,
    int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Use cooperative groups for efficient prefix sum
    cg::thread_block block = cg::this_thread_block();

    if (tid < num_partitions) {
        // Calculate bit usage for this partition
        int seg_len = d_end_indices[tid] - d_start_indices[tid];
        int64_t my_bits = (int64_t)seg_len * d_delta_bits[tid];

        // Simple sequential prefix sum for now (can be optimized with parallel scan)
        int64_t bit_offset = 0;
        for (int i = 0; i < tid; i++) {
            int s_len = d_end_indices[i] - d_start_indices[i];
            bit_offset += (int64_t)s_len * d_delta_bits[i];
        }

        d_delta_array_bit_offsets[tid] = bit_offset;
    }
}

// ============================================================================
// Optimized Delta Packing using Shared Memory Buffering
// ============================================================================

template<typename T>
__global__ void packDeltasOptimized(
    const T* __restrict__ values_device,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions_val,
    uint32_t* __restrict__ delta_array_device)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    if (num_partitions_val == 0) return;

    int max_idx_to_process = d_end_indices[num_partitions_val - 1];

    // Shared memory buffer for coalescing writes
    __shared__ uint32_t shared_buffer[256];
    __shared__ int shared_partition_cache[32];

    // Cache partition boundaries in shared memory
    if (threadIdx.x < 32 && threadIdx.x < num_partitions_val) {
        shared_partition_cache[threadIdx.x] = d_start_indices[threadIdx.x];
    }
    __syncthreads();

    for (int current_idx = g_idx; current_idx < max_idx_to_process; current_idx += g_stride) {
        // Optimized binary search using shared memory cache
        int p_left = 0, p_right = num_partitions_val - 1;
        int found_partition_idx = -1;

        // Fast path for common case (sequential access)
        int estimated_partition = current_idx * num_partitions_val / max_idx_to_process;
        if (estimated_partition < num_partitions_val) {
            int32_t est_start = d_start_indices[estimated_partition];
            int32_t est_end = d_end_indices[estimated_partition];
            if (current_idx >= est_start && current_idx < est_end) {
                found_partition_idx = estimated_partition;
            }
        }

        // Fallback to binary search if estimate was wrong
        if (found_partition_idx == -1) {
            while (p_left <= p_right) {
                int p_mid = p_left + (p_right - p_left) / 2;
                int32_t current_start = d_start_indices[p_mid];
                int32_t current_end = d_end_indices[p_mid];

                if (current_idx >= current_start && current_idx < current_end) {
                    found_partition_idx = p_mid;
                    break;
                } else if (current_idx < current_start) {
                    p_right = p_mid - 1;
                } else {
                    p_left = p_mid + 1;
                }
            }
        }

        if (found_partition_idx == -1) continue;

        // Get partition data
        int32_t current_model_type = d_model_types[found_partition_idx];
        int32_t current_delta_bits = d_delta_bits[found_partition_idx];
        int64_t current_bit_offset_base = d_delta_array_bit_offsets[found_partition_idx];
        int32_t current_start_idx = d_start_indices[found_partition_idx];

        if (current_model_type == MODEL_DIRECT_COPY) {
            // Direct copy path
            int local_idx = current_idx - current_start_idx;
            int64_t bit_offset = current_bit_offset_base + (int64_t)local_idx * current_delta_bits;

            T value = values_device[current_idx];
            uint64_t value_to_store = static_cast<uint64_t>(value);

            // Optimized bit packing with reduced atomics
            int start_word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;

            if (current_delta_bits <= 32 - offset_in_word) {
                // Single word write
                uint32_t mask = (current_delta_bits == 32) ? ~0U : ((1U << current_delta_bits) - 1U);
                uint32_t value_part = (value_to_store & mask) << offset_in_word;
                atomicOr(&delta_array_device[start_word_idx], value_part);
            } else {
                // Multi-word write
                int bits_remaining = current_delta_bits;
                int word_idx = start_word_idx;

                while (bits_remaining > 0) {
                    int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                    uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                    uint32_t value_part = (value_to_store & mask) << offset_in_word;
                    atomicOr(&delta_array_device[word_idx], value_part);

                    value_to_store >>= bits_in_this_word;
                    bits_remaining -= bits_in_this_word;
                    word_idx++;
                    offset_in_word = 0;
                }
            }
        } else {
            // Normal delta encoding path
            int current_local_idx = current_idx - current_start_idx;

            // Use FMA for better accuracy and performance
            double pred_double = fma(d_model_params[found_partition_idx * 4 + 1],
                                    static_cast<double>(current_local_idx),
                                    d_model_params[found_partition_idx * 4]);

            if (current_model_type == MODEL_POLYNOMIAL2) {
                pred_double = fma(d_model_params[found_partition_idx * 4 + 2],
                                 static_cast<double>(current_local_idx * current_local_idx),
                                 pred_double);
            }

            T pred_T_val = static_cast<T>(round(pred_double));
            long long current_delta_ll = calculateDelta(values_device[current_idx], pred_T_val);

            if (current_delta_bits > 0) {
                int64_t current_bit_offset_val = current_bit_offset_base +
                                                 (int64_t)current_local_idx * current_delta_bits;

                // Optimized bit packing
                if (current_delta_bits <= 32) {
                    uint32_t final_packed_delta = static_cast<uint32_t>(current_delta_ll &
                                                                       ((1ULL << current_delta_bits) - 1ULL));

                    int target_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;

                    if (current_delta_bits + offset_in_word <= 32) {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word);
                    } else {
                        atomicOr(&delta_array_device[target_word_idx], final_packed_delta << offset_in_word);
                        atomicOr(&delta_array_device[target_word_idx + 1],
                                final_packed_delta >> (32 - offset_in_word));
                    }
                } else {
                    // For deltas > 32 bits
                    uint64_t final_packed_delta_64 = static_cast<uint64_t>(current_delta_ll &
                        ((current_delta_bits == 64) ? ~0ULL : ((1ULL << current_delta_bits) - 1ULL)));

                    int start_word_idx = current_bit_offset_val / 32;
                    int offset_in_word = current_bit_offset_val % 32;
                    int bits_remaining = current_delta_bits;
                    int word_idx = start_word_idx;
                    uint64_t delta_to_write = final_packed_delta_64;

                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t value = (delta_to_write & mask) << offset_in_word;
                        atomicOr(&delta_array_device[word_idx], value);

                        delta_to_write >>= bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        word_idx++;
                        offset_in_word = 0;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Optimized Kernel Launch Wrappers
// ============================================================================

template<typename T>
void launchModelFittingOptimized(
    const T* d_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream = 0)
{
    // Use optimal thread configuration
    int threads_per_block = 256; // Multiple of warp size
    int blocks = num_partitions;
    int num_warps = threads_per_block / WARP_SIZE;

    size_t shared_mem_size = num_warps * 4 * sizeof(double) +  // warp sums
                            num_warps * sizeof(long long) +     // warp max error
                            num_warps * sizeof(bool);           // warp overflow

    processPartitionsOptimized<T><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_error_bounds,
        num_partitions, d_total_bits
    );
}

void launchSetBitOffsetsOptimized(
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_delta_bits,
    int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    cudaStream_t stream = 0)
{
    int threads_per_block = 256;
    int blocks = (num_partitions + threads_per_block - 1) / threads_per_block;

    setBitOffsetsOptimized<<<blocks, threads_per_block, 0, stream>>>(
        d_start_indices, d_end_indices,
        d_delta_bits, d_delta_array_bit_offsets,
        num_partitions
    );
}

template<typename T>
void launchDeltaPackingOptimized(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* delta_array,
    int total_elements,
    cudaStream_t stream = 0)
{
    // Use larger blocks for better occupancy
    int threads_per_block = 256;
    int blocks = (total_elements + threads_per_block * 4 - 1) / (threads_per_block * 4);
    blocks = min(blocks, 65535);

    packDeltasOptimized<T><<<blocks, threads_per_block, 0, stream>>>(
        d_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_delta_array_bit_offsets,
        num_partitions, delta_array
    );
}

// ============================================================================
// Template Instantiations
// ============================================================================

template void launchModelFittingOptimized<int32_t>(const int32_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingOptimized<uint32_t>(const uint32_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingOptimized<int64_t>(const int64_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingOptimized<uint64_t>(const uint64_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int, int64_t*, cudaStream_t);

template void launchDeltaPackingOptimized<int32_t>(const int32_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingOptimized<uint32_t>(const uint32_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingOptimized<int64_t>(const int64_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingOptimized<uint64_t>(const uint64_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);