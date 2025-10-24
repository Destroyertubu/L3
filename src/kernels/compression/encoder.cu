/**
 * GLECO Encoder Kernels
 *
 * PORTED FROM: /root/autodl-tmp/code/data/SSB/L3/L32.cu
 * PORT DATE: 2025-10-14
 * STATUS: Semantic-preserving port (no algorithmic changes)
 *
 * This file contains the GPU kernels for GLECO compression:
 * - Model fitting (linear regression)
 * - Delta computation
 * - Bit packing
 *
 * CRITICAL: These kernels produce bitstreams that MUST be decodable
 * by the decoder in decompression_kernels.cu. Any change here MUST
 * be accompanied by corresponding decoder changes.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <cmath>
#include "L3_format.hpp"

// Constants from original implementation
#define MAX_DELTA_BITS 64

// ============================================================================
// Helper Functions (Device and Host)
// ============================================================================

/**
 * Check if value might lose precision when converted to double
 * (Used to detect overflow risk for large unsigned values)
 */
template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;  // Signed types within long long range are OK
    } else {
        // For unsigned types, check if value exceeds double precision (2^53)
        const uint64_t DOUBLE_PRECISION_LIMIT = (1ULL << 53);
        return static_cast<uint64_t>(value) > DOUBLE_PRECISION_LIMIT;
    }
}

/**
 * Safe delta calculation handling signed/unsigned arithmetic
 *
 * INVARIANT: Must produce identical results to decoder's applyDelta inverse
 */
template<typename T>
__device__ __host__ inline long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        // For unsigned types
        if (sizeof(T) == 8) {
            // For 64-bit unsigned types (unsigned long long)
            unsigned long long actual_ull = static_cast<unsigned long long>(actual);
            unsigned long long pred_ull = static_cast<unsigned long long>(predicted);

            if (actual_ull >= pred_ull) {
                unsigned long long diff = actual_ull - pred_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return static_cast<long long>(diff);
                } else {
                    return LLONG_MAX;
                }
            } else {
                unsigned long long diff = pred_ull - actual_ull;
                if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                    return -static_cast<long long>(diff);
                } else {
                    return LLONG_MIN;
                }
            }
        } else {
            // For smaller unsigned types, direct conversion is safe
            return static_cast<long long>(actual) - static_cast<long long>(predicted);
        }
    }
}

// ============================================================================
// Model Fitting and Metadata Kernel
// ============================================================================

/**
 * Combined kernel for overflow checking, model fitting, and metadata computation
 *
 * PORTED FROM: L32.cu:709-881 (wprocessPartitionsKernel)
 * ENHANCED: Now stores actual partition min/max values for predicate pushdown
 *
 * Launched with: 1 block per partition, blockDim.x threads per block
 *
 * For each partition:
 * 1. Check if any values might overflow double precision
 * 2. If overflow: use MODEL_DIRECT_COPY (store full values)
 * 3. Else: fit linear model using least squares
 * 4. Compute maximum |delta| across all elements
 * 5. Track actual min/max values in the partition (NEW)
 * 6. Determine delta_bits (bits needed to represent max delta + sign)
 * 7. Store actual min/max values for tight predicate pushdown bounds (NEW)
 * 8. Update total_bits_device atomically
 *
 * Shared memory requirements:
 *   4 * blockDim.x * sizeof(double) + blockDim.x * sizeof(long long) * 3 + blockDim.x * sizeof(bool)
 *
 * NOTE: s_min_delta and s_max_delta arrays are reused to store actual min/max values
 *       (not deltas) for efficient parallel reduction.
 */
template<typename T>
__global__ void wprocessPartitionsKernel(const T* values_device,
                                       int32_t* d_start_indices,
                                       int32_t* d_end_indices,
                                       int32_t* d_model_types,
                                       double* d_model_params,
                                       int32_t* d_delta_bits,
                                       int64_t* d_error_bounds,
                                       T* d_partition_min,
                                       T* d_partition_max,
                                       int num_partitions,
                                       int64_t* total_bits_device) {
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int segment_len = end_idx - start_idx;

    if (segment_len <= 0) return;

    // Shared memory for reduction operations
    extern __shared__ char shared_mem[];
    double* s_sums = reinterpret_cast<double*>(shared_mem);
    long long* s_max_error = reinterpret_cast<long long*>(shared_mem + 4 * blockDim.x * sizeof(double));
    long long* s_min_delta = reinterpret_cast<long long*>(shared_mem + 4 * blockDim.x * sizeof(double) + blockDim.x * sizeof(long long));
    long long* s_max_delta = reinterpret_cast<long long*>(shared_mem + 4 * blockDim.x * sizeof(double) + 2 * blockDim.x * sizeof(long long));
    bool* s_overflow = reinterpret_cast<bool*>(shared_mem + 4 * blockDim.x * sizeof(double) + 3 * blockDim.x * sizeof(long long));

    int tid = threadIdx.x;

    // Phase 1: Check for overflow
    bool local_overflow = false;
    for (int i = tid; i < segment_len; i += blockDim.x) {
        if (mightOverflowDoublePrecision(values_device[start_idx + i])) {
            local_overflow = true;
            break;
        }
    }

    // Reduce overflow flag
    s_overflow[tid] = local_overflow;
    __syncthreads();

    // Simple reduction for overflow flag
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_overflow[tid] = s_overflow[tid] || s_overflow[tid + s];
        }
        __syncthreads();
    }

    bool has_overflow = s_overflow[0];

    if (tid == 0) {
        if (has_overflow) {
            // Direct copy model for overflow
            d_model_types[partition_idx] = MODEL_DIRECT_COPY;
            d_model_params[partition_idx * 4] = 0.0;
            d_model_params[partition_idx * 4 + 1] = 0.0;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
            d_error_bounds[partition_idx] = 0;
            d_delta_bits[partition_idx] = sizeof(T) * 8;

            // For direct copy, we need to scan actual values for min/max
            // This is an exceptional case for very large values
            T local_min = values_device[start_idx];
            T local_max = values_device[start_idx];
            for (int i = 0; i < segment_len; i++) {
                T val = values_device[start_idx + i];
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
            }
            d_partition_min[partition_idx] = local_min;
            d_partition_max[partition_idx] = local_max;
        }
    }
    __syncthreads();

    if (!has_overflow) {
        // Phase 2: Fit linear model
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

        for (int i = tid; i < segment_len; i += blockDim.x) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(values_device[start_idx + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        s_sums[tid] = sum_x;
        s_sums[tid + blockDim.x] = sum_y;
        s_sums[tid + 2 * blockDim.x] = sum_xx;
        s_sums[tid + 3 * blockDim.x] = sum_xy;
        __syncthreads();

        // Reduction for sums
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_sums[tid] += s_sums[tid + s];
                s_sums[tid + blockDim.x] += s_sums[tid + s + blockDim.x];
                s_sums[tid + 2 * blockDim.x] += s_sums[tid + s + 2 * blockDim.x];
                s_sums[tid + 3 * blockDim.x] += s_sums[tid + s + 3 * blockDim.x];
            }
            __syncthreads();
        }

        __shared__ double theta0, theta1;

        if (tid == 0) {
            double n = static_cast<double>(segment_len);
            double determinant = n * s_sums[2 * blockDim.x] - s_sums[0] * s_sums[0];

            if (fabs(determinant) > 1e-10) {
                theta1 = (n * s_sums[3 * blockDim.x] - s_sums[0] * s_sums[blockDim.x]) / determinant;
                theta0 = (s_sums[blockDim.x] - theta1 * s_sums[0]) / n;
            } else {
                theta1 = 0.0;
                theta0 = s_sums[blockDim.x] / n;
            }

            d_model_types[partition_idx] = MODEL_LINEAR;
            d_model_params[partition_idx * 4] = theta0;
            d_model_params[partition_idx * 4 + 1] = theta1;
            d_model_params[partition_idx * 4 + 2] = 0.0;
            d_model_params[partition_idx * 4 + 3] = 0.0;
        }
        __syncthreads();

        theta0 = d_model_params[partition_idx * 4];
        theta1 = d_model_params[partition_idx * 4 + 1];

        // Phase 3: Calculate maximum error and track actual min/max values
        long long max_error = 0;
        T local_min_value = (segment_len > 0) ? values_device[start_idx] : 0;
        T local_max_value = (segment_len > 0) ? values_device[start_idx] : 0;

        for (int i = tid; i < segment_len; i += blockDim.x) {
            T actual_value = values_device[start_idx + i];

            // Track actual min/max values
            if (actual_value < local_min_value) {
                local_min_value = actual_value;
            }
            if (actual_value > local_max_value) {
                local_max_value = actual_value;
            }

            // Calculate delta for error tracking
            double predicted = theta0 + theta1 * i;
            T pred_T = static_cast<T>(round(predicted));
            long long delta = calculateDelta(actual_value, pred_T);
            long long abs_error = (delta < 0) ? -delta : delta;

            if (abs_error > max_error) {
                max_error = abs_error;
            }
        }

        s_max_error[tid] = max_error;

        // Reuse s_min_delta and s_max_delta arrays to store actual min/max values
        // Cast T to long long for reduction (works for both signed and unsigned T up to 64-bit)
        if (std::is_signed<T>::value || sizeof(T) <= 4) {
            s_min_delta[tid] = static_cast<long long>(local_min_value);
            s_max_delta[tid] = static_cast<long long>(local_max_value);
        } else {
            // For unsigned 64-bit, use reinterpret cast
            s_min_delta[tid] = *reinterpret_cast<long long*>(&local_min_value);
            s_max_delta[tid] = *reinterpret_cast<long long*>(&local_max_value);
        }
        __syncthreads();

        // Reduction for maximum error and actual min/max values
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                // Reduce max error
                if (s_max_error[tid + s] > s_max_error[tid]) {
                    s_max_error[tid] = s_max_error[tid + s];
                }
                // Reduce min value (as long long representation)
                if (s_min_delta[tid + s] < s_min_delta[tid]) {
                    s_min_delta[tid] = s_min_delta[tid + s];
                }
                // Reduce max value (as long long representation)
                if (s_max_delta[tid + s] > s_max_delta[tid]) {
                    s_max_delta[tid] = s_max_delta[tid + s];
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            d_error_bounds[partition_idx] = s_max_error[0];

            // Calculate delta bits
            int delta_bits = 0;
            if (s_max_error[0] > 0) {
                long long max_abs_error = s_max_error[0];
                int bits_for_magnitude = 0;
                unsigned long long temp = static_cast<unsigned long long>(max_abs_error);
                while (temp > 0) {
                    bits_for_magnitude++;
                    temp >>= 1;
                }
                delta_bits = bits_for_magnitude + 1; // +1 for sign bit
                delta_bits = min(delta_bits, MAX_DELTA_BITS);
                delta_bits = max(delta_bits, 0);
            }
            d_delta_bits[partition_idx] = delta_bits;

            // Phase 4: Store actual min/max values from the partition data
            // NEW APPROACH: We directly store the real min/max values scanned from data
            // This provides tight bounds for effective predicate pushdown

            T partition_min, partition_max;

            // Convert back from long long representation to type T
            if (std::is_signed<T>::value || sizeof(T) <= 4) {
                partition_min = static_cast<T>(s_min_delta[0]);
                partition_max = static_cast<T>(s_max_delta[0]);
            } else {
                // For unsigned 64-bit, use reinterpret cast back
                long long min_as_ll = s_min_delta[0];
                long long max_as_ll = s_max_delta[0];
                partition_min = *reinterpret_cast<T*>(&min_as_ll);
                partition_max = *reinterpret_cast<T*>(&max_as_ll);
            }

            d_partition_min[partition_idx] = partition_min;
            d_partition_max[partition_idx] = partition_max;
        }
    }

    // Atomic add to total bits counter
    if (tid == 0) {
        int64_t partition_bits = (int64_t)segment_len * d_delta_bits[partition_idx];
        // Use unsigned long long atomicAdd and cast
        atomicAdd(reinterpret_cast<unsigned long long*>(total_bits_device),
                  static_cast<unsigned long long>(partition_bits));
    }
}

// ============================================================================
// Bit Offset Calculation Kernel
// ============================================================================

/**
 * Kernel to set bit offsets based on cumulative sum
 *
 * PORTED FROM: L32.cu:884-900 (setBitOffsetsKernel)
 *
 * Each thread computes the bit offset for one partition by summing
 * all previous partitions' bit usage.
 *
 * INVARIANT: bit_offset[i] = sum(k=0..i-1) { segment_length[k] * delta_bits[k] }
 */
__global__ void setBitOffsetsKernel(int32_t* d_start_indices,
                                   int32_t* d_end_indices,
                                   int32_t* d_delta_bits,
                                   int64_t* d_delta_array_bit_offsets,
                                   int num_partitions) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_partitions) return;

    // Calculate cumulative bit offset
    int64_t bit_offset = 0;
    for (int i = 0; i < tid; i++) {
        int seg_len = d_end_indices[i] - d_start_indices[i];
        bit_offset += (int64_t)seg_len * d_delta_bits[i];
    }

    d_delta_array_bit_offsets[tid] = bit_offset;
}

// ============================================================================
// Delta Packing Kernel
// ============================================================================

/**
 * Optimized delta packing kernel with direct copy support
 *
 * PORTED FROM: L32.cu:1026-1155 (packDeltasKernelOptimized)
 *
 * Grid-stride loop over all elements. Each thread:
 * 1. Binary search to find partition for its element
 * 2. Compute prediction using model parameters
 * 3. Compute delta = actual - predicted
 * 4. Pack delta into delta_array at correct bit offset using atomicOr
 *
 * CRITICAL: delta_array must be initialized to all zeros before launch
 *
 * Bit packing details:
 * - Deltas may span word boundaries (require 2 atomicOr operations)
 * - For MODEL_DIRECT_COPY, store full value instead of delta
 * - Sign information preserved via two's complement representation
 */
template<typename T>
__global__ void packDeltasKernelOptimized(const T* values_device,
                                          const int32_t* d_start_indices,
                                          const int32_t* d_end_indices,
                                          const int32_t* d_model_types,
                                          const double* d_model_params,
                                          const int32_t* d_delta_bits,
                                          const int64_t* d_delta_array_bit_offsets,
                                          int num_partitions_val,
                                          uint32_t* delta_array_device) {
    // Handle both 1D and 2D grids for large datasets
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x +
                blockIdx.y * (gridDim.x * blockDim.x);
    int g_stride = blockDim.x * gridDim.x * gridDim.y;

    if (num_partitions_val == 0) return;

    int max_idx_to_process = d_end_indices[num_partitions_val - 1];

    for (int current_idx = g_idx; current_idx < max_idx_to_process; current_idx += g_stride) {
        // Binary search for partition
        int p_left = 0, p_right = num_partitions_val - 1;
        int found_partition_idx = -1;

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

        if (found_partition_idx == -1) continue;

        // BOUNDS CHECK: Ensure partition index is valid
        if (found_partition_idx < 0 || found_partition_idx >= num_partitions_val) continue;

        // Get partition data using found index
        int32_t current_model_type = d_model_types[found_partition_idx];
        int32_t current_delta_bits = d_delta_bits[found_partition_idx];
        int64_t current_bit_offset_base = d_delta_array_bit_offsets[found_partition_idx];
        int32_t current_start_idx = d_start_indices[found_partition_idx];

        // For direct copy model, we store the full value
        if (current_model_type == MODEL_DIRECT_COPY) {
            int local_idx = current_idx - current_start_idx;
            int64_t bit_offset = current_bit_offset_base +
                                (int64_t)local_idx * current_delta_bits;

            // Store the full value as "delta"
            T value = values_device[current_idx];
            uint64_t value_to_store = static_cast<uint64_t>(value);

            // Pack the value into the delta array
            int start_word_idx = bit_offset / 32;
            int offset_in_word = bit_offset % 32;
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
        } else {
            // Normal delta encoding
            int current_local_idx = current_idx - current_start_idx;

            double pred_double = d_model_params[found_partition_idx * 4] +
                                d_model_params[found_partition_idx * 4 + 1] * current_local_idx;
            if (current_model_type == MODEL_POLYNOMIAL2) {
                pred_double += d_model_params[found_partition_idx * 4 + 2] * current_local_idx * current_local_idx;
            }

            T pred_T_val = static_cast<T>(round(pred_double));
            long long current_delta_ll = calculateDelta(values_device[current_idx], pred_T_val);

            if (current_delta_bits > 0) {
                int64_t current_bit_offset_val = current_bit_offset_base +
                                                 (int64_t)current_local_idx * current_delta_bits;

                // Handle deltas up to 64 bits
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
// Kernel Launch Wrappers (Host Functions)
// ============================================================================

/**
 * Launch model fitting kernel
 *
 * USAGE: Call this after partitioning, before bit packing
 * Now also computes partition min/max bounds from learned models + deltas
 */
template<typename T>
void launchModelFittingKernel(
    const T* d_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream = 0)
{
    int threads_per_block = 256;
    int blocks = num_partitions;

    // Updated shared memory: 4*double + 3*long long + 1*bool per thread
    size_t shared_mem_size = 4 * threads_per_block * sizeof(double) +
                             3 * threads_per_block * sizeof(long long) +
                             threads_per_block * sizeof(bool);

    wprocessPartitionsKernel<T><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_error_bounds,
        d_partition_min, d_partition_max,
        num_partitions, d_total_bits
    );
}

/**
 * Launch bit offset calculation kernel
 */
void launchSetBitOffsetsKernel(
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_delta_bits,
    int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    cudaStream_t stream = 0)
{
    int threads_per_block = 256;
    int blocks = (num_partitions + threads_per_block - 1) / threads_per_block;

    setBitOffsetsKernel<<<blocks, threads_per_block, 0, stream>>>(
        d_start_indices, d_end_indices,
        d_delta_bits, d_delta_array_bit_offsets,
        num_partitions
    );
}

/**
 * Launch delta packing kernel
 *
 * CRITICAL: delta_array must be zeroed before calling this
 */
template<typename T>
void launchDeltaPackingKernel(
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
    int threads_per_block = 256;
    int blocks_1d = (total_elements + threads_per_block - 1) / threads_per_block;

    // Use 2D grid if needed to exceed 1D limit (65535)
    dim3 blocks;
    if (blocks_1d <= 65535) {
        blocks = dim3(blocks_1d, 1, 1);
    } else {
        // Use 2D grid: blocks_x can be 65535, blocks_y allows additional blocks
        blocks.x = 65535;
        blocks.y = (blocks_1d + 65534) / 65535;  // Ceiling division
    }

    packDeltasKernelOptimized<T><<<blocks, threads_per_block, 0, stream>>>(
        d_values,
        d_start_indices, d_end_indices,
        d_model_types, d_model_params,
        d_delta_bits, d_delta_array_bit_offsets,
        num_partitions, delta_array
    );
}

// ============================================================================
// Template Instantiations (for common types)
// ============================================================================

// Explicit instantiations for linker (updated with new signature)
template void launchModelFittingKernel<int32_t>(const int32_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int32_t*, int32_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingKernel<uint32_t>(const uint32_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, uint32_t*, uint32_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingKernel<int64_t>(const int64_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, int64_t*, int64_t*, int, int64_t*, cudaStream_t);
template void launchModelFittingKernel<uint64_t>(const uint64_t*, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*, uint64_t*, uint64_t*, int, int64_t*, cudaStream_t);

template void launchDeltaPackingKernel<int32_t>(const int32_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingKernel<uint32_t>(const uint32_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingKernel<int64_t>(const int64_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
template void launchDeltaPackingKernel<uint64_t>(const uint64_t*, const int32_t*, const int32_t*, const int32_t*, const double*, const int32_t*, const int64_t*, int, uint32_t*, int, cudaStream_t);
