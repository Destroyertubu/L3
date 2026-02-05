/**
 * L3 Optimized Decompression Kernels
 *
 * Optimization Strategy:
 * 1. Template specialization for common bit widths (8, 16)
 * 2. Warp-cooperative loading with vectorized reads
 * 3. Shared memory parameter broadcasting
 * 4. Register tiling for higher ILP
 * 5. Reduced divergence through compile-time branching
 *
 * INVARIANT: Maintains bit-exact compatibility with baseline kernels
 */

#ifndef PHASE2_USE_CP_ASYNC
#define PHASE2_USE_CP_ASYNC 1
#endif
#ifndef PHASE2_PERSISTENT_THREADS
#define PHASE2_PERSISTENT_THREADS 1
#endif
#ifndef PHASE2_VEC_LOADS
#define PHASE2_VEC_LOADS 1
#endif
#ifndef PHASE2_AUTOTUNE
#define PHASE2_AUTOTUNE 1
#endif



#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// OPTIMIZED KERNEL: 8-bit specialized decode
// ============================================================================
// Key optimizations for 8-bit:
// - 4 deltas fit in one 32-bit word (vectorized extraction)
// - Each warp processes 128 deltas per iteration (4 per thread)
// - Coalesced uint4 loads (16 bytes = 128 bits = 16 deltas)
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_8bit_Opt(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int num_partitions,
    const int total_elements,
    T* __restrict__ output)
{
    // Shared memory for partition metadata (broadcast once)
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    } meta;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    // Thread 0 loads metadata into shared memory (broadcast)
    if (threadIdx.x == 0) {
        meta.start_idx = d_start_indices[partition_idx];
        meta.partition_len = d_end_indices[partition_idx] - meta.start_idx;
        meta.model_type = d_model_types[partition_idx];
        meta.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        meta.theta0 = d_model_params[params_idx];
        meta.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    // Warp-level processing
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // Each thread processes 4 consecutive deltas (ILP via register tiling)
    constexpr int DELTAS_PER_THREAD = 4;
    constexpr int DELTAS_PER_WARP = 32 * DELTAS_PER_THREAD;  // 128

    // Process in warp-strided chunks
    for (int chunk_start = warp_id * DELTAS_PER_WARP;
         chunk_start < meta.partition_len;
         chunk_start += num_warps * DELTAS_PER_WARP)
    {
        // Each thread loads one 32-bit word containing 4 packed 8-bit deltas
        const int local_idx = chunk_start + lane_id * DELTAS_PER_THREAD;

        if (local_idx >= meta.partition_len) break;

        // Calculate bit offset for this thread's data
        const int64_t bit_offset = meta.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) << 3);  // * 8
        const int word_idx = bit_offset >> 5;  // / 32
        const int bit_in_word = bit_offset & 31;

        // Load packed deltas (may need 2 words if unaligned)
        uint32_t packed;
        if (bit_in_word == 0) {
            // Aligned: single word load
            packed = __ldg(&delta_array[word_idx]);
        } else {
            // Unaligned: combine two words
            const uint32_t w0 = __ldg(&delta_array[word_idx]);
            const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
            const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
            packed = static_cast<uint32_t>(combined >> bit_in_word);
        }

        // Extract 4 deltas from packed word
        int32_t delta[DELTAS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const uint32_t extracted = (packed >> (i * 8)) & 0xFF;
            // Sign extend from 8 bits
            delta[i] = static_cast<int32_t>(static_cast<int8_t>(extracted));
        }

        // Compute and write outputs
        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_idx = local_idx + i;
            if (elem_idx >= meta.partition_len) break;

            const int global_idx = meta.start_idx + elem_idx;
            if (global_idx >= total_elements) break;

            T final_value;

            if (meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(delta[i]);
            } else {
                // Linear prediction
                const double predicted = fma(meta.theta1,
                                            static_cast<double>(elem_idx),
                                            meta.theta0);
                const T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                if constexpr (std::is_signed<T>::value) {
                    final_value = predicted_T + static_cast<T>(delta[i]);
                } else {
                    final_value = static_cast<T>(
                        static_cast<int64_t>(predicted_T) + delta[i]
                    );
                }
            }

            // Coalesced write
            output[global_idx] = final_value;
        }
    }
}

// ============================================================================
// OPTIMIZED KERNEL: 16-bit specialized decode
// ============================================================================
// Key optimizations for 16-bit:
// - 2 deltas fit in one 32-bit word
// - Each thread processes 2 deltas
// - Warp processes 64 deltas per iteration
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_16bit_Opt(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int num_partitions,
    const int total_elements,
    T* __restrict__ output)
{
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    } meta;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    if (threadIdx.x == 0) {
        meta.start_idx = d_start_indices[partition_idx];
        meta.partition_len = d_end_indices[partition_idx] - meta.start_idx;
        meta.model_type = d_model_types[partition_idx];
        meta.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        meta.theta0 = d_model_params[params_idx];
        meta.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    constexpr int DELTAS_PER_THREAD = 2;
    constexpr int DELTAS_PER_WARP = 32 * DELTAS_PER_THREAD;  // 64

    for (int chunk_start = warp_id * DELTAS_PER_WARP;
         chunk_start < meta.partition_len;
         chunk_start += num_warps * DELTAS_PER_WARP)
    {
        const int local_idx = chunk_start + lane_id * DELTAS_PER_THREAD;

        if (local_idx >= meta.partition_len) break;

        const int64_t bit_offset = meta.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) << 4);  // * 16
        const int word_idx = bit_offset >> 5;
        const int bit_in_word = bit_offset & 31;

        // Load packed deltas
        uint32_t packed;
        if (bit_in_word == 0) {
            packed = __ldg(&delta_array[word_idx]);
        } else {
            const uint32_t w0 = __ldg(&delta_array[word_idx]);
            const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
            const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
            packed = static_cast<uint32_t>(combined >> bit_in_word);
        }

        // Extract 2 deltas
        int32_t delta[DELTAS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const uint32_t extracted = (packed >> (i * 16)) & 0xFFFF;
            // Sign extend from 16 bits
            delta[i] = static_cast<int32_t>(static_cast<int16_t>(extracted));
        }

        // Compute and write outputs
        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_idx = local_idx + i;
            if (elem_idx >= meta.partition_len) break;

            const int global_idx = meta.start_idx + elem_idx;
            if (global_idx >= total_elements) break;

            T final_value;

            if (meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(delta[i]);
            } else {
                const double predicted = fma(meta.theta1,
                                            static_cast<double>(elem_idx),
                                            meta.theta0);
                const T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                if constexpr (std::is_signed<T>::value) {
                    final_value = predicted_T + static_cast<T>(delta[i]);
                } else {
                    final_value = static_cast<T>(
                        static_cast<int64_t>(predicted_T) + delta[i]
                    );
                }
            }

            output[global_idx] = final_value;
        }
    }
}

// ============================================================================
// GENERIC KERNEL: Per-partition delta_bits (fallback for non-8/16-bit)
// ============================================================================
// This kernel reads the actual delta_bits from each partition's metadata
// and handles arbitrary bit widths.
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_Generic_Opt(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int num_partitions,
    const int total_elements,
    T* __restrict__ output)
{
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        int32_t delta_bits;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    } meta;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    // Thread 0 loads metadata
    if (threadIdx.x == 0) {
        meta.start_idx = d_start_indices[partition_idx];
        meta.partition_len = d_end_indices[partition_idx] - meta.start_idx;
        meta.model_type = d_model_types[partition_idx];
        meta.delta_bits = d_delta_bits[partition_idx];  // Per-partition bits!
        meta.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        meta.theta0 = d_model_params[params_idx];
        meta.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    // Handle zero delta_bits case (perfect prediction)
    if (meta.delta_bits == 0) {
        for (int local_idx = threadIdx.x; local_idx < meta.partition_len; local_idx += blockDim.x) {
            const int global_idx = meta.start_idx + local_idx;
            if (global_idx >= total_elements) continue;

            T final_value;
            if (meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                const double predicted = fma(meta.theta1, static_cast<double>(local_idx), meta.theta0);
                final_value = static_cast<T>(__double2ll_rn(predicted));
            }
            output[global_idx] = final_value;
        }
        return;
    }

    // Grid-stride loop for variable bit widths
    for (int local_idx = threadIdx.x; local_idx < meta.partition_len; local_idx += blockDim.x) {
        const int global_idx = meta.start_idx + local_idx;
        if (global_idx >= total_elements) continue;

        // Calculate bit offset for this element
        const int64_t bit_offset = meta.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) * meta.delta_bits);
        const int word_idx = bit_offset >> 5;
        const int bit_in_word = bit_offset & 31;

        T final_value;

        // Handle 64-bit types with deltas > 32 bits
        if constexpr (sizeof(T) == 8) {
            if (meta.delta_bits > 32) {
                // Multi-word extraction for 64-bit deltas
                uint64_t val64 = 0;
                int bits_remaining = meta.delta_bits;
                int current_word_idx = word_idx;
                int current_bit_offset = bit_in_word;
                int shift_amount = 0;

                while (bits_remaining > 0) {
                    int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                    uint32_t word = __ldg(&delta_array[current_word_idx]);
                    uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                    uint32_t extracted_word = (word >> current_bit_offset) & mask;

                    val64 |= (static_cast<uint64_t>(extracted_word) << shift_amount);

                    shift_amount += bits_in_this_word;
                    bits_remaining -= bits_in_this_word;
                    current_word_idx++;
                    current_bit_offset = 0;
                }

                if (meta.model_type == MODEL_DIRECT_COPY) {
                    final_value = static_cast<T>(val64);
                } else {
                    // Sign extend the extracted delta
                    int64_t delta64 = signExtend64(val64, meta.delta_bits);
                    const double predicted = fma(meta.theta1, static_cast<double>(local_idx), meta.theta0);
                    const T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                    if constexpr (std::is_signed<T>::value) {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    } else {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    }
                }

                output[global_idx] = final_value;
                continue;
            }
        }

        // Standard path for <= 32-bit deltas
        // Extract delta (handles cross-word boundaries)
        uint32_t extracted;
        if (bit_in_word + meta.delta_bits <= 32) {
            // Fits in single word
            const uint32_t w0 = __ldg(&delta_array[word_idx]);
            const uint32_t mask = (meta.delta_bits == 32) ? 0xFFFFFFFFU : ((1U << meta.delta_bits) - 1U);
            extracted = (w0 >> bit_in_word) & mask;
        } else {
            // Crosses word boundary
            const uint32_t w0 = __ldg(&delta_array[word_idx]);
            const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
            const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
            const uint32_t mask = (meta.delta_bits == 32) ? 0xFFFFFFFFU : ((1U << meta.delta_bits) - 1U);
            extracted = static_cast<uint32_t>((combined >> bit_in_word) & mask);
        }

        // Sign extend
        int32_t delta;
        if (meta.delta_bits < 32) {
            const uint32_t sign_bit = extracted >> (meta.delta_bits - 1);
            const uint32_t sign_mask = -sign_bit;
            const uint32_t extend_mask = ~((1U << meta.delta_bits) - 1U);
            delta = static_cast<int32_t>(extracted | (sign_mask & extend_mask));
        } else {
            delta = static_cast<int32_t>(extracted);
        }

        if (meta.model_type == MODEL_DIRECT_COPY) {
            // Direct copy: use raw extracted value for unsigned, sign-extended for signed
            if constexpr (std::is_signed<T>::value) {
                final_value = static_cast<T>(delta);
            } else {
                final_value = static_cast<T>(extracted);
            }
        } else {
            const double predicted = fma(meta.theta1,
                                        static_cast<double>(local_idx),
                                        meta.theta0);
            const T predicted_T = static_cast<T>(round(predicted));

            if constexpr (std::is_signed<T>::value) {
                final_value = predicted_T + static_cast<T>(delta);
            } else {
                final_value = static_cast<T>(
                    static_cast<int64_t>(predicted_T) + delta
                );
            }
        }

        output[global_idx] = final_value;
    }
}

// ============================================================================
// Dispatcher function
// ============================================================================

template<typename T>
void decompressL3_Optimized(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output,
    int avg_delta_bits)
{
    dim3 block(256);
    dim3 grid(num_partitions);

    // Always use generic kernel that reads per-partition delta_bits
    // This handles mixed bit-width data correctly
    // Specialized 8-bit and 16-bit kernels only work when ALL partitions have the same bit width
    decompressPartitions_Generic_Opt<<<grid, block>>>(
        d_start_indices, d_end_indices, d_model_types, d_model_params,
        d_delta_bits, d_delta_array_bit_offsets, delta_array,
        num_partitions, total_elements, output
    );
}

// Explicit instantiations
template void decompressL3_Optimized<uint32_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, uint32_t*, int);

template void decompressL3_Optimized<uint64_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, uint64_t*, int);

template void decompressL3_Optimized<int32_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, int32_t*, int);

template void decompressL3_Optimized<int64_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, int64_t*, int);
