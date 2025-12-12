#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include "L3.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ModelType enum is defined in L3_format.hpp
// CompressedDataOpt is defined in L3.h

// Partition metadata optimized for GPU (coalesced access)
// Extended to support all polynomial models and FOR+BitPack
struct PartitionMeta {
    int32_t start_idx;
    int32_t partition_len;
    int32_t model_type;
    int32_t delta_bits;
    double theta0;   // Linear: intercept, FOR: base value
    double theta1;   // Linear: slope
    double theta2;   // Polynomial2: quadratic coefficient
    double theta3;   // Polynomial3: cubic coefficient
    int64_t bit_offset_base;
};

// Apply delta to predicted value
template<typename T>
__device__ __forceinline__ T applyDelta(T predicted, int64_t delta) {
    if constexpr (std::is_signed<T>::value) {
        return predicted + static_cast<T>(delta);
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

/**
 * Compute prediction using polynomial model (Horner's method)
 * Supports: CONSTANT, LINEAR, POLYNOMIAL2, POLYNOMIAL3, FOR_BITPACK
 * Uses __double2ll_rn() for banker's rounding - matches V2 partitioner and encoder
 */
template<typename T>
__device__ __forceinline__
T computePrediction(int32_t model_type, double theta0, double theta1,
                    double theta2, double theta3, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:  // 0
            predicted = theta0;
            break;
        case MODEL_LINEAR:    // 1
            predicted = theta0 + theta1 * x;
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = theta0 + x * (theta1 + x * theta2);
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
            break;
        case MODEL_FOR_BITPACK:  // 4 - FOR: base stored in theta0
            // For FOR+BitPack, return base value directly
            // Delta will be added as unsigned offset
            // 64-bit: base stored as bit pattern via __longlong_as_double in encoder
            // 32-bit: base stored as regular double conversion
            if constexpr (sizeof(T) == 8) {
                return static_cast<T>(__double_as_longlong(theta0));
            } else {
                return static_cast<T>(__double2ll_rn(theta0));
            }
        default:
            // Fallback to linear for unknown types
            predicted = theta0 + theta1 * x;
            break;
    }

    // Use __double2ll_rn for banker's rounding - matches V2 partitioner and encoder
    return static_cast<T>(__double2ll_rn(predicted));
}

// ============================================================================
// OPTIMIZED KERNEL: Warp-cooperative decompression with shared memory staging
// ============================================================================
// Key optimizations:
// 1. Warp loads delta chunks cooperatively into shared memory
// 2. Reduced divergence via early model type specialization
// 3. Vectorized/coalesced memory access
// 4. Register tiling: process multiple elements per thread
// 5. Prefetching of next partition's data
// ============================================================================

// Shared memory buffer size: 256 words + 2 extra for cross-word boundary reads
// This ensures extractBitsFromShared can safely read word_idx+1 even at buffer end
constexpr int SMEM_BUFFER_WORDS = 258;  // 256 + 2 = 1032 bytes
constexpr int SMEM_USABLE_WORDS = 256;  // Actual words for data
constexpr int ELEMENTS_PER_WARP_LOAD = (SMEM_USABLE_WORDS * 32) / 8;  // Approx elements for 8-bit deltas

template<typename T>
__global__ void __launch_bounds__(256, 4)  // Tune for occupancy
decompressPartitionsOptimized(
    const CompressedDataOpt<T> compressed_data,
    T* __restrict__ output)
{
    // Shared memory for metadata and delta staging
    __shared__ PartitionMeta s_meta;
    __shared__ uint32_t s_delta_buffer[SMEM_BUFFER_WORDS];

    const int partition_idx = blockIdx.x;

    if (partition_idx >= compressed_data.num_partitions) {
        return;
    }

    // Load partition metadata (thread 0 only)
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed_data.d_start_indices[partition_idx];
        s_meta.partition_len = compressed_data.d_end_indices[partition_idx] -
                               compressed_data.d_start_indices[partition_idx];
        s_meta.model_type = compressed_data.d_model_types[partition_idx];
        s_meta.delta_bits = compressed_data.d_delta_bits[partition_idx];
        s_meta.bit_offset_base = compressed_data.d_delta_array_bit_offsets[partition_idx];

        // Load all 4 model parameters (supports POLYNOMIAL2/3 and FOR_BITPACK)
        const int params_idx = partition_idx * 4;
        s_meta.theta0 = compressed_data.d_model_params[params_idx];
        s_meta.theta1 = compressed_data.d_model_params[params_idx + 1];
        s_meta.theta2 = compressed_data.d_model_params[params_idx + 2];
        s_meta.theta3 = compressed_data.d_model_params[params_idx + 3];
    }

    __syncthreads();

    // Fast path: pre-unpacked deltas (highest throughput)
    if (compressed_data.d_plain_deltas != nullptr) {
        for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
            const int global_idx = s_meta.start_idx + local_idx;

            if (global_idx >= compressed_data.total_elements) continue;

            T final_value;

            if (s_meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(compressed_data.d_plain_deltas[global_idx]);
            } else {
                // Use computePrediction for all model types (LINEAR, POLY2, POLY3, FOR)
                const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                    s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                const int64_t delta = compressed_data.d_plain_deltas[global_idx];
                final_value = applyDelta(predicted_T, delta);
            }

            output[global_idx] = final_value;
        }
        return;
    }

    // Standard path: bit-packed deltas with warp-cooperative unpacking
    if (s_meta.delta_bits <= 0 || compressed_data.delta_array == nullptr) {
        // No deltas: just write predicted values
        for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
            const int global_idx = s_meta.start_idx + local_idx;
            if (global_idx >= compressed_data.total_elements) continue;

            T final_value;
            if (s_meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                // Use computePrediction for all model types
                final_value = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                    s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
            }

            output[global_idx] = final_value;
        }
        return;
    }

    // Warp-cooperative processing of bit-packed deltas
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Process partition in chunks that fit in shared memory
    // All warps cooperate on the same chunk, then move to next chunk
    const int max_elements_per_chunk = (SMEM_USABLE_WORDS * 32) / max(s_meta.delta_bits, 1);

    for (int chunk_start = 0;
         chunk_start < s_meta.partition_len;
         chunk_start += max_elements_per_chunk)
    {
        const int chunk_size = min(max_elements_per_chunk, s_meta.partition_len - chunk_start);

        if (chunk_size <= 0) break;

        // Calculate how many words to load
        const int words_needed = computeWordsNeeded(chunk_size, s_meta.delta_bits);
        const int words_to_load = min(words_needed + 1, SMEM_BUFFER_WORDS);  // +1 for cross-word reads

        // All threads cooperate to load data into shared memory
        const int64_t chunk_bit_offset = s_meta.bit_offset_base +
                                        (static_cast<int64_t>(chunk_start) * s_meta.delta_bits);
        const int64_t chunk_word_offset = chunk_bit_offset >> 5;
        const int local_bit_offset_base = chunk_bit_offset & 31;

        // Cooperative load using all threads in the block
        for (int w = threadIdx.x; w < words_to_load; w += blockDim.x) {
            s_delta_buffer[w] = __ldg(&compressed_data.delta_array[chunk_word_offset + w]);
        }

        __syncthreads();  // Ensure shared memory is populated

        // All warps process elements from the same chunk
        // Each thread handles multiple elements with block-strided access
        for (int i = threadIdx.x; i < chunk_size; i += blockDim.x) {
            const int local_idx = chunk_start + i;
            const int global_idx = s_meta.start_idx + local_idx;

            if (global_idx >= compressed_data.total_elements) continue;

            // Extract delta from shared memory
            const int local_bit_offset = local_bit_offset_base + i * s_meta.delta_bits;

            T final_value;

            // Handle 64-bit types with deltas > 32 bits
            if constexpr (sizeof(T) == 8) {
                if (s_meta.delta_bits > 32) {
                    // Multi-word extraction for 64-bit deltas
                    const int word_idx = local_bit_offset >> 5;
                    const int bit_in_word = local_bit_offset & 31;

                    uint64_t val64 = 0;
                    int bits_remaining = s_meta.delta_bits;
                    int current_word_idx = word_idx;
                    int current_bit_offset = bit_in_word;
                    int shift_amount = 0;

                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                        uint32_t word = s_delta_buffer[current_word_idx];
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t extracted_word = (word >> current_bit_offset) & mask;

                        val64 |= (static_cast<uint64_t>(extracted_word) << shift_amount);

                        shift_amount += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        current_word_idx++;
                        current_bit_offset = 0;
                    }

                    if (s_meta.model_type == MODEL_DIRECT_COPY) {
                        final_value = static_cast<T>(val64);
                    } else if (s_meta.model_type == MODEL_FOR_BITPACK) {
                        // FOR+BitPack: base + unsigned_delta
                        const T base = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = base + static_cast<T>(val64);  // Delta is unsigned
                    } else {
                        // Sign extend and apply model using computePrediction
                        int64_t delta64 = signExtend64(val64, s_meta.delta_bits);
                        const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = applyDelta(predicted_T, delta64);
                    }
                } else {
                    // Standard 32-bit path for 64-bit types
                    const uint32_t extracted = extractBitsFromShared(s_delta_buffer, local_bit_offset, s_meta.delta_bits);

                    if (s_meta.model_type == MODEL_DIRECT_COPY) {
                        final_value = static_cast<T>(extracted);
                    } else if (s_meta.model_type == MODEL_FOR_BITPACK) {
                        // FOR+BitPack: base + unsigned_delta
                        const T base = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = base + static_cast<T>(extracted);  // Delta is unsigned
                    } else {
                        const int32_t delta = signExtend(extracted, s_meta.delta_bits);
                        const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = applyDelta(predicted_T, static_cast<int64_t>(delta));
                    }
                }
            } else {
                // Path for 32-bit types
                // Note: delta_bits can exceed 32 even for 32-bit types when deltas span a large range
                if (s_meta.delta_bits > 32) {
                    // Multi-word extraction for deltas > 32 bits
                    const int word_idx = local_bit_offset >> 5;
                    const int bit_in_word = local_bit_offset & 31;

                    uint64_t val64 = 0;
                    int bits_remaining = s_meta.delta_bits;
                    int current_word_idx = word_idx;
                    int current_bit_offset = bit_in_word;
                    int shift_amount = 0;

                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                        uint32_t word = s_delta_buffer[current_word_idx];
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t extracted_word = (word >> current_bit_offset) & mask;

                        val64 |= (static_cast<uint64_t>(extracted_word) << shift_amount);

                        shift_amount += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        current_word_idx++;
                        current_bit_offset = 0;
                    }

                    if (s_meta.model_type == MODEL_DIRECT_COPY) {
                        final_value = static_cast<T>(val64);
                    } else if (s_meta.model_type == MODEL_FOR_BITPACK) {
                        const T base = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = base + static_cast<T>(val64);
                    } else {
                        // Sign extend and apply model
                        int64_t delta64 = signExtend64(val64, s_meta.delta_bits);
                        const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = applyDelta(predicted_T, delta64);
                    }
                } else {
                    // Standard path for delta_bits <= 32
                    const uint32_t extracted = extractBitsFromShared(s_delta_buffer, local_bit_offset, s_meta.delta_bits);

                    if (s_meta.model_type == MODEL_DIRECT_COPY) {
                        // Direct copy: extracted value is the final value
                        // For signed types, we need sign extension
                        if constexpr (std::is_signed<T>::value) {
                            const int32_t signed_val = signExtend(extracted, s_meta.delta_bits);
                            final_value = static_cast<T>(signed_val);
                        } else {
                            final_value = static_cast<T>(extracted);
                        }
                    } else if (s_meta.model_type == MODEL_FOR_BITPACK) {
                        // FOR+BitPack: base + unsigned_delta
                        const T base = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = base + static_cast<T>(extracted);  // Delta is unsigned
                    } else {
                        // Model-based: apply delta to prediction using computePrediction
                        const int32_t delta = signExtend(extracted, s_meta.delta_bits);
                        const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                            s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                        final_value = applyDelta(predicted_T, static_cast<int64_t>(delta));
                    }
                }
            }

            // Coalesced write to global memory
            output[global_idx] = final_value;
        }

        __syncthreads();  // Required: ensure all threads finish reading before next chunk overwrites buffer
    }
}

// ============================================================================
// ALTERNATIVE KERNEL: Branchless template specialization for model types
// ============================================================================
// This version reduces divergence by creating separate code paths per model type

template<typename T, int MODEL_TYPE>
__global__ void __launch_bounds__(256, 4)
decompressPartitionsSpecialized(
    const CompressedDataOpt<T> compressed_data,
    T* __restrict__ output)
{
    __shared__ PartitionMeta s_meta;
    __shared__ uint32_t s_delta_buffer[SMEM_BUFFER_WORDS];

    const int partition_idx = blockIdx.x;

    if (partition_idx >= compressed_data.num_partitions) {
        return;
    }

    // Load metadata
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed_data.d_start_indices[partition_idx];
        s_meta.partition_len = compressed_data.d_end_indices[partition_idx] -
                               compressed_data.d_start_indices[partition_idx];
        s_meta.delta_bits = compressed_data.d_delta_bits[partition_idx];
        s_meta.bit_offset_base = compressed_data.d_delta_array_bit_offsets[partition_idx];

        // Load all 4 model parameters
        const int params_idx = partition_idx * 4;
        s_meta.theta0 = compressed_data.d_model_params[params_idx];
        s_meta.theta1 = compressed_data.d_model_params[params_idx + 1];
        s_meta.theta2 = compressed_data.d_model_params[params_idx + 2];
        s_meta.theta3 = compressed_data.d_model_params[params_idx + 3];
        s_meta.model_type = compressed_data.d_model_types[partition_idx];
    }

    __syncthreads();

    // Grid-stride loop with specialization
    for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
        const int global_idx = s_meta.start_idx + local_idx;

        if (global_idx >= compressed_data.total_elements) continue;

        uint32_t extracted = 0;

        if (s_meta.delta_bits > 0 && compressed_data.delta_array != nullptr) {
            const int64_t bit_offset = s_meta.bit_offset_base +
                                      (static_cast<int64_t>(local_idx) * s_meta.delta_bits);
            extracted = extractDeltaDirect(compressed_data.delta_array, bit_offset, s_meta.delta_bits);
        }

        T final_value;

        if constexpr (MODEL_TYPE == MODEL_DIRECT_COPY) {
            final_value = static_cast<T>(extracted);
        } else {
            // Check for FOR_BITPACK at runtime
            if (s_meta.model_type == MODEL_FOR_BITPACK) {
                // FOR+BitPack: base + unsigned_delta
                const T base = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                    s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                final_value = base + static_cast<T>(extracted);  // Delta is unsigned
            } else {
                // Other models: prediction + signed delta
                const int32_t delta = signExtend(extracted, s_meta.delta_bits);
                const T predicted_T = computePrediction<T>(s_meta.model_type, s_meta.theta0,
                    s_meta.theta1, s_meta.theta2, s_meta.theta3, local_idx);
                final_value = applyDelta(predicted_T, static_cast<int64_t>(delta));
            }
        }

        output[global_idx] = final_value;
    }
}

// ============================================================================
// HOST API FUNCTIONS
// ============================================================================

template<typename T>
cudaError_t launchDecompressOptimized(
    const CompressedDataOpt<T>& compressed_data,
    T* output,
    cudaStream_t stream)
{
    if (compressed_data.num_partitions == 0) {
        return cudaSuccess;
    }

    const int num_blocks = compressed_data.num_partitions;
    const int threads_per_block = 256;

    decompressPartitionsOptimized<T><<<num_blocks, threads_per_block, 0, stream>>>(
        compressed_data,
        output
    );

    return cudaGetLastError();
}

// Explicit instantiations for common types
template cudaError_t launchDecompressOptimized<int32_t>(
    const CompressedDataOpt<int32_t>&, int32_t*, cudaStream_t);
template cudaError_t launchDecompressOptimized<int64_t>(
    const CompressedDataOpt<int64_t>&, int64_t*, cudaStream_t);
template cudaError_t launchDecompressOptimized<uint32_t>(
    const CompressedDataOpt<uint32_t>&, uint32_t*, cudaStream_t);
template cudaError_t launchDecompressOptimized<uint64_t>(
    const CompressedDataOpt<uint64_t>&, uint64_t*, cudaStream_t);

// ============================================================================
// Wrapper for CompressedDataL3 format (L3_codec.hpp compatibility)
// ============================================================================

/**
 * Wrapper to decompress using CompressedDataL3 format
 *
 * This function converts CompressedDataL3 to CompressedDataOpt and calls
 * the optimized decompression kernel.
 */
template<typename T>
void launchDecompressOptimized(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream)
{
    // Convert CompressedDataL3 to CompressedDataOpt
    CompressedDataOpt<T> opt_format;
    opt_format.d_start_indices = compressed->d_start_indices;
    opt_format.d_end_indices = compressed->d_end_indices;
    opt_format.d_model_types = compressed->d_model_types;
    opt_format.d_model_params = compressed->d_model_params;
    opt_format.d_delta_bits = compressed->d_delta_bits;
    opt_format.d_delta_array_bit_offsets = compressed->d_delta_array_bit_offsets;
    opt_format.delta_array = compressed->delta_array;
    opt_format.d_plain_deltas = compressed->d_plain_deltas;
    opt_format.num_partitions = compressed->num_partitions;
    opt_format.total_elements = compressed->total_values;

    // Call the optimized decompression kernel
    launchDecompressOptimized(opt_format, d_output, stream);
}

// Explicit instantiations for CompressedDataL3 wrapper
template void launchDecompressOptimized<int32_t>(
    const CompressedDataL3<int32_t>*, int32_t*, cudaStream_t);
template void launchDecompressOptimized<uint32_t>(
    const CompressedDataL3<uint32_t>*, uint32_t*, cudaStream_t);
template void launchDecompressOptimized<int64_t>(
    const CompressedDataL3<int64_t>*, int64_t*, cudaStream_t);
template void launchDecompressOptimized<uint64_t>(
    const CompressedDataL3<uint64_t>*, uint64_t*, cudaStream_t);

// ============================================================================
// SIMPLE DECODER (for debugging/comparison with original)
// ============================================================================

/**
 * Simple grid-stride loop decoder, matching original L32.cu algorithm
 * This is NOT optimized, but should be bit-for-bit identical to encoder
 *
 * Uses CompressedDataOpt format passed by value (device-friendly)
 */
template<typename T>
__global__ void decompressPartitionsSimple(
    const CompressedDataOpt<T> compressed,
    T* __restrict__ output)
{
    __shared__ int32_t s_start;
    __shared__ int32_t s_len;
    __shared__ int32_t s_model_type;
    __shared__ int32_t s_delta_bits;
    __shared__ int64_t s_bit_offset_base;
    __shared__ double s_theta0, s_theta1, s_theta2, s_theta3;

    const int partition_idx = blockIdx.x;

    if (partition_idx >= compressed.num_partitions) return;

    // Load metadata
    if (threadIdx.x == 0) {
        s_start = compressed.d_start_indices[partition_idx];
        s_len = compressed.d_end_indices[partition_idx] - s_start;
        s_model_type = compressed.d_model_types[partition_idx];
        s_delta_bits = compressed.d_delta_bits[partition_idx];
        s_bit_offset_base = compressed.d_delta_array_bit_offsets[partition_idx];
        // Load all 4 model parameters
        const int params_idx = partition_idx * 4;
        s_theta0 = compressed.d_model_params[params_idx];
        s_theta1 = compressed.d_model_params[params_idx + 1];
        s_theta2 = compressed.d_model_params[params_idx + 2];
        s_theta3 = compressed.d_model_params[params_idx + 3];
    }
    __syncthreads();

    // Grid-stride loop (matching original)
    for (int local_idx = threadIdx.x; local_idx < s_len; local_idx += blockDim.x) {
        int global_idx = s_start + local_idx;
        if (global_idx >= compressed.total_elements) continue;

        T final_value;

        if (s_model_type == MODEL_DIRECT_COPY) {
            // Direct copy: extract full value
            if (s_delta_bits > 0) {
                int64_t bit_offset = s_bit_offset_base + (int64_t)local_idx * s_delta_bits;

                // Handle >32-bit values (for uint64_t) - must match encoder.cu:396-406
                if (s_delta_bits > 32 && sizeof(T) == 8) {
                    // Multi-word extraction for >32-bit deltas
                    const int word_idx = bit_offset >> 5;
                    const int bit_in_word = bit_offset & 31;

                    uint64_t val64 = 0;
                    int bits_remaining = s_delta_bits;
                    int current_word_idx = word_idx;
                    int current_bit_offset = bit_in_word;
                    int shift_amount = 0;

                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                        uint32_t word = compressed.delta_array[current_word_idx];
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t extracted = (word >> current_bit_offset) & mask;

                        val64 |= (static_cast<uint64_t>(extracted) << shift_amount);

                        shift_amount += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        current_word_idx++;
                        current_bit_offset = 0;
                    }
                    final_value = static_cast<T>(val64);
                } else {
                    uint32_t extracted = extractDeltaDirect(compressed.delta_array, bit_offset, s_delta_bits);
                    final_value = static_cast<T>(extracted);
                }
            } else {
                final_value = static_cast<T>(0);
            }
        } else {
            // Model-based prediction using computePrediction for all model types
            const T pred_T = computePrediction<T>(s_model_type, s_theta0, s_theta1, s_theta2, s_theta3, local_idx);

            // Extract delta - handle both 32-bit and 64-bit cases
            int64_t delta = 0;
            if (s_delta_bits > 0) {
                int64_t bit_offset = s_bit_offset_base + (int64_t)local_idx * s_delta_bits;

                // Handle >32-bit deltas for 64-bit types (e.g., FOR_BITPACK with large range)
                if (s_delta_bits > 32 && sizeof(T) == 8) {
                    // Multi-word extraction (match encoder pattern)
                    const int word_idx = bit_offset >> 5;
                    const int bit_in_word = bit_offset & 31;

                    uint64_t result = 0;
                    int bits_remaining = s_delta_bits;
                    int current_word_idx = word_idx;
                    int current_bit_offset = bit_in_word;
                    int shift_amount = 0;

                    while (bits_remaining > 0) {
                        int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                        uint32_t word = compressed.delta_array[current_word_idx];
                        uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                        uint32_t extracted_part = (word >> current_bit_offset) & mask;

                        result |= (static_cast<uint64_t>(extracted_part) << shift_amount);

                        shift_amount += bits_in_this_word;
                        bits_remaining -= bits_in_this_word;
                        current_word_idx++;
                        current_bit_offset = 0;
                    }

                    // FOR_BITPACK uses unsigned deltas, others use signed
                    if (s_model_type == MODEL_FOR_BITPACK) {
                        delta = static_cast<int64_t>(result);  // Unsigned delta
                    } else {
                        delta = signExtend64(result, s_delta_bits);  // Signed delta
                    }
                } else {
                    // Standard 32-bit extraction
                    // NOTE: extractDeltaDirect returns int32_t (already sign-extended)
                    // For FOR_BITPACK, we need the original unsigned bits
                    int32_t raw_signed = extractDeltaDirect(compressed.delta_array, bit_offset, s_delta_bits);
                    if (s_model_type == MODEL_FOR_BITPACK) {
                        // FOR_BITPACK uses unsigned deltas - mask off sign extension bits
                        uint32_t unsigned_delta = static_cast<uint32_t>(raw_signed) & ((1U << s_delta_bits) - 1);
                        delta = static_cast<int64_t>(unsigned_delta);
                    } else {
                        // Other models: already sign-extended by extractDeltaDirect
                        delta = static_cast<int64_t>(raw_signed);
                    }
                }
            }

            // Apply delta to prediction
            if (s_model_type == MODEL_FOR_BITPACK) {
                // FOR+BitPack: base + unsigned_delta
                final_value = static_cast<T>(static_cast<int64_t>(pred_T) + delta);
            } else {
                // Other models: prediction + signed_delta
                final_value = applyDelta(pred_T, delta);
            }
        }

        output[global_idx] = final_value;
    }
}

template<typename T>
void launchDecompressSimple(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream)
{
    // Convert CompressedDataL3 to CompressedDataOpt for device access
    CompressedDataOpt<T> opt_format;
    opt_format.d_start_indices = compressed->d_start_indices;
    opt_format.d_end_indices = compressed->d_end_indices;
    opt_format.d_model_types = compressed->d_model_types;
    opt_format.d_model_params = compressed->d_model_params;
    opt_format.d_delta_bits = compressed->d_delta_bits;
    opt_format.d_delta_array_bit_offsets = compressed->d_delta_array_bit_offsets;
    opt_format.delta_array = compressed->delta_array;
    opt_format.d_plain_deltas = compressed->d_plain_deltas;
    opt_format.num_partitions = compressed->num_partitions;
    opt_format.total_elements = compressed->total_values;

    const int num_blocks = compressed->num_partitions;
    const int threads_per_block = 256;

    decompressPartitionsSimple<T><<<num_blocks, threads_per_block, 0, stream>>>(
        opt_format, d_output);
}

// Explicit instantiations for simple decoder
template void launchDecompressSimple<int32_t>(const CompressedDataL3<int32_t>*, int32_t*, cudaStream_t);
template void launchDecompressSimple<uint32_t>(const CompressedDataL3<uint32_t>*, uint32_t*, cudaStream_t);
template void launchDecompressSimple<int64_t>(const CompressedDataL3<int64_t>*, int64_t*, cudaStream_t);
template void launchDecompressSimple<uint64_t>(const CompressedDataL3<uint64_t>*, uint64_t*, cudaStream_t);
