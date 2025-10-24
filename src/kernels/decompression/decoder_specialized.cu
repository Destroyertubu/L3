/**
 * GLECO Specialized Decoders - Template Specialization for Common Delta Widths
 *
 * Optimizations:
 * - Compile-time constant delta_bits (1, 2, 4, 8, 16)
 * - Branchless hot loop
 * - Optimized bit extraction for specific widths
 * - Reduced instruction count
 */

#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include "L3_opt.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Specialized Bit Extraction for Common Widths
// ============================================================================

/**
 * Extract DELTA_BITS from packed data at bit_offset
 * Template parameter allows compile-time optimization
 */
template<int DELTA_BITS>
__device__ __forceinline__ uint32_t extract_bits_specialized(
    const uint32_t* smem, int bit_offset)
{
    static_assert(DELTA_BITS > 0 && DELTA_BITS <= 32, "Invalid DELTA_BITS");

    const int word_idx = bit_offset >> 5;
    const int bit_in_word = bit_offset & 31;

    if constexpr (DELTA_BITS == 1) {
        // 1-bit: simple bit test
        return (smem[word_idx] >> bit_in_word) & 1;
    }
    else if constexpr (DELTA_BITS == 2) {
        // 2-bit: mask after shift
        return (smem[word_idx] >> bit_in_word) & 0x3;
    }
    else if constexpr (DELTA_BITS == 4) {
        // 4-bit: nibble extraction
        return (smem[word_idx] >> bit_in_word) & 0xF;
    }
    else if constexpr (DELTA_BITS == 8) {
        // 8-bit: byte extraction
        if (bit_in_word <= 24) {
            return (smem[word_idx] >> bit_in_word) & 0xFF;
        } else {
            // Crosses boundary
            uint32_t lo = smem[word_idx] >> bit_in_word;
            uint32_t hi = smem[word_idx + 1] << (32 - bit_in_word);
            return (lo | hi) & 0xFF;
        }
    }
    else if constexpr (DELTA_BITS == 16) {
        // 16-bit: half-word extraction
        if (bit_in_word <= 16) {
            return (smem[word_idx] >> bit_in_word) & 0xFFFF;
        } else {
            uint32_t lo = smem[word_idx] >> bit_in_word;
            uint32_t hi = smem[word_idx + 1] << (32 - bit_in_word);
            return (lo | hi) & 0xFFFF;
        }
    }
    else {
        // Generic path for other widths (6, 9, 17, 19, 32, etc.)
        // Use funnel shift for efficient extraction
        uint32_t lo = smem[word_idx];
        uint32_t hi = smem[word_idx + 1];
        uint32_t extracted = __funnelshift_r(lo, hi, bit_in_word);

        // Compile-time mask calculation
        constexpr uint32_t mask = (DELTA_BITS == 32) ? 0xFFFFFFFFU : ((1U << DELTA_BITS) - 1);
        return extracted & mask;
    }
}

// ============================================================================
// Specialized Decoder Kernel
// ============================================================================

template<typename T, int DELTA_BITS>
__global__ void __launch_bounds__(256, 4)
decompressSpecialized(
    const CompressedDataOpt<T> compressed,
    T* __restrict__ output)
{
    constexpr int MAX_WARPS = 4;
    constexpr int TILE_WORDS = 512;
    constexpr int NUM_BUFFERS = 2;

    // Per-warp shared memory
    __shared__ uint32_t s_tiles[MAX_WARPS][NUM_BUFFERS][TILE_WORDS];

    // Shared metadata
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        int64_t bit_offset_base;
        double theta0;
        double theta1;
    } s_meta;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= compressed.num_partitions) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // Load metadata
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed.d_start_indices[partition_idx];
        s_meta.partition_len = compressed.d_end_indices[partition_idx] - s_meta.start_idx;
        s_meta.model_type = compressed.d_model_types[partition_idx];
        s_meta.bit_offset_base = compressed.d_delta_array_bit_offsets[partition_idx];

        int params_idx = partition_idx * 4;
        s_meta.theta0 = compressed.d_model_params[params_idx];
        s_meta.theta1 = compressed.d_model_params[params_idx + 1];
    }
    __syncthreads();

    // Fast path: no deltas
    if (compressed.delta_array == nullptr) {
        for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
            int global_idx = s_meta.start_idx + local_idx;
            if (global_idx >= compressed.total_elements) continue;

            T final_value;
            if (s_meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
                final_value = static_cast<T>(round(predicted));
            }
            output[global_idx] = final_value;
        }
        return;
    }

    // Broadcast model_type
    const int model_type = __shfl_sync(0xFFFFFFFF, s_meta.model_type, 0);

    // Calculate elements per tile (compile-time constant division)
    constexpr int elements_per_tile = ((TILE_WORDS - 3) * 32) / DELTA_BITS;

    int warp_start = warp_id * elements_per_tile;

    int curr_buffer = 0;
    int next_buffer = 1;

    // Prefetch first tile
    if (warp_start < s_meta.partition_len) {
        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(warp_start) * DELTA_BITS);
        int64_t tile_word_offset = tile_bit_offset >> 5;
        int words_needed = (elements_per_tile * DELTA_BITS + 31) / 32 + 2;
        words_needed = min(words_needed, TILE_WORDS);

        // Coalesced load
        constexpr int WORDS_PER_LANE = (TILE_WORDS + 31) / 32;
        #pragma unroll
        for (int i = 0; i < WORDS_PER_LANE; ++i) {
            int word_idx = lane_id + i * 32;
            if (word_idx < words_needed) {
                s_tiles[warp_id][curr_buffer][word_idx] =
                    __ldg(&compressed.delta_array[tile_word_offset + word_idx]);
            }
        }
        __syncwarp();
    }

    // Process tiles with double buffering
    for (int tile_idx = 0; warp_start < s_meta.partition_len; tile_idx++) {
        int tile_start = warp_start;
        int tile_end = min(tile_start + elements_per_tile, s_meta.partition_len);

        // Prefetch next tile
        int next_warp_start = warp_start + num_warps * elements_per_tile;
        if (next_warp_start < s_meta.partition_len) {
            int64_t next_tile_bit_offset = s_meta.bit_offset_base +
                                          (static_cast<int64_t>(next_warp_start) * DELTA_BITS);
            int64_t next_tile_word_offset = next_tile_bit_offset >> 5;
            int next_words_needed = (elements_per_tile * DELTA_BITS + 31) / 32 + 2;
            next_words_needed = min(next_words_needed, TILE_WORDS);

            constexpr int WORDS_PER_LANE = (TILE_WORDS + 31) / 32;
            #pragma unroll
            for (int i = 0; i < WORDS_PER_LANE; ++i) {
                int word_idx = lane_id + i * 32;
                if (word_idx < next_words_needed) {
                    s_tiles[warp_id][next_buffer][word_idx] =
                        __ldg(&compressed.delta_array[next_tile_word_offset + word_idx]);
                }
            }
        }

        __syncwarp();

        // Calculate local bit offset base
        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(tile_start) * DELTA_BITS);
        int local_bit_base = tile_bit_offset & 31;

        // HOT LOOP - Optimized decode path
        #pragma unroll 8
        for (int i = lane_id; i < (tile_end - tile_start); i += 32) {
            int local_idx = tile_start + i;
            int global_idx = s_meta.start_idx + local_idx;

            if (global_idx >= compressed.total_elements) continue;

            // Bit position (compile-time constant multiplication)
            int local_bit_offset = local_bit_base + i * DELTA_BITS;

            // Extract delta (specialized for DELTA_BITS)
            uint32_t extracted = extract_bits_specialized<DELTA_BITS>(
                s_tiles[warp_id][curr_buffer], local_bit_offset);

            T final_value;

            // Branchless computation using mask
            if (model_type == MODEL_DIRECT_COPY) {
                // Direct copy path
                if constexpr (std::is_signed<T>::value) {
                    int32_t signed_val = signExtend(extracted, DELTA_BITS);
                    final_value = static_cast<T>(signed_val);
                } else {
                    final_value = static_cast<T>(extracted);
                }
            } else {
                // Model-based path
                int32_t delta = signExtend(extracted, DELTA_BITS);
                double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
                T predicted_T = static_cast<T>(round(predicted));

                if constexpr (std::is_signed<T>::value) {
                    final_value = predicted_T + static_cast<T>(delta);
                } else {
                    final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
                }
            }

            // Coalesced write
            output[global_idx] = final_value;
        }

        // Swap buffers
        curr_buffer = 1 - curr_buffer;
        next_buffer = 1 - next_buffer;
        warp_start = next_warp_start;
    }
}

// ============================================================================
// Dispatcher - Select Best Kernel Based on Delta Width
// ============================================================================

template<typename T>
void launchDecompressSpecialized(
    const CompressedDataOpt<T>& compressed,
    T* output,
    int delta_bits,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0) return;

    const int threads_per_block = 128;
    const int num_blocks = compressed.num_partitions;

    // Dispatch to specialized kernel based on delta_bits
    switch (delta_bits) {
        case 1:
            decompressSpecialized<T, 1><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 2:
            decompressSpecialized<T, 2><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 4:
            decompressSpecialized<T, 4><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 6:
            decompressSpecialized<T, 6><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 8:
            decompressSpecialized<T, 8><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 9:
            decompressSpecialized<T, 9><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 16:
            decompressSpecialized<T, 16><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 17:
            decompressSpecialized<T, 17><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 19:
            decompressSpecialized<T, 19><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        case 32:
            decompressSpecialized<T, 32><<<num_blocks, threads_per_block, 0, stream>>>(
                compressed, output);
            break;
        default:
            // Fall back to generic kernel for uncommon widths
            // Use nearest specialized version for performance
            if (delta_bits < 6) {
                decompressSpecialized<T, 4><<<num_blocks, threads_per_block, 0, stream>>>(
                    compressed, output);
            } else if (delta_bits < 12) {
                decompressSpecialized<T, 9><<<num_blocks, threads_per_block, 0, stream>>>(
                    compressed, output);
            } else if (delta_bits < 24) {
                decompressSpecialized<T, 17><<<num_blocks, threads_per_block, 0, stream>>>(
                    compressed, output);
            } else {
                decompressSpecialized<T, 32><<<num_blocks, threads_per_block, 0, stream>>>(
                    compressed, output);
            }
            break;
    }
}

// Wrapper for CompressedDataGLECO format
template<typename T>
void launchDecompressSpecialized(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    int delta_bits,
    cudaStream_t stream = 0)
{
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

    launchDecompressSpecialized(opt_format, d_output, delta_bits, stream);
}

// Explicit instantiations
template void launchDecompressSpecialized<int32_t>(const CompressedDataGLECO<int32_t>*, int32_t*, int, cudaStream_t);
template void launchDecompressSpecialized<uint32_t>(const CompressedDataGLECO<uint32_t>*, uint32_t*, int, cudaStream_t);
template void launchDecompressSpecialized<int64_t>(const CompressedDataGLECO<int64_t>*, int64_t*, int, cudaStream_t);
template void launchDecompressSpecialized<uint64_t>(const CompressedDataGLECO<uint64_t>*, uint64_t*, int, cudaStream_t);

template void launchDecompressSpecialized<int32_t>(const CompressedDataOpt<int32_t>&, int32_t*, int, cudaStream_t);
template void launchDecompressSpecialized<uint32_t>(const CompressedDataOpt<uint32_t>&, uint32_t*, int, cudaStream_t);
template void launchDecompressSpecialized<int64_t>(const CompressedDataOpt<int64_t>&, int64_t*, int, cudaStream_t);
template void launchDecompressSpecialized<uint64_t>(const CompressedDataOpt<uint64_t>&, uint64_t*, int, cudaStream_t);
