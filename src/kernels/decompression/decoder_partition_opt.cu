/**
 * L3 Partition-Aware Decompression Kernel
 *
 * Optimized for predicate pushdown queries where we know the exact
 * partition IDs to decompress. Eliminates binary search overhead
 * by using explicit partition IDs.
 *
 * Key optimizations:
 * - Direct partition ID indexing (no binary search)
 * - Warp-cooperative double-buffered loading
 * - Contiguous output layout for efficient post-processing
 * - Prefetch output offsets to reduce global memory latency
 */

#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include "L3.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Configuration Parameters (matching decoder_warp_opt.cu)
// ============================================================================

constexpr int PARTITION_MAX_WARPS = 4;
constexpr int PARTITION_TILE_BYTES = 2048;
constexpr int PARTITION_TILE_WORDS = PARTITION_TILE_BYTES / sizeof(uint32_t);
constexpr int PARTITION_NUM_BUFFERS = 2;
constexpr uint32_t PARTITION_FULL_MASK = 0xFFFFFFFF;

// ============================================================================
// Device Helpers (same as decoder_warp_opt.cu)
// ============================================================================

__device__ __forceinline__ uint32_t extract_bits_partition(
    uint64_t val, int start, int width)
{
    uint32_t lo = static_cast<uint32_t>(val);
    uint32_t hi = static_cast<uint32_t>(val >> 32);
    uint32_t extracted = __funnelshift_r(lo, hi, start);
    uint32_t mask = (1U << width) - 1;
    return extracted & mask;
}

__device__ __forceinline__ uint64_t load_uint64_partition(const uint32_t* ptr, int word_idx)
{
    uint32_t lo = ptr[word_idx];
    uint32_t hi = ptr[word_idx + 1];
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// ============================================================================
// Warp Staging Functions
// ============================================================================

#if __CUDA_ARCH__ >= 800
template<int BUFFER_WORDS>
__device__ __forceinline__ void partition_warp_stage_async(
    const uint32_t* __restrict__ global_ptr,
    uint32_t* __restrict__ shared_ptr,
    int words_to_load,
    int lane_id)
{
    constexpr int WORDS_PER_LANE = (BUFFER_WORDS + WARP_SIZE - 1) / WARP_SIZE;

    #pragma unroll
    for (int i = 0; i < WORDS_PER_LANE; ++i) {
        int word_idx = lane_id + i * WARP_SIZE;
        if (word_idx < words_to_load) {
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&shared_ptr[word_idx]));
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(smem_addr),
                   "l"(&global_ptr[word_idx])
            );
        }
    }
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void partition_warp_wait_async()
{
    asm volatile("cp.async.wait_all;\n");
    __syncwarp(PARTITION_FULL_MASK);
}

#else
template<int BUFFER_WORDS>
__device__ __forceinline__ void partition_warp_stage_sync(
    const uint32_t* __restrict__ global_ptr,
    uint32_t* __restrict__ shared_ptr,
    int words_to_load,
    int lane_id)
{
    constexpr int WORDS_PER_LANE = (BUFFER_WORDS + WARP_SIZE - 1) / WARP_SIZE;

    #pragma unroll
    for (int i = 0; i < WORDS_PER_LANE; ++i) {
        int word_idx = lane_id + i * WARP_SIZE;
        if (word_idx < words_to_load) {
            shared_ptr[word_idx] = __ldg(&global_ptr[word_idx]);
        }
    }
    __syncwarp(PARTITION_FULL_MASK);
}
#endif

// ============================================================================
// Partition-Aware Decompression Kernel
// ============================================================================

/**
 * Each block processes one partition from the partition_ids array.
 * Output is written contiguously: partition 0 data, then partition 1 data, etc.
 */
template<typename T>
__global__ void __launch_bounds__(128, 4)
decompressPartitionsKernel(
    const CompressedDataOpt<T> compressed,
    const int* __restrict__ partition_ids,      // Which partitions to decompress
    const int* __restrict__ output_offsets,     // Where each partition's output starts
    T* __restrict__ output)
{
    __shared__ uint32_t s_tiles[PARTITION_MAX_WARPS][PARTITION_NUM_BUFFERS][PARTITION_TILE_WORDS];

    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        int32_t delta_bits;
        int64_t bit_offset_base;
        double theta0;
        double theta1;
        int32_t output_offset;
    } s_meta;

    // Get the actual partition ID from the input array
    const int block_partition_idx = blockIdx.x;
    const int actual_partition_id = partition_ids[block_partition_idx];

    if (actual_partition_id < 0 || actual_partition_id >= compressed.num_partitions) return;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Load metadata for actual partition
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed.d_start_indices[actual_partition_id];
        s_meta.partition_len = compressed.d_end_indices[actual_partition_id] - s_meta.start_idx;
        s_meta.model_type = compressed.d_model_types[actual_partition_id];
        s_meta.delta_bits = compressed.d_delta_bits[actual_partition_id];
        s_meta.bit_offset_base = compressed.d_delta_array_bit_offsets[actual_partition_id];

        int params_idx = actual_partition_id * 4;
        s_meta.theta0 = compressed.d_model_params[params_idx];
        s_meta.theta1 = compressed.d_model_params[params_idx + 1];

        // Get output offset for contiguous layout
        s_meta.output_offset = output_offsets[block_partition_idx];
    }
    __syncthreads();

    // Fast path: no deltas
    if (s_meta.delta_bits <= 0 || compressed.delta_array == nullptr) {
        for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
            int output_idx = s_meta.output_offset + local_idx;

            T final_value;
            if (s_meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
                final_value = static_cast<T>(__double2ll_rn(predicted));
            }
            output[output_idx] = final_value;
        }
        return;
    }

    const int delta_bits = __shfl_sync(PARTITION_FULL_MASK, s_meta.delta_bits, 0);
    const int model_type = __shfl_sync(PARTITION_FULL_MASK, s_meta.model_type, 0);
    const int elements_per_tile = ((PARTITION_TILE_WORDS - 3) * 32) / max(delta_bits, 1);

    int warp_start = warp_id * elements_per_tile;
    int curr_buffer = 0;
    int next_buffer = 1;

    // Prefetch first tile
    if (warp_start < s_meta.partition_len) {
        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(warp_start) * delta_bits);
        int64_t tile_word_offset = tile_bit_offset >> 5;
        int words_needed = (elements_per_tile * delta_bits + 31) / 32 + 2;
        words_needed = min(words_needed, PARTITION_TILE_WORDS);

#if __CUDA_ARCH__ >= 800
        partition_warp_stage_async<PARTITION_TILE_WORDS>(
            compressed.delta_array + tile_word_offset,
            s_tiles[warp_id][curr_buffer],
            words_needed,
            lane_id
        );
#else
        partition_warp_stage_sync<PARTITION_TILE_WORDS>(
            compressed.delta_array + tile_word_offset,
            s_tiles[warp_id][curr_buffer],
            words_needed,
            lane_id
        );
#endif
    }

    // Process tiles with double buffering
    for (int tile_idx = 0; warp_start < s_meta.partition_len; tile_idx++) {
        int tile_start = warp_start;
        int tile_end = min(tile_start + elements_per_tile, s_meta.partition_len);

        int next_warp_start = warp_start + num_warps * elements_per_tile;
        if (next_warp_start < s_meta.partition_len) {
            int64_t next_tile_bit_offset = s_meta.bit_offset_base +
                                          (static_cast<int64_t>(next_warp_start) * delta_bits);
            int64_t next_tile_word_offset = next_tile_bit_offset >> 5;
            int next_words_needed = (elements_per_tile * delta_bits + 31) / 32 + 2;
            next_words_needed = min(next_words_needed, PARTITION_TILE_WORDS);

#if __CUDA_ARCH__ >= 800
            partition_warp_stage_async<PARTITION_TILE_WORDS>(
                compressed.delta_array + next_tile_word_offset,
                s_tiles[warp_id][next_buffer],
                next_words_needed,
                lane_id
            );
#else
            partition_warp_stage_sync<PARTITION_TILE_WORDS>(
                compressed.delta_array + next_tile_word_offset,
                s_tiles[warp_id][next_buffer],
                next_words_needed,
                lane_id
            );
#endif
        }

#if __CUDA_ARCH__ >= 800
        partition_warp_wait_async();
#endif
        __syncwarp(PARTITION_FULL_MASK);

        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(tile_start) * delta_bits);
        int local_bit_base = tile_bit_offset & 31;

        #pragma unroll 4
        for (int i = lane_id; i < (tile_end - tile_start); i += WARP_SIZE) {
            const int local_idx = tile_start + i;
            const int output_idx = s_meta.output_offset + local_idx;

            const int local_bit_offset = local_bit_base + i * delta_bits;
            const int word_idx = local_bit_offset >> 5;
            const int bit_in_word = local_bit_offset & 31;

            T final_value;

            if (delta_bits > 32 && sizeof(T) == 8) {
                uint64_t val64 = 0;
                int bits_remaining = delta_bits;
                int current_word_idx = word_idx;
                int current_bit_offset = bit_in_word;
                int shift_amount = 0;

                while (bits_remaining > 0) {
                    int bits_in_this_word = min(bits_remaining, 32 - current_bit_offset);
                    uint32_t word = s_tiles[warp_id][curr_buffer][current_word_idx];
                    uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                    uint32_t extracted = (word >> current_bit_offset) & mask;

                    val64 |= (static_cast<uint64_t>(extracted) << shift_amount);

                    shift_amount += bits_in_this_word;
                    bits_remaining -= bits_in_this_word;
                    current_word_idx++;
                    current_bit_offset = 0;
                }

                if (model_type == MODEL_DIRECT_COPY) {
                    final_value = static_cast<T>(val64);
                } else {
                    int64_t delta64 = signExtend64(val64, delta_bits);
                    double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
                    T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                    if constexpr (std::is_signed<T>::value) {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    } else {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    }
                }
            } else {
                uint64_t packed = load_uint64_partition(s_tiles[warp_id][curr_buffer], word_idx);
                uint32_t extracted = extract_bits_partition(packed, bit_in_word, delta_bits);

                if (model_type == MODEL_DIRECT_COPY) {
                    if constexpr (std::is_signed<T>::value) {
                        int32_t signed_val = signExtend(extracted, delta_bits);
                        final_value = static_cast<T>(signed_val);
                    } else {
                        final_value = static_cast<T>(extracted);
                    }
                } else {
                    int32_t delta = signExtend(extracted, delta_bits);
                    double predicted = fma(s_meta.theta1, static_cast<double>(local_idx), s_meta.theta0);
                    T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                    if constexpr (std::is_signed<T>::value) {
                        final_value = predicted_T + static_cast<T>(delta);
                    } else {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
                    }
                }
            }

            output[output_idx] = final_value;
        }

        curr_buffer = 1 - curr_buffer;
        next_buffer = 1 - next_buffer;
        warp_start = next_warp_start;
    }
}

// ============================================================================
// Host Launch Functions
// ============================================================================

/**
 * Launch partition-aware decompression for selected partitions
 *
 * @param compressed      Compressed data structure
 * @param d_partition_ids Device array of partition IDs to decompress
 * @param num_partitions  Number of partitions to decompress
 * @param d_output_offsets Device array of output offsets (prefix sum of partition sizes)
 * @param d_output        Output buffer (must be large enough for all selected partitions)
 * @param stream          CUDA stream
 */
template<typename T>
void launchDecompressPartitions(
    const CompressedDataOpt<T>& compressed,
    const int* d_partition_ids,
    int num_partitions,
    const int* d_output_offsets,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (num_partitions == 0) return;

    const int threads_per_block = 128;  // 4 warps per block
    const int num_blocks = num_partitions;

    decompressPartitionsKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        compressed, d_partition_ids, d_output_offsets, d_output
    );
}

// Wrapper for CompressedDataL3 format
template<typename T>
void launchDecompressPartitions(
    const CompressedDataL3<T>* compressed,
    const int* d_partition_ids,
    int num_partitions,
    const int* d_output_offsets,
    T* d_output,
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

    launchDecompressPartitions(opt_format, d_partition_ids, num_partitions,
                               d_output_offsets, d_output, stream);
}

// Explicit instantiations
template void launchDecompressPartitions<int32_t>(const CompressedDataL3<int32_t>*, const int*, int, const int*, int32_t*, cudaStream_t);
template void launchDecompressPartitions<uint32_t>(const CompressedDataL3<uint32_t>*, const int*, int, const int*, uint32_t*, cudaStream_t);
template void launchDecompressPartitions<int64_t>(const CompressedDataL3<int64_t>*, const int*, int, const int*, int64_t*, cudaStream_t);
template void launchDecompressPartitions<uint64_t>(const CompressedDataL3<uint64_t>*, const int*, int, const int*, uint64_t*, cudaStream_t);

template void launchDecompressPartitions<int32_t>(const CompressedDataOpt<int32_t>&, const int*, int, const int*, int32_t*, cudaStream_t);
template void launchDecompressPartitions<uint32_t>(const CompressedDataOpt<uint32_t>&, const int*, int, const int*, uint32_t*, cudaStream_t);
template void launchDecompressPartitions<int64_t>(const CompressedDataOpt<int64_t>&, const int*, int, const int*, int64_t*, cudaStream_t);
template void launchDecompressPartitions<uint64_t>(const CompressedDataOpt<uint64_t>&, const int*, int, const int*, uint64_t*, cudaStream_t);
