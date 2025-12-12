/**
 * L3 Warp-Optimized Decoder
 *
 * Extreme-performance decompression using:
 * - Double-buffered shared memory staging
 * - Warp-cooperative async loads (cp.async on sm_80+)
 * - Funnel shifts for cross-word bit extraction
 * - Branchless hot loop with predication
 * - Vectorized global memory access
 *
 * Target: 300-500 GB/s on V100
 */

#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include "L3.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================================
// Configuration Parameters
// ============================================================================

// Maximum number of warps per block
constexpr int MAX_WARPS_PER_BLOCK = 4;

// Tile size in bytes for staging (per warp)
constexpr int TILE_BYTES = 2048;  // 2KB per warp
constexpr int TILE_WORDS = TILE_BYTES / sizeof(uint32_t);  // 512 words

// Number of tiles for double buffering
constexpr int NUM_BUFFERS = 2;

// WARP_SIZE is defined in bitpack_utils.cuh

// Full mask for warp operations
constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

// ============================================================================
// Device-side Funnel Shift Helpers
// ============================================================================

/**
 * Extract w bits from a 64-bit value starting at bit position 'start'
 * Uses funnel shift for efficient cross-boundary extraction
 */
__device__ __forceinline__ uint32_t extract_bits_funnel(
    uint64_t val, int start, int width)
{
    // Funnel shift right: extract width bits starting at 'start'
    uint32_t lo = static_cast<uint32_t>(val);
    uint32_t hi = static_cast<uint32_t>(val >> 32);

    // Use CUDA's funnel shift intrinsic
    uint32_t extracted = __funnelshift_r(lo, hi, start);
    uint32_t mask = (1U << width) - 1;

    return extracted & mask;
}

/**
 * Load 64-bit aligned value from shared memory
 */
__device__ __forceinline__ uint64_t load_uint64(const uint32_t* ptr, int word_idx)
{
    uint32_t lo = ptr[word_idx];
    uint32_t hi = ptr[word_idx + 1];
    return (static_cast<uint64_t>(hi) << 32) | lo;
}

// load_uint4_aligned is defined in bitpack_utils.cuh

// ============================================================================
// Warp-Cooperative Async Staging
// ============================================================================

#if __CUDA_ARCH__ >= 800
// Use cp.async on Ampere+
template<int BUFFER_WORDS>
__device__ __forceinline__ void warp_stage_async(
    const uint32_t* __restrict__ global_ptr,
    uint32_t* __restrict__ shared_ptr,
    int words_to_load,
    int lane_id)
{
    // Each thread loads multiple words using cp.async
    constexpr int WORDS_PER_LANE = (BUFFER_WORDS + WARP_SIZE - 1) / WARP_SIZE;

    #pragma unroll
    for (int i = 0; i < WORDS_PER_LANE; ++i) {
        int word_idx = lane_id + i * WARP_SIZE;
        if (word_idx < words_to_load) {
            // Use cp.async for async copy from global to shared
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&shared_ptr[word_idx]));
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 4;\n"
                :: "r"(smem_addr),
                   "l"(&global_ptr[word_idx])
            );
        }
    }

    // Commit this group
    asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void warp_wait_async()
{
    // Wait for all async groups to complete
    asm volatile("cp.async.wait_all;\n");
    __syncwarp(FULL_MASK);
}

#else
// Fallback for pre-Ampere: coalesced loads
template<int BUFFER_WORDS>
__device__ __forceinline__ void warp_stage_sync(
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

    __syncwarp(FULL_MASK);
}
#endif

// Unified staging function that selects async or sync based on architecture
template<int BUFFER_WORDS>
__device__ __forceinline__ void warp_stage_ldg(
    const uint32_t* __restrict__ global_ptr,
    uint32_t* __restrict__ shared_ptr,
    int words_to_load,
    int lane_id)
{
#if __CUDA_ARCH__ >= 800
    warp_stage_async<BUFFER_WORDS>(global_ptr, shared_ptr, words_to_load, lane_id);
#else
    warp_stage_sync<BUFFER_WORDS>(global_ptr, shared_ptr, words_to_load, lane_id);
#endif
}

// Wait for staging to complete
__device__ __forceinline__ void warp_stage_wait()
{
#if __CUDA_ARCH__ >= 800
    warp_wait_async();
#else
    // Already synced in warp_stage_sync
#endif
}

// ============================================================================
// Warp-Optimized Decompression Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressWarpOptimized(
    const CompressedDataOpt<T> compressed,
    T* __restrict__ output)
{
    // Double-buffered shared memory (per-warp to avoid race conditions)
    // Layout: [warp_id][buffer_id][words]
    __shared__ uint32_t s_tiles[MAX_WARPS_PER_BLOCK][NUM_BUFFERS][TILE_WORDS];

    // Shared metadata (loaded once)
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        int32_t delta_bits;
        int64_t bit_offset_base;
        double theta0;
        double theta1;
        double theta2;  // For POLY2
        double theta3;  // For POLY3
    } s_meta;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= compressed.num_partitions) return;

    // Warp and lane identification
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // Load metadata (single thread)
    if (threadIdx.x == 0) {
        s_meta.start_idx = compressed.d_start_indices[partition_idx];
        s_meta.partition_len = compressed.d_end_indices[partition_idx] - s_meta.start_idx;
        s_meta.model_type = compressed.d_model_types[partition_idx];
        s_meta.delta_bits = compressed.d_delta_bits[partition_idx];
        s_meta.bit_offset_base = compressed.d_delta_array_bit_offsets[partition_idx];

        int params_idx = partition_idx * 4;
        s_meta.theta0 = compressed.d_model_params[params_idx];
        s_meta.theta1 = compressed.d_model_params[params_idx + 1];
        s_meta.theta2 = compressed.d_model_params[params_idx + 2];  // For POLY2
        s_meta.theta3 = compressed.d_model_params[params_idx + 3];  // For POLY3
    }
    __syncthreads();

    // Fast path: no deltas
    if (s_meta.delta_bits <= 0 || compressed.delta_array == nullptr) {
        for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
            int global_idx = s_meta.start_idx + local_idx;
            if (global_idx >= compressed.total_elements) continue;

            T final_value;
            if (s_meta.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                // Compute prediction using Horner's method based on model type
                double x = static_cast<double>(local_idx);
                double predicted;
                switch (s_meta.model_type) {
                    case MODEL_FOR_BITPACK: {
                        // FOR: base stored in theta0 (no prediction needed when delta_bits=0)
                        T base;
                        if constexpr (sizeof(T) == 8) {
                            base = static_cast<T>(__double_as_longlong(s_meta.theta0));
                        } else {
                            base = static_cast<T>(__double2int_rn(s_meta.theta0));
                        }
                        final_value = base;
                        break;
                    }
                    case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                        predicted = s_meta.theta0 + x * (s_meta.theta1 + x * s_meta.theta2);
                        final_value = static_cast<T>(__double2ll_rn(predicted));
                        break;
                    case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                        predicted = s_meta.theta0 + x * (s_meta.theta1 + x * (s_meta.theta2 + x * s_meta.theta3));
                        final_value = static_cast<T>(__double2ll_rn(predicted));
                        break;
                    default:  // LINEAR, CONSTANT, etc.
                        predicted = fma(s_meta.theta1, x, s_meta.theta0);
                        final_value = static_cast<T>(__double2ll_rn(predicted));
                        break;
                }
            }
            output[global_idx] = final_value;
        }
        return;
    }

    // Broadcast metadata to registers using shuffle
    const int delta_bits = __shfl_sync(FULL_MASK, s_meta.delta_bits, 0);
    const int model_type = __shfl_sync(FULL_MASK, s_meta.model_type, 0);

    // Calculate elements per tile (ensure we fit in TILE_WORDS with +2 margin for cross-word)
    const int elements_per_tile = ((TILE_WORDS - 3) * 32) / max(delta_bits, 1);

    // Each warp processes its portion
    int warp_start = warp_id * elements_per_tile;

    // Double-buffering indices
    int curr_buffer = 0;
    int next_buffer = 1;

    // Prefetch first tile
    if (warp_start < s_meta.partition_len) {
        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(warp_start) * delta_bits);
        int64_t tile_word_offset = tile_bit_offset >> 5;
        int words_needed = (elements_per_tile * delta_bits + 31) / 32 + 2;  // +2 for cross-word
        words_needed = min(words_needed, TILE_WORDS);

        warp_stage_ldg<TILE_WORDS>(
            compressed.delta_array + tile_word_offset,
            s_tiles[warp_id][curr_buffer],
            words_needed,
            lane_id
        );
    }

    // Process tiles with double buffering
    for (int tile_idx = 0; warp_start < s_meta.partition_len; tile_idx++) {
        int tile_start = warp_start;
        int tile_end = min(tile_start + elements_per_tile, s_meta.partition_len);

        // Prefetch next tile while processing current
        int next_warp_start = warp_start + num_warps * elements_per_tile;
        if (next_warp_start < s_meta.partition_len) {
            int64_t next_tile_bit_offset = s_meta.bit_offset_base +
                                          (static_cast<int64_t>(next_warp_start) * delta_bits);
            int64_t next_tile_word_offset = next_tile_bit_offset >> 5;
            int next_words_needed = (elements_per_tile * delta_bits + 31) / 32 + 2;
            next_words_needed = min(next_words_needed, TILE_WORDS);

            warp_stage_ldg<TILE_WORDS>(
                compressed.delta_array + next_tile_word_offset,
                s_tiles[warp_id][next_buffer],
                next_words_needed,
                lane_id
            );
        }

        // Sync to ensure current tile is ready
        warp_stage_wait();

        // Calculate local bit offset base for this tile
        int64_t tile_bit_offset = s_meta.bit_offset_base +
                                 (static_cast<int64_t>(tile_start) * delta_bits);
        int local_bit_base = tile_bit_offset & 31;

        // Decode loop: each lane processes its assigned elements
        #pragma unroll 4
        for (int i = lane_id; i < (tile_end - tile_start); i += WARP_SIZE) {
            const int local_idx = tile_start + i;
            const int global_idx = s_meta.start_idx + local_idx;

            if (global_idx >= compressed.total_elements) continue;

            // Calculate bit position in tile - optimized with strength reduction
            const int local_bit_offset = local_bit_base + i * delta_bits;
            const int word_idx = local_bit_offset >> 5;
            const int bit_in_word = local_bit_offset & 31;

            T final_value;

            // Handle deltas > 32 bits (including 64-bit) for 64-bit types
            if (delta_bits > 32 && sizeof(T) == 8) {
                // Extract up to 64-bit value using multi-word logic
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

                // Apply model and delta for 64-bit types
                if (model_type == MODEL_DIRECT_COPY) {
                    final_value = static_cast<T>(val64);
                } else if (model_type == MODEL_FOR_BITPACK) {
                    // FOR model: delta is UNSIGNED, base stored in theta0
                    T base = static_cast<T>(__double_as_longlong(s_meta.theta0));
                    final_value = base + static_cast<T>(val64);  // val64 is unsigned delta
                } else {
                    // Sign extend the extracted delta
                    int64_t delta64 = signExtend64(val64, delta_bits);

                    // Compute prediction using Horner's method based on model type
                    double x = static_cast<double>(local_idx);
                    double predicted;
                    switch (model_type) {
                        case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                            predicted = s_meta.theta0 + x * (s_meta.theta1 + x * s_meta.theta2);
                            break;
                        case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                            predicted = s_meta.theta0 + x * (s_meta.theta1 + x * (s_meta.theta2 + x * s_meta.theta3));
                            break;
                        default:  // LINEAR, CONSTANT, etc.
                            predicted = fma(s_meta.theta1, x, s_meta.theta0);
                            break;
                    }
                    // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
                    if constexpr (std::is_unsigned<T>::value) {
                        if (predicted < 0.0) predicted = 0.0;
                    }
                    T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                    if constexpr (std::is_signed<T>::value) {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    } else {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta64);
                    }
                }
            } else {
                // Standard path: up to 32-bit deltas
                // Load 64-bit value for funnel shift
                uint64_t packed = load_uint64(s_tiles[warp_id][curr_buffer], word_idx);

                // Extract using funnel shift (branchless)
                uint32_t extracted = extract_bits_funnel(packed, bit_in_word, delta_bits);

                // Apply model and delta (branchless with predication)
                if (model_type == MODEL_DIRECT_COPY) {
                    // Direct copy path
                    if constexpr (std::is_signed<T>::value) {
                        int32_t signed_val = signExtend(extracted, delta_bits);
                        final_value = static_cast<T>(signed_val);
                    } else {
                        final_value = static_cast<T>(extracted);
                    }
                } else if (model_type == MODEL_FOR_BITPACK) {
                    // FOR model: delta is UNSIGNED, base stored in theta0
                    T base;
                    if constexpr (sizeof(T) == 8) {
                        base = static_cast<T>(__double_as_longlong(s_meta.theta0));
                    } else {
                        base = static_cast<T>(__double2int_rn(s_meta.theta0));
                    }
                    final_value = base + static_cast<T>(extracted);  // extracted is unsigned delta
                } else {
                    // Model-based path
                    int32_t delta = signExtend(extracted, delta_bits);

                    // Compute prediction using Horner's method based on model type
                    double x = static_cast<double>(local_idx);
                    double predicted;
                    switch (model_type) {
                        case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                            predicted = s_meta.theta0 + x * (s_meta.theta1 + x * s_meta.theta2);
                            break;
                        case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                            predicted = s_meta.theta0 + x * (s_meta.theta1 + x * (s_meta.theta2 + x * s_meta.theta3));
                            break;
                        default:  // LINEAR, CONSTANT, etc.
                            predicted = fma(s_meta.theta1, x, s_meta.theta0);
                            break;
                    }
                    // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
                    if constexpr (std::is_unsigned<T>::value) {
                        if (predicted < 0.0) predicted = 0.0;
                    }
                    T predicted_T = static_cast<T>(__double2ll_rn(predicted));

                    if constexpr (std::is_signed<T>::value) {
                        final_value = predicted_T + static_cast<T>(delta);
                    } else {
                        final_value = static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
                    }
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
// Host Launch Function
// ============================================================================

template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataOpt<T>& compressed,
    T* output,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0) return;

    // Use 4 warps per block (128 threads) for balanced occupancy and shared memory
    // Each warp has its own shared memory buffer: s_tiles[warp_id][NUM_BUFFERS][TILE_WORDS]
    const int threads_per_block = 128;  // 4 warps per block
    const int num_blocks = compressed.num_partitions;

    decompressWarpOptimized<T><<<num_blocks, threads_per_block, 0, stream>>>(
        compressed, output
    );
}

// Wrapper for CompressedDataL3 format
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
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

    launchDecompressWarpOpt(opt_format, d_output, stream);
}

// Explicit instantiations
template void launchDecompressWarpOpt<int32_t>(const CompressedDataL3<int32_t>*, int32_t*, cudaStream_t);
template void launchDecompressWarpOpt<uint32_t>(const CompressedDataL3<uint32_t>*, uint32_t*, cudaStream_t);
template void launchDecompressWarpOpt<int64_t>(const CompressedDataL3<int64_t>*, int64_t*, cudaStream_t);
template void launchDecompressWarpOpt<uint64_t>(const CompressedDataL3<uint64_t>*, uint64_t*, cudaStream_t);

template void launchDecompressWarpOpt<int32_t>(const CompressedDataOpt<int32_t>&, int32_t*, cudaStream_t);
template void launchDecompressWarpOpt<uint32_t>(const CompressedDataOpt<uint32_t>&, uint32_t*, cudaStream_t);
template void launchDecompressWarpOpt<int64_t>(const CompressedDataOpt<int64_t>&, int64_t*, cudaStream_t);
template void launchDecompressWarpOpt<uint64_t>(const CompressedDataOpt<uint64_t>&, uint64_t*, cudaStream_t);
