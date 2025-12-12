/**
 * L3 Phase 2 Decompression Kernels
 *
 * Advanced optimizations for Hopper/Ampere:
 * 1. cp.async double/triple-buffering pipeline (SM80+)
 * 2. Persistent threads with global work queue
 * 3. Full bitwidth coverage (1/2/4/8/12/16/24/32-bit) with template specialization
 * 4. Vectorized 128-bit loads (uint4/ulonglong2)
 * 5. Warp-synchronous bit stitching with __shfl_sync
 * 6. Launch bounds tuning and register pressure control
 *
 * INVARIANT: Maintains bit-exact compatibility with baseline protocol
 * Protocol version: matches L3_format.hpp v1.0
 */

#ifndef PHASE2_USE_CP_ASYNC
#define PHASE2_USE_CP_ASYNC 1
#endif
#ifndef PHASE2_PERSISTENT_THREADS
#define PHASE2_PERSISTENT_THREADS 0  // DISABLED by default (poor performance ~7-60 GB/s)
#endif
#ifndef PHASE2_VEC_LOADS
#define PHASE2_VEC_LOADS 1
#endif
#ifndef PHASE2_AUTOTUNE
#define PHASE2_AUTOTUNE 1
#endif

#include "bitpack_utils.cuh"
#include "L3_format.hpp"
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <cmath>

namespace cg = cooperative_groups;

// ============================================================================
// Pipeline Utilities (SM80+)
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Double-buffering: load stage N+1 while computing stage N
template<int BUFFER_WORDS>
__device__ inline void pipelineLoadStage(
    const uint32_t* __restrict__ global_src,
    uint32_t* __restrict__ shared_dst,
    int num_words,
    int lane_id)
{
    constexpr int WORDS_PER_THREAD = (BUFFER_WORDS + 31) / 32;

    #pragma unroll
    for (int i = 0; i < WORDS_PER_THREAD; i++) {
        int idx = lane_id + i * 32;
        if (idx < num_words) {
            __pipeline_memcpy_async(
                &shared_dst[idx],
                &global_src[idx],
                sizeof(uint32_t)
            );
        }
    }
    __pipeline_commit();
}

#endif // __CUDA_ARCH__ >= 800

// ============================================================================
// Warp-Synchronous Bit Extraction with Shuffle
// ============================================================================

// Extract packed bits using warp shuffle for cross-lane word stitching
template<int BITS>
__device__ inline int32_t warpExtractBits(
    uint32_t my_word,
    int my_bit_offset,
    int lane_id)
{
    // For bits <= 16, can fit in single word most of the time
    // For larger bits, may need neighbor's word

    const int bit_in_word = my_bit_offset & 31;
    const bool needs_next = (bit_in_word + BITS) > 32;

    uint32_t next_word = 0;
    if (needs_next) {
        // Get next word from adjacent lane
        next_word = __shfl_sync(0xFFFFFFFF, my_word, lane_id + 1);
    }

    // Combine and extract
    uint64_t combined = (static_cast<uint64_t>(next_word) << 32) | my_word;
    uint32_t extracted = (combined >> bit_in_word) & ((1ULL << BITS) - 1);

    // Sign extend
    if constexpr (BITS < 32) {
        uint32_t sign_bit = extracted >> (BITS - 1);
        uint32_t sign_mask = -sign_bit;
        uint32_t extend_mask = ~((1U << BITS) - 1U);
        extracted |= (sign_mask & extend_mask);
    }

    return static_cast<int32_t>(extracted);
}

// ============================================================================
// Specialized Kernels by Bit Width (with Pipeline Support)
// ============================================================================

// 8-bit kernel with cp.async double-buffering (SM80+)
template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_8bit_Phase2(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int num_partitions,
    const int total_elements,
    T* __restrict__ output,
    int32_t* __restrict__ work_queue = nullptr,
    int persistent_mode = 0)
{
    // Shared memory: metadata + double-buffer for bit data
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        double theta0;
        double theta1;
        int64_t bit_offset_base;

        // Double-buffer for pipeline (2 stages × 128 words each)
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        alignas(16) uint32_t buffer[2][128];
        #endif
    } smem;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    // Persistent threads mode: loop over partitions via work queue
    int partition_idx;

    #if PHASE2_PERSISTENT_THREADS
    if (persistent_mode && work_queue) {
        while (true) {
            if (lane_id == 0 && warp_id == 0) {
                partition_idx = atomicAdd(work_queue, 1);
            }
            partition_idx = __shfl_sync(0xFFFFFFFF, partition_idx, 0);

            if (partition_idx >= num_partitions) break;

            // Process this partition (code below)
            goto process_partition;
        }
        return;
    }
    #endif

    // Standard mode: one partition per block
    partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

process_partition:

    // Load metadata (thread 0 only)
    if (threadIdx.x == 0) {
        smem.start_idx = d_start_indices[partition_idx];
        smem.partition_len = d_end_indices[partition_idx] - smem.start_idx;
        smem.model_type = d_model_types[partition_idx];
        smem.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        smem.theta0 = d_model_params[params_idx];
        smem.theta1 = d_model_params[params_idx + 1];

        // ROUTING VALIDATION: Verify this partition actually has 8-bit deltas
        #ifdef PHASE2_DEBUG_ROUTING
        int actual_delta_bits = d_delta_bits[partition_idx];
        if (actual_delta_bits != 8 && actual_delta_bits != 0) {
            printf("WARNING: 8-bit kernel got partition %d with delta_bits=%d\n",
                   partition_idx, actual_delta_bits);
        }
        #endif
    }

    __syncthreads();

    // Process in warp-strided chunks
    constexpr int DELTAS_PER_THREAD = 4;
    constexpr int DELTAS_PER_WARP = 32 * DELTAS_PER_THREAD;  // 128

    for (int chunk_start = warp_id * DELTAS_PER_WARP;
         chunk_start < smem.partition_len;
         chunk_start += num_warps * DELTAS_PER_WARP)
    {
        const int local_idx = chunk_start + lane_id * DELTAS_PER_THREAD;

        if (local_idx >= smem.partition_len) break;

        // Calculate bit offset
        const int64_t bit_offset = smem.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) << 3);  // * 8

        const int word_idx = bit_offset >> 5;
        const int bit_in_word = bit_offset & 31;

        // Vectorized load: try to load 4 words (128 bits) aligned
        #if PHASE2_VEC_LOADS
        uint32_t packed;
        if ((word_idx & 3) == 0 && bit_in_word == 0) {
            // Aligned case: use uint4 load
            uint4 vec = *reinterpret_cast<const uint4*>(&delta_array[word_idx]);
            packed = vec.x;  // First word contains our 4 8-bit deltas
        } else
        #endif
        {
            // Standard unaligned load
            if (bit_in_word == 0) {
                packed = __ldg(&delta_array[word_idx]);
            } else {
                const uint32_t w0 = __ldg(&delta_array[word_idx]);
                const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
                const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
                packed = static_cast<uint32_t>(combined >> bit_in_word);
            }
        }

        // Extract 4 deltas from packed word
        int32_t delta[DELTAS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const uint32_t extracted = (packed >> (i * 8)) & 0xFF;
            delta[i] = static_cast<int32_t>(static_cast<int8_t>(extracted));
        }

        // Compute predictions and write outputs
        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_idx = local_idx + i;
            if (elem_idx >= smem.partition_len) break;

            const int global_idx = smem.start_idx + elem_idx;
            if (global_idx >= total_elements) break;

            T final_value;

            if (smem.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(delta[i]);
            } else {
                const double predicted = fma(smem.theta1,
                                            static_cast<double>(elem_idx),
                                            smem.theta0);
                const T predicted_T = static_cast<T>(round(predicted));

                if constexpr (std::is_signed<T>::value) {
                    final_value = predicted_T + static_cast<T>(delta[i]);
                } else {
                    final_value = static_cast<T>(
                        static_cast<int64_t>(predicted_T) + delta[i]
                    );
                }
            }

            // Coalesced write (each warp writes consecutive addresses)
            output[global_idx] = final_value;
        }
    }
}

// ============================================================================
// 16-bit kernel with pipeline support
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_16bit_Phase2(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const uint32_t* __restrict__ delta_array,
    const int num_partitions,
    const int total_elements,
    T* __restrict__ output,
    int32_t* __restrict__ work_queue = nullptr,
    int persistent_mode = 0)
{
    __shared__ struct {
        int32_t start_idx;
        int32_t partition_len;
        int32_t model_type;
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    } smem;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    int partition_idx;

    #if PHASE2_PERSISTENT_THREADS
    if (persistent_mode && work_queue) {
        while (true) {
            if (lane_id == 0 && warp_id == 0) {
                partition_idx = atomicAdd(work_queue, 1);
            }
            partition_idx = __shfl_sync(0xFFFFFFFF, partition_idx, 0);

            if (partition_idx >= num_partitions) break;
            goto process_partition;
        }
        return;
    }
    #endif

    partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

process_partition:

    if (threadIdx.x == 0) {
        smem.start_idx = d_start_indices[partition_idx];
        smem.partition_len = d_end_indices[partition_idx] - smem.start_idx;
        smem.model_type = d_model_types[partition_idx];
        smem.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        smem.theta0 = d_model_params[params_idx];
        smem.theta1 = d_model_params[params_idx + 1];

        // ROUTING VALIDATION: Verify this partition actually has 16-bit deltas
        #ifdef PHASE2_DEBUG_ROUTING
        int actual_delta_bits = d_delta_bits[partition_idx];
        if (actual_delta_bits != 16 && actual_delta_bits != 0) {
            printf("WARNING: 16-bit kernel got partition %d with delta_bits=%d\n",
                   partition_idx, actual_delta_bits);
        }
        #endif
    }

    __syncthreads();

    constexpr int DELTAS_PER_THREAD = 2;
    constexpr int DELTAS_PER_WARP = 32 * DELTAS_PER_THREAD;  // 64

    for (int chunk_start = warp_id * DELTAS_PER_WARP;
         chunk_start < smem.partition_len;
         chunk_start += num_warps * DELTAS_PER_WARP)
    {
        const int local_idx = chunk_start + lane_id * DELTAS_PER_THREAD;

        if (local_idx >= smem.partition_len) break;

        const int64_t bit_offset = smem.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) << 4);  // * 16
        const int word_idx = bit_offset >> 5;
        const int bit_in_word = bit_offset & 31;

        uint32_t packed;

        #if PHASE2_VEC_LOADS
        if ((word_idx & 3) == 0 && bit_in_word == 0) {
            uint4 vec = *reinterpret_cast<const uint4*>(&delta_array[word_idx]);
            packed = vec.x;
        } else
        #endif
        {
            if (bit_in_word == 0) {
                packed = __ldg(&delta_array[word_idx]);
            } else {
                const uint32_t w0 = __ldg(&delta_array[word_idx]);
                const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
                const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
                packed = static_cast<uint32_t>(combined >> bit_in_word);
            }
        }

        int32_t delta[DELTAS_PER_THREAD];

        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const uint32_t extracted = (packed >> (i * 16)) & 0xFFFF;
            delta[i] = static_cast<int32_t>(static_cast<int16_t>(extracted));
        }

        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_idx = local_idx + i;
            if (elem_idx >= smem.partition_len) break;

            const int global_idx = smem.start_idx + elem_idx;
            if (global_idx >= total_elements) break;

            T final_value;

            if (smem.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(delta[i]);
            } else {
                const double predicted = fma(smem.theta1,
                                            static_cast<double>(elem_idx),
                                            smem.theta0);
                const T predicted_T = static_cast<T>(round(predicted));

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
// Template-based kernel for arbitrary bit widths
// ============================================================================

template<typename T, int BITS>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_Nbits_Phase2(
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
    } smem;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    if (threadIdx.x == 0) {
        smem.start_idx = d_start_indices[partition_idx];
        smem.partition_len = d_end_indices[partition_idx] - smem.start_idx;
        smem.model_type = d_model_types[partition_idx];
        smem.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        smem.theta0 = d_model_params[params_idx];
        smem.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    // Compute deltas per thread based on bit width
    constexpr int DELTAS_PER_THREAD = (BITS <= 8) ? 4 : (BITS <= 16) ? 2 : 1;
    constexpr int DELTAS_PER_WARP = 32 * DELTAS_PER_THREAD;

    for (int chunk_start = warp_id * DELTAS_PER_WARP;
         chunk_start < smem.partition_len;
         chunk_start += num_warps * DELTAS_PER_WARP)
    {
        const int local_idx = chunk_start + lane_id * DELTAS_PER_THREAD;

        if (local_idx >= smem.partition_len) break;

        int32_t delta[DELTAS_PER_THREAD];

        // Extract deltas
        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_local = local_idx + i;
            if (elem_local >= smem.partition_len) {
                delta[i] = 0;
                continue;
            }

            const int64_t bit_offset = smem.bit_offset_base +
                                       (static_cast<int64_t>(elem_local) * BITS);
            const int word_idx = bit_offset >> 5;
            const int bit_in_word = bit_offset & 31;

            // Load and extract
            const uint32_t w0 = __ldg(&delta_array[word_idx]);
            const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
            const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;

            uint32_t extracted = (combined >> bit_in_word) & ((1ULL << BITS) - 1);

            // Sign extend
            if constexpr (BITS < 32) {
                const uint32_t sign_bit = extracted >> (BITS - 1);
                const uint32_t sign_mask = -sign_bit;
                const uint32_t extend_mask = ~((1U << BITS) - 1U);
                extracted |= (sign_mask & extend_mask);
            }

            delta[i] = static_cast<int32_t>(extracted);
        }

        // Compute and write
        #pragma unroll
        for (int i = 0; i < DELTAS_PER_THREAD; i++) {
            const int elem_idx = local_idx + i;
            if (elem_idx >= smem.partition_len) break;

            const int global_idx = smem.start_idx + elem_idx;
            if (global_idx >= total_elements) break;

            T final_value;

            if (smem.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(delta[i]);
            } else {
                const double predicted = fma(smem.theta1,
                                            static_cast<double>(elem_idx),
                                            smem.theta0);
                const T predicted_T = static_cast<T>(round(predicted));

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
// Special kernel for 0-bit deltas (perfect prediction)
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256)
decompressPartitions_0bit_Phase2(
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
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
    } smem;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    // Load metadata
    if (threadIdx.x == 0) {
        smem.start_idx = d_start_indices[partition_idx];
        smem.partition_len = d_end_indices[partition_idx] - smem.start_idx;
        smem.model_type = d_model_types[partition_idx];

        const int params_idx = partition_idx * 4;
        smem.theta0 = d_model_params[params_idx];
        smem.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    // Process elements: delta is always 0, just compute prediction
    for (int local_idx = threadIdx.x; local_idx < smem.partition_len; local_idx += blockDim.x) {
        const int global_idx = smem.start_idx + local_idx;
        if (global_idx >= total_elements) break;

        T final_value;

        if (smem.model_type == MODEL_DIRECT_COPY) {
            final_value = static_cast<T>(0);  // Delta is 0
        } else {
            // Perfect prediction: predicted value IS the final value
            const double predicted = fma(smem.theta1,
                                        static_cast<double>(local_idx),
                                        smem.theta0);
            final_value = static_cast<T>(round(predicted));
        }

        output[global_idx] = final_value;
    }
}

// ============================================================================
// Generic kernel with per-partition delta_bits (for mixed bit widths)
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(256, 4)
decompressPartitions_Generic_Phase2(
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
        int32_t delta_bits;  // Per-partition!
        double theta0;
        double theta1;
        int64_t bit_offset_base;
    } smem;

    const int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    // Load metadata including per-partition delta_bits
    if (threadIdx.x == 0) {
        smem.start_idx = d_start_indices[partition_idx];
        smem.partition_len = d_end_indices[partition_idx] - smem.start_idx;
        smem.model_type = d_model_types[partition_idx];
        smem.delta_bits = d_delta_bits[partition_idx];
        smem.bit_offset_base = d_delta_array_bit_offsets[partition_idx];

        const int params_idx = partition_idx * 4;
        smem.theta0 = d_model_params[params_idx];
        smem.theta1 = d_model_params[params_idx + 1];
    }

    __syncthreads();

    // Handle zero delta_bits case
    if (smem.delta_bits == 0) {
        for (int local_idx = threadIdx.x; local_idx < smem.partition_len; local_idx += blockDim.x) {
            const int global_idx = smem.start_idx + local_idx;
            if (global_idx >= total_elements) continue;

            T final_value;
            if (smem.model_type == MODEL_DIRECT_COPY) {
                final_value = static_cast<T>(0);
            } else {
                const double predicted = fma(smem.theta1, static_cast<double>(local_idx), smem.theta0);
                final_value = static_cast<T>(round(predicted));
            }
            output[global_idx] = final_value;
        }
        return;
    }

    // Grid-stride loop with runtime delta_bits
    for (int local_idx = threadIdx.x; local_idx < smem.partition_len; local_idx += blockDim.x) {
        const int global_idx = smem.start_idx + local_idx;
        if (global_idx >= total_elements) continue;

        // Calculate bit offset using per-partition delta_bits
        const int64_t bit_offset = smem.bit_offset_base +
                                   (static_cast<int64_t>(local_idx) * smem.delta_bits);
        const int word_idx = bit_offset >> 5;
        const int bit_in_word = bit_offset & 31;

        T final_value;

        // Handle 64-bit types with deltas > 32 bits
        if constexpr (sizeof(T) == 8) {
            if (smem.delta_bits > 32) {
                // Multi-word extraction for 64-bit deltas
                uint64_t val64 = 0;
                int bits_remaining = smem.delta_bits;
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

                if (smem.model_type == MODEL_DIRECT_COPY) {
                    final_value = static_cast<T>(val64);
                } else {
                    // Sign extend the extracted delta
                    int64_t delta64 = signExtend64(val64, smem.delta_bits);
                    const double predicted = fma(smem.theta1, static_cast<double>(local_idx), smem.theta0);
                    const T predicted_T = static_cast<T>(round(predicted));

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
        const uint32_t w0 = __ldg(&delta_array[word_idx]);
        const uint32_t w1 = __ldg(&delta_array[word_idx + 1]);
        const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
        const uint64_t mask = (smem.delta_bits == 32) ? 0xFFFFFFFFULL : ((1ULL << smem.delta_bits) - 1);
        extracted = static_cast<uint32_t>((combined >> bit_in_word) & mask);

        // Sign extend
        int32_t delta;
        if (smem.delta_bits < 32) {
            const uint32_t sign_bit = extracted >> (smem.delta_bits - 1);
            const uint32_t sign_mask = -sign_bit;
            const uint32_t extend_mask = ~((1U << smem.delta_bits) - 1U);
            delta = static_cast<int32_t>(extracted | (sign_mask & extend_mask));
        } else {
            delta = static_cast<int32_t>(extracted);
        }

        if (smem.model_type == MODEL_DIRECT_COPY) {
            // Direct copy: use raw extracted value for unsigned, sign-extended for signed
            if constexpr (std::is_signed<T>::value) {
                final_value = static_cast<T>(delta);
            } else {
                final_value = static_cast<T>(extracted);
            }
        } else {
            const double predicted = fma(smem.theta1,
                                        static_cast<double>(local_idx),
                                        smem.theta0);
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
// Runtime Dispatcher with Full Bitwidth Coverage
// ============================================================================

template<typename T>
void decompressL3_Phase2(
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
    int avg_delta_bits,
    bool use_persistent = false)
{
    dim3 block(256);
    dim3 grid(num_partitions);

    // Always use generic kernel that reads per-partition delta_bits
    // This handles mixed bit-width data correctly
    // Specialized template kernels only work when ALL partitions have the same bit width
    decompressPartitions_Generic_Phase2<<<grid, block>>>(
        d_start_indices, d_end_indices, d_model_types, d_model_params,
        d_delta_bits, d_delta_array_bit_offsets, delta_array,
        num_partitions, total_elements, output);
}

// Explicit template instantiations
template void decompressL3_Phase2<uint32_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, uint32_t*, int, bool);

template void decompressL3_Phase2<uint64_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, uint64_t*, int, bool);

template void decompressL3_Phase2<int32_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, int32_t*, int, bool);

template void decompressL3_Phase2<int64_t>(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, int64_t*, int, bool);
