/**
 * GM (Word-Interleaved) V4 Decoder
 *
 * This decoder matches V4_decoder in all aspects EXCEPT the memory layout:
 *   - V4_decoder: Lane-Major (Vertical) - strided access
 *   - GM_V4_decoder: Word-Interleaved (Transposed) - coalesced access
 *
 * Memory access pattern:
 *   Vertical:   thread L reads addr[L * words_per_lane + w]  (STRIDED)
 *   GM:         thread L reads addr[w * 32 + L]              (COALESCED)
 *
 * All other logic (model types, bit extraction, tail handling) is identical.
 *
 * Date: 2025-01-26
 */

#ifndef GM_DECODER_V4_CUH
#define GM_DECODER_V4_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include "L3_Transposed_format.hpp"
#include "L3_Vertical_format.hpp"  // For constants like MINI_VECTOR_SIZE, VALUES_PER_THREAD
#include "bitpack_utils_Vertical.cuh"  // For mask64_rt, sign_extend_64

namespace GM_V4_decoder {

constexpr int BLOCK_SIZE_V4 = 256;
constexpr int WARP_SIZE_V4 = 32;

// ============================================================================
// Device Helper Functions (same as V4_decoder)
// ============================================================================

template<typename T>
__device__ __forceinline__
T applyDeltaV4(T predicted, int64_t delta) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(static_cast<uint64_t>(predicted) + static_cast<uint64_t>(delta));
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

template<typename T>
__device__ __forceinline__
T computePredictionPolyV4(int32_t model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:
            predicted = params[0];
            break;
        case MODEL_LINEAR:
            predicted = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        default:
            predicted = params[0] + params[1] * x;
            break;
    }

    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(predicted));
    } else {
        return static_cast<T>(__double2ll_rn(predicted));
    }
}

// ============================================================================
// Direct memory access helper for GM format (coalesced pattern)
// ============================================================================
__device__ __forceinline__ void extract_gm_vectorized_4_rt(
    const uint32_t* __restrict__ transposed_array,
    int64_t mv_word_base,
    int lane_id,
    int local_bit,
    int delta_bits,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3)
{
    if (delta_bits <= 0) {
        v0 = v1 = v2 = v3 = 0U;
        return;
    }
    if (delta_bits > 32) delta_bits = 32;

    const uint64_t MASK = (delta_bits == 32) ? 0xFFFFFFFFULL : ((1ULL << delta_bits) - 1ULL);

    // Lambda to extract a single value
    auto extract_one = [&](int value_idx) -> uint32_t {
        int bit_pos = local_bit + value_idx * delta_bits;
        int word_idx = bit_pos >> 5;
        int bit_in_word = bit_pos & 31;

        // GM layout: word address = mv_word_base + word_idx * 32 + lane_id
        uint32_t lo = __ldg(&transposed_array[mv_word_base + word_idx * 32 + lane_id]);
        uint32_t hi = __ldg(&transposed_array[mv_word_base + (word_idx + 1) * 32 + lane_id]);
        uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;

        return static_cast<uint32_t>((combined >> bit_in_word) & MASK);
    };

    v0 = extract_one(0);
    v1 = extract_one(1);
    v2 = extract_one(2);
    v3 = extract_one(3);
}

// ============================================================================
// GM V4 Decoder Kernel (Word-Interleaved Layout) - Direct Memory Access Version
// ============================================================================

template<typename T>
__global__ void decompressGM_AllPartitionsV4(
    const uint32_t* __restrict__ transposed_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_transposed_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Load partition metadata
    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t transposed_base = d_transposed_offsets[pid];
    int model_type = d_model_types[pid];

    // Lazy parameter loading
    double params[4];
    params[0] = d_model_params[pid * 4];
    if (model_type != MODEL_FOR_BITPACK) {
        params[1] = d_model_params[pid * 4 + 1];
        params[2] = d_model_params[pid * 4 + 2];
        params[3] = d_model_params[pid * 4 + 3];
    }

    // Pre-compute base value for FOR_BITPACK
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base_value = static_cast<T>(__double2ll_rn(params[0]));
        }
    }

    // Handle no transposed data case
    if (transposed_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value;
            } else {
                result = computePredictionPolyV4<T>(model_type, params, local_idx);
            }
            output[global_idx] = result;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE_V4;
    int warp_id_in_block = threadIdx.x / WARP_SIZE_V4;
    int warps_per_block = blockDim.x / WARP_SIZE_V4;

    // GM layout calculations
    int bits_per_lane = VALUES_PER_THREAD * delta_bits;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mv = words_per_lane * 32;

    // Mask for delta extraction
    uint64_t mask = Vertical::mask64_rt(delta_bits);

    // Process mini-vectors using DIRECT MEMORY ACCESS (same pattern as V4_decoder)
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // GM layout: word address = mv_word_base + word_in_lane * 32 + lane_id
        int64_t mv_word_base = transposed_base + static_cast<int64_t>(mv_idx) * words_per_mv;

        // Direct memory access path for uint32/int32 types - process 4 values at once
        if constexpr (sizeof(T) <= 4) {
            if (delta_bits == 0) {
                // Zero delta bits - no extraction needed
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; v++) {
                    int global_idx = mv_start_global + v * WARP_SIZE_V4 + lane_id;
                    int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE_V4 + lane_id;
                    T result;
                    if (model_type == MODEL_FOR_BITPACK) {
                        result = base_value;
                    } else {
                        result = computePredictionPolyV4<T>(model_type, params, local_idx_in_partition);
                    }
                    output[global_idx] = result;
                }
            } else {
                // Non-zero delta bits - use vectorized extraction (4 values at a time)
                int local_bit = 0;
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                    // Extract 4 consecutive values at once using direct memory access
                    uint32_t vals[4];
                    extract_gm_vectorized_4_rt(transposed_array, mv_word_base, lane_id,
                                               local_bit, delta_bits,
                                               vals[0], vals[1], vals[2], vals[3]);

                    // Process and output each of the 4 values
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        int vv = v + i;
                        int global_idx = mv_start_global + vv * WARP_SIZE_V4 + lane_id;
                        int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + vv * WARP_SIZE_V4 + lane_id;

                        T result;
                        if (model_type == MODEL_FOR_BITPACK) {
                            result = base_value + static_cast<T>(vals[i]);
                        } else {
                            T predicted = computePredictionPolyV4<T>(model_type, params, local_idx_in_partition);
                            int64_t delta = Vertical::sign_extend_64(vals[i], delta_bits);
                            result = applyDeltaV4(predicted, delta);
                        }
                        output[global_idx] = result;
                    }
                    local_bit += 4 * delta_bits;
                }
            }
        } else {
            // Original path for uint64/int64 types - process 1 value at a time with direct access
            int local_bit = 0;
            #pragma unroll
            for (int v = 0; v < VALUES_PER_THREAD; v++) {
                int global_idx = mv_start_global + v * WARP_SIZE_V4 + lane_id;
                int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE_V4 + lane_id;

                T result;
                if (model_type == MODEL_FOR_BITPACK) {
                    if (delta_bits == 0) {
                        result = base_value;
                    } else {
                        // Direct memory access for GM format
                        int word_idx = local_bit >> 5;
                        int bit_in_word = local_bit & 31;
                        uint32_t lo = __ldg(&transposed_array[mv_word_base + word_idx * 32 + lane_id]);
                        uint32_t hi = __ldg(&transposed_array[mv_word_base + (word_idx + 1) * 32 + lane_id]);
                        uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;
                        uint64_t extracted = (combined >> bit_in_word) & mask;
                        result = base_value + static_cast<T>(extracted);
                    }
                } else if (delta_bits == 0) {
                    result = computePredictionPolyV4<T>(model_type, params, local_idx_in_partition);
                } else {
                    T predicted = computePredictionPolyV4<T>(model_type, params, local_idx_in_partition);
                    // Direct memory access for GM format
                    int word_idx = local_bit >> 5;
                    int bit_in_word = local_bit & 31;
                    uint32_t lo = __ldg(&transposed_array[mv_word_base + word_idx * 32 + lane_id]);
                    uint32_t hi = __ldg(&transposed_array[mv_word_base + (word_idx + 1) * 32 + lane_id]);
                    uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;
                    uint64_t extracted = (combined >> bit_in_word) & mask;
                    int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                    result = applyDeltaV4(predicted, delta);
                }

                output[global_idx] = result;
                local_bit += delta_bits;
            }
        }
    }

    // Handle tail values (sequential bit-packed after mini-vectors)
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int64_t tail_word_base = transposed_base + static_cast<int64_t>(num_mv) * words_per_mv;
    int64_t tail_bit_base = tail_word_base * 32;

    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;

        T result;
        if (model_type == MODEL_FOR_BITPACK) {
            if (delta_bits == 0) {
                result = base_value;
            } else {
                int tail_local_idx = local_idx - tail_start;
                int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
                uint64_t delta = Vertical::extract_branchless_64_rt(transposed_array, bit_offset, delta_bits);
                result = base_value + static_cast<T>(delta);
            }
        } else if (delta_bits == 0) {
            result = computePredictionPolyV4<T>(model_type, params, local_idx);
        } else {
            T predicted = computePredictionPolyV4<T>(model_type, params, local_idx);
            int tail_local_idx = local_idx - tail_start;
            int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(transposed_array, bit_offset, delta_bits);
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
            result = applyDeltaV4(predicted, delta);
        }

        output[global_idx] = result;
    }
}

// ============================================================================
// GM V4 Decoder Wrapper with configurable block size
// ============================================================================

template<typename T>
void decompressV4_GM(
    const CompressedDataTransposed<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0,
    int block_size = 256)
{
    if (compressed.num_partitions == 0) return;

    int np = compressed.num_partitions;

    switch (block_size) {
        case 32:
            decompressGM_AllPartitionsV4<T><<<np, 32, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        case 64:
            decompressGM_AllPartitionsV4<T><<<np, 64, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        case 128:
            decompressGM_AllPartitionsV4<T><<<np, 128, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        case 256:
            decompressGM_AllPartitionsV4<T><<<np, 256, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        case 512:
            decompressGM_AllPartitionsV4<T><<<np, 512, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        case 1024:
            decompressGM_AllPartitionsV4<T><<<np, 1024, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
        default:
            decompressGM_AllPartitionsV4<T><<<np, 256, 0, stream>>>(
                compressed.d_transposed_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_transposed_offsets,
                np, d_output);
            break;
    }
}

// Adaptive block size selection (same as V4_decoder)
__host__ inline int selectOptimalBlockSize(int64_t total_elements, int num_partitions) {
    if (num_partitions == 0) return 256;
    int64_t avg_partition_size = total_elements / num_partitions;

    if (avg_partition_size < 700) {
        return 64;
    } else if (avg_partition_size < 2000) {
        return 128;
    } else {
        return 256;
    }
}

} // namespace GM_V4_decoder

#endif // GM_DECODER_V4_CUH
