/**
 * L3 Transposed (Word-Interleaved) Decoder
 *
 * GPU decoder with COALESCED memory access pattern:
 *
 * KEY DIFFERENCE from Vertical (Lane-Major):
 *   Vertical:   Thread 0 reads addr 0, Thread 1 reads addr N (STRIDED!)
 *   Transposed: Thread 0 reads addr 0, Thread 1 reads addr 1 (COALESCED!)
 *
 * Memory Layout:
 *   Vertical:   [Lane0_W0][Lane0_W1]...[Lane0_WN][Lane1_W0]...[Lane31_WN]
 *   Transposed: [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *
 * Benefits:
 *   - Perfect 128-byte coalesced memory access
 *   - 4x fewer L1 cache transactions
 *   - Single cache line per warp read
 *
 * Platform: SM 8.0+ (Ampere and later)
 * Date: 2025-12-16
 */

#ifndef DECODER_TRANSPOSED_CU
#define DECODER_TRANSPOSED_CU

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "L3_format.hpp"
#include "L3_Transposed_format.hpp"
#include "../utils/bitpack_utils_Transposed.cuh"

namespace Transposed_decoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_WORDS_PER_LANE = 20;  // Max words per lane for register buffer

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * Apply signed delta to predicted value
 */
template<typename T>
__device__ __forceinline__
T applyDelta(T predicted, int64_t delta) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(static_cast<uint64_t>(predicted) + static_cast<uint64_t>(delta));
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

/**
 * Compute prediction using polynomial model (Horner's method)
 */
template<typename T>
__device__ __forceinline__
T computePredictionPoly(int32_t model_type, const double* params, int local_idx) {
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
// TRANSPOSED Decompression Kernel (Coalesced Access)
// ============================================================================

/**
 * Decompress all partitions using TRANSPOSED (Word-Interleaved) format
 *
 * KEY OPTIMIZATION: Coalesced memory access!
 *
 * Memory access pattern for loading lane words:
 *   Vertical (strided):   lane_word_addr = lane_word_start + i
 *                         where lane_word_start = mv_bit_base + lane_id * words_per_lane
 *   Transposed (coalesced): lane_word_addr = mv_word_base + w * WARP_SIZE + lane_id
 *
 * This means:
 *   - All 32 threads read consecutive addresses
 *   - Single 128-byte cache line transaction per warp read
 *   - 4x fewer L1 cache sectors than Vertical
 */
template<typename T>
__global__ void decompressTransposedAllPartitions(
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

    // Load model parameters
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    if (transposed_base < 0) {
        // No transposed data - use prediction only
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                T base;
                if (sizeof(T) == 8) {
                    base = static_cast<T>(__double_as_longlong(params[0]));
                } else {
                    base = static_cast<T>(__double2ll_rn(params[0]));
                }
                result = base;
            } else {
                result = computePredictionPoly<T>(model_type, params, local_idx);
            }
            output[global_idx] = result;
        }
        return;
    }

    // Calculate words per lane and per mini-vector
    int bits_per_lane = VALUES_PER_THREAD * delta_bits;
    int words_per_lane = (bits_per_lane + 31) / 32;
    words_per_lane = min(words_per_lane + 1, MAX_WORDS_PER_LANE);  // +1 for extraction safety
    int words_per_mv = calcWordsPerLane(delta_bits) * WARP_SIZE;

    // Process mini-vectors
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // Mini-vector word base in transposed array
        int64_t mv_word_base = transposed_base + static_cast<int64_t>(mv_idx) * words_per_mv;

        // =========================================================
        // COALESCED LOADING: The key optimization!
        // All 32 threads load consecutive addresses
        // =========================================================
        uint32_t lane_words[MAX_WORDS_PER_LANE];

        // Load words using coalesced pattern
        // Word W for lane L is at: mv_word_base + W * 32 + L
        #pragma unroll
        for (int w = 0; w < MAX_WORDS_PER_LANE; w++) {
            if (w < words_per_lane) {
                // COALESCED ACCESS:
                // Thread 0 reads mv_word_base + w*32 + 0
                // Thread 1 reads mv_word_base + w*32 + 1
                // Thread 2 reads mv_word_base + w*32 + 2
                // ... Thread 31 reads mv_word_base + w*32 + 31
                // All 32 addresses are consecutive!
                int64_t addr = mv_word_base + static_cast<int64_t>(w) * WARP_SIZE + lane_id;
                lane_words[w] = __ldg(&transposed_array[addr]);
            } else {
                lane_words[w] = 0;
            }
        }

        // Extract and decompress 8 values from register buffer
        int local_bit = 0;  // Bit position within lane's data

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE + lane_id;

            // Extract from register buffer
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            // Handle 3-word extraction for delta_bits > 32
            if (delta_bits > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
            }
            extracted &= Transposed::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                T base;
                if (sizeof(T) == 8) {
                    base = static_cast<T>(__double_as_longlong(params[0]));
                } else {
                    base = static_cast<T>(__double2ll_rn(params[0]));
                }
                if (delta_bits == 0) {
                    result = base;
                } else {
                    result = base + static_cast<T>(extracted);
                }
            } else if (delta_bits == 0) {
                result = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
            } else {
                T predicted = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
                int64_t delta = Transposed::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }

            output[global_idx] = result;
            local_bit += delta_bits;
        }
    }

    // Handle tail values (sequential layout after mini-vectors)
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    if (tail_size > 0) {
        // Tail word base (after all mini-vector words)
        int64_t tail_word_base = transposed_base + static_cast<int64_t>(num_mv) * words_per_mv;
        int64_t tail_bit_base = tail_word_base * 32;

        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                T base;
                if (sizeof(T) == 8) {
                    base = static_cast<T>(__double_as_longlong(params[0]));
                } else {
                    base = static_cast<T>(__double2ll_rn(params[0]));
                }
                if (delta_bits == 0) {
                    result = base;
                } else {
                    int tail_local_idx = local_idx - tail_start;
                    int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
                    uint64_t delta = Transposed::extract_branchless_64_rt(transposed_array, bit_offset, delta_bits);
                    result = base + static_cast<T>(delta);
                }
            } else if (delta_bits == 0) {
                result = computePredictionPoly<T>(model_type, params, local_idx);
            } else {
                T predicted = computePredictionPoly<T>(model_type, params, local_idx);
                int tail_local_idx = local_idx - tail_start;
                int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
                uint64_t extracted = Transposed::extract_branchless_64_rt(transposed_array, bit_offset, delta_bits);
                int64_t delta = Transposed::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }

            output[global_idx] = result;
        }
    }
}

// ============================================================================
// Random Access Kernel for Transposed Layout
// ============================================================================

/**
 * Random access decompression using TRANSPOSED coordinate mapping
 *
 * For point queries within mini-vectors, we need to:
 * 1. Find which mini-vector and position within it
 * 2. Map to transposed word address: mv_word_base + word_in_lane * 32 + lane_id
 */
template<typename T>
__global__ void decompressTransposedRandomAccess(
    const uint32_t* __restrict__ transposed_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_transposed_offsets,
    int num_partitions,
    const int* __restrict__ query_indices,
    int num_queries,
    T* __restrict__ output)
{
    int qid = threadIdx.x + blockIdx.x * blockDim.x;
    if (qid >= num_queries) return;

    int idx = query_indices[qid];

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

    if (pid < 0) {
        output[qid] = 0;
        return;
    }

    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int32_t start_idx = d_start_indices[pid];
    int32_t num_mv = d_num_mini_vectors[pid];
    int64_t transposed_base = d_transposed_offsets[pid];

    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    int local_idx = idx - start_idx;

    // Calculate words per mini-vector
    int bits_per_lane = VALUES_PER_THREAD * delta_bits;
    int words_per_lane_base = (bits_per_lane + 31) / 32;
    int words_per_mv = words_per_lane_base * WARP_SIZE;

    // Compute bit offset based on whether we're in mini-vector or tail
    int64_t bit_offset;
    int mv_boundary = num_mv * MINI_VECTOR_SIZE;

    if (local_idx < mv_boundary) {
        // In mini-vector section: use TRANSPOSED coordinate mapping
        int mini_vector_idx = local_idx / MINI_VECTOR_SIZE;
        int local_in_mv = local_idx % MINI_VECTOR_SIZE;
        int lane_id = local_in_mv % LANES_PER_MINI_VECTOR;
        int value_idx = local_in_mv / LANES_PER_MINI_VECTOR;

        // TRANSPOSED bit offset calculation:
        // 1. Mini-vector word base
        int64_t mv_word_base = transposed_base + static_cast<int64_t>(mini_vector_idx) * words_per_mv;

        // 2. Bit position within lane's data
        int bit_in_lane = value_idx * delta_bits;
        int word_in_lane = bit_in_lane / 32;
        int bit_in_word = bit_in_lane % 32;

        // 3. TRANSPOSED word address: base + word_in_lane * 32 + lane_id
        int64_t word_addr = mv_word_base + static_cast<int64_t>(word_in_lane) * WARP_SIZE + lane_id;

        bit_offset = word_addr * 32 + bit_in_word;
    } else {
        // In tail section: sequential layout after all mini-vectors
        int tail_local_idx = local_idx - mv_boundary;
        int64_t tail_word_base = transposed_base + static_cast<int64_t>(num_mv) * words_per_mv;
        bit_offset = tail_word_base * 32 + static_cast<int64_t>(tail_local_idx) * delta_bits;
    }

    // Extract and decompress
    if (model_type == MODEL_FOR_BITPACK) {
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(params[0]));
        }

        bool is_direct_copy = (params[0] == 0.0 && params[1] == 0.0);

        if (delta_bits == 0) {
            output[qid] = base;
        } else if (is_direct_copy) {
            uint64_t extracted = Transposed::extract_branchless_64_rt(
                transposed_array, bit_offset, delta_bits);
            if constexpr (std::is_signed<T>::value) {
                int64_t signed_val = Transposed::sign_extend_64(extracted, delta_bits);
                output[qid] = static_cast<T>(signed_val);
            } else {
                output[qid] = static_cast<T>(extracted);
            }
        } else {
            uint64_t delta = Transposed::extract_branchless_64_rt(
                transposed_array, bit_offset, delta_bits);
            output[qid] = base + static_cast<T>(delta);
        }
    } else if (delta_bits == 0) {
        output[qid] = computePredictionPoly<T>(model_type, params, local_idx);
    } else {
        T pred_val = computePredictionPoly<T>(model_type, params, local_idx);
        uint64_t extracted = Transposed::extract_branchless_64_rt(
            transposed_array, bit_offset, delta_bits);
        int64_t delta = Transposed::sign_extend_64(extracted, delta_bits);
        output[qid] = applyDelta(pred_val, delta);
    }
}

// ============================================================================
// Host API Functions
// ============================================================================

/**
 * Decompress all data using Transposed format
 */
template<typename T>
void decompressAll(
    const CompressedDataTransposed<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0) return;

    int np = compressed.num_partitions;

    if (compressed.d_transposed_deltas != nullptr) {
        // Launch one block per partition
        decompressTransposedAllPartitions<T><<<np, BLOCK_SIZE, 0, stream>>>(
            compressed.d_transposed_deltas,
            compressed.d_start_indices,
            compressed.d_end_indices,
            compressed.d_model_types,
            compressed.d_model_params,
            compressed.d_delta_bits,
            compressed.d_num_mini_vectors,
            compressed.d_transposed_offsets,
            np,
            d_output
        );
    }
}

/**
 * Launch random access decompression
 */
template<typename T>
void decompressRandomAccess(
    const CompressedDataTransposed<T>& compressed,
    const int* d_query_indices,
    int num_queries,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0 || num_queries == 0) return;

    int blocks = (num_queries + BLOCK_SIZE - 1) / BLOCK_SIZE;

    decompressTransposedRandomAccess<T><<<blocks, BLOCK_SIZE, 0, stream>>>(
        compressed.d_transposed_deltas,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_num_mini_vectors,
        compressed.d_transposed_offsets,
        compressed.num_partitions,
        d_query_indices,
        num_queries,
        d_output
    );
}

// ============================================================================
// Template Instantiations
// ============================================================================

template void decompressAll<uint32_t>(
    const CompressedDataTransposed<uint32_t>&, uint32_t*, cudaStream_t);
template void decompressAll<int32_t>(
    const CompressedDataTransposed<int32_t>&, int32_t*, cudaStream_t);
template void decompressAll<uint64_t>(
    const CompressedDataTransposed<uint64_t>&, uint64_t*, cudaStream_t);
template void decompressAll<int64_t>(
    const CompressedDataTransposed<int64_t>&, int64_t*, cudaStream_t);

template void decompressRandomAccess<uint32_t>(
    const CompressedDataTransposed<uint32_t>&, const int*, int, uint32_t*, cudaStream_t);
template void decompressRandomAccess<int32_t>(
    const CompressedDataTransposed<int32_t>&, const int*, int, int32_t*, cudaStream_t);
template void decompressRandomAccess<uint64_t>(
    const CompressedDataTransposed<uint64_t>&, const int*, int, uint64_t*, cudaStream_t);
template void decompressRandomAccess<int64_t>(
    const CompressedDataTransposed<int64_t>&, const int*, int, int64_t*, cudaStream_t);

} // namespace Transposed_decoder

#endif // DECODER_TRANSPOSED_CU
