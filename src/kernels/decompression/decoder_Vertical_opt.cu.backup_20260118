/**
 * L3 Vertical-Optimized Decoder
 *
 * GPU decoder with Vertical-inspired optimizations:
 *
 * 1. BRANCHLESS UNPACKING:
 *    - Eliminates conditional branches in bit extraction
 *    - Uniform execution path for all alignments
 *    - Zero warp divergence
 *
 * 2. REGISTER BUFFERING:
 *    - Prefetches 4 words into registers before extraction
 *    - Amortizes memory latency across multiple values
 *    - Reduces L1 cache pressure
 *
 * 3. INTERLEAVED-ONLY STORAGE (v3.0):
 *    - Single format: mini-vectors + sequential tail
 *    - Supports both batch scan and random access
 *    - Random access via coordinate mapping
 *
 * 4. WARP-COOPERATIVE LOADING:
 *    - 32 threads load 32 words cooperatively
 *    - Coalesced 128-byte cache line reads
 *    - Shared memory staging for complex patterns
 *
 * Platform: SM 8.0+ (Ampere and later)
 * Date: 2025-12-07
 */

#ifndef DECODER_Vertical_OPT_CU
#define DECODER_Vertical_OPT_CU

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <type_traits>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../utils/bitpack_utils_Vertical.cuh"

namespace Vertical_decoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int SMEM_WORDS_PER_WARP = 72;  // Shared memory words per warp (extra for 3-word extraction)

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * Apply signed delta to predicted value, handling uint64_t values > INT64_MAX correctly.
 *
 * For uint64_t when predicted > INT64_MAX, casting to int64_t would overflow.
 * Instead, we handle the signed addition directly using unsigned arithmetic.
 */
template<typename T>
__device__ __forceinline__
T applyDelta(T predicted, int64_t delta) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        // For uint64_t: handle signed addition without overflow
        // This works because two's complement arithmetic is equivalent to unsigned
        // arithmetic modulo 2^64, which is what we want here.
        return static_cast<T>(static_cast<uint64_t>(predicted) + static_cast<uint64_t>(delta));
    } else {
        // For signed types or smaller types, direct arithmetic is fine
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

/**
 * Compute prediction using polynomial model (Horner's method)
 *
 * Supports:
 *   MODEL_CONSTANT (0):    y = a0
 *   MODEL_LINEAR (1):      y = a0 + a1*x
 *   MODEL_POLYNOMIAL2 (2): y = a0 + x*(a1 + x*a2)
 *   MODEL_POLYNOMIAL3 (3): y = a0 + x*(a1 + x*(a2 + x*a3))
 *
 * Horner's method minimizes operations and is numerically stable.
 */
template<typename T>
__device__ __forceinline__
T computePredictionPoly(int32_t model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:  // 0
            predicted = params[0];
            break;
        case MODEL_LINEAR:    // 1
            predicted = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        default:
            // Fallback to linear for unknown types
            predicted = params[0] + params[1] * x;
            break;
    }

    // CRITICAL FIX: For uint64_t types, use __double2ull_rn to avoid overflow
    // when predicted > INT64_MAX. __double2ll_rn would overflow to INT64_MIN.
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(predicted));
    } else {
        // Use __double2ll_rn for banker's rounding - matches encoder's rounding
        return static_cast<T>(__double2ll_rn(predicted));
    }
}

// Legacy function for backward compatibility
template<typename T>
__device__ __forceinline__
T computePrediction(double theta0, double theta1, int local_idx) {
    double predicted = theta0 + theta1 * local_idx;
    // CRITICAL FIX: For uint64_t types, use __double2ull_rn to avoid overflow
    // when predicted > INT64_MAX. __double2ll_rn would overflow to INT64_MIN.
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(predicted));
    } else {
        // Use __double2ll_rn for banker's rounding - matches encoder's rounding
        return static_cast<T>(__double2ll_rn(predicted));
    }
}

// ============================================================================
// Sequential Decompression Kernels (Branchless + Register Buffering)
// ============================================================================

/**
 * Branchless sequential decompression with register buffering
 *
 * Each thread decompresses one value using branchless extraction
 * and applies the linear model prediction.
 */
template<typename T>
__global__ void decompressSequentialBranchless(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    T* __restrict__ output,
    int total_values)
{
    int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int g_stride = blockDim.x * gridDim.x;

    for (int idx = g_idx; idx < total_values; idx += g_stride) {
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

        if (pid < 0) continue;

        int32_t model_type = d_model_types[pid];
        int32_t delta_bits = d_delta_bits[pid];
        int64_t bit_offset_base = d_delta_array_bit_offsets[pid];
        int32_t start_idx = d_start_indices[pid];
        int local_idx = idx - start_idx;

        // Load ALL 4 model parameters for polynomial support
        double params[4];
        params[0] = d_model_params[pid * 4];
        params[1] = d_model_params[pid * 4 + 1];
        params[2] = d_model_params[pid * 4 + 2];
        params[3] = d_model_params[pid * 4 + 3];

        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking / DIRECT_COPY mode
            // Detect which mode based on params[0]:
            // - If params[0] == 0.0: L3's MODEL_DIRECT_COPY (delta IS the value, need sign extend)
            // - If params[0] != 0.0: True FOR+BitPack (val = base + unsigned_delta)
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }

            bool is_direct_copy = (params[0] == 0.0 && params[1] == 0.0);

            if (delta_bits == 0) {
                output[idx] = base;
            } else if (is_direct_copy) {
                // L3's MODEL_DIRECT_COPY: delta IS the actual value
                int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
                uint64_t extracted = Vertical::extract_branchless_64_rt(
                    delta_array, bit_offset, delta_bits);
                // Sign extend for signed types, direct cast for unsigned
                if constexpr (std::is_signed<T>::value) {
                    int64_t signed_val = Vertical::sign_extend_64(extracted, delta_bits);
                    output[idx] = static_cast<T>(signed_val);
                } else {
                    output[idx] = static_cast<T>(extracted);
                }
            } else {
                // True FOR+BitPack: val = base + unsigned_delta
                int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
                uint64_t delta = Vertical::extract_branchless_64_rt(
                    delta_array, bit_offset, delta_bits);
                output[idx] = base + static_cast<T>(delta);
            }
        } else if (delta_bits == 0) {
            // Perfect prediction - use polynomial model
            output[idx] = computePredictionPoly<T>(model_type, params, local_idx);
        } else {
            // Normal case: polynomial prediction + signed delta
            T predicted = computePredictionPoly<T>(model_type, params, local_idx);

            int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(
                delta_array, bit_offset, delta_bits);

            // Sign extend
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);

            // Apply prediction
            output[idx] = applyDelta(predicted, delta);
        }
    }
}

/**
 * Warp-cooperative sequential decompression with register buffering
 *
 * Each warp processes 32 consecutive values:
 * 1. Thread 0 loads partition metadata
 * 2. All threads cooperatively load delta words to shared memory
 * 3. Each thread extracts and decompresses its value
 */
template<typename T>
__global__ void decompressSequentialWarpCooperative(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    T* __restrict__ output,
    int total_values)
{
    __shared__ uint32_t smem_delta[SMEM_WORDS_PER_WARP * (BLOCK_SIZE / WARP_SIZE)];
    __shared__ int32_t smem_pid[BLOCK_SIZE / WARP_SIZE];
    __shared__ int64_t smem_bit_base[BLOCK_SIZE / WARP_SIZE];
    __shared__ double smem_params[4 * (BLOCK_SIZE / WARP_SIZE)];  // All 4 model params
    __shared__ int32_t smem_delta_bits[BLOCK_SIZE / WARP_SIZE];
    __shared__ int32_t smem_model_type[BLOCK_SIZE / WARP_SIZE];
    __shared__ int32_t smem_start_idx[BLOCK_SIZE / WARP_SIZE];

    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // Each warp processes 32 values starting at warp_base_idx
    int warp_base_idx = (blockIdx.x * num_warps + warp_id) * WARP_SIZE;

    if (warp_base_idx >= total_values) return;

    int my_idx = warp_base_idx + lane_id;

    // Thread 0 of each warp finds the partition
    if (lane_id == 0) {
        int left = 0, right = num_partitions - 1;
        int pid = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (warp_base_idx >= d_start_indices[mid] && warp_base_idx < d_end_indices[mid]) {
                pid = mid;
                break;
            } else if (warp_base_idx < d_start_indices[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        smem_pid[warp_id] = pid;
        if (pid >= 0) {
            smem_bit_base[warp_id] = d_delta_array_bit_offsets[pid];
            // Load ALL 4 model parameters
            smem_params[warp_id * 4 + 0] = d_model_params[pid * 4];
            smem_params[warp_id * 4 + 1] = d_model_params[pid * 4 + 1];
            smem_params[warp_id * 4 + 2] = d_model_params[pid * 4 + 2];
            smem_params[warp_id * 4 + 3] = d_model_params[pid * 4 + 3];
            smem_delta_bits[warp_id] = d_delta_bits[pid];
            smem_model_type[warp_id] = d_model_types[pid];
            smem_start_idx[warp_id] = d_start_indices[pid];
        }
    }
    __syncwarp();

    int pid = smem_pid[warp_id];

    // Calculate validity BEFORE any divergent returns
    bool is_valid = (pid >= 0) && (my_idx < total_values);

    // If no thread in this warp is valid, all can return
    unsigned valid_mask = __ballot_sync(0xffffffff, is_valid);
    if (valid_mask == 0) return;

    // From here, some threads may be invalid but must participate in sync
    int64_t bit_base = smem_bit_base[warp_id];
    double params[4];
    params[0] = smem_params[warp_id * 4 + 0];
    params[1] = smem_params[warp_id * 4 + 1];
    params[2] = smem_params[warp_id * 4 + 2];
    params[3] = smem_params[warp_id * 4 + 3];
    int delta_bits = smem_delta_bits[warp_id];
    int model_type = smem_model_type[warp_id];
    int start_idx = smem_start_idx[warp_id];

    int local_idx = my_idx - start_idx;

    // Calculate which words this warp needs
    int64_t first_bit = bit_base + static_cast<int64_t>(warp_base_idx - start_idx) * delta_bits;
    int64_t last_bit = first_bit + 32LL * delta_bits - 1;  // Fixed: 32 values, not 33
    int64_t first_word = first_bit >> 5;
    int64_t last_word = (last_bit >> 5) + 2;  // +2 to ensure we have enough for 3-word extraction
    int num_words = static_cast<int>(last_word - first_word + 1);
    num_words = min(num_words, SMEM_WORDS_PER_WARP);

    // Cooperative load to shared memory - ALL threads participate
    uint32_t* warp_smem = smem_delta + warp_id * SMEM_WORDS_PER_WARP;
    for (int w = lane_id; w < num_words; w += WARP_SIZE) {
        warp_smem[w] = __ldg(&delta_array[first_word + w]);
    }
    __syncwarp();

    // Only valid threads extract and write output
    if (!is_valid) return;

    if (model_type == MODEL_FOR_BITPACK) {
        // FOR+BitPacking / DIRECT_COPY mode
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(params[0]));
        }

        bool is_direct_copy = (params[0] == 0.0 && params[1] == 0.0);

        if (delta_bits == 0) {
            output[my_idx] = base;
        } else {
            int64_t my_bit = first_bit + static_cast<int64_t>(lane_id) * delta_bits;
            int64_t local_bit = my_bit - (first_word << 5);
            int word_in_smem = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            // Handle extraction for values that may span up to 3 words (for delta_bits > 32)
            uint64_t combined = (static_cast<uint64_t>(warp_smem[word_in_smem + 1]) << 32) |
                               warp_smem[word_in_smem];
            uint64_t extracted = (combined >> bit_in_word);

            // If we need more bits from a third word (delta_bits > 32 and misaligned)
            if (delta_bits > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                uint64_t third_word = warp_smem[word_in_smem + 2];
                extracted |= (third_word << bits_from_first_two);
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            if (is_direct_copy) {
                // L3's MODEL_DIRECT_COPY: delta IS the actual value
                if constexpr (std::is_signed<T>::value) {
                    int64_t signed_val = Vertical::sign_extend_64(extracted, delta_bits);
                    output[my_idx] = static_cast<T>(signed_val);
                } else {
                    output[my_idx] = static_cast<T>(extracted);
                }
            } else {
                // True FOR+BitPack: val = base + unsigned_delta
                output[my_idx] = base + static_cast<T>(extracted);
            }
        }
    } else if (delta_bits == 0) {
        // Perfect prediction - use polynomial model
        output[my_idx] = computePredictionPoly<T>(model_type, params, local_idx);
    } else {
        // Normal case: polynomial prediction + signed delta
        T predicted = computePredictionPoly<T>(model_type, params, local_idx);

        int64_t my_bit = first_bit + static_cast<int64_t>(lane_id) * delta_bits;
        int64_t local_bit = my_bit - (first_word << 5);
        int word_in_smem = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        // Handle extraction for values that may span up to 3 words
        uint64_t combined = (static_cast<uint64_t>(warp_smem[word_in_smem + 1]) << 32) |
                           warp_smem[word_in_smem];
        uint64_t extracted = (combined >> bit_in_word);

        // If we need more bits from a third word (delta_bits > 32 and misaligned)
        if (delta_bits > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < delta_bits) {
            int bits_from_first_two = 64 - bit_in_word;
            uint64_t third_word = warp_smem[word_in_smem + 2];
            extracted |= (third_word << bits_from_first_two);
        }
        extracted &= Vertical::mask64_rt(delta_bits);

        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
        output[my_idx] = applyDelta(predicted, delta);
    }
}

// ============================================================================
// Interleaved Decompression Kernels
// ============================================================================

/**
 * Interleaved mini-vector decompression
 *
 * Each block processes one mini-vector (256 values).
 * Each thread handles 8 values in its lane.
 * Output is written in sequential order for cache efficiency.
 */
template<typename T>
__global__ void decompressInterleavedMiniVector(
    const uint32_t* __restrict__ interleaved_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    // Shared memory for partition metadata
    __shared__ int s_partition_start;
    __shared__ int s_delta_bits;
    __shared__ double s_params[4];  // All 4 model params
    __shared__ int64_t s_interleaved_base;
    __shared__ int s_model_type;
    __shared__ int s_mv_in_partition;

    // Thread 0 finds the partition for this mini-vector
    if (threadIdx.x == 0) {
        int mv_global = blockIdx.x;
        int cumulative_mv = 0;

        for (int p = 0; p < num_partitions; p++) {
            int num_mv = d_num_mini_vectors[p];
            if (mv_global < cumulative_mv + num_mv) {
                s_partition_start = d_start_indices[p];
                s_delta_bits = d_delta_bits[p];
                s_params[0] = d_model_params[p * 4];
                s_params[1] = d_model_params[p * 4 + 1];
                s_params[2] = d_model_params[p * 4 + 2];
                s_params[3] = d_model_params[p * 4 + 3];
                s_interleaved_base = d_interleaved_offsets[p];
                s_model_type = d_model_types[p];
                s_mv_in_partition = mv_global - cumulative_mv;
                break;
            }
            cumulative_mv += num_mv;
        }
    }
    __syncthreads();

    int partition_start = s_partition_start;
    int delta_bits = s_delta_bits;
    double params[4];
    params[0] = s_params[0];
    params[1] = s_params[1];
    params[2] = s_params[2];
    params[3] = s_params[3];
    int64_t interleaved_base = s_interleaved_base;
    int model_type = s_model_type;
    int mv_idx = s_mv_in_partition;

    int lane_id = threadIdx.x;
    if (lane_id >= WARP_SIZE) return;

    // Global index where this mini-vector starts
    int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

    // Calculate bit position for this lane's data
    int64_t mv_bit_base = (interleaved_base << 5) +
                         static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;

    // Load lane data into registers (branchless prefetch)
    int64_t lane_word_start = lane_bit_start >> 5;
    int bits_per_lane = VALUES_PER_THREAD * delta_bits;
    int words_per_lane = (bits_per_lane + 31 + 32) / 32;  // +31 for rounding, +32 for alignment padding
    words_per_lane = min(words_per_lane, 20);  // Cap at 20 words (enough for 8x64-bit values with misalignment)

    uint32_t lane_words[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        lane_words[i] = (i < words_per_lane) ? __ldg(&interleaved_array[lane_word_start + i]) : 0;
    }

    // Extract and decompress 8 values
    int local_bit = lane_bit_start & 31;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        // Global output index for this value
        int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
        int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE + lane_id;

        // Extract from register buffer
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint64_t extracted = (combined >> bit_in_word);

        // Handle 3-word extraction for delta_bits > 32 with misalignment
        if (delta_bits > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < delta_bits) {
            int bits_from_first_two = 64 - bit_in_word;
            extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
        }
        extracted &= Vertical::mask64_rt(delta_bits);

        T result;
        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: val = base + delta (unsigned delta)
            // Handle delta_bits == 0 case (all values equal base)
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
            // Perfect prediction - use polynomial model
            result = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
        } else {
            // Normal case: polynomial prediction + signed delta
            T predicted = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
            result = applyDelta(predicted, delta);
        }

        output[global_idx] = result;
        local_bit += delta_bits;
    }
}

/**
 * OPTIMIZED: Decompress all partitions using interleaved format
 *
 * One block per partition. Each block:
 * 1. Loads partition metadata once
 * 2. Processes all mini-vectors in the partition
 * 3. Each thread handles its lane across all mini-vectors
 *
 * This avoids the overhead of:
 * - Host-side cudaMemcpy to count mini-vectors
 * - Per-mini-vector block launches
 * - Redundant sequential decompression
 */
template<typename T>
__global__ void decompressInterleavedAllPartitions(
    const uint32_t* __restrict__ interleaved_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Load partition metadata (all threads load the same, cached in L1)
    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    int model_type = d_model_types[pid];

    // Load model parameters
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    // Skip only if no interleaved format available
    // Note: num_mv == 0 means all values are in tail - we still need to decode them!
    if (interleaved_base < 0) {
        // No interleaved data - use prediction only (delta_bits must be 0)
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
                result = base;  // Only valid when delta_bits == 0
            } else {
                result = computePredictionPoly<T>(model_type, params, local_idx);
            }
            output[global_idx] = result;
        }
        return;
    }

    // Process mini-vectors: each thread in warp handles its lane
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Each warp processes multiple mini-vectors
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        // Global index where this mini-vector starts
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // Calculate bit position for this lane's data
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;

        // Load lane data into registers
        int64_t lane_word_start = lane_bit_start >> 5;
        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 31 + 32) / 32;
        words_per_lane = min(words_per_lane, 20);

        uint32_t lane_words[20];
        #pragma unroll
        for (int i = 0; i < 20; i++) {
            lane_words[i] = (i < words_per_lane) ? __ldg(&interleaved_array[lane_word_start + i]) : 0;
        }

        // Extract and decompress 8 values
        int local_bit = lane_bit_start & 31;

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

            if (delta_bits > 32 && bit_in_word > 0 && (32 - bit_in_word + 32) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
            }
            extracted &= Vertical::mask64_rt(delta_bits);

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
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }

            output[global_idx] = result;
            local_bit += delta_bits;
        }
    }

    // Handle tail values (not in mini-vectors)
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    // Use sequential approach for tail
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;

        // For tail, we need to read from sequential format
        // But since interleaved format packs tails sequentially after mini-vectors,
        // we can still decode them from the interleaved array

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
                // Tail bits start after all mini-vectors
                int64_t tail_bit_base = (interleaved_base << 5) +
                                       static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
                int tail_local_idx = local_idx - tail_start;
                int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
                uint64_t delta = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                result = base + static_cast<T>(delta);
            }
        } else if (delta_bits == 0) {
            result = computePredictionPoly<T>(model_type, params, local_idx);
        } else {
            T predicted = computePredictionPoly<T>(model_type, params, local_idx);
            int64_t tail_bit_base = (interleaved_base << 5) +
                                   static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
            int tail_local_idx = local_idx - tail_start;
            int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
            result = applyDelta(predicted, delta);
        }

        output[global_idx] = result;
    }
}

/**
 * Batch decompression with automatic path selection
 *
 * Selects between sequential and interleaved based on:
 * - Partition size (large → interleaved)
 * - Access pattern (provided via hints)
 * - Format availability
 */
template<typename T>
__global__ void decompressBatchAdaptive(
    const uint32_t* __restrict__ sequential_deltas,
    const uint32_t* __restrict__ interleaved_deltas,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    int partition_id,  // Which partition to decompress (-1 for all)
    T* __restrict__ output,
    int output_offset)
{
    // For now, use sequential path (interleaved handled by separate kernel)
    // This kernel handles partitions without interleaved format

    int pid = partition_id;
    if (pid < 0) {
        // Full scan mode: each block handles one partition
        pid = blockIdx.x;
    }

    if (pid >= num_partitions) return;

    int start_idx = d_start_indices[pid];
    int end_idx = d_end_indices[pid];
    int partition_size = end_idx - start_idx;

    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int64_t bit_base = d_delta_array_bit_offsets[pid];

    // Load ALL 4 model parameters for polynomial support
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    // Each thread processes multiple values
    for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = start_idx + local_idx;
        int out_idx = output_offset + global_idx;

        T result;
        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: val = base + delta (unsigned delta)
            // Handle delta_bits == 0 case (all values equal base)
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }
            if (delta_bits == 0) {
                result = base;
            } else {
                int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
                uint64_t delta = Vertical::extract_branchless_64_rt(
                    sequential_deltas, bit_offset, delta_bits);
                result = base + static_cast<T>(delta);
            }
        } else if (delta_bits == 0) {
            // Perfect prediction - use polynomial model
            result = computePredictionPoly<T>(model_type, params, local_idx);
        } else {
            // Normal case: polynomial prediction + signed delta
            T predicted = computePredictionPoly<T>(model_type, params, local_idx);
            int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(
                sequential_deltas, bit_offset, delta_bits);
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
            result = applyDelta(predicted, delta);
        }

        output[out_idx] = result;
    }
}

// ============================================================================
// Random Access Kernel (Optimized with Register Buffering)
// ============================================================================

/**
 * Random access decompression with register buffering
 *
 * For point queries and small ranges.
 * Uses sequential format for O(1) single-value access.
 * Supports polynomial models (LINEAR, POLYNOMIAL2, POLYNOMIAL3).
 */
template<typename T>
__global__ void decompressRandomAccess(
    const uint32_t* __restrict__ sequential_deltas,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    const int* __restrict__ query_indices,  // Indices to fetch
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
        output[qid] = 0;  // Invalid index
        return;
    }

    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int64_t bit_base = d_delta_array_bit_offsets[pid];
    int32_t start_idx = d_start_indices[pid];

    // Load ALL 4 model parameters for polynomial support
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    int local_idx = idx - start_idx;

    if (model_type == MODEL_FOR_BITPACK) {
        // FOR+BitPacking / DIRECT_COPY mode
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
            // L3's MODEL_DIRECT_COPY: delta IS the actual value
            int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(
                sequential_deltas, bit_offset, delta_bits);
            if constexpr (std::is_signed<T>::value) {
                int64_t signed_val = Vertical::sign_extend_64(extracted, delta_bits);
                output[qid] = static_cast<T>(signed_val);
            } else {
                output[qid] = static_cast<T>(extracted);
            }
        } else {
            // True FOR+BitPack: val = base + unsigned_delta
            int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t delta = Vertical::extract_branchless_64_rt(
                sequential_deltas, bit_offset, delta_bits);
            output[qid] = base + static_cast<T>(delta);
        }
    } else if (delta_bits == 0) {
        // Perfect prediction - use polynomial model
        output[qid] = computePredictionPoly<T>(model_type, params, local_idx);
    } else {
        // Normal case: polynomial prediction + signed delta
        T pred_val = computePredictionPoly<T>(model_type, params, local_idx);
        int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
        uint64_t extracted = Vertical::extract_branchless_64_rt(
            sequential_deltas, bit_offset, delta_bits);
        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
        output[qid] = static_cast<T>(static_cast<int64_t>(pred_val) + delta);
    }
}

/**
 * Random access decompression using INTERLEAVED format (v3.0)
 *
 * Reads from unified interleaved array (mini-vectors + tail).
 * For each query index:
 *   1. Find partition via binary search
 *   2. Determine if in mini-vector or tail section
 *   3. Compute bit offset using coordinate mapping
 *   4. Extract and decompress
 *
 * Coordinate mapping for mini-vectors:
 *   local_idx → (mini_vector_idx, lane_id, value_idx)
 *   mini_vector_idx = local_idx / 256
 *   local_in_mv = local_idx % 256
 *   lane_id = local_in_mv % 32
 *   value_idx = local_in_mv / 32
 */
template<typename T>
__global__ void decompressRandomAccessInterleaved(
    const uint32_t* __restrict__ interleaved_deltas,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
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
        output[qid] = 0;  // Invalid index
        return;
    }

    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int32_t start_idx = d_start_indices[pid];
    int32_t num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];

    // Load model parameters
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    int local_idx = idx - start_idx;

    // Compute bit offset based on whether we're in mini-vector or tail
    int64_t bit_offset;
    int mv_boundary = num_mv * MINI_VECTOR_SIZE;

    if (local_idx < mv_boundary) {
        // In mini-vector section: use interleaved coordinate mapping
        int mini_vector_idx = local_idx / MINI_VECTOR_SIZE;
        int local_in_mv = local_idx % MINI_VECTOR_SIZE;
        int lane_id = local_in_mv % LANES_PER_MINI_VECTOR;
        int value_idx = local_in_mv / LANES_PER_MINI_VECTOR;

        // Bit offset = base + mv_offset + lane_offset + value_offset
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mini_vector_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_offset = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int64_t value_bit_offset = static_cast<int64_t>(value_idx) * delta_bits;
        bit_offset = mv_bit_base + lane_bit_offset + value_bit_offset;
    } else {
        // In tail section: sequential layout after all mini-vectors
        int tail_local_idx = local_idx - mv_boundary;
        int64_t mv_total_bits = static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
        bit_offset = (interleaved_base << 5) + mv_total_bits +
                    static_cast<int64_t>(tail_local_idx) * delta_bits;
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
            uint64_t extracted = Vertical::extract_branchless_64_rt(
                interleaved_deltas, bit_offset, delta_bits);
            if constexpr (std::is_signed<T>::value) {
                int64_t signed_val = Vertical::sign_extend_64(extracted, delta_bits);
                output[qid] = static_cast<T>(signed_val);
            } else {
                output[qid] = static_cast<T>(extracted);
            }
        } else {
            uint64_t delta = Vertical::extract_branchless_64_rt(
                interleaved_deltas, bit_offset, delta_bits);
            output[qid] = base + static_cast<T>(delta);
        }
    } else if (delta_bits == 0) {
        output[qid] = computePredictionPoly<T>(model_type, params, local_idx);
    } else {
        T pred_val = computePredictionPoly<T>(model_type, params, local_idx);
        uint64_t extracted = Vertical::extract_branchless_64_rt(
            interleaved_deltas, bit_offset, delta_bits);
        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
        output[qid] = static_cast<T>(static_cast<int64_t>(pred_val) + delta);
    }
}

// ============================================================================
// Optimized Per-Partition Decoder (Original L3 Architecture + Branchless)
// ============================================================================

/**
 * Optimized decoder using one-warp-per-partition design
 *
 * This approach eliminates:
 * 1. Binary search (warp_id directly maps to partition)
 * 2. Shared memory for metadata (loaded directly per-warp)
 * 3. Most __syncwarp() calls
 *
 * Combined with branchless extraction from Vertical for best performance.
 *
 * Supports polynomial models (LINEAR, POLYNOMIAL2, POLYNOMIAL3) using Horner's method.
 *
 * NOTE: This kernel uses LEGACY d_delta_array_bit_offsets for L3 compatibility.
 *       For new code, use decompressInterleavedAllPartitions instead.
 */
template<typename T>
__global__ void decompressPerPartitionBranchless(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    // Each warp handles one partition
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread_id / WARP_SIZE;
    int lane_id = global_thread_id % WARP_SIZE;

    if (warp_id >= num_partitions) return;

    // Load partition metadata (each thread loads the same data, cached in L1)
    int32_t start_idx = d_start_indices[warp_id];
    int32_t end_idx = d_end_indices[warp_id];
    int32_t model_type = d_model_types[warp_id];
    int32_t delta_bits = d_delta_bits[warp_id];
    int64_t bit_base = d_delta_array_bit_offsets[warp_id];

    // Load ALL 4 model parameters for polynomial support
    double params[4];
    params[0] = d_model_params[warp_id * 4];
    params[1] = d_model_params[warp_id * 4 + 1];
    params[2] = d_model_params[warp_id * 4 + 2];
    params[3] = d_model_params[warp_id * 4 + 3];

    int partition_size = end_idx - start_idx;

    // Each thread processes multiple values within the partition
    for (int local_idx = lane_id; local_idx < partition_size; local_idx += WARP_SIZE) {
        int global_idx = start_idx + local_idx;

        T result;
        if (model_type == MODEL_FOR_BITPACK) {  // MODEL_FOR_BITPACK == MODEL_DIRECT_COPY == 4
            // FOR+BitPacking / DIRECT_COPY mode
            // Detect which mode based on params[0]:
            // - If params[0] == 0.0: L3's MODEL_DIRECT_COPY (delta IS the value, need sign extend)
            // - If params[0] != 0.0: True FOR+BitPack (val = base + unsigned_delta)
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }

            bool is_direct_copy = (params[0] == 0.0 && params[1] == 0.0);

            if (delta_bits == 0) {
                result = base;
            } else if (is_direct_copy) {
                // L3's MODEL_DIRECT_COPY: delta IS the actual value (with sign extension for signed types)
                int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
                uint64_t extracted = Vertical::extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
                // Sign extend for signed types, direct cast for unsigned
                if constexpr (std::is_signed<T>::value) {
                    int64_t signed_val = Vertical::sign_extend_64(extracted, delta_bits);
                    result = static_cast<T>(signed_val);
                } else {
                    result = static_cast<T>(extracted);
                }
            } else {
                // True FOR+BitPack: val = base + unsigned_delta
                int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
                uint64_t delta = Vertical::extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
                result = base + static_cast<T>(delta);
            }
        } else if (delta_bits == 0) {
            // Perfect prediction - use polynomial model
            result = computePredictionPoly<T>(model_type, params, local_idx);
        } else {
            // Normal case: polynomial prediction + signed delta
            T pred_val = computePredictionPoly<T>(model_type, params, local_idx);

            int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
            int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);

            result = static_cast<T>(static_cast<int64_t>(pred_val) + delta);
        }

        output[global_idx] = result;
    }
}

/**
 * Launch optimized per-partition decoder
 */
template<typename T>
void launchDecompressPerPartitionBranchless(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0) return;

    // One warp per partition
    int warps_needed = compressed.num_partitions;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;

    decompressPerPartitionBranchless<T><<<blocks, threads_per_block, 0, stream>>>(
        compressed.d_sequential_deltas,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_delta_array_bit_offsets,
        compressed.num_partitions,
        d_output
    );
}

// ============================================================================
// Host API Functions
// ============================================================================

/**
 * Decompress all data using the best available path
 */
template<typename T>
void decompressAll(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    DecompressMode mode,
    cudaStream_t stream = 0)
{
    if (compressed.num_partitions == 0) return;

    int total_values = compressed.total_values;
    int np = compressed.num_partitions;

    if (mode == DecompressMode::INTERLEAVED &&
        compressed.d_interleaved_deltas != nullptr &&
        compressed.total_interleaved_partitions > 0)
    {
        // OPTIMIZED: Use pre-computed total_interleaved_partitions instead of cudaMemcpy
        // Each interleaved partition has (partition_size / 256) mini-vectors
        // For fixed partition sizes, we can estimate total_mini_vectors efficiently

        // Calculate total mini-vectors: for fixed partition size P, each partition
        // contributes P/256 mini-vectors. With np partitions, total = np * (P/256)
        // Since partition sizes may vary, we use a kernel to decompress all partitions

        // Launch one block per partition, each block handles its mini-vectors internally
        decompressInterleavedAllPartitions<T><<<np, BLOCK_SIZE, 0, stream>>>(
            compressed.d_interleaved_deltas,
            compressed.d_start_indices,
            compressed.d_end_indices,
            compressed.d_model_types,
            compressed.d_model_params,
            compressed.d_delta_bits,
            compressed.d_num_mini_vectors,
            compressed.d_interleaved_offsets,
            np,
            d_output
        );
    }
    else if (mode == DecompressMode::BRANCHLESS)
    {
        // v3.0: BRANCHLESS now uses INTERLEAVED format (sequential removed)
        // Fall back to INTERLEAVED mode which handles both mini-vectors and tail
        decompressInterleavedAllPartitions<T><<<np, BLOCK_SIZE, 0, stream>>>(
            compressed.d_interleaved_deltas,
            compressed.d_start_indices,
            compressed.d_end_indices,
            compressed.d_model_types,
            compressed.d_model_params,
            compressed.d_delta_bits,
            compressed.d_num_mini_vectors,
            compressed.d_interleaved_offsets,
            np,
            d_output
        );
    }
    else if (mode == DecompressMode::SEQUENTIAL)
    {
        // DEPRECATED: Sequential format removed in v3.0
        // Fall back to INTERLEAVED mode which handles both mini-vectors and tail
        decompressInterleavedAllPartitions<T><<<np, BLOCK_SIZE, 0, stream>>>(
            compressed.d_interleaved_deltas,
            compressed.d_start_indices,
            compressed.d_end_indices,
            compressed.d_model_types,
            compressed.d_model_params,
            compressed.d_delta_bits,
            compressed.d_num_mini_vectors,
            compressed.d_interleaved_offsets,
            np,
            d_output
        );
    }
    else  // AUTO mode
    {
        // v3.0: Default to INTERLEAVED (only format available)
        decompressInterleavedAllPartitions<T><<<np, BLOCK_SIZE, 0, stream>>>(
            compressed.d_interleaved_deltas,
            compressed.d_start_indices,
            compressed.d_end_indices,
            compressed.d_model_types,
            compressed.d_model_params,
            compressed.d_delta_bits,
            compressed.d_num_mini_vectors,
            compressed.d_interleaved_offsets,
            np,
            d_output
        );
    }
}

/**
 * Decompress specific indices (random access) - v3.0 Interleaved format
 *
 * Uses coordinate mapping to access values from interleaved array.
 */
template<typename T>
void decompressIndices(
    const CompressedDataVertical<T>& compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (num_indices == 0) return;

    int blocks = (num_indices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Use interleaved random access kernel (v3.0)
    decompressRandomAccessInterleaved<T><<<blocks, BLOCK_SIZE, 0, stream>>>(
        compressed.d_interleaved_deltas,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_num_mini_vectors,
        compressed.d_interleaved_offsets,
        compressed.num_partitions,
        d_indices,
        num_indices,
        d_output
    );
}

/**
 * Decompress single value (optimized for minimum latency) - v3.0 Interleaved format
 */
template<typename T>
T decompressSingleValue(
    const CompressedDataVertical<T>& compressed,
    int index,
    cudaStream_t stream = 0)
{
    int* d_index;
    T* d_result;

    cudaMalloc(&d_index, sizeof(int));
    cudaMalloc(&d_result, sizeof(T));

    cudaMemcpyAsync(d_index, &index, sizeof(int), cudaMemcpyHostToDevice, stream);

    // Use interleaved random access kernel (v3.0)
    decompressRandomAccessInterleaved<T><<<1, 1, 0, stream>>>(
        compressed.d_interleaved_deltas,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_num_mini_vectors,
        compressed.d_interleaved_offsets,
        compressed.num_partitions,
        d_index,
        1,
        d_result
    );

    T result;
    cudaMemcpyAsync(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(d_index);
    cudaFree(d_result);

    return result;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void decompressAll<uint32_t>(
    const CompressedDataVertical<uint32_t>&, uint32_t*, DecompressMode, cudaStream_t);
template void decompressAll<uint64_t>(
    const CompressedDataVertical<uint64_t>&, uint64_t*, DecompressMode, cudaStream_t);
template void decompressAll<int32_t>(
    const CompressedDataVertical<int32_t>&, int32_t*, DecompressMode, cudaStream_t);
template void decompressAll<int64_t>(
    const CompressedDataVertical<int64_t>&, int64_t*, DecompressMode, cudaStream_t);

template void decompressIndices<uint32_t>(
    const CompressedDataVertical<uint32_t>&, const int*, int, uint32_t*, cudaStream_t);
template void decompressIndices<uint64_t>(
    const CompressedDataVertical<uint64_t>&, const int*, int, uint64_t*, cudaStream_t);
template void decompressIndices<int32_t>(
    const CompressedDataVertical<int32_t>&, const int*, int, int32_t*, cudaStream_t);
template void decompressIndices<int64_t>(
    const CompressedDataVertical<int64_t>&, const int*, int, int64_t*, cudaStream_t);

}  // namespace Vertical_decoder

// ============================================================================
// L3-Compatible Wrapper (Outside namespace for global visibility)
// ============================================================================

/**
 * Launch Vertical branchless decoder with L3 format data.
 *
 * This wrapper allows SSB queries to use the Vertical branchless decoder
 * without format conversion, since CompressedDataL3 has compatible fields.
 */
template<typename T>
void launchDecompressVerticalL3(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (compressed->num_partitions == 0) return;

    constexpr int WARP_SIZE = 32;
    int warps_needed = compressed->num_partitions;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;

    // Call the branchless kernel directly with L3 format fields
    Vertical_decoder::decompressPerPartitionBranchless<T><<<blocks, threads_per_block, 0, stream>>>(
        compressed->delta_array,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->num_partitions,
        d_output
    );
}

// Explicit instantiations for L3 wrapper
template void launchDecompressVerticalL3<uint32_t>(
    const CompressedDataL3<uint32_t>*, uint32_t*, cudaStream_t);
template void launchDecompressVerticalL3<int32_t>(
    const CompressedDataL3<int32_t>*, int32_t*, cudaStream_t);
template void launchDecompressVerticalL3<uint64_t>(
    const CompressedDataL3<uint64_t>*, uint64_t*, cudaStream_t);
template void launchDecompressVerticalL3<int64_t>(
    const CompressedDataL3<int64_t>*, int64_t*, cudaStream_t);

#endif // DECODER_Vertical_OPT_CU
