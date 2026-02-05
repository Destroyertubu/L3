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
#include "../utils/finite_diff_shared.cuh"

namespace Vertical_decoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr int SMEM_WORDS_PER_WARP = 72;  // Shared memory words per warp (extra for 3-word extraction)

// Adaptive block size selection based on average partition size
// Empirically determined thresholds from H100 benchmarks
__host__ inline int selectOptimalBlockSize(int64_t total_elements, int num_partitions) {
    if (num_partitions == 0) return BLOCK_SIZE;
    int64_t avg_partition_size = total_elements / num_partitions;

    // - Small partitions (<700): block=64 gives +20% to +85% speedup
    // - Medium partitions (700-2000): block=128 is a good balance
    // - Large partitions (>2000): block=256 gives best performance
    if (avg_partition_size < 700) {
        return 64;
    } else if (avg_partition_size < 2000) {
        return 128;
    } else {
        return 256;
    }
}

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
            // Match encoder: explicit FP64 ops (no FMA fusion)
            predicted = FiniteDiff::d_add(params[0], FiniteDiff::d_mul(params[1], x));
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = FiniteDiff::d_add(
                params[0],
                FiniteDiff::d_mul(
                    x,
                    FiniteDiff::d_add(params[1], FiniteDiff::d_mul(x, params[2]))));
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = FiniteDiff::d_add(
                params[0],
                FiniteDiff::d_mul(
                    x,
                    FiniteDiff::d_add(
                        params[1],
                        FiniteDiff::d_mul(
                            x,
                            FiniteDiff::d_add(params[2], FiniteDiff::d_mul(x, params[3]))))));
            break;
        default:
            // Fallback to linear for unknown types
            predicted = FiniteDiff::d_add(params[0], FiniteDiff::d_mul(params[1], x));
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
    double predicted = FiniteDiff::d_add(theta0, FiniteDiff::d_mul(theta1, static_cast<double>(local_idx)));
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
// Shared helpers for V6 (variable params + fast bit reading)
// ============================================================================

__device__ __forceinline__ int64_t getParamBase(const int64_t* __restrict__ d_param_offsets, int pid) {
    return (d_param_offsets == nullptr) ? (static_cast<int64_t>(pid) * 4) : d_param_offsets[pid];
}

__device__ __forceinline__ void loadModelParams(
    int model_type,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,
    int pid,
    double& p0, double& p1, double& p2, double& p3)
{
    p0 = p1 = p2 = p3 = 0.0;

    const int64_t base = getParamBase(d_param_offsets, pid);
    if (d_param_offsets == nullptr) {
        // Fixed layout: always 4 params per partition.
        p0 = d_model_params[base + 0];
        p1 = d_model_params[base + 1];
        p2 = d_model_params[base + 2];
        p3 = d_model_params[base + 3];
        return;
    }

    // Variable layout: only read the params this model actually stores.
    const int cnt = getParamCount(model_type);
    if (cnt >= 1) p0 = d_model_params[base + 0];
    if (cnt >= 2) p1 = d_model_params[base + 1];
    if (cnt >= 3) p2 = d_model_params[base + 2];
    if (cnt >= 4) p3 = d_model_params[base + 3];
}

// Fast sequential bit reader for <=32-bit values (per-thread).
// Reads a contiguous bitstream starting at start_bit.
struct BitReader32 {
    const uint32_t* __restrict__ ptr;
    uint64_t buf;
    int avail;  // number of valid bits in buf

    __device__ __forceinline__
    BitReader32(const uint32_t* __restrict__ words, uint64_t start_bit) {
        const uint32_t* p = words + (start_bit >> 5);

        // Always load two words up front (safe due to +4 padding in allocation).
        uint32_t lo = __ldg(p);
        uint32_t hi = __ldg(p + 1);
        buf = (static_cast<uint64_t>(hi) << 32) | lo;

        int shift = static_cast<int>(start_bit & 31);
        buf >>= shift;
        avail = 64 - shift;

        ptr = p + 2;
    }

    __device__ __forceinline__
    uint32_t read(int bits, uint64_t mask) {
        // bits must be in [1, 32]
        if (avail < bits) {
            uint32_t next = __ldg(ptr++);
            // When avail < bits <= 32, avail <= 31, so this shift is safe (<32).
            buf |= (static_cast<uint64_t>(next) << avail);
            avail += 32;
        }
        uint32_t v = static_cast<uint32_t>(buf & mask);
        buf >>= bits;
        avail -= bits;
        return v;
    }

    __device__ __forceinline__
    void read4(int bits, uint64_t mask, uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3) {
        v0 = read(bits, mask);
        v1 = read(bits, mask);
        v2 = read(bits, mask);
        v3 = read(bits, mask);
    }
};

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
            // FOR+BitPacking: val = base + unsigned_delta (base stored in params[0])
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }

            if (delta_bits == 0) {
                output[idx] = base;
            } else {
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
        // FOR+BitPacking: val = base + unsigned_delta (base stored in params[0])
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(params[0]));
        }

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
            output[my_idx] = base + static_cast<T>(extracted);
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

    // Fixed layout (pid * 4) - legacy V4 path.
    double params[4];
    params[0] = d_model_params[pid * 4 + 0];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        // Shared memory for RLE decoding
        __shared__ uint32_t s_rle_data[72];  // Buffer for RLE data
        __shared__ T s_run_values[256];
        __shared__ int s_run_counts[256];
        __shared__ int s_run_offsets[257];

        int num_runs = static_cast<int>(params[0]);
        int value_bits = static_cast<int>(params[2]);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        // Single run with count_bits=0: traditional CONSTANT
        if (num_runs == 1 && count_bits == 0) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // Load RLE data
        int bits_per_run = value_bits + count_bits;
        int64_t rle_bit_base = interleaved_base << 5;
        int64_t rle_bits_total = static_cast<int64_t>(num_runs) * bits_per_run;
        int rle_words_needed = (rle_bits_total + 31 + 32) / 32;
        rle_words_needed = min(rle_words_needed, 72);
        int64_t rle_first_word = rle_bit_base >> 5;

        for (int w = threadIdx.x; w < rle_words_needed; w += blockDim.x) {
            s_rle_data[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        // Thread 0 decodes runs
        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;
            for (int r = 0; r < num_runs; r++) {
                int64_t bit_offset = static_cast<int64_t>(r) * bits_per_run;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;

                uint64_t combined = (static_cast<uint64_t>(s_rle_data[word_idx + 1]) << 32) |
                                   s_rle_data[word_idx];
                uint64_t packed = combined >> bit_in_word;

                if (bit_in_word > 0 && (64 - bit_in_word) < bits_per_run && (word_idx + 2) < rle_words_needed) {
                    int bits_from_first_two = 64 - bit_in_word;
                    packed |= (static_cast<uint64_t>(s_rle_data[word_idx + 2]) << bits_from_first_two);
                }

                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));

                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);
                s_run_counts[r] = count;
                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        // All threads expand runs using binary search
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int left = 0, right = num_runs - 1;
            while (left < right) {
                int mid = (left + right + 1) >> 1;
                if (s_run_offsets[mid] <= local_idx) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

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
        // OPTIMIZED: Use minimal buffer size based on data type
        // For uint32_t (delta_bits <= 32): max need = (8*32+63)/32 = 10 words
        // For uint64_t (delta_bits <= 64): max need = (8*64+63)/32 = 18 words
        constexpr int MAX_BUFFER_WORDS = (sizeof(T) <= 4) ? 10 : 18;

        int64_t lane_word_start = lane_bit_start >> 5;
        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 31 + 32) / 32;
        words_per_lane = min(words_per_lane, MAX_BUFFER_WORDS);

        uint32_t lane_words[MAX_BUFFER_WORDS];
        #pragma unroll
        for (int i = 0; i < MAX_BUFFER_WORDS; i++) {
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
 * Optimized version of decompressInterleavedAllPartitions
 *
 * Key optimizations over original version:
 * 1. Uses shared memory instead of 20-word register buffer
 * 2. Warp-cooperative loading for better memory coalescing
 * 3. Reduced register pressure (target: ~34 registers vs 40)
 * 4. Better thread utilization during tail processing
 *
 * Expected improvements:
 * - Higher occupancy due to lower register usage
 * - Better instruction-level parallelism
 * - More active threads per warp
 */
template<typename T>
__global__ void decompressInterleavedAllPartitionsOpt(
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

    // Load partition metadata
    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    int model_type = d_model_types[pid];

    // Fixed layout (pid * 4) - legacy Opt path.
    double params[4];
    params[0] = d_model_params[pid * 4 + 0];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        __shared__ uint32_t s_rle_data[72];
        __shared__ T s_run_values[256];
        __shared__ int s_run_offsets[257];

        int num_runs = static_cast<int>(params[0]);
        int value_bits = static_cast<int>(params[2]);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        if (num_runs == 1 && count_bits == 0) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        int bits_per_run = value_bits + count_bits;
        int64_t rle_bit_base = interleaved_base << 5;
        int64_t rle_bits_total = static_cast<int64_t>(num_runs) * bits_per_run;
        int rle_words_needed = (rle_bits_total + 31 + 32) / 32;
        rle_words_needed = min(rle_words_needed, 72);
        int64_t rle_first_word = rle_bit_base >> 5;

        for (int w = threadIdx.x; w < rle_words_needed; w += blockDim.x) {
            s_rle_data[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;
            for (int r = 0; r < num_runs; r++) {
                int64_t bit_offset = static_cast<int64_t>(r) * bits_per_run;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;

                uint64_t combined = (static_cast<uint64_t>(s_rle_data[word_idx + 1]) << 32) |
                                   s_rle_data[word_idx];
                uint64_t packed = combined >> bit_in_word;

                if (bit_in_word > 0 && (64 - bit_in_word) < bits_per_run && (word_idx + 2) < rle_words_needed) {
                    int bits_from_first_two = 64 - bit_in_word;
                    packed |= (static_cast<uint64_t>(s_rle_data[word_idx + 2]) << bits_from_first_two);
                }

                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));

                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);
                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int left = 0, right = num_runs - 1;
            while (left < right) {
                int mid = (left + right + 1) >> 1;
                if (s_run_offsets[mid] <= local_idx) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

    // Handle no interleaved data case
    if (interleaved_base < 0) {
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

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Process mini-vectors
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // Calculate bit position for this lane's data
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;

        // Load lane data into registers - OPTIMIZED: reduced buffer size
        // For 32-bit data with delta_bits <= 32: need max (8*32+63)/32 = 10 words
        // For most compressed data (delta_bits <= 16): need only 6 words
        int64_t lane_word_start = lane_bit_start >> 5;
        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 31 + 32) / 32;
        words_per_lane = min(words_per_lane, 10);  // Reduced from 20 to 10

        // OPTIMIZATION: Use smaller fixed-size buffer (10 words instead of 20)
        // This reduces register pressure significantly
        uint32_t lane_words[10];
        #pragma unroll
        for (int i = 0; i < 10; i++) {
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

            // Handle 3-word case (only for delta_bits > 32, which is rare for uint32_t)
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

    // Handle tail values
    int tail_start = num_mv * MINI_VECTOR_SIZE;

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
 * V4 Optimized: Decompress all partitions using interleaved format
 *
 * Key optimizations over decompressInterleavedAllPartitionsOpt:
 * 1. Shared memory for warp-cooperative loading (reduces register pressure)
 * 2. Warp-cooperative tail decoding for better memory coalescing
 * 3. Pre-computed extraction flags (reduces branch overhead)
 * 4. Lazy parameter loading (FOR_BITPACK only needs 1 param)
 * 5. Pre-computed base value (moved outside inner loop)
 *
 * One block per partition. Each block:
 * 1. Loads partition metadata once
 * 2. Processes all mini-vectors using shared memory buffer
 * 3. Uses warp-cooperative loading for tail values
 */
template<typename T>
__global__ void decompressInterleavedAllPartitionsV4(
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
    // Shared memory: 72 words per warp for lane data + extra for tail
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;  // 256/32 = 8
    __shared__ uint32_t s_warp_words[WARPS_PER_BLOCK][SMEM_WORDS_PER_WARP];

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

    // V4 OPTIMIZATION: Lazy parameter loading
    // FOR_BITPACK only needs param0, other models need all 4
    double params[4];
    params[0] = d_model_params[pid * 4];
    if (model_type != MODEL_FOR_BITPACK) {
        params[1] = d_model_params[pid * 4 + 1];
        params[2] = d_model_params[pid * 4 + 2];
        params[3] = d_model_params[pid * 4 + 3];
    }

    // V4 OPTIMIZATION: Pre-compute base value for FOR_BITPACK
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base_value = static_cast<T>(__double2ll_rn(params[0]));
        }
    }

    // V4 OPTIMIZATION: Pre-compute needs_third_word flag (outside loops)
    bool needs_third_word = (delta_bits > 32);

    // ========== RLE (CONSTANT) partition special handling ==========
    // MODEL_CONSTANT now uses RLE format: (value-base, count) pairs
    if (model_type == MODEL_CONSTANT) {
        // RLE parameters from model_params
        int num_runs = static_cast<int>(params[0]);       // theta0 = num_runs
        int value_bits = static_cast<int>(params[2]);     // theta2 = value_bits
        int count_bits = delta_bits;                       // delta_bits = count_bits

        // Get base_value from params[1]
        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        // Special case: single run with count_bits=0 means all values are same
        if (num_runs == 1 && count_bits == 0) {
            // Traditional CONSTANT: all values are rle_base_value
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // RLE decoding: sequentially parse (value, count) pairs
        // Use shared memory to buffer RLE data
        int bits_per_run = value_bits + count_bits;
        int64_t rle_bit_base = interleaved_base << 5;  // words to bits

        // Load RLE data to shared memory (warp 0)
        uint32_t* rle_smem = s_warp_words[0];
        int64_t rle_bits_total = static_cast<int64_t>(num_runs) * bits_per_run;
        int rle_words_needed = (rle_bits_total + 31 + 32) / 32;
        rle_words_needed = min(rle_words_needed, SMEM_WORDS_PER_WARP);

        int64_t rle_first_word = rle_bit_base >> 5;

        // Cooperative load
        for (int w = threadIdx.x; w < rle_words_needed; w += blockDim.x) {
            rle_smem[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        // Thread 0 decodes and all threads help write
        // First, decode all runs to shared memory (use s_warp_words[1..7] for values/counts)
        __shared__ T s_run_values[256];     // Store up to 256 runs
        __shared__ int s_run_counts[256];
        __shared__ int s_run_offsets[257];  // Prefix sum of counts

        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;

            for (int r = 0; r < num_runs; r++) {
                // Calculate bit position for this run
                int64_t bit_offset = static_cast<int64_t>(r) * bits_per_run;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;

                // Extract (value_delta, count) pair
                uint64_t combined = (static_cast<uint64_t>(rle_smem[word_idx + 1]) << 32) |
                                   rle_smem[word_idx];
                uint64_t packed = combined >> bit_in_word;

                // Handle case where we need a third word
                if (bit_in_word > 0 && (64 - bit_in_word) < bits_per_run && (word_idx + 2) < rle_words_needed) {
                    int bits_from_first_two = 64 - bit_in_word;
                    packed |= (static_cast<uint64_t>(rle_smem[word_idx + 2]) << bits_from_first_two);
                }

                // Extract value_delta and count
                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));

                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);
                s_run_counts[r] = count;
                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        // All threads expand runs to output using binary search
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            // Binary search to find which run this index belongs to
            int left = 0, right = num_runs - 1;
            while (left < right) {
                int mid = (left + right + 1) >> 1;
                if (s_run_offsets[mid] <= local_idx) {
                    left = mid;
                } else {
                    right = mid - 1;
                }
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

    // Skip only if no interleaved format available
    if (interleaved_base < 0) {
        // No interleaved data - use prediction only (delta_bits must be 0)
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value;
            } else {
                result = computePredictionPoly<T>(model_type, params, local_idx);
            }
            output[global_idx] = result;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Get pointer to this warp's shared memory
    uint32_t* warp_smem = s_warp_words[warp_id_in_block];

    // Process mini-vectors: each warp processes one mini-vector at a time
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // Calculate bit position for this mini-vector
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;

        // V4 OPTIMIZATION: Warp-cooperative loading into shared memory
        // Calculate how many words this warp needs for all 256 values
        int64_t mv_first_word = mv_bit_base >> 5;
        int total_bits = MINI_VECTOR_SIZE * delta_bits;
        int words_needed = (total_bits + 31 + 32) / 32;  // +32 for alignment safety
        words_needed = min(words_needed, SMEM_WORDS_PER_WARP);

        // All 32 threads cooperatively load words
        for (int w = lane_id; w < words_needed; w += WARP_SIZE) {
            warp_smem[w] = __ldg(&interleaved_array[mv_first_word + w]);
        }
        __syncwarp();

        // Each thread processes 8 values from its lane
        int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int local_bit = lane_bit_start & 31;
        int base_word_idx = lane_bit_start >> 5;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE + lane_id;

            // Extract from shared memory buffer
            int word_idx = base_word_idx + (local_bit >> 5);
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(warp_smem[word_idx + 1]) << 32) |
                               warp_smem[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            // V4 OPTIMIZATION: Use pre-computed needs_third_word flag
            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(warp_smem[word_idx + 2]) << bits_from_first_two);
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                // V4: Use pre-computed base_value
                if (delta_bits == 0) {
                    result = base_value;
                } else {
                    result = base_value + static_cast<T>(extracted);
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
        __syncwarp();  // Ensure all threads done before reusing shared memory
    }

    // V4 OPTIMIZATION: Warp-cooperative tail decoding
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    if (tail_size > 0 && delta_bits > 0) {
        // Calculate tail bit range
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
        int64_t tail_first_word = tail_bit_base >> 5;
        int tail_bit_offset_in_first_word = tail_bit_base & 31;

        // Calculate words needed for tail (limit to available shared memory)
        int64_t tail_bits_total = static_cast<int64_t>(tail_size) * delta_bits;
        int tail_words_needed = (tail_bits_total + tail_bit_offset_in_first_word + 31 + 32) / 32;
        tail_words_needed = min(tail_words_needed, SMEM_WORDS_PER_WARP);

        // Warp-cooperative load of tail data (all warps participate)
        __syncthreads();  // Ensure mini-vector processing is complete

        // Only first warp loads tail data to shared memory
        if (warp_id_in_block == 0) {
            for (int w = lane_id; w < tail_words_needed; w += WARP_SIZE) {
                warp_smem[w] = __ldg(&interleaved_array[tail_first_word + w]);
            }
        }
        __syncthreads();  // All threads wait for tail data

        // All threads process tail values from warp 0's shared memory
        uint32_t* tail_smem = s_warp_words[0];

        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            int tail_local_idx = local_idx - tail_start;

            // Calculate bit position within loaded shared memory
            int64_t bit_offset_from_base = static_cast<int64_t>(tail_local_idx) * delta_bits;
            int64_t local_bit = tail_bit_offset_in_first_word + bit_offset_from_base;
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                // Extract from shared memory
                uint64_t combined = (static_cast<uint64_t>(tail_smem[word_idx + 1]) << 32) |
                                   tail_smem[word_idx];
                uint64_t extracted = (combined >> bit_in_word);
                if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                    int bits_from_first_two = 64 - bit_in_word;
                    extracted |= (static_cast<uint64_t>(tail_smem[word_idx + 2]) << bits_from_first_two);
                }
                extracted &= Vertical::mask64_rt(delta_bits);
                result = base_value + static_cast<T>(extracted);
            } else {
                T predicted = computePredictionPoly<T>(model_type, params, local_idx);
                uint64_t combined = (static_cast<uint64_t>(tail_smem[word_idx + 1]) << 32) |
                                   tail_smem[word_idx];
                uint64_t extracted = (combined >> bit_in_word);
                if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                    int bits_from_first_two = 64 - bit_in_word;
                    extracted |= (static_cast<uint64_t>(tail_smem[word_idx + 2]) << bits_from_first_two);
                }
                extracted &= Vertical::mask64_rt(delta_bits);
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }
            output[global_idx] = result;
        }
    } else if (tail_size > 0) {
        // delta_bits == 0 case: just use prediction or base
        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value;
            } else {
                result = computePredictionPoly<T>(model_type, params, local_idx);
            }
            output[global_idx] = result;
        }
    }
}

// ============================================================================
// V5 Optimized Decoder - Cache-Optimized Metadata + Dynamic SharedMem + Prefetch
// ============================================================================

/**
 * V5 Optimized Decoder Kernel
 *
 * Optimizations over V4:
 * 1. CACHE-OPTIMIZED METADATA: Single PartitionMetadataV5 load (64B = 1 cache line)
 *    - Reduces L1 misses from 6+ separate array accesses to 1-2 loads
 *    - Expected L1 hit rate improvement: 35% → 70%+
 *
 * 2. DYNAMIC SHARED MEMORY: Allocates exact amount needed
 *    - No more SMEM_WORDS_PER_WARP = 72 limitation
 *    - Can handle any delta_bits without truncation
 *
 * 3. PREFETCH NEXT MINI-VECTOR: Hides memory latency
 *    - While processing current MV, prefetch next MV's data
 *    - Reduces stalls waiting for global memory
 *
 * 4. BANK CONFLICT PADDING: +8 words per warp
 *    - Reduces shared memory bank conflicts
 *    - Slightly better IPC
 *
 * Expected total improvement: 20-35% decode throughput
 */
template<typename T>
__global__ void decompressInterleavedAllPartitionsV5(
    const uint32_t* __restrict__ interleaved_array,
    const PartitionMetadataV5* __restrict__ d_metadata,
    int num_partitions,
    T* __restrict__ output)
{
    // V5: Dynamic shared memory with bank padding
    extern __shared__ uint32_t s_dynamic_smem[];

    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // V5 OPTIMIZATION 1: Single metadata load (64 bytes = 1 cache line)
    PartitionMetadataV5 meta = d_metadata[pid];

    int partition_start = meta.start_idx;
    int partition_size = meta.end_idx - meta.start_idx;
    int delta_bits = meta.delta_bits;
    int num_mv = meta.num_mini_vectors;
    int64_t interleaved_base = meta.interleaved_offset;
    int model_type = meta.model_type;
    const double* params = meta.params;  // Direct access to embedded params

    // Pre-compute values
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base_value = static_cast<T>(__double2ll_rn(params[0]));
        }
    }
    bool needs_third_word = (delta_bits > 32);

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(params[0]);
        int value_bits = static_cast<int>(params[2]);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        if (num_runs == 1 && count_bits == 0) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // RLE decoding using dynamic shared memory
        int bits_per_run = value_bits + count_bits;
        int64_t rle_bit_base = interleaved_base << 5;
        int64_t rle_bits_total = static_cast<int64_t>(num_runs) * bits_per_run;
        int rle_words_needed = (rle_bits_total + 63) / 32;
        int64_t rle_first_word = rle_bit_base >> 5;

        // Use dynamic smem for RLE data
        uint32_t* rle_smem = s_dynamic_smem;
        for (int w = threadIdx.x; w < rle_words_needed; w += blockDim.x) {
            rle_smem[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        // Decode runs (same as V4)
        __shared__ T s_run_values[256];
        __shared__ int s_run_counts[256];
        __shared__ int s_run_offsets[257];

        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;
            for (int r = 0; r < num_runs; r++) {
                int64_t bit_offset = static_cast<int64_t>(r) * bits_per_run;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;

                uint64_t combined = (static_cast<uint64_t>(rle_smem[word_idx + 1]) << 32) | rle_smem[word_idx];
                uint64_t packed = combined >> bit_in_word;
                if (bit_in_word > 0 && (64 - bit_in_word) < bits_per_run && (word_idx + 2) < rle_words_needed) {
                    packed |= (static_cast<uint64_t>(rle_smem[word_idx + 2]) << (64 - bit_in_word));
                }

                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));
                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);
                s_run_counts[r] = count;
                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int left = 0, right = num_runs - 1;
            while (left < right) {
                int mid = (left + right + 1) >> 1;
                if (s_run_offsets[mid] <= local_idx) left = mid;
                else right = mid - 1;
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

    // Skip if no interleaved data
    if (interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result = (model_type == MODEL_FOR_BITPACK) ? base_value :
                       computePredictionPoly<T>(model_type, params, local_idx);
            output[global_idx] = result;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // V5 OPTIMIZATION 2: Dynamic shared memory with bank padding
    // Calculate words needed per warp: (MINI_VECTOR_SIZE * delta_bits) / 32 + padding
    int bits_per_mv = MINI_VECTOR_SIZE * delta_bits;
    int words_per_warp = (bits_per_mv + 31 + 32) / 32 + 8;  // +8 for bank padding
    uint32_t* warp_smem = s_dynamic_smem + warp_id_in_block * words_per_warp;

    // Process mini-vectors with prefetch
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t mv_first_word = mv_bit_base >> 5;
        int words_needed = (bits_per_mv + 31 + 32) / 32;

        // V5 OPTIMIZATION 3: Prefetch next mini-vector while loading current
        int next_mv_idx = mv_idx + warps_per_block;
        if (next_mv_idx < num_mv) {
            int64_t next_mv_bit_base = (interleaved_base << 5) +
                                       static_cast<int64_t>(next_mv_idx) * MINI_VECTOR_SIZE * delta_bits;
            int64_t next_mv_first_word = next_mv_bit_base >> 5;

            // Prefetch first 4 cache lines of next mini-vector
            if (lane_id < 4) {
                asm volatile("prefetch.global.L1 [%0];" : : "l"(&interleaved_array[next_mv_first_word + lane_id * 32]));
            }
        }

        // Load current mini-vector
        for (int w = lane_id; w < words_needed; w += WARP_SIZE) {
            warp_smem[w] = __ldg(&interleaved_array[mv_first_word + w]);
        }
        __syncwarp();

        // Process values (same extraction logic as V4)
        int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int local_bit = lane_bit_start & 31;
        int base_word_idx = lane_bit_start >> 5;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE + lane_id;

            int word_idx = base_word_idx + (local_bit >> 5);
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(warp_smem[word_idx + 1]) << 32) | warp_smem[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                extracted |= (static_cast<uint64_t>(warp_smem[word_idx + 2]) << (64 - bit_in_word));
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = (delta_bits == 0) ? base_value : base_value + static_cast<T>(extracted);
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
        __syncwarp();
    }

    // Tail processing (same as V4 but using dynamic smem)
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    if (tail_size > 0 && delta_bits > 0) {
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
        int64_t tail_first_word = tail_bit_base >> 5;
        int tail_bit_offset_in_first_word = tail_bit_base & 31;

        int64_t tail_bits_total = static_cast<int64_t>(tail_size) * delta_bits;
        int tail_words_needed = (tail_bits_total + tail_bit_offset_in_first_word + 63) / 32;

        __syncthreads();

        // Use first warp's smem for tail
        uint32_t* tail_smem = s_dynamic_smem;
        if (warp_id_in_block == 0) {
            for (int w = lane_id; w < tail_words_needed; w += WARP_SIZE) {
                tail_smem[w] = __ldg(&interleaved_array[tail_first_word + w]);
            }
        }
        __syncthreads();

        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            int tail_local_idx = local_idx - tail_start;

            int64_t bit_offset_from_base = static_cast<int64_t>(tail_local_idx) * delta_bits;
            int64_t local_bit_pos = tail_bit_offset_in_first_word + bit_offset_from_base;
            int word_idx = local_bit_pos >> 5;
            int bit_in_word = local_bit_pos & 31;

            uint64_t combined = (static_cast<uint64_t>(tail_smem[word_idx + 1]) << 32) | tail_smem[word_idx];
            uint64_t extracted = (combined >> bit_in_word);
            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                extracted |= (static_cast<uint64_t>(tail_smem[word_idx + 2]) << (64 - bit_in_word));
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value + static_cast<T>(extracted);
            } else {
                T predicted = computePredictionPoly<T>(model_type, params, local_idx);
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }
            output[global_idx] = result;
        }
    } else if (tail_size > 0) {
        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result = (model_type == MODEL_FOR_BITPACK) ? base_value :
                       computePredictionPoly<T>(model_type, params, local_idx);
            output[global_idx] = result;
        }
    }
}

// ============================================================================
// V5-Optimized Decoder: Static Shared Memory with Delta-Bits Classification
// ============================================================================

/**
 * V5-Opt Decoder: Optimized version with static shared memory
 *
 * Key optimizations over V5:
 * 1. Static shared memory based on delta_bits class (no dynamic allocation overhead)
 * 2. Conditional prefetch only for large delta_bits
 * 3. Optimized metadata loading with __ldg
 * 4. Reduced bank padding for small delta_bits
 *
 * DELTA_BITS_CLASS:
 *   0 = Small (delta_bits <= 16): 48 words/warp, no prefetch, no padding
 *   1 = Medium (delta_bits <= 32): 72 words/warp, minimal prefetch, 4-word padding
 *   2 = Large (delta_bits > 32): 144 words/warp, full prefetch, 8-word padding
 */
template<typename T, int DELTA_BITS_CLASS>
__global__ void decompressInterleavedAllPartitionsV5_Opt(
    const uint32_t* __restrict__ interleaved_array,
    const PartitionMetadataV5* __restrict__ d_metadata,
    int num_partitions,
    T* __restrict__ output)
{
    // Static shared memory configuration based on delta_bits class
    // MINI_VECTOR_SIZE = 256, so:
    //   Small (<=16 bits): 256*16/32 + extra = 128+16 = 144 words total, 48/warp is enough for loading
    //   Medium (<=32 bits): 256*32/32 + extra = 256+16 = 272 words total, 72/warp
    //   Large (>32 bits): 256*64/32 + extra = 512+32 = 544 words total, 144/warp
    constexpr int SMEM_WORDS_PER_WARP = (DELTA_BITS_CLASS == 0) ? 48 :
                                        (DELTA_BITS_CLASS == 1) ? 72 : 144;
    constexpr int WARPS_PER_BLOCK_MAX = 8;  // For block_size=256

    __shared__ uint32_t s_warp_words[WARPS_PER_BLOCK_MAX][SMEM_WORDS_PER_WARP];

    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Optimized metadata loading using __ldg for read-only data
    const PartitionMetadataV5* meta_ptr = &d_metadata[pid];

    // Load metadata fields using texture cache path
    int partition_start = __ldg(&meta_ptr->start_idx);
    int partition_end = __ldg(&meta_ptr->end_idx);
    int partition_size = partition_end - partition_start;
    int delta_bits = __ldg(&meta_ptr->delta_bits);
    int num_mv = __ldg(&meta_ptr->num_mini_vectors);
    int64_t interleaved_base = __ldg(&meta_ptr->interleaved_offset);
    int model_type = __ldg(&meta_ptr->model_type);

    // Load params - use direct access as they're in the same cache line
    double params[4];
    params[0] = meta_ptr->params[0];
    if (model_type != MODEL_FOR_BITPACK) {
        params[1] = meta_ptr->params[1];
        params[2] = meta_ptr->params[2];
        params[3] = meta_ptr->params[3];
    }

    // Pre-compute values
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base_value = static_cast<T>(__double2ll_rn(params[0]));
        }
    }
    bool needs_third_word = (delta_bits > 32);

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(params[0]);
        int value_bits = static_cast<int>(params[2]);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        if (num_runs == 1 && count_bits == 0) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // RLE decoding
        int bits_per_run = value_bits + count_bits;
        int64_t rle_bit_base = interleaved_base << 5;
        int64_t rle_bits_total = static_cast<int64_t>(num_runs) * bits_per_run;
        int rle_words_needed = (rle_bits_total + 63) / 32;
        int64_t rle_first_word = rle_bit_base >> 5;

        // Use first warp's smem for RLE
        uint32_t* rle_smem = s_warp_words[0];
        int max_rle_words = min(rle_words_needed, SMEM_WORDS_PER_WARP * WARPS_PER_BLOCK_MAX);
        for (int w = threadIdx.x; w < max_rle_words; w += blockDim.x) {
            rle_smem[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        __shared__ T s_run_values[256];
        __shared__ int s_run_counts[256];
        __shared__ int s_run_offsets[257];

        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;
            for (int r = 0; r < num_runs; r++) {
                int64_t bit_offset = static_cast<int64_t>(r) * bits_per_run;
                int word_idx = bit_offset >> 5;
                int bit_in_word = bit_offset & 31;

                uint64_t combined = (static_cast<uint64_t>(rle_smem[word_idx + 1]) << 32) | rle_smem[word_idx];
                uint64_t packed = combined >> bit_in_word;
                if (bit_in_word > 0 && (64 - bit_in_word) < bits_per_run && (word_idx + 2) < max_rle_words) {
                    packed |= (static_cast<uint64_t>(rle_smem[word_idx + 2]) << (64 - bit_in_word));
                }

                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));
                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);
                s_run_counts[r] = count;
                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int left = 0, right = num_runs - 1;
            while (left < right) {
                int mid = (left + right + 1) >> 1;
                if (s_run_offsets[mid] <= local_idx) left = mid;
                else right = mid - 1;
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

    // Skip if no interleaved data
    if (interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result = (model_type == MODEL_FOR_BITPACK) ? base_value :
                       computePredictionPoly<T>(model_type, params, local_idx);
            output[global_idx] = result;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Get this warp's static shared memory
    uint32_t* warp_smem = s_warp_words[warp_id_in_block];

    // Process mini-vectors
    int bits_per_mv = MINI_VECTOR_SIZE * delta_bits;
    int words_needed = (bits_per_mv + 31 + 32) / 32;
    words_needed = min(words_needed, SMEM_WORDS_PER_WARP);

    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t mv_first_word = mv_bit_base >> 5;

        // Conditional prefetch: only for LARGE delta_bits class
        if constexpr (DELTA_BITS_CLASS == 2) {
            int next_mv_idx = mv_idx + warps_per_block;
            if (next_mv_idx < num_mv && lane_id < 4) {
                int64_t next_mv_bit_base = (interleaved_base << 5) +
                                           static_cast<int64_t>(next_mv_idx) * MINI_VECTOR_SIZE * delta_bits;
                int64_t next_mv_first_word = next_mv_bit_base >> 5;
                asm volatile("prefetch.global.L1 [%0];" : : "l"(&interleaved_array[next_mv_first_word + lane_id * 32]));
            }
        }

        // Load current mini-vector
        for (int w = lane_id; w < words_needed; w += WARP_SIZE) {
            warp_smem[w] = __ldg(&interleaved_array[mv_first_word + w]);
        }
        __syncwarp();

        // Process values
        int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int local_bit = lane_bit_start & 31;
        int base_word_idx = lane_bit_start >> 5;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE + lane_id;

            int word_idx = base_word_idx + (local_bit >> 5);
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(warp_smem[word_idx + 1]) << 32) | warp_smem[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                extracted |= (static_cast<uint64_t>(warp_smem[word_idx + 2]) << (64 - bit_in_word));
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = (delta_bits == 0) ? base_value : base_value + static_cast<T>(extracted);
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
        __syncwarp();
    }

    // Tail processing
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    if (tail_size > 0 && delta_bits > 0) {
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
        int64_t tail_first_word = tail_bit_base >> 5;
        int tail_bit_offset_in_first_word = tail_bit_base & 31;

        int64_t tail_bits_total = static_cast<int64_t>(tail_size) * delta_bits;
        int tail_words_needed = (tail_bits_total + tail_bit_offset_in_first_word + 63) / 32;
        tail_words_needed = min(tail_words_needed, SMEM_WORDS_PER_WARP);

        __syncthreads();

        // Use first warp's smem for tail
        uint32_t* tail_smem = s_warp_words[0];
        if (warp_id_in_block == 0) {
            for (int w = lane_id; w < tail_words_needed; w += WARP_SIZE) {
                tail_smem[w] = __ldg(&interleaved_array[tail_first_word + w]);
            }
        }
        __syncthreads();

        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            int tail_local_idx = local_idx - tail_start;

            int64_t bit_offset_from_base = static_cast<int64_t>(tail_local_idx) * delta_bits;
            int64_t local_bit_pos = tail_bit_offset_in_first_word + bit_offset_from_base;
            int word_idx = local_bit_pos >> 5;
            int bit_in_word = local_bit_pos & 31;

            uint64_t combined = (static_cast<uint64_t>(tail_smem[word_idx + 1]) << 32) | tail_smem[word_idx];
            uint64_t extracted = (combined >> bit_in_word);
            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                extracted |= (static_cast<uint64_t>(tail_smem[word_idx + 2]) << (64 - bit_in_word));
            }
            extracted &= Vertical::mask64_rt(delta_bits);

            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value + static_cast<T>(extracted);
            } else {
                T predicted = computePredictionPoly<T>(model_type, params, local_idx);
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDelta(predicted, delta);
            }
            output[global_idx] = result;
        }
    } else if (tail_size > 0) {
        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result = (model_type == MODEL_FOR_BITPACK) ? base_value :
                       computePredictionPoly<T>(model_type, params, local_idx);
            output[global_idx] = result;
        }
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

    if (model_type == MODEL_DIRECT_COPY) {
        // MODEL_DIRECT_COPY: delta_bits store the full value bit pattern.
        if (delta_bits == 0) {
            output[qid] = T(0);
            return;
        }

        int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
        uint64_t extracted = Vertical::extract_branchless_64_rt(
            sequential_deltas, bit_offset, delta_bits);

        if constexpr (std::is_signed<T>::value) {
            output[qid] = static_cast<T>(Vertical::sign_extend_64(extracted, delta_bits));
        } else {
            output[qid] = static_cast<T>(extracted);
        }
        return;
    }

    if (model_type == MODEL_FOR_BITPACK) {
        // FOR+BitPacking: val = base + unsigned_delta (base stored in params[0])
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(params[0]));
        }

        if (delta_bits == 0) {
            output[qid] = base;
        } else {
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
        output[qid] = applyDelta(pred_val, delta);
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
    const int64_t* __restrict__ d_param_offsets,  // nullptr = fixed layout (pid * 4)
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

    double p0, p1, p2, p3;
    loadModelParams(model_type, d_model_params, d_param_offsets, pid, p0, p1, p2, p3);
    double params[4] = {p0, p1, p2, p3};

    int local_idx = idx - start_idx;

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        // params[0]=num_runs (1 => pure constant), params[1]=base_value (bit-pattern preserved for 64-bit).
        const int num_runs = static_cast<int>(p0);

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(p1));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(p1));
        }

        // Single-run CONSTANT: no interleaved payload required.
        if (num_runs <= 1 || interleaved_base < 0) {
            output[qid] = rle_base_value;
            return;
        }

        // RLE v6 bitstream:
        //   header (6 words): 4 floats (value/count intercept/slope) + bits_info + actual_num_runs
        //   residuals start at bit offset 192: values residuals, then counts residuals.
        constexpr int MAX_RLE_RUNS = 512;

        const float* fptr = reinterpret_cast<const float*>(&interleaved_deltas[interleaved_base]);
        const float value_intercept_f = fptr[0];
        const float value_slope_f = fptr[1];
        const float count_intercept_f = fptr[2];
        const float count_slope_f = fptr[3];
        const uint32_t bits_info = __ldg(&interleaved_deltas[interleaved_base + 4]);
        const int value_residual_bits = static_cast<int>(bits_info & 0xFF);
        const int count_residual_bits = static_cast<int>((bits_info >> 8) & 0xFF);
        int actual_num_runs = static_cast<int>(__ldg(&interleaved_deltas[interleaved_base + 5]));
        actual_num_runs = min(actual_num_runs, MAX_RLE_RUNS);

        const uint64_t rle_bit_base = static_cast<uint64_t>(interleaved_base) << 5;
        const uint64_t values_bit_base = rle_bit_base + 192;
        const uint64_t counts_bit_base =
            values_bit_base + static_cast<uint64_t>(actual_num_runs) * static_cast<uint64_t>(value_residual_bits);

        auto extract_signed = [&](uint64_t start_bit, int bits) -> int64_t {
            if (bits <= 0) return 0;
            if (bits <= 32) {
                uint32_t u = Vertical::extract_branchless_32_rt(interleaved_deltas, start_bit, bits);
                return static_cast<int64_t>(Vertical::sign_extend_32(u, bits));
            }
            uint64_t u = Vertical::extract_branchless_64_rt(interleaved_deltas, start_bit, bits);
            return Vertical::sign_extend_64(u, bits);
        };

        int cumulative = 0;
        for (int r = 0; r < actual_num_runs; ++r) {
            const int64_t value_residual =
                extract_signed(values_bit_base + static_cast<uint64_t>(r) * static_cast<uint64_t>(value_residual_bits),
                               value_residual_bits);
            const int64_t count_residual =
                extract_signed(counts_bit_base + static_cast<uint64_t>(r) * static_cast<uint64_t>(count_residual_bits),
                               count_residual_bits);

            double value_pred = static_cast<double>(value_intercept_f) +
                                static_cast<double>(value_slope_f) * static_cast<double>(r);
            int64_t value_delta = static_cast<int64_t>(llrint(value_pred)) + value_residual;

            double count_pred = static_cast<double>(count_intercept_f) +
                                static_cast<double>(count_slope_f) * static_cast<double>(r);
            int count = static_cast<int>(llrint(count_pred)) + static_cast<int>(count_residual);
            if (count <= 0) continue;

            if (local_idx < cumulative + count) {
                output[qid] = applyDelta(rle_base_value, value_delta);
                return;
            }
            cumulative += count;
        }

        // Fallback for malformed data: return base value.
        output[qid] = rle_base_value;
        return;
    }

    // No interleaved payload: prediction-only.
    if (interleaved_base < 0) {
        if (model_type == MODEL_FOR_BITPACK) {
            T base;
            if (sizeof(T) == 8) base = static_cast<T>(__double_as_longlong(p0));
            else base = static_cast<T>(__double2ll_rn(p0));
            output[qid] = base;
        } else {
            output[qid] = computePredictionPoly<T>(model_type, params, local_idx);
        }
        return;
    }

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

        if (delta_bits == 0) {
            output[qid] = base;
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
        output[qid] = applyDelta(pred_val, delta);
    }
}

/**
 * OPTIMIZED Random access decompression using INTERLEAVED format (v3.0)
 *
 * Optimizations over original:
 * 1. Minimize integer math in coordinate mapping (opt path)
 * 2. Keep the random-access hot path branch-light
 */
template<typename T>
__global__ void decompressRandomAccessInterleavedOpt(
    const uint32_t* __restrict__ interleaved_deltas,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,  // nullptr = fixed layout (pid * 4)
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
        output[qid] = 0;
        return;
    }

    // Load metadata
    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int32_t start_idx = d_start_indices[pid];
    int32_t num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];

    double p0, p1, p2, p3;
    loadModelParams(model_type, d_model_params, d_param_offsets, pid, p0, p1, p2, p3);
    double params[4] = {p0, p1, p2, p3};

    int local_idx = idx - start_idx;

    // ========== CONSTANT/RLE (MODEL_CONSTANT) ==========
    if (model_type == MODEL_CONSTANT) {
        const int num_runs = static_cast<int>(p0);

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(p1));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(p1));
        }

        // Single-run CONSTANT (or missing payload): return base value directly.
        if (num_runs <= 1 || interleaved_base < 0) {
            output[qid] = rle_base_value;
            return;
        }

        // RLE v6 bitstream:
        //   header (6 words): 4 floats (value/count intercept/slope) + bits_info + actual_num_runs
        //   residuals start at bit offset 192: values residuals, then counts residuals.
        constexpr int MAX_RLE_RUNS = 512;

        const float* fptr = reinterpret_cast<const float*>(&interleaved_deltas[interleaved_base]);
        const float value_intercept_f = fptr[0];
        const float value_slope_f = fptr[1];
        const float count_intercept_f = fptr[2];
        const float count_slope_f = fptr[3];
        const uint32_t bits_info = __ldg(&interleaved_deltas[interleaved_base + 4]);
        const int value_residual_bits = static_cast<int>(bits_info & 0xFF);
        const int count_residual_bits = static_cast<int>((bits_info >> 8) & 0xFF);
        int actual_num_runs = static_cast<int>(__ldg(&interleaved_deltas[interleaved_base + 5]));
        actual_num_runs = min(actual_num_runs, MAX_RLE_RUNS);

        const uint64_t rle_bit_base = static_cast<uint64_t>(interleaved_base) << 5;
        const uint64_t values_bit_base = rle_bit_base + 192;
        const uint64_t counts_bit_base =
            values_bit_base + static_cast<uint64_t>(actual_num_runs) * static_cast<uint64_t>(value_residual_bits);

        auto extract_signed = [&](uint64_t start_bit, int bits) -> int64_t {
            if (bits <= 0) return 0;
            if (bits <= 32) {
                uint32_t u = Vertical::extract_branchless_32_rt(interleaved_deltas, start_bit, bits);
                return static_cast<int64_t>(Vertical::sign_extend_32(u, bits));
            }
            uint64_t u = Vertical::extract_branchless_64_rt(interleaved_deltas, start_bit, bits);
            return Vertical::sign_extend_64(u, bits);
        };

        int cumulative = 0;
        for (int r = 0; r < actual_num_runs; ++r) {
            const int64_t value_residual =
                extract_signed(values_bit_base + static_cast<uint64_t>(r) * static_cast<uint64_t>(value_residual_bits),
                               value_residual_bits);
            const int64_t count_residual =
                extract_signed(counts_bit_base + static_cast<uint64_t>(r) * static_cast<uint64_t>(count_residual_bits),
                               count_residual_bits);

            double value_pred = static_cast<double>(value_intercept_f) +
                                static_cast<double>(value_slope_f) * static_cast<double>(r);
            int64_t value_delta = static_cast<int64_t>(llrint(value_pred)) + value_residual;

            double count_pred = static_cast<double>(count_intercept_f) +
                                static_cast<double>(count_slope_f) * static_cast<double>(r);
            int count = static_cast<int>(llrint(count_pred)) + static_cast<int>(count_residual);
            if (count <= 0) continue;

            if (local_idx < cumulative + count) {
                output[qid] = applyDelta(rle_base_value, value_delta);
                return;
            }
            cumulative += count;
        }

        output[qid] = rle_base_value;
        return;
    }

    // Prediction-only path (rare): no interleaved payload.
    if (interleaved_base < 0) {
        if (model_type == MODEL_FOR_BITPACK) {
            T base;
            if constexpr (sizeof(T) == 8) base = static_cast<T>(__double_as_longlong(p0));
            else base = static_cast<T>(__double2ll_rn(p0));
            output[qid] = base;
        } else {
            output[qid] = computePredictionPoly<T>(model_type, params, local_idx);
        }
        return;
    }

    // Compute bit offset (mini-vectors are lane-major; tail is sequential).
    const int mv_boundary = num_mv * MINI_VECTOR_SIZE;
    int64_t bit_offset;
    if (local_idx < mv_boundary) {
        const int mini_vector_idx = local_idx / MINI_VECTOR_SIZE;
        const int local_in_mv = local_idx % MINI_VECTOR_SIZE;
        const int lane_id = local_in_mv % LANES_PER_MINI_VECTOR;
        const int value_idx = local_in_mv / LANES_PER_MINI_VECTOR;

        const int64_t mv_bit_base =
            (interleaved_base << 5) +
            static_cast<int64_t>(mini_vector_idx) * MINI_VECTOR_SIZE * delta_bits;
        const int64_t lane_bit_offset = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        const int64_t value_bit_offset = static_cast<int64_t>(value_idx) * delta_bits;
        bit_offset = mv_bit_base + lane_bit_offset + value_bit_offset;
    } else {
        const int tail_idx = local_idx - mv_boundary;
        const int64_t mv_total_bits = static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;
        bit_offset = (interleaved_base << 5) + mv_total_bits + static_cast<int64_t>(tail_idx) * delta_bits;
    }

    // Extract and decompress
    if (model_type == MODEL_FOR_BITPACK) {
        T base;
        if constexpr (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(params[0]));
        }

        if (delta_bits == 0) {
            output[qid] = base;
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
        output[qid] = applyDelta(pred_val, delta);
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

    // ========== RLE (CONSTANT) partition - basic support for legacy kernel ==========
    // Note: Full RLE support is in interleaved format. This handles simple cases.
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(params[0]);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(params[1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(params[1]));
        }

        // For single run or traditional constant: fill with base value
        if (num_runs == 1 || count_bits == 0) {
            for (int local_idx = lane_id; local_idx < partition_size; local_idx += WARP_SIZE) {
                output[start_idx + local_idx] = rle_base_value;
            }
            return;
        }

        // For RLE with multiple runs in sequential format: scan and expand
        int value_bits = static_cast<int>(params[2]);
        int bits_per_run = value_bits + count_bits;

        // Simple sequential scan (single-threaded for correctness in warp)
        if (lane_id == 0) {
            int out_idx = 0;
            for (int r = 0; r < num_runs && out_idx < partition_size; r++) {
                int64_t run_bit_offset = bit_base + static_cast<int64_t>(r) * bits_per_run;
                uint64_t packed = Vertical::extract_branchless_64_rt(delta_array, run_bit_offset, bits_per_run);

                uint64_t value_delta = packed & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((packed >> value_bits) & ((1ULL << count_bits) - 1));

                T value = rle_base_value + static_cast<T>(value_delta);
                for (int c = 0; c < count && out_idx < partition_size; c++, out_idx++) {
                    output[start_idx + out_idx] = value;
                }
            }
        }
        return;
    }

    // ========== DIRECT_COPY special handling (legacy sequential format) ==========
    if (model_type == MODEL_DIRECT_COPY) {
        for (int local_idx = lane_id; local_idx < partition_size; local_idx += WARP_SIZE) {
            int64_t bit_offset = bit_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = Vertical::extract_branchless_64_rt(delta_array, bit_offset, delta_bits);

            if constexpr (std::is_signed<T>::value) {
                output[start_idx + local_idx] = static_cast<T>(Vertical::sign_extend_64(extracted, delta_bits));
            } else {
                output[start_idx + local_idx] = static_cast<T>(extracted);
            }
        }
        return;
    }

    // Each thread processes multiple values within the partition
    for (int local_idx = lane_id; local_idx < partition_size; local_idx += WARP_SIZE) {
        int global_idx = start_idx + local_idx;

        T result;
        if (model_type == MODEL_FOR_BITPACK) {
            // FOR+BitPacking: val = base + unsigned_delta (base stored in params[0])
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

            result = applyDelta(pred_val, delta);
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
// V6 Decoder (SoA metadata + variable params + RLE support)
// ============================================================================

/**
 * V6: One block per partition, lane-major interleaved decoding.
 *
 * Key properties:
 * - Reads metadata in SoA form (coalesced across blocks)
 * - Supports variable-length parameter storage via d_param_offsets
 * - Supports unified CONSTANT/RLE format (MODEL_CONSTANT)
 * - Uses BitReader32 for delta_bits <= 32 (fast sequential bitstream per lane)
 * - Uses shared FiniteDiff FP64 accumulation to match encoder bit-exactly
 */
template<typename T>
__global__ void decompressInterleavedAllPartitionsV6(
    const uint32_t* __restrict__ interleaved_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,  // nullptr = fixed layout (pid * 4)
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    const int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    const int partition_start = d_start_indices[pid];
    const int partition_end = d_end_indices[pid];
    const int partition_size = partition_end - partition_start;
    const int delta_bits = d_delta_bits[pid];
    const int num_mv = d_num_mini_vectors[pid];
    const int64_t interleaved_base = d_interleaved_offsets[pid];
    const int model_type = d_model_types[pid];

    double p0, p1, p2, p3;
    loadModelParams(model_type, d_model_params, d_param_offsets, pid, p0, p1, p2, p3);

    // Precompute base value for FOR_BITPACK.
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(p0));
        } else {
            base_value = static_cast<T>(__double2ll_rn(p0));
        }
    }

    // ---------------------------------------------------------------------
    // Unified CONSTANT/RLE handling
    // params[0]=num_runs, params[1]=base_value, delta_bits=count_bits (0 => single-run)
    // ---------------------------------------------------------------------
    if (model_type == MODEL_CONSTANT) {
        const int num_runs = static_cast<int>(p0);

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(p1));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(p1));
        }

        // Single-run CONSTANT: all values are base_value.
        if (num_runs <= 1) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // RLE: decode header + residuals, then expand runs.
        constexpr int MAX_RLE_RUNS = 512;
        constexpr int MAX_RLE_WORDS = 256;  // 256 * 32 = 8192 bits shared staging

        __shared__ uint32_t s_rle_data[MAX_RLE_WORDS];
        __shared__ T s_run_values[MAX_RLE_RUNS];
        __shared__ int s_run_counts[MAX_RLE_RUNS];
        __shared__ int s_run_offsets[MAX_RLE_RUNS + 1];

        __shared__ float s_value_intercept;
        __shared__ float s_value_slope;
        __shared__ float s_count_intercept;
        __shared__ float s_count_slope;
        __shared__ int s_value_residual_bits;
        __shared__ int s_count_residual_bits;
        __shared__ int s_actual_num_runs;
        __shared__ int s_rle_words_loaded;
        __shared__ bool s_use_smem;

        const uint64_t rle_bit_base = static_cast<uint64_t>(interleaved_base) << 5;
        const int64_t rle_first_word = interleaved_base;

        // Read header (6 words). Keep the exact float->double behavior consistent with encoder.
        if (threadIdx.x == 0) {
            const float* fptr = reinterpret_cast<const float*>(&interleaved_array[rle_first_word]);
            s_value_intercept = fptr[0];
            s_value_slope = fptr[1];
            s_count_intercept = fptr[2];
            s_count_slope = fptr[3];

            const uint32_t bits_info = interleaved_array[rle_first_word + 4];
            s_value_residual_bits = static_cast<int>(bits_info & 0xFF);
            s_count_residual_bits = static_cast<int>((bits_info >> 8) & 0xFF);
            s_actual_num_runs = static_cast<int>(interleaved_array[rle_first_word + 5]);

            // Conservative staging size (includes 2-word padding for 3-word extraction safety).
            const int capped_runs = (s_actual_num_runs > MAX_RLE_RUNS) ? MAX_RLE_RUNS : s_actual_num_runs;
            const int64_t total_bits =
                192 + static_cast<int64_t>(capped_runs) *
                          (static_cast<int64_t>(s_value_residual_bits) + static_cast<int64_t>(s_count_residual_bits));
            int words_needed = static_cast<int>((total_bits + 31) / 32) + 2;

            s_use_smem = (words_needed <= MAX_RLE_WORDS);
            s_rle_words_loaded = s_use_smem ? words_needed : 0;
        }
        __syncthreads();

        const int actual_num_runs = min(s_actual_num_runs, MAX_RLE_RUNS);

        if (s_use_smem) {
            const int words_to_load = s_rle_words_loaded;
            for (int w = threadIdx.x; w < words_to_load; w += blockDim.x) {
                s_rle_data[w] = __ldg(&interleaved_array[rle_first_word + w]);
            }
            __syncthreads();
        }

        // Shared-memory extraction with optional global fallback (handles 3-word crossing).
        auto extract_signed_rle = [&](int64_t bit_offset_rel, int bits) -> int64_t {
            if (bits <= 0) return 0;
            if (bits > 64) bits = 64;

            if (s_use_smem) {
                const int word_idx = static_cast<int>(bit_offset_rel >> 5);
                const int bit_in_word = static_cast<int>(bit_offset_rel & 31);

                // Need up to word_idx+2 for the 3-word stitch.
                if (word_idx + 2 < s_rle_words_loaded) {
                    uint64_t combined =
                        (static_cast<uint64_t>(s_rle_data[word_idx + 1]) << 32) | s_rle_data[word_idx];
                    uint64_t extracted = (combined >> bit_in_word);
                    if (bit_in_word > 0 && (64 - bit_in_word) < bits) {
                        extracted |= (static_cast<uint64_t>(s_rle_data[word_idx + 2]) << (64 - bit_in_word));
                    }
                    extracted &= Vertical::mask64_rt(bits);
                    return Vertical::sign_extend_64(extracted, bits);
                }
            }

            // Global fallback (rare): absolute bit offset into interleaved_array.
            const uint64_t abs_bit = rle_bit_base + static_cast<uint64_t>(bit_offset_rel);
            uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, abs_bit, bits);
            return Vertical::sign_extend_64(extracted, bits);
        };

        // Parallel run decode: each thread decodes one or more runs.
        const int64_t values_bit_base = 192;
        const int64_t counts_bit_base = 192 + static_cast<int64_t>(actual_num_runs) * s_value_residual_bits;
        for (int r = threadIdx.x; r < actual_num_runs; r += blockDim.x) {
            const int64_t value_residual = extract_signed_rle(values_bit_base + static_cast<int64_t>(r) * s_value_residual_bits,
                                                              s_value_residual_bits);
            const int64_t count_residual = extract_signed_rle(counts_bit_base + static_cast<int64_t>(r) * s_count_residual_bits,
                                                              s_count_residual_bits);

            // Reconstruct using LINEAR model (float params stored; compute in double to match encoder).
            double value_pred = static_cast<double>(s_value_intercept) + static_cast<double>(s_value_slope) * static_cast<double>(r);
            int64_t value_delta = static_cast<int64_t>(llrint(value_pred)) + value_residual;
            s_run_values[r] = applyDelta(rle_base_value, value_delta);

            double count_pred = static_cast<double>(s_count_intercept) + static_cast<double>(s_count_slope) * static_cast<double>(r);
            int count = static_cast<int>(llrint(count_pred)) + static_cast<int>(count_residual);
            s_run_counts[r] = count;
        }
        __syncthreads();

        // Prefix sum of run counts -> offsets (warp 0, chunked by 32).
        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            int running_offset = 0;

            for (int base = 0; base < actual_num_runs; base += 32) {
                int r = base + lane;
                int my_count = (r < actual_num_runs) ? s_run_counts[r] : 0;

                #pragma unroll
                for (int offset = 1; offset < 32; offset <<= 1) {
                    int n = __shfl_up_sync(0xFFFFFFFF, my_count, offset);
                    if (lane >= offset) my_count += n;
                }

                if (r < actual_num_runs) {
                    s_run_offsets[r] = running_offset + my_count - s_run_counts[r];
                }
                running_offset += __shfl_sync(0xFFFFFFFF, my_count, 31);
            }

            if (lane == 0) {
                s_run_offsets[actual_num_runs] = running_offset;
            }
        }
        __syncthreads();

        // Expand runs: each warp processes one run at a time (coalesced writes).
        const int warp_id = threadIdx.x / 32;
        const int lane_id = threadIdx.x & 31;
        const int num_warps = blockDim.x / 32;

        for (int r = warp_id; r < actual_num_runs; r += max(num_warps, 1)) {
            int run_start = s_run_offsets[r];
            int run_end = s_run_offsets[r + 1];
            if (run_start >= partition_size) continue;
            run_end = min(run_end, partition_size);

            const T run_value = s_run_values[r];
            for (int i = run_start + lane_id; i < run_end; i += 32) {
                output[partition_start + i] = run_value;
            }
        }
        return;
    }

    // ---------------------------------------------------------------------
    // No interleaved data: prediction-only path (rare)
    // ---------------------------------------------------------------------
    if (interleaved_base < 0) {
        if (model_type == MODEL_FOR_BITPACK) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = base_value;
            }
        } else {
            double params_arr[4] = {p0, p1, p2, p3};
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = computePredictionPoly<T>(model_type, params_arr, local_idx);
            }
        }
        return;
    }

    // ---------------------------------------------------------------------
    // Warp/lane mapping: one lane-group (LANES_PER_MINI_VECTOR threads) per mini-vector.
    // ---------------------------------------------------------------------
    const int lane_id = threadIdx.x % LANES_PER_MINI_VECTOR;
    const int warp_id_in_block = threadIdx.x / LANES_PER_MINI_VECTOR;
    const int warps_per_block = blockDim.x / LANES_PER_MINI_VECTOR;

    double params_arr[4] = {p0, p1, p2, p3};

    // ---------------------------------------------------------------------
    // Mini-vectors (lane-group-per-mini-vector)
    // ---------------------------------------------------------------------
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += max(warps_per_block, 1)) {
        const int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;
        int out = mv_start_global + lane_id;

        if (delta_bits == 0) {
            if (model_type == MODEL_FOR_BITPACK) {
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = base_value;
                    out += LANES_PER_MINI_VECTOR;
                }
            } else if (model_type == MODEL_LINEAR) {
                double y_fp, step_fp;
                FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                      LANES_PER_MINI_VECTOR, y_fp, step_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = FiniteDiff::fp64_to_int<T>(y_fp);
                    out += LANES_PER_MINI_VECTOR;
                    y_fp = FiniteDiff::d_add(y_fp, step_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL2) {
                double y_fp, d1_fp, d2_fp;
                FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                     LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = FiniteDiff::fp64_to_int<T>(y_fp);
                    out += LANES_PER_MINI_VECTOR;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL3) {
                double y_fp, d1_fp, d2_fp, d3_fp;
                FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                     LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp, d3_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = FiniteDiff::fp64_to_int<T>(y_fp);
                    out += LANES_PER_MINI_VECTOR;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                    d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                }
            } else {
                // Fallback: treat as constant prediction params[0].
                const T pred_const = FiniteDiff::fp64_to_int<T>(p0);
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = pred_const;
                    out += LANES_PER_MINI_VECTOR;
                }
            }
        } else {
            const uint64_t mv_bit_base =
                (static_cast<uint64_t>(interleaved_base) << 5) +
                static_cast<uint64_t>(mv_idx) * static_cast<uint64_t>(MINI_VECTOR_SIZE) * static_cast<uint64_t>(delta_bits);
            const uint64_t lane_bit_start =
                mv_bit_base +
                static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(VALUES_PER_THREAD) * static_cast<uint64_t>(delta_bits);

            if (delta_bits <= 32) {
                const uint64_t MASK = (delta_bits == 32) ? 0xFFFFFFFFULL : ((1ULL << delta_bits) - 1ULL);

                if (model_type == MODEL_FOR_BITPACK) {
                    BitReader32 br(interleaved_array, lane_bit_start);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t d0, d1, d2, d3;
                        br.read4(delta_bits, MASK, d0, d1, d2, d3);
                        output[out + 0 * LANES_PER_MINI_VECTOR] = base_value + static_cast<T>(d0);
                        output[out + 1 * LANES_PER_MINI_VECTOR] = base_value + static_cast<T>(d1);
                        output[out + 2 * LANES_PER_MINI_VECTOR] = base_value + static_cast<T>(d2);
                        output[out + 3 * LANES_PER_MINI_VECTOR] = base_value + static_cast<T>(d3);
                        out += 4 * LANES_PER_MINI_VECTOR;
                    }
                } else if (model_type == MODEL_LINEAR) {
                    BitReader32 br(interleaved_array, lane_bit_start);
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                          LANES_PER_MINI_VECTOR, y_fp, step_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t d0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t d1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t d2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t d3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), d0);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 1 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), d1);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 2 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), d2);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 3 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), d3);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        out += 4 * LANES_PER_MINI_VECTOR;
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    BitReader32 br(interleaved_array, lane_bit_start);
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                         LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t dd0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t dd1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t dd2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t dd3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 1 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 2 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 3 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        out += 4 * LANES_PER_MINI_VECTOR;
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    BitReader32 br(interleaved_array, lane_bit_start);
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                         LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp, d3_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t dd0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t dd1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t dd2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t dd3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 1 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 2 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 3 * LANES_PER_MINI_VECTOR] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), dd3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        out += 4 * LANES_PER_MINI_VECTOR;
                    }
                } else {
                    BitReader32 br(interleaved_array, lane_bit_start);
                    const T pred_const = FiniteDiff::fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t dd0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t dd1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t dd2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t dd3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * LANES_PER_MINI_VECTOR] = applyDelta(pred_const, dd0);
                        output[out + 1 * LANES_PER_MINI_VECTOR] = applyDelta(pred_const, dd1);
                        output[out + 2 * LANES_PER_MINI_VECTOR] = applyDelta(pred_const, dd2);
                        output[out + 3 * LANES_PER_MINI_VECTOR] = applyDelta(pred_const, dd3);
                        out += 4 * LANES_PER_MINI_VECTOR;
                    }
                }
            } else {
                // delta_bits > 32: use 64-bit extraction
                if (model_type == MODEL_LINEAR) {
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                          LANES_PER_MINI_VECTOR, y_fp, step_fp);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), delta);
                        out += LANES_PER_MINI_VECTOR;
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                         LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), delta);
                        out += LANES_PER_MINI_VECTOR;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id,
                                                         LANES_PER_MINI_VECTOR, y_fp, d1_fp, d2_fp, d3_fp);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDelta(FiniteDiff::fp64_to_int<T>(y_fp), delta);
                        out += LANES_PER_MINI_VECTOR;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                    }
                } else if (model_type == MODEL_FOR_BITPACK) {
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t d = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        output[out] = base_value + static_cast<T>(d);
                        out += LANES_PER_MINI_VECTOR;
                    }
                } else {
                    const T pred_const = FiniteDiff::fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDelta(pred_const, delta);
                        out += LANES_PER_MINI_VECTOR;
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Tail values (partition_size % MINI_VECTOR_SIZE) - sequential packed after mini-vectors
    // ---------------------------------------------------------------------
    const int tail_start = num_mv * MINI_VECTOR_SIZE;
    const uint64_t tail_bit_base =
        (static_cast<uint64_t>(interleaved_base) << 5) +
        static_cast<uint64_t>(num_mv) * static_cast<uint64_t>(MINI_VECTOR_SIZE) * static_cast<uint64_t>(delta_bits);

    if (tail_start < partition_size) {
        if (model_type == MODEL_FOR_BITPACK) {
            if (delta_bits == 0) {
                for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                    output[partition_start + local_idx] = base_value;
                }
            } else if (delta_bits <= 32) {
                for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                    uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(local_idx - tail_start) * delta_bits;
                    uint32_t d = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    output[partition_start + local_idx] = base_value + static_cast<T>(d);
                }
            } else {
                for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                    uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(local_idx - tail_start) * delta_bits;
                    uint64_t d = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    output[partition_start + local_idx] = base_value + static_cast<T>(d);
                }
            }
        } else if (delta_bits == 0) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = computePredictionPoly<T>(model_type, params_arr, local_idx);
            }
        } else {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                const T pv = computePredictionPoly<T>(model_type, params_arr, local_idx);
                uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(local_idx - tail_start) * delta_bits;

                int64_t delta;
                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    delta = Vertical::sign_extend_64(extracted, delta_bits);
                }

                output[partition_start + local_idx] = applyDelta(pv, delta);
            }
        }
    }
}

// ============================================================================
// Host API Functions
// ============================================================================

/**
 * Decompress all data using the best available path
 * Now with adaptive block size selection for optimal performance
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

    // Select optimal block size based on average partition size
    int block_size = selectOptimalBlockSize(static_cast<int64_t>(total_values), np);

    // Branchless mode: for legacy-fromBase conversions (no interleaved payload).
    if (mode == DecompressMode::BRANCHLESS &&
        compressed.d_sequential_deltas != nullptr &&
        compressed.d_delta_array_bit_offsets != nullptr)
    {
        launchDecompressPerPartitionBranchless(compressed, d_output, stream);
        return;
    }

    // V6 scan decoder is the correctness baseline (SoA metadata + variable params + RLE header format).
    if (compressed.d_interleaved_deltas != nullptr && compressed.total_interleaved_partitions > 0) {
        switch (block_size) {
            case 64:
                decompressInterleavedAllPartitionsV6<T><<<np, 64, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_types,
                    compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, d_output);
                break;
            case 128:
                decompressInterleavedAllPartitionsV6<T><<<np, 128, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_types,
                    compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, d_output);
                break;
            case 256:
            default:
                decompressInterleavedAllPartitionsV6<T><<<np, 256, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_types,
                    compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, d_output);
                break;
        }
        return;
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
        compressed.d_param_offsets,
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
        compressed.d_param_offsets,
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

// ============================================================================
// V13: FOR-Only Optimized Decoder for 32-bit Data
// ============================================================================
// Optimizations over V12:
// 1. FOR-only fast path (no polynomial branches)
// 2. 32-bit extraction (no 64-bit operations for delta_bits <= 32)
// 3. Compile-time masks via template specialization
// 4. No sign extension (FOR uses unsigned deltas)
// 5. Common delta_bits fully unrolled (4, 6, 16, 20)

namespace V13_FOR {

// Compile-time mask for 32-bit values
template<int BITS>
struct Mask32 {
    static constexpr uint32_t value = (BITS >= 32) ? 0xFFFFFFFFu : ((1u << BITS) - 1);
};

// 32-bit extraction - single word case
template<int BITS>
__device__ __forceinline__
uint32_t extract32_single(uint32_t word, int bit_offset) {
    return (word >> bit_offset) & Mask32<BITS>::value;
}

// 32-bit extraction - cross word boundary
template<int BITS>
__device__ __forceinline__
uint32_t extract32_cross(uint32_t w0, uint32_t w1, int bit_offset) {
    uint32_t lo = w0 >> bit_offset;
    uint32_t hi = w1 << (32 - bit_offset);
    return (lo | hi) & Mask32<BITS>::value;
}

// Fully specialized unpack for 8 values with FOR mode
template<int BITS>
__device__ __forceinline__ void unpack8_FOR(
    const uint32_t* __restrict__ data,
    int64_t lane_bit_start,
    uint32_t base,
    uint32_t (&out)[8])
{
    constexpr int BITS_PER_LANE = 8 * BITS;
    constexpr int WORDS_NEEDED = (BITS_PER_LANE + 31) / 32;

    int64_t word_start = lane_bit_start >> 5;
    int local_bit = static_cast<int>(lane_bit_start & 31);

    // Load minimal words needed (+1 for potential boundary crossing)
    uint32_t w[WORDS_NEEDED + 1];
    #pragma unroll
    for (int i = 0; i <= WORDS_NEEDED; i++) {
        w[i] = __ldg(&data[word_start + i]);
    }

    // Extract 8 values with compile-time mask
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint32_t delta;
        if (bit_in_word + BITS <= 32) {
            // Single word extraction
            delta = extract32_single<BITS>(w[word_idx], bit_in_word);
        } else {
            // Cross word boundary
            delta = extract32_cross<BITS>(w[word_idx], w[word_idx + 1], bit_in_word);
        }

        out[v] = base + delta;
        local_bit += BITS;
    }
}

// Specialization for 0-bit (all values equal base)
template<>
__device__ __forceinline__ void unpack8_FOR<0>(
    const uint32_t* __restrict__ data,
    int64_t lane_bit_start,
    uint32_t base,
    uint32_t (&out)[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) out[v] = base;
}

// Runtime dispatcher with common cases at the top
__device__ __forceinline__ void unpack8_FOR_dispatch(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane,
    int delta_bits,
    uint32_t base,
    uint32_t (&out)[8])
{
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane) * 8 * delta_bits;

    // Common cases first for better branch prediction
    switch (delta_bits) {
        case 0:  unpack8_FOR<0>(data, lane_bit_start, base, out); break;
        case 4:  unpack8_FOR<4>(data, lane_bit_start, base, out); break;
        case 6:  unpack8_FOR<6>(data, lane_bit_start, base, out); break;
        case 16: unpack8_FOR<16>(data, lane_bit_start, base, out); break;
        case 20: unpack8_FOR<20>(data, lane_bit_start, base, out); break;
        // Other values
        case 1:  unpack8_FOR<1>(data, lane_bit_start, base, out); break;
        case 2:  unpack8_FOR<2>(data, lane_bit_start, base, out); break;
        case 3:  unpack8_FOR<3>(data, lane_bit_start, base, out); break;
        case 5:  unpack8_FOR<5>(data, lane_bit_start, base, out); break;
        case 7:  unpack8_FOR<7>(data, lane_bit_start, base, out); break;
        case 8:  unpack8_FOR<8>(data, lane_bit_start, base, out); break;
        case 9:  unpack8_FOR<9>(data, lane_bit_start, base, out); break;
        case 10: unpack8_FOR<10>(data, lane_bit_start, base, out); break;
        case 11: unpack8_FOR<11>(data, lane_bit_start, base, out); break;
        case 12: unpack8_FOR<12>(data, lane_bit_start, base, out); break;
        case 13: unpack8_FOR<13>(data, lane_bit_start, base, out); break;
        case 14: unpack8_FOR<14>(data, lane_bit_start, base, out); break;
        case 15: unpack8_FOR<15>(data, lane_bit_start, base, out); break;
        case 17: unpack8_FOR<17>(data, lane_bit_start, base, out); break;
        case 18: unpack8_FOR<18>(data, lane_bit_start, base, out); break;
        case 19: unpack8_FOR<19>(data, lane_bit_start, base, out); break;
        case 21: unpack8_FOR<21>(data, lane_bit_start, base, out); break;
        case 22: unpack8_FOR<22>(data, lane_bit_start, base, out); break;
        case 23: unpack8_FOR<23>(data, lane_bit_start, base, out); break;
        case 24: unpack8_FOR<24>(data, lane_bit_start, base, out); break;
        case 25: unpack8_FOR<25>(data, lane_bit_start, base, out); break;
        case 26: unpack8_FOR<26>(data, lane_bit_start, base, out); break;
        case 27: unpack8_FOR<27>(data, lane_bit_start, base, out); break;
        case 28: unpack8_FOR<28>(data, lane_bit_start, base, out); break;
        case 29: unpack8_FOR<29>(data, lane_bit_start, base, out); break;
        case 30: unpack8_FOR<30>(data, lane_bit_start, base, out); break;
        case 31: unpack8_FOR<31>(data, lane_bit_start, base, out); break;
        case 32: unpack8_FOR<32>(data, lane_bit_start, base, out); break;
        default: unpack8_FOR<16>(data, lane_bit_start, base, out); break;
    }
}

// Multi-warp support version (N values per thread)
template<int BITS, int N>
__device__ __forceinline__ void unpackN_FOR(
    const uint32_t* __restrict__ data,
    int64_t lane_bit_start,
    int start_v,
    uint32_t base,
    uint32_t (&out)[N])
{
    constexpr int BITS_PER_LANE = N * BITS;
    constexpr int WORDS_NEEDED = (BITS_PER_LANE + 31) / 32;

    int64_t adjusted_bit_start = lane_bit_start + static_cast<int64_t>(start_v) * BITS;
    int64_t word_start = adjusted_bit_start >> 5;
    int local_bit = static_cast<int>(adjusted_bit_start & 31);

    uint32_t w[WORDS_NEEDED + 1];
    #pragma unroll
    for (int i = 0; i <= WORDS_NEEDED; i++) {
        w[i] = __ldg(&data[word_start + i]);
    }

    #pragma unroll
    for (int v = 0; v < N; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint32_t delta;
        if (bit_in_word + BITS <= 32) {
            delta = extract32_single<BITS>(w[word_idx], bit_in_word);
        } else {
            delta = extract32_cross<BITS>(w[word_idx], w[word_idx + 1], bit_in_word);
        }

        out[v] = base + delta;
        local_bit += BITS;
    }
}

// Specialization for 0-bit with N values
template<int N>
__device__ __forceinline__ void unpackN_FOR_0bit(
    uint32_t base,
    uint32_t (&out)[N])
{
    #pragma unroll
    for (int v = 0; v < N; v++) out[v] = base;
}

// Runtime dispatcher for N values per thread
template<int N>
__device__ __forceinline__ void unpackN_FOR_dispatch(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane,
    int delta_bits,
    uint32_t base,
    int start_v,
    uint32_t (&out)[N])
{
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane) * 8 * delta_bits;

    // Common cases first for better branch prediction
    switch (delta_bits) {
        case 0:  unpackN_FOR_0bit<N>(base, out); break;
        case 4:  unpackN_FOR<4, N>(data, lane_bit_start, start_v, base, out); break;
        case 6:  unpackN_FOR<6, N>(data, lane_bit_start, start_v, base, out); break;
        case 16: unpackN_FOR<16, N>(data, lane_bit_start, start_v, base, out); break;
        case 20: unpackN_FOR<20, N>(data, lane_bit_start, start_v, base, out); break;
        // Other values
        case 1:  unpackN_FOR<1, N>(data, lane_bit_start, start_v, base, out); break;
        case 2:  unpackN_FOR<2, N>(data, lane_bit_start, start_v, base, out); break;
        case 3:  unpackN_FOR<3, N>(data, lane_bit_start, start_v, base, out); break;
        case 5:  unpackN_FOR<5, N>(data, lane_bit_start, start_v, base, out); break;
        case 7:  unpackN_FOR<7, N>(data, lane_bit_start, start_v, base, out); break;
        case 8:  unpackN_FOR<8, N>(data, lane_bit_start, start_v, base, out); break;
        case 9:  unpackN_FOR<9, N>(data, lane_bit_start, start_v, base, out); break;
        case 10: unpackN_FOR<10, N>(data, lane_bit_start, start_v, base, out); break;
        case 11: unpackN_FOR<11, N>(data, lane_bit_start, start_v, base, out); break;
        case 12: unpackN_FOR<12, N>(data, lane_bit_start, start_v, base, out); break;
        case 13: unpackN_FOR<13, N>(data, lane_bit_start, start_v, base, out); break;
        case 14: unpackN_FOR<14, N>(data, lane_bit_start, start_v, base, out); break;
        case 15: unpackN_FOR<15, N>(data, lane_bit_start, start_v, base, out); break;
        case 17: unpackN_FOR<17, N>(data, lane_bit_start, start_v, base, out); break;
        case 18: unpackN_FOR<18, N>(data, lane_bit_start, start_v, base, out); break;
        case 19: unpackN_FOR<19, N>(data, lane_bit_start, start_v, base, out); break;
        case 21: unpackN_FOR<21, N>(data, lane_bit_start, start_v, base, out); break;
        case 22: unpackN_FOR<22, N>(data, lane_bit_start, start_v, base, out); break;
        case 23: unpackN_FOR<23, N>(data, lane_bit_start, start_v, base, out); break;
        case 24: unpackN_FOR<24, N>(data, lane_bit_start, start_v, base, out); break;
        case 25: unpackN_FOR<25, N>(data, lane_bit_start, start_v, base, out); break;
        case 26: unpackN_FOR<26, N>(data, lane_bit_start, start_v, base, out); break;
        case 27: unpackN_FOR<27, N>(data, lane_bit_start, start_v, base, out); break;
        case 28: unpackN_FOR<28, N>(data, lane_bit_start, start_v, base, out); break;
        case 29: unpackN_FOR<29, N>(data, lane_bit_start, start_v, base, out); break;
        case 30: unpackN_FOR<30, N>(data, lane_bit_start, start_v, base, out); break;
        case 31: unpackN_FOR<31, N>(data, lane_bit_start, start_v, base, out); break;
        case 32: unpackN_FOR<32, N>(data, lane_bit_start, start_v, base, out); break;
        default: unpackN_FOR<16, N>(data, lane_bit_start, start_v, base, out); break;
    }
}

} // namespace V13_FOR

#endif // DECODER_Vertical_OPT_CU
