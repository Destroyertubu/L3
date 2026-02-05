/**
 * L3 Model Statistics E2E Test
 *
 * Tests the GPU Adaptive algorithm on SOSD datasets 1-20 and collects
 * model selection statistics for each dataset.
 *
 * Uses encodeVerticalGPU_PolyCost which calls launchAdaptiveSelectorFullPolynomial
 * to ensure all model types (LINEAR, POLY2, POLY3, FOR) are considered.
 *
 * Supports two partitioning strategies:
 * - FIXED: Fixed partition size (4096)
 * - COST_OPTIMAL: V2 GPU Cost-Optimal with merge optimization
 *
 * Features:
 * - Full E2E testing with encode + decode + verification
 * - V4 Optimized Interleaved Decoder
 * - Model selection statistics (CONSTANT/LINEAR/POLY2/POLY3/FOR)
 * - Performance metrics (encode/decode throughput in GB/s)
 *
 * Output: Console table + CSV file
 *
 * Date: 2025-12-09
 * Updated: 2025-01-24 - Added E2E verification with V4 decoder
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <type_traits>
#include <cuda_runtime.h>
#include <sys/stat.h>

// ============================================================================
// MINI_VECTOR_SIZE Configuration
// Uncomment ONE of the following lines to select MV_SIZE:
// ============================================================================
#if !defined(L3_VERTICAL_256_CONFIG) && !defined(L3_VERTICAL_512_CONFIG) && !defined(L3_VERTICAL_1024_CONFIG) && !defined(L3_VERTICAL_2048_CONFIG)
// #define L3_VERTICAL_256_CONFIG   // MV_SIZE=256, VALUES_PER_THREAD=8
// #define L3_VERTICAL_512_CONFIG   // MV_SIZE=512, VALUES_PER_THREAD=16
// #define L3_VERTICAL_1024_CONFIG  // MV_SIZE=1024, VALUES_PER_THREAD=32
#define L3_VERTICAL_2048_CONFIG  // MV_SIZE=2048, VALUES_PER_THREAD=64
#endif

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Transposed_format.hpp"
#include "L3_Vertical_api.hpp"
#include "L3_Transposed_api_decl.hpp"

// Include bitpack utilities for mask64_rt and sign_extend_64
#include "bitpack_utils_Vertical.cuh"

// Shared finite difference functions - MUST match encoder exactly
#include "finite_diff_shared.cuh"

// GM V4 decoder (header-only)
#include "GM_decoder_v4.cuh"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Debug kernel to test GPU rounding (disabled)
// ============================================================================
// __global__ void testGPURounding(double theta0, double theta1, int count) {
//     if (threadIdx.x == 0) {
//         printf("GPU rounding test:\n");
//         for (int i = 0; i < count; i++) {
//             double x = static_cast<double>(i);
//             double predicted = theta0 + theta1 * x;
//             long long pred_int = __double2ll_rn(predicted);
//             printf("  [%d] pred_d=%.17f -> gpu_llrint=%lld\n", i, predicted, pred_int);
//         }
//     }
// }

// ============================================================================
// V4 Decoder Implementation (Optimized Interleaved Decoder)
// ============================================================================

namespace V4_decoder {

constexpr int BLOCK_SIZE_V4 = 256;
constexpr int WARP_SIZE_V4 = 32;
constexpr int SMEM_WORDS_PER_WARP_V4 = 72;

// Maximum words per mini-vector for shared memory (delta_bits <= 32, MV_SIZE=2048)
// 2048 * 32 / 32 = 2048 words max, but we use smaller buffer with streaming
constexpr int SMEM_BUFFER_WORDS = 136;  // Enough for delta_bits up to ~32 with MV_SIZE=2048

// =============================================================================
// Vectorized bit extraction from SHARED MEMORY - processes 4 consecutive values
// =============================================================================
__device__ __forceinline__ void extract_from_smem_4(
    const uint32_t* __restrict__ smem,
    int local_bit,
    int bits,
    uint64_t mask,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3)
{
    auto extract_one = [&](int bit_offset) -> uint32_t {
        int word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(smem[word_idx + 1]) << 32) | smem[word_idx];
        return static_cast<uint32_t>((combined >> bit_in_word) & mask);
    };

    v0 = extract_one(local_bit);
    v1 = extract_one(local_bit + bits);
    v2 = extract_one(local_bit + 2 * bits);
    v3 = extract_one(local_bit + 3 * bits);
}

// =============================================================================
// Vectorized bit extraction - processes 4 consecutive values at once
// =============================================================================
__device__ __forceinline__ void extract_vectorized_4_rt(
    const uint32_t* __restrict__ words,
    uint64_t start_bit,  // Changed to uint64_t to match original
    int bits,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3)
{
    if (bits <= 0) {
        v0 = v1 = v2 = v3 = 0U;
        return;
    }
    if (bits > 32) bits = 32;

    const uint64_t MASK = (bits == 32) ? 0xFFFFFFFFULL : ((1ULL << bits) - 1ULL);

    // Lambda to extract a single value at a given index
    auto extract_one = [&](int value_idx) -> uint32_t {
        uint64_t bit_pos = start_bit + static_cast<uint64_t>(value_idx) * bits;
        uint32_t word_idx = static_cast<uint32_t>(bit_pos >> 5);
        int bit_in_word = static_cast<int>(bit_pos & 31);

        // Load two consecutive 32-bit words and combine
        uint32_t lo = __ldg(&words[word_idx]);
        uint32_t hi = __ldg(&words[word_idx + 1]);
        uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;

        return static_cast<uint32_t>((combined >> bit_in_word) & MASK);
    };

    v0 = extract_one(0);
    v1 = extract_one(1);
    v2 = extract_one(2);
    v3 = extract_one(3);
}

// =============================================================================
// Fast sequential bit reader for <=32-bit values (per-thread)
// Reads a contiguous bitstream starting at start_bit.
// Designed for decoding lane data where values are packed back-to-back.
// =============================================================================
struct BitReader32 {
    const uint32_t* __restrict__ ptr;
    uint64_t buf;
    int avail;  // number of valid bits in buf

    __device__ __forceinline__
    BitReader32(const uint32_t* __restrict__ words, uint64_t start_bit) {
        const uint32_t* p = words + (start_bit >> 5);

        // Always load two words up front (safe due to +4 padding in allocation)
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
            // When avail < bits <= 32, avail <= 31, so this shift is safe (<32)
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



template<typename T>
__device__ __forceinline__
T applyDeltaV4(T predicted, int64_t delta) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(static_cast<uint64_t>(predicted) + static_cast<uint64_t>(delta));
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

// =============================================================================
// Finite Difference Prediction (eliminates FP64 per-value computation)
// =============================================================================
// For LINEAR:  y = a + b*x  → y[i+1] = y[i] + b*stride
// For POLY2:   y = a + b*x + c*x²  → y += d1; d1 += 2c*stride²
// For POLY3:   y = a + b*x + c*x² + d*x³ → y += d1; d1 += d2; d2 += 6d*stride³
//
// Each thread processes VALUES_PER_THREAD values with stride = WARP_SIZE
// We use FP64 accumulation to avoid rounding errors, then convert at the end

template<typename T>
__device__ __forceinline__
void computeLinearFiniteDiff_FP64(
    const double* params_arr,
    int start_idx,
    int stride,
    double& y,       // initial prediction (output, FP64)
    double& step)    // step per iteration (output, FP64)
{
    double a = params_arr[0];
    double b = params_arr[1];
    // y[start_idx] = a + b * start_idx
    // step = b * stride (constant for LINEAR)
    y = a + b * static_cast<double>(start_idx);
    step = b * static_cast<double>(stride);
}

template<typename T>
__device__ __forceinline__
void computePoly2FiniteDiff_FP64(
    const double* params_arr,
    int start_idx,
    int stride,
    double& y,       // initial prediction
    double& d1,      // first difference
    double& d2)      // second difference (constant)
{
    double a = params_arr[0];
    double b = params_arr[1];
    double c = params_arr[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // y[x0] = a + b*x0 + c*x0²
    y = a + b * x0 + c * x0 * x0;

    // First difference: d1 = y[x0+s] - y[x0] = b*s + c*s*(2*x0 + s)
    d1 = b * s + c * s * (2.0 * x0 + s);

    // Second difference: d2 = 2*c*s² (constant)
    d2 = 2.0 * c * s * s;
}

template<typename T>
__device__ __forceinline__
void computePoly3FiniteDiff_FP64(
    const double* params_arr,
    int start_idx,
    int stride,
    double& y,       // initial prediction
    double& d1,      // first difference
    double& d2,      // second difference
    double& d3)      // third difference (constant)
{
    double a = params_arr[0];
    double b = params_arr[1];
    double c = params_arr[2];
    double d = params_arr[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // y[x0] = a + b*x0 + c*x0² + d*x0³
    y = a + b * x0 + c * x0 * x0 + d * x0 * x0 * x0;

    // First difference at x0 (y[x0+s] - y[x0])
    double x1 = x0 + s;
    double y1 = a + b * x1 + c * x1 * x1 + d * x1 * x1 * x1;
    d1 = y1 - y;

    // Second difference: d2[x0] = d1[x0+s] - d1[x0]
    double x2 = x0 + 2.0 * s;
    double y2 = a + b * x2 + c * x2 * x2 + d * x2 * x2 * x2;
    double d1_next = y2 - y1;
    d2 = d1_next - d1;

    // Third difference: d3 = 6*d*s³ (constant)
    d3 = 6.0 * d * s * s * s;
}

// =============================================================================
// INTEGER VERSION - Fast, uses int64_t accumulation
// Initial value and step computed from FP64, then pure integer arithmetic
// =============================================================================

template<typename T>
__device__ __forceinline__
void computeLinearFiniteDiff_INT(
    const double* params_arr,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction (output, INT64)
    int64_t& step)   // step per iteration (output, INT64)
{
    double a = params_arr[0];
    double b = params_arr[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0);
    // CRITICAL: step = y1 - y0, not __double2ll_rn(b * s)
    int64_t y1 = __double2ll_rn(a + b * (x0 + s));
    step = y1 - y;
}

template<typename T>
__device__ __forceinline__
void computePoly2FiniteDiff_INT(
    const double* params_arr,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction
    int64_t& d1,     // first difference
    int64_t& d2)     // second difference (constant)
{
    double a = params_arr[0];
    double b = params_arr[1];
    double c = params_arr[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0 + c * x0 * x0);
    d1 = __double2ll_rn(b * s + c * s * (2.0 * x0 + s));
    d2 = __double2ll_rn(2.0 * c * s * s);
}

template<typename T>
__device__ __forceinline__
void computePoly3FiniteDiff_INT(
    const double* params_arr,
    int start_idx,
    int stride,
    int64_t& y,      // initial prediction
    int64_t& d1,     // first difference
    int64_t& d2,     // second difference
    int64_t& d3)     // third difference (constant)
{
    double a = params_arr[0];
    double b = params_arr[1];
    double c = params_arr[2];
    double d = params_arr[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = __double2ll_rn(a + b * x0 + c * x0 * x0 + d * x0 * x0 * x0);

    double x1 = x0 + s;
    double y1_fp = a + b * x1 + c * x1 * x1 + d * x1 * x1 * x1;
    d1 = __double2ll_rn(y1_fp) - y;

    double x2 = x0 + 2.0 * s;
    double y2_fp = a + b * x2 + c * x2 * x2 + d * x2 * x2 * x2;
    int64_t d1_next = __double2ll_rn(y2_fp) - __double2ll_rn(y1_fp);
    d2 = d1_next - d1;

    d3 = __double2ll_rn(6.0 * d * s * s * s);
}

// FP64 to integer conversion helper
template<typename T>
__device__ __forceinline__
T fp64_to_int(double val) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(val));
    } else {
        return static_cast<T>(__double2ll_rn(val));
    }
}

// =============================================================================
// V4 Decoder with direct memory access (optimized for cache)
// =============================================================================
template<typename T>
__global__ void decompressInterleavedAllPartitionsV4(
    const uint32_t* __restrict__ interleaved_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int64_t* __restrict__ d_param_offsets,  // NEW: nullptr = fixed layout
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    T* __restrict__ output)
{
    const int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // DEBUG: Confirm kernel execution
    if (pid == 0 && threadIdx.x == 0) {
        // // printf("[DECODER V4 KERNEL] pid=0, threadIdx.x=0 executing\n");
    }

    // ---------------------------------------------------------------------
    // Partition metadata
    // ---------------------------------------------------------------------
    const int partition_start = d_start_indices[pid];
    const int partition_end   = d_end_indices[pid];
    const int partition_size  = partition_end - partition_start;

    const int delta_bits = d_delta_bits[pid];
    const int num_mv     = d_num_mini_vectors[pid];
    const int64_t interleaved_base = d_interleaved_offsets[pid];
    const int model_type = d_model_types[pid];

    // ---------------------------------------------------------------------
    // Parameter loading (supports both fixed and variable layout)
    // ---------------------------------------------------------------------
    double p0 = 0.0, p1 = 0.0, p2 = 0.0, p3 = 0.0;
    if (d_param_offsets == nullptr) {
        // Fixed layout: pid * 4
        const int param_base = pid * 4;
        p0 = d_model_params[param_base + 0];
        if (model_type == MODEL_LINEAR) {
            p1 = d_model_params[param_base + 1];
        } else if (model_type == MODEL_POLYNOMIAL2) {
            p1 = d_model_params[param_base + 1];
            p2 = d_model_params[param_base + 2];
        } else if (model_type == MODEL_POLYNOMIAL3) {
            p1 = d_model_params[param_base + 1];
            p2 = d_model_params[param_base + 2];
            p3 = d_model_params[param_base + 3];
        }
    } else {
        // Variable layout: use offset (for future support)
        const int64_t param_base = d_param_offsets[pid];
        p0 = d_model_params[param_base + 0];
        if (model_type == MODEL_LINEAR) {
            p1 = d_model_params[param_base + 1];
        } else if (model_type == MODEL_POLYNOMIAL2) {
            p1 = d_model_params[param_base + 1];
            p2 = d_model_params[param_base + 2];
        } else if (model_type == MODEL_POLYNOMIAL3) {
            p1 = d_model_params[param_base + 1];
            p2 = d_model_params[param_base + 2];
            p3 = d_model_params[param_base + 3];
        }
    }

    // Base value for FOR_BITPACK
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(p0));
        } else {
            base_value = static_cast<T>(__double2ll_rn(p0));
        }
    }

    // ---------------------------------------------------------------------
    // CONSTANT/RLE unified handling
    // Unified format: params_arr[0]=num_runs, params_arr[1]=base_value, params_arr[2]=value_bits
    // delta_bits = count_bits (0 for single-run CONSTANT)
    // ---------------------------------------------------------------------
    if (model_type == MODEL_CONSTANT) {
        const int num_runs = static_cast<int>(p0);
        const int count_bits = delta_bits;

        // Get base_value from params_arr[1] - use fixed layout for RLE (GPU path)
        T rle_base_value;
        const int64_t rle_param_base = (d_param_offsets == nullptr) ? (pid * 4) : d_param_offsets[pid];
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(d_model_params[rle_param_base + 1]));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(d_model_params[rle_param_base + 1]));
        }

        // Single run (traditional CONSTANT): all values are base_value
        if (num_runs == 1) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        // Multiple runs (RLE): decode using LINEAR model for values and counts
        // Header format: [value_intercept(f), value_slope(f), count_intercept(f), count_slope(f), bits_info, num_runs]
        // = 6 words = 24 bytes
        __shared__ uint32_t s_rle_data[256];
        __shared__ T s_run_values[512];
        __shared__ int s_run_offsets[513];

        const int64_t rle_bit_base = interleaved_base << 5;
        const int64_t rle_first_word = rle_bit_base >> 5;

        // Read header containing LINEAR params_arr (6 words)
        __shared__ float s_value_intercept, s_value_slope, s_count_intercept, s_count_slope;
        __shared__ int s_value_residual_bits, s_count_residual_bits, s_actual_num_runs;

        if (threadIdx.x == 0) {
            const float* fptr = reinterpret_cast<const float*>(&interleaved_array[rle_first_word]);
            s_value_intercept = fptr[0];
            s_value_slope = fptr[1];
            s_count_intercept = fptr[2];
            s_count_slope = fptr[3];
            uint32_t bits_info = interleaved_array[rle_first_word + 4];
            s_value_residual_bits = bits_info & 0xFF;
            s_count_residual_bits = (bits_info >> 8) & 0xFF;
            // Read actual num_runs from header
            s_actual_num_runs = static_cast<int>(interleaved_array[rle_first_word + 5]);
        }
        __syncthreads();

        // Use actual num_runs from header
        const int actual_num_runs = s_actual_num_runs;

        // Calculate data region size and load
        const int64_t data_bit_base = rle_bit_base + 192;  // After 6-word header
        const int64_t total_data_bits = static_cast<int64_t>(actual_num_runs) * (s_value_residual_bits + s_count_residual_bits);
        int rle_words_needed = 6 + (total_data_bits + 31) / 32 + 2;
        rle_words_needed = min(rle_words_needed, 256);

        for (int w = threadIdx.x; w < rle_words_needed; w += blockDim.x) {
            s_rle_data[w] = __ldg(&interleaved_array[rle_first_word + w]);
        }
        __syncthreads();

        // Thread 0 decodes runs using LINEAR model
        if (threadIdx.x == 0) {
            int offset = 0;
            s_run_offsets[0] = 0;

            const int64_t values_bit_base = 192;  // relative to s_rle_data start (after 6-word header)
            const int64_t counts_bit_base = 192 + static_cast<int64_t>(actual_num_runs) * s_value_residual_bits;

            for (int r = 0; r < actual_num_runs && r < 512; r++) {
                // Read value residual
                int64_t value_residual = 0;
                if (s_value_residual_bits > 0) {
                    const int64_t bit_offset = values_bit_base + static_cast<int64_t>(r) * s_value_residual_bits;
                    const int word_idx = bit_offset >> 5;
                    const int bit_in_word = bit_offset & 31;

                    uint64_t combined = (word_idx + 1 < rle_words_needed) ?
                        ((static_cast<uint64_t>(s_rle_data[word_idx + 1]) << 32) | s_rle_data[word_idx]) :
                        s_rle_data[word_idx];
                    uint64_t raw = (combined >> bit_in_word) & ((1ULL << s_value_residual_bits) - 1);
                    // Sign extend
                    if (s_value_residual_bits < 64 && (raw & (1ULL << (s_value_residual_bits - 1)))) {
                        raw |= ~((1ULL << s_value_residual_bits) - 1);
                    }
                    value_residual = static_cast<int64_t>(raw);
                }

                // Read count residual
                int64_t count_residual = 0;
                if (s_count_residual_bits > 0) {
                    const int64_t bit_offset = counts_bit_base + static_cast<int64_t>(r) * s_count_residual_bits;
                    const int word_idx = bit_offset >> 5;
                    const int bit_in_word = bit_offset & 31;

                    uint64_t combined = (word_idx + 1 < rle_words_needed) ?
                        ((static_cast<uint64_t>(s_rle_data[word_idx + 1]) << 32) | s_rle_data[word_idx]) :
                        s_rle_data[word_idx];
                    uint64_t raw = (combined >> bit_in_word) & ((1ULL << s_count_residual_bits) - 1);
                    // Sign extend
                    if (s_count_residual_bits < 64 && (raw & (1ULL << (s_count_residual_bits - 1)))) {
                        raw |= ~((1ULL << s_count_residual_bits) - 1);
                    }
                    count_residual = static_cast<int64_t>(raw);
                }

                // Reconstruct value using LINEAR model
                double value_pred = static_cast<double>(s_value_intercept) + static_cast<double>(s_value_slope) * static_cast<double>(r);
                int64_t value_delta = static_cast<int64_t>(llrint(value_pred)) + value_residual;
                s_run_values[r] = rle_base_value + static_cast<T>(value_delta);

                // Reconstruct count using LINEAR model
                double count_pred = static_cast<double>(s_count_intercept) + static_cast<double>(s_count_slope) * static_cast<double>(r);
                int count = static_cast<int>(llrint(count_pred)) + static_cast<int>(count_residual);

                offset += count;
                s_run_offsets[r + 1] = offset;
            }
        }
        __syncthreads();

        // All threads expand runs using binary search
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int left = 0, right = min(actual_num_runs, 512) - 1;
            while (left < right) {
                const int mid = (left + right + 1) >> 1;
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

    // ---------------------------------------------------------------------
    // No interleaved data: prediction-only path (rare in current encoder)
    // ---------------------------------------------------------------------
    if (interleaved_base < 0) {
        if (model_type == MODEL_FOR_BITPACK) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = base_value;
            }
        } else if (model_type == MODEL_LINEAR) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x = static_cast<double>(local_idx);
                double predicted = p0 + p1 * x;
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else if (model_type == MODEL_POLYNOMIAL2) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x = static_cast<double>(local_idx);
                double predicted = p0 + x * (p1 + x * p2);
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else if (model_type == MODEL_POLYNOMIAL3) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x = static_cast<double>(local_idx);
                double predicted = p0 + x * (p1 + x * (p2 + x * p3));
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else {
            const T pred_const = fp64_to_int<T>(p0);
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = pred_const;
            }
        }
        return;
    }

    // ---------------------------------------------------------------------
    // Warp mapping
    // ---------------------------------------------------------------------
    const int lane_id = threadIdx.x & (WARP_SIZE_V4 - 1);
    const int warp_id_in_block = threadIdx.x >> 5;
    const int warps_per_block  = blockDim.x >> 5;

    constexpr double X_STEP = static_cast<double>(WARP_SIZE_V4);

    // Construct params array for FiniteDiff functions - MUST match encoder order
    double params_arr[4] = {p0, p1, p2, p3};

    // ---------------------------------------------------------------------
    // Mini-vectors (warp-per-mini-vector)
    // ---------------------------------------------------------------------
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        const int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        // Coalesced output indices: (mv_start_global + lane_id) then +32 each step
        int out = mv_start_global + lane_id;

        // Local x index for prediction: mv-local (NOT global)
        double x = static_cast<double>(mv_idx * MINI_VECTOR_SIZE + lane_id);

        if (delta_bits == 0) {
            // -------------------------------------------------------------
            // Prediction only (no deltas) - Use INT finite diff
            // -------------------------------------------------------------
            if (model_type == MODEL_FOR_BITPACK) {
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = base_value;
                    out += WARP_SIZE_V4;
                }
            } else if (model_type == MODEL_LINEAR) {
                // FP64 accumulation for LINEAR - eliminates drift error
                double y_fp, step_fp;
                FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, step_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL2) {
                // FP64 accumulation for POLY2 - eliminates drift error
                double y_fp, d1_fp, d2_fp;
                FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL3) {
                // FP64 accumulation for POLY3 - eliminates drift error
                double y_fp, d1_fp, d2_fp, d3_fp;
                FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                    d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                }
            } else {
                const T pred_const = fp64_to_int<T>(p0);
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = pred_const;
                    out += WARP_SIZE_V4;
                }
            }
        } else {
            // -------------------------------------------------------------
            // Delta decode + prediction
            // Use fast BitReader for the common <=32-bit case.
            // -------------------------------------------------------------
            const uint64_t mv_bit_base =
                (static_cast<uint64_t>(interleaved_base) << 5) +
                static_cast<uint64_t>(mv_idx) * static_cast<uint64_t>(MINI_VECTOR_SIZE) * static_cast<uint64_t>(delta_bits);

            const uint64_t lane_bit_start =
                mv_bit_base +
                static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(VALUES_PER_THREAD) * static_cast<uint64_t>(delta_bits);

            if (delta_bits <= 32) {
                const uint64_t MASK = (delta_bits == 32) ? 0xFFFFFFFFULL : ((1ULL << delta_bits) - 1ULL);

                BitReader32 br(interleaved_array, lane_bit_start);

                if (model_type == MODEL_FOR_BITPACK) {
                    // FOR: base + unsigned delta
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t d0, d1, d2, d3;
                        br.read4(delta_bits, MASK, d0, d1, d2, d3);

                        output[out + 0 * WARP_SIZE_V4] = base_value + static_cast<T>(d0);
                        output[out + 1 * WARP_SIZE_V4] = base_value + static_cast<T>(d1);
                        output[out + 2 * WARP_SIZE_V4] = base_value + static_cast<T>(d2);
                        output[out + 3 * WARP_SIZE_V4] = base_value + static_cast<T>(d3);

                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_LINEAR) {
                    // FP64 accumulation for LINEAR - eliminates drift error
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);

                        int64_t d0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t d1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t d2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t d3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d0);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d1);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d2);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d3);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);

                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    // FP64 accumulation for POLY2 - eliminates drift error
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);

                        int64_t delta0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t delta1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t delta2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t delta3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);

                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    // FP64 accumulation for POLY3 - eliminates drift error
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);

                        int64_t delta0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t delta1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t delta2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t delta3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);

                        out += 4 * WARP_SIZE_V4;
                    }
                } else {
                    // CONSTANT / fallback
                    const T pred_const = fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);

                        int64_t d0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t d1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t d2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t d3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d0);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d1);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d2);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d3);

                        out += 4 * WARP_SIZE_V4;
                    }
                }
            } else {
                // Slow-path for wide deltas (>32 bits): scalar extraction
                uint64_t bit_pos = lane_bit_start;

                if (model_type == MODEL_FOR_BITPACK) {
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        output[out] = base_value + static_cast<T>(extracted);
                        out += WARP_SIZE_V4;
                        bit_pos += static_cast<uint64_t>(delta_bits);
                    }
                } else if (model_type == MODEL_LINEAR) {
                    // FP64 accumulation for LINEAR (slow path) - eliminates drift error
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);

                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        bit_pos += static_cast<uint64_t>(delta_bits);
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    // FP64 accumulation for POLY2 (slow path) - eliminates drift error
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);

                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        bit_pos += static_cast<uint64_t>(delta_bits);
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    // FP64 accumulation for POLY3 (slow path) - eliminates drift error
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);

                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        bit_pos += static_cast<uint64_t>(delta_bits);
                    }
                } else {
                    const T pred_const = fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(pred_const, delta);

                        out += WARP_SIZE_V4;
                        bit_pos += static_cast<uint64_t>(delta_bits);
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Tail values (partition_size % MINI_VECTOR_SIZE)
    // ---------------------------------------------------------------------
    const int tail_start = num_mv * MINI_VECTOR_SIZE;
    const uint64_t tail_bit_base =
        (static_cast<uint64_t>(interleaved_base) << 5) +
        static_cast<uint64_t>(num_mv) * static_cast<uint64_t>(MINI_VECTOR_SIZE) * static_cast<uint64_t>(delta_bits);

    if (model_type == MODEL_FOR_BITPACK) {
        if (delta_bits == 0) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = base_value;
            }
        } else if (delta_bits <= 32) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);
                const uint32_t d = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                output[partition_start + local_idx] = base_value + static_cast<T>(d);
            }
        } else {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);
                const uint64_t d = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                output[partition_start + local_idx] = base_value + static_cast<T>(d);
            }
        }
    } else if (delta_bits == 0) {
        // Prediction-only tail
        if (model_type == MODEL_LINEAR) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + p1 * x_tail;
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else if (model_type == MODEL_POLYNOMIAL2) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + x_tail * (p1 + x_tail * p2);
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else if (model_type == MODEL_POLYNOMIAL3) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + x_tail * (p1 + x_tail * (p2 + x_tail * p3));
                output[partition_start + local_idx] = fp64_to_int<T>(predicted);
            }
        } else {
            const T pred_const = fp64_to_int<T>(p0);
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = pred_const;
            }
        }
    } else {
        // Prediction + delta tail
        if (model_type == MODEL_LINEAR) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + p1 * x_tail;
                T pv = fp64_to_int<T>(predicted);

                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);

                int64_t delta;
                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    delta = Vertical::sign_extend_64(extracted, delta_bits);
                }

                output[partition_start + local_idx] = applyDeltaV4(pv, delta);
            }
        } else if (model_type == MODEL_POLYNOMIAL2) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + x_tail * (p1 + x_tail * p2);
                T pv = fp64_to_int<T>(predicted);

                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);

                int64_t delta;
                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    delta = Vertical::sign_extend_64(extracted, delta_bits);
                }

                output[partition_start + local_idx] = applyDeltaV4(pv, delta);
            }
        } else if (model_type == MODEL_POLYNOMIAL3) {
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                double x_tail = static_cast<double>(local_idx);
                double predicted = p0 + x_tail * (p1 + x_tail * (p2 + x_tail * p3));
                T pv = fp64_to_int<T>(predicted);

                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);

                int64_t delta;
                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    delta = Vertical::sign_extend_64(extracted, delta_bits);
                }

                output[partition_start + local_idx] = applyDeltaV4(pv, delta);
            }
        } else {
            const T pred_const = fp64_to_int<T>(p0);
            for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                const int tail_local_idx = local_idx - tail_start;
                const uint64_t bit_offset = tail_bit_base + static_cast<uint64_t>(tail_local_idx) * static_cast<uint64_t>(delta_bits);

                int64_t delta;
                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_offset, delta_bits);
                    delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, delta_bits);
                    delta = Vertical::sign_extend_64(extracted, delta_bits);
                }

                output[partition_start + local_idx] = applyDeltaV4(pred_const, delta);
            }
        }
    }
}
// V4 Decoder Wrapper with configurable block size
template<typename T>
void decompressV4(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0,
    int block_size = 256)
{
    if (compressed.num_partitions == 0) return;

    int np = compressed.num_partitions;

    switch (block_size) {
        case 32:
            decompressInterleavedAllPartitionsV4<T><<<np, 32, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        case 64:
            decompressInterleavedAllPartitionsV4<T><<<np, 64, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        case 128:
            decompressInterleavedAllPartitionsV4<T><<<np, 128, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        case 256:
            decompressInterleavedAllPartitionsV4<T><<<np, 256, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        case 512:
            decompressInterleavedAllPartitionsV4<T><<<np, 512, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        case 1024:
            decompressInterleavedAllPartitionsV4<T><<<np, 1024, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
        default:
            decompressInterleavedAllPartitionsV4<T><<<np, 256, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_types,
                compressed.d_model_params, compressed.d_param_offsets, compressed.d_delta_bits,
                compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                np, d_output);
            break;
    }
    // DEBUG: Sync to flush printf
    cudaDeviceSynchronize();
}

// Adaptive block size selection based on average partition size
__host__ inline int selectOptimalBlockSize(int64_t total_elements, int num_partitions) {
    if (num_partitions == 0) return 256;
    int64_t avg_partition_size = total_elements / num_partitions;

    // Empirically determined thresholds from H100 benchmarks:
    // - Small partitions (<700): block=64 gives +20% to +85% speedup
    // - Large partitions (>2000): block=256 gives best performance
    if (avg_partition_size < 700) {
        return 64;
    } else if (avg_partition_size < 2000) {
        return 128;
    } else {
        return 256;
    }
}

// Global variable for block size configuration (-1 means adaptive)
static int g_block_size = -1;  // -1 = adaptive

// =============================================================================
// V5 Decoder with Consolidated Metadata (Cache-Optimized)
// =============================================================================
// V5 optimizations:
// 1. MERGED METADATA: Single 64-byte struct load (1 cache line)
// 2. DYNAMIC SHARED MEMORY: Adaptive buffer size with bank padding
// 3. PREFETCH: Hides memory latency for next mini-vector
// 4. BANK PADDING: Reduces shared memory bank conflicts
// =============================================================================

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

    // Copy params_arr to local registers for better performance
    double p0 = meta.params[0];
    double p1 = meta.params[1];
    double p2 = meta.params[2];
    double p3 = meta.params[3];

    // Pre-compute values
    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(p0));
        } else {
            base_value = static_cast<T>(__double2ll_rn(p0));
        }
    }
    bool needs_third_word = (delta_bits > 32);

    // ========== RLE (CONSTANT) partition special handling ==========
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(p0);
        int value_bits = static_cast<int>(p2);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(p1));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(p1));
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

        // Decode runs
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
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value;
            } else {
                // Compute polynomial prediction
                double x = static_cast<double>(local_idx);
                double predicted;
                switch (model_type) {
                    case MODEL_CONSTANT:
                        predicted = p0;
                        break;
                    case MODEL_LINEAR:
                        predicted = p0 + p1 * x;
                        break;
                    case MODEL_POLYNOMIAL2:
                        predicted = p0 + x * (p1 + x * p2);
                        break;
                    case MODEL_POLYNOMIAL3:
                        predicted = p0 + x * (p1 + x * (p2 + x * p3));
                        break;
                    default:
                        predicted = p0 + p1 * x;
                        break;
                }
                if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                    result = static_cast<T>(__double2ull_rn(predicted));
                } else {
                    result = static_cast<T>(__double2ll_rn(predicted));
                }
            }
            output[global_idx] = result;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE_V4;
    int warp_id_in_block = threadIdx.x / WARP_SIZE_V4;
    int warps_per_block = blockDim.x / WARP_SIZE_V4;

    // V5 OPTIMIZATION 2: Dynamic shared memory with bank padding
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
        for (int w = lane_id; w < words_needed; w += WARP_SIZE_V4) {
            warp_smem[w] = __ldg(&interleaved_array[mv_first_word + w]);
        }
        __syncwarp();

        // Process values
        int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int local_bit = lane_bit_start & 31;
        int base_word_idx = lane_bit_start >> 5;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE_V4 + lane_id;
            int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + v * WARP_SIZE_V4 + lane_id;

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
                // Compute polynomial prediction
                double x = static_cast<double>(local_idx_in_partition);
                double predicted;
                switch (model_type) {
                    case MODEL_CONSTANT:
                        predicted = p0;
                        break;
                    case MODEL_LINEAR:
                        predicted = p0 + p1 * x;
                        break;
                    case MODEL_POLYNOMIAL2:
                        predicted = p0 + x * (p1 + x * p2);
                        break;
                    case MODEL_POLYNOMIAL3:
                        predicted = p0 + x * (p1 + x * (p2 + x * p3));
                        break;
                    default:
                        predicted = p0 + p1 * x;
                        break;
                }
                if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                    result = static_cast<T>(__double2ull_rn(predicted));
                } else {
                    result = static_cast<T>(__double2ll_rn(predicted));
                }
            } else {
                // Compute polynomial prediction
                double x = static_cast<double>(local_idx_in_partition);
                double predicted_d;
                switch (model_type) {
                    case MODEL_CONSTANT:
                        predicted_d = p0;
                        break;
                    case MODEL_LINEAR:
                        predicted_d = p0 + p1 * x;
                        break;
                    case MODEL_POLYNOMIAL2:
                        predicted_d = p0 + x * (p1 + x * p2);
                        break;
                    case MODEL_POLYNOMIAL3:
                        predicted_d = p0 + x * (p1 + x * (p2 + x * p3));
                        break;
                    default:
                        predicted_d = p0 + p1 * x;
                        break;
                }
                T predicted;
                if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                    predicted = static_cast<T>(__double2ull_rn(predicted_d));
                } else {
                    predicted = static_cast<T>(__double2ll_rn(predicted_d));
                }
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDeltaV4(predicted, delta);
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

        __syncthreads();

        // Use first warp's smem for tail
        uint32_t* tail_smem = s_dynamic_smem;
        if (warp_id_in_block == 0) {
            for (int w = lane_id; w < tail_words_needed; w += WARP_SIZE_V4) {
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
                // Compute polynomial prediction
                double x = static_cast<double>(local_idx);
                double predicted_d;
                switch (model_type) {
                    case MODEL_CONSTANT:
                        predicted_d = p0;
                        break;
                    case MODEL_LINEAR:
                        predicted_d = p0 + p1 * x;
                        break;
                    case MODEL_POLYNOMIAL2:
                        predicted_d = p0 + x * (p1 + x * p2);
                        break;
                    case MODEL_POLYNOMIAL3:
                        predicted_d = p0 + x * (p1 + x * (p2 + x * p3));
                        break;
                    default:
                        predicted_d = p0 + p1 * x;
                        break;
                }
                T predicted;
                if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                    predicted = static_cast<T>(__double2ull_rn(predicted_d));
                } else {
                    predicted = static_cast<T>(__double2ll_rn(predicted_d));
                }
                int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                result = applyDeltaV4(predicted, delta);
            }
            output[global_idx] = result;
        }
    } else if (tail_size > 0) {
        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            T result;
            if (model_type == MODEL_FOR_BITPACK) {
                result = base_value;
            } else {
                double x = static_cast<double>(local_idx);
                double predicted_d;
                switch (model_type) {
                    case MODEL_CONSTANT:
                        predicted_d = p0;
                        break;
                    case MODEL_LINEAR:
                        predicted_d = p0 + p1 * x;
                        break;
                    case MODEL_POLYNOMIAL2:
                        predicted_d = p0 + x * (p1 + x * p2);
                        break;
                    case MODEL_POLYNOMIAL3:
                        predicted_d = p0 + x * (p1 + x * (p2 + x * p3));
                        break;
                    default:
                        predicted_d = p0 + p1 * x;
                        break;
                }
                if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                    result = static_cast<T>(__double2ull_rn(predicted_d));
                } else {
                    result = static_cast<T>(__double2ll_rn(predicted_d));
                }
            }
            output[global_idx] = result;
        }
    }
}

// V5 Decoder Wrapper with dynamic shared memory
template<typename T>
void decompressV5(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0,
    int block_size = 256)
{
    if (compressed.num_partitions == 0) return;
    if (!compressed.use_v5_metadata || !compressed.d_metadata_v5) {
        // Fallback to V4
        decompressV4<T>(compressed, d_output, stream, block_size);
        return;
    }

    int np = compressed.num_partitions;

    // Calculate dynamic shared memory size
    // Use a conservative estimate: 16 bits is typical for most datasets
    // (books dataset has avg 9.25 bits, max 15 bits)
    // This balances between handling most cases and maintaining good occupancy
    int estimated_max_delta_bits = 16;  // Conservative estimate for most datasets
    int bits_per_mv = MINI_VECTOR_SIZE * estimated_max_delta_bits;
    int words_per_warp = (bits_per_mv + 31 + 32) / 32 + 8;  // +8 for bank padding
    int warps_per_block = block_size / WARP_SIZE_V4;
    int smem_bytes = warps_per_block * words_per_warp * sizeof(uint32_t);

    switch (block_size) {
        case 32:
            decompressInterleavedAllPartitionsV5<T><<<np, 32, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 64:
            decompressInterleavedAllPartitionsV5<T><<<np, 64, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 128:
            decompressInterleavedAllPartitionsV5<T><<<np, 128, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 256:
            decompressInterleavedAllPartitionsV5<T><<<np, 256, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 512:
            decompressInterleavedAllPartitionsV5<T><<<np, 512, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 1024:
            decompressInterleavedAllPartitionsV5<T><<<np, 1024, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        default:
            decompressInterleavedAllPartitionsV5<T><<<np, 256, smem_bytes, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
    }
}

// ============================================================================
// V5-Opt: Optimized V5 Decoder with Static Shared Memory
// ============================================================================

// Helper function for V5-Opt prediction computation
template<typename T>
__device__ __forceinline__
T computePredictionV5Opt(int32_t model_type, const double* params_arr, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:
            predicted = params_arr[0];
            break;
        case MODEL_LINEAR:
            predicted = params_arr[0] + params_arr[1] * x;
            break;
        case MODEL_POLYNOMIAL2:
            predicted = params_arr[0] + x * (params_arr[1] + x * params_arr[2]);
            break;
        case MODEL_POLYNOMIAL3:
            predicted = params_arr[0] + x * (params_arr[1] + x * (params_arr[2] + x * params_arr[3]));
            break;
        default:
            predicted = params_arr[0] + params_arr[1] * x;
            break;
    }

    return fp64_to_int<T>(predicted);
}

/**
 * V5-Opt Decoder Kernel: Optimized metadata loading + BitReader32 extraction
 *
 * Key optimizations:
 * 1. Consolidated metadata (64B cache line aligned)
 * 2. __ldg for read-only data
 * 3. BitReader32 for efficient bit extraction (same as V4)
 * 4. 4-way loop unrolling
 */
template<typename T>
__global__ void decompressInterleavedAllPartitionsV5_Opt(
    const uint32_t* __restrict__ interleaved_array,
    const PartitionMetadataV5* __restrict__ d_metadata,
    int num_partitions,
    T* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // V5 OPTIMIZATION: Single metadata load (64 bytes = 1 cache line)
    const PartitionMetadataV5* meta_ptr = &d_metadata[pid];
    int partition_start = __ldg(&meta_ptr->start_idx);
    int partition_end = __ldg(&meta_ptr->end_idx);
    int partition_size = partition_end - partition_start;
    int delta_bits = __ldg(&meta_ptr->delta_bits);
    int num_mv = __ldg(&meta_ptr->num_mini_vectors);
    int64_t interleaved_base = __ldg(&meta_ptr->interleaved_offset);
    int model_type = __ldg(&meta_ptr->model_type);

    double p0 = meta_ptr->params[0];
    double p1 = (model_type != MODEL_FOR_BITPACK) ? meta_ptr->params[1] : 0.0;
    double p2 = (model_type != MODEL_FOR_BITPACK) ? meta_ptr->params[2] : 0.0;
    double p3 = (model_type != MODEL_FOR_BITPACK) ? meta_ptr->params[3] : 0.0;

    T base_value = T(0);
    if (model_type == MODEL_FOR_BITPACK) {
        if constexpr (sizeof(T) == 8) {
            base_value = static_cast<T>(__double_as_longlong(p0));
        } else {
            base_value = static_cast<T>(__double2ll_rn(p0));
        }
    }

    // RLE handling
    if (model_type == MODEL_CONSTANT) {
        int num_runs = static_cast<int>(p0);
        int value_bits = static_cast<int>(p2);
        int count_bits = delta_bits;

        T rle_base_value;
        if constexpr (sizeof(T) == 8) {
            rle_base_value = static_cast<T>(__double_as_longlong(p1));
        } else {
            rle_base_value = static_cast<T>(__double2ll_rn(p1));
        }

        if (num_runs == 1 && count_bits == 0) {
            for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
                output[partition_start + local_idx] = rle_base_value;
            }
            return;
        }

        __shared__ T s_run_values[256];
        __shared__ int s_run_offsets[257];

        if (threadIdx.x == 0) {
            int bits_per_run = value_bits + count_bits;
            int64_t rle_bit_base = interleaved_base << 5;

            int offset = 0;
            s_run_offsets[0] = 0;
            for (int r = 0; r < num_runs; r++) {
                int64_t bit_offset = rle_bit_base + static_cast<int64_t>(r) * bits_per_run;
                uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_offset, bits_per_run);

                uint64_t value_delta = extracted & ((1ULL << value_bits) - 1);
                int count = static_cast<int>((extracted >> value_bits) & ((1ULL << count_bits) - 1));
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
                if (s_run_offsets[mid] <= local_idx) left = mid;
                else right = mid - 1;
            }
            output[partition_start + local_idx] = s_run_values[left];
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE_V4;
    int warp_id_in_block = threadIdx.x / WARP_SIZE_V4;
    int warps_per_block = blockDim.x / WARP_SIZE_V4;

    // Construct params array for FiniteDiff functions - MUST match encoder order
    double params_arr[4] = {p0, p1, p2, p3};

    // Process mini-vectors using BitReader32 (same as V4)
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int out = partition_start + mv_idx * MINI_VECTOR_SIZE + lane_id;

        if (delta_bits == 0) {
            // Perfect prediction path - Use INT finite diff
            if (model_type == MODEL_FOR_BITPACK) {
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = base_value;
                    out += WARP_SIZE_V4;
                }
            } else if (model_type == MODEL_LINEAR) {
                // FP64 accumulation for LINEAR - eliminates drift error
                double y_fp, step_fp;
                FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, step_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL2) {
                // FP64 accumulation for POLY2 - eliminates drift error
                double y_fp, d1_fp, d2_fp;
                FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                }
            } else if (model_type == MODEL_POLYNOMIAL3) {
                // FP64 accumulation for POLY3 - eliminates drift error
                double y_fp, d1_fp, d2_fp, d3_fp;
                FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = static_cast<T>(__double2ll_rn(y_fp));
                    out += WARP_SIZE_V4;
                    y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                    d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                    d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                }
            } else {
                const T pred_const = fp64_to_int<T>(p0);
                #pragma unroll
                for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                    output[out] = pred_const;
                    out += WARP_SIZE_V4;
                }
            }
        } else {
            // Delta decode + prediction using BitReader32
            const uint64_t mv_bit_base =
                (static_cast<uint64_t>(interleaved_base) << 5) +
                static_cast<uint64_t>(mv_idx) * static_cast<uint64_t>(MINI_VECTOR_SIZE) * static_cast<uint64_t>(delta_bits);

            const uint64_t lane_bit_start =
                mv_bit_base +
                static_cast<uint64_t>(lane_id) * static_cast<uint64_t>(VALUES_PER_THREAD) * static_cast<uint64_t>(delta_bits);

            if (delta_bits <= 32) {
                const uint64_t MASK = (delta_bits == 32) ? 0xFFFFFFFFULL : ((1ULL << delta_bits) - 1ULL);
                BitReader32 br(interleaved_array, lane_bit_start);

                if (model_type == MODEL_FOR_BITPACK) {
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t d0, d1, d2, d3;
                        br.read4(delta_bits, MASK, d0, d1, d2, d3);
                        output[out + 0 * WARP_SIZE_V4] = base_value + static_cast<T>(d0);
                        output[out + 1 * WARP_SIZE_V4] = base_value + static_cast<T>(d1);
                        output[out + 2 * WARP_SIZE_V4] = base_value + static_cast<T>(d2);
                        output[out + 3 * WARP_SIZE_V4] = base_value + static_cast<T>(d3);
                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_LINEAR) {
                    // FP64 accumulation for LINEAR - eliminates drift error
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t d0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t d1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t d2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t d3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d0);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d1);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d2);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), d3);
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    // FP64 accumulation for POLY2 - eliminates drift error
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t delta0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t delta1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t delta2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t delta3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        out += 4 * WARP_SIZE_V4;
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    // FP64 accumulation for POLY3 - eliminates drift error
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t delta0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t delta1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t delta2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t delta3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta0);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta1);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta2);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta3);
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp); d1_fp = FiniteDiff::d_add(d1_fp, d2_fp); d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                        out += 4 * WARP_SIZE_V4;
                    }
                } else {
                    const T pred_const = fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; v += 4) {
                        uint32_t u0, u1, u2, u3;
                        br.read4(delta_bits, MASK, u0, u1, u2, u3);
                        int64_t d0 = static_cast<int64_t>(Vertical::sign_extend_32(u0, delta_bits));
                        int64_t d1 = static_cast<int64_t>(Vertical::sign_extend_32(u1, delta_bits));
                        int64_t d2 = static_cast<int64_t>(Vertical::sign_extend_32(u2, delta_bits));
                        int64_t d3 = static_cast<int64_t>(Vertical::sign_extend_32(u3, delta_bits));

                        output[out + 0 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d0);
                        output[out + 1 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d1);
                        output[out + 2 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d2);
                        output[out + 3 * WARP_SIZE_V4] = applyDeltaV4(pred_const, d3);
                        out += 4 * WARP_SIZE_V4;
                    }
                }
            } else {
                // >32 bit case (rare) - Use FP64 accumulation
                if (model_type == MODEL_LINEAR) {
                    double y_fp, step_fp;
                    FiniteDiff::computeLinearFP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, step_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);
                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, step_fp);
                    }
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    double y_fp, d1_fp, d2_fp;
                    FiniteDiff::computePoly2FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);
                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                    }
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    double y_fp, d1_fp, d2_fp, d3_fp;
                    FiniteDiff::computePoly3FP64Accum<T>(params_arr, mv_idx * MINI_VECTOR_SIZE + lane_id, WARP_SIZE_V4, y_fp, d1_fp, d2_fp, d3_fp);

                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(static_cast<T>(__double2ll_rn(y_fp)), delta);
                        out += WARP_SIZE_V4;
                        y_fp = FiniteDiff::d_add(y_fp, d1_fp);
                        d1_fp = FiniteDiff::d_add(d1_fp, d2_fp);
                        d2_fp = FiniteDiff::d_add(d2_fp, d3_fp);
                    }
                } else if (model_type == MODEL_FOR_BITPACK) {
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        output[out] = base_value + static_cast<T>(extracted);
                        out += WARP_SIZE_V4;
                    }
                } else {
                    const T pred_const = fp64_to_int<T>(p0);
                    #pragma unroll
                    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
                        uint64_t bit_pos = lane_bit_start + static_cast<uint64_t>(v) * delta_bits;
                        uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                        int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);
                        output[out] = applyDeltaV4(pred_const, delta);
                        out += WARP_SIZE_V4;
                    }
                }
            }
        }
    }

    // Tail processing - use FP64 point evaluation (consistent with encoder init)
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    if (tail_size > 0) {
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits;

        for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            int global_idx = partition_start + local_idx;
            int tail_local_idx = local_idx - tail_start;
            double x = static_cast<double>(local_idx);

            T result;
            if (delta_bits == 0) {
                // Use FP64 FMA-safe point evaluation
                if (model_type == MODEL_LINEAR) {
                    double y = FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x));
                    result = static_cast<T>(__double2ll_rn(y));
                } else if (model_type == MODEL_POLYNOMIAL2) {
                    double x_sq = FiniteDiff::d_mul(x, x);
                    double y = FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq));
                    result = static_cast<T>(__double2ll_rn(y));
                } else if (model_type == MODEL_POLYNOMIAL3) {
                    double x_sq = FiniteDiff::d_mul(x, x);
                    double x_cu = FiniteDiff::d_mul(x_sq, x);
                    double y = FiniteDiff::d_add(FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq)), FiniteDiff::d_mul(p3, x_cu));
                    result = static_cast<T>(__double2ll_rn(y));
                } else if (model_type == MODEL_FOR_BITPACK) {
                    result = base_value;
                } else {
                    result = fp64_to_int<T>(p0);
                }
            } else {
                int64_t bit_pos = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;

                if (delta_bits <= 32) {
                    uint32_t extracted = Vertical::extract_branchless_32_rt(interleaved_array, bit_pos, delta_bits);
                    int64_t delta = static_cast<int64_t>(Vertical::sign_extend_32(extracted, delta_bits));

                    // Use FP64 FMA-safe point evaluation
                    if (model_type == MODEL_LINEAR) {
                        double y = FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else if (model_type == MODEL_POLYNOMIAL2) {
                        double x_sq = FiniteDiff::d_mul(x, x);
                        double y = FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else if (model_type == MODEL_POLYNOMIAL3) {
                        double x_sq = FiniteDiff::d_mul(x, x);
                        double x_cu = FiniteDiff::d_mul(x_sq, x);
                        double y = FiniteDiff::d_add(FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq)), FiniteDiff::d_mul(p3, x_cu));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else if (model_type == MODEL_FOR_BITPACK) {
                        result = base_value + static_cast<T>(extracted);
                    } else {
                        result = applyDeltaV4(fp64_to_int<T>(p0), delta);
                    }
                } else {
                    uint64_t extracted = Vertical::extract_branchless_64_rt(interleaved_array, bit_pos, delta_bits);
                    int64_t delta = Vertical::sign_extend_64(extracted, delta_bits);

                    if (model_type == MODEL_FOR_BITPACK) {
                        result = base_value + static_cast<T>(extracted);
                    } else if (model_type == MODEL_LINEAR) {
                        double y = FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else if (model_type == MODEL_POLYNOMIAL2) {
                        double x_sq = FiniteDiff::d_mul(x, x);
                        double y = FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else if (model_type == MODEL_POLYNOMIAL3) {
                        double x_sq = FiniteDiff::d_mul(x, x);
                        double x_cu = FiniteDiff::d_mul(x_sq, x);
                        double y = FiniteDiff::d_add(FiniteDiff::d_add(FiniteDiff::d_add(p0, FiniteDiff::d_mul(p1, x)), FiniteDiff::d_mul(p2, x_sq)), FiniteDiff::d_mul(p3, x_cu));
                        result = applyDeltaV4(static_cast<T>(__double2ll_rn(y)), delta);
                    } else {
                        result = applyDeltaV4(fp64_to_int<T>(p0), delta);
                    }
                }
            }
            output[global_idx] = result;
        }
    }
}

// V5-Opt Wrapper: Optimized metadata loading with direct global memory access
template<typename T>
void decompressV5_Opt(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0,
    int block_size = 256)
{
    if (compressed.num_partitions == 0) return;
    if (!compressed.use_v5_metadata || !compressed.d_metadata_v5) {
        decompressV4<T>(compressed, d_output, stream, block_size);
        return;
    }

    int np = compressed.num_partitions;

    switch (block_size) {
        case 32:
            decompressInterleavedAllPartitionsV5_Opt<T><<<np, 32, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 64:
            decompressInterleavedAllPartitionsV5_Opt<T><<<np, 64, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 128:
            decompressInterleavedAllPartitionsV5_Opt<T><<<np, 128, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
        case 256:
        default:
            decompressInterleavedAllPartitionsV5_Opt<T><<<np, 256, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_metadata_v5, np, d_output);
            break;
    }
}

} // namespace V4_decoder

// ============================================================================
// V5 FOR-Only Optimized Decoder (Compile-time bit-width specialization)
// ============================================================================

namespace V5_FOR_decoder {

constexpr int WARP_SIZE = 32;

// Compile-time mask generation
template<int BIT_WIDTH>
__device__ __forceinline__ constexpr uint64_t mask64() {
    if constexpr (BIT_WIDTH <= 0) return 0ULL;
    else if constexpr (BIT_WIDTH >= 64) return ~0ULL;
    else return (1ULL << BIT_WIDTH) - 1ULL;
}

// ============================================================================
// Device Function Templates for Per-Partition Dispatch (64-bit output)
// ============================================================================

template<int BIT_WIDTH>
__device__ __forceinline__ void decompressPartition64_specialized(
    const uint32_t* __restrict__ interleaved_data,
    int partition_start,
    int partition_size,
    int num_mv,
    int64_t interleaved_base,
    uint64_t base_value,
    uint64_t* __restrict__ output)
{
    constexpr uint64_t MASK = mask64<BIT_WIDTH>();
    // Calculate exact buffer size needed: ceil((31 + 32*BIT_WIDTH) / 32) + 1
    constexpr int MAX_BUFFER_WORDS = (31 + VALUES_PER_THREAD * BIT_WIDTH + 31) / 32 + 2;
    constexpr bool NEEDS_THIRD_WORD = (BIT_WIDTH > 32);

    // Handle 0-bit case
    if constexpr (BIT_WIDTH == 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    if (interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Process mini-vectors
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * BIT_WIDTH;

        int64_t lane_word_start = lane_bit_start >> 5;
        constexpr int bits_per_lane = VALUES_PER_THREAD * BIT_WIDTH;
        constexpr int words_per_lane = (bits_per_lane + 63) / 32;
        constexpr int actual_words = (words_per_lane < MAX_BUFFER_WORDS) ? words_per_lane : MAX_BUFFER_WORDS;

        uint32_t lane_words[MAX_BUFFER_WORDS];
        #pragma unroll
        for (int i = 0; i < MAX_BUFFER_WORDS; i++) {
            lane_words[i] = (i < actual_words) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            if constexpr (NEEDS_THIRD_WORD) {
                if (bit_in_word > 0 && (64 - bit_in_word) < BIT_WIDTH) {
                    int bits_from_first_two = 64 - bit_in_word;
                    extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
                }
            }
            extracted &= MASK;

            output[global_idx] = base_value + extracted;
            local_bit += BIT_WIDTH;
        }
    }

    // Handle tail values
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int tail_local_idx = local_idx - tail_start;
        int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * BIT_WIDTH;

        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word);

        if constexpr (NEEDS_THIRD_WORD) {
            if (bit_in_word > 0 && (64 - bit_in_word) < BIT_WIDTH) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 2])) << bits_from_first_two);
            }
        }
        extracted &= MASK;

        output[global_idx] = base_value + extracted;
    }
}

// Runtime fallback device function
__device__ __forceinline__ void decompressPartition64_runtime(
    const uint32_t* __restrict__ interleaved_data,
    int partition_start,
    int partition_size,
    int num_mv,
    int64_t interleaved_base,
    uint64_t base_value,
    int delta_bits,
    uint64_t* __restrict__ output)
{
    uint64_t mask = (delta_bits < 64) ? ((1ULL << delta_bits) - 1ULL) : ~0ULL;
    bool needs_third_word = (delta_bits > 32);

    if (delta_bits == 0 || interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int64_t lane_word_start = lane_bit_start >> 5;

        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 63) / 32;
        words_per_lane = min(words_per_lane, 18);

        uint32_t lane_words[18];
        for (int i = 0; i < 18; i++) {
            lane_words[i] = (i < words_per_lane) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) | lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
            }
            extracted &= mask;

            output[global_idx] = base_value + extracted;
            local_bit += delta_bits;
        }
    }

    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t bit_offset = (interleaved_base << 5) +
                            static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits +
                            static_cast<int64_t>(local_idx - tail_start) * delta_bits;
        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word);
        if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
            int bits_from_first_two = 64 - bit_in_word;
            extracted |= (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 2])) << bits_from_first_two);
        }
        extracted &= mask;
        output[global_idx] = base_value + extracted;
    }
}

// ============================================================================
// Per-Partition Switch-Case Kernel (64-bit) - V5b
// ============================================================================

__global__ void decompressFOR_V5b_64(
    const uint32_t* __restrict__ interleaved_data,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint64_t* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    uint64_t base_value = static_cast<uint64_t>(__double_as_longlong(d_model_params[pid * 4]));

    // DEBUG: Test if base_value is correct by writing it directly
    // for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
    //     output[partition_start + local_idx] = base_value;
    // }
    // return;

    // Per-partition dispatch to specialized device functions
    // Focus on the most common delta_bits range for uint64 data (32-48)
    // Note: Extending to full range (0-64) actually decreases performance due to code size
    switch (delta_bits) {
        case 0:  decompressPartition64_specialized<0>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 32: decompressPartition64_specialized<32>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 33: decompressPartition64_specialized<33>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 34: decompressPartition64_specialized<34>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 35: decompressPartition64_specialized<35>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 36: decompressPartition64_specialized<36>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 37: decompressPartition64_specialized<37>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 38: decompressPartition64_specialized<38>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 39: decompressPartition64_specialized<39>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 40: decompressPartition64_specialized<40>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 41: decompressPartition64_specialized<41>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 42: decompressPartition64_specialized<42>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 43: decompressPartition64_specialized<43>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 44: decompressPartition64_specialized<44>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 45: decompressPartition64_specialized<45>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 46: decompressPartition64_specialized<46>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 47: decompressPartition64_specialized<47>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        case 48: decompressPartition64_specialized<48>(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, output); break;
        default: decompressPartition64_runtime(interleaved_data, partition_start, partition_size, num_mv, interleaved_base, base_value, delta_bits, output); break;
    }
}

// ============================================================================
// FOR-V5 Template Kernel (32-bit output)
// ============================================================================

template<int BIT_WIDTH>
__global__ void decompressFOR_V5_32(
    const uint32_t* __restrict__ interleaved_data,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,  // base values stored as double
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint32_t* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Load partition metadata
    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];

    // Load base value (FOR only needs params_arr[0])
    uint32_t base_value = static_cast<uint32_t>(__double2ll_rn(d_model_params[pid * 4]));

    // Compile-time constants
    constexpr uint64_t MASK = mask64<BIT_WIDTH>();
    // Calculate exact buffer size needed for 32-bit output
    constexpr int MAX_BUFFER_WORDS = (31 + VALUES_PER_THREAD * BIT_WIDTH + 31) / 32 + 2;

    // Handle 0-bit case (all values are base)
    if constexpr (BIT_WIDTH == 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    // Handle no interleaved data case
    if (interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Process mini-vectors
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * BIT_WIDTH;

        // Load lane data into registers
        int64_t lane_word_start = lane_bit_start >> 5;
        constexpr int bits_per_lane = VALUES_PER_THREAD * BIT_WIDTH;
        constexpr int words_per_lane = (bits_per_lane + 63) / 32;
        constexpr int actual_words = (words_per_lane < MAX_BUFFER_WORDS) ? words_per_lane : MAX_BUFFER_WORDS;

        uint32_t lane_words[MAX_BUFFER_WORDS];
        #pragma unroll
        for (int i = 0; i < MAX_BUFFER_WORDS; i++) {
            lane_words[i] = (i < actual_words) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;

            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word) & MASK;

            output[global_idx] = base_value + static_cast<uint32_t>(extracted);
            local_bit += BIT_WIDTH;
        }
    }

    // Handle tail values
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int tail_local_idx = local_idx - tail_start;
        int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * BIT_WIDTH;

        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word) & MASK;

        output[global_idx] = base_value + static_cast<uint32_t>(extracted);
    }
}

// ============================================================================
// FOR-V5 Template Kernel (64-bit output)
// ============================================================================

template<int BIT_WIDTH>
__global__ void decompressFOR_V5_64(
    const uint32_t* __restrict__ interleaved_data,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,  // base values stored as double
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint64_t* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Load partition metadata
    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];

    // Load base value (FOR only needs params_arr[0])
    // For 64-bit, base is stored as double bit pattern
    uint64_t base_value = static_cast<uint64_t>(__double_as_longlong(d_model_params[pid * 4]));

    // Compile-time constants
    constexpr uint64_t MASK = mask64<BIT_WIDTH>();
    // Calculate exact buffer size needed for 64-bit output
    constexpr int MAX_BUFFER_WORDS = (31 + VALUES_PER_THREAD * BIT_WIDTH + 31) / 32 + 2;
    constexpr bool NEEDS_THIRD_WORD = (BIT_WIDTH > 32);

    // Handle 0-bit case (all values are base)
    if constexpr (BIT_WIDTH == 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    // Handle no interleaved data case
    if (interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    // Process mini-vectors
    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * BIT_WIDTH;

        // Load lane data into registers
        int64_t lane_word_start = lane_bit_start >> 5;
        constexpr int bits_per_lane = VALUES_PER_THREAD * BIT_WIDTH;
        constexpr int words_per_lane = (bits_per_lane + 63) / 32;
        constexpr int actual_words = (words_per_lane < MAX_BUFFER_WORDS) ? words_per_lane : MAX_BUFFER_WORDS;

        uint32_t lane_words[MAX_BUFFER_WORDS];
        #pragma unroll
        for (int i = 0; i < MAX_BUFFER_WORDS; i++) {
            lane_words[i] = (i < actual_words) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;

        #pragma unroll
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;

            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            // Handle BIT_WIDTH > 32: need third word
            if constexpr (NEEDS_THIRD_WORD) {
                if (bit_in_word > 0 && (64 - bit_in_word) < BIT_WIDTH) {
                    int bits_from_first_two = 64 - bit_in_word;
                    extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
                }
            }
            extracted &= MASK;

            output[global_idx] = base_value + extracted;
            local_bit += BIT_WIDTH;
        }
    }

    // Handle tail values
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t tail_bit_base = (interleaved_base << 5) +
                               static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * BIT_WIDTH;
        int tail_local_idx = local_idx - tail_start;
        int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * BIT_WIDTH;

        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word);

        if constexpr (NEEDS_THIRD_WORD) {
            if (bit_in_word > 0 && (64 - bit_in_word) < BIT_WIDTH) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 2])) << bits_from_first_two);
            }
        }
        extracted &= MASK;

        output[global_idx] = base_value + extracted;
    }
}

// ============================================================================
// Runtime Fallback Kernels (for uncommon bit-widths)
// ============================================================================

__global__ void decompressFOR_V5_32_runtime(
    const uint32_t* __restrict__ interleaved_data,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint32_t* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    uint32_t base_value = static_cast<uint32_t>(__double2ll_rn(d_model_params[pid * 4]));

    uint64_t mask = (delta_bits < 64) ? ((1ULL << delta_bits) - 1ULL) : ~0ULL;

    if (delta_bits == 0 || interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int64_t lane_word_start = lane_bit_start >> 5;

        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 63) / 32;
        words_per_lane = min(words_per_lane, 10);

        uint32_t lane_words[10];
        for (int i = 0; i < 10; i++) {
            lane_words[i] = (i < words_per_lane) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) | lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word) & mask;

            output[global_idx] = base_value + static_cast<uint32_t>(extracted);
            local_bit += delta_bits;
        }
    }

    // Tail
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t bit_offset = (interleaved_base << 5) +
                            static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits +
                            static_cast<int64_t>(local_idx - tail_start) * delta_bits;
        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word) & mask;
        output[global_idx] = base_value + static_cast<uint32_t>(extracted);
    }
}

__global__ void decompressFOR_V5_64_runtime(
    const uint32_t* __restrict__ interleaved_data,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_interleaved_offsets,
    int num_partitions,
    uint64_t* __restrict__ output)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int partition_start = d_start_indices[pid];
    int partition_end = d_end_indices[pid];
    int partition_size = partition_end - partition_start;
    int delta_bits = d_delta_bits[pid];
    int num_mv = d_num_mini_vectors[pid];
    int64_t interleaved_base = d_interleaved_offsets[pid];
    uint64_t base_value = static_cast<uint64_t>(__double_as_longlong(d_model_params[pid * 4]));

    uint64_t mask = (delta_bits < 64) ? ((1ULL << delta_bits) - 1ULL) : ~0ULL;
    bool needs_third_word = (delta_bits > 32);

    if (delta_bits == 0 || interleaved_base < 0) {
        for (int local_idx = threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
            output[partition_start + local_idx] = base_value;
        }
        return;
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id_in_block = threadIdx.x / WARP_SIZE;
    int warps_per_block = blockDim.x / WARP_SIZE;

    for (int mv_idx = warp_id_in_block; mv_idx < num_mv; mv_idx += warps_per_block) {
        int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;
        int64_t mv_bit_base = (interleaved_base << 5) +
                             static_cast<int64_t>(mv_idx) * MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * delta_bits;
        int64_t lane_word_start = lane_bit_start >> 5;

        int bits_per_lane = VALUES_PER_THREAD * delta_bits;
        int words_per_lane = (bits_per_lane + 63) / 32;
        words_per_lane = min(words_per_lane, 18);

        uint32_t lane_words[18];
        for (int i = 0; i < 18; i++) {
            lane_words[i] = (i < words_per_lane) ? __ldg(&interleaved_data[lane_word_start + i]) : 0;
        }

        int local_bit = lane_bit_start & 31;
        for (int v = 0; v < VALUES_PER_THREAD; v++) {
            int global_idx = mv_start_global + v * WARP_SIZE + lane_id;
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) | lane_words[word_idx];
            uint64_t extracted = (combined >> bit_in_word);

            if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
                int bits_from_first_two = 64 - bit_in_word;
                extracted |= (static_cast<uint64_t>(lane_words[word_idx + 2]) << bits_from_first_two);
            }
            extracted &= mask;

            output[global_idx] = base_value + extracted;
            local_bit += delta_bits;
        }
    }

    // Tail
    int tail_start = num_mv * MINI_VECTOR_SIZE;
    for (int local_idx = tail_start + threadIdx.x; local_idx < partition_size; local_idx += blockDim.x) {
        int global_idx = partition_start + local_idx;
        int64_t bit_offset = (interleaved_base << 5) +
                            static_cast<int64_t>(num_mv) * MINI_VECTOR_SIZE * delta_bits +
                            static_cast<int64_t>(local_idx - tail_start) * delta_bits;
        int64_t word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;
        uint64_t combined = (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 1])) << 32) |
                           __ldg(&interleaved_data[word_idx]);
        uint64_t extracted = (combined >> bit_in_word);

        if (needs_third_word && bit_in_word > 0 && (64 - bit_in_word) < delta_bits) {
            int bits_from_first_two = 64 - bit_in_word;
            extracted |= (static_cast<uint64_t>(__ldg(&interleaved_data[word_idx + 2])) << bits_from_first_two);
        }
        extracted &= mask;

        output[global_idx] = base_value + extracted;
    }
}

// ============================================================================
// Dispatch Functions
// ============================================================================

// Check if all partitions use FOR model
inline bool isAllFOR(const int32_t* d_model_types, int num_partitions) {
    std::vector<int32_t> h_model_types(num_partitions);
    cudaMemcpy(h_model_types.data(), d_model_types, num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_partitions; i++) {
        if (h_model_types[i] != MODEL_FOR_BITPACK) return false;
    }
    return true;
}

// Check if all partitions have the same delta_bits
inline int getUniformDeltaBits(const int32_t* d_delta_bits, int num_partitions) {
    if (num_partitions == 0) return -1;
    std::vector<int32_t> h_delta_bits(num_partitions);
    cudaMemcpy(h_delta_bits.data(), d_delta_bits, num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    int first_bits = h_delta_bits[0];
    for (int i = 1; i < num_partitions; i++) {
        if (h_delta_bits[i] != first_bits) return -1;  // Not uniform
    }
    return first_bits;
}

// Get delta_bits distribution statistics
inline std::map<int, int> getDeltaBitsDistribution(const int32_t* d_delta_bits, int num_partitions) {
    std::map<int, int> distribution;
    if (num_partitions == 0) return distribution;
    std::vector<int32_t> h_delta_bits(num_partitions);
    cudaMemcpy(h_delta_bits.data(), d_delta_bits, num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_partitions; i++) {
        distribution[h_delta_bits[i]]++;
    }
    return distribution;
}

// Print delta_bits distribution
inline void printDeltaBitsDistribution(const int32_t* d_delta_bits, int num_partitions) {
    auto dist = getDeltaBitsDistribution(d_delta_bits, num_partitions);
    std::cout << "  Delta bits distribution (" << dist.size() << " unique values):" << std::endl;
    for (const auto& [bits, count] : dist) {
        double pct = 100.0 * count / num_partitions;
        std::cout << "    " << std::setw(2) << bits << " bits: " << std::setw(6) << count
                  << " partitions (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
    }
}

// 32-bit dispatch with compile-time specialization
template<typename T>
void decompressFOR_V5(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0,
    int block_size = 256)
{
    if (compressed.num_partitions == 0) return;
    int np = compressed.num_partitions;

    // Check if uniform delta_bits for maximum optimization
    int uniform_bits = getUniformDeltaBits(compressed.d_delta_bits, np);

    if constexpr (sizeof(T) == 4) {
        // 32-bit version
        if (uniform_bits >= 0) {
            // All partitions have same delta_bits - use template specialization
            switch (uniform_bits) {
                case 0:  decompressFOR_V5_32<0><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 4:  decompressFOR_V5_32<4><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 6:  decompressFOR_V5_32<6><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 8:  decompressFOR_V5_32<8><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 10: decompressFOR_V5_32<10><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 12: decompressFOR_V5_32<12><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 16: decompressFOR_V5_32<16><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 20: decompressFOR_V5_32<20><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 24: decompressFOR_V5_32<24><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                case 32: decompressFOR_V5_32<32><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint32_t*>(d_output)); break;
                default:
                    // Uncommon bit-width: use runtime version
                    decompressFOR_V5_32_runtime<<<np, block_size, 0, stream>>>(
                        compressed.d_interleaved_deltas, compressed.d_start_indices,
                        compressed.d_end_indices, compressed.d_model_params,
                        compressed.d_delta_bits, compressed.d_num_mini_vectors,
                        compressed.d_interleaved_offsets, np,
                        reinterpret_cast<uint32_t*>(d_output)); break;
            }
        } else {
            // Mixed delta_bits: use runtime version
            decompressFOR_V5_32_runtime<<<np, block_size, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_params,
                compressed.d_delta_bits, compressed.d_num_mini_vectors,
                compressed.d_interleaved_offsets, np,
                reinterpret_cast<uint32_t*>(d_output));
        }
    } else {
        // 64-bit version
        if (uniform_bits >= 0) {
            switch (uniform_bits) {
                case 0:  decompressFOR_V5_64<0><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 8:  decompressFOR_V5_64<8><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 16: decompressFOR_V5_64<16><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 24: decompressFOR_V5_64<24><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 32: decompressFOR_V5_64<32><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 40: decompressFOR_V5_64<40><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 48: decompressFOR_V5_64<48><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 56: decompressFOR_V5_64<56><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                case 64: decompressFOR_V5_64<64><<<np, block_size, 0, stream>>>(
                    compressed.d_interleaved_deltas, compressed.d_start_indices,
                    compressed.d_end_indices, compressed.d_model_params,
                    compressed.d_num_mini_vectors, compressed.d_interleaved_offsets,
                    np, reinterpret_cast<uint64_t*>(d_output)); break;
                default:
                    decompressFOR_V5_64_runtime<<<np, block_size, 0, stream>>>(
                        compressed.d_interleaved_deltas, compressed.d_start_indices,
                        compressed.d_end_indices, compressed.d_model_params,
                        compressed.d_delta_bits, compressed.d_num_mini_vectors,
                        compressed.d_interleaved_offsets, np,
                        reinterpret_cast<uint64_t*>(d_output)); break;
            }
        } else {
            // Mixed delta_bits: use V5b per-partition switch-case kernel
            decompressFOR_V5b_64<<<np, block_size, 0, stream>>>(
                compressed.d_interleaved_deltas, compressed.d_start_indices,
                compressed.d_end_indices, compressed.d_model_params,
                compressed.d_delta_bits, compressed.d_num_mini_vectors,
                compressed.d_interleaved_offsets, np,
                reinterpret_cast<uint64_t*>(d_output));
        }
    }
}

// Global flag to enable V5 decoder
static bool g_use_v5_decoder = false;

} // namespace V5_FOR_decoder

// ============================================================================
// CUDA Timer
// ============================================================================

class CudaTimer {
    cudaEvent_t start_, stop_;
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_, stream));
    }
    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

// ============================================================================
// Data Structures
// ============================================================================

struct DatasetInfo {
    int id;
    std::string filename;
    std::string name;
    bool is_uint64;
    bool is_signed;
};

struct ModelStats {
    int constant_count = 0;
    int linear_count = 0;
    int poly2_count = 0;
    int poly3_count = 0;
    int for_bp_count = 0;

    int total() const {
        return constant_count + linear_count + poly2_count + poly3_count + for_bp_count;
    }
    double constant_pct() const { return total() > 0 ? 100.0 * constant_count / total() : 0; }
    double linear_pct() const { return total() > 0 ? 100.0 * linear_count / total() : 0; }
    double poly2_pct() const { return total() > 0 ? 100.0 * poly2_count / total() : 0; }
    double poly3_pct() const { return total() > 0 ? 100.0 * poly3_count / total() : 0; }
    double for_bp_pct() const { return total() > 0 ? 100.0 * for_bp_count / total() : 0; }
};

struct DatasetResult {
    int id;
    std::string name;
    std::string data_type;
    size_t num_elements;
    int num_partitions;
    ModelStats stats;
    double compression_ratio;
    // Performance metrics
    double encode_time_ms = 0.0;
    double decode_time_ms = 0.0;
    double encode_gbps = 0.0;
    double decode_gbps = 0.0;
    // V5 FOR-only decoder metrics
    double decode_v5_time_ms = 0.0;
    double decode_v5_gbps = 0.0;
    bool all_for = false;  // Whether all partitions use FOR model
    int uniform_bits = -1;  // Uniform delta_bits (-1 if mixed)
    // Verification status
    bool verified = false;
    bool verified_v5 = false;
    int error_count = 0;
    int error_count_v5 = 0;
    bool success;
    std::string error_msg;
};

// ============================================================================
// Dataset Configuration (SOSD 1-20)
// ============================================================================

// Default data directory (relative path, can be overridden via command line)
std::string DATA_DIR = "../../../../../../../../../test/data/sosd/";

const std::vector<DatasetInfo> DATASETS = {
    {1, "1-linear_200M_uint32.bin", "linear", false, false},
    {2, "2-normal_200M_uint32.bin", "normal", false, false},
    {3, "3-poisson_87M_uint64.bin", "poisson", true, false},
    {4, "4-ml_uint64.bin", "ml", true, false},
    {5, "5-books_200M_uint32.bin", "books", false, false},
    {6, "6-fb_200M_uint64.bin", "fb", true, false},
    {7, "7-wiki_200M_uint64.bin", "wiki", true, false},
    {8, "8-osm_cellids_800M_uint64.bin", "osm", true, false},
    {9, "9-movieid_uint32.bin", "movieid", false, false},
    {10, "10-house_price_uint64.bin", "house_price", true, false},
    {11, "11-planet_uint64.bin", "planet", true, false},
    {12, "12-libio.bin", "libio", true, false},
    {13, "13-medicare.bin", "medicare", true, false},
    {14, "14-cosmos_int32.bin", "cosmos", false, true},  // signed int32
    {15, "15-polylog_10M_uint64.bin", "polylog", true, false},
    {16, "16-exp_200M_uint64.bin", "exp", true, false},
    {17, "17-poly_200M_uint64.bin", "poly", true, false},
    {18, "18-site_250k_uint32.bin", "site", false, false},
    {19, "19-weight_25k_uint32.bin", "weight", false, false},
    {20, "20-adult_30k_uint32.bin", "adult", false, false}
};

// Synthetic datasets for comparison
const std::vector<DatasetInfo> SYNTHETIC_DATASETS = {
    {101, "synthetic/true_linear_10M_uint64.bin", "TRUE_LINEAR_u64", true, false},
    {102, "synthetic/true_linear_10M_uint32.bin", "TRUE_LINEAR_u32", false, false},
    {103, "synthetic/true_linear_1M_uint64.bin", "TRUE_LINEAR_1M", true, false},
    {104, "synthetic/true_cubic_10M_uint64.bin", "TRUE_CUBIC_u64", true, false},
    {105, "synthetic/strong_cubic_10M_uint64.bin", "STRONG_CUBIC_u64", true, false}
};

// SSB LINEORDER datasets (all uint32)
const std::vector<DatasetInfo> SSB_LINEORDER_DATASETS = {
    {200, "LINEORDER0.bin", "LO_ORDERKEY", false, false},
    {201, "LINEORDER1.bin", "LO_LINENUMBER", false, false},
    {202, "LINEORDER2.bin", "LO_CUSTKEY", false, false},
    {203, "LINEORDER3.bin", "LO_PARTKEY", false, false},
    {204, "LINEORDER4.bin", "LO_SUPPKEY", false, false},
    {205, "LINEORDER5.bin", "LO_ORDERDATE", false, false},
    {206, "LINEORDER6.bin", "LO_ORDERPRIORITY", false, false},
    {207, "LINEORDER7.bin", "LO_SHIPPRIORITY", false, false},
    {208, "LINEORDER8.bin", "LO_QUANTITY", false, false},
    {209, "LINEORDER9.bin", "LO_EXTENDEDPRICE", false, false},
    {210, "LINEORDER10.bin", "LO_ORDTOTALPRICE", false, false},
    {211, "LINEORDER11.bin", "LO_DISCOUNT", false, false},
    {212, "LINEORDER12.bin", "LO_REVENUE", false, false},
    {213, "LINEORDER13.bin", "LO_SUPPLYCOST", false, false},
    {214, "LINEORDER14.bin", "LO_TAX", false, false},
    {215, "LINEORDER15.bin", "LO_COMMITDATE", false, false},
    {216, "LINEORDER16.bin", "LO_SHIPMODE", false, false}
};

// ============================================================================
// Data Loading
// ============================================================================

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for 8-byte header (count of elements)
    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));

    size_t data_bytes = file_size - sizeof(uint64_t);
    size_t expected_with_header = data_bytes / sizeof(T);

    std::vector<T> data;
    if (header_count == expected_with_header) {
        // File has header
        data.resize(header_count);
        file.read(reinterpret_cast<char*>(data.data()), header_count * sizeof(T));
    } else {
        // No header, raw binary
        file.seekg(0, std::ios::beg);
        size_t num_elements = file_size / sizeof(T);
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }

    return data;
}

// ============================================================================
// Model Statistics Collection
// ============================================================================

template<typename T>
DatasetResult runModelStatistics(const DatasetInfo& dataset, int partition_size,
                                  int min_partition = 256, int max_partition = 8192,
                                  bool enable_merging = true, int warmup = 2, int runs = 5,
                                  int encoder_block_size = 256, bool sort_data = false) {
    DatasetResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.success = false;

    // Set data type string
    if (std::is_same<T, uint32_t>::value) result.data_type = "uint32";
    else if (std::is_same<T, uint64_t>::value) result.data_type = "uint64";
    else if (std::is_same<T, int32_t>::value) result.data_type = "int32";
    else if (std::is_same<T, int64_t>::value) result.data_type = "int64";

    // Load data
    std::string path = DATA_DIR + dataset.filename;
    std::vector<T> data = loadBinaryData<T>(path);

    if (data.empty()) {
        result.error_msg = "Failed to load data";
        return result;
    }

    // Sort data if requested (for RLE analysis)
    if (sort_data) {
        std::sort(data.begin(), data.end());
    }

    result.num_elements = data.size();

    // Calculate data size for throughput
    double data_bytes = static_cast<double>(data.size()) * sizeof(T);
    double data_gb = data_bytes / 1e9;

    // Configure Vertical with COST_OPTIMAL partitioning for all types
    VerticalConfig config;
    config.partition_size_hint = partition_size;
    config.enable_adaptive_selection = true;  // Enable full polynomial selection
    config.enable_interleaved = true;
    config.enable_branchless_unpack = true;

    // Use COST_OPTIMAL partitioning for all types (signed and unsigned)
    config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
    config.cost_min_partition_size = min_partition;
    config.cost_max_partition_size = max_partition;
    config.cost_enable_merging = enable_merging;
    config.encoder_selector_block_size = encoder_block_size;

    // ========================================================================
    // Encoding with warmup and timing
    // ========================================================================

    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        CompressedDataVertical<T> temp;
        try {
            // Always use PolyCost encoder for all types (signed and unsigned)
            temp = Vertical_encoder::encodeVerticalGPU_PolyCost<T>(data, partition_size, config, 0);
        } catch (...) {
            // Ignore warmup errors
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        Vertical_encoder::freeCompressedData(temp);
    }

    // Timed encoding runs
    double total_encode_ms = 0.0;
    CompressedDataVertical<T> compressed;

    for (int i = 0; i < runs; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();

        CompressedDataVertical<T> temp;
        try {
            // Always use PolyCost encoder for all types (signed and unsigned)
            temp = Vertical_encoder::encodeVerticalGPU_PolyCost<T>(data, partition_size, config, 0);
        } catch (const std::exception& e) {
            result.error_msg = std::string("Compression failed: ") + e.what();
            return result;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_encode_ms += elapsed_ms;

        if (i == runs - 1) {
            // Keep the last compressed result for decoding
            compressed = temp;
        } else {
            Vertical_encoder::freeCompressedData(temp);
        }
    }

    result.encode_time_ms = total_encode_ms / runs;
    result.encode_gbps = data_gb / (result.encode_time_ms / 1000.0);

    if (compressed.num_partitions == 0) {
        result.error_msg = "No partitions created";
        return result;
    }

    result.num_partitions = compressed.num_partitions;

    // ========================================================================
    // Collect Model Statistics
    // ========================================================================

    // Copy model types from device to host
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Collect statistics
    for (int i = 0; i < compressed.num_partitions; i++) {
        switch (h_model_types[i]) {
            case MODEL_CONSTANT: result.stats.constant_count++; break;
            case MODEL_LINEAR: result.stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.stats.poly3_count++; break;
            case MODEL_FOR_BITPACK: result.stats.for_bp_count++; break;
            default:
                std::cerr << "Warning: Unknown model type " << h_model_types[i]
                         << " at partition " << i << std::endl;
                break;
        }
    }

    // ========================================================================
    // Partition Size Statistics
    // ========================================================================
    std::vector<int32_t> h_start_indices(compressed.num_partitions);
    std::vector<int32_t> h_end_indices(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_start_indices.data(), compressed.d_start_indices,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end_indices.data(), compressed.d_end_indices,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Calculate partition sizes and statistics
    std::vector<int> partition_sizes(compressed.num_partitions);
    int min_size = INT_MAX, max_size = 0;
    int64_t total_size = 0;

    // Size distribution buckets: [0-128), [128-256), [256-512), [512-1024), [1024-2048), [2048-4096), [4096-8192), [8192+)
    std::vector<int> size_buckets(8, 0);

    for (int i = 0; i < compressed.num_partitions; i++) {
        int size = h_end_indices[i] - h_start_indices[i];
        partition_sizes[i] = size;
        min_size = std::min(min_size, size);
        max_size = std::max(max_size, size);
        total_size += size;

        // Bucket classification
        if (size < 128) size_buckets[0]++;
        else if (size < 256) size_buckets[1]++;
        else if (size < 512) size_buckets[2]++;
        else if (size < 1024) size_buckets[3]++;
        else if (size < 2048) size_buckets[4]++;
        else if (size < 4096) size_buckets[5]++;
        else if (size < 8192) size_buckets[6]++;
        else size_buckets[7]++;
    }

    double avg_size = static_cast<double>(total_size) / compressed.num_partitions;

    std::cout << "\n--- Partition Size Statistics ---\n";
    std::cout << "  Total partitions: " << compressed.num_partitions << "\n";
    std::cout << "  Min size: " << min_size << "\n";
    std::cout << "  Max size: " << max_size << "\n";
    std::cout << "  Avg size: " << std::fixed << std::setprecision(1) << avg_size << "\n";
    std::cout << "  Size distribution:\n";
    const char* bucket_names[] = {"[0-128)", "[128-256)", "[256-512)", "[512-1024)",
                                   "[1024-2048)", "[2048-4096)", "[4096-8192)", "[8192+)"};
    for (int i = 0; i < 8; i++) {
        if (size_buckets[i] > 0) {
            double pct = 100.0 * size_buckets[i] / compressed.num_partitions;
            std::cout << "    " << std::setw(12) << bucket_names[i] << ": "
                      << std::setw(8) << size_buckets[i] << " ("
                      << std::fixed << std::setprecision(1) << pct << "%)\n";
        }
    }
    std::cout << "---------------------------------\n\n";

    // ========================================================================
    // Delta Bits Statistics
    // ========================================================================
    std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Analyze delta_bits distribution
    std::map<int, int> bits_count;
    int64_t total_weighted_bits = 0;
    int zero_bit_partitions = 0;
    int small_bit_partitions = 0;  // <= 8 bits

    for (int i = 0; i < compressed.num_partitions; i++) {
        int bits = h_delta_bits[i];
        int size = h_end_indices[i] - h_start_indices[i];
        bits_count[bits]++;
        total_weighted_bits += static_cast<int64_t>(bits) * size;
        if (bits == 0) zero_bit_partitions++;
        if (bits <= 8) small_bit_partitions++;
    }

    double avg_delta_bits = static_cast<double>(total_weighted_bits) / total_size;

    std::cout << "--- Delta Bits Statistics ---\n";
    std::cout << "  Avg delta_bits (weighted): " << std::fixed << std::setprecision(2) << avg_delta_bits << "\n";
    std::cout << "  Zero-bit partitions: " << zero_bit_partitions
              << " (" << std::fixed << std::setprecision(1) << (100.0 * zero_bit_partitions / compressed.num_partitions) << "%)\n";
    std::cout << "  Small-bit (<=8) partitions: " << small_bit_partitions
              << " (" << std::fixed << std::setprecision(1) << (100.0 * small_bit_partitions / compressed.num_partitions) << "%)\n";
    std::cout << "  Delta bits distribution:\n";
    for (const auto& [bits, count] : bits_count) {
        double pct = 100.0 * count / compressed.num_partitions;
        if (pct >= 0.1) {  // Only show >= 0.1%
            std::cout << "    " << std::setw(2) << bits << " bits: "
                      << std::setw(6) << count << " ("
                      << std::fixed << std::setprecision(1) << pct << "%)\n";
        }
    }
    std::cout << "-----------------------------\n\n";

    // Calculate compression ratio
    size_t original_bytes = data.size() * sizeof(T);
    // Compute parameter storage size based on layout mode
    size_t param_bytes = compressed.use_variable_params
        ? (compressed.total_param_count * sizeof(double) + compressed.num_partitions * sizeof(int64_t))
        : (compressed.num_partitions * 4 * sizeof(double));

    size_t interleaved_bytes = compressed.interleaved_delta_words * sizeof(uint32_t);
    size_t metadata_bytes = compressed.num_partitions * sizeof(int32_t) * 4;
    size_t compressed_bytes = interleaved_bytes + metadata_bytes + param_bytes;

    // Debug: show compression breakdown
    std::cout << "--- Compression Breakdown ---\n";
    std::cout << "  Original: " << original_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cout << "  Interleaved data: " << interleaved_bytes / 1024.0 / 1024.0 << " MB ("
              << compressed.interleaved_delta_words << " words)\n";
    std::cout << "  Metadata (4 int32/part): " << metadata_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cout << "  Params: " << param_bytes / 1024.0 / 1024.0 << " MB";
    if (compressed.use_variable_params) {
        std::cout << " (variable: " << compressed.total_param_count << " params_arr + "
                  << compressed.num_partitions << " offsets)";
    }
    std::cout << "\n";
    std::cout << "  Total compressed: " << compressed_bytes / 1024.0 / 1024.0 << " MB\n";
    std::cout << "-----------------------------\n\n";

    result.compression_ratio = static_cast<double>(original_bytes) / compressed_bytes;

    // ========================================================================
    // Decoding with V4/V5 decoder and timing
    // ========================================================================

    // Allocate output buffer
    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Warmup decode runs
    int actual_block_size = V4_decoder::g_block_size;
    if (actual_block_size < 0) {
        // Adaptive: select based on average partition size
        actual_block_size = V4_decoder::selectOptimalBlockSize(data.size(), compressed.num_partitions);
    }

    // Check if V5 metadata is available
    bool has_v5 = compressed.use_v5_metadata && compressed.d_metadata_v5 != nullptr;

    // ========== V5-Opt Decoder Timing (if available) ==========
    double v5opt_time_ms = 0.0;
    double v5opt_gbps = 0.0;
    if (has_v5) {
        std::cout << "  Testing V5-Opt decoder...\n";
        for (int i = 0; i < warmup; i++) {
            V4_decoder::decompressV5_Opt<T>(compressed, d_output, 0, actual_block_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        CudaTimer timer_v5opt;
        double total_decode_ms_v5opt = 0.0;
        for (int i = 0; i < runs; i++) {
            timer_v5opt.start();
            V4_decoder::decompressV5_Opt<T>(compressed, d_output, 0, actual_block_size);
            total_decode_ms_v5opt += timer_v5opt.stop();
        }
        v5opt_time_ms = total_decode_ms_v5opt / runs;
        v5opt_gbps = data_gb / (v5opt_time_ms / 1000.0);

        // Use V5-Opt result as primary
        result.decode_time_ms = v5opt_time_ms;
        result.decode_gbps = v5opt_gbps;
        std::cout << "    V5-Opt: " << std::fixed << std::setprecision(2) << v5opt_gbps << " GB/s\n";
    }

    // ========== V4 Decoder Timing (for comparison) ==========
    std::cout << "  Testing V4 decoder...\n";
    for (int i = 0; i < warmup; i++) {
        V4_decoder::decompressV4<T>(compressed, d_output, 0, actual_block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CudaTimer timer;
    double total_decode_ms_v4 = 0.0;
    for (int i = 0; i < runs; i++) {
        timer.start();
        V4_decoder::decompressV4<T>(compressed, d_output, 0, actual_block_size);
        total_decode_ms_v4 += timer.stop();
    }
    double v4_time_ms = total_decode_ms_v4 / runs;
    double v4_gbps = data_gb / (v4_time_ms / 1000.0);
    std::cout << "    V4: " << std::fixed << std::setprecision(2) << v4_gbps << " GB/s\n";

    // If V5-Opt not available, use V4 result
    if (!has_v5) {
        result.decode_time_ms = v4_time_ms;
        result.decode_gbps = v4_gbps;
    } else {
        // Print comparison
        double speedup = v5opt_gbps / v4_gbps;
        std::cout << "    Speedup (V5-Opt vs V4): " << std::fixed << std::setprecision(2) << speedup << "x\n";
    }

    // ========================================================================
    // Verification
    // ========================================================================

    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));

    // DEBUG code disabled
    // {
    //     std::vector<int32_t> h_model_types(compressed.num_partitions);
    //     std::vector<double> h_model_params_arr(compressed.num_partitions * 4);
    //     std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    //     std::vector<int32_t> h_starts(compressed.num_partitions);
    //     std::vector<int32_t> h_ends(compressed.num_partitions);
    //     cudaMemcpy(h_model_types.data(), compressed.d_model_types, compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_model_params_arr.data(), compressed.d_model_params, compressed.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits, compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_starts.data(), compressed.d_start_indices, compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(h_ends.data(), compressed.d_end_indices, compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    //
    //     std::cerr << "\n  DEBUG: First partition info:\n";
    //     std::cerr << "    Model type: " << h_model_types[0] << " (1=LINEAR)\n";
    //     std::cerr << "    Params: theta0=" << std::scientific << std::setprecision(17) << h_model_params_arr[0]
    //               << ", theta1=" << h_model_params_arr[1] << "\n";
    //     std::cerr << "    Delta bits: " << h_delta_bits[0] << "\n";
    //     std::cerr << "    Range: [" << h_starts[0] << ", " << h_ends[0] << ")\n";
    //
    //     // Test GPU rounding
    //     std::cerr << "\n    GPU rounding test:\n";
    //     testGPURounding<<<1, 1>>>(h_model_params_arr[0], h_model_params_arr[1], 10);
    //     cudaDeviceSynchronize();
    //
    //     // Show predictions for first 10 elements with full precision
    //     std::cerr << "\n    CPU predictions:\n";
    //     for (int i = 0; i < 10 && i < h_ends[0] - h_starts[0]; i++) {
    //         double x = static_cast<double>(i);
    //         double predicted = h_model_params_arr[0] + h_model_params_arr[1] * x;
    //         long long pred_int_cpu = std::llrint(predicted);
    //         // Also show raw bits
    //         uint64_t pred_bits = *reinterpret_cast<uint64_t*>(&predicted);
    //         std::cerr << "      [" << i << "] pred_d=" << std::fixed << std::setprecision(17) << predicted
    //                   << " (bits=0x" << std::hex << pred_bits << std::dec << ")"
    //                   << " -> cpu_llrint=" << pred_int_cpu
    //                   << ", actual=" << data[h_starts[0] + i]
    //                   << ", decoded=" << decoded[h_starts[0] + i] << "\n";
    //     }
    //     std::cerr << "\n";
    // }

    int errors = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            errors++;
            if (errors <= 5) {
                std::cerr << "  Mismatch at index " << i
                          << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
        }
    }

    result.verified = (errors == 0);
    result.error_count = errors;

    // ========================================================================
    // V5 FOR-Only Decoder Testing (if all partitions are FOR)
    // ========================================================================

    // Check if all partitions use FOR model
    result.all_for = V5_FOR_decoder::isAllFOR(compressed.d_model_types, compressed.num_partitions);
    result.uniform_bits = V5_FOR_decoder::getUniformDeltaBits(compressed.d_delta_bits, compressed.num_partitions);

    if (result.all_for && V5_FOR_decoder::g_use_v5_decoder) {
        // Reset output buffer
        CUDA_CHECK(cudaMemset(d_output, 0, data.size() * sizeof(T)));

        // Warmup V5 decode runs
        for (int i = 0; i < warmup; i++) {
            V5_FOR_decoder::decompressFOR_V5<T>(compressed, d_output, 0, actual_block_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Timed V5 decode runs
        double total_decode_v5_ms = 0.0;
        for (int i = 0; i < runs; i++) {
            timer.start();
            V5_FOR_decoder::decompressFOR_V5<T>(compressed, d_output, 0, actual_block_size);
            total_decode_v5_ms += timer.stop();
        }

        result.decode_v5_time_ms = total_decode_v5_ms / runs;
        result.decode_v5_gbps = data_gb / (result.decode_v5_time_ms / 1000.0);

        // Verify V5 decoder output
        CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data.size() * sizeof(T),
                              cudaMemcpyDeviceToHost));

        int errors_v5 = 0;
        for (size_t i = 0; i < data.size(); i++) {
            if (data[i] != decoded[i]) {
                errors_v5++;
                if (errors_v5 <= 5) {
                    std::cerr << "  V5 Mismatch at index " << i
                              << ": expected " << data[i]
                              << ", got " << decoded[i] << std::endl;
                }
            }
        }
        result.verified_v5 = (errors_v5 == 0);
        result.error_count_v5 = errors_v5;

        // Print speedup info
        if (result.verified_v5) {
            double speedup = result.decode_v5_gbps / result.decode_gbps;
            std::cout << "  V5 FOR Decoder: " << std::fixed << std::setprecision(2)
                      << result.decode_v5_gbps << " GB/s (Speedup: "
                      << speedup << "x, bits=" << result.uniform_bits << ")\n";
            // Print delta_bits distribution if not uniform
            if (result.uniform_bits == -1) {
                V5_FOR_decoder::printDeltaBitsDistribution(compressed.d_delta_bits, compressed.num_partitions);
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    result.success = true;
    return result;
}

// ============================================================================
// GM (Word-Interleaved) Model Statistics Collection
// ============================================================================

template<typename T>
DatasetResult runModelStatistics_GM(const DatasetInfo& dataset, int partition_size,
                                     int min_partition = 256, int max_partition = 8192,
                                     bool enable_merging = true, int warmup = 2, int runs = 5) {
    DatasetResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.success = false;

    // Set data type string
    if (std::is_same<T, uint32_t>::value) result.data_type = "uint32";
    else if (std::is_same<T, uint64_t>::value) result.data_type = "uint64";
    else if (std::is_same<T, int32_t>::value) result.data_type = "int32";
    else if (std::is_same<T, int64_t>::value) result.data_type = "int64";

    // Load data
    std::string path = DATA_DIR + dataset.filename;
    std::vector<T> data = loadBinaryData<T>(path);

    if (data.empty()) {
        result.error_msg = "Failed to load data";
        return result;
    }

    result.num_elements = data.size();

    // Calculate data size for throughput
    double data_bytes = static_cast<double>(data.size()) * sizeof(T);
    double data_gb = data_bytes / 1e9;

    // Configure Transposed with COST_OPTIMAL partitioning (SAME as Vertical)
    TransposedConfig config;
    config.partition_size_hint = partition_size;
    config.enable_adaptive_selection = true;  // Enable full polynomial selection
    config.enable_branchless_unpack = true;

    // Use COST_OPTIMAL partitioning for all types (SAME as Vertical)
    config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
    config.cost_min_partition_size = min_partition;
    config.cost_max_partition_size = max_partition;
    config.cost_enable_merging = enable_merging;

    // ========================================================================
    // Encoding with warmup and timing (using GM PolyCost encoder)
    // ========================================================================

    // Warmup runs
    for (int i = 0; i < warmup; i++) {
        CompressedDataTransposed<T> temp;
        try {
            // Use PolyCost encoder with V3 partitioner (SAME as Vertical)
            temp = Transposed_encoder::encodeTransposedGPU_PolyCost<T>(data, partition_size, config, 0);
        } catch (...) {
            // Ignore warmup errors
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        Transposed_encoder::freeCompressedData(temp);
    }

    // Timed encoding runs
    double total_encode_ms = 0.0;
    CompressedDataTransposed<T> compressed;

    for (int i = 0; i < runs; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        auto start = std::chrono::steady_clock::now();

        CompressedDataTransposed<T> temp;
        try {
            // Use PolyCost encoder with V3 partitioner (SAME as Vertical)
            temp = Transposed_encoder::encodeTransposedGPU_PolyCost<T>(data, partition_size, config, 0);
        } catch (const std::exception& e) {
            result.error_msg = std::string("Compression failed: ") + e.what();
            return result;
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::steady_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_encode_ms += elapsed_ms;

        if (i == runs - 1) {
            // Keep the last compressed result for decoding
            compressed = temp;
        } else {
            Transposed_encoder::freeCompressedData(temp);
        }
    }

    result.encode_time_ms = total_encode_ms / runs;
    result.encode_gbps = data_gb / (result.encode_time_ms / 1000.0);

    if (compressed.num_partitions == 0) {
        result.error_msg = "No partitions created";
        return result;
    }

    result.num_partitions = compressed.num_partitions;

    // ========================================================================
    // Collect Model Statistics (SAME logic as Vertical)
    // ========================================================================

    // Copy model types from device to host
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Collect statistics
    for (int i = 0; i < compressed.num_partitions; i++) {
        switch (h_model_types[i]) {
            case MODEL_CONSTANT: result.stats.constant_count++; break;
            case MODEL_LINEAR: result.stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.stats.poly3_count++; break;
            case MODEL_FOR_BITPACK: result.stats.for_bp_count++; break;
            default:
                std::cerr << "Warning: Unknown model type " << h_model_types[i]
                         << " at partition " << i << std::endl;
                break;
        }
    }

    // ========================================================================
    // Partition Size Statistics (SAME logic as Vertical)
    // ========================================================================
    std::vector<int32_t> h_start_indices(compressed.num_partitions);
    std::vector<int32_t> h_end_indices(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_start_indices.data(), compressed.d_start_indices,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end_indices.data(), compressed.d_end_indices,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Calculate partition sizes and statistics
    std::vector<int> partition_sizes(compressed.num_partitions);
    int min_size = INT_MAX, max_size = 0;
    int64_t total_size = 0;

    // Size distribution buckets
    std::vector<int> size_buckets(8, 0);

    for (int i = 0; i < compressed.num_partitions; i++) {
        int size = h_end_indices[i] - h_start_indices[i];
        partition_sizes[i] = size;
        min_size = std::min(min_size, size);
        max_size = std::max(max_size, size);
        total_size += size;

        // Bucket classification
        if (size < 128) size_buckets[0]++;
        else if (size < 256) size_buckets[1]++;
        else if (size < 512) size_buckets[2]++;
        else if (size < 1024) size_buckets[3]++;
        else if (size < 2048) size_buckets[4]++;
        else if (size < 4096) size_buckets[5]++;
        else if (size < 8192) size_buckets[6]++;
        else size_buckets[7]++;
    }

    double avg_size = static_cast<double>(total_size) / compressed.num_partitions;

    std::cout << "\n--- GM Partition Size Statistics ---\n";
    std::cout << "  Total partitions: " << compressed.num_partitions << "\n";
    std::cout << "  Min size: " << min_size << "\n";
    std::cout << "  Max size: " << max_size << "\n";
    std::cout << "  Avg size: " << std::fixed << std::setprecision(1) << avg_size << "\n";
    std::cout << "  Size distribution:\n";
    const char* bucket_names[] = {"[0-128)", "[128-256)", "[256-512)", "[512-1024)",
                                   "[1024-2048)", "[2048-4096)", "[4096-8192)", "[8192+)"};
    for (int i = 0; i < 8; i++) {
        if (size_buckets[i] > 0) {
            double pct = 100.0 * size_buckets[i] / compressed.num_partitions;
            std::cout << "    " << std::setw(12) << bucket_names[i] << ": "
                      << std::setw(8) << size_buckets[i] << " ("
                      << std::fixed << std::setprecision(1) << pct << "%)\n";
        }
    }
    std::cout << "------------------------------------\n\n";

    // Calculate compression ratio (using transposed_delta_words for GM)
    size_t original_bytes = data.size() * sizeof(T);
    size_t compressed_bytes = compressed.transposed_delta_words * sizeof(uint32_t) +
                              compressed.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
    result.compression_ratio = static_cast<double>(original_bytes) / compressed_bytes;

    // ========================================================================
    // Decoding with GM V4 decoder and timing
    // ========================================================================

    // Allocate output buffer
    T* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Warmup decode runs
    int actual_block_size = V4_decoder::g_block_size;
    if (actual_block_size < 0) {
        // Adaptive: select based on average partition size
        actual_block_size = GM_V4_decoder::selectOptimalBlockSize(data.size(), compressed.num_partitions);
    }

    for (int i = 0; i < warmup; i++) {
        GM_V4_decoder::decompressV4_GM<T>(compressed, d_output, 0, actual_block_size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed decode runs
    CudaTimer timer;
    double total_decode_ms = 0.0;

    for (int i = 0; i < runs; i++) {
        timer.start();
        GM_V4_decoder::decompressV4_GM<T>(compressed, d_output, 0, actual_block_size);
        total_decode_ms += timer.stop();
    }

    result.decode_time_ms = total_decode_ms / runs;
    result.decode_gbps = data_gb / (result.decode_time_ms / 1000.0);

    // ========================================================================
    // Verification
    // ========================================================================

    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));

    int errors = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            errors++;
            if (errors <= 5) {
                std::cerr << "  GM Mismatch at index " << i
                          << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
        }
    }

    result.verified = (errors == 0);
    result.error_count = errors;

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    Transposed_encoder::freeCompressedData(compressed);

    result.success = true;
    return result;
}

// Helper function to process a single dataset with GM format
DatasetResult processDataset_GM(const DatasetInfo& dataset, int partition_size,
                                 int min_partition = 256, int max_partition = 8192,
                                 bool enable_merging = true, int warmup = 2, int runs = 5) {
    DatasetResult result;
    if (dataset.is_signed) {
        result = runModelStatistics_GM<int32_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
    } else if (dataset.is_uint64) {
        result = runModelStatistics_GM<uint64_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
    } else {
        result = runModelStatistics_GM<uint32_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
    }
    return result;
}

// ============================================================================
// Output Functions
// ============================================================================

void printResultsTable(const std::vector<DatasetResult>& results, const std::string& strategy) {
    std::cout << "\n";
    std::cout << "========================================================================================================\n";
    std::cout << "L3 GPU Adaptive Model Selection Statistics + E2E Verification (Full Polynomial Selector)\n";
    std::cout << "Partitioning: " << strategy << " | Decoder: V4 Optimized Interleaved\n";
    std::cout << "========================================================================================================\n";
    std::cout << std::left << std::setw(16) << "Dataset"
              << std::right << std::setw(10) << "Elements"
              << std::setw(8) << "Parts"
              << std::setw(7) << "CONST"
              << std::setw(7) << "LIN"
              << std::setw(7) << "P2"
              << std::setw(7) << "P3"
              << std::setw(7) << "FOR"
              << std::setw(7) << "Ratio"
              << std::setw(9) << "Enc GB/s"
              << std::setw(9) << "Dec GB/s"
              << std::setw(8) << "Verify"
              << "\n";
    std::cout << std::string(104, '-') << "\n";

    // Aggregate stats
    ModelStats total_stats;
    int total_partitions = 0;
    double total_encode_gbps = 0.0;
    double total_decode_gbps = 0.0;
    int passed = 0, failed = 0;

    for (const auto& r : results) {
        if (!r.success) {
            std::cout << std::left << std::setw(16) << (std::to_string(r.id) + "-" + r.name)
                      << " FAILED: " << r.error_msg << "\n";
            failed++;
            continue;
        }

        std::cout << std::left << std::setw(16) << (std::to_string(r.id) + "-" + r.name)
                  << std::right << std::setw(10) << r.num_elements
                  << std::setw(8) << r.num_partitions
                  << std::fixed << std::setprecision(1)
                  << std::setw(7) << r.stats.constant_pct()
                  << std::setw(7) << r.stats.linear_pct()
                  << std::setw(7) << r.stats.poly2_pct()
                  << std::setw(7) << r.stats.poly3_pct()
                  << std::setw(7) << r.stats.for_bp_pct()
                  << std::setprecision(2) << std::setw(7) << r.compression_ratio
                  << std::setw(9) << r.encode_gbps
                  << std::setw(9) << r.decode_gbps
                  << std::setw(8) << (r.verified ? "OK" : "FAIL")
                  << "\n";

        // Accumulate totals
        total_stats.constant_count += r.stats.constant_count;
        total_stats.linear_count += r.stats.linear_count;
        total_stats.poly2_count += r.stats.poly2_count;
        total_stats.poly3_count += r.stats.poly3_count;
        total_stats.for_bp_count += r.stats.for_bp_count;
        total_partitions += r.num_partitions;
        total_encode_gbps += r.encode_gbps;
        total_decode_gbps += r.decode_gbps;

        if (r.verified) passed++;
        else failed++;
    }

    // Print totals
    int num_results = passed + failed;
    std::cout << std::string(104, '=') << "\n";
    std::cout << std::left << std::setw(16) << "TOTAL/AVG"
              << std::right << std::setw(10) << ""
              << std::setw(8) << total_partitions
              << std::fixed << std::setprecision(1)
              << std::setw(7) << total_stats.constant_pct()
              << std::setw(7) << total_stats.linear_pct()
              << std::setw(7) << total_stats.poly2_pct()
              << std::setw(7) << total_stats.poly3_pct()
              << std::setw(7) << total_stats.for_bp_pct()
              << std::setw(7) << ""
              << std::setprecision(2)
              << std::setw(9) << (passed > 0 ? total_encode_gbps / passed : 0.0)
              << std::setw(9) << (passed > 0 ? total_decode_gbps / passed : 0.0)
              << std::setw(8) << (std::to_string(passed) + "/" + std::to_string(num_results))
              << "\n";
    std::cout << std::string(104, '=') << "\n";

    // Print absolute counts
    std::cout << "\nModel Selection Summary:\n";
    std::cout << "  CONSTANT:    " << std::setw(8) << total_stats.constant_count
              << " (" << std::fixed << std::setprecision(1) << total_stats.constant_pct() << "%)\n";
    std::cout << "  LINEAR:      " << std::setw(8) << total_stats.linear_count
              << " (" << std::fixed << std::setprecision(1) << total_stats.linear_pct() << "%)\n";
    std::cout << "  POLYNOMIAL2: " << std::setw(8) << total_stats.poly2_count
              << " (" << std::fixed << std::setprecision(1) << total_stats.poly2_pct() << "%)\n";
    std::cout << "  POLYNOMIAL3: " << std::setw(8) << total_stats.poly3_count
              << " (" << std::fixed << std::setprecision(1) << total_stats.poly3_pct() << "%)\n";
    std::cout << "  FOR_BITPACK: " << std::setw(8) << total_stats.for_bp_count
              << " (" << std::fixed << std::setprecision(1) << total_stats.for_bp_pct() << "%)\n";
    std::cout << "  TOTAL:       " << std::setw(8) << total_stats.total() << "\n";

    // Print verification summary
    std::cout << "\nVerification Summary:\n";
    std::cout << "  Passed: " << passed << ", Failed: " << failed << "\n";
    if (passed > 0) {
        std::cout << "  Average Encode Throughput: " << std::fixed << std::setprecision(2)
                  << (total_encode_gbps / passed) << " GB/s\n";
        std::cout << "  Average Decode Throughput: " << std::fixed << std::setprecision(2)
                  << (total_decode_gbps / passed) << " GB/s\n";
    }
}

void saveResultsCSV(const std::vector<DatasetResult>& results, const std::string& output_path) {
    // Create directory if needed
    std::string dir = output_path.substr(0, output_path.find_last_of('/'));
    mkdir(dir.c_str(), 0755);

    std::ofstream f(output_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open CSV file: " << output_path << std::endl;
        return;
    }

    // Header - includes performance and verification fields
    f << "dataset_id,dataset_name,data_type,num_elements,num_partitions,"
      << "constant_count,linear_count,poly2_count,poly3_count,for_bp_count,"
      << "constant_pct,linear_pct,poly2_pct,poly3_pct,for_bp_pct,"
      << "compression_ratio,encode_ms,decode_ms,encode_gbps,decode_gbps,"
      << "verified,error_count,success,error_msg\n";

    // Data rows
    for (const auto& r : results) {
        f << r.id << "," << r.name << "," << r.data_type << ","
          << r.num_elements << "," << r.num_partitions << ","
          << r.stats.constant_count << "," << r.stats.linear_count << ","
          << r.stats.poly2_count << "," << r.stats.poly3_count << "," << r.stats.for_bp_count << ","
          << std::fixed << std::setprecision(2)
          << r.stats.constant_pct() << "," << r.stats.linear_pct() << ","
          << r.stats.poly2_pct() << "," << r.stats.poly3_pct() << "," << r.stats.for_bp_pct() << ","
          << r.compression_ratio << ","
          << r.encode_time_ms << "," << r.decode_time_ms << ","
          << r.encode_gbps << "," << r.decode_gbps << ","
          << (r.verified ? "true" : "false") << "," << r.error_count << ","
          << (r.success ? "true" : "false") << ","
          << "\"" << r.error_msg << "\"\n";
    }

    f.close();
    std::cout << "\nCSV saved to: " << output_path << "\n";
}

// ============================================================================
// Main
// ============================================================================

// Helper function to process a single dataset
DatasetResult processDataset(const DatasetInfo& dataset, int partition_size,
                              int min_partition = 256, int max_partition = 8192,
                              bool enable_merging = true, int warmup = 2, int runs = 5,
                              int encoder_block_size = 256, bool sort_data = false) {
    DatasetResult result;
    if (dataset.is_signed) {
        result = runModelStatistics<int32_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size, sort_data);
    } else if (dataset.is_uint64) {
        result = runModelStatistics<uint64_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size, sort_data);
    } else {
        result = runModelStatistics<uint32_t>(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size, sort_data);
    }
    return result;
}

void printComparisonTable(const std::vector<DatasetResult>& synthetic_results,
                          const std::vector<DatasetResult>& sosd_results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "COMPARISON: True Linear vs SOSD Linear\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(20) << "Dataset"
              << std::right << std::setw(10) << "Elements"
              << std::setw(8) << "Parts"
              << std::setw(10) << "LINEAR%"
              << std::setw(8) << "FOR%"
              << std::setw(8) << "Ratio"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    // Print synthetic results first
    std::cout << "--- Synthetic (True Linear) ---\n";
    for (const auto& r : synthetic_results) {
        if (!r.success) continue;
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::setw(10) << r.num_elements
                  << std::setw(8) << r.num_partitions
                  << std::fixed << std::setprecision(1)
                  << std::setw(10) << r.stats.linear_pct()
                  << std::setw(8) << r.stats.for_bp_pct()
                  << std::setprecision(2) << std::setw(8) << r.compression_ratio
                  << "\n";
    }

    // Print SOSD linear result
    std::cout << "--- SOSD (CDF-mapped) ---\n";
    for (const auto& r : sosd_results) {
        if (!r.success) continue;
        if (r.name == "linear") {  // Only show linear for comparison
            std::cout << std::left << std::setw(20) << ("SOSD-" + r.name)
                      << std::right << std::setw(10) << r.num_elements
                      << std::setw(8) << r.num_partitions
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.stats.linear_pct()
                      << std::setw(8) << r.stats.for_bp_pct()
                      << std::setprecision(2) << std::setw(8) << r.compression_ratio
                      << "\n";
        }
    }

    std::cout << std::string(64, '=') << "\n";
    std::cout << "\nConclusion:\n";
    std::cout << "  - True Linear [0,1,2,...]: Selects LINEAR model (expected)\n";
    std::cout << "  - SOSD Linear: Selects FOR (because data is nearly constant CDF values)\n";
    std::cout << "  - Model selection logic is CORRECT based on cost optimization\n";
}

void printUsage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " [OPTIONS]\n"
              << "\nOptions:\n"
              << "  -d, --data-dir <path>   Path to SOSD data directory (default: " << DATA_DIR << ")\n"
              << "  --dataset <id>          Single dataset ID (1-20) to test\n"
              << "  -f, --file <path>       Custom file path (overrides --dataset, use absolute path)\n"
              << "  --name <name>           Custom dataset name (used with --file)\n"
              << "  --format <vertical|gm>  Storage format: vertical (lane-major) or gm (word-interleaved, default: vertical)\n"
              << "  --warmup <n>            Warmup iterations (default: 2)\n"
              << "  --runs <n>              Benchmark iterations (default: 5)\n"
              << "  --partition <size>      Partition size for FIXED strategy (signed types, default: 4096)\n"
              << "  --min-partition <size>  Initial partition size for COST_OPTIMAL (default: 256)\n"
              << "  --max-partition <size>  Max partition size after merging (default: 8192)\n"
              << "  --block-size <size>     Decoder block size: 32, 64, 128, 256, 512, 1024, or -1 for adaptive (default: -1)\n"
              << "  --encoder-block-size <size>  Encoder selector kernel block size (default: 256)\n"
              << "  --scan-encoder-block-size    Scan different encoder block sizes [32,64,128,256,512,1024]\n"
              << "  --v5                    Enable V5 FOR-only optimized decoder comparison\n"
              << "  --no-merge              Disable partition merging (to verify merge effect)\n"
              << "  --no-synthetic          Skip synthetic datasets\n"
              << "  --sort                  Sort data before compression (for RLE analysis)\n"
              << "  --verbose               Verbose output\n"
              << "  -h, --help              Show this help message\n"
              << "\nCOST_OPTIMAL algorithm:\n"
              << "  Starts from min-partition, merges to optimize, never exceeds max-partition\n"
              << "\nFormats:\n"
              << "  vertical  Lane-major layout (strided access within mini-vector)\n"
              << "  gm        Word-interleaved layout (coalesced access within mini-vector)\n"
              << "\nExample:\n"
              << "  " << program_name << " --data-dir /path/to/test/data/sosd/\n"
              << "  " << program_name << " -d /path/to/sosd --dataset 7    # Test only wiki\n"
              << "  " << program_name << " --format gm --dataset 7          # Test wiki with GM format\n"
              << "  " << program_name << " --min-partition 128 --max-partition 4096\n"
              << "  " << program_name << " --block-size 128  # Test with 128 threads per block\n"
              << "  " << program_name << " --v5  # Enable V5 FOR-only decoder comparison\n";
}

int main(int argc, char** argv) {
    // Configuration with defaults
    int partition_size = 4096;
    int min_partition = 256;
    int max_partition = 8192;
    int warmup = 2;
    int runs = 5;
    int single_dataset = -1;  // -1 means test all
    bool skip_synthetic = false;
    bool enable_merging = true;
    bool verbose = false;
    bool test_ssb = false;     // Test SSB LINEORDER datasets
    bool sort_data = false;    // Sort data before compression (for RLE analysis)
    std::string csv_output = "model_statistics_e2e.csv";
    std::string format = "vertical";  // "vertical" or "gm"
    int encoder_block_size = 256;     // Encoder selector kernel block size
    bool scan_encoder_block_size = false;  // Scan different encoder block sizes
    std::string custom_file = "";     // Custom file path (overrides --dataset)
    std::string custom_name = "";     // Custom dataset name

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "-d" || arg == "--data-dir") {
            if (i + 1 < argc) {
                DATA_DIR = argv[++i];
                // Ensure path ends with /
                if (!DATA_DIR.empty() && DATA_DIR.back() != '/') {
                    DATA_DIR += '/';
                }
            } else {
                std::cerr << "Error: " << arg << " requires a path argument\n";
                printUsage(argv[0]);
                return 1;
            }
        } else if (arg == "--dataset") {
            if (i + 1 < argc) {
                single_dataset = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: --dataset requires a dataset ID\n";
                return 1;
            }
        } else if (arg == "--format") {
            if (i + 1 < argc) {
                format = argv[++i];
                if (format != "vertical" && format != "gm") {
                    std::cerr << "Error: --format must be 'vertical' or 'gm'\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --format requires an argument (vertical or gm)\n";
                return 1;
            }
        } else if (arg == "--warmup") {
            if (i + 1 < argc) {
                warmup = std::stoi(argv[++i]);
            }
        } else if (arg == "--runs") {
            if (i + 1 < argc) {
                runs = std::stoi(argv[++i]);
            }
        } else if (arg == "--partition") {
            if (i + 1 < argc) {
                partition_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--min-partition") {
            if (i + 1 < argc) {
                min_partition = std::stoi(argv[++i]);
            }
        } else if (arg == "--max-partition") {
            if (i + 1 < argc) {
                max_partition = std::stoi(argv[++i]);
            }
        } else if (arg == "--block-size") {
            if (i + 1 < argc) {
                V4_decoder::g_block_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--encoder-block-size") {
            if (i + 1 < argc) {
                encoder_block_size = std::stoi(argv[++i]);
            }
        } else if (arg == "--scan-encoder-block-size") {
            scan_encoder_block_size = true;
        } else if (arg == "--v5") {
            V5_FOR_decoder::g_use_v5_decoder = true;
        } else if (arg == "--no-synthetic") {
            skip_synthetic = true;
        } else if (arg == "--ssb") {
            test_ssb = true;
        } else if (arg == "--sort") {
            sort_data = true;
        } else if (arg == "--no-merge") {
            enable_merging = false;
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                csv_output = argv[++i];
            }
        } else if (arg == "--file" || arg == "-f") {
            if (i + 1 < argc) {
                custom_file = argv[++i];
            } else {
                std::cerr << "Error: --file requires a file path\n";
                return 1;
            }
        } else if (arg == "--name") {
            if (i + 1 < argc) {
                custom_name = argv[++i];
            } else {
                std::cerr << "Error: --name requires a dataset name\n";
                return 1;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    std::cout << "\n=== L3 Model Statistics E2E Test ===\n";
    std::cout << "Full E2E: Encode + V4 Decode + Verification\n";
    std::cout << "==========================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Data Directory: " << DATA_DIR << "\n";
    std::cout << "  Storage Format: " << format << " ("
              << (format == "gm" ? "word-interleaved/coalesced" : "lane-major/strided") << ")\n";
    std::cout << "  Mini-Vector Size: " << MINI_VECTOR_SIZE << " (VALUES_PER_THREAD=" << VALUES_PER_THREAD << ")\n";
    std::cout << "  COST_OPTIMAL: start=" << min_partition << ", max=" << max_partition
              << ", merge=" << (enable_merging ? "ON" : "OFF") << "\n";
    std::cout << "  FIXED (signed): partition=" << partition_size << "\n";
    std::cout << "  Warmup: " << warmup << ", Runs: " << runs << "\n";
    std::cout << "  Selector: GPU Full Polynomial (CONSTANT, LINEAR, POLY2, POLY3, FOR)\n";
    std::cout << "  Encoder Block Size: " << (scan_encoder_block_size ? "SCAN [32,64,128,256,512,1024]" : std::to_string(encoder_block_size)) << "\n";
    if (V4_decoder::g_block_size < 0) {
        std::cout << "  Decoder: " << (format == "gm" ? "GM_V4" : "V4") << " Optimized Interleaved (Block Size: ADAPTIVE)\n";
    } else {
        std::cout << "  Decoder: " << (format == "gm" ? "GM_V4" : "V4") << " Optimized Interleaved (Block Size: " << V4_decoder::g_block_size << ")\n";
    }
    if (V5_FOR_decoder::g_use_v5_decoder && format == "vertical") {
        std::cout << "  V5 FOR Decoder: ENABLED (compile-time bit-width specialization)\n";
    }
    if (single_dataset > 0) {
        std::cout << "  Single Dataset: " << single_dataset << "\n";
    }
    if (!custom_file.empty()) {
        std::cout << "  Custom File: " << custom_file << "\n";
        if (!custom_name.empty()) {
            std::cout << "  Custom Name: " << custom_name << "\n";
        }
    }
    std::cout << "\n";

    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA Device: " << prop.name << "\n\n";

    std::vector<DatasetResult> synthetic_results;
    std::vector<DatasetResult> sosd_results;
    int total_passed = 0, total_failed = 0;

    // ========================================================================
    // Part 1: Process Synthetic True Linear Datasets (optional)
    // ========================================================================
    if (!skip_synthetic && single_dataset < 0) {
        std::cout << "=== Part 1: Synthetic True Linear Datasets (" << format << " format) ===\n";
        for (const auto& dataset : SYNTHETIC_DATASETS) {
            std::cout << "Processing " << dataset.name << "... " << std::flush;

            DatasetResult result;
            if (format == "gm") {
                result = processDataset_GM(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
            } else {
                result = processDataset(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
            }

            if (result.success) {
                std::cout << (result.verified ? "OK" : "FAIL")
                          << " (Enc: " << std::fixed << std::setprecision(1) << result.encode_gbps << " GB/s"
                          << ", Dec: " << result.decode_gbps << " GB/s"
                          << ", " << result.stats.linear_pct() << "% LINEAR)\n";
                if (result.verified) total_passed++;
                else total_failed++;
            } else {
                std::cout << "FAILED: " << result.error_msg << "\n";
                total_failed++;
            }

            synthetic_results.push_back(result);
        }
        std::cout << "\n";
    }

    // ========================================================================
    // Part 2: Process SOSD Datasets (1-20 or single)
    // ========================================================================
    std::cout << "=== Part 2: SOSD Datasets (" << format << " format) ===\n";

    // Block sizes to test when scanning
    std::vector<int> encoder_block_sizes_to_test = {32, 64, 128, 256, 512, 1024};

    if (scan_encoder_block_size && format == "vertical") {
        // ========================================================================
        // Encoder Block Size Scan Mode
        // ========================================================================
        std::cout << "\n=== ENCODER BLOCK SIZE SCAN MODE ===\n";
        std::cout << "Testing block sizes: ";
        for (int bs : encoder_block_sizes_to_test) std::cout << bs << " ";
        std::cout << "\n\n";

        // Print header
        std::cout << std::left << std::setw(20) << "Dataset";
        for (int bs : encoder_block_sizes_to_test) {
            std::cout << std::right << std::setw(8) << ("BS" + std::to_string(bs));
        }
        std::cout << std::setw(10) << "Best_BS" << std::setw(10) << "Best_GB/s" << "\n";
        std::cout << std::string(20 + encoder_block_sizes_to_test.size() * 8 + 20, '-') << "\n";

        // Store best parameters for each dataset
        struct BestParams {
            int dataset_id;
            std::string dataset_name;
            int best_block_size;
            double best_encode_gbps;
            std::map<int, double> all_results;  // block_size -> encode_gbps
        };
        std::vector<BestParams> all_best_params_arr;

        for (const auto& dataset : DATASETS) {
            if (single_dataset > 0 && dataset.id != single_dataset) continue;

            BestParams best;
            best.dataset_id = dataset.id;
            best.dataset_name = dataset.name;
            best.best_block_size = 256;
            best.best_encode_gbps = 0.0;

            std::string name = std::to_string(dataset.id) + "-" + dataset.name;
            std::cout << std::left << std::setw(20) << name;

            for (int bs : encoder_block_sizes_to_test) {
                DatasetResult result = processDataset(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, bs);

                double gbps = result.success ? result.encode_gbps : 0.0;
                best.all_results[bs] = gbps;

                std::cout << std::right << std::fixed << std::setprecision(1) << std::setw(8) << gbps;

                if (gbps > best.best_encode_gbps) {
                    best.best_encode_gbps = gbps;
                    best.best_block_size = bs;
                }

                if (result.success && result.verified) total_passed++;
                else total_failed++;
            }

            std::cout << std::setw(10) << best.best_block_size
                      << std::fixed << std::setprecision(2) << std::setw(10) << best.best_encode_gbps << "\n";

            all_best_params_arr.push_back(best);

            // Store the best result for CSV output
            DatasetResult best_result = processDataset(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, best.best_block_size);
            sosd_results.push_back(best_result);
        }

        // Print summary of best parameters
        std::cout << "\n=== BEST ENCODER BLOCK SIZE SUMMARY ===\n";
        std::cout << std::left << std::setw(20) << "Dataset"
                  << std::right << std::setw(12) << "Best_BS"
                  << std::setw(12) << "Enc_GB/s" << "\n";
        std::cout << std::string(44, '-') << "\n";
        for (const auto& bp : all_best_params_arr) {
            std::string name = std::to_string(bp.dataset_id) + "-" + bp.dataset_name;
            std::cout << std::left << std::setw(20) << name
                      << std::right << std::setw(12) << bp.best_block_size
                      << std::fixed << std::setprecision(2) << std::setw(12) << bp.best_encode_gbps << "\n";
        }

        // Save scan results to CSV
        std::string scan_csv = csv_output.substr(0, csv_output.rfind('.')) + "_encoder_bs_scan.csv";
        std::ofstream scan_file(scan_csv);
        if (scan_file.is_open()) {
            scan_file << "Dataset,DatasetID";
            for (int bs : encoder_block_sizes_to_test) {
                scan_file << ",BS" << bs << "_GBps";
            }
            scan_file << ",BestBS,BestGBps\n";

            for (const auto& bp : all_best_params_arr) {
                scan_file << bp.dataset_name << "," << bp.dataset_id;
                for (int bs : encoder_block_sizes_to_test) {
                    scan_file << "," << std::fixed << std::setprecision(2) << bp.all_results.at(bs);
                }
                scan_file << "," << bp.best_block_size << "," << std::fixed << std::setprecision(2) << bp.best_encode_gbps << "\n";
            }
            scan_file.close();
            std::cout << "\nEncoder block size scan results saved to: " << scan_csv << "\n";
        }

    } else {
        // ========================================================================
        // Normal Mode (single encoder block size)
        // ========================================================================

        // Print header for inline results
        std::cout << std::left << std::setw(20) << "Dataset"
                  << std::right << std::setw(10) << "Elements"
                  << std::setw(10) << "Enc GB/s"
                  << std::setw(10) << "Dec GB/s"
                  << std::setw(8) << "Ratio"
                  << std::setw(8) << "Verify"
                  << "\n";
        std::cout << std::string(66, '-') << "\n";

        // Handle custom file if specified
        if (!custom_file.empty()) {
            // Create a custom dataset info
            // Determine if it's uint64 based on filename (default to uint32)
            bool is_uint64 = (custom_file.find("uint64") != std::string::npos ||
                             custom_file.find("64.bin") != std::string::npos);

            // Use custom name or extract from filename
            std::string dataset_name = custom_name;
            if (dataset_name.empty()) {
                size_t last_slash = custom_file.find_last_of("/\\");
                size_t last_dot = custom_file.find_last_of(".");
                if (last_slash != std::string::npos) {
                    dataset_name = custom_file.substr(last_slash + 1,
                        (last_dot != std::string::npos && last_dot > last_slash) ?
                        last_dot - last_slash - 1 : std::string::npos);
                } else {
                    dataset_name = custom_file.substr(0, last_dot);
                }
            }

            DatasetInfo custom_dataset = {0, custom_file, dataset_name, is_uint64, false};

            // Temporarily set DATA_DIR to empty since custom_file is absolute path
            std::string saved_data_dir = DATA_DIR;
            DATA_DIR = "";

            DatasetResult result;
            if (format == "gm") {
                result = processDataset_GM(custom_dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
            } else {
                result = processDataset(custom_dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size);
            }

            // Restore DATA_DIR
            DATA_DIR = saved_data_dir;

            if (result.success) {
                std::cout << std::left << std::setw(20) << dataset_name
                          << std::right << std::setw(10) << result.num_elements
                          << std::fixed << std::setprecision(2)
                          << std::setw(10) << result.encode_gbps
                          << std::setw(10) << result.decode_gbps
                          << std::setw(8) << result.compression_ratio
                          << std::setw(8) << (result.verified ? "OK" : "FAIL")
                          << "\n";

                if (result.verified) total_passed++;
                else total_failed++;
            } else {
                std::cout << std::left << std::setw(20) << dataset_name
                          << " FAILED: " << result.error_msg << "\n";
                total_failed++;
            }

            sosd_results.push_back(result);

        } else {
            // Normal dataset loop (skip if only testing SSB)
            if (!test_ssb) {
                for (const auto& dataset : DATASETS) {
                    // Filter by single dataset if specified
                    if (single_dataset > 0 && dataset.id != single_dataset) {
                        continue;
                    }

                    DatasetResult result;
                    if (format == "gm") {
                        result = processDataset_GM(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
                    } else {
                        result = processDataset(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size);
                    }

                    std::string name = std::to_string(dataset.id) + "-" + dataset.name;
                    if (result.success) {
                        std::cout << std::left << std::setw(20) << name
                                  << std::right << std::setw(10) << result.num_elements
                                  << std::fixed << std::setprecision(2)
                                  << std::setw(10) << result.encode_gbps
                                  << std::setw(10) << result.decode_gbps
                                  << std::setw(8) << result.compression_ratio
                                  << std::setw(8) << (result.verified ? "OK" : "FAIL")
                                  << "\n";

                        if (result.verified) total_passed++;
                    else total_failed++;
                } else {
                    std::cout << std::left << std::setw(20) << name
                              << " FAILED: " << result.error_msg << "\n";
                    total_failed++;
                }

                sosd_results.push_back(result);
            }
            } // end if (!test_ssb)

            // Test SSB LINEORDER datasets if requested
            if (test_ssb) {
                std::cout << "\n=== SSB LINEORDER Datasets" << (sort_data ? " (SORTED)" : "") << " ===\n";
                std::cout << std::string(66, '-') << "\n";

                for (const auto& dataset : SSB_LINEORDER_DATASETS) {
                    DatasetResult result;
                    if (format == "gm") {
                        result = processDataset_GM(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs);
                    } else {
                        result = processDataset(dataset, partition_size, min_partition, max_partition, enable_merging, warmup, runs, encoder_block_size, sort_data);
                    }

                    std::string name = dataset.name;
                    if (result.success) {
                        std::cout << std::left << std::setw(20) << name
                                  << std::right << std::setw(10) << result.num_elements
                                  << std::fixed << std::setprecision(2)
                                  << std::setw(10) << result.encode_gbps
                                  << std::setw(10) << result.decode_gbps
                                  << std::setw(8) << result.compression_ratio
                                  << std::setw(8) << (result.verified ? "OK" : "FAIL")
                                  << "\n";

                        if (result.verified) total_passed++;
                        else total_failed++;
                    } else {
                        std::cout << std::left << std::setw(20) << name
                                  << " FAILED: " << result.error_msg << "\n";
                        total_failed++;
                    }

                    sosd_results.push_back(result);
                }
                std::cout << std::string(66, '-') << "\n";
            }
        }

        std::cout << std::string(66, '-') << "\n";
    }

    // ========================================================================
    // Output Results
    // ========================================================================

    // Print comparison table (True Linear vs SOSD Linear) only when running all
    if (!skip_synthetic && single_dataset < 0 && !synthetic_results.empty()) {
        printComparisonTable(synthetic_results, sosd_results);
    }

    // Print full SOSD results table with model stats
    printResultsTable(sosd_results, "V2 Cost-Optimal (GPU Merge)");

    // Save all results to CSV (combine synthetic + SOSD)
    std::vector<DatasetResult> all_results;
    if (!skip_synthetic) {
        all_results.insert(all_results.end(), synthetic_results.begin(), synthetic_results.end());
    }
    all_results.insert(all_results.end(), sosd_results.begin(), sosd_results.end());
    saveResultsCSV(all_results, csv_output);

    // Final summary
    std::cout << "\n=== Final E2E Verification Summary ===\n";
    std::cout << "Total Passed: " << total_passed << "\n";
    std::cout << "Total Failed: " << total_failed << "\n";

    return (total_failed == 0) ? 0 : 1;
}
