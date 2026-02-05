/**
 * L3 Transposed (Word-Interleaved) Encoder
 *
 * This encoder creates compressed data in the Transposed (Word-Interleaved) format
 * for optimal GPU memory coalescing during decompression.
 *
 * KEY DIFFERENCE from Vertical (Lane-Major):
 *   Vertical:   [Lane0_W0][Lane0_W1]...[Lane0_WN][Lane1_W0]...[Lane31_WN]
 *   Transposed: [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *
 * Memory Access Pattern:
 *   Vertical:   Thread 0 reads addr 0, Thread 1 reads addr N (STRIDED!)
 *   Transposed: Thread 0 reads addr 0, Thread 1 reads addr 1 (COALESCED!)
 *
 * Platform: SM 8.0+ (Ampere and later)
 * Date: 2025-12-16
 */

#ifndef ENCODER_TRANSPOSED_CU
#define ENCODER_TRANSPOSED_CU

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>

#include "L3_format.hpp"
#include "L3_Transposed_format.hpp"
#include "../utils/bitpack_utils_Transposed.cuh"
#include "adaptive_selector.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"
#include "encoder_cost_optimal_gpu_merge_v3.cuh"  // V3 partitioner for PolyCost

namespace Transposed_encoder {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;
constexpr double MODEL_OVERHEAD_BYTES = 64.0;

// ============================================================================
// Device Helper Functions
// ============================================================================

/**
 * Compute prediction using polynomial model (Horner's method) - matches decoder exactly
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

template<typename T>
__device__ __forceinline__ int64_t calculateDelta(T actual, T predicted) {
    if (actual >= predicted) {
        return static_cast<int64_t>(actual - predicted);
    } else {
        return -static_cast<int64_t>(predicted - actual);
    }
}

// ============================================================================
// TRANSPOSED Packing Kernel (Word-Interleaved)
// ============================================================================

/**
 * Convert values to TRANSPOSED (Word-Interleaved) mini-vector format
 *
 * Each block processes one mini-vector (256 values).
 * Each thread handles 8 values in its lane.
 *
 * KEY DIFFERENCE from Vertical:
 *   Vertical:   word_addr = mv_base + lane_id * words_per_lane + word_in_lane
 *   Transposed: word_addr = mv_base + word_in_lane * 32 + lane_id
 *
 * This means all 32 threads' word[W] are stored consecutively,
 * enabling coalesced memory access during decompression!
 */
template<typename T>
__global__ void convertToTransposedKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int64_t* __restrict__ d_transposed_offsets,
    int num_partitions,
    uint32_t* __restrict__ transposed_array)
{
    // Use shared memory to find partition for this block
    __shared__ int s_pid;
    __shared__ int s_mv_in_partition;
    __shared__ int s_partition_start;
    __shared__ int s_delta_bits;
    __shared__ double s_params[4];
    __shared__ int64_t s_transposed_base;
    __shared__ int s_model_type;
    __shared__ bool s_valid;

    // Thread 0 finds the partition
    if (threadIdx.x == 0) {
        int mv_global = blockIdx.x;
        int cumulative_mv = 0;
        s_valid = false;

        for (int p = 0; p < num_partitions; p++) {
            int num_mv = d_num_mini_vectors[p];
            if (mv_global < cumulative_mv + num_mv) {
                s_pid = p;
                s_mv_in_partition = mv_global - cumulative_mv;
                s_partition_start = d_start_indices[p];
                s_delta_bits = d_delta_bits[p];
                s_params[0] = d_model_params[p * 4];
                s_params[1] = d_model_params[p * 4 + 1];
                s_params[2] = d_model_params[p * 4 + 2];
                s_params[3] = d_model_params[p * 4 + 3];
                s_transposed_base = d_transposed_offsets[p];
                s_model_type = d_model_types[p];
                s_valid = true;
                break;
            }
            cumulative_mv += num_mv;
        }
    }
    __syncthreads();

    if (!s_valid) return;

    int pid = s_pid;
    int mv_idx = s_mv_in_partition;
    int partition_start = s_partition_start;
    int delta_bits = s_delta_bits;
    double params[4];
    params[0] = s_params[0];
    params[1] = s_params[1];
    params[2] = s_params[2];
    params[3] = s_params[3];
    int64_t transposed_base = s_transposed_base;
    int model_type = s_model_type;

    // Each thread is a lane (0-31), processes 8 values
    int lane_id = threadIdx.x;
    if (lane_id >= WARP_SIZE) return;

    // Global indices this lane processes
    int mv_start_global = partition_start + mv_idx * MINI_VECTOR_SIZE;

    // Calculate words per lane and mini-vector word base
    int bits_per_lane = delta_bits * VALUES_PER_THREAD;  // 8 values * bit_width
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mv = words_per_lane * WARP_SIZE;

    // Mini-vector word base in transposed array
    int64_t mv_word_base = transposed_base + static_cast<int64_t>(mv_idx) * words_per_mv;

    // Process 8 values for this lane
    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; v++) {
        // Global index: lane_id + v * 32 within mini-vector
        int local_idx_in_mv = lane_id + v * WARP_SIZE;
        int global_idx = mv_start_global + local_idx_in_mv;
        int local_idx_in_partition = mv_idx * MINI_VECTOR_SIZE + local_idx_in_mv;

        uint64_t packed_value;

        if (model_type == MODEL_FOR_BITPACK) {
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }
            packed_value = static_cast<uint64_t>(values[global_idx] - base);
        } else {
            T pred_val = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
            int64_t delta = calculateDelta(values[global_idx], pred_val);

            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;
        }

        // Calculate bit position within lane's data
        int bit_in_lane = v * delta_bits;

        // Pack using TRANSPOSED layout
        // Use packToTransposed from bitpack_utils_Transposed.cuh
        Transposed::packToTransposed(
            transposed_array,
            mv_word_base,
            lane_id,
            bit_in_lane,
            packed_value,
            delta_bits
        );
    }
}

// ============================================================================
// Tail Values Packing Kernel (Sequential - same as Vertical)
// ============================================================================

/**
 * Pack tail values (partition_size % 256) into transposed array
 *
 * Tail data is placed immediately after the mini-vectors.
 * Uses sequential packing (same as Vertical) since there's no warp-level pattern.
 */
template<typename T>
__global__ void packTailValuesKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int32_t* __restrict__ d_num_mini_vectors,
    const int32_t* __restrict__ d_tail_sizes,
    const int64_t* __restrict__ d_transposed_offsets,
    int num_partitions,
    uint32_t* __restrict__ transposed_array)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int tail_size = d_tail_sizes[pid];
    if (tail_size == 0) return;

    int64_t transposed_base = d_transposed_offsets[pid];
    if (transposed_base < 0) return;

    int num_mv = d_num_mini_vectors[pid];
    int delta_bits = d_delta_bits[pid];
    int model_type = d_model_types[pid];
    int partition_start = d_start_indices[pid];

    // Load model parameters
    double params[4];
    params[0] = d_model_params[pid * 4];
    params[1] = d_model_params[pid * 4 + 1];
    params[2] = d_model_params[pid * 4 + 2];
    params[3] = d_model_params[pid * 4 + 3];

    // Calculate words per lane and mini-vector
    int bits_per_lane = delta_bits * VALUES_PER_THREAD;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mv = words_per_lane * WARP_SIZE;

    // Tail starts after all mini-vectors
    int tail_start_local = num_mv * MINI_VECTOR_SIZE;
    int tail_start_global = partition_start + tail_start_local;

    // Tail word base (after all mini-vector words)
    int64_t tail_word_base = transposed_base + static_cast<int64_t>(num_mv) * words_per_mv;
    int64_t tail_bit_base = tail_word_base * 32;

    // Each thread handles multiple tail values
    for (int i = threadIdx.x; i < tail_size; i += blockDim.x) {
        int global_idx = tail_start_global + i;
        int local_idx_in_partition = tail_start_local + i;

        uint64_t packed_value;

        if (model_type == MODEL_FOR_BITPACK) {
            T base;
            if (sizeof(T) == 8) {
                base = static_cast<T>(__double_as_longlong(params[0]));
            } else {
                base = static_cast<T>(__double2ll_rn(params[0]));
            }
            packed_value = static_cast<uint64_t>(values[global_idx] - base);
        } else {
            T pred_val = computePredictionPoly<T>(model_type, params, local_idx_in_partition);
            int64_t delta = calculateDelta(values[global_idx], pred_val);

            uint64_t mask = (delta_bits == 64) ? ~0ULL : ((1ULL << delta_bits) - 1ULL);
            packed_value = static_cast<uint64_t>(delta) & mask;
        }

        if (delta_bits == 0) continue;

        // Sequential packing for tail values
        int64_t bit_offset = tail_bit_base + static_cast<int64_t>(i) * delta_bits;
        int word_idx = bit_offset >> 5;
        int bit_in_word = bit_offset & 31;

        // First word
        int bits_in_first = min(delta_bits, 32 - bit_in_word);
        uint32_t mask_first = (bits_in_first == 32) ? ~0U : ((1U << bits_in_first) - 1U);
        uint32_t val_first = static_cast<uint32_t>(packed_value & mask_first) << bit_in_word;
        atomicOr(&transposed_array[word_idx], val_first);

        int bits_remaining = delta_bits - bits_in_first;
        if (bits_remaining > 0) {
            uint64_t shifted = packed_value >> bits_in_first;
            int bits_in_second = min(bits_remaining, 32);
            uint32_t mask_second = (bits_in_second == 32) ? ~0U : ((1U << bits_in_second) - 1U);
            uint32_t val_second = static_cast<uint32_t>(shifted & mask_second);
            atomicOr(&transposed_array[word_idx + 1], val_second);

            bits_remaining -= bits_in_second;
            if (bits_remaining > 0) {
                uint32_t val_third = static_cast<uint32_t>(shifted >> 32);
                atomicOr(&transposed_array[word_idx + 2], val_third);
            }
        }
    }
}

// ============================================================================
// GPU Kernels for Metadata Computation
// ============================================================================

/**
 * Kernel: Generate fixed-size partition indices directly on GPU
 */
__global__ void generateFixedPartitionsKernel(
    int32_t* __restrict__ start_indices,
    int32_t* __restrict__ end_indices,
    int partition_size,
    int total_elements,
    int num_partitions)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_partitions) return;

    start_indices[i] = i * partition_size;
    end_indices[i] = min((i + 1) * partition_size, total_elements);
}

/**
 * Kernel: Unpack ModelDecision structures into separate metadata arrays
 */
template<typename T>
__global__ void unpackDecisionsKernel(
    const adaptive_selector::ModelDecision<T>* __restrict__ decisions,
    int32_t* __restrict__ model_types,
    double* __restrict__ model_params,
    int32_t* __restrict__ delta_bits,
    int64_t* __restrict__ error_bounds,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    const auto& dec = decisions[pid];
    model_types[pid] = dec.model_type;
    delta_bits[pid] = dec.delta_bits;
    error_bounds[pid] = dec.max_val - dec.min_val;

    #pragma unroll
    for (int j = 0; j < 4; j++) {
        model_params[pid * 4 + j] = dec.params[j];
    }
}

/**
 * Kernel: Compute transposed metadata for each partition
 */
__global__ void computeTransposedMetadataKernel(
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    const int32_t* __restrict__ delta_bits,
    int32_t* __restrict__ num_mini_vectors,
    int32_t* __restrict__ tail_sizes,
    int64_t* __restrict__ word_counts,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    int psize = end_indices[pid] - start_indices[pid];
    int bit_width = delta_bits[pid];

    num_mini_vectors[pid] = psize / MINI_VECTOR_SIZE;
    tail_sizes[pid] = psize % MINI_VECTOR_SIZE;

    // Calculate words for transposed layout
    int bits_per_lane = bit_width * VALUES_PER_THREAD;
    int words_per_lane = (bits_per_lane + 31) / 32;
    int words_per_mv = words_per_lane * WARP_SIZE;

    int64_t mv_words = static_cast<int64_t>(num_mini_vectors[pid]) * words_per_mv;
    int64_t tail_bits = static_cast<int64_t>(tail_sizes[pid]) * bit_width;
    int64_t tail_words = (tail_bits + 31) / 32;

    word_counts[pid] = mv_words + tail_words;
}

/**
 * Kernel: Set transposed offsets from prefix sum results
 */
__global__ void setTransposedOffsetsKernel(
    const int64_t* __restrict__ prefix_sums,
    int64_t* __restrict__ transposed_offsets,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    transposed_offsets[pid] = prefix_sums[pid];
}

/**
 * Kernel: Count total mini-vectors
 */
__global__ void countTotalMiniVectorsKernel(
    const int32_t* __restrict__ num_mini_vectors,
    int num_partitions,
    int* __restrict__ total_count)
{
    __shared__ int shared_sum[256];

    int tid = threadIdx.x;
    int local_sum = 0;

    for (int i = tid; i < num_partitions; i += blockDim.x) {
        local_sum += num_mini_vectors[i];
    }

    shared_sum[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *total_count = shared_sum[0];
    }
}

/**
 * GPU Kernel: Merge theta arrays into model_params
 */
__global__ void mergeModelParamsKernel(
    const double* __restrict__ theta0,
    const double* __restrict__ theta1,
    const double* __restrict__ theta2,
    const double* __restrict__ theta3,
    double* __restrict__ model_params,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    model_params[pid * 4 + 0] = theta0[pid];
    model_params[pid * 4 + 1] = theta1[pid];
    model_params[pid * 4 + 2] = theta2[pid];
    model_params[pid * 4 + 3] = theta3[pid];
}

/**
 * Kernel: Compute partition min/max values for predicate pushdown
 */
template<typename T>
__global__ void computePartitionMinMaxKernel(
    const T* __restrict__ values,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    int num_partitions,
    T* __restrict__ d_partition_min,
    T* __restrict__ d_partition_max)
{
    extern __shared__ char shared_mem[];
    T* s_min = reinterpret_cast<T*>(shared_mem);
    T* s_max = reinterpret_cast<T*>(shared_mem + blockDim.x * sizeof(T));

    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = d_start_indices[pid];
    int end = d_end_indices[pid];
    int size = end - start;

    if (size == 0) {
        if (threadIdx.x == 0) {
            d_partition_min[pid] = T(0);
            d_partition_max[pid] = T(0);
        }
        return;
    }

    T local_min = values[start];
    T local_max = values[start];

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        T val = values[start + i];
        local_min = (val < local_min) ? val : local_min;
        local_max = (val > local_max) ? val : local_max;
    }

    s_min[threadIdx.x] = local_min;
    s_max[threadIdx.x] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            T other_min = s_min[threadIdx.x + stride];
            T other_max = s_max[threadIdx.x + stride];
            s_min[threadIdx.x] = (other_min < s_min[threadIdx.x]) ? other_min : s_min[threadIdx.x];
            s_max[threadIdx.x] = (other_max > s_max[threadIdx.x]) ? other_max : s_max[threadIdx.x];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_partition_min[pid] = s_min[0];
        d_partition_max[pid] = s_max[0];
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert TransposedConfig to CostOptimalConfig
 */
inline CostOptimalConfig TransposedToCostOptimalConfig(const TransposedConfig& config, int partition_size) {
    CostOptimalConfig cost_config;
    cost_config.analysis_block_size = config.cost_analysis_block_size;
    cost_config.min_partition_size = config.cost_min_partition_size;
    cost_config.max_partition_size = config.cost_max_partition_size;
    cost_config.breakpoint_threshold = config.cost_breakpoint_threshold;
    cost_config.merge_benefit_threshold = config.cost_merge_benefit_threshold;
    cost_config.max_merge_rounds = config.cost_max_merge_rounds;
    cost_config.enable_merging = config.cost_enable_merging;
    cost_config.enable_polynomial_models = config.enable_adaptive_selection;
    return cost_config;
}

/**
 * Copy GPUPartitionResult to device arrays (templated for V2/V3 compatibility)
 */
template<typename T, typename GPUPartitionResultT>
void copyGPUPartitionResultToDevice(
    const GPUPartitionResultT& gpu_result,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    cudaStream_t stream)
{
    int np = gpu_result.num_partitions;

    cudaMemcpyAsync(d_start_indices, gpu_result.d_starts, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_end_indices, gpu_result.d_ends, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_model_types, gpu_result.d_model_types, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_delta_bits, gpu_result.d_delta_bits, np * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_error_bounds, gpu_result.d_max_errors, np * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);

    int threads = 256;
    int blocks = (np + threads - 1) / threads;
    mergeModelParamsKernel<<<blocks, threads, 0, stream>>>(
        gpu_result.d_theta0,
        gpu_result.d_theta1,
        gpu_result.d_theta2,
        gpu_result.d_theta3,
        d_model_params,
        np);
}

// ============================================================================
// Main Encoder Function
// ============================================================================

/**
 * Full GPU Pipeline Encoder for Transposed (Word-Interleaved) Layout
 *
 * Steps:
 * 1. Data upload
 * 2. Partition generation/cost-optimal partitioning
 * 3. Model selection
 * 4. Transposed metadata computation
 * 5. Mini-vector delta packing (TRANSPOSED layout)
 * 6. Tail delta packing (sequential)
 */
template<typename T, typename PartitionerT>
CompressedDataTransposed<T> encodeTransposedGPU_Internal(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config,
    cudaStream_t stream = 0)
{
    CompressedDataTransposed<T> result;

    // For COST_OPTIMAL, partition_size is ignored (uses config.cost_min/max_partition_size)
    bool use_cost_optimal = (config.partitioning_strategy == PartitioningStrategy::COST_OPTIMAL);
    if (data.empty() || (!use_cost_optimal && partition_size <= 0)) {
        return result;
    }

    size_t n = data.size();
    int num_partitions;

    // ========== Step 1: Upload data to GPU ==========
    T* d_data;
    cudaMalloc(&d_data, n * sizeof(T));
    cudaMemcpyAsync(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice, stream);

    // ========== CUDA Events for kernel timing ==========
    cudaEvent_t kernel_start, kernel_end;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_end);

    // Temporary arrays
    int32_t* d_start_indices_temp = nullptr;
    int32_t* d_end_indices_temp = nullptr;
    adaptive_selector::ModelDecision<T>* d_decisions = nullptr;

    if (use_cost_optimal) {
        // ========== COST_OPTIMAL Path ==========
        CostOptimalConfig cost_config = TransposedToCostOptimalConfig(config, partition_size);

        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        PartitionerT partitioner(data, cost_config, stream);
        auto gpu_partitions = partitioner.partitionGPU();
        num_partitions = gpu_partitions.num_partitions;

        if (num_partitions == 0) {
            cudaFree(d_data);
            return result;
        }

        result.num_partitions = num_partitions;
        result.total_values = n;

        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_transposed_offsets, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        copyGPUPartitionResultToDevice<T>(
            gpu_partitions,
            result.d_start_indices, result.d_end_indices,
            result.d_model_types, result.d_model_params,
            result.d_delta_bits, result.d_error_bounds,
            stream);

    } else {
        // ========== FIXED Path ==========
        cudaStreamSynchronize(stream);
        cudaEventRecord(kernel_start, stream);

        num_partitions = (n + partition_size - 1) / partition_size;

        result.num_partitions = num_partitions;
        result.total_values = n;

        int threads = BLOCK_SIZE;
        int blocks_np = (num_partitions + threads - 1) / threads;

        cudaMalloc(&d_start_indices_temp, num_partitions * sizeof(int32_t));
        cudaMalloc(&d_end_indices_temp, num_partitions * sizeof(int32_t));

        generateFixedPartitionsKernel<<<blocks_np, threads, 0, stream>>>(
            d_start_indices_temp,
            d_end_indices_temp,
            partition_size,
            static_cast<int>(n),
            num_partitions
        );

        cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<T>));

        if (config.enable_adaptive_selection) {
            adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions, d_decisions, 256, stream);
        } else {
            adaptive_selector::launchFixedModelSelector<T>(
                d_data, d_start_indices_temp, d_end_indices_temp, num_partitions,
                config.fixed_model_type, d_decisions, 256, stream);
        }

        cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
        cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_num_mini_vectors, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_tail_sizes, num_partitions * sizeof(int32_t));
        cudaMalloc(&result.d_transposed_offsets, num_partitions * sizeof(int64_t));
        cudaMalloc(&result.d_partition_min_values, num_partitions * sizeof(T));
        cudaMalloc(&result.d_partition_max_values, num_partitions * sizeof(T));

        cudaMemcpyAsync(result.d_start_indices, d_start_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result.d_end_indices, d_end_indices_temp,
                        num_partitions * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        int threads_unpack = BLOCK_SIZE;
        int blocks_unpack = (num_partitions + threads_unpack - 1) / threads_unpack;
        unpackDecisionsKernel<T><<<blocks_unpack, threads_unpack, 0, stream>>>(
            d_decisions,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_error_bounds,
            num_partitions
        );
    }

    // ========== Common Path: Compute transposed metadata and pack deltas ==========
    int threads = BLOCK_SIZE;
    int blocks_np = (num_partitions + threads - 1) / threads;

    // ========== Step 2: Compute transposed metadata ==========
    int64_t* d_word_counts;
    cudaMalloc(&d_word_counts, num_partitions * sizeof(int64_t));

    computeTransposedMetadataKernel<<<blocks_np, threads, 0, stream>>>(
        result.d_start_indices,
        result.d_end_indices,
        result.d_delta_bits,
        result.d_num_mini_vectors,
        result.d_tail_sizes,
        d_word_counts,
        num_partitions
    );

    // GPU prefix sum for word offsets
    int64_t* d_word_prefix;
    cudaMalloc(&d_word_prefix, num_partitions * sizeof(int64_t));

    thrust::device_ptr<int64_t> wcount_ptr(d_word_counts);
    thrust::device_ptr<int64_t> wprefix_ptr(d_word_prefix);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        wcount_ptr, wcount_ptr + num_partitions,
        wprefix_ptr
    );

    setTransposedOffsetsKernel<<<blocks_np, threads, 0, stream>>>(
        d_word_prefix,
        result.d_transposed_offsets,
        num_partitions
    );

    // Get total words and mini-vector count
    int64_t h_last_wcount, h_last_wprefix;
    cudaMemcpyAsync(&h_last_wcount, d_word_counts + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_wprefix, d_word_prefix + num_partitions - 1,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);

    int total_mini_vectors = 0;
    int* d_total_mv;
    cudaMalloc(&d_total_mv, sizeof(int));
    countTotalMiniVectorsKernel<<<1, 256, 0, stream>>>(
        result.d_num_mini_vectors,
        num_partitions,
        d_total_mv
    );
    cudaMemcpyAsync(&total_mini_vectors, d_total_mv, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    result.transposed_delta_words = h_last_wprefix + h_last_wcount + 4;

    // Allocate transposed deltas
    if (result.transposed_delta_words > 0) {
        cudaMalloc(&result.d_transposed_deltas,
                  result.transposed_delta_words * sizeof(uint32_t));
        cudaMemsetAsync(result.d_transposed_deltas, 0,
                       result.transposed_delta_words * sizeof(uint32_t), stream);
    }

    cudaFree(d_word_counts);
    cudaFree(d_word_prefix);
    cudaFree(d_total_mv);

    // ========== Step 3: Pack mini-vector deltas (TRANSPOSED layout) ==========
    if (total_mini_vectors > 0) {
        convertToTransposedKernel<T><<<total_mini_vectors, WARP_SIZE, 0, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            result.d_model_types,
            result.d_model_params,
            result.d_delta_bits,
            result.d_num_mini_vectors,
            result.d_transposed_offsets,
            num_partitions,
            result.d_transposed_deltas
        );
    }

    // ========== Step 4: Pack tail values (sequential) ==========
    packTailValuesKernel<T><<<num_partitions, BLOCK_SIZE, 0, stream>>>(
        d_data,
        result.d_start_indices,
        result.d_end_indices,
        result.d_model_types,
        result.d_model_params,
        result.d_delta_bits,
        result.d_num_mini_vectors,
        result.d_tail_sizes,
        result.d_transposed_offsets,
        num_partitions,
        result.d_transposed_deltas
    );

    // ========== Step 5: Compute partition min/max ==========
    {
        int minmax_block_size = 256;
        size_t minmax_shared_mem = 2 * minmax_block_size * sizeof(T);
        computePartitionMinMaxKernel<T><<<num_partitions, minmax_block_size, minmax_shared_mem, stream>>>(
            d_data,
            result.d_start_indices,
            result.d_end_indices,
            num_partitions,
            result.d_partition_min_values,
            result.d_partition_max_values
        );
    }

    // ========== Record kernel end time ==========
    cudaEventRecord(kernel_end, stream);

    // ========== Cleanup ==========
    cudaStreamSynchronize(stream);

    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_end);
    result.kernel_time_ms = kernel_ms;
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_end);

    cudaFree(d_data);
    if (d_start_indices_temp) cudaFree(d_start_indices_temp);
    if (d_end_indices_temp) cudaFree(d_end_indices_temp);
    if (d_decisions) cudaFree(d_decisions);

    return result;
}

/**
 * Free compressed data
 */
template<typename T>
void freeCompressedData(CompressedDataTransposed<T>& data)
{
    if (data.d_start_indices) cudaFree(data.d_start_indices);
    if (data.d_end_indices) cudaFree(data.d_end_indices);
    if (data.d_model_types) cudaFree(data.d_model_types);
    if (data.d_model_params) cudaFree(data.d_model_params);
    if (data.d_delta_bits) cudaFree(data.d_delta_bits);
    if (data.d_error_bounds) cudaFree(data.d_error_bounds);
    if (data.d_partition_min_values) cudaFree(data.d_partition_min_values);
    if (data.d_partition_max_values) cudaFree(data.d_partition_max_values);
    if (data.d_transposed_deltas) cudaFree(data.d_transposed_deltas);
    if (data.d_num_mini_vectors) cudaFree(data.d_num_mini_vectors);
    if (data.d_tail_sizes) cudaFree(data.d_tail_sizes);
    if (data.d_transposed_offsets) cudaFree(data.d_transposed_offsets);
    if (data.d_self) cudaFree(data.d_self);

    data = CompressedDataTransposed<T>();
}

// ============================================================================
// Public API Functions
// ============================================================================

/**
 * encodeTransposedGPU - Uses V2 partitioner (compatible with original API)
 */
template<typename T>
CompressedDataTransposed<T> encodeTransposedGPU(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config,
    cudaStream_t stream = 0)
{
    return encodeTransposedGPU_Internal<T, GPUCostOptimalPartitionerV2<T>>(
        data, partition_size, config, stream);
}

/**
 * encodeTransposedGPU_PolyCost - Uses V3 partitioner with full polynomial model selection
 * This is the recommended function for best compression ratio with adaptive models.
 */
template<typename T>
CompressedDataTransposed<T> encodeTransposedGPU_PolyCost(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config,
    cudaStream_t stream = 0)
{
    return encodeTransposedGPU_Internal<T, GPUCostOptimalPartitionerV3<T>>(
        data, partition_size, config, stream);
}

// ============================================================================
// Template Instantiations
// ============================================================================

// Explicit instantiations for encodeTransposedGPU
template CompressedDataTransposed<uint32_t> encodeTransposedGPU<uint32_t>(
    const std::vector<uint32_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<int32_t> encodeTransposedGPU<int32_t>(
    const std::vector<int32_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<uint64_t> encodeTransposedGPU<uint64_t>(
    const std::vector<uint64_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<int64_t> encodeTransposedGPU<int64_t>(
    const std::vector<int64_t>&, int, const TransposedConfig&, cudaStream_t);

// Explicit instantiations for encodeTransposedGPU_PolyCost
template CompressedDataTransposed<uint32_t> encodeTransposedGPU_PolyCost<uint32_t>(
    const std::vector<uint32_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<int32_t> encodeTransposedGPU_PolyCost<int32_t>(
    const std::vector<int32_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<uint64_t> encodeTransposedGPU_PolyCost<uint64_t>(
    const std::vector<uint64_t>&, int, const TransposedConfig&, cudaStream_t);
template CompressedDataTransposed<int64_t> encodeTransposedGPU_PolyCost<int64_t>(
    const std::vector<int64_t>&, int, const TransposedConfig&, cudaStream_t);

template void freeCompressedData<uint32_t>(CompressedDataTransposed<uint32_t>&);
template void freeCompressedData<int32_t>(CompressedDataTransposed<int32_t>&);
template void freeCompressedData<uint64_t>(CompressedDataTransposed<uint64_t>&);
template void freeCompressedData<int64_t>(CompressedDataTransposed<int64_t>&);

} // namespace Transposed_encoder

#endif // ENCODER_TRANSPOSED_CU
