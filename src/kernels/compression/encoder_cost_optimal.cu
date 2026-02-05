/**
 * Cost-Optimal Partitioning Algorithm for L3
 *
 * Implements a GPU-parallel partitioning algorithm based on actual compression cost
 * rather than variance heuristics. This avoids partition explosion on high-variance
 * data while maintaining good compression on linear data.
 */

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "finite_diff_shared.cuh"

// ============================================================================
// Constants
// ============================================================================

#define WARP_SIZE 32
#define MODEL_OVERHEAD_BYTES 64.0  // Bytes for partition metadata

// ============================================================================
// Device Helper Functions
// ============================================================================

template<typename T>
__device__ __host__ inline bool mightOverflowDouble(T value) {
    if (std::is_signed<T>::value) {
        return false;
    } else {
        const double MAX_SAFE = 9007199254740992.0; // 2^53
        return static_cast<double>(value) > MAX_SAFE;
    }
}

__device__ inline double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline long long warpReduceMax(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__device__ inline long long blockReduceMax(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);

    return val;
}

__device__ inline int computeBitsForValue(unsigned long long val) {
    if (val == 0) return 0;
    return 64 - __clzll(val);
}

// ============================================================================
// Stage 1: Delta-bits Computation Kernel
// ============================================================================

template<typename T>
__global__ void computeDeltaBitsKernel(
    const T* __restrict__ data,
    int data_size,
    int block_size,
    int* __restrict__ delta_bits_per_block,
    int num_blocks)
{
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    int start = bid * block_size;
    int end = min(start + block_size, data_size);
    int n = end - start;

    if (n <= 0) {
        if (threadIdx.x == 0) delta_bits_per_block[bid] = 0;
        return;
    }

    // Check for overflow values
    __shared__ int has_overflow;
    if (threadIdx.x == 0) has_overflow = 0;
    __syncthreads();

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (mightOverflowDouble(data[i])) {
            atomicExch(&has_overflow, 1);
            break;
        }
    }
    __syncthreads();

    if (has_overflow) {
        if (threadIdx.x == 0) {
            delta_bits_per_block[bid] = sizeof(T) * 8;  // Full width
        }
        return;
    }

    // Linear regression: y = theta0 + theta1 * x
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double x = static_cast<double>(i - start);
        double y = static_cast<double>(data[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    __shared__ double s_theta0, s_theta1;
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;
        if (fabs(det) > 1e-10) {
            s_theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            s_theta0 = (sum_y - s_theta1 * sum_x) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
    }
    __syncthreads();

    // Compute max error using INT finite diff (must match encoder/decoder EXACTLY)
    // Encoder uses stride=32 (WARP_SIZE), so we must simulate the same accumulation path.
    constexpr int ENCODER_STRIDE = 32;
    int lane = threadIdx.x % ENCODER_STRIDE;
    int iter_offset = threadIdx.x / ENCODER_STRIDE;

    double params_linear[2] = {s_theta0, s_theta1};
    int64_t y_int, step_int;
    FiniteDiff::computeLinearINT<T>(params_linear, lane, ENCODER_STRIDE, y_int, step_int);

    long long local_max_error = 0;

    int num_iter_groups = blockDim.x / ENCODER_STRIDE;
    if (num_iter_groups == 0) num_iter_groups = 1;
    int max_iters = (n + ENCODER_STRIDE - 1) / ENCODER_STRIDE;

    // Advance y_int to the starting iteration for this thread
    for (int skip = 0; skip < iter_offset && skip < max_iters; skip++) {
        y_int += step_int;
    }

    for (int iter = iter_offset; iter < max_iters; iter += num_iter_groups) {
        int local_idx = lane + iter * ENCODER_STRIDE;
        if (local_idx >= n) break;

        int global_idx = start + local_idx;
        T pred_val = static_cast<T>(y_int);
        long long delta;
        if (data[global_idx] >= pred_val) {
            delta = static_cast<long long>(data[global_idx] - pred_val);
        } else {
            delta = -static_cast<long long>(pred_val - data[global_idx]);
        }
        local_max_error = max(local_max_error, llabs(delta));

        // Advance by num_iter_groups iterations
        for (int adv = 0; adv < num_iter_groups && (iter + adv + 1) < max_iters; adv++) {
            y_int += step_int;
        }
    }

    long long max_error = blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        }
        delta_bits_per_block[bid] = bits;
    }
}

// ============================================================================
// Stage 2: Breakpoint Detection Kernel
// ============================================================================

__global__ void detectBreakpointsKernel(
    const int* __restrict__ delta_bits,
    int* __restrict__ is_breakpoint,
    int num_blocks,
    int threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks) return;

    if (i == 0) {
        is_breakpoint[i] = 1;  // First block is always a breakpoint
    } else {
        int diff = abs(delta_bits[i] - delta_bits[i - 1]);
        is_breakpoint[i] = (diff >= threshold) ? 1 : 0;
    }
}

// ============================================================================
// Stage 3: Partition Creation Kernels
// ============================================================================

__global__ void countPartitionsInSegmentsKernel(
    const int* __restrict__ breakpoint_positions,
    int num_breakpoints,
    int data_size,
    int min_partition_size,
    int max_partition_size,
    int* __restrict__ partition_counts)
{
    int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_idx >= num_breakpoints) return;

    int seg_start = breakpoint_positions[seg_idx];
    int seg_end = (seg_idx + 1 < num_breakpoints) ?
                  breakpoint_positions[seg_idx + 1] : data_size;
    int seg_len = seg_end - seg_start;

    if (seg_len <= 0) {
        partition_counts[seg_idx] = 0;
        return;
    }

    // Start from min_partition_size (warp-aligned)
    int part_size = ((min_partition_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Count partitions
    int count = 0;
    for (int pos = seg_start; pos < seg_end; pos += part_size) {
        count++;
    }

    partition_counts[seg_idx] = count;
}

__global__ void writePartitionsKernel(
    const int* __restrict__ breakpoint_positions,
    int num_breakpoints,
    int data_size,
    int min_partition_size,
    int max_partition_size,
    const int* __restrict__ partition_offsets,
    int* __restrict__ partition_starts,
    int* __restrict__ partition_ends)
{
    int seg_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg_idx >= num_breakpoints) return;

    int seg_start = breakpoint_positions[seg_idx];
    int seg_end = (seg_idx + 1 < num_breakpoints) ?
                  breakpoint_positions[seg_idx + 1] : data_size;
    int seg_len = seg_end - seg_start;

    if (seg_len <= 0) return;

    // Start from min_partition_size (warp-aligned)
    int part_size = ((min_partition_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    int write_pos = partition_offsets[seg_idx];
    int local_idx = 0;

    for (int pos = seg_start; pos < seg_end; pos += part_size) {
        partition_starts[write_pos + local_idx] = pos;
        partition_ends[write_pos + local_idx] = min(pos + part_size, seg_end);
        local_idx++;
    }
}

// ============================================================================
// Stage 4: Model Fitting Kernel (reused logic)
// ============================================================================

template<typename T>
__global__ void fitPartitionsKernel(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    float* __restrict__ costs,
    int num_partitions)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = partition_starts[pid];
    int end = partition_ends[pid];
    int n = end - start;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            costs[pid] = 0.0f;
        }
        return;
    }

    // Check for overflow
    __shared__ int has_overflow;
    if (threadIdx.x == 0) has_overflow = 0;
    __syncthreads();

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (mightOverflowDouble(data[i])) {
            atomicExch(&has_overflow, 1);
            break;
        }
    }
    __syncthreads();

    if (has_overflow) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            delta_bits_array[pid] = sizeof(T) * 8;
            max_errors[pid] = 0;
            costs[pid] = MODEL_OVERHEAD_BYTES + n * sizeof(T);
        }
        return;
    }

    // Linear regression
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double x = static_cast<double>(i - start);
        double y = static_cast<double>(data[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    __shared__ double s_theta0, s_theta1;
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;
        if (fabs(det) > 1e-10) {
            s_theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            s_theta0 = (sum_y - s_theta1 * sum_x) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
        model_types[pid] = MODEL_LINEAR;
        theta0_array[pid] = s_theta0;
        theta1_array[pid] = s_theta1;
    }
    __syncthreads();

    // Compute max error using INT finite diff (must match encoder/decoder EXACTLY)
    // Encoder uses stride=32 (WARP_SIZE), so we must simulate the same accumulation path.
    // Each thread simulates one encoder lane (tid % 32) and processes values at
    // local_idx = lane, lane+32, lane+64, ...
    constexpr int ENCODER_STRIDE = 32;
    int lane = threadIdx.x % ENCODER_STRIDE;  // Which encoder lane this thread simulates
    int iter_offset = threadIdx.x / ENCODER_STRIDE;  // Which iteration group

    double params_linear[2] = {s_theta0, s_theta1};
    int64_t y_int, step_int;
    FiniteDiff::computeLinearINT<T>(params_linear, lane, ENCODER_STRIDE, y_int, step_int);

    long long local_max_error = 0;

    // Process values exactly as encoder would: lane, lane+32, lane+64, ...
    // But distribute work across threads: thread tid handles iterations iter_offset, iter_offset + (blockDim/32), ...
    int num_iter_groups = blockDim.x / ENCODER_STRIDE;
    int max_iters = (n + ENCODER_STRIDE - 1) / ENCODER_STRIDE;  // Total iterations per lane

    // Advance y_int to the starting iteration for this thread
    for (int skip = 0; skip < iter_offset && skip < max_iters; skip++) {
        y_int += step_int;
    }

    for (int iter = iter_offset; iter < max_iters; iter += num_iter_groups) {
        int local_idx = lane + iter * ENCODER_STRIDE;
        if (local_idx >= n) break;

        int global_idx = start + local_idx;
        T pred_val = static_cast<T>(y_int);
        long long delta;
        if (data[global_idx] >= pred_val) {
            delta = static_cast<long long>(data[global_idx] - pred_val);
        } else {
            delta = -static_cast<long long>(pred_val - data[global_idx]);
        }
        local_max_error = max(local_max_error, llabs(delta));

        // Advance by num_iter_groups iterations
        for (int adv = 0; adv < num_iter_groups && (iter + adv + 1) < max_iters; adv++) {
            y_int += step_int;
        }
    }

    long long max_error = blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        max_errors[pid] = max_error;
        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        }
        delta_bits_array[pid] = bits;

        // Compute cost
        float delta_bytes = static_cast<float>(n) * bits / 8.0f;
        float original_bytes = static_cast<float>(n * sizeof(T));
        float compressed_cost = MODEL_OVERHEAD_BYTES + delta_bytes;
        costs[pid] = fminf(compressed_cost, original_bytes);
    }
}

// ============================================================================
// Stage 5: Merge Cost Evaluation Kernel
// ============================================================================

template<typename T>
__global__ void evaluateMergeCostKernel(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    const float* __restrict__ current_costs,
    float* __restrict__ merge_benefits,
    int* __restrict__ merged_delta_bits,
    double* __restrict__ merged_theta0,
    double* __restrict__ merged_theta1,
    int num_partitions,
    int max_partition_size)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions - 1) return;

    int start = partition_starts[pid];
    int end = partition_ends[pid + 1];
    int n = end - start;

    // Check if merged partition would be too large
    if (n > max_partition_size) {
        if (threadIdx.x == 0) {
            merge_benefits[pid] = -1.0f;  // Cannot merge
        }
        return;
    }

    // Fit model for merged partition
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double x = static_cast<double>(i - start);
        double y = static_cast<double>(data[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    __shared__ double s_theta0, s_theta1;
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;
        if (fabs(det) > 1e-10) {
            s_theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            s_theta0 = (sum_y - s_theta1 * sum_x) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
        merged_theta0[pid] = s_theta0;
        merged_theta1[pid] = s_theta1;
    }
    __syncthreads();

    // Compute max error for merged partition using INT finite diff (must match encoder)
    constexpr int ENCODER_STRIDE = 32;
    int lane = threadIdx.x % ENCODER_STRIDE;
    int iter_offset = threadIdx.x / ENCODER_STRIDE;

    double params_linear[2] = {s_theta0, s_theta1};
    int64_t y_int, step_int;
    FiniteDiff::computeLinearINT<T>(params_linear, lane, ENCODER_STRIDE, y_int, step_int);

    long long local_max_error = 0;

    int num_iter_groups = blockDim.x / ENCODER_STRIDE;
    if (num_iter_groups == 0) num_iter_groups = 1;
    int max_iters = (n + ENCODER_STRIDE - 1) / ENCODER_STRIDE;

    // Advance y_int to the starting iteration for this thread
    for (int skip = 0; skip < iter_offset && skip < max_iters; skip++) {
        y_int += step_int;
    }

    for (int iter = iter_offset; iter < max_iters; iter += num_iter_groups) {
        int local_idx = lane + iter * ENCODER_STRIDE;
        if (local_idx >= n) break;

        int global_idx = start + local_idx;
        T pred_val = static_cast<T>(y_int);
        long long delta;
        if (data[global_idx] >= pred_val) {
            delta = static_cast<long long>(data[global_idx] - pred_val);
        } else {
            delta = -static_cast<long long>(pred_val - data[global_idx]);
        }
        local_max_error = max(local_max_error, llabs(delta));

        // Advance by num_iter_groups iterations
        for (int adv = 0; adv < num_iter_groups && (iter + adv + 1) < max_iters; adv++) {
            y_int += step_int;
        }
    }

    long long max_error = blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        }
        merged_delta_bits[pid] = bits;

        // Compute merged cost
        float delta_bytes = static_cast<float>(n) * bits / 8.0f;
        float merged_cost = MODEL_OVERHEAD_BYTES + delta_bytes;

        // Compute benefit
        float separate_cost = current_costs[pid] + current_costs[pid + 1];
        float benefit = (separate_cost - merged_cost) / separate_cost;

        merge_benefits[pid] = benefit;
    }
}

// ============================================================================
// Stage 6: Parallel Odd-Even Merge Decision Kernel
// ============================================================================

__global__ void markMergesKernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,  // 1 = merge with next, 0 = keep separate
    int num_partitions,
    int phase,  // 0 = even pairs (0-1, 2-3, ...), 1 = odd pairs (1-2, 3-4, ...)
    float threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int pid = idx * 2 + phase;

    if (pid >= num_partitions - 1) return;

    // Check if this pair should merge
    if (merge_benefits[pid] >= threshold) {
        // Make sure neither partition is already marked for merging
        // Also check if pid is already absorbed by a previous merge (pid-1 has flag=1)
        bool pid_available = (merge_flags[pid] == 0);
        bool pid_not_absorbed = (pid == 0 || merge_flags[pid - 1] == 0);
        bool pid_plus1_available = (merge_flags[pid + 1] == 0);

        if (pid_available && pid_not_absorbed && pid_plus1_available) {
            merge_flags[pid] = 1;  // Mark for merge
        }
    }
}

// ============================================================================
// Cost-Optimal Partitioner Class
// ============================================================================

template<typename T>
class CostOptimalPartitioner {
private:
    T* d_data;
    const std::vector<T>& h_data_ref;
    int data_size;
    CostOptimalConfig config;
    cudaStream_t stream;

    // Refit a partition on host (for boundary adjustments)
    void refitPartition(PartitionInfo& info) {
        int start = info.start_idx;
        int end = info.end_idx;
        int n = end - start;

        if (n <= 0) return;

        // Linear regression
        double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
        for (int i = 0; i < n; i++) {
            double x = static_cast<double>(i);
            double y = static_cast<double>(h_data_ref[start + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }

        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;

        double theta0, theta1;
        if (std::fabs(det) > 1e-10) {
            theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            theta0 = (sum_y - theta1 * sum_x) / dn;
        } else {
            theta1 = 0.0;
            theta0 = sum_y / dn;
        }

        info.model_type = MODEL_LINEAR;
        info.model_params[0] = theta0;
        info.model_params[1] = theta1;

        // Compute max error using INT finite diff (matches GPU encoder/decoder EXACTLY)
        // Encoder uses stride=32 (WARP_SIZE), so we must simulate the same accumulation path.
        // Each encoder lane processes: lane, lane+32, lane+64, ...
        constexpr int ENCODER_STRIDE = 32;

        long long max_error = 0;
        for (int lane = 0; lane < ENCODER_STRIDE && lane < n; lane++) {
            // Initialize INT finite diff for this lane
            double y0_fp = theta0 + theta1 * lane;
            double y1_fp = theta0 + theta1 * (lane + ENCODER_STRIDE);
            int64_t y_int = std::llrint(y0_fp);
            int64_t step_int = std::llrint(y1_fp) - y_int;

            // Process all values for this lane: lane, lane+32, lane+64, ...
            for (int local_idx = lane; local_idx < n; local_idx += ENCODER_STRIDE) {
                T pred_val = static_cast<T>(y_int);
                long long delta;
                if (h_data_ref[start + local_idx] >= pred_val) {
                    delta = static_cast<long long>(h_data_ref[start + local_idx] - pred_val);
                } else {
                    delta = -static_cast<long long>(pred_val - h_data_ref[start + local_idx]);
                }
                max_error = std::max(max_error, std::llabs(delta));
                y_int += step_int;  // INT accumulation
            }
        }

        info.error_bound = max_error;
        int bits = 0;
        if (max_error > 0) {
            bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
        }
        info.delta_bits = bits;
    }

public:
    CostOptimalPartitioner(const std::vector<T>& data,
                           const CostOptimalConfig& cfg,
                           cudaStream_t cuda_stream = 0)
        : h_data_ref(data),
          data_size(data.size()),
          config(cfg),
          stream(cuda_stream)
    {
        cudaMalloc(&d_data, data_size * sizeof(T));
        cudaMemcpyAsync(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    ~CostOptimalPartitioner() {
        if (d_data) cudaFree(d_data);
    }

    std::vector<PartitionInfo> partition() {
        if (data_size == 0) return {};

        // ================================================================
        // Stage 1: Compute delta-bits for each analysis block
        // ================================================================
        int num_analysis_blocks = (data_size + config.analysis_block_size - 1) /
                                  config.analysis_block_size;

        int* d_delta_bits;
        cudaMalloc(&d_delta_bits, num_analysis_blocks * sizeof(int));

        int threads = 256;
        int blocks = num_analysis_blocks;
        computeDeltaBitsKernel<T><<<blocks, threads, 0, stream>>>(
            d_data, data_size, config.analysis_block_size,
            d_delta_bits, num_analysis_blocks);

        // ================================================================
        // Stage 2: Detect breakpoints
        // ================================================================
        int* d_is_breakpoint;
        cudaMalloc(&d_is_breakpoint, num_analysis_blocks * sizeof(int));

        blocks = (num_analysis_blocks + 255) / 256;
        detectBreakpointsKernel<<<blocks, 256, 0, stream>>>(
            d_delta_bits, d_is_breakpoint, num_analysis_blocks,
            config.breakpoint_threshold);

        // Compact breakpoints to get positions
        std::vector<int> h_is_breakpoint(num_analysis_blocks);
        cudaMemcpy(h_is_breakpoint.data(), d_is_breakpoint,
                   num_analysis_blocks * sizeof(int), cudaMemcpyDeviceToHost);

        std::vector<int> breakpoint_positions;
        for (int i = 0; i < num_analysis_blocks; i++) {
            if (h_is_breakpoint[i]) {
                breakpoint_positions.push_back(i * config.analysis_block_size);
            }
        }
        // Always include data end as implicit breakpoint
        int num_breakpoints = breakpoint_positions.size();

        if (num_breakpoints == 0) {
            breakpoint_positions.push_back(0);
            num_breakpoints = 1;
        }

        // ================================================================
        // Stage 3: Create partitions within segments
        // ================================================================
        int* d_breakpoint_positions;
        int* d_partition_counts;
        int* d_partition_offsets;

        cudaMalloc(&d_breakpoint_positions, num_breakpoints * sizeof(int));
        cudaMalloc(&d_partition_counts, num_breakpoints * sizeof(int));
        cudaMalloc(&d_partition_offsets, (num_breakpoints + 1) * sizeof(int));

        cudaMemcpy(d_breakpoint_positions, breakpoint_positions.data(),
                   num_breakpoints * sizeof(int), cudaMemcpyHostToDevice);

        blocks = (num_breakpoints + 255) / 256;
        countPartitionsInSegmentsKernel<<<blocks, 256, 0, stream>>>(
            d_breakpoint_positions, num_breakpoints, data_size,
            config.min_partition_size, config.max_partition_size, d_partition_counts);

        // Prefix sum to get offsets
        thrust::device_ptr<int> counts_ptr(d_partition_counts);
        thrust::device_ptr<int> offsets_ptr(d_partition_offsets);
        thrust::exclusive_scan(counts_ptr, counts_ptr + num_breakpoints, offsets_ptr);

        // Get total partition count
        int h_total_partitions;
        int h_last_count;
        cudaMemcpy(&h_total_partitions, d_partition_offsets + num_breakpoints - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_last_count, d_partition_counts + num_breakpoints - 1,
                   sizeof(int), cudaMemcpyDeviceToHost);
        h_total_partitions += h_last_count;

        if (h_total_partitions == 0) {
            // Fallback: single partition
            h_total_partitions = 1;
        }

        // Allocate partition arrays
        int* d_partition_starts;
        int* d_partition_ends;
        cudaMalloc(&d_partition_starts, h_total_partitions * sizeof(int));
        cudaMalloc(&d_partition_ends, h_total_partitions * sizeof(int));

        writePartitionsKernel<<<blocks, 256, 0, stream>>>(
            d_breakpoint_positions, num_breakpoints, data_size,
            config.min_partition_size, config.max_partition_size, d_partition_offsets,
            d_partition_starts, d_partition_ends);

        cudaStreamSynchronize(stream);

        // ================================================================
        // Stage 4: Fit models for all partitions
        // ================================================================
        int* d_model_types;
        double* d_theta0;
        double* d_theta1;
        int* d_delta_bits_parts;
        long long* d_max_errors;
        float* d_costs;

        cudaMalloc(&d_model_types, h_total_partitions * sizeof(int));
        cudaMalloc(&d_theta0, h_total_partitions * sizeof(double));
        cudaMalloc(&d_theta1, h_total_partitions * sizeof(double));
        cudaMalloc(&d_delta_bits_parts, h_total_partitions * sizeof(int));
        cudaMalloc(&d_max_errors, h_total_partitions * sizeof(long long));
        cudaMalloc(&d_costs, h_total_partitions * sizeof(float));

        fitPartitionsKernel<T><<<h_total_partitions, 256, 0, stream>>>(
            d_data, d_partition_starts, d_partition_ends,
            d_model_types, d_theta0, d_theta1, d_delta_bits_parts,
            d_max_errors, d_costs, h_total_partitions);

        // ================================================================
        // Stage 5: Cost-based merging (if enabled)
        // ================================================================
        if (config.enable_merging && h_total_partitions > 1) {
            float* d_merge_benefits;
            int* d_merged_delta_bits;
            double* d_merged_theta0;
            double* d_merged_theta1;
            int* d_merge_flags;

            cudaMalloc(&d_merge_benefits, h_total_partitions * sizeof(float));
            cudaMalloc(&d_merged_delta_bits, h_total_partitions * sizeof(int));
            cudaMalloc(&d_merged_theta0, h_total_partitions * sizeof(double));
            cudaMalloc(&d_merged_theta1, h_total_partitions * sizeof(double));
            cudaMalloc(&d_merge_flags, h_total_partitions * sizeof(int));

            for (int round = 0; round < config.max_merge_rounds; round++) {
                // Reset merge flags
                cudaMemset(d_merge_flags, 0, h_total_partitions * sizeof(int));

                // Evaluate merge costs
                evaluateMergeCostKernel<T><<<h_total_partitions, 256, 0, stream>>>(
                    d_data, d_partition_starts, d_partition_ends, d_costs,
                    d_merge_benefits, d_merged_delta_bits,
                    d_merged_theta0, d_merged_theta1,
                    h_total_partitions, config.max_partition_size);

                // Even phase
                blocks = (h_total_partitions / 2 + 255) / 256;
                if (blocks > 0) {
                    markMergesKernel<<<blocks, 256, 0, stream>>>(
                        d_merge_benefits, d_merge_flags, h_total_partitions,
                        0, config.merge_benefit_threshold);
                }

                // Odd phase
                if (blocks > 0) {
                    markMergesKernel<<<blocks, 256, 0, stream>>>(
                        d_merge_benefits, d_merge_flags, h_total_partitions,
                        1, config.merge_benefit_threshold);
                }

                cudaStreamSynchronize(stream);

                // Apply merges on host (simpler than GPU compaction for now)
                std::vector<int> h_merge_flags(h_total_partitions);
                std::vector<int> h_starts(h_total_partitions);
                std::vector<int> h_ends(h_total_partitions);
                std::vector<float> h_costs(h_total_partitions);
                std::vector<int> h_model_types(h_total_partitions);
                std::vector<double> h_theta0_arr(h_total_partitions);
                std::vector<double> h_theta1_arr(h_total_partitions);
                std::vector<int> h_delta_bits_arr(h_total_partitions);
                std::vector<long long> h_max_errors_arr(h_total_partitions);

                cudaMemcpy(h_merge_flags.data(), d_merge_flags,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_starts.data(), d_partition_starts,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_ends.data(), d_partition_ends,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_costs.data(), d_costs,
                           h_total_partitions * sizeof(float), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_model_types.data(), d_model_types,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_theta0_arr.data(), d_theta0,
                           h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_theta1_arr.data(), d_theta1,
                           h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_delta_bits_arr.data(), d_delta_bits_parts,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_max_errors_arr.data(), d_max_errors,
                           h_total_partitions * sizeof(long long), cudaMemcpyDeviceToHost);

                // Also get merged model parameters
                std::vector<double> h_merged_theta0(h_total_partitions);
                std::vector<double> h_merged_theta1(h_total_partitions);
                std::vector<int> h_merged_delta_bits(h_total_partitions);
                cudaMemcpy(h_merged_theta0.data(), d_merged_theta0,
                           h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_merged_theta1.data(), d_merged_theta1,
                           h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_merged_delta_bits.data(), d_merged_delta_bits,
                           h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);

                // Count merges
                int merge_count = 0;
                for (int i = 0; i < h_total_partitions; i++) {
                    if (h_merge_flags[i]) merge_count++;
                }

                if (merge_count == 0) break;  // No more merges possible

                // Apply merges - create new partition lists
                std::vector<int> new_starts, new_ends;
                std::vector<float> new_costs;
                std::vector<int> new_model_types;
                std::vector<double> new_theta0, new_theta1;
                std::vector<int> new_delta_bits;
                std::vector<long long> new_max_errors;

                for (int i = 0; i < h_total_partitions; i++) {
                    if (h_merge_flags[i] && i + 1 < h_total_partitions) {
                        // Merge i with i+1
                        new_starts.push_back(h_starts[i]);
                        new_ends.push_back(h_ends[i + 1]);
                        new_model_types.push_back(MODEL_LINEAR);
                        new_theta0.push_back(h_merged_theta0[i]);
                        new_theta1.push_back(h_merged_theta1[i]);
                        new_delta_bits.push_back(h_merged_delta_bits[i]);
                        // Recompute cost
                        int n = h_ends[i + 1] - h_starts[i];
                        float delta_bytes = static_cast<float>(n) * h_merged_delta_bits[i] / 8.0f;
                        new_costs.push_back(MODEL_OVERHEAD_BYTES + delta_bytes);
                        new_max_errors.push_back(0);  // Will be recomputed if needed
                        i++;  // Skip next partition (already merged)
                    } else if (i > 0 && h_merge_flags[i - 1]) {
                        // This partition was merged with previous, skip
                        continue;
                    } else {
                        // Keep as is
                        new_starts.push_back(h_starts[i]);
                        new_ends.push_back(h_ends[i]);
                        new_costs.push_back(h_costs[i]);
                        new_model_types.push_back(h_model_types[i]);
                        new_theta0.push_back(h_theta0_arr[i]);
                        new_theta1.push_back(h_theta1_arr[i]);
                        new_delta_bits.push_back(h_delta_bits_arr[i]);
                        new_max_errors.push_back(h_max_errors_arr[i]);
                    }
                }

                // Update partition count and arrays
                h_total_partitions = new_starts.size();

                if (h_total_partitions <= 1) break;

                // Copy back to device
                cudaMemcpy(d_partition_starts, new_starts.data(),
                           h_total_partitions * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_partition_ends, new_ends.data(),
                           h_total_partitions * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_costs, new_costs.data(),
                           h_total_partitions * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(d_model_types, new_model_types.data(),
                           h_total_partitions * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_theta0, new_theta0.data(),
                           h_total_partitions * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_theta1, new_theta1.data(),
                           h_total_partitions * sizeof(double), cudaMemcpyHostToDevice);
                cudaMemcpy(d_delta_bits_parts, new_delta_bits.data(),
                           h_total_partitions * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_max_errors, new_max_errors.data(),
                           h_total_partitions * sizeof(long long), cudaMemcpyHostToDevice);
            }

            cudaFree(d_merge_benefits);
            cudaFree(d_merged_delta_bits);
            cudaFree(d_merged_theta0);
            cudaFree(d_merged_theta1);
            cudaFree(d_merge_flags);
        }

        // ================================================================
        // Stage 6: Copy results back and create PartitionInfo vector
        // ================================================================
        std::vector<int> h_starts(h_total_partitions);
        std::vector<int> h_ends(h_total_partitions);
        std::vector<int> h_model_types(h_total_partitions);
        std::vector<double> h_theta0_arr(h_total_partitions);
        std::vector<double> h_theta1_arr(h_total_partitions);
        std::vector<int> h_delta_bits_arr(h_total_partitions);
        std::vector<long long> h_max_errors_arr(h_total_partitions);

        cudaMemcpy(h_starts.data(), d_partition_starts,
                   h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ends.data(), d_partition_ends,
                   h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_model_types.data(), d_model_types,
                   h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_theta0_arr.data(), d_theta0,
                   h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_theta1_arr.data(), d_theta1,
                   h_total_partitions * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta_bits_arr.data(), d_delta_bits_parts,
                   h_total_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max_errors_arr.data(), d_max_errors,
                   h_total_partitions * sizeof(long long), cudaMemcpyDeviceToHost);

        // Build result vector
        std::vector<PartitionInfo> result;
        result.reserve(h_total_partitions);

        for (int i = 0; i < h_total_partitions; i++) {
            PartitionInfo info;
            info.start_idx = h_starts[i];
            info.end_idx = h_ends[i];
            info.model_type = h_model_types[i];
            info.model_params[0] = h_theta0_arr[i];
            info.model_params[1] = h_theta1_arr[i];
            info.model_params[2] = 0.0;
            info.model_params[3] = 0.0;
            info.delta_bits = h_delta_bits_arr[i];
            info.delta_array_bit_offset = 0;
            info.error_bound = h_max_errors_arr[i];
            result.push_back(info);
        }

        // Sort by start index
        std::sort(result.begin(), result.end(),
                  [](const PartitionInfo& a, const PartitionInfo& b) {
                      return a.start_idx < b.start_idx;
                  });

        // Ensure complete coverage
        if (!result.empty()) {
            bool needs_refit_first = (result[0].start_idx != 0);
            result[0].start_idx = 0;

            bool needs_refit_last = (result.back().end_idx != data_size);
            result.back().end_idx = data_size;

            // Fix any gaps
            for (size_t i = 0; i < result.size() - 1; i++) {
                if (result[i].end_idx != result[i + 1].start_idx) {
                    result[i].end_idx = result[i + 1].start_idx;
                }
            }

            // Refit adjusted partitions
            if (needs_refit_first) {
                refitPartition(result[0]);
            }
            if (needs_refit_last) {
                refitPartition(result.back());
            }
        }

        // Cleanup
        cudaFree(d_delta_bits);
        cudaFree(d_is_breakpoint);
        cudaFree(d_breakpoint_positions);
        cudaFree(d_partition_counts);
        cudaFree(d_partition_offsets);
        cudaFree(d_partition_starts);
        cudaFree(d_partition_ends);
        cudaFree(d_model_types);
        cudaFree(d_theta0);
        cudaFree(d_theta1);
        cudaFree(d_delta_bits_parts);
        cudaFree(d_max_errors);
        cudaFree(d_costs);

        return result;
    }
};

// ============================================================================
// Public API
// ============================================================================

template<typename T>
std::vector<PartitionInfo> createPartitionsCostOptimal(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    int* num_partitions_out,
    cudaStream_t stream)
{
    CostOptimalPartitioner<T> partitioner(data, config, stream);
    std::vector<PartitionInfo> result = partitioner.partition();

    if (num_partitions_out) {
        *num_partitions_out = static_cast<int>(result.size());
    }

    return result;
}

// Explicit template instantiation
template class CostOptimalPartitioner<uint64_t>;
template class CostOptimalPartitioner<int64_t>;
template class CostOptimalPartitioner<uint32_t>;
template class CostOptimalPartitioner<int32_t>;

template std::vector<PartitionInfo> createPartitionsCostOptimal<uint64_t>(
    const std::vector<uint64_t>&, const CostOptimalConfig&, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostOptimal<int64_t>(
    const std::vector<int64_t>&, const CostOptimalConfig&, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostOptimal<uint32_t>(
    const std::vector<uint32_t>&, const CostOptimalConfig&, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostOptimal<int32_t>(
    const std::vector<int32_t>&, const CostOptimalConfig&, int*, cudaStream_t);
