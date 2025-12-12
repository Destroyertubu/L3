/**
 * GPU-Parallel Merge Implementation for Cost-Optimal Partitioning
 *
 * This file implements GPU-accelerated merge operations using stream compaction,
 * replacing the CPU-based merge loop in the original implementation.
 *
 * Key features:
 * 1. All merge operations run on GPU
 * 2. Uses thrust::exclusive_scan for stream compaction
 * 3. Double buffering minimizes memory allocation overhead
 * 4. Only partition count is transferred back to CPU
 *
 * Author: Claude Code Assistant
 * Date: 2025-12-06
 */

#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"

// ============================================================================
// Constants
// ============================================================================

#define WARP_SIZE 32
#define GPU_MERGE_BLOCK_SIZE 256

// Maximum integer that can be exactly represented as double (2^53)
// Values larger than this lose precision when converted to double,
// making linear/polynomial models unreliable. Force FOR+BitPack for such data.
constexpr uint64_t GPU_MERGE_DOUBLE_PRECISION_MAX = 9007199254740992ULL;  // 2^53

// ============================================================================
// Device Helper Functions (reused from encoder_cost_optimal.cu)
// ============================================================================

namespace gpu_merge {

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

// Kahan summation pair: (sum, compensation)
struct KahanPair {
    double sum;
    double comp;  // Compensation for lost low-order bits
};

__device__ inline KahanPair kahanAdd(KahanPair kp, double val) {
    double y = val - kp.comp;
    double t = kp.sum + y;
    kp.comp = (t - kp.sum) - y;
    kp.sum = t;
    return kp;
}

__device__ inline KahanPair kahanWarpReduceSum(KahanPair kp) {
    // Reduce both sum and compensation across warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_sum = __shfl_down_sync(0xffffffff, kp.sum, offset);
        double other_comp = __shfl_down_sync(0xffffffff, kp.comp, offset);
        // Combine using Kahan addition
        kp = kahanAdd(kp, other_sum);
        kp.comp += other_comp;  // Accumulate compensation
    }
    return kp;
}

__device__ inline double blockReduceSumKahan(double val) {
    __shared__ double shared_sum[32];
    __shared__ double shared_comp[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    // Start with Kahan pair
    KahanPair kp = {val, 0.0};
    kp = kahanWarpReduceSum(kp);

    if (lane == 0) {
        shared_sum[wid] = kp.sum;
        shared_comp[wid] = kp.comp;
    }
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> 5)) {
        kp.sum = shared_sum[lane];
        kp.comp = shared_comp[lane];
    } else {
        kp.sum = 0.0;
        kp.comp = 0.0;
    }

    if (wid == 0) {
        kp = kahanWarpReduceSum(kp);
    }

    // Return corrected sum
    return kp.sum + kp.comp;
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

// Unsigned versions for proper uint64_t handling (values > 2^63 corrupt when cast to signed)
__device__ inline unsigned long long warpReduceMaxUnsigned(unsigned long long val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

__device__ inline unsigned long long warpReduceMinUnsigned(unsigned long long val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other < val) ? other : val;
    }
    return val;
}

__device__ inline unsigned long long blockReduceMaxUnsigned(unsigned long long val) {
    __shared__ unsigned long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMaxUnsigned(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0ULL;
    if (wid == 0) val = warpReduceMaxUnsigned(val);
    return val;
}

__device__ inline unsigned long long blockReduceMinUnsigned(unsigned long long val) {
    __shared__ unsigned long long shared_min[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMinUnsigned(val);
    if (lane == 0) shared_min[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared_min[lane] : ULLONG_MAX;
    if (wid == 0) val = warpReduceMinUnsigned(val);
    return val;
}

} // namespace gpu_merge

// ============================================================================
// GPU Merge Context Implementation
// ============================================================================

template<typename T>
cudaError_t GPUMergeContext<T>::allocate(size_t max_partitions) {
    if (allocated && capacity >= max_partitions) return cudaSuccess;

    free();

    cudaError_t err;

    // Allocate double buffers
    err = buffer_A.allocate(max_partitions);
    if (err != cudaSuccess) return err;
    err = buffer_B.allocate(max_partitions);
    if (err != cudaSuccess) return err;

    current = &buffer_A;
    next = &buffer_B;

    // Allocate merge evaluation arrays
    err = cudaMalloc(&merge_benefits, max_partitions * sizeof(float));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_delta_bits, max_partitions * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta0, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta1, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merge_flags, max_partitions * sizeof(int));
    if (err != cudaSuccess) return err;

    // Allocate merged statistics arrays for O(1) merge propagation
    err = cudaMalloc(&merged_sum_x, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_y, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_xx, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_xy, max_partitions * sizeof(double));
    if (err != cudaSuccess) return err;

    // Allocate stream compaction arrays
    err = cudaMalloc(&output_slots, max_partitions * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&output_indices, max_partitions * sizeof(int));
    if (err != cudaSuccess) return err;
    err = cudaMalloc(&is_merge_base, max_partitions * sizeof(int));
    if (err != cudaSuccess) return err;

    capacity = max_partitions;
    allocated = true;
    return cudaSuccess;
}

template<typename T>
void GPUMergeContext<T>::free() {
    buffer_A.free();
    buffer_B.free();
    current = next = nullptr;

    if (merge_benefits) cudaFree(merge_benefits);
    if (merged_delta_bits) cudaFree(merged_delta_bits);
    if (merged_theta0) cudaFree(merged_theta0);
    if (merged_theta1) cudaFree(merged_theta1);
    if (merge_flags) cudaFree(merge_flags);
    if (merged_sum_x) cudaFree(merged_sum_x);
    if (merged_sum_y) cudaFree(merged_sum_y);
    if (merged_sum_xx) cudaFree(merged_sum_xx);
    if (merged_sum_xy) cudaFree(merged_sum_xy);
    if (output_slots) cudaFree(output_slots);
    if (output_indices) cudaFree(output_indices);
    if (is_merge_base) cudaFree(is_merge_base);
    if (d_temp_storage) cudaFree(d_temp_storage);

    merge_benefits = nullptr;
    merged_delta_bits = nullptr;
    merged_theta0 = merged_theta1 = nullptr;
    merge_flags = nullptr;
    merged_sum_x = merged_sum_y = merged_sum_xx = merged_sum_xy = nullptr;
    output_slots = output_indices = is_merge_base = nullptr;
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;

    capacity = 0;
    allocated = false;
}

// Explicit instantiation
template struct GPUMergeContext<int32_t>;
template struct GPUMergeContext<uint32_t>;
template struct GPUMergeContext<int64_t>;
template struct GPUMergeContext<uint64_t>;

// ============================================================================
// Stream Compaction Kernels
// ============================================================================

/**
 * Compute output slot for each partition
 * - Partition merged with previous: slot = 0 (skip)
 * - Partition that merges with next: slot = 1, is_merge_base = 1
 * - Partition kept as is: slot = 1, is_merge_base = 0
 */
__global__ void computeOutputSlotKernel(
    const int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    // Check if this partition was merged with the previous one
    bool prev_merged = (pid > 0) && (merge_flags[pid - 1] == 1);

    // Check if this partition merges with the next one
    bool curr_merges = (merge_flags[pid] == 1) && (pid + 1 < num_partitions);

    if (prev_merged) {
        // This partition was merged into the previous, skip it
        output_slots[pid] = 0;
        is_merge_base[pid] = 0;
    } else {
        // This partition contributes to output
        output_slots[pid] = 1;
        is_merge_base[pid] = curr_merges ? 1 : 0;
    }
}

/**
 * Apply merges in parallel using precomputed output indices.
 * OPTIMIZATION: Also propagates cached statistics for O(1) merge in next round.
 */
template<typename T>
__global__ void applyMergesKernel(
    // Input arrays (old partitions)
    const int* __restrict__ old_starts,
    const int* __restrict__ old_ends,
    const int* __restrict__ old_model_types,
    const double* __restrict__ old_theta0,
    const double* __restrict__ old_theta1,
    const int* __restrict__ old_delta_bits,
    const float* __restrict__ old_costs,
    const long long* __restrict__ old_max_errors,
    // Old cached statistics
    const double* __restrict__ old_sum_x,
    const double* __restrict__ old_sum_y,
    const double* __restrict__ old_sum_xx,
    const double* __restrict__ old_sum_xy,
    // Merge information
    const int* __restrict__ output_slots,
    const int* __restrict__ output_indices,
    const int* __restrict__ is_merge_base,
    const double* __restrict__ merged_theta0,
    const double* __restrict__ merged_theta1,
    const int* __restrict__ merged_delta_bits,
    // Merged statistics (precomputed during merge evaluation)
    const double* __restrict__ merged_sum_x,
    const double* __restrict__ merged_sum_y,
    const double* __restrict__ merged_sum_xx,
    const double* __restrict__ merged_sum_xy,
    // Output arrays (new partitions)
    int* __restrict__ new_starts,
    int* __restrict__ new_ends,
    int* __restrict__ new_model_types,
    double* __restrict__ new_theta0,
    double* __restrict__ new_theta1,
    int* __restrict__ new_delta_bits,
    float* __restrict__ new_costs,
    long long* __restrict__ new_max_errors,
    // New cached statistics
    double* __restrict__ new_sum_x,
    double* __restrict__ new_sum_y,
    double* __restrict__ new_sum_xx,
    double* __restrict__ new_sum_xy,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;

    // Skip partitions that don't contribute to output
    if (output_slots[pid] == 0) return;

    int out_idx = output_indices[pid];

    if (is_merge_base[pid]) {
        // Write merged partition
        new_starts[out_idx] = old_starts[pid];
        new_ends[out_idx] = old_ends[pid + 1];  // Use next partition's end
        new_model_types[out_idx] = MODEL_LINEAR;
        new_theta0[out_idx] = merged_theta0[pid];
        new_theta1[out_idx] = merged_theta1[pid];
        new_delta_bits[out_idx] = merged_delta_bits[pid];

        // Recompute cost for merged partition
        int n = old_ends[pid + 1] - old_starts[pid];
        float delta_bytes = static_cast<float>(n) * merged_delta_bits[pid] / 8.0f;
        new_costs[out_idx] = GPU_MERGE_MODEL_OVERHEAD_BYTES + delta_bytes;
        new_max_errors[out_idx] = 0;  // Will be recomputed if needed

        // Propagate merged statistics for next round
        new_sum_x[out_idx] = merged_sum_x[pid];
        new_sum_y[out_idx] = merged_sum_y[pid];
        new_sum_xx[out_idx] = merged_sum_xx[pid];
        new_sum_xy[out_idx] = merged_sum_xy[pid];
    } else {
        // Copy original partition
        new_starts[out_idx] = old_starts[pid];
        new_ends[out_idx] = old_ends[pid];
        new_model_types[out_idx] = old_model_types[pid];
        new_theta0[out_idx] = old_theta0[pid];
        new_theta1[out_idx] = old_theta1[pid];
        new_delta_bits[out_idx] = old_delta_bits[pid];
        new_costs[out_idx] = old_costs[pid];
        new_max_errors[out_idx] = old_max_errors[pid];

        // Copy original statistics
        new_sum_x[out_idx] = old_sum_x[pid];
        new_sum_y[out_idx] = old_sum_y[pid];
        new_sum_xx[out_idx] = old_sum_xx[pid];
        new_sum_xy[out_idx] = old_sum_xy[pid];
    }
}

// Note: applyMergesKernel explicit instantiations not needed since it's called from
// template functions that are already explicitly instantiated

/**
 * Count merge flags using atomic reduction
 */
__global__ void countMergeFlagsKernel(
    const int* __restrict__ merge_flags,
    int* __restrict__ count,
    int num_partitions)
{
    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int local_count = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_partitions;
         i += blockDim.x * gridDim.x) {
        if (merge_flags[i]) local_count++;
    }

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(&s_count, local_count);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, s_count);
    }
}

// ============================================================================
// Merge Evaluation and Marking Kernels (from encoder_cost_optimal.cu)
// ============================================================================

/**
 * OPTIMIZED: Evaluate merge cost using CACHED statistics for O(1) theta computation.
 *
 * Key insight: When merging partition A (n_a elements) with B (n_b elements):
 * - B's local indices shift by n_a after merge
 * - New statistics can be computed in O(1) from cached values:
 *   - sum_x_c  = sum_x_a + sum_x_b + n_a * n_b
 *   - sum_y_c  = sum_y_a + sum_y_b
 *   - sum_xx_c = sum_xx_a + sum_xx_b + 2*n_a*sum_x_b + n_a²*n_b
 *   - sum_xy_c = sum_xy_a + sum_xy_b + n_a*sum_y_b
 *
 * This reduces merge evaluation from O(2n) to O(n) per partition pair
 * (only one pass needed for max_error, theta is O(1)).
 */
template<typename T>
__global__ void gpuMergeEvaluateMergeCostKernel(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    const float* __restrict__ current_costs,
    // Cached statistics for O(1) theta computation
    const double* __restrict__ cached_sum_x,
    const double* __restrict__ cached_sum_y,
    const double* __restrict__ cached_sum_xx,
    const double* __restrict__ cached_sum_xy,
    // Output
    float* __restrict__ merge_benefits,
    int* __restrict__ merged_delta_bits,
    double* __restrict__ merged_theta0,
    double* __restrict__ merged_theta1,
    // Also output merged statistics for propagation
    double* __restrict__ merged_sum_x,
    double* __restrict__ merged_sum_y,
    double* __restrict__ merged_sum_xx,
    double* __restrict__ merged_sum_xy,
    int num_partitions,
    int max_partition_size)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions - 1) return;

    int start_a = partition_starts[pid];
    int end_a = partition_ends[pid];
    int start_b = partition_starts[pid + 1];
    int end_b = partition_ends[pid + 1];

    int n_a = end_a - start_a;
    int n_b = end_b - start_b;
    int n_c = n_a + n_b;

    // Check if merged partition would be too large
    if (n_c > max_partition_size) {
        if (threadIdx.x == 0) {
            merge_benefits[pid] = -1.0f;  // Cannot merge
        }
        return;
    }

    // O(1) computation of merged statistics using cached values
    __shared__ double s_theta0, s_theta1;
    __shared__ double s_sum_x, s_sum_y, s_sum_xx, s_sum_xy;

    if (threadIdx.x == 0) {
        // Load cached statistics
        double sum_x_a = cached_sum_x[pid];
        double sum_y_a = cached_sum_y[pid];
        double sum_xx_a = cached_sum_xx[pid];
        double sum_xy_a = cached_sum_xy[pid];

        double sum_x_b = cached_sum_x[pid + 1];
        double sum_y_b = cached_sum_y[pid + 1];
        double sum_xx_b = cached_sum_xx[pid + 1];
        double sum_xy_b = cached_sum_xy[pid + 1];

        // Compute merged statistics (B's indices shift by n_a)
        double dn_a = static_cast<double>(n_a);
        double dn_b = static_cast<double>(n_b);

        s_sum_x = sum_x_a + sum_x_b + dn_a * dn_b;
        s_sum_y = sum_y_a + sum_y_b;
        s_sum_xx = sum_xx_a + sum_xx_b + 2.0 * dn_a * sum_x_b + dn_a * dn_a * dn_b;
        s_sum_xy = sum_xy_a + sum_xy_b + dn_a * sum_y_b;

        // Compute theta from merged statistics
        double dn_c = static_cast<double>(n_c);
        double det = dn_c * s_sum_xx - s_sum_x * s_sum_x;
        if (fabs(det) > 1e-10) {
            s_theta1 = (dn_c * s_sum_xy - s_sum_x * s_sum_y) / det;
            s_theta0 = (s_sum_y - s_theta1 * s_sum_x) / dn_c;
        } else {
            s_theta1 = 0.0;
            s_theta0 = s_sum_y / dn_c;
        }

        merged_theta0[pid] = s_theta0;
        merged_theta1[pid] = s_theta1;

        // Save merged statistics for propagation
        merged_sum_x[pid] = s_sum_x;
        merged_sum_y[pid] = s_sum_y;
        merged_sum_xx[pid] = s_sum_xx;
        merged_sum_xy[pid] = s_sum_xy;
    }
    __syncthreads();

    // Single pass to compute max_error (this is the only data access needed)
    // Also track max value for precision check
    int merged_start = start_a;
    int merged_end = end_b;

    long long local_max_error = 0;
    unsigned long long local_max_ull = 0ULL;
    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        int local_idx = i - merged_start;
        T val = data[i];
        double predicted = s_theta0 + s_theta1 * local_idx;
        T pred_val = static_cast<T>(__double2ll_rn(predicted));
        long long delta;
        if (val >= pred_val) {
            delta = static_cast<long long>(val - pred_val);
        } else {
            delta = -static_cast<long long>(pred_val - val);
        }
        local_max_error = max(local_max_error, llabs(delta));

        // Track max value as unsigned for precision check
        unsigned long long val_ull = static_cast<unsigned long long>(val);
        local_max_ull = max(local_max_ull, val_ull);
    }

    long long max_error = gpu_merge::blockReduceMax(local_max_error);

    // Reduce max value for precision check
    unsigned long long global_max_ull = gpu_merge::blockReduceMaxUnsigned(local_max_ull);

    if (threadIdx.x == 0) {
        // CRITICAL: For uint64_t values > 2^53, double precision is insufficient
        // for accurate linear model predictions. Disallow merge in this case.
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (global_max_ull > GPU_MERGE_DOUBLE_PRECISION_MAX) {
                merge_benefits[pid] = -1.0f;  // Don't merge
                return;
            }
        }

        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = gpu_merge::computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        }
        merged_delta_bits[pid] = bits;

        // Compute merged cost
        float delta_bytes = static_cast<float>(n_c) * bits / 8.0f;
        float merged_cost = GPU_MERGE_MODEL_OVERHEAD_BYTES + delta_bytes;

        // Compute benefit: (separate_cost - merged_cost) / separate_cost
        float separate_cost = current_costs[pid] + current_costs[pid + 1];
        float benefit = (separate_cost - merged_cost) / separate_cost;

        merge_benefits[pid] = benefit;
    }
}

/**
 * Mark partitions for merging using Odd-Even strategy
 * NOTE: This intentionally matches the original CPU version's behavior,
 * which can mark overlapping merges (e.g., both 2->3 and 3->4).
 * The sequential apply loop and post-processing handle these edge cases.
 */
__global__ void gpuMergeMarkMergesKernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,
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
        // NOTE: This matches original CPU behavior and can create overlapping marks
        if (merge_flags[pid] == 0 && merge_flags[pid + 1] == 0) {
            merge_flags[pid] = 1;  // Mark for merge
        }
    }
}

// ============================================================================
// Delta-bits and Partition Creation Kernels
// ============================================================================

template<typename T>
__global__ void gpuMergeComputeDeltaBitsKernel(
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

    // Note: No overflow check here - matches CPU behavior which doesn't special-case
    // overflow in delta-bits analysis. This allows consistent partition boundaries.

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

    sum_x = gpu_merge::blockReduceSum(sum_x);
    sum_y = gpu_merge::blockReduceSum(sum_y);
    sum_xx = gpu_merge::blockReduceSum(sum_xx);
    sum_xy = gpu_merge::blockReduceSum(sum_xy);

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

    // Compute max error - handle uint64 overflow when diff > LLONG_MAX
    long long local_max_error = 0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double predicted = s_theta0 + s_theta1 * local_idx;
        T pred_val = static_cast<T>(__double2ll_rn(predicted));
        long long delta;
        if (data[i] >= pred_val) {
            unsigned long long diff = static_cast<unsigned long long>(data[i]) - static_cast<unsigned long long>(pred_val);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                local_max_error = LLONG_MAX;  // Overflow
            } else {
                delta = static_cast<long long>(diff);
                local_max_error = max(local_max_error, delta);
            }
        } else {
            unsigned long long diff = static_cast<unsigned long long>(pred_val) - static_cast<unsigned long long>(data[i]);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                local_max_error = LLONG_MAX;  // Overflow
            } else {
                delta = -static_cast<long long>(diff);
                local_max_error = max(local_max_error, llabs(delta));
            }
        }
    }

    long long max_error = gpu_merge::blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = gpu_merge::computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        }
        delta_bits_per_block[bid] = bits;
    }
}

__global__ void gpuMergeDetectBreakpointsKernel(
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

/**
 * Count partitions within each segment between breakpoints
 */
__global__ void gpuMergeCountPartitionsInSegmentsKernel(
    const int* __restrict__ breakpoint_positions,
    int num_breakpoints,
    int data_size,
    int target_partition_size,
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

    // Determine partition size for this segment
    int part_size = target_partition_size;

    // Adjust to fit segment better
    int num_parts = (seg_len + part_size - 1) / part_size;
    if (num_parts > 0) {
        // Try to make partitions more even
        part_size = (seg_len + num_parts - 1) / num_parts;
        // Align to warp size
        part_size = ((part_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        // Clamp to bounds
        part_size = max(min_partition_size, min(max_partition_size, part_size));
    }

    // Count partitions
    int count = 0;
    for (int pos = seg_start; pos < seg_end; pos += part_size) {
        count++;
    }

    partition_counts[seg_idx] = count;
}

/**
 * Write partition boundaries for all segments
 */
__global__ void gpuMergeWritePartitionsKernel(
    const int* __restrict__ breakpoint_positions,
    int num_breakpoints,
    int data_size,
    int target_partition_size,
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

    // Same logic as counting kernel
    int part_size = target_partition_size;
    int num_parts = (seg_len + part_size - 1) / part_size;
    if (num_parts > 0) {
        part_size = (seg_len + num_parts - 1) / num_parts;
        part_size = ((part_size + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        part_size = max(min_partition_size, min(max_partition_size, part_size));
    }

    int write_pos = partition_offsets[seg_idx];
    int local_idx = 0;

    for (int pos = seg_start; pos < seg_end; pos += part_size) {
        partition_starts[write_pos + local_idx] = pos;
        partition_ends[write_pos + local_idx] = min(pos + part_size, seg_end);
        local_idx++;
    }
}

/**
 * Fit linear model for each partition and CACHE statistics for O(1) merge evaluation.
 *
 * OPTIMIZATION: Saves sum_x, sum_y, sum_xx, sum_xy so that merge evaluation
 * can compute merged theta without re-scanning data.
 */
template<typename T>
__global__ void gpuMergeFitPartitionsKernel(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    double* __restrict__ theta2_array,
    double* __restrict__ theta3_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    float* __restrict__ costs,
    // Cached statistics for O(1) merge
    double* __restrict__ cached_sum_x,
    double* __restrict__ cached_sum_y,
    double* __restrict__ cached_sum_xx,
    double* __restrict__ cached_sum_xy,
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
            theta2_array[pid] = 0.0;  // Initialize polynomial coefficients
            theta3_array[pid] = 0.0;
            cached_sum_x[pid] = 0.0;
            cached_sum_y[pid] = 0.0;
            cached_sum_xx[pid] = 0.0;
            cached_sum_xy[pid] = 0.0;
        }
        return;
    }

    // ========== PASS 1: Compute y_mean for numerical stability ==========
    // For large uint64_t values (e.g., 10^15), directly accumulating sum_y
    // causes precision loss. Using centered y values improves stability.
    // Also track min/max for FOR+BitPack model as fallback for large values.

    double local_sum_y_pass1 = 0.0;

    // Track min/max for FOR model (needed when values > 2^53)
    unsigned long long local_min_ull, local_max_ull;
    if constexpr (sizeof(T) == 8) {
        local_min_ull = ULLONG_MAX;
        local_max_ull = 0ULL;
    } else {
        local_min_ull = 0xFFFFFFFFULL;
        local_max_ull = 0ULL;
    }

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        double y = static_cast<double>(val);
        local_sum_y_pass1 += y;

        // Track min/max as unsigned
        unsigned long long val_ull = static_cast<unsigned long long>(val);
        local_min_ull = min(local_min_ull, val_ull);
        local_max_ull = max(local_max_ull, val_ull);
    }
    double sum_y_pass1 = gpu_merge::blockReduceSum(local_sum_y_pass1);

    // Reduce min/max
    unsigned long long global_min_ull = gpu_merge::blockReduceMinUnsigned(local_min_ull);
    __syncthreads();
    unsigned long long global_max_ull = gpu_merge::blockReduceMaxUnsigned(local_max_ull);

    __shared__ double s_y_mean;
    __shared__ unsigned long long s_global_min, s_global_max;
    __shared__ bool s_force_for_bitpack;
    if (threadIdx.x == 0) {
        s_y_mean = sum_y_pass1 / static_cast<double>(n);
        s_global_min = global_min_ull;
        s_global_max = global_max_ull;

        // CRITICAL: For uint64_t values > 2^53, double precision is insufficient
        // for accurate linear model predictions. Force FOR+BitPack in this case.
        s_force_for_bitpack = false;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (s_global_max > GPU_MERGE_DOUBLE_PRECISION_MAX) {
                s_force_for_bitpack = true;
            }
        }
    }
    __syncthreads();
    double y_mean = s_y_mean;

    // If we need to force FOR+BitPack, skip the expensive linear fitting
    if (s_force_for_bitpack) {
        if (threadIdx.x == 0) {
            // Compute FOR+BitPack model
            unsigned long long range = s_global_max - s_global_min;
            int for_bits = (range > 0) ? gpu_merge::computeBitsForValue(range) : 0;

            model_types[pid] = MODEL_FOR_BITPACK;
            // Store base value using bit-pattern copy for 64-bit types
            theta0_array[pid] = __longlong_as_double(static_cast<long long>(s_global_min));
            theta1_array[pid] = 0.0;
            theta2_array[pid] = 0.0;
            theta3_array[pid] = 0.0;
            max_errors[pid] = 0;
            delta_bits_array[pid] = for_bits;

            float delta_bytes = static_cast<float>(n) * for_bits / 8.0f;
            costs[pid] = static_cast<float>(sizeof(T)) + delta_bytes;  // FOR overhead = sizeof(T)

            // Still cache statistics for merge evaluation (won't be used but needed for compatibility)
            cached_sum_x[pid] = 0.0;
            cached_sum_y[pid] = sum_y_pass1;
            cached_sum_xx[pid] = 0.0;
            cached_sum_xy[pid] = 0.0;
        }
        return;
    }

    // ========== PASS 2: Compute centered statistics ==========
    // y_centered = y - y_mean, which has much smaller magnitude
    double sum_x = 0.0, sum_y_centered = 0.0, sum_xx = 0.0, sum_xy_centered = 0.0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double x = static_cast<double>(i - start);
        double y_centered = static_cast<double>(data[i]) - y_mean;
        sum_x += x;
        sum_y_centered += y_centered;  // Should be ~0
        sum_xx += x * x;
        sum_xy_centered += x * y_centered;
    }

    sum_x = gpu_merge::blockReduceSum(sum_x);
    sum_y_centered = gpu_merge::blockReduceSum(sum_y_centered);
    sum_xx = gpu_merge::blockReduceSum(sum_xx);
    sum_xy_centered = gpu_merge::blockReduceSum(sum_xy_centered);

    __shared__ double s_theta0, s_theta1, s_y_mean_final, s_x_mean_final;
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double x_mean = sum_x / dn;

        // Compute theta1 using centered statistics
        // sum_x2_centered = sum((x - x_mean)^2) = sum_xx - n * x_mean^2
        double sum_x2_centered = sum_xx - dn * x_mean * x_mean;
        // sum_xy_centered already uses centered y
        // Adjust for centered x: cov(x,y) = sum_xy_c - x_mean * sum_y_c
        double cov_xy = sum_xy_centered - x_mean * sum_y_centered;

        if (fabs(sum_x2_centered) > 1e-10) {
            s_theta1 = cov_xy / sum_x2_centered;
        } else {
            s_theta1 = 0.0;
        }
        // CRITICAL FIX: Avoid precision loss when computing theta0
        // The direct formula theta0 = y_mean - theta1 * x_mean causes
        // catastrophic cancellation when both terms are large (~10^13) but nearly equal.
        // Solution: Store y_mean and x_mean for centered prediction in max_error calc.
        // Final theta0 for storage will be computed using first data point for better precision.
        s_y_mean_final = y_mean;
        s_x_mean_final = x_mean;

        // Cache ORIGINAL (non-centered) statistics for O(1) merge evaluation
        // These are needed for merge cost computation compatibility
        double sum_y_original = sum_y_pass1;
        double sum_xy_original = sum_xy_centered + x_mean * sum_y_centered + sum_x * y_mean;
        // sum_xy_original = sum(x * y) = sum(x * (y_c + y_mean)) = sum_xy_c + y_mean * sum_x
        // But we need the actual sum_xy, let's recompute correctly:
        // sum_xy_centered = sum(x * y_centered) = sum(x*y) - y_mean*sum_x
        // So: sum_xy_original = sum_xy_centered + y_mean * sum_x
        sum_xy_original = sum_xy_centered + y_mean * sum_x;

        cached_sum_x[pid] = sum_x;
        cached_sum_y[pid] = sum_y_original;
        cached_sum_xx[pid] = sum_xx;
        cached_sum_xy[pid] = sum_xy_original;
    }
    __syncthreads();

    // CRITICAL FIX: Compute max_error using ANCHORED prediction (theta0 = data[start])
    // This MUST match what the encoder will use, not the centered prediction!
    // The encoder uses: pred = theta0 + theta1 * x where theta0 = data[start]
    // Centered prediction would give different residuals, causing verification failures.
    double anchored_theta0 = static_cast<double>(data[start]);

    long long local_max_error = 0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        // Use ANCHORED prediction: matches encoder exactly
        double x = static_cast<double>(local_idx);
        double predicted = anchored_theta0 + s_theta1 * x;
        T pred_val;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            pred_val = static_cast<T>(__double2ull_rn(predicted));
        } else {
            pred_val = static_cast<T>(__double2ll_rn(predicted));
        }
        long long delta;
        if (data[i] >= pred_val) {
            unsigned long long diff = static_cast<unsigned long long>(data[i]) - static_cast<unsigned long long>(pred_val);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                local_max_error = LLONG_MAX;  // Overflow - force DIRECT_COPY
            } else {
                delta = static_cast<long long>(diff);
                local_max_error = max(local_max_error, delta);
            }
        } else {
            unsigned long long diff = static_cast<unsigned long long>(pred_val) - static_cast<unsigned long long>(data[i]);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                local_max_error = LLONG_MAX;  // Overflow - force DIRECT_COPY
            } else {
                delta = -static_cast<long long>(diff);
                local_max_error = max(local_max_error, llabs(delta));
            }
        }
    }

    long long max_error = gpu_merge::blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        // Store anchored theta0
        s_theta0 = anchored_theta0;

        // If max_error = LLONG_MAX, delta overflows - force DIRECT_COPY
        if (max_error == LLONG_MAX) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            theta2_array[pid] = 0.0;
            theta3_array[pid] = 0.0;
            max_errors[pid] = 0;
            delta_bits_array[pid] = sizeof(T) * 8;
            costs[pid] = static_cast<float>(n * sizeof(T));
        } else {
            model_types[pid] = MODEL_LINEAR;
            theta0_array[pid] = s_theta0;
            theta1_array[pid] = s_theta1;
            theta2_array[pid] = 0.0;  // Initialize polynomial coefficients to 0 for LINEAR model
            theta3_array[pid] = 0.0;
            max_errors[pid] = max_error;

            int bits = 0;
            if (max_error > 0) {
                // +2 for sign bit + safety margin for floating-point rounding
                bits = gpu_merge::computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
            }
            delta_bits_array[pid] = bits;

            // Compute cost
            float delta_bytes = static_cast<float>(n) * bits / 8.0f;
            float original_bytes = static_cast<float>(n * sizeof(T));
            float compressed_cost = GPU_MERGE_MODEL_OVERHEAD_BYTES + delta_bytes;
            costs[pid] = fminf(compressed_cost, original_bytes);
        }
    }
}

/**
 * Polynomial refit kernel - upgrades partitions from LINEAR to POLY2/POLY3 if beneficial.
 *
 * This kernel runs AFTER initial LINEAR fitting and evaluates whether polynomial
 * models provide better compression. It only modifies model_types and theta arrays,
 * preserving the cached LINEAR statistics for merge operations.
 *
 * Cost model:
 * - LINEAR: 16 bytes overhead + (n * bits / 8)
 * - POLY2:  24 bytes overhead + (n * bits / 8)
 * - POLY3:  32 bytes overhead + (n * bits / 8)
 */
template<typename T>
__global__ void gpuMergeRefitPolynomialKernel(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    double* __restrict__ theta2_array,
    double* __restrict__ theta3_array,
    int* __restrict__ delta_bits_array,
    float* __restrict__ costs,
    int num_partitions,
    int poly_min_size,      // Minimum size for POLY2
    int cubic_min_size,     // Minimum size for POLY3
    float cost_threshold)   // Require this much improvement (e.g., 0.95 = 5% better)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = partition_starts[pid];
    int end = partition_ends[pid];
    int n = end - start;

    // Skip small partitions
    if (n < poly_min_size) return;

    // Compute statistics for polynomial fitting
    double local_sum_y = 0.0, local_sum_xy = 0.0;
    double local_sum_x2y = 0.0, local_sum_x3y = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);
        double x2 = x * x;
        double x3 = x2 * x;
        double y = static_cast<double>(data[i]);

        local_sum_y += y;
        local_sum_xy += x * y;
        local_sum_x2y += x2 * y;
        local_sum_x3y += x3 * y;
    }

    double sum_y = gpu_merge::blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xy = gpu_merge::blockReduceSum(local_sum_xy);
    __syncthreads();
    double sum_x2y = gpu_merge::blockReduceSum(local_sum_x2y);
    __syncthreads();
    double sum_x3y = gpu_merge::blockReduceSum(local_sum_x3y);
    __syncthreads();

    // Shared memory for model parameters
    __shared__ double s_linear_params[4];
    __shared__ double s_poly2_params[4];
    __shared__ double s_poly3_params[4];
    __shared__ float s_linear_cost, s_poly2_cost, s_poly3_cost;
    __shared__ int s_linear_bits, s_poly2_bits, s_poly3_bits;

    // Thread 0 computes model coefficients
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);

        // Sum formulas for x^k
        double sx = dn * (dn - 1.0) / 2.0;
        double sx2 = dn * (dn - 1.0) * (2.0 * dn - 1.0) / 6.0;
        double sx3 = sx * sx;
        double sx4 = dn * (dn - 1.0) * (2.0 * dn - 1.0) * (3.0 * dn * dn - 3.0 * dn - 1.0) / 30.0;

        // LINEAR fit
        double det = dn * sx2 - sx * sx;
        if (fabs(det) > 1e-10) {
            s_linear_params[1] = (dn * sum_xy - sx * sum_y) / det;
            s_linear_params[0] = (sum_y - s_linear_params[1] * sx) / dn;
        } else {
            s_linear_params[1] = 0.0;
            s_linear_params[0] = sum_y / dn;
        }
        s_linear_params[2] = 0.0;
        s_linear_params[3] = 0.0;

        // POLY2 fit (if large enough)
        if (n >= poly_min_size) {
            // 3x3 system for quadratic
            double a00 = dn, a01 = sx, a02 = sx2;
            double a10 = sx, a11 = sx2, a12 = sx3;
            double a20 = sx2, a21 = sx3, a22 = sx4;
            double b0 = sum_y, b1 = sum_xy, b2 = sum_x2y;

            // Simple Gaussian elimination
            double det2 = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
            if (fabs(det2) > 1e-10) {
                s_poly2_params[0] = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det2;
                s_poly2_params[1] = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det2;
                s_poly2_params[2] = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det2;
            } else {
                s_poly2_params[0] = s_linear_params[0];
                s_poly2_params[1] = s_linear_params[1];
                s_poly2_params[2] = 0.0;
            }
            s_poly2_params[3] = 0.0;
        } else {
            for (int i = 0; i < 4; i++) s_poly2_params[i] = s_linear_params[i];
        }

        // POLY3 fit using 4x4 Gaussian elimination (cubic polynomial: a + bx + cx^2 + dx^3)
        if (n >= cubic_min_size) {
            // Build the normal equations matrix for least squares cubic fit
            // [n,    sx,   sx2,  sx3 ] [a]   [sum_y  ]
            // [sx,   sx2,  sx3,  sx4 ] [b] = [sum_xy ]
            // [sx2,  sx3,  sx4,  sx5 ] [c]   [sum_x2y]
            // [sx3,  sx4,  sx5,  sx6 ] [d]   [sum_x3y]

            // Compute higher order sums using Faulhaber's formulas
            // sx5 = sum(i^5) for i=0..n-1
            // sx6 = sum(i^6) for i=0..n-1
            double n2 = dn * dn;
            double n3 = n2 * dn;
            double n4 = n3 * dn;

            // Faulhaber formula for sum of 5th powers: n^2(n-1)^2(2n^2-2n-1)/12
            double nm1 = dn - 1.0;
            double sx5 = n2 * nm1 * nm1 * (2.0 * n2 - 2.0 * dn - 1.0) / 12.0;

            // Faulhaber formula for sum of 6th powers: n(n-1)(2n-1)(3n^4-6n^3-n^2+4n+2)/42
            double sx6 = dn * nm1 * (2.0 * dn - 1.0) * (3.0 * n4 - 6.0 * n3 - n2 + 4.0 * dn + 2.0) / 42.0;

            // 4x4 augmented matrix for Gaussian elimination
            double A[4][5] = {
                {dn,  sx,  sx2, sx3, sum_y},
                {sx,  sx2, sx3, sx4, sum_xy},
                {sx2, sx3, sx4, sx5, sum_x2y},
                {sx3, sx4, sx5, sx6, sum_x3y}
            };

            // Gaussian elimination with partial pivoting
            bool singular = false;
            for (int col = 0; col < 4 && !singular; col++) {
                // Find pivot
                int max_row = col;
                double max_val = fabs(A[col][col]);
                for (int row = col + 1; row < 4; row++) {
                    if (fabs(A[row][col]) > max_val) {
                        max_val = fabs(A[row][col]);
                        max_row = row;
                    }
                }

                if (max_val < 1e-12) {
                    singular = true;
                    break;
                }

                // Swap rows
                if (max_row != col) {
                    for (int k = 0; k < 5; k++) {
                        double tmp = A[col][k];
                        A[col][k] = A[max_row][k];
                        A[max_row][k] = tmp;
                    }
                }

                // Eliminate
                for (int row = col + 1; row < 4; row++) {
                    double factor = A[row][col] / A[col][col];
                    for (int k = col; k < 5; k++) {
                        A[row][k] -= factor * A[col][k];
                    }
                }
            }

            if (!singular && fabs(A[3][3]) > 1e-12) {
                // Back substitution
                double sol[4];
                for (int i = 3; i >= 0; i--) {
                    sol[i] = A[i][4];
                    for (int j = i + 1; j < 4; j++) {
                        sol[i] -= A[i][j] * sol[j];
                    }
                    sol[i] /= A[i][i];
                }

                s_poly3_params[0] = sol[0];  // a (intercept)
                s_poly3_params[1] = sol[1];  // b (linear coef)
                s_poly3_params[2] = sol[2];  // c (quadratic coef)
                s_poly3_params[3] = sol[3];  // d (cubic coef)
            } else {
                // Fall back to POLY2 if singular
                s_poly3_params[0] = s_poly2_params[0];
                s_poly3_params[1] = s_poly2_params[1];
                s_poly3_params[2] = s_poly2_params[2];
                s_poly3_params[3] = 0.0;
            }
        } else {
            for (int i = 0; i < 4; i++) s_poly3_params[i] = s_poly2_params[i];
        }
    }
    __syncthreads();

    // Compute max errors for all models
    long long linear_max_err = 0, poly2_max_err = 0, poly3_max_err = 0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);
        T val = data[i];

        // LINEAR prediction
        double pred_linear = s_linear_params[0] + s_linear_params[1] * x;
        T pv_linear = static_cast<T>(__double2ll_rn(pred_linear));
        long long err_linear = (val >= pv_linear) ? (long long)(val - pv_linear) : -(long long)(pv_linear - val);
        linear_max_err = max(linear_max_err, llabs(err_linear));

        // POLY2 prediction (Horner's method)
        double pred_poly2 = s_poly2_params[0] + x * (s_poly2_params[1] + x * s_poly2_params[2]);
        T pv_poly2 = static_cast<T>(__double2ll_rn(pred_poly2));
        long long err_poly2 = (val >= pv_poly2) ? (long long)(val - pv_poly2) : -(long long)(pv_poly2 - val);
        poly2_max_err = max(poly2_max_err, llabs(err_poly2));

        // POLY3 prediction (Horner's method)
        double pred_poly3 = s_poly3_params[0] + x * (s_poly3_params[1] + x * (s_poly3_params[2] + x * s_poly3_params[3]));
        T pv_poly3 = static_cast<T>(__double2ll_rn(pred_poly3));
        long long err_poly3 = (val >= pv_poly3) ? (long long)(val - pv_poly3) : -(long long)(pv_poly3 - val);
        poly3_max_err = max(poly3_max_err, llabs(err_poly3));
    }

    linear_max_err = gpu_merge::blockReduceMax(linear_max_err);
    __syncthreads();
    poly2_max_err = gpu_merge::blockReduceMax(poly2_max_err);
    __syncthreads();
    poly3_max_err = gpu_merge::blockReduceMax(poly3_max_err);
    __syncthreads();

    // Thread 0 selects best model
    if (threadIdx.x == 0) {
        // Compute bits for each model (+2 for sign bit + safety margin for floating-point rounding)
        s_linear_bits = (linear_max_err > 0) ? gpu_merge::computeBitsForValue((unsigned long long)linear_max_err) + 2 : 0;
        s_poly2_bits = (poly2_max_err > 0) ? gpu_merge::computeBitsForValue((unsigned long long)poly2_max_err) + 2 : 0;
        s_poly3_bits = (poly3_max_err > 0) ? gpu_merge::computeBitsForValue((unsigned long long)poly3_max_err) + 2 : 0;

        // Compute costs (overhead + delta bytes)
        float fn = static_cast<float>(n);
        s_linear_cost = 16.0f + fn * s_linear_bits / 8.0f;  // LINEAR: 16 bytes overhead
        s_poly2_cost = 24.0f + fn * s_poly2_bits / 8.0f;    // POLY2: 24 bytes overhead
        s_poly3_cost = 32.0f + fn * s_poly3_bits / 8.0f;    // POLY3: 32 bytes overhead

        // Select best model (require cost_threshold improvement)
        int best_model = MODEL_LINEAR;
        float best_cost = s_linear_cost;
        double best_params[4] = {s_linear_params[0], s_linear_params[1], 0.0, 0.0};
        int best_bits = s_linear_bits;

        if (n >= poly_min_size && s_poly2_cost < best_cost * cost_threshold) {
            best_model = MODEL_POLYNOMIAL2;
            best_cost = s_poly2_cost;
            best_params[0] = s_poly2_params[0];
            best_params[1] = s_poly2_params[1];
            best_params[2] = s_poly2_params[2];
            best_params[3] = 0.0;
            best_bits = s_poly2_bits;
        }

        // POLY3 selection - enabled now with proper 4x4 solver
        if (n >= cubic_min_size && s_poly3_cost < best_cost * cost_threshold) {
            best_model = MODEL_POLYNOMIAL3;
            best_cost = s_poly3_cost;
            best_params[0] = s_poly3_params[0];
            best_params[1] = s_poly3_params[1];
            best_params[2] = s_poly3_params[2];
            best_params[3] = s_poly3_params[3];
            best_bits = s_poly3_bits;
        }

        // Update partition info if polynomial is better
        if (best_model != MODEL_LINEAR) {
            model_types[pid] = best_model;
            theta0_array[pid] = best_params[0];
            theta1_array[pid] = best_params[1];
            theta2_array[pid] = best_params[2];
            theta3_array[pid] = best_params[3];
            delta_bits_array[pid] = best_bits;
            costs[pid] = best_cost;
        }
    }
}

// ============================================================================
// GPUCostOptimalPartitioner Implementation
// ============================================================================

template<typename T>
GPUCostOptimalPartitioner<T>::GPUCostOptimalPartitioner(
    const std::vector<T>& data,
    const CostOptimalConfig& cfg,
    cudaStream_t cuda_stream)
    : h_data_ref(data),
      data_size(data.size()),
      config(cfg),
      stream(cuda_stream)
{
    cudaMalloc(&d_data, data_size * sizeof(T));
    cudaMemcpy(d_data, data.data(), data_size * sizeof(T), cudaMemcpyHostToDevice);

    // Allocate context with maximum possible partitions
    size_t max_partitions = (data_size + config.min_partition_size - 1) / config.min_partition_size;
    ctx.allocate(max_partitions);
}

template<typename T>
GPUCostOptimalPartitioner<T>::~GPUCostOptimalPartitioner() {
    if (d_data) cudaFree(d_data);
}

template<typename T>
void GPUCostOptimalPartitioner<T>::refitPartition(PartitionInfo& info) {
    int start = info.start_idx;
    int end = info.end_idx;
    int n = end - start;

    if (n <= 0) return;

    // Linear regression on host
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

    // Compute max error - handle uint64 overflow when diff > LLONG_MAX
    long long max_error = 0;
    bool overflow = false;
    for (int i = 0; i < n; i++) {
        double predicted = theta0 + theta1 * i;
        T pred_val = static_cast<T>(std::llrint(predicted));
        if (h_data_ref[start + i] >= pred_val) {
            unsigned long long diff = static_cast<unsigned long long>(h_data_ref[start + i]) - static_cast<unsigned long long>(pred_val);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                overflow = true;
                max_error = LLONG_MAX;
            } else {
                max_error = std::max(max_error, static_cast<long long>(diff));
            }
        } else {
            unsigned long long diff = static_cast<unsigned long long>(pred_val) - static_cast<unsigned long long>(h_data_ref[start + i]);
            if (diff > static_cast<unsigned long long>(LLONG_MAX)) {
                overflow = true;
                max_error = LLONG_MAX;
            } else {
                max_error = std::max(max_error, static_cast<long long>(diff));
            }
        }
    }

    // If overflow, force DIRECT_COPY
    if (overflow) {
        info.model_type = MODEL_DIRECT_COPY;
        info.model_params[0] = 0.0;
        info.model_params[1] = 0.0;
        info.model_params[2] = 0.0;
        info.model_params[3] = 0.0;
        info.error_bound = 0;
        info.delta_bits = sizeof(T) * 8;
        return;
    }

    info.error_bound = max_error;
    int bits = 0;
    if (max_error > 0) {
        // +2 for sign bit + safety margin for CPU/GPU rounding differences
        // std::llrint (CPU) and __double2ll_rn (GPU) may round slightly differently
        bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 2;
    }
    info.delta_bits = bits;
}

template<typename T>
int GPUCostOptimalPartitioner<T>::applyMergesGPU(int num_partitions) {
    if (num_partitions <= 1) return num_partitions;

    int blocks = (num_partitions + GPU_MERGE_BLOCK_SIZE - 1) / GPU_MERGE_BLOCK_SIZE;

    // Step 1: Compute output slots
    computeOutputSlotKernel<<<blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        ctx.merge_flags,
        ctx.output_slots,
        ctx.is_merge_base,
        num_partitions);

    // Step 2: Exclusive scan to compute output indices
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(ctx.output_slots),
        thrust::device_pointer_cast(ctx.output_slots + num_partitions),
        thrust::device_pointer_cast(ctx.output_indices));

    // Step 3: Get new partition count (only D2H transfer in merge process)
    int last_slot, last_idx;
    cudaMemcpyAsync(&last_slot, ctx.output_slots + num_partitions - 1,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_idx, ctx.output_indices + num_partitions - 1,
                    sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int new_count = last_idx + last_slot;

    if (new_count == 0 || new_count >= num_partitions) {
        // No merges occurred
        return num_partitions;
    }

    // Step 4: Apply merges in parallel (with statistics propagation)
    applyMergesKernel<T><<<blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        // Old buffers (current)
        ctx.current->starts,
        ctx.current->ends,
        ctx.current->model_types,
        ctx.current->theta0,
        ctx.current->theta1,
        ctx.current->delta_bits,
        ctx.current->costs,
        ctx.current->max_errors,
        // Old cached statistics
        ctx.current->sum_x,
        ctx.current->sum_y,
        ctx.current->sum_xx,
        ctx.current->sum_xy,
        // Merge info
        ctx.output_slots,
        ctx.output_indices,
        ctx.is_merge_base,
        ctx.merged_theta0,
        ctx.merged_theta1,
        ctx.merged_delta_bits,
        // Merged statistics (precomputed)
        ctx.merged_sum_x,
        ctx.merged_sum_y,
        ctx.merged_sum_xx,
        ctx.merged_sum_xy,
        // New buffers (next)
        ctx.next->starts,
        ctx.next->ends,
        ctx.next->model_types,
        ctx.next->theta0,
        ctx.next->theta1,
        ctx.next->delta_bits,
        ctx.next->costs,
        ctx.next->max_errors,
        // New cached statistics
        ctx.next->sum_x,
        ctx.next->sum_y,
        ctx.next->sum_xx,
        ctx.next->sum_xy,
        num_partitions);

    // Step 5: Swap buffers
    ctx.swap();

    return new_count;
}

template<typename T>
std::vector<PartitionInfo> GPUCostOptimalPartitioner<T>::partition() {
    // ================================================================
    // Stage 1: Compute delta-bits per analysis block
    // ================================================================
    int num_analysis_blocks = (data_size + config.analysis_block_size - 1) / config.analysis_block_size;

    int* d_delta_bits_per_block;
    cudaMalloc(&d_delta_bits_per_block, num_analysis_blocks * sizeof(int));

    gpuMergeComputeDeltaBitsKernel<T><<<num_analysis_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        d_data, data_size, config.analysis_block_size,
        d_delta_bits_per_block, num_analysis_blocks);

    // ================================================================
    // Stage 2: Detect breakpoints
    // ================================================================
    int* d_is_breakpoint;
    cudaMalloc(&d_is_breakpoint, num_analysis_blocks * sizeof(int));

    int bp_blocks = (num_analysis_blocks + GPU_MERGE_BLOCK_SIZE - 1) / GPU_MERGE_BLOCK_SIZE;
    gpuMergeDetectBreakpointsKernel<<<bp_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        d_delta_bits_per_block, d_is_breakpoint,
        num_analysis_blocks, config.breakpoint_threshold);

    // Copy breakpoints to host to collect positions (matches original behavior)
    std::vector<int> h_is_breakpoint(num_analysis_blocks);
    cudaMemcpy(h_is_breakpoint.data(), d_is_breakpoint,
               num_analysis_blocks * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<int> breakpoint_positions;
    for (int i = 0; i < num_analysis_blocks; i++) {
        if (h_is_breakpoint[i]) {
            breakpoint_positions.push_back(i * config.analysis_block_size);
        }
    }

    int num_breakpoints = breakpoint_positions.size();
    if (num_breakpoints == 0) {
        breakpoint_positions.push_back(0);
        num_breakpoints = 1;
    }

    cudaFree(d_delta_bits_per_block);
    cudaFree(d_is_breakpoint);

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

    int seg_blocks = (num_breakpoints + GPU_MERGE_BLOCK_SIZE - 1) / GPU_MERGE_BLOCK_SIZE;
    gpuMergeCountPartitionsInSegmentsKernel<<<seg_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        d_breakpoint_positions, num_breakpoints, data_size,
        config.target_partition_size, config.min_partition_size,
        config.max_partition_size, d_partition_counts);

    // Prefix sum to get offsets
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(d_partition_counts),
        thrust::device_pointer_cast(d_partition_counts + num_breakpoints),
        thrust::device_pointer_cast(d_partition_offsets));

    // Get total partition count
    int h_total_partitions;
    int h_last_count;
    cudaMemcpy(&h_total_partitions, d_partition_offsets + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_count, d_partition_counts + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    h_total_partitions += h_last_count;

    int num_partitions = h_total_partitions;
    if (num_partitions == 0) {
        // Fallback: single partition
        num_partitions = 1;
        int zero = 0;
        cudaMemcpy(ctx.current->starts, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.current->ends, &data_size, sizeof(int), cudaMemcpyHostToDevice);
    } else {
        // Write partition boundaries
        gpuMergeWritePartitionsKernel<<<seg_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
            d_breakpoint_positions, num_breakpoints, data_size,
            config.target_partition_size, config.min_partition_size,
            config.max_partition_size, d_partition_offsets,
            ctx.current->starts, ctx.current->ends);
    }

    cudaFree(d_breakpoint_positions);
    cudaFree(d_partition_counts);
    cudaFree(d_partition_offsets);

    cudaStreamSynchronize(stream);

    // ================================================================
    // Stage 4: Fit models for each partition (and cache statistics)
    // ================================================================
    gpuMergeFitPartitionsKernel<T><<<num_partitions, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
        d_data,
        ctx.current->starts,
        ctx.current->ends,
        ctx.current->model_types,
        ctx.current->theta0,
        ctx.current->theta1,
        ctx.current->theta2,
        ctx.current->theta3,
        ctx.current->delta_bits,
        ctx.current->max_errors,
        ctx.current->costs,
        // Cache statistics for O(1) merge evaluation
        ctx.current->sum_x,
        ctx.current->sum_y,
        ctx.current->sum_xx,
        ctx.current->sum_xy,
        num_partitions);

    // ================================================================
    // Stage 4.5: Initial Polynomial Model Selection (if enabled)
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        gpuMergeRefitPolynomialKernel<T><<<num_partitions, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
            d_data,
            ctx.current->starts,
            ctx.current->ends,
            ctx.current->model_types,
            ctx.current->theta0,
            ctx.current->theta1,
            ctx.current->theta2,
            ctx.current->theta3,
            ctx.current->delta_bits,
            ctx.current->costs,
            num_partitions,
            config.polynomial_min_size,
            config.cubic_min_size,
            config.polynomial_cost_threshold);
    }

    // ================================================================
    // Stage 5-6: GPU-Parallel Merge Loop (OPTIMIZED with cached statistics)
    // ================================================================
    if (config.enable_merging && num_partitions > 1) {
        for (int round = 0; round < config.max_merge_rounds; round++) {
            // Clear merge flags
            cudaMemsetAsync(ctx.merge_flags, 0, num_partitions * sizeof(int), stream);

            // Evaluate merge benefits using O(1) theta computation from cached statistics
            gpuMergeEvaluateMergeCostKernel<T><<<num_partitions, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
                d_data,
                ctx.current->starts,
                ctx.current->ends,
                ctx.current->costs,
                // Cached statistics for O(1) theta computation
                ctx.current->sum_x,
                ctx.current->sum_y,
                ctx.current->sum_xx,
                ctx.current->sum_xy,
                // Output
                ctx.merge_benefits,
                ctx.merged_delta_bits,
                ctx.merged_theta0,
                ctx.merged_theta1,
                // Merged statistics for propagation
                ctx.merged_sum_x,
                ctx.merged_sum_y,
                ctx.merged_sum_xx,
                ctx.merged_sum_xy,
                num_partitions,
                config.max_partition_size);

            // Mark merges - Even phase (pairs 0-1, 2-3, 4-5, ...)
            int mark_blocks = (num_partitions / 2 + GPU_MERGE_BLOCK_SIZE - 1) / GPU_MERGE_BLOCK_SIZE;
            if (mark_blocks == 0) mark_blocks = 1;
            gpuMergeMarkMergesKernel<<<mark_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
                ctx.merge_benefits, ctx.merge_flags, num_partitions,
                0, config.merge_benefit_threshold);

            // Mark merges - Odd phase (pairs 1-2, 3-4, 5-6, ...)
            gpuMergeMarkMergesKernel<<<mark_blocks, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
                ctx.merge_benefits, ctx.merge_flags, num_partitions,
                1, config.merge_benefit_threshold);

            // Apply merges using GPU stream compaction
            int prev_count = num_partitions;
            num_partitions = applyMergesGPU(num_partitions);

            if (num_partitions == prev_count || num_partitions <= 1) {
                break;  // No more merges possible
            }
        }
    }

    // ================================================================
    // Stage 6.5: Post-Merge Polynomial Re-evaluation (if enabled)
    // Re-evaluate merged partitions (which use LINEAR) for polynomial
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        gpuMergeRefitPolynomialKernel<T><<<num_partitions, GPU_MERGE_BLOCK_SIZE, 0, stream>>>(
            d_data,
            ctx.current->starts,
            ctx.current->ends,
            ctx.current->model_types,
            ctx.current->theta0,
            ctx.current->theta1,
            ctx.current->theta2,
            ctx.current->theta3,
            ctx.current->delta_bits,
            ctx.current->costs,
            num_partitions,
            config.polynomial_min_size,
            config.cubic_min_size,
            config.polynomial_cost_threshold);
    }

    // ================================================================
    // Stage 7: Copy results back and build PartitionInfo vector
    // ================================================================
    std::vector<int> h_starts(num_partitions);
    std::vector<int> h_ends(num_partitions);
    std::vector<int> h_model_types(num_partitions);
    std::vector<double> h_theta0(num_partitions);
    std::vector<double> h_theta1(num_partitions);
    std::vector<double> h_theta2(num_partitions);
    std::vector<double> h_theta3(num_partitions);
    std::vector<int> h_delta_bits(num_partitions);
    std::vector<long long> h_max_errors(num_partitions);

    cudaMemcpy(h_starts.data(), ctx.current->starts, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ends.data(), ctx.current->ends, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_model_types.data(), ctx.current->model_types, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta0.data(), ctx.current->theta0, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta1.data(), ctx.current->theta1, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta2.data(), ctx.current->theta2, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_theta3.data(), ctx.current->theta3, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits.data(), ctx.current->delta_bits, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_errors.data(), ctx.current->max_errors, num_partitions * sizeof(long long), cudaMemcpyDeviceToHost);

    std::vector<PartitionInfo> result;
    result.reserve(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        PartitionInfo info;
        info.start_idx = h_starts[i];
        info.end_idx = h_ends[i];
        info.model_type = h_model_types[i];
        info.model_params[0] = h_theta0[i];
        info.model_params[1] = h_theta1[i];
        info.model_params[2] = h_theta2[i];
        info.model_params[3] = h_theta3[i];
        info.delta_bits = h_delta_bits[i];
        info.delta_array_bit_offset = 0;
        info.error_bound = h_max_errors[i];
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

    return result;
}

template<typename T>
void GPUCostOptimalPartitioner<T>::getStats(int& num_partitions, float& avg_partition_size) const {
    // This would need to track stats during partition()
    num_partitions = 0;
    avg_partition_size = 0.0f;
}

// Explicit template instantiation
template class GPUCostOptimalPartitioner<int32_t>;
template class GPUCostOptimalPartitioner<uint32_t>;
template class GPUCostOptimalPartitioner<int64_t>;
template class GPUCostOptimalPartitioner<uint64_t>;

// ============================================================================
// Validation Function
// ============================================================================

template<typename T>
bool validateGPUMerge(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    bool verbose)
{
    // Run GPU version
    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    // Run original CPU version
    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    // Compare results
    if (gpu_result.size() != cpu_result.size()) {
        if (verbose) {
            std::cerr << "Partition count mismatch: GPU=" << gpu_result.size()
                      << " CPU=" << cpu_result.size() << std::endl;
        }
        return false;
    }

    for (size_t i = 0; i < gpu_result.size(); i++) {
        const auto& g = gpu_result[i];
        const auto& c = cpu_result[i];

        if (g.start_idx != c.start_idx || g.end_idx != c.end_idx) {
            if (verbose) {
                std::cerr << "Partition " << i << " boundary mismatch: "
                          << "GPU=[" << g.start_idx << "," << g.end_idx << "] "
                          << "CPU=[" << c.start_idx << "," << c.end_idx << "]" << std::endl;
            }
            return false;
        }

        // Allow small floating point differences in model params
        if (std::fabs(g.model_params[0] - c.model_params[0]) > 1e-6 ||
            std::fabs(g.model_params[1] - c.model_params[1]) > 1e-6) {
            if (verbose) {
                std::cerr << "Partition " << i << " model params mismatch" << std::endl;
            }
            return false;
        }
    }

    if (verbose) {
        std::cout << "GPU merge validation passed! Partitions: " << gpu_result.size() << std::endl;
    }

    return true;
}

// Explicit instantiation
template bool validateGPUMerge<int32_t>(const std::vector<int32_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMerge<uint32_t>(const std::vector<uint32_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMerge<int64_t>(const std::vector<int64_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMerge<uint64_t>(const std::vector<uint64_t>&, const CostOptimalConfig&, bool);
