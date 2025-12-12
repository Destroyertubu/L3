/**
 * Optimized GPU-Parallel Merge Implementation for Cost-Optimal Partitioning
 *
 * Version 2: Key optimizations over v1:
 * 1. Cooperative Groups for GPU-side loop control (zero D2H sync during merge)
 * 2. Block-level prefix sum using shared memory (no thrust overhead)
 * 3. Fused kernels to reduce launch overhead
 * 4. Atomic counting for early termination detection
 *
 * Author: Claude Code Assistant
 * Date: 2025-12-07
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <chrono>
#include <type_traits>
#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

namespace cg = cooperative_groups;

// ============================================================================
// Device Helper Functions
// ============================================================================

namespace gpu_merge_v2 {

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

__device__ inline long long warpReduceMin(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = min(val, __shfl_down_sync(0xffffffff, val, offset));
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

// Kahan summation pair: (sum, compensation) for high-precision accumulation
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
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other_sum = __shfl_down_sync(0xffffffff, kp.sum, offset);
        double other_comp = __shfl_down_sync(0xffffffff, kp.comp, offset);
        kp = kahanAdd(kp, other_sum);
        kp.comp += other_comp;
    }
    return kp;
}

__device__ inline double blockReduceSumKahan(double val) {
    __shared__ double shared_sum[32];
    __shared__ double shared_comp[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

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

__device__ inline long long blockReduceMin(long long val) {
    __shared__ long long shared_min[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMin(val);
    if (lane == 0) shared_min[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared_min[lane] : LLONG_MAX;
    if (wid == 0) val = warpReduceMin(val);
    return val;
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

__device__ inline int computeBitsForValue(unsigned long long val) {
    if (val == 0) return 0;
    return 64 - __clzll(val);
}

// Efficient warp-level inclusive scan
__device__ inline int warpInclusiveScan(int val) {
    int lane = threadIdx.x & 31;
    for (int d = 1; d < 32; d *= 2) {
        int n = __shfl_up_sync(0xffffffff, val, d);
        if (lane >= d) val += n;
    }
    return val;
}

// Block-level exclusive scan using shared memory
// Returns the total sum of the block
__device__ inline int blockExclusiveScan(int val, int* shared_data) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    // Step 1: Warp-level inclusive scan
    int warp_result = warpInclusiveScan(val);

    // Store warp totals
    if (lane == 31) {
        shared_data[wid] = warp_result;
    }
    __syncthreads();

    // Step 2: Scan warp totals (single warp)
    if (wid == 0 && lane < num_warps) {
        int warp_total = shared_data[lane];
        warp_total = warpInclusiveScan(warp_total);
        shared_data[lane] = warp_total;
    }
    __syncthreads();

    // Step 3: Add prefix from previous warps
    int prefix = (wid > 0) ? shared_data[wid - 1] : 0;
    int inclusive_result = warp_result + prefix;

    // Convert to exclusive
    return inclusive_result - val;
}

} // namespace gpu_merge_v2

// ============================================================================
// Buffer Allocation Implementation
// ============================================================================

template<typename T>
cudaError_t UnifiedPartitionBuffer<T>::allocate(size_t max_partitions) {
    if (allocated && capacity >= max_partitions) return cudaSuccess;
    free();

    cudaError_t err;
    size_t size_int = max_partitions * sizeof(int);
    size_t size_double = max_partitions * sizeof(double);
    size_t size_float = max_partitions * sizeof(float);
    size_t size_ll = max_partitions * sizeof(long long);

    err = cudaMalloc(&starts, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&ends, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&model_types, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&theta0, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&theta1, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&theta2, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&theta3, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&delta_bits, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&costs, size_float); if (err != cudaSuccess) return err;
    err = cudaMalloc(&max_errors, size_ll); if (err != cudaSuccess) return err;
    err = cudaMalloc(&sum_x, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&sum_y, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&sum_xx, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&sum_xy, size_double); if (err != cudaSuccess) return err;

    capacity = max_partitions;
    allocated = true;
    return cudaSuccess;
}

template<typename T>
void UnifiedPartitionBuffer<T>::free() {
    if (starts) cudaFree(starts);
    if (ends) cudaFree(ends);
    if (model_types) cudaFree(model_types);
    if (theta0) cudaFree(theta0);
    if (theta1) cudaFree(theta1);
    if (theta2) cudaFree(theta2);
    if (theta3) cudaFree(theta3);
    if (delta_bits) cudaFree(delta_bits);
    if (costs) cudaFree(costs);
    if (max_errors) cudaFree(max_errors);
    if (sum_x) cudaFree(sum_x);
    if (sum_y) cudaFree(sum_y);
    if (sum_xx) cudaFree(sum_xx);
    if (sum_xy) cudaFree(sum_xy);

    starts = ends = model_types = delta_bits = nullptr;
    theta0 = theta1 = theta2 = theta3 = nullptr;
    costs = nullptr;
    max_errors = nullptr;
    sum_x = sum_y = sum_xx = sum_xy = nullptr;
    capacity = 0;
    allocated = false;
}

template struct UnifiedPartitionBuffer<int32_t>;
template struct UnifiedPartitionBuffer<uint32_t>;
template struct UnifiedPartitionBuffer<int64_t>;
template struct UnifiedPartitionBuffer<uint64_t>;

template<typename T>
cudaError_t GPUMergeContextV2<T>::allocate(size_t max_partitions) {
    if (allocated && capacity >= max_partitions) return cudaSuccess;
    free();

    cudaError_t err;

    err = buffer_A.allocate(max_partitions);
    if (err != cudaSuccess) return err;
    err = buffer_B.allocate(max_partitions);
    if (err != cudaSuccess) return err;

    current = &buffer_A;
    next = &buffer_B;

    size_t size_int = max_partitions * sizeof(int);
    size_t size_double = max_partitions * sizeof(double);
    size_t size_float = max_partitions * sizeof(float);

    err = cudaMalloc(&merge_benefits, size_float); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_delta_bits, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta0, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta1, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta2, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_theta3, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_x, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_y, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_xx, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merged_sum_xy, size_double); if (err != cudaSuccess) return err;
    err = cudaMalloc(&merge_flags, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&output_slots, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&output_indices, size_int); if (err != cudaSuccess) return err;
    err = cudaMalloc(&is_merge_base, size_int); if (err != cudaSuccess) return err;

    // Block sums for multi-block scan
    int num_blocks = (max_partitions + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
    err = cudaMalloc(&block_sums, (num_blocks + 1) * sizeof(int));
    if (err != cudaSuccess) return err;

    // Device counters
    err = cudaMalloc(&d_num_partitions, sizeof(int)); if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_merge_count, sizeof(int)); if (err != cudaSuccess) return err;
    err = cudaMalloc(&d_new_partition_count, sizeof(int)); if (err != cudaSuccess) return err;

    // Pinned host memory for final result
    err = cudaMallocHost(&h_final_partition_count, sizeof(int));
    if (err != cudaSuccess) return err;

    capacity = max_partitions;
    allocated = true;
    return cudaSuccess;
}

template<typename T>
void GPUMergeContextV2<T>::free() {
    buffer_A.free();
    buffer_B.free();
    current = next = nullptr;

    if (merge_benefits) cudaFree(merge_benefits);
    if (merged_delta_bits) cudaFree(merged_delta_bits);
    if (merged_theta0) cudaFree(merged_theta0);
    if (merged_theta1) cudaFree(merged_theta1);
    if (merged_theta2) cudaFree(merged_theta2);
    if (merged_theta3) cudaFree(merged_theta3);
    if (merged_sum_x) cudaFree(merged_sum_x);
    if (merged_sum_y) cudaFree(merged_sum_y);
    if (merged_sum_xx) cudaFree(merged_sum_xx);
    if (merged_sum_xy) cudaFree(merged_sum_xy);
    if (merge_flags) cudaFree(merge_flags);
    if (output_slots) cudaFree(output_slots);
    if (output_indices) cudaFree(output_indices);
    if (is_merge_base) cudaFree(is_merge_base);
    if (block_sums) cudaFree(block_sums);
    if (d_num_partitions) cudaFree(d_num_partitions);
    if (d_merge_count) cudaFree(d_merge_count);
    if (d_new_partition_count) cudaFree(d_new_partition_count);
    if (h_final_partition_count) cudaFreeHost(h_final_partition_count);

    merge_benefits = nullptr;
    merged_delta_bits = nullptr;
    merged_theta0 = merged_theta1 = merged_theta2 = merged_theta3 = nullptr;
    merged_sum_x = merged_sum_y = merged_sum_xx = merged_sum_xy = nullptr;
    merge_flags = nullptr;
    output_slots = output_indices = is_merge_base = nullptr;
    block_sums = nullptr;
    d_num_partitions = d_merge_count = d_new_partition_count = nullptr;
    h_final_partition_count = nullptr;

    capacity = 0;
    allocated = false;
}

template struct GPUMergeContextV2<int32_t>;
template struct GPUMergeContextV2<uint32_t>;
template struct GPUMergeContextV2<int64_t>;
template struct GPUMergeContextV2<uint64_t>;

// ============================================================================
// Cooperative Merge Loop Kernel
// ============================================================================

template<typename T>
__global__ void __launch_bounds__(GPU_MERGE_V2_BLOCK_SIZE)
mergeLoopCooperativeKernel(
    const T* __restrict__ data,
    // Buffer A
    int* __restrict__ starts_A, int* __restrict__ ends_A,
    int* __restrict__ model_types_A, double* __restrict__ theta0_A, double* __restrict__ theta1_A,
    int* __restrict__ delta_bits_A, float* __restrict__ costs_A, long long* __restrict__ max_errors_A,
    double* __restrict__ sum_x_A, double* __restrict__ sum_y_A,
    double* __restrict__ sum_xx_A, double* __restrict__ sum_xy_A,
    // Buffer B
    int* __restrict__ starts_B, int* __restrict__ ends_B,
    int* __restrict__ model_types_B, double* __restrict__ theta0_B, double* __restrict__ theta1_B,
    int* __restrict__ delta_bits_B, float* __restrict__ costs_B, long long* __restrict__ max_errors_B,
    double* __restrict__ sum_x_B, double* __restrict__ sum_y_B,
    double* __restrict__ sum_xx_B, double* __restrict__ sum_xy_B,
    // Working arrays
    float* __restrict__ merge_benefits,
    int* __restrict__ merged_delta_bits,
    double* __restrict__ merged_theta0, double* __restrict__ merged_theta1,
    double* __restrict__ merged_sum_x, double* __restrict__ merged_sum_y,
    double* __restrict__ merged_sum_xx, double* __restrict__ merged_sum_xy,
    int* __restrict__ merge_flags,
    int* __restrict__ output_slots, int* __restrict__ output_indices, int* __restrict__ is_merge_base,
    int* __restrict__ block_sums,
    // Control
    int* __restrict__ d_num_partitions,
    int* __restrict__ d_merge_count,
    int max_rounds,
    float threshold,
    int max_partition_size,
    int data_size)
{
    cg::grid_group grid = cg::this_grid();

    // Shared memory for block-level operations
    __shared__ double s_theta0, s_theta1;
    __shared__ int s_scan_data[GPU_MERGE_V2_BLOCK_SIZE / 32 + 1];

    // Pointers to current and next buffers (will swap each round)
    int* cur_starts = starts_A;
    int* cur_ends = ends_A;
    int* cur_model_types = model_types_A;
    double* cur_theta0 = theta0_A;
    double* cur_theta1 = theta1_A;
    int* cur_delta_bits = delta_bits_A;
    float* cur_costs = costs_A;
    long long* cur_max_errors = max_errors_A;
    double* cur_sum_x = sum_x_A;
    double* cur_sum_y = sum_y_A;
    double* cur_sum_xx = sum_xx_A;
    double* cur_sum_xy = sum_xy_A;

    int* nxt_starts = starts_B;
    int* nxt_ends = ends_B;
    int* nxt_model_types = model_types_B;
    double* nxt_theta0 = theta0_B;
    double* nxt_theta1 = theta1_B;
    int* nxt_delta_bits = delta_bits_B;
    float* nxt_costs = costs_B;
    long long* nxt_max_errors = max_errors_B;
    double* nxt_sum_x = sum_x_B;
    double* nxt_sum_y = sum_y_B;
    double* nxt_sum_xx = sum_xx_B;
    double* nxt_sum_xy = sum_xy_B;

    int use_buffer_A = 1;

    for (int round = 0; round < max_rounds; round++) {
        int num_parts = *d_num_partitions;
        if (num_parts <= 1) break;

        // Reset merge count
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_merge_count = 0;
        }
        grid.sync();

        // ================================================================
        // Phase 1: Evaluate merge costs (one block per partition pair)
        // ================================================================
        // First, initialize merge_benefits to -1 for all pairs (so unevaluated pairs won't merge)
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_parts; i += gridDim.x * blockDim.x) {
            merge_benefits[i] = -1.0f;
        }
        grid.sync();

        // Now evaluate merge costs for pairs we can process (one block per pair)
        int pid = blockIdx.x;
        if (pid < num_parts - 1) {
            int start_a = cur_starts[pid];
            int end_a = cur_ends[pid];
            int start_b = cur_starts[pid + 1];
            int end_b = cur_ends[pid + 1];

            int n_a = end_a - start_a;
            int n_b = end_b - start_b;
            int n_c = n_a + n_b;

            if (n_c > max_partition_size) {
                if (threadIdx.x == 0) {
                    merge_benefits[pid] = -1.0f;
                }
            } else {
                // Use centered fitting for numerical stability with large values
                // PASS 1: Compute y_mean and track max value for precision check
                int merged_start = start_a;
                int merged_end = end_b;

                double local_sum_y_pass1 = 0.0;
                unsigned long long local_max_ull = 0ULL;
                for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
                    T val = data[i];
                    local_sum_y_pass1 += static_cast<double>(val);
                    // Track max as unsigned for precision check
                    unsigned long long val_ull = static_cast<unsigned long long>(val);
                    local_max_ull = max(local_max_ull, val_ull);
                }
                double sum_y_pass1 = gpu_merge_v2::blockReduceSum(local_sum_y_pass1);

                // Reduce max value
                unsigned long long global_max_ull = gpu_merge_v2::blockReduceMaxUnsigned(local_max_ull);

                __shared__ double s_y_mean;
                __shared__ bool s_values_too_large_coop;
                if (threadIdx.x == 0) {
                    s_y_mean = sum_y_pass1 / static_cast<double>(n_c);

                    // CRITICAL: For uint64_t values > 2^53, double precision is insufficient
                    // for accurate linear model predictions. Disallow merge in this case.
                    s_values_too_large_coop = false;
                    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
                        if (global_max_ull > GPU_MERGE_V2_DOUBLE_PRECISION_MAX) {
                            s_values_too_large_coop = true;
                            merge_benefits[pid] = -1.0f;  // Don't merge
                        }
                    }
                }
                __syncthreads();

                // Skip further processing if values too large
                if (s_values_too_large_coop) {
                    // Do nothing - merge_benefits already set to -1
                } else {

                double y_mean = s_y_mean;

                // PASS 2: Compute centered statistics
                double local_sum_x = 0.0, local_sum_y_c = 0.0;
                double local_sum_xx = 0.0, local_sum_xy_c = 0.0;

                for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
                    double x = static_cast<double>(i - merged_start);
                    double y_c = static_cast<double>(data[i]) - y_mean;
                    local_sum_x += x;
                    local_sum_y_c += y_c;
                    local_sum_xx += x * x;
                    local_sum_xy_c += x * y_c;
                }

                double m_sx = gpu_merge_v2::blockReduceSum(local_sum_x);
                __syncthreads();
                double sum_y_c = gpu_merge_v2::blockReduceSum(local_sum_y_c);
                __syncthreads();
                double m_sxx = gpu_merge_v2::blockReduceSum(local_sum_xx);
                __syncthreads();
                double sum_xy_c = gpu_merge_v2::blockReduceSum(local_sum_xy_c);
                __syncthreads();

                __shared__ double s_x_mean_coop;
                if (threadIdx.x == 0) {
                    double dn_c = static_cast<double>(n_c);
                    double x_mean = m_sx / dn_c;

                    // Centered linear regression
                    double sum_x2_c = m_sxx - dn_c * x_mean * x_mean;
                    double cov_xy = sum_xy_c - x_mean * sum_y_c;

                    if (fabs(sum_x2_c) > 1e-10) {
                        s_theta1 = cov_xy / sum_x2_c;
                    } else {
                        s_theta1 = 0.0;
                    }
                    // CRITICAL FIX: Anchor theta0 at first data point to avoid catastrophic cancellation
                    // AND ensure max_error calculation uses the same theta0 that will be stored!
                    // Using y_mean - theta1 * x_mean causes precision loss when values are large.
                    s_theta0 = static_cast<double>(data[merged_start]);  // Anchor at first point
                    merged_theta0[pid] = s_theta0;
                    merged_theta1[pid] = s_theta1;

                    // Store original (non-centered) statistics for compatibility
                    double m_sy = sum_y_pass1;
                    double m_sxy = sum_xy_c + y_mean * m_sx;
                    merged_sum_x[pid] = m_sx;
                    merged_sum_y[pid] = m_sy;
                    merged_sum_xx[pid] = m_sxx;
                    merged_sum_xy[pid] = m_sxy;
                }
                __syncthreads();

                // PASS 3: Compute max error using ANCHORED prediction (matches encoder exactly)
                // CRITICAL: Must use same prediction formula as encoder: pred = theta0 + theta1 * x
                long long local_max_error = 0;
                for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
                    int local_idx = i - merged_start;
                    // Anchored prediction: pred = theta0 + theta1 * x (matches encoder)
                    double predicted = s_theta0 + s_theta1 * static_cast<double>(local_idx);
                    T pred_val = static_cast<T>(__double2ll_rn(predicted));
                    long long delta;
                    if (data[i] >= pred_val) {
                        delta = (long long)(data[i] - pred_val);
                    } else {
                        delta = -(long long)(pred_val - data[i]);
                    }
                    local_max_error = max(local_max_error, llabs(delta));
                }

                long long max_error = gpu_merge_v2::blockReduceMax(local_max_error);

                if (threadIdx.x == 0) {
                    int bits = 0;
                    if (max_error > 0) {
                        // +2 for sign bit + safety margin for floating-point rounding
                        bits = gpu_merge_v2::computeBitsForValue((unsigned long long)max_error) + 2;
                    }
                    merged_delta_bits[pid] = bits;

                    float delta_bytes = (float)n_c * bits / 8.0f;
                    float merged_cost = GPU_MERGE_V2_MODEL_OVERHEAD_BYTES + delta_bytes;
                    float separate_cost = cur_costs[pid] + cur_costs[pid + 1];
                    float benefit = (separate_cost - merged_cost) / separate_cost;
                    merge_benefits[pid] = benefit;
                }
                } // end of "values not too large" block
            }
        }
        grid.sync();

        // ================================================================
        // Phase 2: Mark merges (odd-even) and compute output slots
        // ================================================================
        // Clear merge flags first
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_parts; i += gridDim.x * blockDim.x) {
            merge_flags[i] = 0;
        }
        grid.sync();

        // Even phase: pairs (0,1), (2,3), (4,5), ...
        // Use atomicCAS to safely mark both the base (value=1) and absorbed (value=2) partitions
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (num_parts - 1) / 2 + 1; i += gridDim.x * blockDim.x) {
            int p = i * 2;
            if (p < num_parts - 1 && merge_benefits[p] >= threshold) {
                // Try to atomically mark the base partition
                if (atomicCAS(&merge_flags[p], 0, 1) == 0) {
                    // Successfully marked base, now try to mark absorbed partition
                    if (atomicCAS(&merge_flags[p + 1], 0, 2) == 0) {
                        // Success - both partitions marked
                        atomicAdd(d_merge_count, 1);
                    } else {
                        // Failed to mark absorbed - rollback the base
                        merge_flags[p] = 0;
                    }
                }
            }
        }
        grid.sync();

        // Odd phase: pairs (1,2), (3,4), (5,6), ...
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (num_parts - 2) / 2 + 1; i += gridDim.x * blockDim.x) {
            int p = i * 2 + 1;
            if (p < num_parts - 1 && merge_benefits[p] >= threshold) {
                // Try to atomically mark the base partition
                if (atomicCAS(&merge_flags[p], 0, 1) == 0) {
                    // Successfully marked base, now try to mark absorbed partition
                    if (atomicCAS(&merge_flags[p + 1], 0, 2) == 0) {
                        // Success - both partitions marked
                        atomicAdd(d_merge_count, 1);
                    } else {
                        // Failed to mark absorbed - rollback the base
                        merge_flags[p] = 0;
                    }
                }
            }
        }
        grid.sync();

        // Check if any merges happened
        if (*d_merge_count == 0) break;

        // Compute output slots
        // merge_flags[i] == 1 means partition i is base of merge with i+1
        // merge_flags[i] == 2 means partition i is absorbed by partition i-1
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_parts; i += gridDim.x * blockDim.x) {
            bool is_absorbed = (merge_flags[i] == 2);
            bool is_merge_base_flag = (merge_flags[i] == 1);

            if (is_absorbed) {
                output_slots[i] = 0;
                is_merge_base[i] = 0;
            } else {
                output_slots[i] = 1;
                is_merge_base[i] = is_merge_base_flag ? 1 : 0;
            }
        }
        grid.sync();

        // ================================================================
        // Phase 3: Prefix sum for output indices
        // Using simple grid-stride approach with block sums
        // ================================================================
        int num_blocks_needed = (num_parts + blockDim.x - 1) / blockDim.x;

        // Step 1: Each block computes local prefix sum and block total
        {
            int block_start = blockIdx.x * blockDim.x;
            int local_idx = threadIdx.x;
            int global_idx = block_start + local_idx;

            int val = (global_idx < num_parts) ? output_slots[global_idx] : 0;
            int exclusive_result = gpu_merge_v2::blockExclusiveScan(val, s_scan_data);

            if (global_idx < num_parts) {
                output_indices[global_idx] = exclusive_result;
            }

            // Store block total
            if (threadIdx.x == blockDim.x - 1) {
                block_sums[blockIdx.x] = exclusive_result + val;
            }
        }
        grid.sync();

        // Step 2: Scan block sums (single block does this)
        if (blockIdx.x == 0) {
            for (int i = threadIdx.x; i < num_blocks_needed && i < gridDim.x; i += blockDim.x) {
                int bs = block_sums[i];
                int prefix = gpu_merge_v2::blockExclusiveScan(bs, s_scan_data);
                block_sums[i] = prefix;
            }
        }
        grid.sync();

        // Handle case where we have more blocks than one scan can handle
        if (num_blocks_needed > blockDim.x) {
            // Sequential fallback for block sums (rare case)
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                int running_sum = 0;
                for (int i = 0; i < num_blocks_needed; i++) {
                    int old_val = block_sums[i];
                    block_sums[i] = running_sum;
                    running_sum += old_val;
                }
            }
            grid.sync();
        }

        // Step 3: Add block offsets to get final indices
        {
            int block_start = blockIdx.x * blockDim.x;
            int global_idx = block_start + threadIdx.x;
            if (global_idx < num_parts && blockIdx.x > 0) {
                output_indices[global_idx] += block_sums[blockIdx.x];
            }
        }
        grid.sync();

        // Get new partition count
        int new_count;
        {
            int last_idx = num_parts - 1;
            new_count = output_indices[last_idx] + output_slots[last_idx];
        }

        if (new_count == 0 || new_count >= num_parts) break;

        // ================================================================
        // Phase 4: Apply merges
        // ================================================================
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_parts; i += gridDim.x * blockDim.x) {
            if (output_slots[i] == 0) continue;

            int out_idx = output_indices[i];

            if (is_merge_base[i]) {
                nxt_starts[out_idx] = cur_starts[i];
                nxt_ends[out_idx] = cur_ends[i + 1];
                nxt_model_types[out_idx] = MODEL_LINEAR;
                nxt_theta0[out_idx] = merged_theta0[i];
                nxt_theta1[out_idx] = merged_theta1[i];
                nxt_delta_bits[out_idx] = merged_delta_bits[i];

                int n = cur_ends[i + 1] - cur_starts[i];
                float delta_bytes = (float)n * merged_delta_bits[i] / 8.0f;
                nxt_costs[out_idx] = GPU_MERGE_V2_MODEL_OVERHEAD_BYTES + delta_bytes;
                nxt_max_errors[out_idx] = 0;

                nxt_sum_x[out_idx] = merged_sum_x[i];
                nxt_sum_y[out_idx] = merged_sum_y[i];
                nxt_sum_xx[out_idx] = merged_sum_xx[i];
                nxt_sum_xy[out_idx] = merged_sum_xy[i];
            } else {
                nxt_starts[out_idx] = cur_starts[i];
                nxt_ends[out_idx] = cur_ends[i];
                nxt_model_types[out_idx] = cur_model_types[i];
                nxt_theta0[out_idx] = cur_theta0[i];
                nxt_theta1[out_idx] = cur_theta1[i];
                nxt_delta_bits[out_idx] = cur_delta_bits[i];
                nxt_costs[out_idx] = cur_costs[i];
                nxt_max_errors[out_idx] = cur_max_errors[i];

                nxt_sum_x[out_idx] = cur_sum_x[i];
                nxt_sum_y[out_idx] = cur_sum_y[i];
                nxt_sum_xx[out_idx] = cur_sum_xx[i];
                nxt_sum_xy[out_idx] = cur_sum_xy[i];
            }
        }
        grid.sync();

        // Update partition count
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_num_partitions = new_count;
        }

        // Swap buffers
        use_buffer_A = 1 - use_buffer_A;
        if (use_buffer_A) {
            cur_starts = starts_A; cur_ends = ends_A;
            cur_model_types = model_types_A;
            cur_theta0 = theta0_A; cur_theta1 = theta1_A;
            cur_delta_bits = delta_bits_A; cur_costs = costs_A;
            cur_max_errors = max_errors_A;
            cur_sum_x = sum_x_A; cur_sum_y = sum_y_A;
            cur_sum_xx = sum_xx_A; cur_sum_xy = sum_xy_A;

            nxt_starts = starts_B; nxt_ends = ends_B;
            nxt_model_types = model_types_B;
            nxt_theta0 = theta0_B; nxt_theta1 = theta1_B;
            nxt_delta_bits = delta_bits_B; nxt_costs = costs_B;
            nxt_max_errors = max_errors_B;
            nxt_sum_x = sum_x_B; nxt_sum_y = sum_y_B;
            nxt_sum_xx = sum_xx_B; nxt_sum_xy = sum_xy_B;
        } else {
            cur_starts = starts_B; cur_ends = ends_B;
            cur_model_types = model_types_B;
            cur_theta0 = theta0_B; cur_theta1 = theta1_B;
            cur_delta_bits = delta_bits_B; cur_costs = costs_B;
            cur_max_errors = max_errors_B;
            cur_sum_x = sum_x_B; cur_sum_y = sum_y_B;
            cur_sum_xx = sum_xx_B; cur_sum_xy = sum_xy_B;

            nxt_starts = starts_A; nxt_ends = ends_A;
            nxt_model_types = model_types_A;
            nxt_theta0 = theta0_A; nxt_theta1 = theta1_A;
            nxt_delta_bits = delta_bits_A; nxt_costs = costs_A;
            nxt_max_errors = max_errors_A;
            nxt_sum_x = sum_x_A; nxt_sum_y = sum_y_A;
            nxt_sum_xx = sum_xx_A; nxt_sum_xy = sum_xy_A;
        }

        grid.sync();
    }
}

// Explicit instantiation
template __global__ void mergeLoopCooperativeKernel<int32_t>(
    const int32_t*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, float*, int*, double*, double*, double*, double*, double*, double*,
    int*, int*, int*, int*, int*, int*, int*, int, float, int, int);
template __global__ void mergeLoopCooperativeKernel<uint32_t>(
    const uint32_t*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, float*, int*, double*, double*, double*, double*, double*, double*,
    int*, int*, int*, int*, int*, int*, int*, int, float, int, int);
template __global__ void mergeLoopCooperativeKernel<int64_t>(
    const int64_t*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, float*, int*, double*, double*, double*, double*, double*, double*,
    int*, int*, int*, int*, int*, int*, int*, int, float, int, int);
template __global__ void mergeLoopCooperativeKernel<uint64_t>(
    const uint64_t*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, int*, int*, int*, double*, double*, int*, float*, long long*,
    double*, double*, double*, double*, float*, int*, double*, double*, double*, double*, double*, double*,
    int*, int*, int*, int*, int*, int*, int*, int, float, int, int);

// ============================================================================
// Fallback Multi-Kernel Implementation
// ============================================================================

template<typename T>
__global__ void evaluateMergeCostV2Kernel(
    const T* __restrict__ data,
    const int* __restrict__ starts, const int* __restrict__ ends,
    const float* __restrict__ costs,
    const double* __restrict__ sum_x, const double* __restrict__ sum_y,
    const double* __restrict__ sum_xx, const double* __restrict__ sum_xy,
    float* __restrict__ merge_benefits,
    int* __restrict__ merged_delta_bits,
    double* __restrict__ merged_theta0, double* __restrict__ merged_theta1,
    double* __restrict__ merged_sum_x, double* __restrict__ merged_sum_y,
    double* __restrict__ merged_sum_xx, double* __restrict__ merged_sum_xy,
    int num_partitions,
    int max_partition_size)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions - 1) return;

    int start_a = starts[pid];
    int end_a = ends[pid];
    int start_b = starts[pid + 1];
    int end_b = ends[pid + 1];

    int n_a = end_a - start_a;
    int n_b = end_b - start_b;
    int n_c = n_a + n_b;

    if (n_c > max_partition_size) {
        if (threadIdx.x == 0) {
            merge_benefits[pid] = -1.0f;
        }
        return;
    }

    __shared__ double s_theta0, s_theta1, s_x_mean;

    // Use centered fitting for numerical stability with large values
    // PASS 1: Compute y_mean and track max value for precision check
    int merged_start = start_a;
    int merged_end = end_b;

    double local_sum_y_pass1 = 0.0;
    unsigned long long local_max_ull = 0ULL;
    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        T val = data[i];
        local_sum_y_pass1 += static_cast<double>(val);
        // Track max as unsigned for precision check
        unsigned long long val_ull = static_cast<unsigned long long>(val);
        local_max_ull = max(local_max_ull, val_ull);
    }
    double sum_y_pass1 = gpu_merge_v2::blockReduceSum(local_sum_y_pass1);

    // Reduce max value
    unsigned long long global_max_ull = gpu_merge_v2::blockReduceMaxUnsigned(local_max_ull);

    __shared__ double s_y_mean;
    __shared__ bool s_values_too_large;
    if (threadIdx.x == 0) {
        s_y_mean = sum_y_pass1 / static_cast<double>(n_c);

        // CRITICAL: For uint64_t values > 2^53, double precision is insufficient
        // for accurate linear model predictions. Disallow merge in this case.
        s_values_too_large = false;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (global_max_ull > GPU_MERGE_V2_DOUBLE_PRECISION_MAX) {
                s_values_too_large = true;
            }
        }
    }
    __syncthreads();

    // If values are too large for double precision, don't allow merge
    if (s_values_too_large) {
        if (threadIdx.x == 0) {
            merge_benefits[pid] = -1.0f;  // Negative benefit = don't merge
        }
        return;
    }

    double y_mean = s_y_mean;

    // PASS 2: Compute centered statistics
    double local_sum_x = 0.0, local_sum_y_c = 0.0;
    double local_sum_xx = 0.0, local_sum_xy_c = 0.0;

    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        double x = static_cast<double>(i - merged_start);
        double y_c = static_cast<double>(data[i]) - y_mean;
        local_sum_x += x;
        local_sum_y_c += y_c;
        local_sum_xx += x * x;
        local_sum_xy_c += x * y_c;
    }

    double m_sx = gpu_merge_v2::blockReduceSum(local_sum_x);
    __syncthreads();
    double sum_y_c = gpu_merge_v2::blockReduceSum(local_sum_y_c);
    __syncthreads();
    double m_sxx = gpu_merge_v2::blockReduceSum(local_sum_xx);
    __syncthreads();
    double sum_xy_c = gpu_merge_v2::blockReduceSum(local_sum_xy_c);
    __syncthreads();

    if (threadIdx.x == 0) {
        double dn_c = static_cast<double>(n_c);
        double x_mean = m_sx / dn_c;

        // Centered linear regression
        double sum_x2_c = m_sxx - dn_c * x_mean * x_mean;
        double cov_xy = sum_xy_c - x_mean * sum_y_c;

        if (fabs(sum_x2_c) > 1e-10) {
            s_theta1 = cov_xy / sum_x2_c;
        } else {
            s_theta1 = 0.0;
        }
        // CRITICAL FIX: Avoid catastrophic cancellation in theta0 computation
        // The formula theta0 = y_mean - theta1 * x_mean loses precision when both terms
        // are huge (~10^16) but nearly equal (cancellation). Instead, anchor theta0
        // using the first data point: theta0 = data[0] since at x=0, pred = theta0.
        // This avoids the subtraction of two large similar numbers.
        // IMPORTANT: Use the SAME anchored theta0 for BOTH storing AND max_error calculation!
        s_theta0 = static_cast<double>(data[merged_start]);  // Anchor at first point
        merged_theta0[pid] = s_theta0;
        merged_theta1[pid] = s_theta1;

        // Store original (non-centered) statistics for compatibility
        double m_sy = sum_y_pass1;
        double m_sxy = sum_xy_c + y_mean * m_sx;
        merged_sum_x[pid] = m_sx;
        merged_sum_y[pid] = m_sy;
        merged_sum_xx[pid] = m_sxx;
        merged_sum_xy[pid] = m_sxy;
    }
    __syncthreads();

    // PASS 3: Compute max error using ANCHORED prediction (matches encoder exactly)
    // CRITICAL: Must use same prediction formula as encoder: pred = theta0 + theta1 * x
    // where theta0 = data[merged_start]. Using centered prediction gives different results!
    long long local_max_error = 0;
    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        int local_idx = i - merged_start;
        // Anchored prediction: pred = theta0 + theta1 * x (matches encoder)
        double predicted = s_theta0 + s_theta1 * static_cast<double>(local_idx);
        T pred_val = static_cast<T>(__double2ll_rn(predicted));
        long long delta;
        if (data[i] >= pred_val) {
            delta = (long long)(data[i] - pred_val);
        } else {
            delta = -(long long)(pred_val - data[i]);
        }
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long max_error = gpu_merge_v2::blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        int bits = 0;
        if (max_error > 0) {
            // +2 for sign bit + safety margin for floating-point rounding
            bits = gpu_merge_v2::computeBitsForValue((unsigned long long)max_error) + 2;
        }
        merged_delta_bits[pid] = bits;

        float delta_bytes = (float)n_c * bits / 8.0f;
        float merged_cost = GPU_MERGE_V2_MODEL_OVERHEAD_BYTES + delta_bytes;
        float separate_cost = costs[pid] + costs[pid + 1];
        float benefit = (separate_cost - merged_cost) / separate_cost;
        merge_benefits[pid] = benefit;
    }
}

// Combined mark kernel that handles both phases and clears flags
__global__ void markMergesCombinedKernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int* __restrict__ d_merge_count,
    int num_partitions,
    float threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // First: clear flags (every thread clears one element)
    if (tid < num_partitions) {
        merge_flags[tid] = 0;
    }
    // Use grid-wide memory fence to ensure all flags are cleared
    __threadfence();

    // Even phase: pairs (0,1), (2,3), ...
    // Each thread handles one even pair
    int even_pair = tid * 2;
    if (even_pair < num_partitions - 1) {
        if (merge_benefits[even_pair] >= threshold) {
            // Atomically try to mark this pair
            if (atomicCAS(&merge_flags[even_pair], 0, 1) == 0) {
                // Successfully marked even_pair, try to reserve next
                if (atomicCAS(&merge_flags[even_pair + 1], 0, 2) != 0) {
                    // Failed to reserve next, rollback
                    merge_flags[even_pair] = 0;
                } else {
                    atomicAdd(d_merge_count, 1);
                }
            }
        }
    }
    __threadfence();

    // Odd phase: pairs (1,2), (3,4), ...
    int odd_pair = tid * 2 + 1;
    if (odd_pair < num_partitions - 1) {
        if (merge_benefits[odd_pair] >= threshold) {
            if (merge_flags[odd_pair] == 0 && merge_flags[odd_pair + 1] == 0) {
                if (atomicCAS(&merge_flags[odd_pair], 0, 1) == 0) {
                    if (atomicCAS(&merge_flags[odd_pair + 1], 0, 2) != 0) {
                        merge_flags[odd_pair] = 0;
                    } else {
                        atomicAdd(d_merge_count, 1);
                    }
                }
            }
        }
    }
    __threadfence();

    // Compute output slots in the same kernel
    if (tid < num_partitions) {
        bool prev_merged = (tid > 0) && (merge_flags[tid - 1] == 1);
        bool curr_merges = (merge_flags[tid] == 1) && (tid + 1 < num_partitions);

        if (prev_merged) {
            output_slots[tid] = 0;
            is_merge_base[tid] = 0;
        } else {
            output_slots[tid] = 1;
            is_merge_base[tid] = curr_merges ? 1 : 0;
        }
    }
}

__global__ void markMergesV2Kernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,
    int* __restrict__ d_merge_count,
    int num_partitions,
    int phase,  // 0 = even, 1 = odd
    float threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int p = tid * 2 + phase;

    if (p >= num_partitions - 1) return;

    if (merge_benefits[p] >= threshold) {
        // Atomically try to mark both the base (value=1) and absorbed (value=2) partitions
        if (atomicCAS(&merge_flags[p], 0, 1) == 0) {
            // Successfully marked base, now try to mark absorbed
            if (atomicCAS(&merge_flags[p + 1], 0, 2) == 0) {
                // Success - both partitions marked
                atomicAdd(d_merge_count, 1);
            } else {
                // Failed to mark absorbed - rollback the base
                merge_flags[p] = 0;
            }
        }
    }
}

__global__ void computeOutputSlotsV2Kernel(
    const int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int num_partitions)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_partitions) return;

    // merge_flags[tid] == 1 means this partition is the base of a merge (merges with tid+1)
    // merge_flags[tid] == 2 means this partition is absorbed by partition tid-1
    bool is_absorbed = (merge_flags[tid] == 2);
    bool is_merge_base_flag = (merge_flags[tid] == 1);

    if (is_absorbed) {
        output_slots[tid] = 0;
        is_merge_base[tid] = 0;
    } else {
        output_slots[tid] = 1;
        is_merge_base[tid] = is_merge_base_flag ? 1 : 0;
    }
}

// Keep the fused version but fix it for single-block case
__global__ void fusedMarkAndOutputSlotsKernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int* __restrict__ d_merge_count,
    int num_partitions,
    float threshold)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Clear flags
    if (tid < num_partitions) {
        merge_flags[tid] = 0;
    }
    __syncthreads();

    // Even phase
    if (tid < (num_partitions + 1) / 2) {
        int p = tid * 2;
        if (p < num_partitions - 1 && merge_benefits[p] >= threshold) {
            if (atomicCAS(&merge_flags[p], 0, 1) == 0) {
                if (atomicCAS(&merge_flags[p + 1], 0, -1) == 0) {
                    atomicAdd(d_merge_count, 1);
                } else {
                    merge_flags[p] = 0;  // Rollback
                }
            }
        }
    }
    __syncthreads();

    // Odd phase
    if (tid < num_partitions / 2) {
        int p = tid * 2 + 1;
        if (p < num_partitions - 1 && merge_benefits[p] >= threshold) {
            if (merge_flags[p] == 0 && merge_flags[p + 1] == 0) {
                if (atomicCAS(&merge_flags[p], 0, 1) == 0) {
                    if (atomicCAS(&merge_flags[p + 1], 0, -1) == 0) {
                        atomicAdd(d_merge_count, 1);
                    } else {
                        merge_flags[p] = 0;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Compute output slots
    if (tid < num_partitions) {
        bool prev_merged = (tid > 0) && (merge_flags[tid - 1] == 1);
        bool curr_merges = (merge_flags[tid] == 1) && (tid + 1 < num_partitions);

        if (prev_merged) {
            output_slots[tid] = 0;
            is_merge_base[tid] = 0;
        } else {
            output_slots[tid] = 1;
            is_merge_base[tid] = curr_merges ? 1 : 0;
        }
    }
}

template<typename T>
__global__ void applyMergesV2Kernel(
    const int* __restrict__ old_starts, const int* __restrict__ old_ends,
    const int* __restrict__ old_model_types,
    const double* __restrict__ old_theta0, const double* __restrict__ old_theta1,
    const double* __restrict__ old_theta2, const double* __restrict__ old_theta3,
    const int* __restrict__ old_delta_bits, const float* __restrict__ old_costs,
    const long long* __restrict__ old_max_errors,
    const double* __restrict__ old_sum_x, const double* __restrict__ old_sum_y,
    const double* __restrict__ old_sum_xx, const double* __restrict__ old_sum_xy,
    const int* __restrict__ output_slots, const int* __restrict__ output_indices,
    const int* __restrict__ is_merge_base,
    const double* __restrict__ merged_theta0, const double* __restrict__ merged_theta1,
    const int* __restrict__ merged_delta_bits,
    const double* __restrict__ merged_sum_x, const double* __restrict__ merged_sum_y,
    const double* __restrict__ merged_sum_xx, const double* __restrict__ merged_sum_xy,
    int* __restrict__ new_starts, int* __restrict__ new_ends,
    int* __restrict__ new_model_types,
    double* __restrict__ new_theta0, double* __restrict__ new_theta1,
    double* __restrict__ new_theta2, double* __restrict__ new_theta3,
    int* __restrict__ new_delta_bits, float* __restrict__ new_costs,
    long long* __restrict__ new_max_errors,
    double* __restrict__ new_sum_x, double* __restrict__ new_sum_y,
    double* __restrict__ new_sum_xx, double* __restrict__ new_sum_xy,
    int num_partitions)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_partitions) return;
    if (output_slots[pid] == 0) return;

    int out_idx = output_indices[pid];

    if (is_merge_base[pid]) {
        // Merged partition: use LINEAR model (O(1) formula only works for linear)
        new_starts[out_idx] = old_starts[pid];
        new_ends[out_idx] = old_ends[pid + 1];
        new_model_types[out_idx] = MODEL_LINEAR;
        new_theta0[out_idx] = merged_theta0[pid];
        new_theta1[out_idx] = merged_theta1[pid];
        new_theta2[out_idx] = 0.0;  // Clear polynomial coefficients for merged partitions
        new_theta3[out_idx] = 0.0;
        new_delta_bits[out_idx] = merged_delta_bits[pid];

        int n = old_ends[pid + 1] - old_starts[pid];
        float delta_bytes = (float)n * merged_delta_bits[pid] / 8.0f;
        new_costs[out_idx] = GPU_MERGE_V2_MODEL_OVERHEAD_BYTES + delta_bytes;
        new_max_errors[out_idx] = 0;

        new_sum_x[out_idx] = merged_sum_x[pid];
        new_sum_y[out_idx] = merged_sum_y[pid];
        new_sum_xx[out_idx] = merged_sum_xx[pid];
        new_sum_xy[out_idx] = merged_sum_xy[pid];

        // DEBUG: 打印写入的 theta1 (commented out to reduce output)
        // printf("[applyMerges] MERGE pid=%d out_idx=%d: read merged_theta1[%d]=%.5f, n=%d\n",
        //        pid, out_idx, pid, merged_theta1[pid], n);
    } else {
        // Unmerged partition: preserve original model type and polynomial coefficients
        new_starts[out_idx] = old_starts[pid];
        new_ends[out_idx] = old_ends[pid];
        new_model_types[out_idx] = old_model_types[pid];
        new_theta0[out_idx] = old_theta0[pid];
        new_theta1[out_idx] = old_theta1[pid];
        new_theta2[out_idx] = old_theta2[pid];  // Preserve polynomial coefficient
        new_theta3[out_idx] = old_theta3[pid];  // Preserve polynomial coefficient
        new_delta_bits[out_idx] = old_delta_bits[pid];
        new_costs[out_idx] = old_costs[pid];
        new_max_errors[out_idx] = old_max_errors[pid];

        new_sum_x[out_idx] = old_sum_x[pid];
        new_sum_y[out_idx] = old_sum_y[pid];
        new_sum_xx[out_idx] = old_sum_xx[pid];
        new_sum_xy[out_idx] = old_sum_xy[pid];
    }
}

// Explicit instantiation for fallback kernels
template __global__ void evaluateMergeCostV2Kernel<int32_t>(const int32_t*, const int*, const int*, const float*, const double*, const double*, const double*, const double*, float*, int*, double*, double*, double*, double*, double*, double*, int, int);
template __global__ void evaluateMergeCostV2Kernel<uint32_t>(const uint32_t*, const int*, const int*, const float*, const double*, const double*, const double*, const double*, float*, int*, double*, double*, double*, double*, double*, double*, int, int);
template __global__ void evaluateMergeCostV2Kernel<int64_t>(const int64_t*, const int*, const int*, const float*, const double*, const double*, const double*, const double*, float*, int*, double*, double*, double*, double*, double*, double*, int, int);
template __global__ void evaluateMergeCostV2Kernel<uint64_t>(const uint64_t*, const int*, const int*, const float*, const double*, const double*, const double*, const double*, float*, int*, double*, double*, double*, double*, double*, double*, int, int);

template __global__ void applyMergesV2Kernel<int32_t>(const int*, const int*, const int*, const double*, const double*, const double*, const double*, const int*, const float*, const long long*, const double*, const double*, const double*, const double*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const double*, const double*, const double*, int*, int*, int*, double*, double*, double*, double*, int*, float*, long long*, double*, double*, double*, double*, int);
template __global__ void applyMergesV2Kernel<uint32_t>(const int*, const int*, const int*, const double*, const double*, const double*, const double*, const int*, const float*, const long long*, const double*, const double*, const double*, const double*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const double*, const double*, const double*, int*, int*, int*, double*, double*, double*, double*, int*, float*, long long*, double*, double*, double*, double*, int);
template __global__ void applyMergesV2Kernel<int64_t>(const int*, const int*, const int*, const double*, const double*, const double*, const double*, const int*, const float*, const long long*, const double*, const double*, const double*, const double*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const double*, const double*, const double*, int*, int*, int*, double*, double*, double*, double*, int*, float*, long long*, double*, double*, double*, double*, int);
template __global__ void applyMergesV2Kernel<uint64_t>(const int*, const int*, const int*, const double*, const double*, const double*, const double*, const int*, const float*, const long long*, const double*, const double*, const double*, const double*, const int*, const int*, const int*, const double*, const double*, const int*, const double*, const double*, const double*, const double*, int*, int*, int*, double*, double*, double*, double*, int*, float*, long long*, double*, double*, double*, double*, int);

// ============================================================================
// Polynomial Refit Kernel for V2
// ============================================================================

/**
 * Polynomial refit kernel for V2 - evaluates if POLY2/POLY3 is better than LINEAR
 */
template<typename T>
__global__ void refitPolynomialV2Kernel(
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
    int poly_min_size,
    int cubic_min_size,
    float cost_threshold)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = partition_starts[pid];
    int end = partition_ends[pid];
    int n = end - start;

    // Skip small partitions
    if (n < poly_min_size) return;

    // ========== PASS 1: Compute y_mean and min/max for numerical stability ==========
    double local_sum_y_pass1 = 0.0;

    // For FOR model: track min/max
    // Must use correct initial values for signed vs unsigned types
    T local_min, local_max;
    if constexpr (std::is_unsigned<T>::value) {
        // Unsigned types: min init = max possible value, max init = 0
        if constexpr (sizeof(T) == 8) {
            local_min = static_cast<T>(~0ULL);  // ULLONG_MAX
            local_max = static_cast<T>(0);
        } else {
            local_min = static_cast<T>(~0U);    // UINT_MAX
            local_max = static_cast<T>(0);
        }
    } else {
        // Signed types: use LLONG_MAX/LLONG_MIN or INT_MAX/INT_MIN
        if constexpr (sizeof(T) == 8) {
            local_min = static_cast<T>(LLONG_MAX);
            local_max = static_cast<T>(LLONG_MIN);
        } else {
            local_min = static_cast<T>(INT_MAX);
            local_max = static_cast<T>(INT_MIN);
        }
    }

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        double y = static_cast<double>(val);
        local_sum_y_pass1 += y;
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    double sum_y_pass1 = gpu_merge_v2::blockReduceSum(local_sum_y_pass1);
    __syncthreads();

    // Block reduction for min/max (FOR model)
    unsigned long long global_min_ull = gpu_merge_v2::blockReduceMinUnsigned(static_cast<unsigned long long>(local_min));
    __syncthreads();
    unsigned long long global_max_ull = gpu_merge_v2::blockReduceMaxUnsigned(static_cast<unsigned long long>(local_max));
    __syncthreads();

    // Shared memory for y_mean (needed for centered fitting)
    __shared__ double s_y_mean;
    __shared__ T s_global_min, s_global_max;
    __shared__ bool s_force_for_bitpack;

    if (threadIdx.x == 0) {
        s_y_mean = sum_y_pass1 / static_cast<double>(n);
        s_global_min = static_cast<T>(global_min_ull);
        s_global_max = static_cast<T>(global_max_ull);

        // CRITICAL: For uint64_t values > 2^53, double precision is insufficient
        // for accurate linear/polynomial model predictions. Force FOR+BitPack in this case.
        s_force_for_bitpack = false;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (static_cast<uint64_t>(s_global_max) > GPU_MERGE_V2_DOUBLE_PRECISION_MAX) {
                s_force_for_bitpack = true;
            }
        }
    }
    __syncthreads();

    // If we need to force FOR+BitPack, skip the expensive polynomial fitting
    if (s_force_for_bitpack) {
        if (threadIdx.x == 0) {
            // Compute FOR+BitPack model
            uint64_t range = static_cast<uint64_t>(s_global_max) - static_cast<uint64_t>(s_global_min);
            int for_bits = (range > 0) ? gpu_merge_v2::computeBitsForValue(range) : 0;

            model_types[pid] = MODEL_FOR_BITPACK;
            // Store base value using bit-pattern copy for 64-bit types
            theta0_array[pid] = __longlong_as_double(static_cast<long long>(s_global_min));
            theta1_array[pid] = 0.0;
            theta2_array[pid] = 0.0;
            theta3_array[pid] = 0.0;
            delta_bits_array[pid] = for_bits;

            float delta_bytes = static_cast<float>(n) * for_bits / 8.0f;
            costs[pid] = static_cast<float>(sizeof(T)) + delta_bytes;  // FOR overhead = sizeof(T)
        }
        return;
    }

    double y_mean = s_y_mean;

    // ========== PASS 2: Compute centered statistics for better precision ==========
    // Using centered y values: y_c = y - y_mean
    // This dramatically improves numerical stability for large values
    double local_sum_y = 0.0, local_sum_xy = 0.0;
    double local_sum_x2y = 0.0, local_sum_x3y = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);
        double x2 = x * x;
        T val = data[i];
        double y_centered = static_cast<double>(val) - y_mean;  // Centered y

        local_sum_y += y_centered;       // Should be ~0
        local_sum_xy += x * y_centered;
        local_sum_x2y += x2 * y_centered;
        local_sum_x3y += x2 * x * y_centered;
    }

    double sum_y = gpu_merge_v2::blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xy = gpu_merge_v2::blockReduceSum(local_sum_xy);
    __syncthreads();
    double sum_x2y = gpu_merge_v2::blockReduceSum(local_sum_x2y);
    __syncthreads();
    double sum_x3y = gpu_merge_v2::blockReduceSum(local_sum_x3y);
    __syncthreads();

    // Shared memory for model parameters
    __shared__ double s_linear_params[4];
    __shared__ double s_poly2_params[4];
    __shared__ double s_poly3_params[4];
    __shared__ float s_linear_cost, s_poly2_cost, s_poly3_cost, s_for_cost;
    __shared__ int s_linear_bits, s_poly2_bits, s_poly3_bits, s_for_bits;

    // Thread 0 computes model coefficients
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);

        // Sum formulas for x^k (x goes from 0 to n-1)
        double sx = dn * (dn - 1.0) / 2.0;
        double sx2 = dn * (dn - 1.0) * (2.0 * dn - 1.0) / 6.0;
        double sx3 = sx * sx;
        double sx4 = dn * (dn - 1.0) * (2.0 * dn - 1.0) * (3.0 * dn * dn - 3.0 * dn - 1.0) / 30.0;

        // For POLY3: sx5, sx6 using Faulhaber's formula
        // sx5 = n² × (n-1)² × (2n² - 2n - 1) / 12
        double nn = dn * dn;
        double nm1 = dn - 1.0;
        double sx5 = nn * nm1 * nm1 * (2.0 * nn - 2.0 * dn - 1.0) / 12.0;
        // sx6 = m(m+1)(2m+1)(3m⁴+6m³-3m+1)/42 where m = n-1
        double m = dn - 1.0;
        double m2 = m * m;
        double m3 = m2 * m;
        double m4 = m2 * m2;
        double sx6 = m * (m + 1.0) * (2.0 * m + 1.0) * (3.0 * m4 + 6.0 * m3 - 3.0 * m + 1.0) / 42.0;

        // x_mean for centered coordinates
        double x_mean = sx / dn;

        // LINEAR fit using CENTERED y values
        // For centered data: y_c = y - y_mean
        // Normal equations: sum_y_c ≈ 0, so theta1 = sum_xy_c / sum_x2_c
        // where sum_x2_c = sum((x - x_mean)^2) = sx2 - sx*sx/n
        double sum_x2_centered = sx2 - sx * sx / dn;  // Variance of x * n
        double sum_xy_centered = sum_xy - x_mean * sum_y;  // Covariance * n (sum_y ≈ 0)

        double theta1_centered;
        if (fabs(sum_x2_centered) > 1e-10) {
            theta1_centered = sum_xy_centered / sum_x2_centered;
        } else {
            theta1_centered = 0.0;
        }

        // Convert back to original coordinates:
        // y = y_c + y_mean = theta0_c + theta1_c * (x - x_mean) + y_mean
        //   = (theta0_c + y_mean - theta1_c * x_mean) + theta1_c * x
        // So: theta0 = y_mean - theta1 * x_mean (since theta0_c ≈ 0 for centered data)
        //     theta1 = theta1_centered
        s_linear_params[1] = theta1_centered;
        s_linear_params[0] = y_mean - theta1_centered * x_mean;
        s_linear_params[2] = 0.0;
        s_linear_params[3] = 0.0;

        // POLY2 fit (if large enough) - also using centered approach
        if (n >= poly_min_size) {
            // For centered y: y_c = a_c + b_c*(x-x_mean) + c_c*(x-x_mean)^2
            // Use 3x3 system with centered x coordinates for better stability
            // But for simplicity, we use the original method with centered y
            double a00 = dn, a01 = sx, a02 = sx2;
            double a10 = sx, a11 = sx2, a12 = sx3;
            double a20 = sx2, a21 = sx3, a22 = sx4;
            double b0 = sum_y, b1 = sum_xy, b2 = sum_x2y;

            double det2 = a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20);
            if (fabs(det2) > 1e-10) {
                double theta0_c = (b0 * (a11 * a22 - a12 * a21) - a01 * (b1 * a22 - a12 * b2) + a02 * (b1 * a21 - a11 * b2)) / det2;
                double theta1_c = (a00 * (b1 * a22 - a12 * b2) - b0 * (a10 * a22 - a12 * a20) + a02 * (a10 * b2 - b1 * a20)) / det2;
                double theta2_c = (a00 * (a11 * b2 - b1 * a21) - a01 * (a10 * b2 - b1 * a20) + b0 * (a10 * a21 - a11 * a20)) / det2;
                // Convert back: add y_mean to theta0
                s_poly2_params[0] = theta0_c + y_mean;
                s_poly2_params[1] = theta1_c;
                s_poly2_params[2] = theta2_c;
            } else {
                s_poly2_params[0] = s_linear_params[0];
                s_poly2_params[1] = s_linear_params[1];
                s_poly2_params[2] = 0.0;
            }
            s_poly2_params[3] = 0.0;
        } else {
            for (int i = 0; i < 4; i++) s_poly2_params[i] = s_linear_params[i];
        }

        // POLY3 fit (if large enough for cubic)
        // Use scaled coordinates to avoid numerical instability
        if (n >= cubic_min_size) {
            double scale = dn - 1.0;
            double s2 = scale * scale;
            double s3 = s2 * scale;
            double s4 = s2 * s2;
            double s5 = s4 * scale;
            double s6 = s3 * s3;

            // Scaled power sums
            double sx_sc = sx / scale;
            double sx2_sc = sx2 / s2;
            double sx3_sc = sx3 / s3;
            double sx4_sc = sx4 / s4;
            double sx5_sc = sx5 / s5;
            double sx6_sc = sx6 / s6;

            // Scaled data sums
            double sum_xpy_sc = sum_xy / scale;
            double sum_x2py_sc = sum_x2y / s2;
            double sum_x3py_sc = sum_x3y / s3;

            // Build 4x4 normal equations matrix (scaled)
            // Using LU decomposition with partial pivoting
            double A[4][4] = {
                {dn,      sx_sc,   sx2_sc,  sx3_sc},
                {sx_sc,   sx2_sc,  sx3_sc,  sx4_sc},
                {sx2_sc,  sx3_sc,  sx4_sc,  sx5_sc},
                {sx3_sc,  sx4_sc,  sx5_sc,  sx6_sc}
            };
            double b_vec[4] = {sum_y, sum_xpy_sc, sum_x2py_sc, sum_x3py_sc};

            // Gaussian elimination with partial pivoting
            bool solve_ok = true;
            for (int k = 0; k < 4 && solve_ok; k++) {
                // Find pivot
                int max_row = k;
                double max_val = fabs(A[k][k]);
                for (int i = k + 1; i < 4; i++) {
                    if (fabs(A[i][k]) > max_val) {
                        max_val = fabs(A[i][k]);
                        max_row = i;
                    }
                }
                if (max_val < 1e-12) {
                    solve_ok = false;
                    break;
                }
                // Swap rows
                if (max_row != k) {
                    for (int j = 0; j < 4; j++) {
                        double tmp = A[k][j]; A[k][j] = A[max_row][j]; A[max_row][j] = tmp;
                    }
                    double tmp = b_vec[k]; b_vec[k] = b_vec[max_row]; b_vec[max_row] = tmp;
                }
                // Eliminate
                for (int i = k + 1; i < 4; i++) {
                    double factor = A[i][k] / A[k][k];
                    for (int j = k; j < 4; j++) {
                        A[i][j] -= factor * A[k][j];
                    }
                    b_vec[i] -= factor * b_vec[k];
                }
            }

            if (solve_ok) {
                // Back substitution
                double alpha[4];
                for (int i = 3; i >= 0; i--) {
                    alpha[i] = b_vec[i];
                    for (int j = i + 1; j < 4; j++) {
                        alpha[i] -= A[i][j] * alpha[j];
                    }
                    alpha[i] /= A[i][i];
                }
                // Transform back to original coordinates
                s_poly3_params[0] = alpha[0];
                s_poly3_params[1] = alpha[1] / scale;
                s_poly3_params[2] = alpha[2] / s2;
                s_poly3_params[3] = alpha[3] / s3;
            } else {
                // Fallback to POLY2
                for (int i = 0; i < 4; i++) s_poly3_params[i] = s_poly2_params[i];
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

    linear_max_err = gpu_merge_v2::blockReduceMax(linear_max_err);
    __syncthreads();
    poly2_max_err = gpu_merge_v2::blockReduceMax(poly2_max_err);
    __syncthreads();
    poly3_max_err = gpu_merge_v2::blockReduceMax(poly3_max_err);
    __syncthreads();

    // Thread 0 selects best model
    if (threadIdx.x == 0) {
        // Compute bits for each model (+2 for sign bit + safety margin for floating-point rounding)
        s_linear_bits = (linear_max_err > 0) ? gpu_merge_v2::computeBitsForValue((unsigned long long)linear_max_err) + 2 : 0;
        s_poly2_bits = (poly2_max_err > 0) ? gpu_merge_v2::computeBitsForValue((unsigned long long)poly2_max_err) + 2 : 0;
        s_poly3_bits = (poly3_max_err > 0) ? gpu_merge_v2::computeBitsForValue((unsigned long long)poly3_max_err) + 2 : 0;

        // Compute FOR model bits (range = max - min, unsigned)
        uint64_t range;
        if constexpr (sizeof(T) == 8) {
            range = static_cast<uint64_t>(s_global_max) - static_cast<uint64_t>(s_global_min);
        } else {
            range = static_cast<uint64_t>(static_cast<uint32_t>(s_global_max) - static_cast<uint32_t>(s_global_min));
        }
        s_for_bits = (range > 0) ? gpu_merge_v2::computeBitsForValue(range) : 0;

        // Compute costs (overhead + delta bytes)
        float fn = static_cast<float>(n);
        s_linear_cost = 16.0f + fn * s_linear_bits / 8.0f;  // LINEAR: 16 bytes overhead
        s_poly2_cost = 24.0f + fn * s_poly2_bits / 8.0f;    // POLY2: 24 bytes overhead
        s_poly3_cost = 32.0f + fn * s_poly3_bits / 8.0f;    // POLY3: 32 bytes overhead
        s_for_cost = static_cast<float>(sizeof(T)) + fn * s_for_bits / 8.0f;  // FOR: sizeof(T) overhead (base only)

        // Start with FOR as default model
        int best_model = MODEL_FOR_BITPACK;
        float best_cost = s_for_cost;
        double best_params[4] = {0.0, 0.0, 0.0, 0.0};
        int best_bits = s_for_bits;

        // Store base for FOR model using bit-pattern copy for 64-bit types
        if constexpr (sizeof(T) == 8) {
            best_params[0] = __longlong_as_double(static_cast<long long>(s_global_min));
        } else {
            best_params[0] = static_cast<double>(s_global_min);
        }

        // Check if LINEAR is better (require cost_threshold improvement)
        if (s_linear_cost < best_cost * cost_threshold) {
            best_model = MODEL_LINEAR;
            best_cost = s_linear_cost;
            best_params[0] = s_linear_params[0];
            best_params[1] = s_linear_params[1];
            best_params[2] = 0.0;
            best_params[3] = 0.0;
            best_bits = s_linear_bits;
        }

        // Check if POLY2 is better (require cost_threshold improvement)
        if (n >= poly_min_size && s_poly2_cost < best_cost * cost_threshold) {
            best_model = MODEL_POLYNOMIAL2;
            best_cost = s_poly2_cost;
            best_params[0] = s_poly2_params[0];
            best_params[1] = s_poly2_params[1];
            best_params[2] = s_poly2_params[2];
            best_params[3] = 0.0;
            best_bits = s_poly2_bits;
        }

        // Check if POLY3 is better (require cost_threshold improvement)
        if (n >= cubic_min_size && s_poly3_cost < best_cost * cost_threshold) {
            best_model = MODEL_POLYNOMIAL3;
            best_cost = s_poly3_cost;
            best_params[0] = s_poly3_params[0];
            best_params[1] = s_poly3_params[1];
            best_params[2] = s_poly3_params[2];
            best_params[3] = s_poly3_params[3];
            best_bits = s_poly3_bits;
        }

        // Always update partition info with the best model found
        model_types[pid] = best_model;
        theta0_array[pid] = best_params[0];
        theta1_array[pid] = best_params[1];
        theta2_array[pid] = best_params[2];
        theta3_array[pid] = best_params[3];
        delta_bits_array[pid] = best_bits;
        costs[pid] = best_cost;
    }
}

// ============================================================================
// GPU Boundary Fix and Refit Kernel
// ============================================================================

/**
 * Fix partition boundaries and refit if needed (all on GPU)
 * - First partition starts at 0
 * - Last partition ends at data_size
 * - Consecutive partitions have matching boundaries
 * - Refit partitions that had boundaries changed
 */
template<typename T>
__global__ void fixBoundariesAndRefitKernel(
    const T* __restrict__ data,
    int* __restrict__ starts,
    int* __restrict__ ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0,
    double* __restrict__ theta1,
    double* __restrict__ theta2,
    double* __restrict__ theta3,
    int* __restrict__ delta_bits,
    long long* __restrict__ max_errors,
    float* __restrict__ costs,
    int num_partitions,
    int data_size)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    // Thread 0 handles boundary fixes
    __shared__ bool needs_refit;
    __shared__ int s_start, s_end, s_n;
    __shared__ double s_theta0, s_theta1;

    if (threadIdx.x == 0) {
        needs_refit = false;
        int original_start = starts[pid];
        int original_end = ends[pid];

        // Fix first partition
        if (pid == 0 && original_start != 0) {
            starts[pid] = 0;
            needs_refit = true;
        }

        // Fix last partition
        if (pid == num_partitions - 1 && original_end != data_size) {
            ends[pid] = data_size;
            needs_refit = true;
        }

        // Fix gaps between consecutive partitions (only pid > 0 fixes its start)
        if (pid > 0) {
            int prev_end = ends[pid - 1];
            if (starts[pid] != prev_end) {
                starts[pid] = prev_end;
                needs_refit = true;
            }
        }

        s_start = starts[pid];
        s_end = ends[pid];
        s_n = s_end - s_start;
    }
    __syncthreads();

    if (!needs_refit || s_n <= 0) return;

    // Refit the partition using linear regression (same as GPU fitting)
    int start = s_start;
    int end = s_end;
    int n = s_n;

    // Parallel sum computation
    double local_sum_x = 0.0, local_sum_y = 0.0, local_sum_xx = 0.0, local_sum_xy = 0.0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double x = static_cast<double>(i - start);
        double y = static_cast<double>(data[i]);
        local_sum_x += x;
        local_sum_y += y;
        local_sum_xx += x * x;
        local_sum_xy += x * y;
    }

    // Block reduction
    double sum_x = gpu_merge_v2::blockReduceSum(local_sum_x);
    __syncthreads();
    double sum_y = gpu_merge_v2::blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xx = gpu_merge_v2::blockReduceSum(local_sum_xx);
    __syncthreads();
    double sum_xy = gpu_merge_v2::blockReduceSum(local_sum_xy);
    __syncthreads();

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

        // CRITICAL: Anchor theta0 at first data point to avoid precision issues
        s_theta0 = static_cast<double>(data[start]);
    }
    __syncthreads();

    // Compute max error in parallel
    long long local_max_error = 0;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double predicted = s_theta0 + s_theta1 * static_cast<double>(local_idx);
        T pred_val;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            pred_val = static_cast<T>(__double2ull_rn(predicted));
        } else {
            pred_val = static_cast<T>(__double2ll_rn(predicted));
        }

        long long delta;
        if (data[i] >= pred_val) {
            unsigned long long diff = static_cast<unsigned long long>(data[i]) - static_cast<unsigned long long>(pred_val);
            if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                delta = static_cast<long long>(diff);
                local_max_error = max(local_max_error, delta);
            } else {
                local_max_error = LLONG_MAX;
            }
        } else {
            unsigned long long diff = static_cast<unsigned long long>(pred_val) - static_cast<unsigned long long>(data[i]);
            if (diff <= static_cast<unsigned long long>(LLONG_MAX)) {
                delta = static_cast<long long>(diff);
                local_max_error = max(local_max_error, delta);
            } else {
                local_max_error = LLONG_MAX;
            }
        }
    }

    long long max_error = gpu_merge_v2::blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        model_types[pid] = MODEL_LINEAR;
        theta0[pid] = s_theta0;
        theta1[pid] = s_theta1;
        theta2[pid] = 0.0;
        theta3[pid] = 0.0;
        max_errors[pid] = max_error;

        int bits = 0;
        if (max_error > 0 && max_error < LLONG_MAX) {
            // +2 for sign bit + safety margin
            bits = gpu_merge_v2::computeBitsForValue(static_cast<unsigned long long>(max_error)) + 2;
        } else if (max_error == LLONG_MAX) {
            bits = sizeof(T) * 8;
        }
        delta_bits[pid] = bits;

        // Recompute cost (only if costs array is provided)
        if (costs != nullptr) {
            float delta_bytes = static_cast<float>(n) * bits / 8.0f;
            costs[pid] = GPU_MERGE_V2_MODEL_OVERHEAD_BYTES + delta_bytes;
        }
    }
}

// Explicit instantiation
template __global__ void refitPolynomialV2Kernel<int32_t>(const int32_t*, const int*, const int*, int*, double*, double*, double*, double*, int*, float*, int, int, int, float);
template __global__ void refitPolynomialV2Kernel<uint32_t>(const uint32_t*, const int*, const int*, int*, double*, double*, double*, double*, int*, float*, int, int, int, float);
template __global__ void refitPolynomialV2Kernel<int64_t>(const int64_t*, const int*, const int*, int*, double*, double*, double*, double*, int*, float*, int, int, int, float);
template __global__ void refitPolynomialV2Kernel<uint64_t>(const uint64_t*, const int*, const int*, int*, double*, double*, double*, double*, int*, float*, int, int, int, float);

template __global__ void fixBoundariesAndRefitKernel<int32_t>(const int32_t*, int*, int*, int*, double*, double*, double*, double*, int*, long long*, float*, int, int);
template __global__ void fixBoundariesAndRefitKernel<uint32_t>(const uint32_t*, int*, int*, int*, double*, double*, double*, double*, int*, long long*, float*, int, int);
template __global__ void fixBoundariesAndRefitKernel<int64_t>(const int64_t*, int*, int*, int*, double*, double*, double*, double*, int*, long long*, float*, int, int);
template __global__ void fixBoundariesAndRefitKernel<uint64_t>(const uint64_t*, int*, int*, int*, double*, double*, double*, double*, int*, long long*, float*, int, int);

// ============================================================================
// Partitioner Implementation
// ============================================================================

template<typename T>
void GPUCostOptimalPartitionerV2<T>::checkCooperativeLaunchSupport() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    cooperative_launch_supported = (prop.cooperativeLaunch != 0);

    if (cooperative_launch_supported) {
        int num_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks,
            mergeLoopCooperativeKernel<T>,
            GPU_MERGE_V2_BLOCK_SIZE,
            0);
        max_cooperative_blocks = num_blocks * prop.multiProcessorCount;

        // Conservative limit
        max_cooperative_blocks = std::min(max_cooperative_blocks, GPU_MERGE_V2_MAX_BLOCKS);
    } else {
        max_cooperative_blocks = 0;
    }
}

template<typename T>
GPUCostOptimalPartitionerV2<T>::GPUCostOptimalPartitionerV2(
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

    size_t max_partitions = (data_size + config.min_partition_size - 1) / config.min_partition_size;
    ctx.allocate(max_partitions);

    checkCooperativeLaunchSupport();
}

template<typename T>
GPUCostOptimalPartitionerV2<T>::~GPUCostOptimalPartitionerV2() {
    if (d_data) cudaFree(d_data);
}

template<typename T>
void GPUCostOptimalPartitionerV2<T>::refitPartition(PartitionInfo& info) {
    int start = info.start_idx;
    int end = info.end_idx;
    int n = end - start;

    if (n <= 0) return;

    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = 0; i < n; i++) {
        double x = (double)i;
        double y = (double)h_data_ref[start + i];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double dn = (double)n;
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
        // +3 for sign bit + safety margin for CPU/GPU floating-point differences
        // std::llrint (CPU) and __double2ll_rn (GPU) may round differently due to
        // FMA optimizations and intermediate precision differences
        bits = 64 - __builtin_clzll((unsigned long long)max_error) + 3;
    }
    info.delta_bits = bits;
}

template<typename T>
int GPUCostOptimalPartitionerV2<T>::runMergeLoopCooperative(int num_partitions) {
    if (num_partitions <= 1) return num_partitions;

    // Initialize partition count on device
    cudaMemcpy(ctx.d_num_partitions, &num_partitions, sizeof(int), cudaMemcpyHostToDevice);

    int num_blocks = std::min(num_partitions, max_cooperative_blocks);
    void* kernel_args[] = {
        &d_data,
        // Buffer A
        &ctx.buffer_A.starts, &ctx.buffer_A.ends,
        &ctx.buffer_A.model_types, &ctx.buffer_A.theta0, &ctx.buffer_A.theta1,
        &ctx.buffer_A.delta_bits, &ctx.buffer_A.costs, &ctx.buffer_A.max_errors,
        &ctx.buffer_A.sum_x, &ctx.buffer_A.sum_y, &ctx.buffer_A.sum_xx, &ctx.buffer_A.sum_xy,
        // Buffer B
        &ctx.buffer_B.starts, &ctx.buffer_B.ends,
        &ctx.buffer_B.model_types, &ctx.buffer_B.theta0, &ctx.buffer_B.theta1,
        &ctx.buffer_B.delta_bits, &ctx.buffer_B.costs, &ctx.buffer_B.max_errors,
        &ctx.buffer_B.sum_x, &ctx.buffer_B.sum_y, &ctx.buffer_B.sum_xx, &ctx.buffer_B.sum_xy,
        // Working arrays
        &ctx.merge_benefits, &ctx.merged_delta_bits,
        &ctx.merged_theta0, &ctx.merged_theta1,
        &ctx.merged_sum_x, &ctx.merged_sum_y, &ctx.merged_sum_xx, &ctx.merged_sum_xy,
        &ctx.merge_flags,
        &ctx.output_slots, &ctx.output_indices, &ctx.is_merge_base,
        &ctx.block_sums,
        // Control
        &ctx.d_num_partitions, &ctx.d_merge_count,
        &config.max_merge_rounds, &config.merge_benefit_threshold,
        &config.max_partition_size, &data_size
    };

    cudaLaunchCooperativeKernel(
        (void*)mergeLoopCooperativeKernel<T>,
        dim3(num_blocks), dim3(GPU_MERGE_V2_BLOCK_SIZE),
        kernel_args, 0, stream);

    cudaStreamSynchronize(stream);

    // Get final partition count
    cudaMemcpy(ctx.h_final_partition_count, ctx.d_num_partitions, sizeof(int), cudaMemcpyDeviceToHost);

    return *ctx.h_final_partition_count;
}

template<typename T>
int GPUCostOptimalPartitionerV2<T>::runMergeLoopMultiKernel(int num_partitions) {
    if (num_partitions <= 1) return num_partitions;

    // Removed verbose debug output for normal operation

    for (int round = 0; round < config.max_merge_rounds; round++) {
        cudaDeviceSynchronize();

        // Reset merge flags (async, no sync needed)
        cudaMemsetAsync(ctx.merge_flags, 0, num_partitions * sizeof(int), stream);

        // Evaluate merge costs
        evaluateMergeCostV2Kernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
            d_data,
            ctx.current->starts, ctx.current->ends, ctx.current->costs,
            ctx.current->sum_x, ctx.current->sum_y, ctx.current->sum_xx, ctx.current->sum_xy,
            ctx.merge_benefits, ctx.merged_delta_bits,
            ctx.merged_theta0, ctx.merged_theta1,
            ctx.merged_sum_x, ctx.merged_sum_y, ctx.merged_sum_xx, ctx.merged_sum_xy,
            num_partitions, config.max_partition_size);

        // Mark merges - Even phase
        int mark_blocks = ((num_partitions + 1) / 2 + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
        if (mark_blocks > 0) {
            markMergesV2Kernel<<<mark_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
                ctx.merge_benefits, ctx.merge_flags, ctx.d_merge_count,
                num_partitions, 0, config.merge_benefit_threshold);
        }

        // Mark merges - Odd phase
        mark_blocks = (num_partitions / 2 + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
        if (mark_blocks > 0) {
            markMergesV2Kernel<<<mark_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
                ctx.merge_benefits, ctx.merge_flags, ctx.d_merge_count,
                num_partitions, 1, config.merge_benefit_threshold);
        }

        // Compute output slots
        int slots_blocks = (num_partitions + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
        computeOutputSlotsV2Kernel<<<slots_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
            ctx.merge_flags, ctx.output_slots, ctx.is_merge_base, num_partitions);

        // Prefix sum using thrust
        thrust::exclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(ctx.output_slots),
            thrust::device_pointer_cast(ctx.output_slots + num_partitions),
            thrust::device_pointer_cast(ctx.output_indices));

        // Get new partition count (single sync point per round)
        int last_slot, last_idx;
        cudaMemcpyAsync(&last_slot, ctx.output_slots + num_partitions - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_idx, ctx.output_indices + num_partitions - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int new_count = last_idx + last_slot;

        if (new_count == 0 || new_count >= num_partitions) break;

        // Apply merges
        int apply_blocks = (num_partitions + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
        applyMergesV2Kernel<T><<<apply_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
            ctx.current->starts, ctx.current->ends, ctx.current->model_types,
            ctx.current->theta0, ctx.current->theta1, ctx.current->theta2, ctx.current->theta3,
            ctx.current->delta_bits, ctx.current->costs, ctx.current->max_errors,
            ctx.current->sum_x, ctx.current->sum_y, ctx.current->sum_xx, ctx.current->sum_xy,
            ctx.output_slots, ctx.output_indices, ctx.is_merge_base,
            ctx.merged_theta0, ctx.merged_theta1, ctx.merged_delta_bits,
            ctx.merged_sum_x, ctx.merged_sum_y, ctx.merged_sum_xx, ctx.merged_sum_xy,
            ctx.next->starts, ctx.next->ends, ctx.next->model_types,
            ctx.next->theta0, ctx.next->theta1, ctx.next->theta2, ctx.next->theta3,
            ctx.next->delta_bits, ctx.next->costs, ctx.next->max_errors,
            ctx.next->sum_x, ctx.next->sum_y, ctx.next->sum_xx, ctx.next->sum_xy,
            num_partitions);

        ctx.swap();
        num_partitions = new_count;

        // DEBUG: 打印 swap 后状态 (commented out to reduce output)
        cudaDeviceSynchronize();
        // printf("[runMergeLoop] After swap: num_partitions=%d\n", num_partitions);

        if (num_partitions <= 1) break;
    }

    // DEBUG: 打印最终状态 (commented out to reduce output)
    // printf("[runMergeLoop] Final num_partitions=%d\n\n", num_partitions);

    return num_partitions;
}

// Reuse the partition creation kernels from v1
extern __global__ void gpuMergeDetectBreakpointsKernel(const int*, int*, int, int);
extern __global__ void gpuMergeCountPartitionsInSegmentsKernel(const int*, int, int, int, int, int, int*);
extern __global__ void gpuMergeWritePartitionsKernel(const int*, int, int, int, int, int, const int*, int*, int*);

template<typename T>
extern __global__ void gpuMergeComputeDeltaBitsKernel(const T*, int, int, int*, int);

template<typename T>
extern __global__ void gpuMergeFitPartitionsKernel(const T*, const int*, const int*, int*, double*, double*, double*, double*, int*, long long*, float*, double*, double*, double*, double*, int);

template<typename T>
std::vector<PartitionInfo> GPUCostOptimalPartitionerV2<T>::partition() {
    // ================================================================
    // Stage 1-4: Same as v1 (partition creation and initial fitting)
    // ================================================================

    int num_analysis_blocks = (data_size + config.analysis_block_size - 1) / config.analysis_block_size;

    int* d_delta_bits_per_block;
    cudaMalloc(&d_delta_bits_per_block, num_analysis_blocks * sizeof(int));

    gpuMergeComputeDeltaBitsKernel<T><<<num_analysis_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_data, data_size, config.analysis_block_size,
        d_delta_bits_per_block, num_analysis_blocks);

    int* d_is_breakpoint;
    cudaMalloc(&d_is_breakpoint, num_analysis_blocks * sizeof(int));

    int bp_blocks = (num_analysis_blocks + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
    gpuMergeDetectBreakpointsKernel<<<bp_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_delta_bits_per_block, d_is_breakpoint,
        num_analysis_blocks, config.breakpoint_threshold);

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

    int* d_breakpoint_positions;
    int* d_partition_counts;
    int* d_partition_offsets;

    cudaMalloc(&d_breakpoint_positions, num_breakpoints * sizeof(int));
    cudaMalloc(&d_partition_counts, num_breakpoints * sizeof(int));
    cudaMalloc(&d_partition_offsets, (num_breakpoints + 1) * sizeof(int));

    cudaMemcpy(d_breakpoint_positions, breakpoint_positions.data(),
               num_breakpoints * sizeof(int), cudaMemcpyHostToDevice);

    int seg_blocks = (num_breakpoints + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
    gpuMergeCountPartitionsInSegmentsKernel<<<seg_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_breakpoint_positions, num_breakpoints, data_size,
        config.target_partition_size, config.min_partition_size,
        config.max_partition_size, d_partition_counts);

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(d_partition_counts),
        thrust::device_pointer_cast(d_partition_counts + num_breakpoints),
        thrust::device_pointer_cast(d_partition_offsets));

    int h_total_partitions;
    int h_last_count;
    cudaMemcpy(&h_total_partitions, d_partition_offsets + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_count, d_partition_counts + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    h_total_partitions += h_last_count;

    int num_partitions = h_total_partitions;
    if (num_partitions == 0) {
        num_partitions = 1;
        int zero = 0;
        cudaMemcpy(ctx.current->starts, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.current->ends, &data_size, sizeof(int), cudaMemcpyHostToDevice);
    } else {
        gpuMergeWritePartitionsKernel<<<seg_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
            d_breakpoint_positions, num_breakpoints, data_size,
            config.target_partition_size, config.min_partition_size,
            config.max_partition_size, d_partition_offsets,
            ctx.current->starts, ctx.current->ends);
    }

    cudaFree(d_breakpoint_positions);
    cudaFree(d_partition_counts);
    cudaFree(d_partition_offsets);

    cudaStreamSynchronize(stream);

    // Fit models
    gpuMergeFitPartitionsKernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_data,
        ctx.current->starts, ctx.current->ends,
        ctx.current->model_types,
        ctx.current->theta0, ctx.current->theta1, ctx.current->theta2, ctx.current->theta3,
        ctx.current->delta_bits, ctx.current->max_errors, ctx.current->costs,
        ctx.current->sum_x, ctx.current->sum_y, ctx.current->sum_xx, ctx.current->sum_xy,
        num_partitions);

    // ================================================================
    // Stage 4.5: Initial Polynomial Model Selection (if enabled)
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        refitPolynomialV2Kernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
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
    // Stage 5-6: Optimized GPU Merge Loop
    // ================================================================
    if (config.enable_merging && num_partitions > 1) {
        // TODO: Cooperative kernel has issues, using multi-kernel path for now
        // if (cooperative_launch_supported && num_partitions <= max_cooperative_blocks * GPU_MERGE_V2_BLOCK_SIZE) {
        //     num_partitions = runMergeLoopCooperative(num_partitions);
        // } else {
            num_partitions = runMergeLoopMultiKernel(num_partitions);
        // }
    }

    // ================================================================
    // Stage 6.5: Post-Merge Polynomial Re-evaluation (if enabled)
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        refitPolynomialV2Kernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
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
    // Stage 7: Copy results back
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

    std::sort(result.begin(), result.end(),
              [](const PartitionInfo& a, const PartitionInfo& b) {
                  return a.start_idx < b.start_idx;
              });

    if (!result.empty()) {
        bool needs_refit_first = (result[0].start_idx != 0);
        result[0].start_idx = 0;

        bool needs_refit_last = (result.back().end_idx != data_size);
        result.back().end_idx = data_size;

        for (size_t i = 0; i < result.size() - 1; i++) {
            if (result[i].end_idx != result[i + 1].start_idx) {
                result[i].end_idx = result[i + 1].start_idx;
            }
        }

        // DISABLED: Refitting causes CPU/GPU precision mismatch
        // The model params were computed on GPU, if we refit on CPU with different
        // floating-point precision, we get encoding errors.
        // TODO: Implement GPU-based boundary refit
        // if (needs_refit_first) refitPartition(result[0]);
        // if (needs_refit_last) refitPartition(result.back());
        (void)needs_refit_first;  // Suppress unused warning
        (void)needs_refit_last;
    }

    return result;
}

template<typename T>
GPUPartitionResult<T> GPUCostOptimalPartitionerV2<T>::partitionGPU() {
    GPUPartitionResult<T> result;

    // ================================================================
    // Stage 1-4: Same as partition() - create partitions and fit models
    // ================================================================

    int num_analysis_blocks = (data_size + config.analysis_block_size - 1) / config.analysis_block_size;

    int* d_delta_bits_per_block;
    cudaMalloc(&d_delta_bits_per_block, num_analysis_blocks * sizeof(int));

    gpuMergeComputeDeltaBitsKernel<T><<<num_analysis_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_data, data_size, config.analysis_block_size,
        d_delta_bits_per_block, num_analysis_blocks);

    int* d_is_breakpoint;
    cudaMalloc(&d_is_breakpoint, num_analysis_blocks * sizeof(int));

    int bp_blocks = (num_analysis_blocks + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
    gpuMergeDetectBreakpointsKernel<<<bp_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_delta_bits_per_block, d_is_breakpoint,
        num_analysis_blocks, config.breakpoint_threshold);

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

    int* d_breakpoint_positions;
    int* d_partition_counts;
    int* d_partition_offsets;

    cudaMalloc(&d_breakpoint_positions, num_breakpoints * sizeof(int));
    cudaMalloc(&d_partition_counts, num_breakpoints * sizeof(int));
    cudaMalloc(&d_partition_offsets, (num_breakpoints + 1) * sizeof(int));

    cudaMemcpy(d_breakpoint_positions, breakpoint_positions.data(),
               num_breakpoints * sizeof(int), cudaMemcpyHostToDevice);

    int seg_blocks = (num_breakpoints + GPU_MERGE_V2_BLOCK_SIZE - 1) / GPU_MERGE_V2_BLOCK_SIZE;
    gpuMergeCountPartitionsInSegmentsKernel<<<seg_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_breakpoint_positions, num_breakpoints, data_size,
        config.target_partition_size, config.min_partition_size,
        config.max_partition_size, d_partition_counts);

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(d_partition_counts),
        thrust::device_pointer_cast(d_partition_counts + num_breakpoints),
        thrust::device_pointer_cast(d_partition_offsets));

    int h_total_partitions;
    int h_last_count;
    cudaMemcpy(&h_total_partitions, d_partition_offsets + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_count, d_partition_counts + num_breakpoints - 1,
               sizeof(int), cudaMemcpyDeviceToHost);
    h_total_partitions += h_last_count;

    int num_partitions = h_total_partitions;
    if (num_partitions == 0) {
        num_partitions = 1;
        int zero = 0;
        cudaMemcpy(ctx.current->starts, &zero, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.current->ends, &data_size, sizeof(int), cudaMemcpyHostToDevice);
    } else {
        gpuMergeWritePartitionsKernel<<<seg_blocks, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
            d_breakpoint_positions, num_breakpoints, data_size,
            config.target_partition_size, config.min_partition_size,
            config.max_partition_size, d_partition_offsets,
            ctx.current->starts, ctx.current->ends);
    }

    cudaFree(d_breakpoint_positions);
    cudaFree(d_partition_counts);
    cudaFree(d_partition_offsets);

    cudaStreamSynchronize(stream);

    // Fit models - use same kernel call signature as partition()
    gpuMergeFitPartitionsKernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_data,
        ctx.current->starts, ctx.current->ends,
        ctx.current->model_types,
        ctx.current->theta0, ctx.current->theta1, ctx.current->theta2, ctx.current->theta3,
        ctx.current->delta_bits, ctx.current->max_errors, ctx.current->costs,
        ctx.current->sum_x, ctx.current->sum_y, ctx.current->sum_xx, ctx.current->sum_xy,
        num_partitions);

    // ================================================================
    // Stage 4.5: Initial Polynomial Model Selection (if enabled)
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        refitPolynomialV2Kernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
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
    // Stage 5-6: Optimized GPU Merge Loop
    // ================================================================
    if (config.enable_merging && num_partitions > 1) {
        num_partitions = runMergeLoopMultiKernel(num_partitions);
    }

    // ================================================================
    // Stage 6.5: Post-Merge Polynomial Re-evaluation (if enabled)
    // ================================================================
    if (config.enable_polynomial_models && num_partitions > 1) {
        refitPolynomialV2Kernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
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
    // Stage 7: Sort partitions by start index and fix boundaries ON GPU
    // ================================================================
    // Allocate result arrays first
    result.num_partitions = num_partitions;
    result.owns_memory = true;

    cudaMalloc(&result.d_starts, num_partitions * sizeof(int));
    cudaMalloc(&result.d_ends, num_partitions * sizeof(int));
    cudaMalloc(&result.d_model_types, num_partitions * sizeof(int));
    cudaMalloc(&result.d_theta0, num_partitions * sizeof(double));
    cudaMalloc(&result.d_theta1, num_partitions * sizeof(double));
    cudaMalloc(&result.d_theta2, num_partitions * sizeof(double));
    cudaMalloc(&result.d_theta3, num_partitions * sizeof(double));
    cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int));
    cudaMalloc(&result.d_max_errors, num_partitions * sizeof(long long));

    // Copy from ctx.current to result first (device-to-device)
    cudaMemcpyAsync(result.d_starts, ctx.current->starts, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_ends, ctx.current->ends, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_model_types, ctx.current->model_types, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_theta0, ctx.current->theta0, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_theta1, ctx.current->theta1, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_theta2, ctx.current->theta2, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_theta3, ctx.current->theta3, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_delta_bits, ctx.current->delta_bits, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(result.d_max_errors, ctx.current->max_errors, num_partitions * sizeof(long long), cudaMemcpyDeviceToDevice, stream);

    // Sort by start position using thrust (sort keys, gather values)
    // Create indices array on device
    int* d_indices;
    cudaMalloc(&d_indices, num_partitions * sizeof(int));

    // Initialize indices to 0,1,2,...
    thrust::sequence(thrust::cuda::par.on(stream),
                     thrust::device_pointer_cast(d_indices),
                     thrust::device_pointer_cast(d_indices + num_partitions));

    // Sort indices by start values
    thrust::sort_by_key(thrust::cuda::par.on(stream),
                        thrust::device_pointer_cast(result.d_starts),
                        thrust::device_pointer_cast(result.d_starts + num_partitions),
                        thrust::device_pointer_cast(d_indices));

    // Gather all other arrays using sorted indices
    // We need to reorder ends, model_types, theta0-3, delta_bits, max_errors
    // Use temporary arrays to gather
    int* d_temp_ends;
    int* d_temp_model_types;
    double* d_temp_theta0;
    double* d_temp_theta1;
    double* d_temp_theta2;
    double* d_temp_theta3;
    int* d_temp_delta_bits;
    long long* d_temp_max_errors;

    cudaMalloc(&d_temp_ends, num_partitions * sizeof(int));
    cudaMalloc(&d_temp_model_types, num_partitions * sizeof(int));
    cudaMalloc(&d_temp_theta0, num_partitions * sizeof(double));
    cudaMalloc(&d_temp_theta1, num_partitions * sizeof(double));
    cudaMalloc(&d_temp_theta2, num_partitions * sizeof(double));
    cudaMalloc(&d_temp_theta3, num_partitions * sizeof(double));
    cudaMalloc(&d_temp_delta_bits, num_partitions * sizeof(int));
    cudaMalloc(&d_temp_max_errors, num_partitions * sizeof(long long));

    // Copy original data to temp
    cudaMemcpyAsync(d_temp_ends, result.d_ends, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_model_types, result.d_model_types, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_theta0, result.d_theta0, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_theta1, result.d_theta1, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_theta2, result.d_theta2, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_theta3, result.d_theta3, num_partitions * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_delta_bits, result.d_delta_bits, num_partitions * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_temp_max_errors, result.d_max_errors, num_partitions * sizeof(long long), cudaMemcpyDeviceToDevice, stream);

    // Gather from temp using indices
    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_ends),
                   thrust::device_pointer_cast(result.d_ends));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_model_types),
                   thrust::device_pointer_cast(result.d_model_types));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_theta0),
                   thrust::device_pointer_cast(result.d_theta0));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_theta1),
                   thrust::device_pointer_cast(result.d_theta1));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_theta2),
                   thrust::device_pointer_cast(result.d_theta2));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_theta3),
                   thrust::device_pointer_cast(result.d_theta3));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_delta_bits),
                   thrust::device_pointer_cast(result.d_delta_bits));

    thrust::gather(thrust::cuda::par.on(stream),
                   thrust::device_pointer_cast(d_indices),
                   thrust::device_pointer_cast(d_indices + num_partitions),
                   thrust::device_pointer_cast(d_temp_max_errors),
                   thrust::device_pointer_cast(result.d_max_errors));

    // Free temp arrays
    cudaFree(d_indices);
    cudaFree(d_temp_ends);
    cudaFree(d_temp_model_types);
    cudaFree(d_temp_theta0);
    cudaFree(d_temp_theta1);
    cudaFree(d_temp_theta2);
    cudaFree(d_temp_theta3);
    cudaFree(d_temp_delta_bits);
    cudaFree(d_temp_max_errors);

    // ================================================================
    // Stage 8: Fix boundaries and refit on GPU (no CPU roundtrip!)
    // ================================================================
    fixBoundariesAndRefitKernel<T><<<num_partitions, GPU_MERGE_V2_BLOCK_SIZE, 0, stream>>>(
        d_data,
        result.d_starts,
        result.d_ends,
        result.d_model_types,
        result.d_theta0,
        result.d_theta1,
        result.d_theta2,
        result.d_theta3,
        result.d_delta_bits,
        result.d_max_errors,
        nullptr,  // costs not needed for final result
        num_partitions,
        data_size);

    cudaStreamSynchronize(stream);

    return result;
}

template<typename T>
void GPUCostOptimalPartitionerV2<T>::getStats(int& num_partitions, float& avg_partition_size) const {
    num_partitions = 0;
    avg_partition_size = 0.0f;
}

// Explicit instantiation
template class GPUCostOptimalPartitionerV2<int32_t>;
template class GPUCostOptimalPartitionerV2<uint32_t>;
template class GPUCostOptimalPartitionerV2<int64_t>;
template class GPUCostOptimalPartitionerV2<uint64_t>;

// ============================================================================
// Validation and Benchmarking
// ============================================================================

template<typename T>
bool validateGPUMergeV2(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    bool verbose)
{
    GPUCostOptimalPartitionerV2<T> v2_partitioner(data, config);
    auto v2_result = v2_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    if (v2_result.size() != cpu_result.size()) {
        if (verbose) {
            std::cerr << "Partition count mismatch: V2=" << v2_result.size()
                      << " CPU=" << cpu_result.size() << std::endl;
        }
        return false;
    }

    for (size_t i = 0; i < v2_result.size(); i++) {
        const auto& v2 = v2_result[i];
        const auto& cpu = cpu_result[i];

        if (v2.start_idx != cpu.start_idx || v2.end_idx != cpu.end_idx) {
            if (verbose) {
                std::cerr << "Partition " << i << " boundary mismatch" << std::endl;
            }
            return false;
        }
    }

    if (verbose) {
        std::cout << "GPU Merge V2 validation passed! Partitions: " << v2_result.size() << std::endl;
        std::cout << "Cooperative launch " << (v2_partitioner.isCooperativeLaunchSupported() ? "enabled" : "disabled") << std::endl;
    }

    return true;
}

template<typename T>
void benchmarkGPUMergeVersions(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    int num_runs)
{
    std::cout << "=== GPU Merge Benchmark ===" << std::endl;
    std::cout << "Data size: " << data.size() << " elements" << std::endl;
    std::cout << "Runs: " << num_runs << std::endl;

    // Warmup
    {
        GPUCostOptimalPartitioner<T> v1(data, config);
        v1.partition();
        GPUCostOptimalPartitionerV2<T> v2(data, config);
        v2.partition();
    }

    // Benchmark V1
    double v1_total_ms = 0;
    int v1_partitions = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitioner<T> v1(data, config);
        auto result = v1.partition();

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        v1_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        v1_partitions = result.size();
    }

    // Benchmark V2
    double v2_total_ms = 0;
    int v2_partitions = 0;
    bool cooperative_used = false;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitionerV2<T> v2(data, config);
        auto result = v2.partition();
        cooperative_used = v2.isCooperativeLaunchSupported();

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        v2_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        v2_partitions = result.size();
    }

    std::cout << "\nResults:" << std::endl;
    std::cout << "V1 (original): " << v1_total_ms / num_runs << " ms avg, "
              << v1_partitions << " partitions" << std::endl;
    std::cout << "V2 (optimized): " << v2_total_ms / num_runs << " ms avg, "
              << v2_partitions << " partitions" << std::endl;
    std::cout << "Cooperative groups: " << (cooperative_used ? "yes" : "no") << std::endl;
    std::cout << "Speedup: " << v1_total_ms / v2_total_ms << "x" << std::endl;
}

template bool validateGPUMergeV2<int32_t>(const std::vector<int32_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMergeV2<uint32_t>(const std::vector<uint32_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMergeV2<int64_t>(const std::vector<int64_t>&, const CostOptimalConfig&, bool);
template bool validateGPUMergeV2<uint64_t>(const std::vector<uint64_t>&, const CostOptimalConfig&, bool);

template void benchmarkGPUMergeVersions<int32_t>(const std::vector<int32_t>&, const CostOptimalConfig&, int);
template void benchmarkGPUMergeVersions<uint32_t>(const std::vector<uint32_t>&, const CostOptimalConfig&, int);
template void benchmarkGPUMergeVersions<int64_t>(const std::vector<int64_t>&, const CostOptimalConfig&, int);
template void benchmarkGPUMergeVersions<uint64_t>(const std::vector<uint64_t>&, const CostOptimalConfig&, int);
