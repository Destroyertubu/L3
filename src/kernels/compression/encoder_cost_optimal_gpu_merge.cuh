/**
 * GPU-Parallel Merge Implementation for Cost-Optimal Partitioning
 *
 * This file provides GPU-accelerated merge operations using stream compaction,
 * replacing the CPU-based merge loop in the original implementation.
 *
 * Key innovations:
 * 1. GPU stream compaction using thrust::exclusive_scan
 * 2. Double buffering for efficient memory management
 * 3. Minimal CPU-GPU data transfer (only partition count)
 *
 * Author: Claude Code Assistant
 * Date: 2025-12-06
 */

#ifndef ENCODER_COST_OPTIMAL_GPU_MERGE_CUH
#define ENCODER_COST_OPTIMAL_GPU_MERGE_CUH

#include <cuda_runtime.h>
#include <vector>
#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"

// ============================================================================
// Constants
// ============================================================================

constexpr float GPU_MERGE_MODEL_OVERHEAD_BYTES = 64.0f;

// ============================================================================
// Data Structures
// ============================================================================

/**
 * Double buffer structure for partition arrays
 * Allows efficient ping-pong between merge rounds without reallocation
 *
 * OPTIMIZATION: Caches linear regression statistics (sum_x, sum_y, sum_xx, sum_xy)
 * to enable O(1) merge evaluation instead of O(n) data scanning.
 */
template<typename T>
struct PartitionBuffers {
    int* starts;
    int* ends;
    int* model_types;
    double* theta0;
    double* theta1;
    double* theta2;  // Polynomial coefficient for POLY2/POLY3
    double* theta3;  // Polynomial coefficient for POLY3
    int* delta_bits;
    float* costs;
    long long* max_errors;

    // Auxiliary arrays for merge
    T* partition_mins;
    T* partition_maxs;

    // Cached statistics for O(1) merge evaluation
    // These allow computing merged theta0/theta1 without re-scanning data
    double* sum_x;   // Σi (local index)
    double* sum_y;   // Σdata[i]
    double* sum_xx;  // Σi²
    double* sum_xy;  // Σi*data[i]

    bool allocated;
    size_t capacity;

    PartitionBuffers() : allocated(false), capacity(0),
        starts(nullptr), ends(nullptr), model_types(nullptr),
        theta0(nullptr), theta1(nullptr), theta2(nullptr), theta3(nullptr),
        delta_bits(nullptr), costs(nullptr), max_errors(nullptr),
        partition_mins(nullptr), partition_maxs(nullptr),
        sum_x(nullptr), sum_y(nullptr), sum_xx(nullptr), sum_xy(nullptr) {}

    cudaError_t allocate(size_t max_partitions) {
        if (allocated && capacity >= max_partitions) return cudaSuccess;

        free();

        cudaError_t err;
        err = cudaMalloc(&starts, max_partitions * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&ends, max_partitions * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&model_types, max_partitions * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&theta0, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&theta1, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&theta2, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&theta3, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&delta_bits, max_partitions * sizeof(int));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&costs, max_partitions * sizeof(float));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&max_errors, max_partitions * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&partition_mins, max_partitions * sizeof(T));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&partition_maxs, max_partitions * sizeof(T));
        if (err != cudaSuccess) return err;

        // Allocate cached statistics
        err = cudaMalloc(&sum_x, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&sum_y, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&sum_xx, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&sum_xy, max_partitions * sizeof(double));
        if (err != cudaSuccess) return err;

        allocated = true;
        capacity = max_partitions;
        return cudaSuccess;
    }

    void free() {
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
        if (partition_mins) cudaFree(partition_mins);
        if (partition_maxs) cudaFree(partition_maxs);
        if (sum_x) cudaFree(sum_x);
        if (sum_y) cudaFree(sum_y);
        if (sum_xx) cudaFree(sum_xx);
        if (sum_xy) cudaFree(sum_xy);

        starts = ends = model_types = delta_bits = nullptr;
        theta0 = theta1 = theta2 = theta3 = nullptr;
        costs = nullptr;
        max_errors = nullptr;
        partition_mins = partition_maxs = nullptr;
        sum_x = sum_y = sum_xx = sum_xy = nullptr;
        allocated = false;
        capacity = 0;
    }

    ~PartitionBuffers() {
        free();
    }
};

/**
 * GPU Merge context containing all temporary buffers
 */
template<typename T>
struct GPUMergeContext {
    // Double buffers
    PartitionBuffers<T> buffer_A;
    PartitionBuffers<T> buffer_B;
    PartitionBuffers<T>* current;
    PartitionBuffers<T>* next;

    // Merge evaluation arrays
    float* merge_benefits;
    int* merged_delta_bits;
    double* merged_theta0;
    double* merged_theta1;
    int* merge_flags;

    // Merged statistics for propagation (computed during merge evaluation)
    double* merged_sum_x;
    double* merged_sum_y;
    double* merged_sum_xx;
    double* merged_sum_xy;

    // Stream compaction arrays
    int* output_slots;
    int* output_indices;
    int* is_merge_base;

    // Scan temporary storage
    void* d_temp_storage;
    size_t temp_storage_bytes;

    size_t capacity;
    bool allocated;

    GPUMergeContext() : allocated(false), capacity(0),
        current(nullptr), next(nullptr),
        merge_benefits(nullptr), merged_delta_bits(nullptr),
        merged_theta0(nullptr), merged_theta1(nullptr),
        merge_flags(nullptr),
        merged_sum_x(nullptr), merged_sum_y(nullptr),
        merged_sum_xx(nullptr), merged_sum_xy(nullptr),
        output_slots(nullptr),
        output_indices(nullptr), is_merge_base(nullptr),
        d_temp_storage(nullptr), temp_storage_bytes(0) {}

    cudaError_t allocate(size_t max_partitions);
    void free();
    void swap() { std::swap(current, next); }

    ~GPUMergeContext() { free(); }
};

// ============================================================================
// GPU Kernels Declaration
// ============================================================================

/**
 * Compute output slot for each partition in stream compaction
 *
 * Rules:
 * - merge_flags[i-1] = 1 -> output_slots[i] = 0 (merged with previous, skip)
 * - merge_flags[i] = 1   -> output_slots[i] = 1 (merge base, output merged result)
 * - otherwise            -> output_slots[i] = 1 (keep as is)
 */
__global__ void computeOutputSlotKernel(
    const int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int num_partitions);

/**
 * Apply merges in parallel using precomputed output indices
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
    // Merge information
    const int* __restrict__ output_slots,
    const int* __restrict__ output_indices,
    const int* __restrict__ is_merge_base,
    const double* __restrict__ merged_theta0,
    const double* __restrict__ merged_theta1,
    const int* __restrict__ merged_delta_bits,
    // Output arrays (new partitions)
    int* __restrict__ new_starts,
    int* __restrict__ new_ends,
    int* __restrict__ new_model_types,
    double* __restrict__ new_theta0,
    double* __restrict__ new_theta1,
    int* __restrict__ new_delta_bits,
    float* __restrict__ new_costs,
    long long* __restrict__ new_max_errors,
    int num_partitions);

/**
 * Count merge flags (reduction kernel)
 */
__global__ void countMergeFlagsKernel(
    const int* __restrict__ merge_flags,
    int* __restrict__ count,
    int num_partitions);

// ============================================================================
// GPU-Parallel Cost-Optimal Partitioner Class
// ============================================================================

/**
 * GPU-accelerated Cost-Optimal Partitioner with parallel merge
 *
 * This class extends the original CostOptimalPartitioner with:
 * - GPU-parallel merge application using stream compaction
 * - Double buffering for efficient memory management
 * - Minimal CPU-GPU data transfer
 */
template<typename T>
class GPUCostOptimalPartitioner {
private:
    T* d_data;
    const std::vector<T>& h_data_ref;
    int data_size;
    CostOptimalConfig config;
    cudaStream_t stream;

    GPUMergeContext<T> ctx;

    // Host-side refit for boundary adjustments
    void refitPartition(PartitionInfo& info);

    // GPU merge implementation
    int applyMergesGPU(int num_partitions);

public:
    GPUCostOptimalPartitioner(const std::vector<T>& data,
                              const CostOptimalConfig& cfg,
                              cudaStream_t cuda_stream = 0);

    ~GPUCostOptimalPartitioner();

    /**
     * Create partitions using GPU-accelerated cost-optimal algorithm
     */
    std::vector<PartitionInfo> partition();

    /**
     * Get compression statistics
     */
    void getStats(int& num_partitions, float& avg_partition_size) const;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compare GPU and CPU merge results for validation
 */
template<typename T>
bool validateGPUMerge(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    bool verbose = false);

#endif // ENCODER_COST_OPTIMAL_GPU_MERGE_CUH
