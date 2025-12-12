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

#ifndef ENCODER_COST_OPTIMAL_GPU_MERGE_V2_CUH
#define ENCODER_COST_OPTIMAL_GPU_MERGE_V2_CUH

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"

namespace cg = cooperative_groups;

// ============================================================================
// Constants
// ============================================================================

constexpr float GPU_MERGE_V2_MODEL_OVERHEAD_BYTES = 64.0f;
constexpr int GPU_MERGE_V2_BLOCK_SIZE = 256;
constexpr int GPU_MERGE_V2_MAX_BLOCKS = 512;  // Conservative limit for cooperative launch

// Maximum integer that can be exactly represented as double (2^53)
// Values larger than this lose precision when converted to double,
// making linear/polynomial models unreliable. Force FOR+BitPack for such data.
constexpr uint64_t GPU_MERGE_V2_DOUBLE_PRECISION_MAX = 9007199254740992ULL;  // 2^53

// ============================================================================
// Unified Buffer Structure (single allocation, better cache utilization)
// ============================================================================

template<typename T>
struct UnifiedPartitionBuffer {
    // Core partition data (AoS would be better for cache, but keeping SoA for compatibility)
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

    // Cached statistics for O(1) merge
    double* sum_x;
    double* sum_y;
    double* sum_xx;
    double* sum_xy;

    size_t capacity;
    bool allocated;

    UnifiedPartitionBuffer() : capacity(0), allocated(false),
        starts(nullptr), ends(nullptr), model_types(nullptr),
        theta0(nullptr), theta1(nullptr), theta2(nullptr), theta3(nullptr),
        delta_bits(nullptr), costs(nullptr), max_errors(nullptr),
        sum_x(nullptr), sum_y(nullptr), sum_xx(nullptr), sum_xy(nullptr) {}

    cudaError_t allocate(size_t max_partitions);
    void free();
    ~UnifiedPartitionBuffer() { free(); }
};

// ============================================================================
// Optimized GPU Merge Context
// ============================================================================

template<typename T>
struct GPUMergeContextV2 {
    // Double buffers for ping-pong
    UnifiedPartitionBuffer<T> buffer_A;
    UnifiedPartitionBuffer<T> buffer_B;
    UnifiedPartitionBuffer<T>* current;
    UnifiedPartitionBuffer<T>* next;

    // Merge working arrays
    float* merge_benefits;
    int* merged_delta_bits;
    double* merged_theta0;
    double* merged_theta1;
    double* merged_theta2;  // Polynomial coefficient for POLY2/POLY3
    double* merged_theta3;  // Polynomial coefficient for POLY3
    double* merged_sum_x;
    double* merged_sum_y;
    double* merged_sum_xx;
    double* merged_sum_xy;
    int* merge_flags;

    // Stream compaction arrays
    int* output_slots;
    int* output_indices;
    int* is_merge_base;

    // Block-level scan intermediates
    int* block_sums;

    // Atomic counters (pinned memory for fast access)
    int* d_num_partitions;      // Device: current partition count
    int* d_merge_count;         // Device: merges in current round
    int* d_new_partition_count; // Device: partition count after merge

    // Host-side pinned memory for final result
    int* h_final_partition_count;

    size_t capacity;
    bool allocated;

    GPUMergeContextV2() : allocated(false), capacity(0),
        current(nullptr), next(nullptr),
        merge_benefits(nullptr), merged_delta_bits(nullptr),
        merged_theta0(nullptr), merged_theta1(nullptr),
        merged_theta2(nullptr), merged_theta3(nullptr),
        merged_sum_x(nullptr), merged_sum_y(nullptr),
        merged_sum_xx(nullptr), merged_sum_xy(nullptr),
        merge_flags(nullptr),
        output_slots(nullptr), output_indices(nullptr), is_merge_base(nullptr),
        block_sums(nullptr),
        d_num_partitions(nullptr), d_merge_count(nullptr), d_new_partition_count(nullptr),
        h_final_partition_count(nullptr) {}

    cudaError_t allocate(size_t max_partitions);
    void free();
    void swap() { std::swap(current, next); }
    ~GPUMergeContextV2() { free(); }
};

// ============================================================================
// GPU Partition Result (keeps data on GPU, avoids CPU roundtrip)
// ============================================================================

template<typename T>
struct GPUPartitionResult {
    int* d_starts;
    int* d_ends;
    int* d_model_types;
    double* d_theta0;
    double* d_theta1;
    double* d_theta2;
    double* d_theta3;
    int* d_delta_bits;
    long long* d_max_errors;
    int num_partitions;
    bool owns_memory;  // If true, destructor will free memory

    GPUPartitionResult() : d_starts(nullptr), d_ends(nullptr), d_model_types(nullptr),
        d_theta0(nullptr), d_theta1(nullptr), d_theta2(nullptr), d_theta3(nullptr),
        d_delta_bits(nullptr), d_max_errors(nullptr), num_partitions(0), owns_memory(false) {}

    void free() {
        if (owns_memory) {
            if (d_starts) cudaFree(d_starts);
            if (d_ends) cudaFree(d_ends);
            if (d_model_types) cudaFree(d_model_types);
            if (d_theta0) cudaFree(d_theta0);
            if (d_theta1) cudaFree(d_theta1);
            if (d_theta2) cudaFree(d_theta2);
            if (d_theta3) cudaFree(d_theta3);
            if (d_delta_bits) cudaFree(d_delta_bits);
            if (d_max_errors) cudaFree(d_max_errors);
        }
        d_starts = d_ends = d_model_types = d_delta_bits = nullptr;
        d_theta0 = d_theta1 = d_theta2 = d_theta3 = nullptr;
        d_max_errors = nullptr;
        num_partitions = 0;
        owns_memory = false;
    }

    ~GPUPartitionResult() { free(); }
};

// ============================================================================
// Optimized GPU Partitioner Class
// ============================================================================

template<typename T>
class GPUCostOptimalPartitionerV2 {
private:
    T* d_data;
    const std::vector<T>& h_data_ref;
    int data_size;
    CostOptimalConfig config;
    cudaStream_t stream;

    GPUMergeContextV2<T> ctx;

    // Check if cooperative launch is supported
    bool cooperative_launch_supported;
    int max_cooperative_blocks;

    void checkCooperativeLaunchSupport();
    void refitPartition(PartitionInfo& info);

    // Two implementation paths
    int runMergeLoopCooperative(int num_partitions);  // Single kernel, cooperative groups
    int runMergeLoopMultiKernel(int num_partitions);  // Fallback: optimized multi-kernel

public:
    GPUCostOptimalPartitionerV2(const std::vector<T>& data,
                                 const CostOptimalConfig& cfg,
                                 cudaStream_t cuda_stream = 0);
    ~GPUCostOptimalPartitionerV2();

    std::vector<PartitionInfo> partition();

    // New: Keep partition results on GPU (avoids GPU->CPU->GPU roundtrip)
    GPUPartitionResult<T> partitionGPU();

    void getStats(int& num_partitions, float& avg_partition_size) const;

    // For benchmarking
    bool isCooperativeLaunchSupported() const { return cooperative_launch_supported; }
};

// ============================================================================
// Kernel Declarations
// ============================================================================

// Cooperative kernel: all merge rounds in one launch
template<typename T>
__global__ void mergeLoopCooperativeKernel(
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
    int data_size);

// Fallback kernels for non-cooperative path
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
    int max_partition_size);

__global__ void fusedMarkAndOutputSlotsKernel(
    const float* __restrict__ merge_benefits,
    int* __restrict__ merge_flags,
    int* __restrict__ output_slots,
    int* __restrict__ is_merge_base,
    int* __restrict__ d_merge_count,  // atomic counter
    int num_partitions,
    float threshold);

__global__ void blockPrefixSumKernel(
    const int* __restrict__ input,
    int* __restrict__ output,
    int* __restrict__ block_sums,
    int n);

__global__ void addBlockOffsetsKernel(
    int* __restrict__ data,
    const int* __restrict__ block_sums,
    int n);

template<typename T>
__global__ void applyMergesV2Kernel(
    // Old buffers
    const int* __restrict__ old_starts, const int* __restrict__ old_ends,
    const int* __restrict__ old_model_types,
    const double* __restrict__ old_theta0, const double* __restrict__ old_theta1,
    const double* __restrict__ old_theta2, const double* __restrict__ old_theta3,
    const int* __restrict__ old_delta_bits, const float* __restrict__ old_costs,
    const long long* __restrict__ old_max_errors,
    const double* __restrict__ old_sum_x, const double* __restrict__ old_sum_y,
    const double* __restrict__ old_sum_xx, const double* __restrict__ old_sum_xy,
    // Merge info
    const int* __restrict__ output_slots, const int* __restrict__ output_indices,
    const int* __restrict__ is_merge_base,
    const double* __restrict__ merged_theta0, const double* __restrict__ merged_theta1,
    const int* __restrict__ merged_delta_bits,
    const double* __restrict__ merged_sum_x, const double* __restrict__ merged_sum_y,
    const double* __restrict__ merged_sum_xx, const double* __restrict__ merged_sum_xy,
    // New buffers
    int* __restrict__ new_starts, int* __restrict__ new_ends,
    int* __restrict__ new_model_types,
    double* __restrict__ new_theta0, double* __restrict__ new_theta1,
    double* __restrict__ new_theta2, double* __restrict__ new_theta3,
    int* __restrict__ new_delta_bits, float* __restrict__ new_costs,
    long long* __restrict__ new_max_errors,
    double* __restrict__ new_sum_x, double* __restrict__ new_sum_y,
    double* __restrict__ new_sum_xx, double* __restrict__ new_sum_xy,
    int num_partitions);

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
bool validateGPUMergeV2(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    bool verbose = false);

// Compare V1 vs V2 performance
template<typename T>
void benchmarkGPUMergeVersions(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    int num_runs = 10);

#endif // ENCODER_COST_OPTIMAL_GPU_MERGE_V2_CUH
