#pragma once

#include <cuda_runtime.h>
#include <cstdint>

/**
 * GPU-Friendly Learned Index for Partition Lookup
 *
 * Inspired by ALEX (SIGMOD'20), but optimized for GPU execution:
 * - Linear model for each fanout node
 * - Compact representation suitable for device memory
 * - O(1) expected lookup time
 *
 * Trade-off: Slightly larger memory footprint vs binary search,
 * but much faster lookups (especially for random access patterns)
 */

// Simplified linear model for GPU
struct LinearModel {
    double slope;      // y = slope * x + intercept
    double intercept;

    __device__ __forceinline__ int predict(int key) const {
        double pred = slope * key + intercept;
        return static_cast<int>(pred);
    }
};

/**
 * Learned Index Structure for Partition Lookup
 *
 * Two-level structure:
 * - Root model: Maps global_idx to approximate partition
 * - Correction vector: Small adjustments for each partition
 */
struct PartitionLearnedIndex {
    // Root linear model
    LinearModel root_model;

    // Partition boundaries (for validation/correction)
    int32_t* d_start_indices;  // Start index of each partition
    int32_t* d_end_indices;    // End index of each partition

    int num_partitions;

    // Error bounds (maximum prediction error)
    int max_error;  // Maximum distance between prediction and actual partition
};

/**
 * Build learned index from partition metadata (CPU side)
 *
 * NOTE: This function is provided for reference but not currently used
 * in the simplified GPU implementation which builds the model on-the-fly
 * from partition metadata already in CompressedDataL3 structure.
 *
 * ALGORITHM:
 * 1. Fit linear model: partition_idx = slope * global_idx + intercept
 * 2. Compute maximum prediction error
 * 3. Upload to device memory
 */
// Commented out to avoid CUDA_CHECK macro dependency issues
// Will be implemented when needed for more complex learned index structures
/*
inline PartitionLearnedIndex* buildPartitionLearnedIndex(
    const int32_t* h_start_indices,
    const int32_t* h_end_indices,
    int num_partitions)
{
    PartitionLearnedIndex* index = new PartitionLearnedIndex();
    index->num_partitions = num_partitions;

    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    int n = num_partitions;

    for (int i = 0; i < num_partitions; i++) {
        double x = (h_start_indices[i] + h_end_indices[i]) / 2.0;
        double y = i;

        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    double slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    double intercept = (sum_y - slope * sum_x) / n;

    index->root_model.slope = slope;
    index->root_model.intercept = intercept;

    int max_error = 0;
    for (int i = 0; i < num_partitions; i++) {
        int mid = (h_start_indices[i] + h_end_indices[i]) / 2;
        int pred = static_cast<int>(slope * mid + intercept);
        int error = abs(pred - i);
        max_error = max(max_error, error);
    }
    index->max_error = max_error;

    cudaMalloc(&index->d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&index->d_end_indices, num_partitions * sizeof(int32_t));
    cudaMemcpy(index->d_start_indices, h_start_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(index->d_end_indices, h_end_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);

    return index;
}

inline void freePartitionLearnedIndex(PartitionLearnedIndex* index) {
    if (index) {
        if (index->d_start_indices) cudaFree(index->d_start_indices);
        if (index->d_end_indices) cudaFree(index->d_end_indices);
        delete index;
    }
}
*/

/**
 * Lookup partition using learned index (GPU device function)
 *
 * ALGORITHM:
 * 1. Use linear model to predict approximate partition
 * 2. Search within [pred - max_error, pred + max_error] window
 * 3. Return exact partition
 *
 * COMPLEXITY: O(1) expected, O(max_error) worst case
 *
 * For well-distributed data, max_error is typically 1-3,
 * making this much faster than O(log n) binary search
 */
__device__ __forceinline__ int findPartitionWithLearnedIndex(
    const PartitionLearnedIndex* index,
    int global_idx)
{
    // Step 1: Predict using linear model
    int pred = index->root_model.predict(global_idx);

    // Step 2: Clamp to valid range
    int search_start = max(0, pred - index->max_error);
    int search_end = min(index->num_partitions - 1, pred + index->max_error);

    // Step 3: Linear search within small window
    // Most of the time, pred is exact or off by 1
    for (int i = search_start; i <= search_end; i++) {
        int start = index->d_start_indices[i];
        int end = index->d_end_indices[i];

        if (global_idx >= start && global_idx < end) {
            return i;
        }
    }

    // Fallback: Should rarely reach here if model is well-fitted
    // Binary search as safety net
    int left = 0, right = index->num_partitions - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        int start = index->d_start_indices[mid];
        int end = index->d_end_indices[mid];

        if (global_idx >= start && global_idx < end) {
            return mid;
        } else if (global_idx < start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return 0;  // Should never reach here
}

/**
 * Alternative: Even simpler "piecewise linear" index
 *
 * For uniform-ish partitions, we can use a single linear model
 * with very small error bounds
 */
__device__ __forceinline__ int findPartitionSimple(
    const LinearModel* model,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    int num_partitions,
    int global_idx)
{
    // Direct prediction
    int pred = model->predict(global_idx);
    pred = max(0, min(num_partitions - 1, pred));

    // Check predicted partition
    if (global_idx >= d_start_indices[pred] && global_idx < d_end_indices[pred]) {
        return pred;
    }

    // Check neighbors (usually sufficient)
    if (pred > 0 && global_idx >= d_start_indices[pred-1] && global_idx < d_end_indices[pred-1]) {
        return pred - 1;
    }
    if (pred < num_partitions - 1 && global_idx >= d_start_indices[pred+1] && global_idx < d_end_indices[pred+1]) {
        return pred + 1;
    }

    // Rare case: binary search fallback
    int left = 0, right = num_partitions - 1;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (global_idx >= d_start_indices[mid] && global_idx < d_end_indices[mid]) {
            return mid;
        } else if (global_idx < d_start_indices[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return 0;
}

// Note: CUDA_CHECK macro is defined in ssb_utils.h
// Avoid redefinition by checking if it's already defined
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
#endif
