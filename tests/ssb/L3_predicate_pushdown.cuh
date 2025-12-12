#pragma once

#include "L3_codec.hpp"
#include "L3_ra_utils.cuh"
#include <cuda_runtime.h>
#include <cstdio>

/**
 * Predicate Pushdown into Compressed Domain
 *
 * INNOVATION: Filter data BEFORE decompression using model predictions + bounds
 *
 * ALGORITHM:
 * 1. For each partition, compute predicted value range using model
 * 2. Compare with actual min/max bounds to get conservative estimate
 * 3. If partition range doesn't overlap filter predicate, SKIP entire partition
 * 4. Otherwise, decompress and check each value
 *
 * ACADEMIC VALUE:
 * - Novel approach: Pushes predicate evaluation into compressed domain
 * - Leverages learned model's prediction capability for pruning
 * - Reduces decompression overhead for selective queries
 *
 * EXPECTED SPEEDUP:
 * - For highly selective filters (< 5%): 1.5-3× faster
 * - For range queries on sorted/semi-sorted data: 2-5× faster
 * - For random data: Minimal overhead, ~5-10% slower due to bound checking
 */

/**
 * Partition-Level Predicate Evaluation
 *
 * Determine if a partition can be pruned without decompression
 *
 * RETURNS:
 * - 0: Partition definitely does NOT contain matching values (PRUNE)
 * - 1: Partition MAY contain matching values (DECOMPRESS & CHECK)
 */
template<typename T>
__device__ __forceinline__ bool canPartitionMatch(
    const CompressedDataL3<T>* compressed,
    int partition_idx,
    T filter_min,
    T filter_max)
{
    // Get actual min/max bounds for this partition
    T partition_min = compressed->d_partition_min_values[partition_idx];
    T partition_max = compressed->d_partition_max_values[partition_idx];

    // Check overlap: [partition_min, partition_max] ∩ [filter_min, filter_max]
    // No overlap if: partition_max < filter_min OR partition_min > filter_max
    if (partition_max < filter_min || partition_min > filter_max) {
        return false;  // Partition range doesn't overlap filter range - PRUNE!
    }

    return true;  // Partition may contain matches - decompress and check
}

/**
 * Predicate Pushdown Stage 1: Partition-Level Filtering with Pruning
 *
 * OPTIMIZATION: Skip entire partitions that cannot match the filter
 *
 * ALGORITHM:
 * 1. Each block processes one partition
 * 2. Check partition bounds against filter predicate
 * 3. If partition can be pruned, skip entirely (no decompression!)
 * 4. Otherwise, decompress partition and filter normally
 *
 * PERFORMANCE:
 * - Best case (sorted data, selective filter): Skip 90%+ partitions
 * - Worst case (random data): Small overhead from bound checking
 */
template<typename T>
__global__ void stage1_filter_with_predicate_pushdown(
    const CompressedDataL3<T>* c_column,
    int num_entries,
    T filter_min,
    T filter_max,
    int* d_candidate_indices,
    int* d_num_candidates,
    unsigned long long* d_partitions_pruned)  // Stats: # partitions skipped
{
    __shared__ PartitionMeta s_meta;
    __shared__ bool s_partition_can_match;
    __shared__ int s_candidate_buffer[512];
    __shared__ int s_buffer_count;

    int partition_idx = blockIdx.x;
    if (partition_idx >= c_column->num_partitions) return;

    // Thread 0: Check if this partition can possibly match the filter
    if (threadIdx.x == 0) {
        s_partition_can_match = canPartitionMatch(c_column, partition_idx, filter_min, filter_max);

        // If partition cannot match, record pruning stats
        if (!s_partition_can_match) {
            atomicAdd(d_partitions_pruned, 1);
        }

        // Load partition metadata (only if we need to decompress)
        if (s_partition_can_match) {
            loadPartitionMeta(c_column, partition_idx, s_meta);
        }
        s_buffer_count = 0;
    }
    __syncthreads();

    // Early exit: If partition was pruned, entire block skips decompression!
    if (!s_partition_can_match) {
        return;  // 🎯 PRUNING: Skip this partition entirely!
    }

    // Decompress and filter (only if partition passed pruning)
    for (int local_idx = threadIdx.x; local_idx < s_meta.partition_len; local_idx += blockDim.x) {
        int global_idx = s_meta.start_idx + local_idx;
        if (global_idx >= num_entries) continue;

        // Decompress value
        T value = decompressWithMeta(c_column, global_idx, s_meta);

        // Apply filter
        if (value >= filter_min && value < filter_max) {
            // Add to shared memory buffer
            int buffer_pos = atomicAdd(&s_buffer_count, 1);

            if (buffer_pos < 512) {
                s_candidate_buffer[buffer_pos] = global_idx;
            } else {
                // Buffer full, write directly to global memory
                int pos = atomicAdd(d_num_candidates, 1);
                if (pos < 50000000) {
                    d_candidate_indices[pos] = global_idx;
                }
            }
        }
    }

    __syncthreads();

    // Batch write buffered candidates to global memory
    int buffer_size = min(s_buffer_count, 512);

    if (buffer_size > 0) {
        __shared__ int global_start_pos;
        if (threadIdx.x == 0) {
            global_start_pos = atomicAdd(d_num_candidates, buffer_size);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < buffer_size; i += blockDim.x) {
            int global_pos = global_start_pos + i;
            if (global_pos < 50000000) {
                d_candidate_indices[global_pos] = s_candidate_buffer[i];
            }
        }
    }
}

/**
 * Model-Based Partition Pruning (Alternative Strategy)
 *
 * ALGORITHM:
 * - Use model prediction + error bounds to estimate value range
 * - Conservative estimate: [min_prediction - max_error, max_prediction + max_error]
 * - Compare with actual partition bounds for tighter estimate
 *
 * This is MORE conservative but avoids needing actual min/max bounds
 */
template<typename T>
__device__ __forceinline__ bool canPartitionMatchModelBased(
    const CompressedDataL3<T>* compressed,
    int partition_idx,
    T filter_min,
    T filter_max)
{
    // Get partition info
    int start_idx = compressed->d_start_indices[partition_idx];
    int end_idx = compressed->d_end_indices[partition_idx];
    int partition_len = end_idx - start_idx;

    // Get model parameters
    double theta0 = compressed->d_model_params[partition_idx * 4];
    double theta1 = compressed->d_model_params[partition_idx * 4 + 1];

    // Get error bound
    int64_t error_bound = compressed->d_error_bounds[partition_idx];

    // Compute predicted value range
    // First value: theta0 + theta1 * 0
    // Last value: theta0 + theta1 * (partition_len - 1)
    double pred_first = theta0;
    double pred_last = theta0 + theta1 * (partition_len - 1);

    // Conservative range with error bounds
    T min_possible = static_cast<T>(min(pred_first, pred_last) - error_bound);
    T max_possible = static_cast<T>(max(pred_first, pred_last) + error_bound);

    // Check overlap
    if (max_possible < filter_min || min_possible > filter_max) {
        return false;  // Cannot match - PRUNE!
    }

    return true;  // May match - decompress
}

/**
 * Hybrid Strategy: Use both actual bounds AND model-based bounds
 *
 * ALGORITHM:
 * 1. If actual bounds available, use them (most accurate)
 * 2. Otherwise, fall back to model-based estimation
 *
 * This provides flexibility for different compression modes
 */
template<typename T>
__device__ __forceinline__ bool canPartitionMatchHybrid(
    const CompressedDataL3<T>* compressed,
    int partition_idx,
    T filter_min,
    T filter_max)
{
    // Strategy 1: Use actual bounds if available
    if (compressed->d_partition_min_values != nullptr &&
        compressed->d_partition_max_values != nullptr) {
        return canPartitionMatch(compressed, partition_idx, filter_min, filter_max);
    }

    // Strategy 2: Fall back to model-based estimation
    return canPartitionMatchModelBased(compressed, partition_idx, filter_min, filter_max);
}

/**
 * Statistics for analyzing predicate pushdown effectiveness
 */
struct PredicatePushdownStats {
    unsigned long long partitions_total;
    unsigned long long partitions_pruned;
    unsigned long long elements_skipped;
    double pruning_ratio;  // partitions_pruned / partitions_total
    double speedup;        // vs non-pushdown version
};

/**
 * Print predicate pushdown statistics
 */
inline void printPredicatePushdownStats(const PredicatePushdownStats& stats) {
    printf("Predicate Pushdown Statistics:\n");
    printf("  Total partitions: %llu\n", stats.partitions_total);
    printf("  Partitions pruned: %llu (%.1f%%)\n",
           stats.partitions_pruned,
           100.0 * stats.partitions_pruned / stats.partitions_total);
    printf("  Elements skipped: %llu\n", stats.elements_skipped);
    printf("  Speedup: %.2fx\n", stats.speedup);
}

