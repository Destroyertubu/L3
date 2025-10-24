#pragma once

#include "L3_codec.hpp"
#include "L3_alex_index.cuh"
#include <cuda_runtime.h>

// Random Access Helper Functions for GLECO Compressed Data

// ============================================================================
// Configuration: Choose partition lookup method
// ============================================================================

// Set to 1 to use ALEX learned index, 0 to use binary search
// NOTE: Benchmarks show binary search is FASTER (1.5-1.6× speedup) due to:
//   - Better GPU memory access patterns
//   - No floating-point division overhead
//   - Less warp divergence
// Learned index kept for reference/future optimization
#define USE_LEARNED_INDEX 0

/**
 * Binary search to find partition containing global_idx (Baseline)
 *
 * COMPLEXITY: O(log n)
 * USAGE: Fallback when learned index not available
 */
template<typename T>
__device__ __forceinline__ int findPartitionBinarySearch(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
    int left = 0, right = compressed->num_partitions - 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        int start = compressed->d_start_indices[mid];
        int end = compressed->d_end_indices[mid];

        if (global_idx >= start && global_idx < end) {
            return mid;
        } else if (global_idx < start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return 0; // Fallback (should not reach here if index is valid)
}

/**
 * Learned index lookup (ALEX-inspired)
 *
 * COMPLEXITY: O(1) expected, O(log n) worst case
 * PERFORMANCE: Typically 3-5× faster than binary search for random access
 *
 * ALGORITHM:
 * 1. Use linear model to predict partition (O(1))
 * 2. Check predicted partition and neighbors (O(1) expected)
 * 3. Binary search fallback if needed (rare)
 */
template<typename T>
__device__ __forceinline__ int findPartitionLearned(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
    // Linear model prediction
    // Assume partitions are roughly uniform, so: partition_idx ≈ global_idx / avg_partition_size
    int num_partitions = compressed->num_partitions;
    int total_elements = compressed->d_end_indices[num_partitions - 1];
    double avg_partition_size = (double)total_elements / num_partitions;

    int pred = (int)(global_idx / avg_partition_size);
    pred = max(0, min(num_partitions - 1, pred));

    // Check predicted partition
    if (global_idx >= compressed->d_start_indices[pred] &&
        global_idx < compressed->d_end_indices[pred]) {
        return pred;
    }

    // Check neighbors (handles most off-by-one cases)
    if (pred > 0 && global_idx >= compressed->d_start_indices[pred-1] &&
        global_idx < compressed->d_end_indices[pred-1]) {
        return pred - 1;
    }
    if (pred < num_partitions - 1 && global_idx >= compressed->d_start_indices[pred+1] &&
        global_idx < compressed->d_end_indices[pred+1]) {
        return pred + 1;
    }

    // Rare case: binary search fallback
    return findPartitionBinarySearch(compressed, global_idx);
}

/**
 * Main partition lookup function (dispatch based on config)
 */
template<typename T>
__device__ __forceinline__ int findPartition(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
#if USE_LEARNED_INDEX
    return findPartitionLearned(compressed, global_idx);
#else
    return findPartitionBinarySearch(compressed, global_idx);
#endif
}

/**
 * Extract delta from bit-packed array (optimized version)
 */
template<typename T>
__device__ __forceinline__ long long extractDelta_Device(
    const uint32_t* delta_array,
    int64_t bit_offset,
    int delta_bits)
{
    if (delta_bits == 0) return 0;

    int64_t word_offset = bit_offset >> 5;  // Divide by 32
    int bit_in_word = bit_offset & 31;      // Modulo 32

    uint32_t word1 = delta_array[word_offset];
    uint32_t extracted = word1 >> bit_in_word;

    // Check if we need second word
    int bits_in_first = 32 - bit_in_word;
    if (delta_bits > bits_in_first) {
        uint32_t word2 = delta_array[word_offset + 1];
        uint32_t mask2 = (1U << (delta_bits - bits_in_first)) - 1;
        extracted |= (word2 & mask2) << bits_in_first;
    }

    // Mask to delta_bits
    uint32_t mask = (delta_bits == 32) ? ~0U : ((1U << delta_bits) - 1);
    extracted &= mask;

    // Sign extend - ALWAYS for int types (residuals are always signed)
    uint32_t sign_bit = extracted >> (delta_bits - 1);
    if (sign_bit) {
        uint32_t sign_extend = ~mask;
        extracted |= sign_extend;
    }

    // Cast to signed, then to long long to preserve sign
    int32_t signed_extracted = static_cast<int32_t>(extracted);
    return static_cast<long long>(signed_extracted);
}

/**
 * Random access decompression of a single element
 * Fast path using d_plain_deltas if available
 */
template<typename T>
__device__ __forceinline__ T randomAccessDecompress(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
    // Find partition
    int partition_idx = findPartition(compressed, global_idx);

    // Load partition metadata
    int start_idx = compressed->d_start_indices[partition_idx];
    int local_idx = global_idx - start_idx;

    // Fast path: use pre-unpacked deltas
    if (compressed->d_plain_deltas != nullptr) {
        long long delta = compressed->d_plain_deltas[global_idx];
        double theta0 = compressed->d_model_params[partition_idx * 4];
        double theta1 = compressed->d_model_params[partition_idx * 4 + 1];
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
        T predicted_T = static_cast<T>(__double2int_rn(predicted));

        if constexpr (std::is_signed<T>::value) {
            return predicted_T + static_cast<T>(delta);
        } else {
            return static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
        }
    }

    // Standard path: extract from bit-packed array
    int delta_bits = compressed->d_delta_bits[partition_idx];
    long long delta = 0;

    if (delta_bits > 0 && compressed->delta_array != nullptr) {
        int64_t bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
        int64_t bit_offset = bit_offset_base + (int64_t)local_idx * delta_bits;
        delta = extractDelta_Device<T>(compressed->delta_array, bit_offset, delta_bits);
    }

    // Apply model
    double theta0 = compressed->d_model_params[partition_idx * 4];
    double theta1 = compressed->d_model_params[partition_idx * 4 + 1];
    double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
    T predicted_T = static_cast<T>(__double2int_rn(predicted));

    if constexpr (std::is_signed<T>::value) {
        return predicted_T + static_cast<T>(delta);
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
    }
}

/**
 * Partition metadata cache structure
 */
struct PartitionMeta {
    int start_idx;
    int partition_len;
    int delta_bits;
    int64_t bit_offset_base;
    double theta0;
    double theta1;
};

/**
 * Load partition metadata into shared memory
 */
template<typename T>
__device__ __forceinline__ void loadPartitionMeta(
    const CompressedDataGLECO<T>* compressed,
    int partition_idx,
    PartitionMeta& meta)
{
    meta.start_idx = compressed->d_start_indices[partition_idx];
    meta.partition_len = compressed->d_end_indices[partition_idx] - meta.start_idx;
    meta.delta_bits = compressed->d_delta_bits[partition_idx];
    meta.bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
    meta.theta0 = compressed->d_model_params[partition_idx * 4];
    meta.theta1 = compressed->d_model_params[partition_idx * 4 + 1];
}

/**
 * Decompress value using cached partition metadata
 */
template<typename T>
__device__ __forceinline__ T decompressWithMeta(
    const CompressedDataGLECO<T>* compressed,
    int global_idx,
    const PartitionMeta& meta)
{
    int local_idx = global_idx - meta.start_idx;

    // Fast path
    if (compressed->d_plain_deltas != nullptr) {
        long long delta = compressed->d_plain_deltas[global_idx];
        double predicted = fma(meta.theta1, static_cast<double>(local_idx), meta.theta0);
        T predicted_T = static_cast<T>(__double2int_rn(predicted));

        if constexpr (std::is_signed<T>::value) {
            return predicted_T + static_cast<T>(delta);
        } else {
            return static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
        }
    }

    // Standard path
    long long delta = 0;
    if (meta.delta_bits > 0 && compressed->delta_array != nullptr) {
        int64_t bit_offset = meta.bit_offset_base + (int64_t)local_idx * meta.delta_bits;
        delta = extractDelta_Device<T>(compressed->delta_array, bit_offset, meta.delta_bits);
    }

    double predicted = fma(meta.theta1, static_cast<double>(local_idx), meta.theta0);
    T predicted_T = static_cast<T>(__double2int_rn(predicted));

    if constexpr (std::is_signed<T>::value) {
        return predicted_T + static_cast<T>(delta);
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted_T) + delta);
    }
}
