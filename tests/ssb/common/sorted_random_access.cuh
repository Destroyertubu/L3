/**
 * @file sorted_random_access.cuh
 * @brief Index Sorting for Cache-Affine Random Access
 *
 * This file implements CUB-based index sorting to improve L2 cache hit rate
 * during random access of compressed data.
 *
 * Key optimization:
 * - Sort indices by partition ID before random access
 * - Consecutive indices access same partition -> better cache locality
 * - Uses CUB DeviceRadixSort for GPU-efficient sorting
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cub/cub.cuh>
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

namespace ssb {

// ============================================================================
// Partition ID Computation
// ============================================================================

/**
 * @brief Kernel to compute partition ID for each index using binary search
 *
 * @param indices Global element indices to look up
 * @param partition_starts Array of partition start indices (sorted)
 * @param num_partitions Number of partitions
 * @param num_indices Number of indices to process
 * @param partition_ids Output: partition ID for each index
 */
__global__ void computePartitionIdsKernel(
    const int* __restrict__ indices,
    const int32_t* __restrict__ partition_starts,
    int num_partitions,
    int num_indices,
    uint16_t* __restrict__ partition_ids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    int global_idx = indices[idx];

    // Binary search to find partition containing global_idx
    int lo = 0, hi = num_partitions - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (partition_starts[mid] <= global_idx) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    partition_ids[idx] = static_cast<uint16_t>(lo);
}

/**
 * @brief Kernel to reorder output values back to original index order
 *
 * After sorting and decompressing, values are in sorted order.
 * This kernel puts them back in the original user-requested order.
 *
 * @param sorted_values Values in sorted (partition-grouped) order
 * @param permutation Permutation array: permutation[i] = original position of sorted element i
 * @param num_values Number of values
 * @param output Output in original order
 */
template<typename T>
__global__ void reorderOutputKernel(
    const T* __restrict__ sorted_values,
    const int* __restrict__ permutation,
    int num_values,
    T* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;

    // permutation[idx] tells us where sorted_values[idx] should go
    output[permutation[idx]] = sorted_values[idx];
}

// ============================================================================
// Sorted Random Access Implementation
// ============================================================================

/**
 * @brief Sorted random access with partition-based index sorting
 *
 * Steps:
 * 1. Compute partition ID for each index (binary search)
 * 2. Sort indices by partition ID (CUB RadixSort)
 * 3. Decompress in sorted order (better cache locality)
 * 4. Reorder output to match original index order
 *
 * @tparam T Element type (uint32_t, uint64_t, etc.)
 * @param compressed Compressed column data
 * @param d_indices Device array of indices to access (not modified)
 * @param num_indices Number of indices
 * @param d_output Output array in original index order
 * @param stream CUDA stream
 */
template<typename T>
void sortedRandomAccessVertical(
    const CompressedDataVertical<T>& compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (num_indices == 0) return;

    // Threshold: only sort if enough indices to benefit
    constexpr int SORT_THRESHOLD = 10000;
    if (num_indices < SORT_THRESHOLD) {
        // Direct random access for small batches
        Vertical_decoder::decompressIndices(compressed, d_indices, num_indices, d_output, stream);
        return;
    }

    // =========================================
    // Step 1: Compute partition IDs
    // =========================================
    uint16_t* d_partition_ids;
    cudaMalloc(&d_partition_ids, num_indices * sizeof(uint16_t));

    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;
    computePartitionIdsKernel<<<blocks, threads, 0, stream>>>(
        d_indices,
        compressed.d_start_indices,
        compressed.num_partitions,
        num_indices,
        d_partition_ids);

    // =========================================
    // Step 2: Sort indices by partition ID
    // =========================================
    // We need to keep track of original positions for reordering
    int* d_sorted_indices;
    int* d_original_positions;
    int* d_sorted_positions;
    uint16_t* d_sorted_partition_ids;

    cudaMalloc(&d_sorted_indices, num_indices * sizeof(int));
    cudaMalloc(&d_original_positions, num_indices * sizeof(int));
    cudaMalloc(&d_sorted_positions, num_indices * sizeof(int));
    cudaMalloc(&d_sorted_partition_ids, num_indices * sizeof(uint16_t));

    // Initialize original positions as sequence 0, 1, 2, ...
    auto initPositions = [] __device__ (int idx) { return idx; };
    thrust::tabulate(thrust::cuda::par.on(stream),
                     d_original_positions, d_original_positions + num_indices,
                     initPositions);

    // Copy indices (CUB sort modifies keys in place)
    cudaMemcpyAsync(d_sorted_indices, d_indices, num_indices * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Determine temporary storage requirements
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // Sort (partition_ids, indices) pairs, also permuting positions
    // First pass: get temp storage size
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_sorted_indices, d_sorted_indices,  // indices sorted by partition_id
        num_indices, 0, 16, stream);  // 16 bits for partition_ids

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // We need to sort (partition_id, (index, position)) triples
    // CUB SortPairs only handles key-value, so we'll do two sorts or use SortPairs twice

    // Alternative: use stable sort with positions as secondary key
    // Actually, let's use a simpler approach: sort indices, track permutation

    // Create combined key-value: key = partition_id, value = index
    // Then separately track positions

    // First: sort indices by partition_id
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_sorted_indices, d_sorted_indices,  // sort indices
        num_indices, 0, 16, stream);

    // Now d_sorted_indices contains indices sorted by partition
    // We need permutation: where does each sorted element belong in output?
    // Solution: sort positions the same way

    cudaMemcpyAsync(d_sorted_positions, d_original_positions,
                    num_indices * sizeof(int), cudaMemcpyDeviceToDevice, stream);

    // Reset and sort positions by same partition_ids
    cudaMemcpyAsync(d_partition_ids + num_indices/2, d_partition_ids,
                    num_indices * sizeof(uint16_t) / 2, cudaMemcpyDeviceToDevice, stream);

    // Actually, we need a cleaner approach. Let's re-do:
    // Create (partition_id, original_position) pairs
    // Sort by partition_id
    // Result: sorted_positions[i] = original position that will be processed at step i

    // Re-compute partition IDs (quick)
    computePartitionIdsKernel<<<blocks, threads, 0, stream>>>(
        d_indices, compressed.d_start_indices, compressed.num_partitions,
        num_indices, d_partition_ids);

    // Sort: key=partition_id, value=original_position
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_original_positions, d_sorted_positions,
        num_indices, 0, 16, stream);

    // Now d_sorted_positions tells us which original index to process at each step
    // We need to gather the actual indices
    // d_gathered_indices[i] = d_indices[d_sorted_positions[i]]
    int* d_gathered_indices;
    cudaMalloc(&d_gathered_indices, num_indices * sizeof(int));

    // Gather kernel
    auto gatherKernel = [=] __device__ (int idx) {
        if (idx < num_indices) {
            return d_indices[d_sorted_positions[idx]];
        }
        return 0;
    };

    // Simple gather kernel
    auto gather = [d_indices, d_sorted_positions, d_gathered_indices, num_indices] __global__ () {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_indices) {
            d_gathered_indices[idx] = d_indices[d_sorted_positions[idx]];
        }
    };

    // Launch gather
    [&]() {
        int* indices_ptr = const_cast<int*>(d_indices);
        auto gatherLambda = [indices_ptr, d_sorted_positions, d_gathered_indices, num_indices]
            __global__ (void) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < num_indices) {
                d_gathered_indices[idx] = indices_ptr[d_sorted_positions[idx]];
            }
        };
    }();

    // Actually use a proper kernel
    cudaFree(d_gathered_indices);
    cudaMalloc(&d_gathered_indices, num_indices * sizeof(int));

    // =========================================
    // Step 3: Decompress in sorted order
    // =========================================
    T* d_sorted_output;
    cudaMalloc(&d_sorted_output, num_indices * sizeof(T));

    // Simple gather and decompress
    // Gather indices based on sorted positions
    struct GatherKernelParams {
        const int* indices;
        const int* positions;
        int* gathered;
        int n;
    };

    // Launch a simple gather kernel
    {
        auto gatherFn = [] __device__ (const int* indices, const int* positions,
                                       int* gathered, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                gathered[idx] = indices[positions[idx]];
            }
        };

        // Use a named kernel instead
    }

    // Simplified: create gathered indices array
    cudaMemcpyAsync(d_gathered_indices, d_indices, num_indices * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    // Sort the indices directly by partition (simpler approach)
    // We already have partition IDs, just sort indices by them
    computePartitionIdsKernel<<<blocks, threads, 0, stream>>>(
        d_indices, compressed.d_start_indices, compressed.num_partitions,
        num_indices, d_partition_ids);

    cudaMemcpyAsync(d_gathered_indices, d_indices, num_indices * sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_gathered_indices, d_sorted_indices,
        num_indices, 0, 16, stream);

    // Decompress sorted indices
    Vertical_decoder::decompressIndices(compressed, d_sorted_indices, num_indices,
                                         d_sorted_output, stream);

    // =========================================
    // Step 4: Reorder to original order
    // =========================================
    // We need inverse permutation
    // sorted_position[i] = original position of sorted element i
    // We computed: d_sorted_positions[i] = original position at sorted step i

    // Reorder
    reorderOutputKernel<<<blocks, threads, 0, stream>>>(
        d_sorted_output, d_sorted_positions, num_indices, d_output);

    // =========================================
    // Cleanup
    // =========================================
    cudaFree(d_temp_storage);
    cudaFree(d_partition_ids);
    cudaFree(d_sorted_partition_ids);
    cudaFree(d_sorted_indices);
    cudaFree(d_original_positions);
    cudaFree(d_sorted_positions);
    cudaFree(d_gathered_indices);
    cudaFree(d_sorted_output);
}

// ============================================================================
// Simplified Version (more reliable)
// ============================================================================

/**
 * @brief Gather kernel for index reordering
 */
__global__ void gatherIndicesKernel(
    const int* __restrict__ src_indices,
    const int* __restrict__ gather_positions,
    int num_indices,
    int* __restrict__ dst_indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;
    dst_indices[idx] = src_indices[gather_positions[idx]];
}

/**
 * @brief Scatter kernel for output reordering
 */
template<typename T>
__global__ void scatterOutputKernel(
    const T* __restrict__ src_values,
    const int* __restrict__ scatter_positions,
    int num_values,
    T* __restrict__ dst_values)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_values) return;
    dst_values[scatter_positions[idx]] = src_values[idx];
}

/**
 * @brief Simplified sorted random access (cleaner implementation)
 */
template<typename T>
void sortedRandomAccessSimple(
    const CompressedDataVertical<T>& compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    cudaStream_t stream = 0)
{
    if (num_indices == 0) return;

    constexpr int SORT_THRESHOLD = 5000;
    if (num_indices < SORT_THRESHOLD) {
        Vertical_decoder::decompressIndices(compressed, d_indices, num_indices, d_output, stream);
        return;
    }

    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;

    // Allocate working arrays
    uint16_t *d_partition_ids, *d_sorted_partition_ids;
    int *d_positions, *d_sorted_positions;
    int *d_sorted_indices;
    T *d_sorted_values;

    cudaMalloc(&d_partition_ids, num_indices * sizeof(uint16_t));
    cudaMalloc(&d_sorted_partition_ids, num_indices * sizeof(uint16_t));
    cudaMalloc(&d_positions, num_indices * sizeof(int));
    cudaMalloc(&d_sorted_positions, num_indices * sizeof(int));
    cudaMalloc(&d_sorted_indices, num_indices * sizeof(int));
    cudaMalloc(&d_sorted_values, num_indices * sizeof(T));

    // Step 1: Compute partition IDs
    computePartitionIdsKernel<<<blocks, threads, 0, stream>>>(
        d_indices, compressed.d_start_indices, compressed.num_partitions,
        num_indices, d_partition_ids);

    // Step 2: Initialize position array [0, 1, 2, ...]
    auto initPosKernel = [] __device__ (int* positions, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) positions[idx] = idx;
    };
    // Use thrust
    thrust::sequence(thrust::cuda::par.on(stream), d_positions, d_positions + num_indices);

    // Step 3: Sort (partition_id, position) pairs
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_positions, d_sorted_positions,
        num_indices, 0, 16, stream);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_partition_ids, d_sorted_partition_ids,
        d_positions, d_sorted_positions,
        num_indices, 0, 16, stream);

    // Step 4: Gather indices in sorted order
    gatherIndicesKernel<<<blocks, threads, 0, stream>>>(
        d_indices, d_sorted_positions, num_indices, d_sorted_indices);

    // Step 5: Decompress in sorted order (cache-friendly)
    Vertical_decoder::decompressIndices(compressed, d_sorted_indices, num_indices,
                                         d_sorted_values, stream);

    // Step 6: Scatter back to original order
    scatterOutputKernel<<<blocks, threads, 0, stream>>>(
        d_sorted_values, d_sorted_positions, num_indices, d_output);

    // Cleanup
    cudaFree(d_temp);
    cudaFree(d_partition_ids);
    cudaFree(d_sorted_partition_ids);
    cudaFree(d_positions);
    cudaFree(d_sorted_positions);
    cudaFree(d_sorted_indices);
    cudaFree(d_sorted_values);
}

}  // namespace ssb
