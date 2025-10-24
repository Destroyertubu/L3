/**
 * Partition Bounds Computation Kernel
 *
 * Computes min/max value bounds for each partition
 * Used for predicate pushdown optimization
 *
 * DATE: 2025-10-17
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "L3_format.hpp"

/**
 * Kernel to compute min/max bounds for each partition
 *
 * Each block processes one partition using parallel reduction
 *
 * PARAMETERS:
 * - d_values: Device array of original values
 * - d_start_indices: Start index of each partition
 * - d_end_indices: End index of each partition
 * - d_partition_min: Output array of minimum values [num_partitions]
 * - d_partition_max: Output array of maximum values [num_partitions]
 * - num_partitions: Number of partitions
 */
template<typename T>
__global__ void computePartitionBoundsKernel(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int start_idx = d_start_indices[partition_idx];
    int end_idx = d_end_indices[partition_idx];
    int partition_len = end_idx - start_idx;

    if (partition_len <= 0) return;

    // Shared memory for reduction
    extern __shared__ char shared_mem[];
    T* s_min = reinterpret_cast<T*>(shared_mem);
    T* s_max = reinterpret_cast<T*>(shared_mem + blockDim.x * sizeof(T));

    int tid = threadIdx.x;

    // Initialize with first element (all threads will get valid values from loop)
    T local_min = d_values[start_idx];
    T local_max = d_values[start_idx];

    // Process all elements in partition with grid-stride loop
    for (int i = tid; i < partition_len; i += blockDim.x) {
        T value = d_values[start_idx + i];
        if (value < local_min) local_min = value;
        if (value > local_max) local_max = value;
    }

    // Store in shared memory
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    // Parallel reduction to find global min/max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min[tid + s] < s_min[tid]) {
                s_min[tid] = s_min[tid + s];
            }
            if (s_max[tid + s] > s_max[tid]) {
                s_max[tid] = s_max[tid + s];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes result
    if (tid == 0) {
        d_partition_min[partition_idx] = s_min[0];
        d_partition_max[partition_idx] = s_max[0];
    }
}

/**
 * Host wrapper to launch partition bounds kernel
 */
template<typename T>
void launchComputePartitionBounds(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions,
    cudaStream_t stream = 0)
{
    int threads_per_block = 256;
    int blocks = num_partitions;

    size_t shared_mem_size = 2 * threads_per_block * sizeof(T);

    computePartitionBoundsKernel<T><<<blocks, threads_per_block, shared_mem_size, stream>>>(
        d_values,
        d_start_indices,
        d_end_indices,
        d_partition_min,
        d_partition_max,
        num_partitions
    );
}

// Template instantiations for common types
template void launchComputePartitionBounds<int32_t>(
    const int32_t*, const int32_t*, const int32_t*, int32_t*, int32_t*, int, cudaStream_t);
template void launchComputePartitionBounds<uint32_t>(
    const uint32_t*, const int32_t*, const int32_t*, uint32_t*, uint32_t*, int, cudaStream_t);
template void launchComputePartitionBounds<int64_t>(
    const int64_t*, const int32_t*, const int32_t*, int64_t*, int64_t*, int, cudaStream_t);
template void launchComputePartitionBounds<uint64_t>(
    const uint64_t*, const int32_t*, const int32_t*, uint64_t*, uint64_t*, int, cudaStream_t);
