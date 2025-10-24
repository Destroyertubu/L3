/**
 * GLECO Random Access Kernel Implementations
 *
 * High-performance CUDA kernels for random access to compressed data.
 *
 * VERSION: 1.1.0 - Updated with 64-bit delta support
 * DATE: 2025-10-23
 */

#include "L3_random_access.hpp"
#include "L3_format.hpp"
#include "bitpack_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ============================================================================
// Device Utilities
// ============================================================================

/**
 * Extract delta from bit-packed array (device function)
 * UPDATED: Now supports 64-bit deltas using optimized 128-bit window extraction
 */
template<typename T>
__device__ __forceinline__ int64_t extractDelta(
    const uint32_t* delta_array,
    int64_t bit_offset,
    int bit_width)
{
    if (bit_width <= 0) return 0;
    if (bit_width > 64) bit_width = 64;  // Clamp to max supported

    // Use optimized 64-bit extractor from bitpack_utils.cuh
    uint64_t extracted = extract_bits_upto64_runtime(delta_array, bit_offset, bit_width);

    // Sign extension for signed deltas
    if (bit_width < 64) {
        uint64_t sign_bit = (extracted >> (bit_width - 1)) & 1ULL;
        if (sign_bit) {
            // Sign extend: set all upper bits
            uint64_t extend_mask = ~((1ULL << bit_width) - 1ULL);
            extracted |= extend_mask;
        }
    }

    return static_cast<int64_t>(extracted);
}

/**
 * Apply delta to predicted value
 */
template<typename T>
__device__ __forceinline__ T applyDelta(T predicted, int64_t delta)
{
    if constexpr (std::is_signed<T>::value) {
        return predicted + static_cast<T>(delta);
    } else {
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

// ============================================================================
// Partition Lookup Functions
// ============================================================================

/**
 * Binary search to find partition containing global_idx
 */
template<typename T>
__device__ int findPartition(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
    int left = 0;
    int right = compressed->num_partitions - 1;
    int partition_idx = 0;

    while (left <= right) {
        int mid = (left + right) >> 1;
        int start = compressed->d_start_indices[mid];
        int end = compressed->d_end_indices[mid];

        if (global_idx >= start && global_idx < end) {
            partition_idx = mid;
            break;
        } else if (global_idx < start) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return partition_idx;
}

/**
 * Cached partition lookup using shared memory LRU cache
 */
template<typename T, int CACHE_SIZE = 8>
__device__ int findPartitionCached(
    const CompressedDataGLECO<T>* compressed,
    int global_idx,
    int* s_cache_partition_ids,
    int* s_cache_start_indices,
    int* s_cache_end_indices)
{
    // Check cache first
    for (int i = 0; i < CACHE_SIZE; i++) {
        int pid = s_cache_partition_ids[i];
        if (pid >= 0 && global_idx >= s_cache_start_indices[i] &&
            global_idx < s_cache_end_indices[i]) {
            return pid;  // Cache hit
        }
    }

    // Cache miss: do binary search
    int partition_idx = findPartition(compressed, global_idx);

    // Update cache (simple round-robin replacement)
    int cache_slot = threadIdx.x % CACHE_SIZE;
    s_cache_partition_ids[cache_slot] = partition_idx;
    s_cache_start_indices[cache_slot] = compressed->d_start_indices[partition_idx];
    s_cache_end_indices[cache_slot] = compressed->d_end_indices[partition_idx];

    return partition_idx;
}

// ============================================================================
// Single-Element Random Access
// ============================================================================

/**
 * Random access a single element (device function)
 */
template<typename T>
__device__ T randomAccessElement(
    const CompressedDataGLECO<T>* compressed,
    int global_idx)
{
    // Find partition
    int partition_idx = findPartition(compressed, global_idx);

    // Load partition metadata
    int start_idx = compressed->d_start_indices[partition_idx];
    int model_type = compressed->d_model_types[partition_idx];
    int delta_bits = compressed->d_delta_bits[partition_idx];
    int64_t bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
    double theta0 = compressed->d_model_params[partition_idx * 4];
    double theta1 = compressed->d_model_params[partition_idx * 4 + 1];

    int local_idx = global_idx - start_idx;

    // Extract delta (now 64-bit)
    int64_t delta = 0;
    if (compressed->d_plain_deltas != nullptr) {
        // Fast path: pre-unpacked deltas
        delta = compressed->d_plain_deltas[global_idx];
    } else if (delta_bits > 0 && compressed->delta_array != nullptr) {
        // Bit-packed deltas (supports up to 64-bit)
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
        delta = extractDelta<T>(compressed->delta_array, bit_offset, delta_bits);
    }

    // Compute final value
    T final_value;
    if (model_type == MODEL_DIRECT_COPY) {
        final_value = static_cast<T>(delta);
    } else {
        // Linear model prediction
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
        // CRITICAL: Must use same rounding as encoder/decoder (round()) for bit-exact correctness
        T predicted_T = static_cast<T>(round(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    return final_value;
}

// ============================================================================
// Kernel: Simple Random Access (One Thread Per Index)
// ============================================================================

template<typename T>
__global__ void randomAccessMultipleKernel(
    const CompressedDataGLECO<T>* compressed,
    const int* __restrict__ d_indices,
    int num_indices,
    T* __restrict__ d_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    int global_idx = d_indices[idx];
    d_output[idx] = randomAccessElement(compressed, global_idx);
}

// ============================================================================
// Kernel: Batch Random Access with Partition Grouping
// ============================================================================

/**
 * Batch random access kernel with shared memory caching
 *
 * Each block processes a batch of indices that (ideally) belong to
 * the same or nearby partitions for better cache locality.
 */
template<typename T>
__global__ void randomAccessBatchKernel(
    const CompressedDataGLECO<T>* compressed,
    const int* __restrict__ d_indices,
    int num_indices,
    T* __restrict__ d_output,
    bool use_cache)
{
    constexpr int CACHE_SIZE = 8;

    // Shared memory for partition cache
    __shared__ int s_cache_partition_ids[CACHE_SIZE];
    __shared__ int s_cache_start_indices[CACHE_SIZE];
    __shared__ int s_cache_end_indices[CACHE_SIZE];

    // Initialize cache
    if (threadIdx.x < CACHE_SIZE) {
        s_cache_partition_ids[threadIdx.x] = -1;
        s_cache_start_indices[threadIdx.x] = 0;
        s_cache_end_indices[threadIdx.x] = 0;
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    int global_idx = d_indices[idx];

    // Find partition (with or without cache)
    int partition_idx;
    if (use_cache) {
        partition_idx = findPartitionCached<T, CACHE_SIZE>(
            compressed, global_idx,
            s_cache_partition_ids, s_cache_start_indices, s_cache_end_indices);
    } else {
        partition_idx = findPartition(compressed, global_idx);
    }

    // Load partition metadata
    int start_idx = compressed->d_start_indices[partition_idx];
    int model_type = compressed->d_model_types[partition_idx];
    int delta_bits = compressed->d_delta_bits[partition_idx];
    int64_t bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
    double theta0 = compressed->d_model_params[partition_idx * 4];
    double theta1 = compressed->d_model_params[partition_idx * 4 + 1];

    int local_idx = global_idx - start_idx;

    // Extract delta (64-bit support)
    int64_t delta = 0;
    if (compressed->d_plain_deltas != nullptr) {
        delta = compressed->d_plain_deltas[global_idx];
    } else if (delta_bits > 0 && compressed->delta_array != nullptr) {
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
        delta = extractDelta<T>(compressed->delta_array, bit_offset, delta_bits);
    }

    // Compute final value
    T final_value;
    if (model_type == MODEL_DIRECT_COPY) {
        final_value = static_cast<T>(delta);
    } else {
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
        // CRITICAL: Must use same rounding as encoder/decoder (round()) for bit-exact correctness
        T predicted_T = static_cast<T>(round(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    d_output[idx] = final_value;
}

// ============================================================================
// Kernel: Random Access with Pre-computed Partitions
// ============================================================================

/**
 * Random access when partition indices are known
 * (avoids binary search overhead)
 */
template<typename T>
__global__ void randomAccessWithPartitionsKernel(
    const CompressedDataGLECO<T>* compressed,
    const int* __restrict__ d_indices,
    const int* __restrict__ d_partition_ids,
    int num_indices,
    T* __restrict__ d_output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    int global_idx = d_indices[idx];
    int partition_idx = d_partition_ids[idx];

    // Load partition metadata
    int start_idx = compressed->d_start_indices[partition_idx];
    int model_type = compressed->d_model_types[partition_idx];
    int delta_bits = compressed->d_delta_bits[partition_idx];
    int64_t bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
    double theta0 = compressed->d_model_params[partition_idx * 4];
    double theta1 = compressed->d_model_params[partition_idx * 4 + 1];

    int local_idx = global_idx - start_idx;

    // Extract delta (64-bit support)
    int64_t delta = 0;
    if (compressed->d_plain_deltas != nullptr) {
        delta = compressed->d_plain_deltas[global_idx];
    } else if (delta_bits > 0 && compressed->delta_array != nullptr) {
        int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
        delta = extractDelta<T>(compressed->delta_array, bit_offset, delta_bits);
    }

    // Compute final value
    T final_value;
    if (model_type == MODEL_DIRECT_COPY) {
        final_value = static_cast<T>(delta);
    } else {
        double predicted = fma(theta1, static_cast<double>(local_idx), theta0);
        // CRITICAL: Must use same rounding as encoder/decoder (round()) for bit-exact correctness
        T predicted_T = static_cast<T>(round(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    d_output[idx] = final_value;
}

// ============================================================================
// Kernel: Range-Based Random Access
// ============================================================================

/**
 * Access a contiguous range of elements
 * More efficient than individual random accesses
 */
template<typename T>
__global__ void randomAccessRangeKernel(
    const CompressedDataGLECO<T>* compressed,
    int start_idx,
    int end_idx,
    T* __restrict__ d_output)
{
    __shared__ int s_partition_idx;
    __shared__ int s_start_idx;
    __shared__ int s_model_type;
    __shared__ int s_delta_bits;
    __shared__ int64_t s_bit_offset_base;
    __shared__ double s_theta0, s_theta1;

    int global_idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= end_idx) return;

    // Find partition (coalesced within warp)
    int partition_idx = findPartition(compressed, global_idx);

    // Load metadata (once per warp/block)
    if (threadIdx.x == 0) {
        s_partition_idx = partition_idx;
        s_start_idx = compressed->d_start_indices[partition_idx];
        s_model_type = compressed->d_model_types[partition_idx];
        s_delta_bits = compressed->d_delta_bits[partition_idx];
        s_bit_offset_base = compressed->d_delta_array_bit_offsets[partition_idx];
        s_theta0 = compressed->d_model_params[partition_idx * 4];
        s_theta1 = compressed->d_model_params[partition_idx * 4 + 1];
    }
    __syncthreads();

    int local_idx = global_idx - s_start_idx;

    // Extract delta (64-bit support)
    int64_t delta = 0;
    if (compressed->d_plain_deltas != nullptr) {
        delta = compressed->d_plain_deltas[global_idx];
    } else if (s_delta_bits > 0 && compressed->delta_array != nullptr) {
        int64_t bit_offset = s_bit_offset_base + static_cast<int64_t>(local_idx) * s_delta_bits;
        delta = extractDelta<T>(compressed->delta_array, bit_offset, s_delta_bits);
    }

    // Compute final value
    T final_value;
    if (s_model_type == MODEL_DIRECT_COPY) {
        final_value = static_cast<T>(delta);
    } else {
        double predicted = fma(s_theta1, static_cast<double>(local_idx), s_theta0);
        // CRITICAL: Must use same rounding as encoder/decoder (round()) for bit-exact correctness
        T predicted_T = static_cast<T>(round(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    int output_idx = global_idx - start_idx;
    d_output[output_idx] = final_value;
}

// ============================================================================
// Host API Implementations
// ============================================================================

template<typename T>
cudaError_t randomAccessMultiple(
    const CompressedDataGLECO<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    const RandomAccessConfig* config,
    RandomAccessStats* stats,
    cudaStream_t stream)
{
    if (num_indices == 0) return cudaSuccess;

    const int threads_per_block = 256;
    const int num_blocks = (num_indices + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    if (stats != nullptr) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    // Use d_self for device-side access if available
    const CompressedDataGLECO<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessMultipleKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, d_indices, num_indices, d_output);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    if (stats != nullptr) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        stats->time_ms = ms;
        stats->num_accesses = num_indices;
        stats->throughput_gbps = (num_indices * sizeof(T) / 1e9) / (ms / 1e3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return cudaSuccess;
}

template<typename T>
cudaError_t randomAccessBatch(
    const CompressedDataGLECO<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    const RandomAccessConfig& config,
    RandomAccessStats* stats,
    cudaStream_t stream)
{
    if (num_indices == 0) return cudaSuccess;

    const int threads_per_block = 256;
    const int num_blocks = (num_indices + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start, stop;
    if (stats != nullptr) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
    }

    // Use d_self for device-side access if available
    const CompressedDataGLECO<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessBatchKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, d_indices, num_indices, d_output, config.enable_partition_cache);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    if (stats != nullptr) {
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        stats->time_ms = ms;
        stats->num_accesses = num_indices;
        stats->throughput_gbps = (num_indices * sizeof(T) / 1e9) / (ms / 1e3);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    return cudaSuccess;
}

template<typename T>
cudaError_t randomAccessWithPartitions(
    const CompressedDataGLECO<T>* compressed,
    const int* d_indices,
    const int* d_partition_ids,
    int num_indices,
    T* d_output,
    cudaStream_t stream)
{
    if (num_indices == 0) return cudaSuccess;

    const int threads_per_block = 256;
    const int num_blocks = (num_indices + threads_per_block - 1) / threads_per_block;

    // Use d_self for device-side access if available
    const CompressedDataGLECO<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessWithPartitionsKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, d_indices, d_partition_ids, num_indices, d_output);

    return cudaGetLastError();
}

template<typename T>
cudaError_t randomAccessRange(
    const CompressedDataGLECO<T>* compressed,
    int start_idx,
    int end_idx,
    T* d_output,
    cudaStream_t stream)
{
    if (start_idx >= end_idx) return cudaSuccess;

    int num_elements = end_idx - start_idx;
    const int threads_per_block = 256;
    const int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    // Use d_self for device-side access if available
    const CompressedDataGLECO<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessRangeKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, start_idx, end_idx, d_output);

    return cudaGetLastError();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// randomAccessElement (device function)
template __device__ int32_t randomAccessElement(const CompressedDataGLECO<int32_t>*, int);
template __device__ uint32_t randomAccessElement(const CompressedDataGLECO<uint32_t>*, int);
template __device__ int64_t randomAccessElement(const CompressedDataGLECO<int64_t>*, int);
template __device__ uint64_t randomAccessElement(const CompressedDataGLECO<uint64_t>*, int);

// randomAccessMultiple
template cudaError_t randomAccessMultiple<int32_t>(
    const CompressedDataGLECO<int32_t>*, const int*, int, int32_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<uint32_t>(
    const CompressedDataGLECO<uint32_t>*, const int*, int, uint32_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<int64_t>(
    const CompressedDataGLECO<int64_t>*, const int*, int, int64_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<uint64_t>(
    const CompressedDataGLECO<uint64_t>*, const int*, int, uint64_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);

// randomAccessBatch
template cudaError_t randomAccessBatch<int32_t>(
    const CompressedDataGLECO<int32_t>*, const int*, int, int32_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<uint32_t>(
    const CompressedDataGLECO<uint32_t>*, const int*, int, uint32_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<int64_t>(
    const CompressedDataGLECO<int64_t>*, const int*, int, int64_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<uint64_t>(
    const CompressedDataGLECO<uint64_t>*, const int*, int, uint64_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);

// randomAccessWithPartitions
template cudaError_t randomAccessWithPartitions<int32_t>(
    const CompressedDataGLECO<int32_t>*, const int*, const int*, int, int32_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<uint32_t>(
    const CompressedDataGLECO<uint32_t>*, const int*, const int*, int, uint32_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<int64_t>(
    const CompressedDataGLECO<int64_t>*, const int*, const int*, int, int64_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<uint64_t>(
    const CompressedDataGLECO<uint64_t>*, const int*, const int*, int, uint64_t*, cudaStream_t);

// randomAccessRange
template cudaError_t randomAccessRange<int32_t>(
    const CompressedDataGLECO<int32_t>*, int, int, int32_t*, cudaStream_t);
template cudaError_t randomAccessRange<uint32_t>(
    const CompressedDataGLECO<uint32_t>*, int, int, uint32_t*, cudaStream_t);
template cudaError_t randomAccessRange<int64_t>(
    const CompressedDataGLECO<int64_t>*, int, int, int64_t*, cudaStream_t);
template cudaError_t randomAccessRange<uint64_t>(
    const CompressedDataGLECO<uint64_t>*, int, int, uint64_t*, cudaStream_t);
