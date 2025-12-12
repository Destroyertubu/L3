/**
 * L3 Random Access Kernel Implementations
 *
 * High-performance CUDA kernels for random access to compressed data.
 *
 * VERSION: 1.2.0 - Updated with branchless extraction (matches Vertical optimization)
 * DATE: 2025-12-08
 */

#include "L3_random_access.hpp"
#include "L3_format.hpp"
#include "bitpack_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>

// ============================================================================
// Device Utilities - Branchless Bit Extraction (Vertical-style optimization)
// ============================================================================

/**
 * Branchless 64-bit mask generation
 */
__device__ __forceinline__ uint64_t mask64_branchless(int k) {
    // Branchless: use shift trick
    // For k >= 64, returns ~0ULL; for k <= 0, returns 0
    uint64_t full_mask = ~0ULL;
    int shift = 64 - k;
    // Clamp shift to [0, 63] to avoid undefined behavior
    shift = (shift < 0) ? 0 : ((shift > 63) ? 63 : shift);
    return full_mask >> shift;
}

/**
 * Branchless sign extension (matches Vertical sign_extend_64)
 */
__device__ __forceinline__ int64_t signExtendBranchless(uint64_t value, int bit_width) {
    if (bit_width <= 0 || bit_width >= 64) {
        return static_cast<int64_t>(value);
    }
    // Branchless sign extension using arithmetic shift
    int shift = 64 - bit_width;
    return static_cast<int64_t>(value << shift) >> shift;
}

/**
 * Branchless 64-bit extraction from bit-packed array
 *
 * OPTIMIZATION (matching Vertical):
 * - Always loads 128-bit window (2 x 64-bit words)
 * - Always performs shift+OR (no boundary check)
 * - Eliminates warp divergence
 */
__device__ __forceinline__ uint64_t extractBranchless64(
    const uint32_t* __restrict__ words,
    int64_t start_bit,
    int bits)
{
    if (bits <= 0) return 0ULL;
    if (bits > 64) bits = 64;

    // Compute 64-bit word index and bit offset
    const uint64_t word64_idx = start_bit >> 6;    // start_bit / 64
    const int bit_offset = start_bit & 63;         // start_bit % 64

    const uint64_t* __restrict__ p64 = reinterpret_cast<const uint64_t*>(words);

    // BRANCHLESS: Always load both words
    const uint64_t lo = __ldg(&p64[word64_idx]);
    const uint64_t hi = __ldg(&p64[word64_idx + 1]);

    // BRANCHLESS stitch: always compute both parts and combine
    // When bit_offset == 0: shifted_hi becomes 0 (harmless)
    const uint64_t shifted_lo = lo >> bit_offset;
    const uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));

    return (shifted_lo | shifted_hi) & mask64_branchless(bits);
}

/**
 * Extract delta from bit-packed array (device function)
 *
 * VERSION 1.2.0: Now uses branchless extraction matching Vertical
 */
template<typename T>
__device__ __forceinline__ int64_t extractDelta(
    const uint32_t* delta_array,
    int64_t bit_offset,
    int bit_width)
{
    if (bit_width <= 0) return 0;
    if (bit_width > 64) bit_width = 64;

    // Use branchless extraction
    uint64_t extracted = extractBranchless64(delta_array, bit_offset, bit_width);

    // Use branchless sign extension
    return signExtendBranchless(extracted, bit_width);
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
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* compressed,
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
    double theta2 = compressed->d_model_params[partition_idx * 4 + 2];  // For POLY2
    double theta3 = compressed->d_model_params[partition_idx * 4 + 3];  // For POLY3

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
        // Compute prediction using Horner's method based on model type
        double x = static_cast<double>(local_idx);
        double predicted;
        switch (model_type) {
            case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                predicted = theta0 + x * (theta1 + x * theta2);
                break;
            case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
                break;
            default:  // LINEAR, CONSTANT, etc.
                predicted = fma(theta1, x, theta0);
                break;
        }
        // CRITICAL: Must use __double2ll_rn (banker's rounding) to match V2 partitioner
        // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
        if constexpr (std::is_unsigned<T>::value) {
            if (predicted < 0.0) predicted = 0.0;
        }
        T predicted_T = static_cast<T>(__double2ll_rn(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    return final_value;
}

// ============================================================================
// Kernel: Simple Random Access (One Thread Per Index)
// ============================================================================

template<typename T>
__global__ void randomAccessMultipleKernel(
    const CompressedDataL3<T>* compressed,
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
// Kernel: Optimized Random Access (Vertical-style direct pointers)
// ============================================================================

/**
 * Optimized random access kernel with direct pointer parameters
 *
 * This kernel matches Vertical structure exactly:
 * - Direct array pointer parameters (no struct indirection)
 * - Binary search for partition lookup
 * - Branchless bit extraction
 *
 * VERSION: 1.2.0 - Matches Vertical random access optimization
 */
template<typename T>
__global__ void randomAccessOptimizedKernel(
    const uint32_t* __restrict__ delta_array,
    const int32_t* __restrict__ d_start_indices,
    const int32_t* __restrict__ d_end_indices,
    const int32_t* __restrict__ d_model_types,
    const double* __restrict__ d_model_params,
    const int32_t* __restrict__ d_delta_bits,
    const int64_t* __restrict__ d_delta_array_bit_offsets,
    int num_partitions,
    const int* __restrict__ query_indices,
    int num_queries,
    T* __restrict__ output)
{
    int qid = threadIdx.x + blockIdx.x * blockDim.x;
    if (qid >= num_queries) return;

    int idx = query_indices[qid];

    // Binary search for partition (same as Vertical)
    int left = 0, right = num_partitions - 1;
    int pid = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (idx >= d_start_indices[mid] && idx < d_end_indices[mid]) {
            pid = mid;
            break;
        } else if (idx < d_start_indices[mid]) {
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    if (pid < 0) {
        output[qid] = 0;  // Invalid index
        return;
    }

    int32_t model_type = d_model_types[pid];
    int32_t delta_bits = d_delta_bits[pid];
    int32_t start_idx = d_start_indices[pid];
    int64_t bit_offset_base = d_delta_array_bit_offsets[pid];

    // Load model parameters
    double theta0 = d_model_params[pid * 4];
    double theta1 = d_model_params[pid * 4 + 1];
    double theta2 = d_model_params[pid * 4 + 2];  // For POLY2
    double theta3 = d_model_params[pid * 4 + 3];  // For POLY3

    int local_idx = idx - start_idx;

    // Compute bit offset (sequential layout)
    int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;

    // Extract delta using branchless extraction
    if (model_type == MODEL_DIRECT_COPY || model_type == MODEL_FOR_BITPACK) {
        // FOR/BitPack model: delta = value - base
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(theta0));
        } else {
            base = static_cast<T>(__double2ll_rn(theta0));
        }

        if (delta_bits == 0) {
            output[qid] = base;
        } else {
            uint64_t delta = extractBranchless64(delta_array, bit_offset, delta_bits);

            // CRITICAL FIX: Use model_type to distinguish between DIRECT_COPY and FOR_BITPACK
            // - MODEL_DIRECT_COPY: deltas are SIGNED values, need sign extension
            // - MODEL_FOR_BITPACK: deltas are UNSIGNED (value - base), no sign extension
            if (model_type == MODEL_DIRECT_COPY) {
                // Sign extend for direct copy (signed values)
                int64_t signed_val = signExtendBranchless(delta, delta_bits);
                output[qid] = static_cast<T>(signed_val);
            } else {
                // FOR_BITPACK: unsigned delta
                output[qid] = base + static_cast<T>(delta);
            }
        }
    } else {
        // Linear/Polynomial model
        double x = static_cast<double>(local_idx);
        double predicted;
        switch (model_type) {
            case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                predicted = theta0 + x * (theta1 + x * theta2);
                break;
            case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
                break;
            default:  // LINEAR, CONSTANT, etc.
                predicted = fma(theta1, x, theta0);
                break;
        }

        // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
        if constexpr (std::is_unsigned<T>::value) {
            if (predicted < 0.0) predicted = 0.0;
        }

        if (delta_bits == 0) {
            output[qid] = static_cast<T>(__double2ll_rn(predicted));
        } else {
            T pred_val = static_cast<T>(__double2ll_rn(predicted));
            uint64_t extracted = extractBranchless64(delta_array, bit_offset, delta_bits);
            int64_t delta = signExtendBranchless(extracted, delta_bits);
            output[qid] = static_cast<T>(static_cast<int64_t>(pred_val) + delta);
        }
    }
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
    const CompressedDataL3<T>* compressed,
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
    double theta2 = compressed->d_model_params[partition_idx * 4 + 2];  // For POLY2
    double theta3 = compressed->d_model_params[partition_idx * 4 + 3];  // For POLY3

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
        // Compute prediction using Horner's method based on model type
        double x = static_cast<double>(local_idx);
        double predicted;
        switch (model_type) {
            case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                predicted = theta0 + x * (theta1 + x * theta2);
                break;
            case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
                break;
            default:  // LINEAR, CONSTANT, etc.
                predicted = fma(theta1, x, theta0);
                break;
        }
        // CRITICAL: Must use __double2ll_rn (banker's rounding) to match V2 partitioner
        // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
        if constexpr (std::is_unsigned<T>::value) {
            if (predicted < 0.0) predicted = 0.0;
        }
        T predicted_T = static_cast<T>(__double2ll_rn(predicted));
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
    const CompressedDataL3<T>* compressed,
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
    double theta2 = compressed->d_model_params[partition_idx * 4 + 2];  // For POLY2
    double theta3 = compressed->d_model_params[partition_idx * 4 + 3];  // For POLY3

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
        // Compute prediction using Horner's method based on model type
        double x = static_cast<double>(local_idx);
        double predicted;
        switch (model_type) {
            case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                predicted = theta0 + x * (theta1 + x * theta2);
                break;
            case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                predicted = theta0 + x * (theta1 + x * (theta2 + x * theta3));
                break;
            default:  // LINEAR, CONSTANT, etc.
                predicted = fma(theta1, x, theta0);
                break;
        }
        // CRITICAL: Must use __double2ll_rn (banker's rounding) to match V2 partitioner
        // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
        if constexpr (std::is_unsigned<T>::value) {
            if (predicted < 0.0) predicted = 0.0;
        }
        T predicted_T = static_cast<T>(__double2ll_rn(predicted));
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
    const CompressedDataL3<T>* compressed,
    int start_idx,
    int end_idx,
    T* __restrict__ d_output)
{
    __shared__ int s_partition_idx;
    __shared__ int s_start_idx;
    __shared__ int s_model_type;
    __shared__ int s_delta_bits;
    __shared__ int64_t s_bit_offset_base;
    __shared__ double s_theta0, s_theta1, s_theta2, s_theta3;

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
        s_theta2 = compressed->d_model_params[partition_idx * 4 + 2];  // For POLY2
        s_theta3 = compressed->d_model_params[partition_idx * 4 + 3];  // For POLY3
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
        // Compute prediction using Horner's method based on model type
        double x = static_cast<double>(local_idx);
        double predicted;
        switch (s_model_type) {
            case MODEL_POLYNOMIAL2:  // Horner: a0 + x*(a1 + x*a2)
                predicted = s_theta0 + x * (s_theta1 + x * s_theta2);
                break;
            case MODEL_POLYNOMIAL3:  // Horner: a0 + x*(a1 + x*(a2 + x*a3))
                predicted = s_theta0 + x * (s_theta1 + x * (s_theta2 + x * s_theta3));
                break;
            default:  // LINEAR, CONSTANT, etc.
                predicted = fma(s_theta1, x, s_theta0);
                break;
        }
        // CRITICAL: Must use __double2ll_rn (banker's rounding) to match V2 partitioner
        // CRITICAL: For unsigned types, clamp negative predictions to 0 to match encoder
        if constexpr (std::is_unsigned<T>::value) {
            if (predicted < 0.0) predicted = 0.0;
        }
        T predicted_T = static_cast<T>(__double2ll_rn(predicted));
        final_value = applyDelta(predicted_T, delta);
    }

    int output_idx = global_idx - start_idx;
    d_output[output_idx] = final_value;
}

// ============================================================================
// Host API Implementations
// ============================================================================

/**
 * Optimized random access with direct pointer parameters (Vertical-style)
 *
 * This function uses the optimized kernel that matches Vertical structure exactly:
 * - Direct array pointer parameters (no struct indirection)
 * - Branchless bit extraction
 */
template<typename T>
cudaError_t randomAccessOptimized(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
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

    // Launch optimized kernel with direct pointers
    randomAccessOptimizedKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        compressed->delta_array,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->num_partitions,
        d_indices,
        num_indices,
        d_output);

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

// Explicit instantiations for randomAccessOptimized
template cudaError_t randomAccessOptimized<uint32_t>(
    const CompressedDataL3<uint32_t>*, const int*, int, uint32_t*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessOptimized<uint64_t>(
    const CompressedDataL3<uint64_t>*, const int*, int, uint64_t*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessOptimized<int32_t>(
    const CompressedDataL3<int32_t>*, const int*, int, int32_t*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessOptimized<int64_t>(
    const CompressedDataL3<int64_t>*, const int*, int, int64_t*, RandomAccessStats*, cudaStream_t);

template<typename T>
cudaError_t randomAccessMultiple(
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

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
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

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
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessWithPartitionsKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, d_indices, d_partition_ids, num_indices, d_output);

    return cudaGetLastError();
}

template<typename T>
cudaError_t randomAccessRange(
    const CompressedDataL3<T>* compressed,
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
    const CompressedDataL3<T>* d_compressed_ptr = compressed->d_self ? compressed->d_self : compressed;

    randomAccessRangeKernel<T><<<num_blocks, threads_per_block, 0, stream>>>(
        d_compressed_ptr, start_idx, end_idx, d_output);

    return cudaGetLastError();
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// randomAccessElement (device function)
template __device__ int32_t randomAccessElement(const CompressedDataL3<int32_t>*, int);
template __device__ uint32_t randomAccessElement(const CompressedDataL3<uint32_t>*, int);
template __device__ int64_t randomAccessElement(const CompressedDataL3<int64_t>*, int);
template __device__ uint64_t randomAccessElement(const CompressedDataL3<uint64_t>*, int);

// randomAccessMultiple
template cudaError_t randomAccessMultiple<int32_t>(
    const CompressedDataL3<int32_t>*, const int*, int, int32_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<uint32_t>(
    const CompressedDataL3<uint32_t>*, const int*, int, uint32_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<int64_t>(
    const CompressedDataL3<int64_t>*, const int*, int, int64_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessMultiple<uint64_t>(
    const CompressedDataL3<uint64_t>*, const int*, int, uint64_t*,
    const RandomAccessConfig*, RandomAccessStats*, cudaStream_t);

// randomAccessBatch
template cudaError_t randomAccessBatch<int32_t>(
    const CompressedDataL3<int32_t>*, const int*, int, int32_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<uint32_t>(
    const CompressedDataL3<uint32_t>*, const int*, int, uint32_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<int64_t>(
    const CompressedDataL3<int64_t>*, const int*, int, int64_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);
template cudaError_t randomAccessBatch<uint64_t>(
    const CompressedDataL3<uint64_t>*, const int*, int, uint64_t*,
    const RandomAccessConfig&, RandomAccessStats*, cudaStream_t);

// randomAccessWithPartitions
template cudaError_t randomAccessWithPartitions<int32_t>(
    const CompressedDataL3<int32_t>*, const int*, const int*, int, int32_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<uint32_t>(
    const CompressedDataL3<uint32_t>*, const int*, const int*, int, uint32_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<int64_t>(
    const CompressedDataL3<int64_t>*, const int*, const int*, int, int64_t*, cudaStream_t);
template cudaError_t randomAccessWithPartitions<uint64_t>(
    const CompressedDataL3<uint64_t>*, const int*, const int*, int, uint64_t*, cudaStream_t);

// randomAccessRange
template cudaError_t randomAccessRange<int32_t>(
    const CompressedDataL3<int32_t>*, int, int, int32_t*, cudaStream_t);
template cudaError_t randomAccessRange<uint32_t>(
    const CompressedDataL3<uint32_t>*, int, int, uint32_t*, cudaStream_t);
template cudaError_t randomAccessRange<int64_t>(
    const CompressedDataL3<int64_t>*, int, int, int64_t*, cudaStream_t);
template cudaError_t randomAccessRange<uint64_t>(
    const CompressedDataL3<uint64_t>*, int, int, uint64_t*, cudaStream_t);
