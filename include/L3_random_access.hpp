/**
 * L3 Random Access API
 *
 * High-performance random access interface for L3 compressed data.
 * Provides multiple access patterns optimized for different use cases.
 *
 * VERSION: 1.0.0
 * DATE: 2025-10-14
 *
 * KEY FEATURES:
 * - Single-element random access
 * - Batch random access with partition grouping
 * - Range-based random access
 * - Partition metadata caching
 * - Warp-cooperative decompression
 */

#ifndef L3_RANDOM_ACCESS_HPP
#define L3_RANDOM_ACCESS_HPP

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "L3_format.hpp"

// ============================================================================
// Random Access Configuration
// ============================================================================

/**
 * Configuration for random access operations
 */
struct RandomAccessConfig {
    bool enable_partition_cache;    // Use shared memory cache for metadata
    bool enable_batch_grouping;     // Group by partition for batch access
    int cache_size;                 // Number of partitions to cache (default: 8)
    int batch_threshold;            // Min batch size for grouping (default: 64)

    RandomAccessConfig()
        : enable_partition_cache(true),
          enable_batch_grouping(true),
          cache_size(8),
          batch_threshold(64) {}
};

/**
 * Statistics for random access operations
 */
struct RandomAccessStats {
    int64_t num_accesses;           // Total random accesses
    int64_t cache_hits;             // Partition cache hits
    int64_t cache_misses;           // Partition cache misses
    double avg_search_depth;        // Average binary search depth
    double throughput_gbps;         // Throughput (GB/s)
    double time_ms;                 // Total time (milliseconds)

    RandomAccessStats()
        : num_accesses(0), cache_hits(0), cache_misses(0),
          avg_search_depth(0.0), throughput_gbps(0.0), time_ms(0.0) {}
};

// ============================================================================
// Single-Element Random Access API
// ============================================================================

/**
 * Random access a single element from compressed data
 *
 * This is a device-side function that can be called from kernels.
 *
 * PARAMETERS:
 * - compressed: Compressed data structure (device pointer)
 * - global_idx: Global element index to access
 *
 * RETURNS: Decompressed value at global_idx
 *
 * PERFORMANCE:
 * - Binary search: O(log n) where n = num_partitions
 * - Decompression: O(1) per element
 * - Suitable for: Low-volume random access, scattered patterns
 *
 * USAGE:
 *   __global__ void myKernel(CompressedDataL3<int>* data) {
 *       int idx = blockIdx.x * blockDim.x + threadIdx.x;
 *       int value = randomAccessElement(data, idx);
 *   }
 */
template<typename T>
__device__ T randomAccessElement(
    const CompressedDataL3<T>* compressed,
    int global_idx);

/**
 * Host-side wrapper: Random access multiple elements
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - d_indices: Array of global indices to access (device memory)
 * - num_indices: Number of indices
 * - d_output: Output array (device memory, pre-allocated)
 * - config: Random access configuration (optional)
 * - stats: Output statistics (optional)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * ALGORITHM:
 * 1. Launch kernel with one thread per index
 * 2. Each thread performs random access independently
 * 3. Write results to d_output
 *
 * PERFORMANCE:
 * - Best for: Small to medium batch sizes (< 100K elements)
 * - Overhead: Binary search per access
 * - Memory pattern: Non-coalesced (random)
 */
template<typename T>
cudaError_t randomAccessMultiple(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    const RandomAccessConfig* config = nullptr,
    RandomAccessStats* stats = nullptr,
    cudaStream_t stream = 0);

// ============================================================================
// Optimized Random Access API (Vertical-style)
// ============================================================================

/**
 * Optimized random access using direct pointer parameters
 *
 * This API uses the same kernel structure as Vertical random access:
 * - Direct array pointer parameters (no struct indirection)
 * - Branchless bit extraction (matches Vertical optimization)
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - d_indices: Array of global indices (device memory)
 * - num_indices: Number of indices
 * - d_output: Output array (device memory, pre-allocated)
 * - stats: Output statistics (optional)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * PERFORMANCE:
 * - Matches Vertical random access performance
 * - No struct indirection overhead
 * - Branchless bit extraction
 */
template<typename T>
cudaError_t randomAccessOptimized(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    RandomAccessStats* stats = nullptr,
    cudaStream_t stream = 0);

// ============================================================================
// Batch Random Access API (Optimized)
// ============================================================================

/**
 * Batch random access with partition grouping
 *
 * This is the RECOMMENDED API for high-volume random access.
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - d_indices: Array of global indices (device memory)
 * - num_indices: Number of indices
 * - d_output: Output array (device memory, pre-allocated)
 * - config: Random access configuration
 * - stats: Output statistics (optional)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * ALGORITHM:
 * 1. Sort indices by partition (optional, controlled by config)
 * 2. Process in batches, one batch per partition
 * 3. Cache partition metadata in shared memory
 * 4. Warp-cooperative decompression within partition
 *
 * PERFORMANCE:
 * - Best for: Large batch sizes (> 100K elements)
 * - Speedup: 1.5-3x vs simple random access
 * - Memory pattern: Improved locality after sorting
 *
 * TRADE-OFFS:
 * - Pro: Better cache locality, reduced binary searches
 * - Con: Sorting overhead (amortized over large batches)
 */
template<typename T>
cudaError_t randomAccessBatch(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    const RandomAccessConfig& config = RandomAccessConfig(),
    RandomAccessStats* stats = nullptr,
    cudaStream_t stream = 0);

/**
 * Batch random access with pre-computed partition indices
 *
 * Use this when you already know which partition each index belongs to.
 * This avoids the binary search overhead entirely.
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - d_indices: Array of global indices (device memory)
 * - d_partition_ids: Partition index for each element (device memory)
 * - num_indices: Number of indices
 * - d_output: Output array (device memory, pre-allocated)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * PERFORMANCE:
 * - Best for: Repeated access patterns with cached partition mappings
 * - Speedup: 2-4x vs simple random access (eliminates binary search)
 */
template<typename T>
cudaError_t randomAccessWithPartitions(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    const int* d_partition_ids,
    int num_indices,
    T* d_output,
    cudaStream_t stream = 0);

// ============================================================================
// Range-Based Random Access API
// ============================================================================

/**
 * Random access a range of elements
 *
 * Optimized for accessing a contiguous range within compressed data.
 * More efficient than accessing each element individually.
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - start_idx: Starting global index (inclusive)
 * - end_idx: Ending global index (exclusive)
 * - d_output: Output array (device memory, pre-allocated)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * ALGORITHM:
 * 1. Find starting and ending partitions
 * 2. For each partition in range:
 *    a. Load metadata to shared memory
 *    b. Decompress elements in range cooperatively
 * 3. Write results to d_output
 *
 * PERFORMANCE:
 * - Best for: Contiguous or semi-contiguous ranges
 * - Speedup: 5-10x vs simple random access (coalesced patterns)
 */
template<typename T>
cudaError_t randomAccessRange(
    const CompressedDataL3<T>* compressed,
    int start_idx,
    int end_idx,
    T* d_output,
    cudaStream_t stream = 0);

// ============================================================================
// Partition Lookup Utilities
// ============================================================================

/**
 * Find partition index for a given global index (device function)
 *
 * Uses binary search on partition boundaries.
 *
 * PARAMETERS:
 * - compressed: Compressed data
 * - global_idx: Global element index
 *
 * RETURNS: Partition index
 */
template<typename T>
__device__ int findPartition(
    const CompressedDataL3<T>* compressed,
    int global_idx);

/**
 * Build partition lookup table (host function)
 *
 * Pre-computes partition indices for a range of global indices.
 * Useful for repeated random access with same index set.
 *
 * PARAMETERS:
 * - compressed: Compressed data (host structure with device pointers)
 * - h_indices: Array of global indices (host memory)
 * - num_indices: Number of indices
 * - h_partition_ids: Output array (host memory, pre-allocated)
 *
 * RETURNS: Number of unique partitions accessed
 *
 * USAGE:
 *   std::vector<int> indices = {...};
 *   std::vector<int> partition_ids(indices.size());
 *   int num_partitions = buildPartitionLookup(compressed,
 *       indices.data(), indices.size(), partition_ids.data());
 */
template<typename T>
int buildPartitionLookupTable(
    const CompressedDataL3<T>* compressed,
    const int* h_indices,
    int num_indices,
    int* h_partition_ids);

// ============================================================================
// Advanced Features
// ============================================================================

/**
 * Random access with predicate filtering
 *
 * Combines random access with on-the-fly filtering.
 * Only accesses and outputs elements that match a predicate.
 *
 * PARAMETERS:
 * - compressed: Compressed data (device memory)
 * - d_indices: Array of candidate indices (device memory)
 * - num_indices: Number of candidate indices
 * - predicate: Device function pointer for filtering
 * - d_output: Output array (device memory, pre-allocated)
 * - d_output_count: Number of elements written (device memory)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * EXAMPLE:
 *   __device__ bool myFilter(int value) { return value > 100; }
 *   randomAccessWithPredicate(compressed, indices, n, myFilter, output, count);
 */
template<typename T>
cudaError_t randomAccessWithPredicate(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    bool (*predicate)(T),
    T* d_output,
    int* d_output_count,
    cudaStream_t stream = 0);

/**
 * Two-stage random access for multi-column queries
 *
 * Optimized for queries that filter on one column and access others.
 *
 * PARAMETERS:
 * - filter_column: Compressed column for filtering (device memory)
 * - filter_min, filter_max: Filter range (inclusive)
 * - access_columns: Array of additional columns to access (device memory)
 * - num_access_columns: Number of additional columns
 * - num_total_elements: Total elements in each column
 * - d_output_indices: Output candidate indices (device memory, pre-allocated)
 * - d_output_values: Output values [num_candidates * num_columns] (device memory)
 * - d_num_candidates: Number of candidates found (device memory)
 * - stream: CUDA stream
 *
 * RETURNS: cudaError_t
 *
 * ALGORITHM:
 * - Stage 1: Scan filter_column, generate candidate indices
 * - Stage 2: Random access other columns only for candidates
 *
 * This is the pattern used in SSB queries (see q12_L3_ra.cu example).
 */
template<typename T>
cudaError_t randomAccessTwoStage(
    const CompressedDataL3<T>* filter_column,
    T filter_min,
    T filter_max,
    const CompressedDataL3<T>** access_columns,
    int num_access_columns,
    int num_total_elements,
    int* d_output_indices,
    T* d_output_values,
    int* d_num_candidates,
    cudaStream_t stream = 0);

// ============================================================================
// Benchmarking and Profiling
// ============================================================================

/**
 * Benchmark random access performance
 *
 * PARAMETERS:
 * - compressed: Compressed data
 * - num_accesses: Number of random accesses to perform
 * - access_pattern: "random", "sequential", "strided", "clustered"
 * - warmup_iters: Number of warmup iterations
 * - timed_iters: Number of timed iterations
 * - stats: Output statistics
 *
 * RETURNS: cudaError_t
 */
template<typename T>
cudaError_t benchmarkRandomAccess(
    const CompressedDataL3<T>* compressed,
    int num_accesses,
    const char* access_pattern,
    int warmup_iters,
    int timed_iters,
    RandomAccessStats* stats);

/**
 * Profile random access to identify bottlenecks
 *
 * Breaks down time into:
 * - Binary search time
 * - Metadata load time
 * - Delta extraction time
 * - Model evaluation time
 *
 * PARAMETERS:
 * - compressed: Compressed data
 * - d_indices: Array of indices to access
 * - num_indices: Number of indices
 * - d_profile_output: Output profile data (device memory)
 * - stream: CUDA stream
 */
template<typename T>
cudaError_t profileRandomAccess(
    const CompressedDataL3<T>* compressed,
    const int* d_indices,
    int num_indices,
    float* d_profile_output,
    cudaStream_t stream = 0);

#endif // L3_RANDOM_ACCESS_HPP
