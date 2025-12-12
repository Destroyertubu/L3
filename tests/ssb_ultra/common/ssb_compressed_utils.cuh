/**
 * @file ssb_compressed_utils.cuh
 * @brief L3 Compressed Data Access Utilities for SSB Queries
 *
 * This file provides high-level wrappers for L3 compression APIs:
 * - Full decompression (for decompress_first approach)
 * - Partition-level decompression (for fused_query approach)
 * - Predicate pushdown using partition min/max bounds (for predicate_pushdown approach)
 * - Batch random access
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <iostream>

// L3 compression headers
#include "L3_format.hpp"
#include "L3_codec.hpp"
#include "L3_random_access.hpp"
#include "L3_Vertical_api.hpp"
#include "L3_Vertical_format.hpp"

// ============================================================================
// Forward Declarations for L3 Decompression Kernels (global namespace)
// ============================================================================

// From decoder_warp_opt.cu - declaration only (defined in .cu file)
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream);

// From decoder_partition_opt.cu - partition-aware decompression
template<typename T>
void launchDecompressPartitions(
    const CompressedDataL3<T>* compressed,
    const int* d_partition_ids,
    int num_partitions,
    const int* d_output_offsets,
    T* d_output,
    cudaStream_t stream);

// From decoder_Vertical_opt.cu - Vertical branchless decoder (L3-compatible)
template<typename T>
void launchDecompressVerticalL3(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream);

namespace ssb {

// ============================================================================
// Utility Kernels (defined first for use in accessor)
// ============================================================================

/**
 * @brief Kernel to fill an array with sequential indices
 */
__global__ inline void fillIndicesKernel(int* indices, int start, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        indices[idx] = start + idx;
    }
}

/**
 * @brief Kernel to filter values and collect passing indices
 */
template<typename T>
__global__ void filterRangeKernel(
    const T* values,
    int start_idx,
    int count,
    T min_val, T max_val,
    int* candidate_indices,
    int* num_candidates)
{
    int local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_idx >= count) return;

    T value = values[local_idx];

    // Check if value is in range
    if (value >= min_val && value <= max_val) {
        // Atomically add to candidate list
        int pos = atomicAdd(num_candidates, 1);
        candidate_indices[pos] = start_idx + local_idx;
    }
}

// ============================================================================
// Compressed Column Accessor
// ============================================================================

/**
 * @brief High-level accessor for L3 compressed columns
 *
 * Provides simplified interface for SSB query operations on compressed data.
 * Supports three access patterns:
 * 1. Full decompression - decompress entire column
 * 2. Partition decompression - decompress specific partitions
 * 3. Random access - access individual elements by index
 */
template<typename T>
class CompressedColumnAccessor {
public:
    // Pointer to compressed data (owned by caller)
    const CompressedDataL3<T>* compressed_;

    // Cached metadata on host
    int num_partitions_;
    int total_elements_;

    // Host copies of partition bounds (for predicate pushdown)
    std::vector<int32_t> h_start_indices_;
    std::vector<int32_t> h_end_indices_;
    std::vector<T> h_partition_min_;
    std::vector<T> h_partition_max_;
    bool bounds_cached_;

public:
    CompressedColumnAccessor()
        : compressed_(nullptr), num_partitions_(0), total_elements_(0), bounds_cached_(false) {}

    explicit CompressedColumnAccessor(const CompressedDataL3<T>* compressed)
        : compressed_(compressed), bounds_cached_(false) {
        if (compressed_) {
            num_partitions_ = compressed_->num_partitions;
            total_elements_ = compressed_->total_values;
        } else {
            num_partitions_ = 0;
            total_elements_ = 0;
        }
    }

    /**
     * @brief Decompress entire column to GPU memory
     *
     * Uses high-performance warp-optimized decoder (300-500 GB/s target)
     *
     * @param d_output Pre-allocated device output array (size: total_elements_)
     * @param stream CUDA stream
     */
    void decompressAll(T* d_output, cudaStream_t stream = 0) const {
        if (!compressed_ || total_elements_ == 0) return;

        // Warp-optimized decoder with cp.async double-buffering (fastest)
        launchDecompressWarpOpt(compressed_, d_output, stream);
        // Alternative: launchDecompressVerticalL3(compressed_, d_output, stream);
    }

    /**
     * @brief Decompress a single partition
     *
     * @param partition_id Partition index
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     * @return Number of elements decompressed
     */
    int decompressPartition(int partition_id, T* d_output, cudaStream_t stream = 0) const {
        if (!compressed_ || partition_id < 0 || partition_id >= num_partitions_) {
            return 0;
        }

        // Get partition bounds
        int32_t start_idx, end_idx;
        cudaMemcpyAsync(&start_idx, compressed_->d_start_indices + partition_id,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&end_idx, compressed_->d_end_indices + partition_id,
                        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int partition_size = end_idx - start_idx;

        // Generate indices for this partition
        int* d_indices;
        cudaMalloc(&d_indices, partition_size * sizeof(int));

        // Launch kernel to fill indices: start_idx, start_idx+1, ..., end_idx-1
        int threads = 256;
        int blocks = (partition_size + threads - 1) / threads;
        fillIndicesKernel<<<blocks, threads, 0, stream>>>(d_indices, start_idx, partition_size);

        // Random access these indices
        RandomAccessConfig config;
        ::randomAccessBatch(compressed_, d_indices, partition_size, d_output, config, nullptr, stream);

        cudaFree(d_indices);
        return partition_size;
    }

    /**
     * @brief Decompress a range of elements
     *
     * @param start_idx Start index (inclusive)
     * @param end_idx End index (exclusive)
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     */
    void decompressRange(int start_idx, int end_idx, T* d_output, cudaStream_t stream = 0) const {
        if (!compressed_ || start_idx >= end_idx) return;

        int range_size = end_idx - start_idx;

        // Generate indices for this range
        int* d_indices;
        cudaMalloc(&d_indices, range_size * sizeof(int));

        int threads = 256;
        int blocks = (range_size + threads - 1) / threads;
        fillIndicesKernel<<<blocks, threads, 0, stream>>>(d_indices, start_idx, range_size);

        // Random access these indices
        RandomAccessConfig config;
        ::randomAccessBatch(compressed_, d_indices, range_size, d_output, config, nullptr, stream);

        cudaFree(d_indices);
    }

    /**
     * @brief Batch random access by indices
     *
     * @param d_indices Device array of indices to access
     * @param num_indices Number of indices
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     */
    void randomAccessBatchIndices(const int* d_indices, int num_indices, T* d_output,
                           cudaStream_t stream = 0) const {
        if (!compressed_ || num_indices == 0) return;

        RandomAccessConfig config;
        config.enable_batch_grouping = true;  // Optimize for batch access

        ::randomAccessBatch(compressed_, d_indices, num_indices, d_output, config, nullptr, stream);
    }

    /**
     * @brief Cache partition bounds for predicate pushdown
     *
     * Copies partition min/max values to host memory for fast filtering
     */
    void cachePartitionBounds() {
        if (!compressed_ || num_partitions_ == 0) return;
        if (bounds_cached_) return;  // Already cached

        h_start_indices_.resize(num_partitions_);
        h_end_indices_.resize(num_partitions_);
        h_partition_min_.resize(num_partitions_);
        h_partition_max_.resize(num_partitions_);

        cudaMemcpy(h_start_indices_.data(), compressed_->d_start_indices,
                   num_partitions_ * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end_indices_.data(), compressed_->d_end_indices,
                   num_partitions_ * sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (compressed_->d_partition_min_values && compressed_->d_partition_max_values) {
            cudaMemcpy(h_partition_min_.data(), compressed_->d_partition_min_values,
                       num_partitions_ * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_partition_max_.data(), compressed_->d_partition_max_values,
                       num_partitions_ * sizeof(T), cudaMemcpyDeviceToHost);
        }

        bounds_cached_ = true;
    }

    /**
     * @brief Get partitions that may contain values in range [min_val, max_val]
     *
     * This is the core function for predicate pushdown optimization.
     * Uses partition min/max metadata to skip partitions that definitely
     * don't contain matching values.
     *
     * @param min_val Minimum value in filter range (inclusive)
     * @param max_val Maximum value in filter range (inclusive)
     * @param partition_ids Output: indices of candidate partitions
     */
    void getCandidatePartitions(T min_val, T max_val,
                                std::vector<int>& partition_ids) const {
        partition_ids.clear();

        if (!bounds_cached_) {
            // No bounds cached, return all partitions
            for (int i = 0; i < num_partitions_; ++i) {
                partition_ids.push_back(i);
            }
            return;
        }

        // Check each partition's bounds against filter range
        for (int i = 0; i < num_partitions_; ++i) {
            T part_min = h_partition_min_[i];
            T part_max = h_partition_max_[i];

            // Check for overlap: partition range [part_min, part_max] intersects [min_val, max_val]
            // Overlap if: part_max >= min_val AND part_min <= max_val
            if (part_max >= min_val && part_min <= max_val) {
                partition_ids.push_back(i);
            }
        }
    }

    /**
     * @brief Get partitions where a specific value may exist
     *
     * @param value Value to search for
     * @param partition_ids Output: indices of candidate partitions
     */
    void getCandidatePartitionsForValue(T value, std::vector<int>& partition_ids) const {
        getCandidatePartitions(value, value, partition_ids);
    }

    /**
     * @brief Get total element count for a list of partitions
     *
     * Useful for pre-allocating output buffers before decompression.
     *
     * @param partition_ids List of partition IDs
     * @return Total number of elements across all specified partitions
     */
    int getTotalElementsForPartitions(const std::vector<int>& partition_ids) const {
        if (!bounds_cached_) return 0;

        int total = 0;
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                total += h_end_indices_[pid] - h_start_indices_[pid];
            }
        }
        return total;
    }

    /**
     * @brief Directly decompress multiple partitions using optimized kernel (NO binary search!)
     *
     * This is the key optimization for predicate pushdown queries.
     * Instead of using generic random access with O(log N) binary search per element,
     * this method directly invokes the warp-optimized decompression kernel
     * for the specified partitions.
     *
     * Performance: ~300-500 GB/s vs ~1-5 GB/s for random access
     *
     * @param partition_ids List of partition IDs to decompress
     * @param d_output Pre-allocated device output array (size: getTotalElementsForPartitions)
     * @param stream CUDA stream
     * @return Total elements decompressed
     */
    int decompressPartitionsDirect(const std::vector<int>& partition_ids,
                                   T* d_output,
                                   cudaStream_t stream = 0) const {
        if (!compressed_ || partition_ids.empty()) return 0;

        // Ensure bounds are cached
        if (!bounds_cached_) {
            // Const-cast workaround: cache bounds
            const_cast<CompressedColumnAccessor*>(this)->cachePartitionBounds();
        }

        int num_partitions = partition_ids.size();

        // Calculate output offsets (prefix sum of partition sizes)
        std::vector<int> output_offsets(num_partitions);
        int offset = 0;
        for (int i = 0; i < num_partitions; ++i) {
            output_offsets[i] = offset;
            int pid = partition_ids[i];
            if (pid >= 0 && pid < num_partitions_) {
                offset += h_end_indices_[pid] - h_start_indices_[pid];
            }
        }
        int total_elements = offset;

        if (total_elements == 0) return 0;

        // Allocate device arrays for partition IDs and offsets
        int* d_partition_ids;
        int* d_output_offsets;
        cudaMalloc(&d_partition_ids, num_partitions * sizeof(int));
        cudaMalloc(&d_output_offsets, num_partitions * sizeof(int));

        // Copy to device
        cudaMemcpyAsync(d_partition_ids, partition_ids.data(),
                        num_partitions * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_output_offsets, output_offsets.data(),
                        num_partitions * sizeof(int), cudaMemcpyHostToDevice, stream);

        // Launch optimized partition decompression kernel
        ::launchDecompressPartitions(compressed_, d_partition_ids, num_partitions,
                                     d_output_offsets, d_output, stream);

        // Cleanup
        cudaStreamSynchronize(stream);
        cudaFree(d_partition_ids);
        cudaFree(d_output_offsets);

        return total_elements;
    }

    /**
     * @brief Get partition sizes for a list of partition IDs
     *
     * @param partition_ids List of partition IDs
     * @return Vector of partition sizes (same order as input)
     */
    std::vector<int> getPartitionSizes(const std::vector<int>& partition_ids) const {
        std::vector<int> sizes;
        if (!bounds_cached_) return sizes;

        sizes.reserve(partition_ids.size());
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                sizes.push_back(h_end_indices_[pid] - h_start_indices_[pid]);
            } else {
                sizes.push_back(0);
            }
        }
        return sizes;
    }

    /**
     * @brief Get start index for a partition
     */
    int getPartitionStartIndex(int partition_id) const {
        if (!bounds_cached_ || partition_id < 0 || partition_id >= num_partitions_) {
            return -1;
        }
        return h_start_indices_[partition_id];
    }

    /**
     * @brief Get partition index containing a global element index
     *
     * Uses binary search on partition boundaries.
     *
     * @param global_idx Global element index
     * @return Partition index, or -1 if not found
     */
    int getPartitionForIndex(int global_idx) const {
        if (!bounds_cached_ || global_idx < 0 || global_idx >= total_elements_) {
            return -1;
        }

        // Binary search
        int lo = 0, hi = num_partitions_ - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (global_idx < h_start_indices_[mid]) {
                hi = mid - 1;
            } else if (global_idx >= h_end_indices_[mid]) {
                lo = mid + 1;
            } else {
                return mid;  // Found: start <= idx < end
            }
        }
        return -1;
    }

    // Getters
    int getNumPartitions() const { return num_partitions_; }
    int getTotalElements() const { return total_elements_; }
    const CompressedDataL3<T>* getCompressed() const { return compressed_; }
};

// ============================================================================
// Vertical Compressed Column Accessor (Interleaved Format)
// ============================================================================

/**
 * @brief High-level accessor for Vertical compressed columns (interleaved format)
 *
 * Uses the interleaved decoder for ~1.4x faster decompression than L3.
 * Provides simplified interface for SSB query operations on compressed data.
 */
template<typename T>
class CompressedColumnAccessorVertical {
public:
    // Pointer to compressed data (owned by caller)
    const CompressedDataVertical<T>* compressed_;

    // Cached metadata on host
    int num_partitions_;
    int total_elements_;

    // Host copies of partition bounds (for predicate pushdown)
    std::vector<int32_t> h_start_indices_;
    std::vector<int32_t> h_end_indices_;
    std::vector<T> h_partition_min_;
    std::vector<T> h_partition_max_;
    bool bounds_cached_;       // Whether start/end indices are cached
    bool minmax_available_;    // Whether min/max are available for filtering

    // Cached fully decompressed data (for efficient range access)
    mutable T* d_decompressed_cache_;
    mutable bool cache_valid_;

public:
    CompressedColumnAccessorVertical()
        : compressed_(nullptr), num_partitions_(0), total_elements_(0),
          bounds_cached_(false), minmax_available_(false),
          d_decompressed_cache_(nullptr), cache_valid_(false) {}

    explicit CompressedColumnAccessorVertical(const CompressedDataVertical<T>* compressed)
        : compressed_(compressed), bounds_cached_(false), minmax_available_(false),
          d_decompressed_cache_(nullptr), cache_valid_(false) {
        if (compressed_) {
            num_partitions_ = compressed_->num_partitions;
            total_elements_ = compressed_->total_values;
        } else {
            num_partitions_ = 0;
            total_elements_ = 0;
        }
    }

    ~CompressedColumnAccessorVertical() {
        if (d_decompressed_cache_) {
            cudaFree(d_decompressed_cache_);
            d_decompressed_cache_ = nullptr;
        }
    }

    // Delete copy to prevent double-free
    CompressedColumnAccessorVertical(const CompressedColumnAccessorVertical&) = delete;
    CompressedColumnAccessorVertical& operator=(const CompressedColumnAccessorVertical&) = delete;

    // Allow move
    CompressedColumnAccessorVertical(CompressedColumnAccessorVertical&& other) noexcept
        : compressed_(other.compressed_), num_partitions_(other.num_partitions_),
          total_elements_(other.total_elements_),
          h_start_indices_(std::move(other.h_start_indices_)),
          h_end_indices_(std::move(other.h_end_indices_)),
          h_partition_min_(std::move(other.h_partition_min_)),
          h_partition_max_(std::move(other.h_partition_max_)),
          bounds_cached_(other.bounds_cached_), minmax_available_(other.minmax_available_),
          d_decompressed_cache_(other.d_decompressed_cache_), cache_valid_(other.cache_valid_) {
        other.d_decompressed_cache_ = nullptr;
        other.cache_valid_ = false;
    }

    CompressedColumnAccessorVertical& operator=(CompressedColumnAccessorVertical&& other) noexcept {
        if (this != &other) {
            if (d_decompressed_cache_) cudaFree(d_decompressed_cache_);
            compressed_ = other.compressed_;
            num_partitions_ = other.num_partitions_;
            total_elements_ = other.total_elements_;
            h_start_indices_ = std::move(other.h_start_indices_);
            h_end_indices_ = std::move(other.h_end_indices_);
            h_partition_min_ = std::move(other.h_partition_min_);
            h_partition_max_ = std::move(other.h_partition_max_);
            bounds_cached_ = other.bounds_cached_;
            minmax_available_ = other.minmax_available_;
            d_decompressed_cache_ = other.d_decompressed_cache_;
            cache_valid_ = other.cache_valid_;
            other.d_decompressed_cache_ = nullptr;
            other.cache_valid_ = false;
        }
        return *this;
    }

    /**
     * @brief Ensure decompressed cache is valid
     */
    void ensureDecompressedCache(cudaStream_t stream = 0) const {
        if (cache_valid_) return;
        if (!compressed_ || total_elements_ == 0) return;

        if (!d_decompressed_cache_) {
            cudaMalloc(&d_decompressed_cache_, total_elements_ * sizeof(T));
        }
        decompressAll(d_decompressed_cache_, stream);
        cudaStreamSynchronize(stream);
        cache_valid_ = true;
    }

    /**
     * @brief Decompress entire column to GPU memory using Vertical interleaved decoder
     *
     * Uses high-performance interleaved decoder (~1400 GB/s on H20)
     *
     * @param d_output Pre-allocated device output array (size: total_elements_)
     * @param stream CUDA stream
     */
    void decompressAll(T* d_output, cudaStream_t stream = 0) const {
        if (!compressed_ || total_elements_ == 0) return;

        // Use Vertical interleaved decoder for maximum throughput
        Vertical_decoder::decompressAll(*compressed_, d_output, DecompressMode::INTERLEAVED, stream);
    }

    /**
     * @brief Batch random access by indices
     *
     * @param d_indices Device array of indices to access
     * @param num_indices Number of indices
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     */
    void randomAccessBatchIndices(const int* d_indices, int num_indices, T* d_output,
                           cudaStream_t stream = 0) const {
        if (!compressed_ || num_indices == 0) return;

        // Use Vertical random access decoder
        Vertical_decoder::decompressIndices(*compressed_, d_indices, num_indices, d_output, stream);
    }

    /**
     * @brief Cache partition bounds for predicate pushdown
     *
     * Copies partition min/max values to host memory for fast filtering
     */
    void cachePartitionBounds() {
        if (!compressed_ || num_partitions_ == 0) return;
        if (bounds_cached_) return;  // Already cached

        h_start_indices_.resize(num_partitions_);
        h_end_indices_.resize(num_partitions_);
        h_partition_min_.resize(num_partitions_);
        h_partition_max_.resize(num_partitions_);

        cudaMemcpy(h_start_indices_.data(), compressed_->d_start_indices,
                   num_partitions_ * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end_indices_.data(), compressed_->d_end_indices,
                   num_partitions_ * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Start/end indices are now cached
        bounds_cached_ = true;

        // Check if min/max are available for predicate pushdown
        if (compressed_->d_partition_min_values && compressed_->d_partition_max_values) {
            cudaMemcpy(h_partition_min_.data(), compressed_->d_partition_min_values,
                       num_partitions_ * sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_partition_max_.data(), compressed_->d_partition_max_values,
                       num_partitions_ * sizeof(T), cudaMemcpyDeviceToHost);
            minmax_available_ = true;
        } else {
            // No min/max data - getCandidatePartitions will return all partitions
            minmax_available_ = false;
        }
    }

    /**
     * @brief Get partitions that may contain values in range [min_val, max_val]
     *
     * This is the core function for predicate pushdown optimization.
     *
     * @param min_val Minimum value in filter range (inclusive)
     * @param max_val Maximum value in filter range (inclusive)
     * @param partition_ids Output: indices of candidate partitions
     */
    void getCandidatePartitions(T min_val, T max_val,
                                std::vector<int>& partition_ids) const {
        partition_ids.clear();

        if (!minmax_available_) {
            // No min/max data available, return all partitions (safe fallback)
            for (int i = 0; i < num_partitions_; ++i) {
                partition_ids.push_back(i);
            }
            return;
        }

        // Check each partition's bounds against filter range
        for (int i = 0; i < num_partitions_; ++i) {
            T part_min = h_partition_min_[i];
            T part_max = h_partition_max_[i];

            // Check for overlap
            if (part_max >= min_val && part_min <= max_val) {
                partition_ids.push_back(i);
            }
        }
    }

    /**
     * @brief Decompress a range of elements
     *
     * Uses caching for efficient repeated range access (e.g., chunk processing)
     *
     * @param start_idx Start index (inclusive)
     * @param end_idx End index (exclusive)
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     */
    void decompressRange(int start_idx, int end_idx, T* d_output, cudaStream_t stream = 0) const {
        if (!compressed_ || start_idx >= end_idx) return;

        int range_size = end_idx - start_idx;

        // Fast path: if range covers entire data, use decompressAll directly
        if (start_idx == 0 && end_idx >= total_elements_) {
            decompressAll(d_output, stream);
            return;
        }

        // For any significant range (> 10K elements), use cached full decompression
        // This is much faster than random access and amortizes decompression cost
        constexpr int CACHE_THRESHOLD = 10000;
        if (range_size > CACHE_THRESHOLD) {
            // Ensure we have decompressed data in cache
            ensureDecompressedCache(stream);

            // Copy the requested range from cache
            cudaMemcpyAsync(d_output, d_decompressed_cache_ + start_idx,
                           range_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
            return;
        }

        // Small ranges: use random access
        int* d_indices;
        cudaMalloc(&d_indices, range_size * sizeof(int));

        int threads = 256;
        int blocks = (range_size + threads - 1) / threads;
        fillIndicesKernel<<<blocks, threads, 0, stream>>>(d_indices, start_idx, range_size);

        Vertical_decoder::decompressIndices(*compressed_, d_indices, range_size, d_output, stream);

        cudaFree(d_indices);
    }

    /**
     * @brief Get total element count for a list of partitions
     *
     * @param partition_ids List of partition IDs
     * @return Total number of elements across all specified partitions
     */
    int getTotalElementsForPartitions(const std::vector<int>& partition_ids) const {
        if (!bounds_cached_) return 0;

        int total = 0;
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                total += h_end_indices_[pid] - h_start_indices_[pid];
            }
        }
        return total;
    }

    /**
     * @brief Get start index for a partition
     */
    int getPartitionStartIndex(int partition_id) const {
        if (!bounds_cached_ || partition_id < 0 || partition_id >= num_partitions_) {
            return -1;
        }
        return h_start_indices_[partition_id];
    }

    /**
     * @brief Get partition sizes for a list of partition IDs
     *
     * @param partition_ids List of partition IDs
     * @return Vector of partition sizes (same order as input)
     */
    std::vector<int> getPartitionSizes(const std::vector<int>& partition_ids) const {
        std::vector<int> sizes;
        if (!bounds_cached_) return sizes;

        sizes.reserve(partition_ids.size());
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                sizes.push_back(h_end_indices_[pid] - h_start_indices_[pid]);
            } else {
                sizes.push_back(0);
            }
        }
        return sizes;
    }

    /**
     * @brief Directly decompress multiple partitions (optimized for predicate pushdown)
     *
     * This decompresses only the specified partitions and packs them contiguously
     * in the output buffer.
     *
     * IMPORTANT: Different columns may have different partition structures due to
     * adaptive partitioning. If partition_ids from one column are used to access
     * another column, results may be incorrect. When in doubt, use decompressAll().
     *
     * @param partition_ids List of partition IDs to decompress
     * @param d_output Pre-allocated device output array
     * @param stream CUDA stream
     * @return Total elements decompressed
     */
    int decompressPartitionsDirect(const std::vector<int>& partition_ids,
                                   T* d_output,
                                   cudaStream_t stream = 0) const {
        if (!compressed_ || partition_ids.empty()) return 0;

        // Ensure bounds are cached (for start/end indices)
        if (!bounds_cached_) {
            const_cast<CompressedColumnAccessorVertical*>(this)->cachePartitionBounds();
        }

        // Safety check: if the number of partition IDs doesn't match this column's
        // partition count, the IDs likely came from a different column.
        // In this case, it's unsafe to use partition-based decompression because
        // partition boundaries differ across columns.
        // Just decompress everything for correctness.
        if (static_cast<int>(partition_ids.size()) != num_partitions_) {
            decompressAll(d_output, stream);
            return total_elements_;
        }

        // Fast path: if all partitions requested in order, use batch decompression
        bool all_sequential = true;
        for (size_t i = 0; i < partition_ids.size(); ++i) {
            if (partition_ids[i] != static_cast<int>(i)) {
                all_sequential = false;
                break;
            }
        }
        if (all_sequential) {
            decompressAll(d_output, stream);
            return total_elements_;
        }

        // Calculate total elements for selected partitions
        int total_elements = 0;
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                total_elements += h_end_indices_[pid] - h_start_indices_[pid];
            }
        }

        if (total_elements == 0) return 0;

        // Use cached decompressed data for efficiency
        ensureDecompressedCache(stream);

        // Copy selected partitions from cache to output
        int output_offset = 0;
        for (int pid : partition_ids) {
            if (pid >= 0 && pid < num_partitions_) {
                int start = h_start_indices_[pid];
                int size = h_end_indices_[pid] - start;
                cudaMemcpyAsync(d_output + output_offset, d_decompressed_cache_ + start,
                               size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
                output_offset += size;
            }
        }

        return total_elements;
    }

    /**
     * @brief Get partition index containing a global element index
     *
     * Uses binary search on partition boundaries.
     *
     * @param global_idx Global element index
     * @return Partition index, or -1 if not found
     */
    int getPartitionForIndex(int global_idx) const {
        if (!bounds_cached_ || global_idx < 0 || global_idx >= total_elements_) {
            return -1;
        }

        // Binary search
        int lo = 0, hi = num_partitions_ - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            if (global_idx < h_start_indices_[mid]) {
                hi = mid - 1;
            } else if (global_idx >= h_end_indices_[mid]) {
                lo = mid + 1;
            } else {
                return mid;  // Found: start <= idx < end
            }
        }
        return -1;
    }

    // Getters
    int getNumPartitions() const { return num_partitions_; }
    int getTotalElements() const { return total_elements_; }
    const CompressedDataVertical<T>* getCompressed() const { return compressed_; }
};

// ============================================================================
// Predicate Pushdown Helper Functions
// ============================================================================

/**
 * @brief Filter indices based on predicate using partition min/max pruning
 *
 * Two-stage filtering:
 * 1. Stage 1: Use partition metadata to identify candidate partitions
 * 2. Stage 2: Decompress only candidate partitions and apply exact filter
 *
 * @param accessor Column accessor
 * @param min_val Minimum filter value
 * @param max_val Maximum filter value
 * @param d_candidate_indices Output: device array of indices passing filter
 * @param d_num_candidates Output: number of candidates (device memory)
 * @param stream CUDA stream
 */
template<typename T>
void filterWithPredicatePushdown(
    CompressedColumnAccessor<T>& accessor,
    T min_val, T max_val,
    int* d_candidate_indices,
    int* d_num_candidates,
    cudaStream_t stream = 0)
{
    // Ensure bounds are cached
    accessor.cachePartitionBounds();

    // Stage 1: Get candidate partitions using metadata
    std::vector<int> candidate_partitions;
    accessor.getCandidatePartitions(min_val, max_val, candidate_partitions);

    if (candidate_partitions.empty()) {
        cudaMemsetAsync(d_num_candidates, 0, sizeof(int), stream);
        return;
    }

    // Initialize count to 0
    cudaMemsetAsync(d_num_candidates, 0, sizeof(int), stream);

    // Calculate total elements in candidate partitions
    int total_candidate_elements = 0;
    for (int pid : candidate_partitions) {
        total_candidate_elements += accessor.h_end_indices_[pid] - accessor.h_start_indices_[pid];
    }

    // Allocate temporary buffer for decompressed values
    T* d_temp_values;
    cudaMalloc(&d_temp_values, total_candidate_elements * sizeof(T));

    // Stage 2: Decompress candidate partitions and filter
    int offset = 0;
    for (int pid : candidate_partitions) {
        int start = accessor.h_start_indices_[pid];
        int size = accessor.h_end_indices_[pid] - start;

        // Decompress this partition's range
        accessor.decompressRange(start, start + size, d_temp_values + offset, stream);

        // Apply exact filter and collect passing indices
        int threads = 256;
        int blocks = (size + threads - 1) / threads;

        filterRangeKernel<T><<<blocks, threads, 0, stream>>>(
            d_temp_values + offset, start, size,
            min_val, max_val,
            d_candidate_indices, d_num_candidates
        );

        offset += size;
    }

    cudaFree(d_temp_values);
}

// ============================================================================
// Multi-Column Accessor for Join Queries
// ============================================================================

/**
 * @brief Accessor for multiple compressed columns (for join operations)
 *
 * Useful for queries that need to access multiple LINEORDER columns
 * that have the same partitioning scheme.
 */
template<typename T>
class MultiColumnAccessor {
public:
    std::vector<CompressedColumnAccessor<T>> columns_;
    int num_columns_;
    int total_elements_;

    MultiColumnAccessor() : num_columns_(0), total_elements_(0) {}

    void addColumn(const CompressedDataL3<T>* compressed) {
        columns_.emplace_back(compressed);
        num_columns_++;
        if (total_elements_ == 0 && compressed) {
            total_elements_ = compressed->total_values;
        }
    }

    /**
     * @brief Decompress all columns to GPU memory
     *
     * @param d_outputs Array of device output pointers (one per column)
     * @param stream CUDA stream
     */
    void decompressAll(T** d_outputs, cudaStream_t stream = 0) const {
        for (int i = 0; i < num_columns_; ++i) {
            columns_[i].decompressAll(d_outputs[i], stream);
        }
    }

    /**
     * @brief Random access multiple columns at same indices
     *
     * @param d_indices Device array of indices
     * @param num_indices Number of indices
     * @param d_outputs Array of device output pointers (one per column)
     * @param stream CUDA stream
     */
    void randomAccessBatchIndices(const int* d_indices, int num_indices,
                           T** d_outputs, cudaStream_t stream = 0) const {
        for (int i = 0; i < num_columns_; ++i) {
            columns_[i].randomAccessBatchIndices(d_indices, num_indices, d_outputs[i], stream);
        }
    }

    /**
     * @brief Cache partition bounds for all columns
     */
    void cacheAllBounds() {
        for (auto& col : columns_) {
            col.cachePartitionBounds();
        }
    }

    const CompressedColumnAccessor<T>& operator[](int idx) const { return columns_[idx]; }
    CompressedColumnAccessor<T>& operator[](int idx) { return columns_[idx]; }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Print compression statistics for a column
 */
template<typename T>
void printCompressionStats(const CompressedDataL3<T>* compressed, const char* name) {
    if (!compressed) return;

    size_t original_bytes = compressed->total_values * sizeof(T);
    size_t compressed_bytes = compressed->delta_array_words * sizeof(uint32_t);

    // Add metadata overhead
    compressed_bytes += compressed->num_partitions * (
        sizeof(int32_t) * 2 +   // start/end indices
        sizeof(int32_t) * 2 +   // model_type, delta_bits
        sizeof(double) * 4 +    // model_params
        sizeof(int64_t) * 2     // bit_offset, error_bound
    );

    double ratio = (double)original_bytes / compressed_bytes;

    std::cout << name << ": "
              << compressed->total_values << " elements, "
              << compressed->num_partitions << " partitions, "
              << "ratio: " << ratio << "x"
              << std::endl;
}

/**
 * @brief Kernel to copy selected elements based on flags
 */
template<typename T>
__global__ void copySelectedKernel(
    const T* input,
    const int* selection_flags,
    int count,
    T* output,
    int* output_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (selection_flags[idx]) {
        int pos = atomicAdd(output_count, 1);
        output[pos] = input[idx];
    }
}

/**
 * @brief Kernel to scatter decompressed values to output based on indices
 */
template<typename T>
__global__ void scatterKernel(
    const T* values,
    const int* indices,
    int count,
    T* output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    output[indices[idx]] = values[idx];
}

}  // namespace ssb
