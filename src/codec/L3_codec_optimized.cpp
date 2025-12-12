/**
 * L3 Optimized Codec Implementation
 * Using optimized GPU kernels for better compression performance
 */

#include "L3_codec.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Forward declarations of optimized kernels
template<typename T>
void launchModelFittingOptimized(
    const T* d_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream = 0);

void launchSetBitOffsetsOptimized(
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_delta_bits,
    int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    cudaStream_t stream = 0);

template<typename T>
void launchDeltaPackingOptimized(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* delta_array,
    int total_elements,
    cudaStream_t stream = 0);

// Warp-optimized decoder (for decompression)
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream);

// ============================================================================
// Adaptive Partitioning
// ============================================================================

template<typename T>
int createAdaptivePartitions(
    const T* d_data,
    int num_elements,
    int target_partition_size,
    int32_t* d_start_indices,
    int32_t* d_end_indices)
{
    // For now, use fixed-size partitions with optimized size
    // Future: implement variance-based adaptive partitioning
    int partition_size = target_partition_size;

    // Align partition size to warp boundaries for better performance
    partition_size = ((partition_size + 31) / 32) * 32;

    int num_partitions = (num_elements + partition_size - 1) / partition_size;

    std::vector<int32_t> h_start(num_partitions);
    std::vector<int32_t> h_end(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        h_start[i] = i * partition_size;
        h_end[i] = std::min((i + 1) * partition_size, num_elements);
    }

    CUDA_CHECK(cudaMemcpy(d_start_indices, h_start.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end_indices, h_end.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));

    return num_partitions;
}

// ============================================================================
// Optimized Compression Implementation
// ============================================================================

template<typename T>
CompressedDataL3<T>* compressDataOptimized(
    const T* h_data,
    int num_elements,
    int partition_size,
    CompressionStats* stats)
{
    // Create CUDA streams for overlapping operations
    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream1));

    // Allocate compressed data structure
    CompressedDataL3<T>* compressed = new CompressedDataL3<T>();
    compressed->total_values = num_elements;

    // Asynchronously upload data to GPU
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(T)));
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, num_elements * sizeof(T),
                               cudaMemcpyHostToDevice, stream1));

    // Compute number of partitions (optimize for warp alignment)
    int aligned_partition_size = ((partition_size + 31) / 32) * 32;
    int num_partitions = (num_elements + aligned_partition_size - 1) / aligned_partition_size;
    compressed->num_partitions = num_partitions;

    // Allocate metadata arrays (use stream2 for parallel allocation)
    CUDA_CHECK(cudaMalloc(&compressed->d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_error_bounds, num_partitions * sizeof(int64_t)));

    // Create adaptive partitions
    createAdaptivePartitions(d_data, num_elements, aligned_partition_size,
                            compressed->d_start_indices,
                            compressed->d_end_indices);

    // Allocate total_bits counter on device
    int64_t* d_total_bits;
    CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream1));

    // Wait for data upload to complete
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Launch optimized model fitting kernel
    launchModelFittingOptimized<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_error_bounds,
        num_partitions,
        d_total_bits,
        stream1
    );

    // Get total bits used
    int64_t total_bits;
    CUDA_CHECK(cudaMemcpyAsync(&total_bits, d_total_bits, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Compute bit offsets using optimized kernel
    launchSetBitOffsetsOptimized(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions,
        stream1
    );

    // Allocate delta array
    int64_t delta_array_words = (total_bits + 31) / 32;
    compressed->delta_array_words = delta_array_words;

    CUDA_CHECK(cudaMalloc(&compressed->delta_array, delta_array_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(compressed->delta_array, 0,
                              delta_array_words * sizeof(uint32_t), stream1));

    // Launch optimized delta packing kernel
    launchDeltaPackingOptimized<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions,
        compressed->delta_array,
        num_elements,
        stream1
    );

    // Record timing
    CUDA_CHECK(cudaEventRecord(stop, stream1));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total_bits));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));

    // Fill statistics
    if (stats) {
        stats->original_bytes = num_elements * sizeof(T);
        stats->compressed_bytes = num_partitions * (sizeof(int32_t) * 2 + sizeof(double) * 4 +
                                                    sizeof(int32_t) + sizeof(int64_t) * 2) +
                                 delta_array_words * sizeof(uint32_t);
        stats->compression_ratio = static_cast<double>(stats->original_bytes) / stats->compressed_bytes;
        stats->compression_time_ms = elapsed_ms;
        stats->compression_throughput_gbps = (stats->original_bytes / 1e9) / (elapsed_ms / 1000.0);
        stats->num_partitions = num_partitions;
        stats->total_bits_used = total_bits;

        // Compute average delta bits (WEIGHTED by partition length)
        std::vector<int32_t> h_delta_bits(num_partitions);
        std::vector<int32_t> h_start_indices(num_partitions);
        std::vector<int32_t> h_end_indices(num_partitions);

        CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_start_indices.data(), compressed->d_start_indices,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end_indices.data(), compressed->d_end_indices,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

        // Calculate weighted average: sum(delta_bits[p] * part_len[p]) / sum(part_len[p])
        int64_t sum_weighted_bits = 0;
        int64_t sum_partition_lengths = 0;

        for (int p = 0; p < num_partitions; p++) {
            int64_t part_len = h_end_indices[p] - h_start_indices[p];
            sum_weighted_bits += static_cast<int64_t>(h_delta_bits[p]) * part_len;
            sum_partition_lengths += part_len;
        }

        stats->avg_delta_bits = sum_partition_lengths > 0 ?
            static_cast<int>((sum_weighted_bits + sum_partition_lengths / 2) / sum_partition_lengths) : 0;
    }

    return compressed;
}

template<typename T>
CompressedDataL3<T>* compressDataOptimized(
    const std::vector<T>& h_data,
    int partition_size,
    CompressionStats* stats)
{
    return compressDataOptimized(h_data.data(), h_data.size(), partition_size, stats);
}

// ============================================================================
// Free Compressed Data (same as original)
// ============================================================================

template<typename T>
void freeCompressedDataOptimized(CompressedDataL3<T>* compressed)
{
    if (!compressed) return;

    if (compressed->d_start_indices) cudaFree(compressed->d_start_indices);
    if (compressed->d_end_indices) cudaFree(compressed->d_end_indices);
    if (compressed->d_model_types) cudaFree(compressed->d_model_types);
    if (compressed->d_model_params) cudaFree(compressed->d_model_params);
    if (compressed->d_delta_bits) cudaFree(compressed->d_delta_bits);
    if (compressed->d_delta_array_bit_offsets) cudaFree(compressed->d_delta_array_bit_offsets);
    if (compressed->d_error_bounds) cudaFree(compressed->d_error_bounds);
    if (compressed->delta_array) cudaFree(compressed->delta_array);
    if (compressed->d_plain_deltas) cudaFree(compressed->d_plain_deltas);

    delete compressed;
}

// ============================================================================
// Decompression (using warp-optimized decoder)
// ============================================================================

template<typename T>
int decompressDataOptimized(
    const CompressedDataL3<T>* compressed,
    T* h_output,
    int output_capacity,
    DecompressionStats* stats)
{
    if (!compressed || !h_output) return 0;
    if (output_capacity < compressed->total_values) return 0;

    // Allocate device output
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, compressed->total_values * sizeof(T)));

    // Create CUDA events for kernel-only timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start IMMEDIATELY before kernel launch
    CUDA_CHECK(cudaEventRecord(start));

    // Launch decompression kernel (from decoder_warp_opt.cu)
    launchDecompressWarpOpt(compressed, d_output, 0);

    // Record stop IMMEDIATELY after kernel launch
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure KERNEL-ONLY execution time
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Download result
    CUDA_CHECK(cudaMemcpy(h_output, d_output, compressed->total_values * sizeof(T),
                          cudaMemcpyDeviceToHost));

    // Fill statistics
    if (stats) {
        stats->decompressed_bytes = compressed->total_values * sizeof(T);
        stats->compressed_bytes = compressed->delta_array_words * sizeof(uint32_t);
        stats->decompression_time_ms = elapsed_ms;
        stats->decompression_throughput_gbps = (stats->decompressed_bytes / 1e9) / (elapsed_ms / 1000.0);
        stats->total_elements = compressed->total_values;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return compressed->total_values;
}

template<typename T>
int decompressDataOptimized(
    const CompressedDataL3<T>* compressed,
    std::vector<T>& h_output,
    DecompressionStats* stats)
{
    h_output.resize(compressed->total_values);
    return decompressDataOptimized(compressed, h_output.data(), h_output.size(), stats);
}

// ============================================================================
// Template Instantiations
// ============================================================================

template CompressedDataL3<int32_t>* compressDataOptimized(const int32_t*, int, int, CompressionStats*);
template CompressedDataL3<uint32_t>* compressDataOptimized(const uint32_t*, int, int, CompressionStats*);
template CompressedDataL3<int64_t>* compressDataOptimized(const int64_t*, int, int, CompressionStats*);
template CompressedDataL3<uint64_t>* compressDataOptimized(const uint64_t*, int, int, CompressionStats*);

template CompressedDataL3<int32_t>* compressDataOptimized(const std::vector<int32_t>&, int, CompressionStats*);
template CompressedDataL3<uint32_t>* compressDataOptimized(const std::vector<uint32_t>&, int, CompressionStats*);
template CompressedDataL3<int64_t>* compressDataOptimized(const std::vector<int64_t>&, int, CompressionStats*);
template CompressedDataL3<uint64_t>* compressDataOptimized(const std::vector<uint64_t>&, int, CompressionStats*);

template void freeCompressedDataOptimized<int32_t>(CompressedDataL3<int32_t>*);
template void freeCompressedDataOptimized<uint32_t>(CompressedDataL3<uint32_t>*);
template void freeCompressedDataOptimized<int64_t>(CompressedDataL3<int64_t>*);
template void freeCompressedDataOptimized<uint64_t>(CompressedDataL3<uint64_t>*);

template int decompressDataOptimized(const CompressedDataL3<int32_t>*, int32_t*, int, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<uint32_t>*, uint32_t*, int, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<int64_t>*, int64_t*, int, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<uint64_t>*, uint64_t*, int, DecompressionStats*);

template int decompressDataOptimized(const CompressedDataL3<int32_t>*, std::vector<int32_t>&, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<uint32_t>*, std::vector<uint32_t>&, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<int64_t>*, std::vector<int64_t>&, DecompressionStats*);
template int decompressDataOptimized(const CompressedDataL3<uint64_t>*, std::vector<uint64_t>&, DecompressionStats*);