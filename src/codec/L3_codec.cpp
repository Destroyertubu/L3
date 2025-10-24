/**
 * GLECO Codec Implementation
 *
 * Simple, principle-preserving implementation of compression/decompression API
 *
 * PORT DATE: 2025-10-14
 * STATUS: Baseline implementation with fixed-size partitioning
 */

#include "L3_codec.hpp"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================================
// Fixed-Size Partitioning
// ============================================================================

template<typename T>
int createFixedSizePartitions(
    const T* d_data,
    int num_elements,
    int partition_size,
    int32_t* d_start_indices,
    int32_t* d_end_indices)
{
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

// Template instantiations
template int createFixedSizePartitions<int32_t>(const int32_t*, int, int, int32_t*, int32_t*);
template int createFixedSizePartitions<uint32_t>(const uint32_t*, int, int, int32_t*, int32_t*);
template int createFixedSizePartitions<int64_t>(const int64_t*, int, int, int32_t*, int32_t*);
template int createFixedSizePartitions<uint64_t>(const uint64_t*, int, int, int32_t*, int32_t*);

// ============================================================================
// Variable-Length Partitioning Support
// ============================================================================

template<typename T>
int uploadPrecomputedPartitions(
    const std::vector<PartitionInfo>& partitions,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds)
{
    int num_partitions = partitions.size();

    std::vector<int32_t> h_start(num_partitions);
    std::vector<int32_t> h_end(num_partitions);
    std::vector<int32_t> h_model_types(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int32_t> h_delta_bits(num_partitions);
    std::vector<int64_t> h_error_bounds(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        h_start[i] = partitions[i].start_idx;
        h_end[i] = partitions[i].end_idx;
        h_model_types[i] = partitions[i].model_type;
        h_delta_bits[i] = partitions[i].delta_bits;
        h_error_bounds[i] = partitions[i].error_bound;

        for (int j = 0; j < 4; j++) {
            h_model_params[i * 4 + j] = partitions[i].model_params[j];
        }
    }

    CUDA_CHECK(cudaMemcpy(d_start_indices, h_start.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end_indices, h_end.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_model_types, h_model_types.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_model_params, h_model_params.data(),
                          num_partitions * 4 * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta_bits, h_delta_bits.data(),
                          num_partitions * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_error_bounds, h_error_bounds.data(),
                          num_partitions * sizeof(int64_t),
                          cudaMemcpyHostToDevice));

    return num_partitions;
}

template int uploadPrecomputedPartitions<uint64_t>(
    const std::vector<PartitionInfo>&, int32_t*, int32_t*, int32_t*, double*, int32_t*, int64_t*);

// ============================================================================
// Compression Implementation
// ============================================================================

template<typename T>
CompressedDataGLECO<T>* compressData(
    const T* h_data,
    int num_elements,
    int partition_size,
    CompressionStats* stats)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Allocate compressed data structure
    CompressedDataGLECO<T>* compressed = new CompressedDataGLECO<T>();
    compressed->total_values = num_elements;

    // Upload data to GPU
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));

    // Compute number of partitions
    int num_partitions = (num_elements + partition_size - 1) / partition_size;
    compressed->num_partitions = num_partitions;

    // Allocate metadata arrays
    CUDA_CHECK(cudaMalloc(&compressed->d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_error_bounds, num_partitions * sizeof(int64_t)));

    // Allocate partition min/max bounds (computed from learned models)
    CUDA_CHECK(cudaMalloc(&compressed->d_partition_min_values, num_partitions * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&compressed->d_partition_max_values, num_partitions * sizeof(T)));

    // Create partitions
    createFixedSizePartitions(d_data, num_elements, partition_size,
                             compressed->d_start_indices,
                             compressed->d_end_indices);

    // Allocate total_bits counter on device
    int64_t* d_total_bits;
    CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(d_total_bits, 0, sizeof(int64_t)));

    // Launch model fitting kernel (now also computes min/max from learned models)
    launchModelFittingKernel<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_error_bounds,
        compressed->d_partition_min_values,
        compressed->d_partition_max_values,
        num_partitions,
        d_total_bits
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get total bits used
    int64_t total_bits;
    CUDA_CHECK(cudaMemcpy(&total_bits, d_total_bits, sizeof(int64_t), cudaMemcpyDeviceToHost));

    // Compute bit offsets
    launchSetBitOffsetsKernel(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate delta array
    int64_t delta_array_words = (total_bits + 31) / 32;
    compressed->delta_array_words = delta_array_words;

    CUDA_CHECK(cudaMalloc(&compressed->delta_array, delta_array_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed->delta_array, 0, delta_array_words * sizeof(uint32_t)));

    // Launch delta packing kernel
    launchDeltaPackingKernel<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions,
        compressed->delta_array,
        num_elements
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create device-side copy of struct for d_self
    CUDA_CHECK(cudaMalloc(&compressed->d_self, sizeof(CompressedDataGLECO<T>)));
    CUDA_CHECK(cudaMemcpy(compressed->d_self, compressed, sizeof(CompressedDataGLECO<T>), cudaMemcpyHostToDevice));

    // Record timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_total_bits));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

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
CompressedDataGLECO<T>* compressData(
    const std::vector<T>& h_data,
    int partition_size,
    CompressionStats* stats)
{
    return compressData(h_data.data(), h_data.size(), partition_size, stats);
}

// Template instantiations
template CompressedDataGLECO<int32_t>* compressData(const int32_t*, int, int, CompressionStats*);
template CompressedDataGLECO<uint32_t>* compressData(const uint32_t*, int, int, CompressionStats*);
template CompressedDataGLECO<int64_t>* compressData(const int64_t*, int, int, CompressionStats*);
template CompressedDataGLECO<uint64_t>* compressData(const uint64_t*, int, int, CompressionStats*);

template CompressedDataGLECO<int32_t>* compressData(const std::vector<int32_t>&, int, CompressionStats*);
template CompressedDataGLECO<uint32_t>* compressData(const std::vector<uint32_t>&, int, CompressionStats*);
template CompressedDataGLECO<int64_t>* compressData(const std::vector<int64_t>&, int, CompressionStats*);
template CompressedDataGLECO<uint64_t>* compressData(const std::vector<uint64_t>&, int, CompressionStats*);

// ============================================================================
// Compression with Precomputed Partitions (Variable-Length)
// ============================================================================

template<typename T>
CompressedDataGLECO<T>* compressDataWithPartitions(
    const T* h_data,
    int num_elements,
    const std::vector<PartitionInfo>& partitions,
    CompressionStats* stats)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Allocate compressed data structure
    CompressedDataGLECO<T>* compressed = new CompressedDataGLECO<T>();
    compressed->total_values = num_elements;
    int num_partitions = partitions.size();
    compressed->num_partitions = num_partitions;

    // Upload data to GPU
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, num_elements * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));

    // Allocate metadata arrays
    CUDA_CHECK(cudaMalloc(&compressed->d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_error_bounds, num_partitions * sizeof(int64_t)));

    // Allocate partition min/max bounds
    CUDA_CHECK(cudaMalloc(&compressed->d_partition_min_values, num_partitions * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&compressed->d_partition_max_values, num_partitions * sizeof(T)));

    // Upload precomputed partition metadata (already fitted by variable-length encoder)
    uploadPrecomputedPartitions<T>(
        partitions,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_error_bounds
    );

    // Compute bit offsets from delta_bits
    launchSetBitOffsetsKernel(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate total bits needed
    std::vector<int32_t> h_start(num_partitions);
    std::vector<int32_t> h_end(num_partitions);
    std::vector<int32_t> h_delta_bits(num_partitions);

    CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                         num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                         num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                         num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    int64_t total_bits = 0;
    for (int i = 0; i < num_partitions; i++) {
        int64_t part_len = h_end[i] - h_start[i];
        total_bits += part_len * h_delta_bits[i];
    }

    // Allocate delta array
    int64_t delta_array_words = (total_bits + 31) / 32;
    compressed->delta_array_words = delta_array_words;

    CUDA_CHECK(cudaMalloc(&compressed->delta_array, delta_array_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed->delta_array, 0, delta_array_words * sizeof(uint32_t)));

    // Launch delta packing kernel (using precomputed model params)
    launchDeltaPackingKernel<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        num_partitions,
        compressed->delta_array,
        num_elements
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute accurate partition min/max bounds from actual data
    launchComputePartitionBounds<T>(
        d_data,
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_partition_min_values,
        compressed->d_partition_max_values,
        num_partitions
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create device-side copy of struct for d_self
    CUDA_CHECK(cudaMalloc(&compressed->d_self, sizeof(CompressedDataGLECO<T>)));
    CUDA_CHECK(cudaMemcpy(compressed->d_self, compressed, sizeof(CompressedDataGLECO<T>), cudaMemcpyHostToDevice));

    // Record timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

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
        int64_t sum_weighted_bits = 0;
        int64_t sum_partition_lengths = 0;

        for (int p = 0; p < num_partitions; p++) {
            int64_t part_len = h_end[p] - h_start[p];
            sum_weighted_bits += static_cast<int64_t>(h_delta_bits[p]) * part_len;
            sum_partition_lengths += part_len;
        }

        stats->avg_delta_bits = sum_partition_lengths > 0 ?
            static_cast<int>((sum_weighted_bits + sum_partition_lengths / 2) / sum_partition_lengths) : 0;
    }

    return compressed;
}

template<typename T>
CompressedDataGLECO<T>* compressDataWithPartitions(
    const std::vector<T>& h_data,
    const std::vector<PartitionInfo>& partitions,
    CompressionStats* stats)
{
    return compressDataWithPartitions(h_data.data(), h_data.size(), partitions, stats);
}

// Template instantiations
template CompressedDataGLECO<uint64_t>* compressDataWithPartitions(
    const uint64_t*, int, const std::vector<PartitionInfo>&, CompressionStats*);
template CompressedDataGLECO<uint64_t>* compressDataWithPartitions(
    const std::vector<uint64_t>&, const std::vector<PartitionInfo>&, CompressionStats*);

// ============================================================================
// Free Compressed Data
// ============================================================================

template<typename T>
void freeCompressedData(CompressedDataGLECO<T>* compressed)
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
    if (compressed->d_partition_min_values) cudaFree(compressed->d_partition_min_values);
    if (compressed->d_partition_max_values) cudaFree(compressed->d_partition_max_values);
    if (compressed->d_self) cudaFree(compressed->d_self);

    delete compressed;
}

template void freeCompressedData<int32_t>(CompressedDataGLECO<int32_t>*);
template void freeCompressedData<uint32_t>(CompressedDataGLECO<uint32_t>*);
template void freeCompressedData<int64_t>(CompressedDataGLECO<int64_t>*);
template void freeCompressedData<uint64_t>(CompressedDataGLECO<uint64_t>*);

// ============================================================================
// Decompression (stub - will link to existing decompression kernels)
// ============================================================================

// Forward declare decompression kernel launchers
template<typename T>
void launchDecompressOptimized(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    cudaStream_t stream
);

// Simple non-optimized decoder (for debugging)
template<typename T>
void launchDecompressSimple(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    cudaStream_t stream
);

// Warp-optimized decoder (extreme performance)
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    cudaStream_t stream
);

template<typename T>
int decompressData(
    const CompressedDataGLECO<T>* compressed,
    T* h_output,
    int output_capacity,
    DecompressionStats* stats)
{
    if (!compressed || !h_output) return 0;
    if (output_capacity < compressed->total_values) return 0;

    // Allocate device output (NOT counted in timing)
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, compressed->total_values * sizeof(T)));

    // Create CUDA events for kernel-only timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start IMMEDIATELY before kernel launch
    CUDA_CHECK(cudaEventRecord(start));

    // Launch decompression kernel (from decoder_warp_opt.cu)
    // Use warp-optimized decoder for extreme performance
    launchDecompressWarpOpt(compressed, d_output, 0);

    // Record stop IMMEDIATELY after kernel launch (before sync or memcpy)
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Measure KERNEL-ONLY execution time
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Download result (NOT counted in timing)
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
int decompressData(
    const CompressedDataGLECO<T>* compressed,
    std::vector<T>& h_output,
    DecompressionStats* stats)
{
    h_output.resize(compressed->total_values);
    return decompressData(compressed, h_output.data(), h_output.size(), stats);
}

// Template instantiations
template int decompressData(const CompressedDataGLECO<int32_t>*, int32_t*, int, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<uint32_t>*, uint32_t*, int, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<int64_t>*, int64_t*, int, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<uint64_t>*, uint64_t*, int, DecompressionStats*);

template int decompressData(const CompressedDataGLECO<int32_t>*, std::vector<int32_t>&, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<uint32_t>*, std::vector<uint32_t>&, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<int64_t>*, std::vector<int64_t>&, DecompressionStats*);
template int decompressData(const CompressedDataGLECO<uint64_t>*, std::vector<uint64_t>&, DecompressionStats*);

// ============================================================================
// Roundtrip Test
// ============================================================================

template<typename T>
bool testRoundtrip(
    const std::vector<T>& original_data,
    int partition_size,
    bool verbose)
{
    if (verbose) {
        std::cout << "Testing roundtrip with " << original_data.size()
                  << " elements, partition_size=" << partition_size << std::endl;
    }

    // Compress
    CompressionStats comp_stats;
    CompressedDataGLECO<T>* compressed = compressData(original_data, partition_size, &comp_stats);

    if (verbose) {
        std::cout << "Compressed: " << comp_stats.original_bytes << " -> "
                  << comp_stats.compressed_bytes << " bytes ("
                  << comp_stats.compression_ratio << "x)" << std::endl;
    }

    // Decompress
    std::vector<T> decompressed;
    DecompressionStats decomp_stats;
    decompressData(compressed, decompressed, &decomp_stats);

    if (verbose) {
        std::cout << "Decompressed " << decomp_stats.total_elements << " elements" << std::endl;
    }

    // Verify
    bool identical = (original_data.size() == decompressed.size());
    if (identical) {
        for (size_t i = 0; i < original_data.size(); i++) {
            if (original_data[i] != decompressed[i]) {
                if (verbose) {
                    std::cout << "Mismatch at index " << i << ": original="
                              << original_data[i] << ", decompressed="
                              << decompressed[i] << std::endl;
                }
                identical = false;
                break;
            }
        }
    }

    // Cleanup
    freeCompressedData(compressed);

    if (verbose) {
        std::cout << "Roundtrip test: " << (identical ? "PASSED" : "FAILED") << std::endl;
    }

    return identical;
}

// Template instantiations
template bool testRoundtrip(const std::vector<int32_t>&, int, bool);
template bool testRoundtrip(const std::vector<uint32_t>&, int, bool);
template bool testRoundtrip(const std::vector<int64_t>&, int, bool);
template bool testRoundtrip(const std::vector<uint64_t>&, int, bool);

// ============================================================================
// Verification
// ============================================================================

template<typename T>
bool verifyCompressedData(const CompressedDataGLECO<T>* compressed)
{
    if (!compressed) return false;
    if (compressed->num_partitions <= 0) return false;
    if (compressed->total_values <= 0) return false;

    // Download metadata to verify
    std::vector<int32_t> h_start(compressed->num_partitions);
    std::vector<int32_t> h_end(compressed->num_partitions);

    CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                          compressed->num_partitions * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                          compressed->num_partitions * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    // Check partition invariants
    if (h_start[0] != 0) return false;
    if (h_end[compressed->num_partitions - 1] != compressed->total_values) return false;

    for (int i = 0; i < compressed->num_partitions; i++) {
        if (h_start[i] >= h_end[i]) return false;
        if (i > 0 && h_end[i-1] > h_start[i]) return false;  // Overlapping
    }

    return true;
}

template bool verifyCompressedData(const CompressedDataGLECO<int32_t>*);
template bool verifyCompressedData(const CompressedDataGLECO<uint32_t>*);
template bool verifyCompressedData(const CompressedDataGLECO<int64_t>*);
template bool verifyCompressedData(const CompressedDataGLECO<uint64_t>*);
