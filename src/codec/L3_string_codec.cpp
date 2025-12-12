/**
 * L3 String Compression Codec Implementation
 *
 * High-level API for GPU-accelerated string compression/decompression
 * using learned models with linear regression.
 *
 * Based on LeCo string support with GPU optimizations.
 */

#include "L3_string_format.hpp"
#include "L3_string_utils.hpp"
#include "L3_format.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
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

// Forward declarations of kernel launch functions
void launchEncodeStringsToUint64(
    const char* d_strings,
    const int32_t* d_string_offsets,
    const int8_t* d_string_lengths,
    int32_t num_strings,
    int32_t min_char,
    int32_t shift_bits,
    int32_t max_length,
    uint64_t* d_encoded_values,
    int8_t* d_original_lengths,
    cudaStream_t stream);

void launchFitStringModel(
    const uint64_t* d_encoded_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream);

void launchSetStringBitOffsets(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_delta_bits,
    int64_t* d_bit_offsets,
    int num_partitions,
    cudaStream_t stream);

void launchPackStringDeltas(
    const uint64_t* d_encoded_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    int total_elements,
    uint32_t* delta_array,
    cudaStream_t stream);

void launchDecompressToEncodedValues(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    int num_partitions,
    uint64_t* d_encoded_values,
    cudaStream_t stream);

void launchReconstructStrings(
    const uint64_t* d_encoded_values,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int32_t num_strings,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream);

void launchDecompressToStrings(
    const uint32_t* delta_array,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_bit_offsets,
    const int8_t* d_original_lengths,
    const char* d_common_prefix,
    int32_t common_prefix_length,
    int32_t max_encoded_length,
    int32_t min_char,
    int32_t shift_bits,
    int num_partitions,
    int32_t output_string_stride,
    char* d_output_strings,
    cudaStream_t stream);

// ============================================================================
// Helper: Create partitions for string data
// ============================================================================

static int createStringPartitions(
    int num_strings,
    int partition_size,
    int32_t* d_start_indices,
    int32_t* d_end_indices)
{
    // Align partition size to warp boundaries
    int aligned_size = ((partition_size + 31) / 32) * 32;
    int num_partitions = (num_strings + aligned_size - 1) / aligned_size;

    std::vector<int32_t> h_start(num_partitions);
    std::vector<int32_t> h_end(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        h_start[i] = i * aligned_size;
        h_end[i] = std::min((i + 1) * aligned_size, num_strings);
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
// String Compression API
// ============================================================================

/**
 * Compress a collection of strings using GPU-accelerated learned compression
 *
 * @param strings Input strings to compress
 * @param config Compression configuration (auto-configured if empty)
 * @param stats Output statistics (optional)
 * @return Pointer to compressed data structure
 */
CompressedStringData* compressStrings(
    const std::vector<std::string>& strings,
    StringCompressionConfig config,
    StringCompressionStats* stats)
{
    if (strings.empty()) return nullptr;

    // Auto-configure if not set
    if (config.min_char == 0 && config.max_char == 255) {
        config = autoConfigureStringCompression(strings, true, config.partition_size);
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    int num_strings = strings.size();

    // Allocate compressed data structure
    CompressedStringData* compressed = new CompressedStringData();
    compressed->total_strings = num_strings;
    compressed->mode = config.mode;
    compressed->min_char = config.min_char;
    compressed->max_char = config.max_char;
    compressed->shift_bits = config.shift_bits;
    compressed->max_encoded_length = config.max_string_length;
    compressed->common_prefix_length = config.common_prefix.size();

    // Prepare string data for GPU upload
    // 1. Remove common prefix and flatten strings
    std::vector<char> flattened;
    std::vector<int32_t> offsets(num_strings);
    std::vector<int8_t> lengths(num_strings);

    size_t prefix_len = config.common_prefix.size();
    int64_t original_bytes = 0;

    for (int i = 0; i < num_strings; i++) {
        offsets[i] = flattened.size();
        std::string suffix = strings[i].substr(prefix_len);
        lengths[i] = static_cast<int8_t>(std::min(suffix.size(),
                                                   size_t(config.max_string_length)));

        // Pad to max length
        while (suffix.size() < size_t(config.max_string_length)) {
            suffix += static_cast<char>(config.min_char);
        }
        suffix = suffix.substr(0, config.max_string_length);

        flattened.insert(flattened.end(), suffix.begin(), suffix.end());
        original_bytes += strings[i].size();
    }

    // Upload string data to GPU
    char* d_strings;
    int32_t* d_string_offsets;
    int8_t* d_string_lengths;
    uint64_t* d_encoded_values;

    CUDA_CHECK(cudaMalloc(&d_strings, flattened.size() * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&d_string_offsets, num_strings * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_string_lengths, num_strings * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_encoded_values, num_strings * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_original_lengths, num_strings * sizeof(int8_t)));

    CUDA_CHECK(cudaMemcpyAsync(d_strings, flattened.data(),
                               flattened.size() * sizeof(char),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_string_offsets, offsets.data(),
                               num_strings * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_string_lengths, lengths.data(),
                               num_strings * sizeof(int8_t),
                               cudaMemcpyHostToDevice, stream));

    // Upload common prefix
    if (compressed->common_prefix_length > 0) {
        CUDA_CHECK(cudaMalloc(&compressed->d_common_prefix, compressed->common_prefix_length));
        CUDA_CHECK(cudaMemcpyAsync(compressed->d_common_prefix, config.common_prefix.data(),
                                   compressed->common_prefix_length,
                                   cudaMemcpyHostToDevice, stream));
    }

    // Step 1: Encode strings to uint64_t values
    launchEncodeStringsToUint64(
        d_strings, d_string_offsets, d_string_lengths,
        num_strings, config.min_char, config.shift_bits, config.max_string_length,
        d_encoded_values, compressed->d_original_lengths, stream);

    // Step 2: Create partitions
    int aligned_partition_size = ((config.partition_size + 31) / 32) * 32;
    int num_partitions = (num_strings + aligned_partition_size - 1) / aligned_partition_size;
    compressed->num_partitions = num_partitions;

    // Allocate partition metadata
    CUDA_CHECK(cudaMalloc(&compressed->d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed->d_error_bounds, num_partitions * sizeof(int64_t)));

    createStringPartitions(num_strings, aligned_partition_size,
                          compressed->d_start_indices, compressed->d_end_indices);

    // Step 3: Fit models and calculate delta bits
    int64_t* d_total_bits;
    CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
    CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    launchFitStringModel(
        d_encoded_values,
        compressed->d_start_indices, compressed->d_end_indices,
        compressed->d_model_types, compressed->d_model_params,
        compressed->d_delta_bits, compressed->d_error_bounds,
        num_partitions, d_total_bits, stream);

    // Get total bits
    int64_t total_bits;
    CUDA_CHECK(cudaMemcpyAsync(&total_bits, d_total_bits, sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Step 4: Calculate bit offsets
    launchSetStringBitOffsets(
        compressed->d_start_indices, compressed->d_end_indices,
        compressed->d_delta_bits, compressed->d_delta_array_bit_offsets,
        num_partitions, stream);

    // Step 5: Allocate and pack delta array
    int64_t delta_array_words = (total_bits + 31) / 32;
    compressed->delta_array_words = delta_array_words;

    CUDA_CHECK(cudaMalloc(&compressed->delta_array, delta_array_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(compressed->delta_array, 0,
                              delta_array_words * sizeof(uint32_t), stream));

    launchPackStringDeltas(
        d_encoded_values,
        compressed->d_start_indices, compressed->d_end_indices,
        compressed->d_model_types, compressed->d_model_params,
        compressed->d_delta_bits, compressed->d_delta_array_bit_offsets,
        num_partitions, num_strings, compressed->delta_array, stream);

    // Record timing
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    compressed->compression_time_ms = elapsed_ms;

    // Calculate compressed size
    int64_t compressed_bytes = num_partitions * (
        sizeof(int32_t) * 2 +           // start/end indices
        sizeof(int32_t) +               // model type
        sizeof(double) * 4 +            // model params
        sizeof(int32_t) +               // delta bits
        sizeof(int64_t) * 2             // bit offset + error bound
    ) + delta_array_words * sizeof(uint32_t)
      + num_strings * sizeof(int8_t)    // original lengths
      + compressed->common_prefix_length;

    compressed->compression_ratio = static_cast<double>(original_bytes) / compressed_bytes;

    // Fill statistics
    if (stats) {
        stats->original_bytes = original_bytes;
        stats->compressed_bytes = compressed_bytes;
        stats->compression_ratio = compressed->compression_ratio;
        stats->compression_time_ms = elapsed_ms;
        stats->compression_throughput_gbps = (original_bytes / 1e9) / (elapsed_ms / 1000.0);
        stats->num_partitions = num_partitions;
        stats->total_strings = num_strings;
        stats->common_prefix_length = compressed->common_prefix_length;
        stats->total_bits_used = total_bits;

        // Calculate weighted average delta bits
        std::vector<int32_t> h_delta_bits(num_partitions);
        std::vector<int32_t> h_start(num_partitions);
        std::vector<int32_t> h_end(num_partitions);

        CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                              num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

        int64_t sum_weighted = 0, sum_lengths = 0;
        for (int p = 0; p < num_partitions; p++) {
            int64_t len = h_end[p] - h_start[p];
            sum_weighted += h_delta_bits[p] * len;
            sum_lengths += len;
        }
        stats->avg_delta_bits = sum_lengths > 0 ? static_cast<int32_t>(sum_weighted / sum_lengths) : 0;
    }

    // Cleanup temporary allocations
    CUDA_CHECK(cudaFree(d_strings));
    CUDA_CHECK(cudaFree(d_string_offsets));
    CUDA_CHECK(cudaFree(d_string_lengths));
    CUDA_CHECK(cudaFree(d_encoded_values));
    CUDA_CHECK(cudaFree(d_total_bits));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return compressed;
}

// ============================================================================
// String Decompression API
// ============================================================================

/**
 * Decompress strings from compressed data
 *
 * @param compressed Compressed string data
 * @param config Compression configuration (must match compression config)
 * @param output Output vector for decompressed strings
 * @param stats Output statistics (optional)
 * @return Number of strings decompressed
 */
int decompressStrings(
    const CompressedStringData* compressed,
    const StringCompressionConfig& config,
    std::vector<std::string>& output,
    StringDecompressionStats* stats)
{
    if (!compressed || compressed->total_strings == 0) return 0;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    int num_strings = compressed->total_strings;

    // Allocate device memory for output
    int max_string_len = compressed->common_prefix_length + compressed->max_encoded_length + 1;
    int output_stride = ((max_string_len + 3) / 4) * 4;  // Align to 4 bytes

    char* d_output_strings;
    CUDA_CHECK(cudaMalloc(&d_output_strings, num_strings * output_stride * sizeof(char)));
    CUDA_CHECK(cudaMemsetAsync(d_output_strings, 0, num_strings * output_stride * sizeof(char), stream));

    // Launch decompression kernel
    launchDecompressToStrings(
        compressed->delta_array,
        compressed->d_start_indices, compressed->d_end_indices,
        compressed->d_model_params, compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->d_original_lengths,
        compressed->d_common_prefix, compressed->common_prefix_length,
        compressed->max_encoded_length,
        compressed->min_char, compressed->shift_bits,
        compressed->num_partitions, output_stride,
        d_output_strings, stream);

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Download results
    std::vector<char> h_output(num_strings * output_stride);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output_strings,
                          num_strings * output_stride * sizeof(char),
                          cudaMemcpyDeviceToHost));

    // Convert to strings
    output.resize(num_strings);
    int64_t total_bytes = 0;

    for (int i = 0; i < num_strings; i++) {
        const char* str_ptr = h_output.data() + i * output_stride;
        output[i] = std::string(str_ptr);
        total_bytes += output[i].size();
    }

    // Fill statistics
    if (stats) {
        stats->total_strings = num_strings;
        stats->decompressed_bytes = total_bytes;
        stats->compressed_bytes = compressed->delta_array_words * sizeof(uint32_t);
        stats->decompression_time_ms = elapsed_ms;
        stats->decompression_throughput_gbps = (total_bytes / 1e9) / (elapsed_ms / 1000.0);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output_strings));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return num_strings;
}

// ============================================================================
// Free Compressed Data
// ============================================================================

void freeCompressedStringData(CompressedStringData* compressed)
{
    if (!compressed) return;

    if (compressed->d_start_indices) cudaFree(compressed->d_start_indices);
    if (compressed->d_end_indices) cudaFree(compressed->d_end_indices);
    if (compressed->d_model_types) cudaFree(compressed->d_model_types);
    if (compressed->d_model_params) cudaFree(compressed->d_model_params);
    if (compressed->d_delta_bits) cudaFree(compressed->d_delta_bits);
    if (compressed->d_delta_array_bit_offsets) cudaFree(compressed->d_delta_array_bit_offsets);
    if (compressed->d_error_bounds) cudaFree(compressed->d_error_bounds);
    if (compressed->d_original_lengths) cudaFree(compressed->d_original_lengths);
    if (compressed->d_common_prefix) cudaFree(compressed->d_common_prefix);
    if (compressed->delta_array) cudaFree(compressed->delta_array);
    if (compressed->d_encoded_values_64) cudaFree(compressed->d_encoded_values_64);
    if (compressed->d_encoded_values_128) cudaFree(compressed->d_encoded_values_128);

    delete compressed;
}

// ============================================================================
// Utility Functions
// ============================================================================

void printStringCompressionStats(const StringCompressionStats& stats)
{
    std::cout << "=== String Compression Statistics ===" << std::endl;
    std::cout << "Total strings:         " << stats.total_strings << std::endl;
    std::cout << "Original size:         " << stats.original_bytes << " bytes" << std::endl;
    std::cout << "Compressed size:       " << stats.compressed_bytes << " bytes" << std::endl;
    std::cout << "Compression ratio:     " << stats.compression_ratio << "x" << std::endl;
    std::cout << "Compression time:      " << stats.compression_time_ms << " ms" << std::endl;
    std::cout << "Throughput:            " << stats.compression_throughput_gbps << " GB/s" << std::endl;
    std::cout << "Num partitions:        " << stats.num_partitions << std::endl;
    std::cout << "Common prefix length:  " << stats.common_prefix_length << std::endl;
    std::cout << "Avg delta bits:        " << stats.avg_delta_bits << std::endl;
}

void printStringDecompressionStats(const StringDecompressionStats& stats)
{
    std::cout << "=== String Decompression Statistics ===" << std::endl;
    std::cout << "Total strings:         " << stats.total_strings << std::endl;
    std::cout << "Decompressed size:     " << stats.decompressed_bytes << " bytes" << std::endl;
    std::cout << "Decompression time:    " << stats.decompression_time_ms << " ms" << std::endl;
    std::cout << "Throughput:            " << stats.decompression_throughput_gbps << " GB/s" << std::endl;
}
