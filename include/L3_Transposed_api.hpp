/**
 * L3 Transposed (Word-Interleaved) API
 *
 * Public API for encoding and decoding using the Transposed memory layout.
 *
 * KEY DIFFERENCE from Vertical (Lane-Major):
 *   Vertical:   [Lane0_W0][Lane0_W1]...[Lane0_WN][Lane1_W0]...[Lane31_WN]
 *   Transposed: [L0_W0][L1_W0]...[L31_W0][L0_W1][L1_W1]...[L31_WN]
 *
 * Benefits:
 *   - Perfect memory coalescing: 32 threads read 32 consecutive words
 *   - Single 128-byte cache line transaction per warp read
 *   - 4x better L1 cache efficiency
 *
 * Date: 2025-12-16
 */

#ifndef L3_TRANSPOSED_API_HPP
#define L3_TRANSPOSED_API_HPP

#include "L3_Transposed_format.hpp"
#include "../src/kernels/compression/encoder_Transposed.cu"
#include "../src/kernels/decompression/decoder_Transposed.cu"

namespace L3 {
namespace Transposed {

// ============================================================================
// Encoder API
// ============================================================================

/**
 * Encode data using Transposed (Word-Interleaved) layout
 *
 * @param data         Input data vector
 * @param partition_size  Size of each partition
 * @param config       Transposed encoding configuration
 * @param stream       CUDA stream (default: 0)
 * @return Compressed data in Transposed format
 */
template<typename T>
CompressedDataTransposed<T> encode(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config = TransposedConfig::defaultConfig(),
    cudaStream_t stream = 0)
{
    return Transposed_encoder::encodeTransposedGPU<T>(data, partition_size, config, stream);
}

/**
 * Encode with cost-optimal partitioning
 */
template<typename T>
CompressedDataTransposed<T> encodeCostOptimal(
    const std::vector<T>& data,
    int partition_size_hint = 4096,
    cudaStream_t stream = 0)
{
    return Transposed_encoder::encodeTransposedGPU<T>(
        data, partition_size_hint, TransposedConfig::costOptimal(), stream);
}

/**
 * Free compressed data
 */
template<typename T>
void freeCompressed(CompressedDataTransposed<T>& compressed)
{
    Transposed_encoder::freeCompressedData<T>(compressed);
}

// ============================================================================
// Decoder API
// ============================================================================

/**
 * Decompress all data
 *
 * @param compressed   Compressed data in Transposed format
 * @param d_output     Device pointer for output (must be pre-allocated)
 * @param stream       CUDA stream (default: 0)
 */
template<typename T>
void decode(
    const CompressedDataTransposed<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0)
{
    Transposed_decoder::decompressAll<T>(compressed, d_output, stream);
}

/**
 * Random access decompression
 *
 * @param compressed       Compressed data
 * @param d_query_indices  Device pointer to indices to query
 * @param num_queries      Number of queries
 * @param d_output         Device pointer for output
 * @param stream           CUDA stream
 */
template<typename T>
void decodeRandomAccess(
    const CompressedDataTransposed<T>& compressed,
    const int* d_query_indices,
    int num_queries,
    T* d_output,
    cudaStream_t stream = 0)
{
    Transposed_decoder::decompressRandomAccess<T>(
        compressed, d_query_indices, num_queries, d_output, stream);
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get compressed size in bytes
 */
template<typename T>
size_t getCompressedSizeBytes(const CompressedDataTransposed<T>& compressed)
{
    size_t size = 0;

    // Partition metadata
    int np = compressed.num_partitions;
    size += np * sizeof(int32_t);  // start_indices
    size += np * sizeof(int32_t);  // end_indices
    size += np * sizeof(int32_t);  // model_types
    size += np * 4 * sizeof(double);  // model_params
    size += np * sizeof(int32_t);  // delta_bits
    size += np * sizeof(int64_t);  // error_bounds

    // Transposed metadata
    size += np * sizeof(int32_t);  // num_mini_vectors
    size += np * sizeof(int32_t);  // tail_sizes
    size += np * sizeof(int64_t);  // transposed_offsets

    // Transposed delta data
    size += compressed.transposed_delta_words * sizeof(uint32_t);

    return size;
}

/**
 * Get compression ratio
 */
template<typename T>
double getCompressionRatio(const CompressedDataTransposed<T>& compressed)
{
    size_t original_bytes = compressed.total_values * sizeof(T);
    size_t compressed_bytes = getCompressedSizeBytes(compressed);
    return static_cast<double>(original_bytes) / compressed_bytes;
}

/**
 * Print compression statistics
 */
template<typename T>
void printStats(const CompressedDataTransposed<T>& compressed)
{
    size_t original_bytes = compressed.total_values * sizeof(T);
    size_t compressed_bytes = getCompressedSizeBytes(compressed);

    printf("=== Transposed Compression Statistics ===\n");
    printf("  Total values:      %d\n", compressed.total_values);
    printf("  Num partitions:    %d\n", compressed.num_partitions);
    printf("  Original size:     %.2f MB\n", original_bytes / 1024.0 / 1024.0);
    printf("  Compressed size:   %.2f MB\n", compressed_bytes / 1024.0 / 1024.0);
    printf("  Compression ratio: %.2fx\n", getCompressionRatio(compressed));
    printf("  Kernel time:       %.4f ms\n", compressed.kernel_time_ms);
    printf("  Memory layout:     %s\n",
           compressed.layout == MemoryLayout::WORD_INTERLEAVED ? "Word-Interleaved (Transposed)" : "Lane-Major");
    printf("=========================================\n");
}

} // namespace Transposed
} // namespace L3

#endif // L3_TRANSPOSED_API_HPP
