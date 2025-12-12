/**
 * L3/L3 Vertical API Header
 *
 * Declares the public API for Vertical encoder and decoder.
 * These functions are implemented in encoder_Vertical_opt.cu and decoder_Vertical_opt.cu
 */

#ifndef L3_Vertical_API_HPP
#define L3_Vertical_API_HPP

#include "L3_Vertical_format.hpp"
#include "L3_codec.hpp"  // For PartitionInfo
#include <vector>
#include <cuda_runtime.h>

namespace Vertical_encoder {

/**
 * Create fixed-size partitions for encoding
 */
template<typename T>
std::vector<PartitionInfo> createFixedPartitions(int data_size, int partition_size);

/**
 * Encode data using Vertical compression with dual format support
 *
 * @param data       Input data (host vector)
 * @param partitions Partition boundaries (non-const, will be modified)
 * @param config     Encoder configuration
 * @param stream     CUDA stream (optional, default 0)
 * @return CompressedDataVertical containing both sequential and interleaved formats
 */
template<typename T>
CompressedDataVertical<T> encodeVertical(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions,
    const VerticalConfig& config,
    cudaStream_t stream = 0);

/**
 * Encode data using GPU-only pipeline with dynamic allocation
 *
 * @param data           Input data (host vector)
 * @param partition_size Fixed partition size
 * @param config         Encoder configuration
 * @param stream         CUDA stream (optional, default 0)
 * @return CompressedDataVertical containing both sequential and interleaved formats
 */
template<typename T>
CompressedDataVertical<T> encodeVerticalGPU(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0);

/**
 * Encode data using GPU-only pipeline with zero mid-pipeline synchronization
 *
 * Uses pre-allocation to eliminate intermediate cudaStreamSynchronize calls.
 * Trade-off: Uses more memory (worst-case 64-bit per value) but achieves
 * zero mid-pipeline CPU-GPU synchronization for maximum throughput.
 *
 * @param data           Input data (host vector)
 * @param partition_size Fixed partition size
 * @param config         Encoder configuration
 * @param stream         CUDA stream (optional, default 0)
 * @return CompressedDataVertical containing both sequential and interleaved formats
 */
template<typename T>
CompressedDataVertical<T> encodeVerticalGPU_ZeroSync(
    const std::vector<T>& data,
    int partition_size,
    const VerticalConfig& config,
    cudaStream_t stream = 0);

/**
 * Free compressed data structure
 */
template<typename T>
void freeCompressedData(CompressedDataVertical<T>& data);

// Explicit instantiations (defined in encoder_Vertical_opt.cu)
extern template std::vector<PartitionInfo> createFixedPartitions<uint32_t>(int, int);
extern template std::vector<PartitionInfo> createFixedPartitions<uint64_t>(int, int);
extern template std::vector<PartitionInfo> createFixedPartitions<int32_t>(int, int);
extern template std::vector<PartitionInfo> createFixedPartitions<int64_t>(int, int);

extern template CompressedDataVertical<uint32_t> encodeVertical<uint32_t>(
    const std::vector<uint32_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<uint64_t> encodeVertical<uint64_t>(
    const std::vector<uint64_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int32_t> encodeVertical<int32_t>(
    const std::vector<int32_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int64_t> encodeVertical<int64_t>(
    const std::vector<int64_t>&, std::vector<PartitionInfo>&, const VerticalConfig&, cudaStream_t);

extern template CompressedDataVertical<uint32_t> encodeVerticalGPU<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<uint64_t> encodeVerticalGPU<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int32_t> encodeVerticalGPU<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int64_t> encodeVerticalGPU<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

extern template CompressedDataVertical<uint32_t> encodeVerticalGPU_ZeroSync<uint32_t>(
    const std::vector<uint32_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<uint64_t> encodeVerticalGPU_ZeroSync<uint64_t>(
    const std::vector<uint64_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int32_t> encodeVerticalGPU_ZeroSync<int32_t>(
    const std::vector<int32_t>&, int, const VerticalConfig&, cudaStream_t);
extern template CompressedDataVertical<int64_t> encodeVerticalGPU_ZeroSync<int64_t>(
    const std::vector<int64_t>&, int, const VerticalConfig&, cudaStream_t);

extern template void freeCompressedData<uint32_t>(CompressedDataVertical<uint32_t>&);
extern template void freeCompressedData<uint64_t>(CompressedDataVertical<uint64_t>&);
extern template void freeCompressedData<int32_t>(CompressedDataVertical<int32_t>&);
extern template void freeCompressedData<int64_t>(CompressedDataVertical<int64_t>&);

} // namespace Vertical_encoder


namespace Vertical_decoder {

/**
 * Decompress all data using the specified mode
 *
 * @param compressed Compressed data structure
 * @param d_output   Output buffer (device memory, must be pre-allocated)
 * @param mode       Decompression mode (SEQUENTIAL, INTERLEAVED, BRANCHLESS, AUTO)
 * @param stream     CUDA stream for async execution
 */
template<typename T>
void decompressAll(
    const CompressedDataVertical<T>& compressed,
    T* d_output,
    DecompressMode mode,
    cudaStream_t stream = 0);

/**
 * Decompress specific indices (random access)
 *
 * @param compressed Compressed data structure
 * @param d_indices  Array of indices to decompress (device memory)
 * @param num_indices Number of indices
 * @param d_output   Output buffer (device memory)
 * @param stream     CUDA stream
 */
template<typename T>
void decompressIndices(
    const CompressedDataVertical<T>& compressed,
    const int* d_indices,
    int num_indices,
    T* d_output,
    cudaStream_t stream = 0);

// Explicit instantiations (defined in decoder_Vertical_opt.cu)
extern template void decompressAll<uint32_t>(
    const CompressedDataVertical<uint32_t>&, uint32_t*, DecompressMode, cudaStream_t);
extern template void decompressAll<uint64_t>(
    const CompressedDataVertical<uint64_t>&, uint64_t*, DecompressMode, cudaStream_t);
extern template void decompressAll<int32_t>(
    const CompressedDataVertical<int32_t>&, int32_t*, DecompressMode, cudaStream_t);
extern template void decompressAll<int64_t>(
    const CompressedDataVertical<int64_t>&, int64_t*, DecompressMode, cudaStream_t);

extern template void decompressIndices<uint32_t>(
    const CompressedDataVertical<uint32_t>&, const int*, int, uint32_t*, cudaStream_t);
extern template void decompressIndices<uint64_t>(
    const CompressedDataVertical<uint64_t>&, const int*, int, uint64_t*, cudaStream_t);
extern template void decompressIndices<int32_t>(
    const CompressedDataVertical<int32_t>&, const int*, int, int32_t*, cudaStream_t);
extern template void decompressIndices<int64_t>(
    const CompressedDataVertical<int64_t>&, const int*, int, int64_t*, cudaStream_t);

} // namespace Vertical_decoder

#endif // L3_Vertical_API_HPP
