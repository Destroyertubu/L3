/**
 * L3 Transposed API Declarations
 *
 * Declaration-only header for Transposed encoder and decoder.
 * Uses extern template to avoid multiple definition issues when linking.
 *
 * The implementations are in:
 *   - encoder_Transposed.cu
 *   - decoder_Transposed.cu
 */

#ifndef L3_TRANSPOSED_API_DECL_HPP
#define L3_TRANSPOSED_API_DECL_HPP

#include "L3_Transposed_format.hpp"
#include <vector>
#include <cuda_runtime.h>

namespace Transposed_encoder {

/**
 * Encode data using Transposed (Word-Interleaved) layout
 */
template<typename T>
CompressedDataTransposed<T> encodeTransposedGPU(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config,
    cudaStream_t stream = 0);

/**
 * Encode data using Transposed layout with V3 partitioner (PolyCost)
 * Uses full polynomial model selection for better compression.
 */
template<typename T>
CompressedDataTransposed<T> encodeTransposedGPU_PolyCost(
    const std::vector<T>& data,
    int partition_size,
    const TransposedConfig& config,
    cudaStream_t stream = 0);

/**
 * Free compressed data
 */
template<typename T>
void freeCompressedData(CompressedDataTransposed<T>& data);

// Extern template declarations for encodeTransposedGPU
extern template CompressedDataTransposed<uint32_t> encodeTransposedGPU<uint32_t>(
    const std::vector<uint32_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<int32_t> encodeTransposedGPU<int32_t>(
    const std::vector<int32_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<uint64_t> encodeTransposedGPU<uint64_t>(
    const std::vector<uint64_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<int64_t> encodeTransposedGPU<int64_t>(
    const std::vector<int64_t>&, int, const TransposedConfig&, cudaStream_t);

// Extern template declarations for encodeTransposedGPU_PolyCost
extern template CompressedDataTransposed<uint32_t> encodeTransposedGPU_PolyCost<uint32_t>(
    const std::vector<uint32_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<int32_t> encodeTransposedGPU_PolyCost<int32_t>(
    const std::vector<int32_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<uint64_t> encodeTransposedGPU_PolyCost<uint64_t>(
    const std::vector<uint64_t>&, int, const TransposedConfig&, cudaStream_t);
extern template CompressedDataTransposed<int64_t> encodeTransposedGPU_PolyCost<int64_t>(
    const std::vector<int64_t>&, int, const TransposedConfig&, cudaStream_t);

extern template void freeCompressedData<uint32_t>(CompressedDataTransposed<uint32_t>&);
extern template void freeCompressedData<int32_t>(CompressedDataTransposed<int32_t>&);
extern template void freeCompressedData<uint64_t>(CompressedDataTransposed<uint64_t>&);
extern template void freeCompressedData<int64_t>(CompressedDataTransposed<int64_t>&);

} // namespace Transposed_encoder


namespace Transposed_decoder {

/**
 * Decompress all data using Transposed format
 */
template<typename T>
void decompressAll(
    const CompressedDataTransposed<T>& compressed,
    T* d_output,
    cudaStream_t stream = 0);

/**
 * Random access decompression
 */
template<typename T>
void decompressRandomAccess(
    const CompressedDataTransposed<T>& compressed,
    const int* d_query_indices,
    int num_queries,
    T* d_output,
    cudaStream_t stream = 0);

// Extern template declarations
extern template void decompressAll<uint32_t>(
    const CompressedDataTransposed<uint32_t>&, uint32_t*, cudaStream_t);
extern template void decompressAll<int32_t>(
    const CompressedDataTransposed<int32_t>&, int32_t*, cudaStream_t);
extern template void decompressAll<uint64_t>(
    const CompressedDataTransposed<uint64_t>&, uint64_t*, cudaStream_t);
extern template void decompressAll<int64_t>(
    const CompressedDataTransposed<int64_t>&, int64_t*, cudaStream_t);

extern template void decompressRandomAccess<uint32_t>(
    const CompressedDataTransposed<uint32_t>&, const int*, int, uint32_t*, cudaStream_t);
extern template void decompressRandomAccess<int32_t>(
    const CompressedDataTransposed<int32_t>&, const int*, int, int32_t*, cudaStream_t);
extern template void decompressRandomAccess<uint64_t>(
    const CompressedDataTransposed<uint64_t>&, const int*, int, uint64_t*, cudaStream_t);
extern template void decompressRandomAccess<int64_t>(
    const CompressedDataTransposed<int64_t>&, const int*, int, int64_t*, cudaStream_t);

} // namespace Transposed_decoder

#endif // L3_TRANSPOSED_API_DECL_HPP
