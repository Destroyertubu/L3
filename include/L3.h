#ifndef L3_OPT_H
#define L3_OPT_H

#include <cuda_runtime.h>
#include <cstdint>
#include <string>
#include "L3_format.hpp"

// Compressed data structure (compatible with original)
template<typename T>
struct CompressedDataOpt {
    // Metadata arrays (SoA layout)
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    int32_t* d_model_types;
    double* d_model_params;
    int32_t* d_delta_bits;
    int64_t* d_delta_array_bit_offsets;

    // Bit-packed delta array
    uint32_t* delta_array;

    // Optional: pre-unpacked deltas
    int64_t* d_plain_deltas;

    // Metadata
    int num_partitions;
    int total_elements;
};

// Benchmark result structure
struct BenchResult {
    std::string config_name;
    int64_t elements;
    int bitwidth;
    int num_partitions;
    float kernel_time_ms;
    double throughput_gbps;
    int warmup_iters;
    int timed_iters;
};

// ============================================================================
// Decoder Function Declarations
// ============================================================================

// Standard decoder (decompression_kernels.cu)
template<typename T>
cudaError_t launchDecompressOptimized(
    const CompressedDataOpt<T>& compressed_data,
    T* output,
    cudaStream_t stream);

template<typename T>
void launchDecompressOptimized(
    const CompressedDataL3<T>* compressed,
    T* output,
    cudaStream_t stream);

template<typename T>
void launchDecompressSimple(
    const CompressedDataL3<T>* compressed,
    T* output,
    cudaStream_t stream);

// Warp-optimized decoder (decoder_warp_opt.cu)
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
    T* output,
    cudaStream_t stream);

template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataOpt<T>& compressed,
    T* output,
    cudaStream_t stream);

// Bit-width specialized decoder (decoder_specialized.cu)
template<typename T>
void launchDecompressSpecialized(
    const CompressedDataL3<T>* compressed,
    T* output,
    int delta_bits_hint,
    cudaStream_t stream);

template<typename T>
void launchDecompressSpecialized(
    const CompressedDataOpt<T>& compressed,
    T* output,
    int delta_bits_hint,
    cudaStream_t stream);

// Phase2 decoder with cp.async pipeline (decompression_kernels_phase2.cu)
template<typename T>
void decompressL3_Phase2(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output,
    int avg_delta_bits,
    bool use_persistent);

// Phase2 bucket-based decoder (decompression_kernels_phase2_bucket.cu)
template<typename T>
void decompressL3_Phase2_Bucket(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const uint8_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets_int64,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output);

// 8/16-bit optimized decoder (decompression_kernels_opt.cu)
template<typename T>
void decompressL3_Optimized(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output,
    int avg_delta_bits);

// ============================================================================
// Utility Functions
// ============================================================================

void printBenchResult(const BenchResult& result);
void writeBenchResultToCSV(const std::string& filename, const BenchResult& result, bool write_header = false);

#endif // L3_OPT_H
