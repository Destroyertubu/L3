#ifndef L3_OPT_H
#define L3_OPT_H

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

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

// Function declarations
template<typename T>
cudaError_t launchDecompressOptimized(
    const CompressedDataOpt<T>& compressed_data,
    T* output,
    cudaStream_t stream = 0);

void printBenchResult(const BenchResult& result);
void writeBenchResultToCSV(const std::string& filename, const BenchResult& result, bool write_header = false);

#endif // L3_OPT_H
