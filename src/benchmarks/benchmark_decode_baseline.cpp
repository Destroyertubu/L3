/**
 * L3 Decompression Kernel Baseline Benchmark
 *
 * Purpose: Measure pure kernel decompression performance across different bit widths
 * Metrics: Median kernel time (ms), throughput (GB/s)
 *
 * Test matrix:
 * - Bit widths: 1, 2, 4, 8, 12, 16, 24, 32
 * - Data sizes: 10M, 50M, 200M elements
 * - Warmup: 5 iterations
 * - Measurement: 20 iterations (report median)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include "L3_codec.hpp"

// Forward declaration of kernel launcher
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream
);

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Generate synthetic data with specific bit width characteristics
template<typename T>
std::vector<T> generateDataForBitWidth(size_t num_elements, int target_bits) {
    std::vector<T> data(num_elements);

    // Generate linear data with controlled deltas
    T max_delta = (1ULL << target_bits) - 1;
    if (target_bits >= 32) {
        max_delta = std::numeric_limits<T>::max() / num_elements;
    }

    T value = 0;
    for (size_t i = 0; i < num_elements; i++) {
        data[i] = value;
        // Create linear pattern with small deltas
        value += (max_delta / 10) + (i % 3);
    }

    return data;
}

// Benchmark result structure
struct BenchmarkResult {
    int bit_width;
    size_t num_elements;
    double median_kernel_time_ms;
    double median_throughput_gbps;
    double compression_ratio;
    int num_partitions;

    void print() const {
        std::cout << std::setw(8) << bit_width << " bits | "
                  << std::setw(12) << num_elements << " elem | "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << median_kernel_time_ms << " ms | "
                  << std::setw(8) << std::fixed << std::setprecision(2)
                  << median_throughput_gbps << " GB/s | "
                  << std::setw(6) << std::fixed << std::setprecision(2)
                  << compression_ratio << "x | "
                  << std::setw(6) << num_partitions << " parts" << std::endl;
    }

    void writeCSV(std::ofstream& csv) const {
        csv << bit_width << ","
            << num_elements << ","
            << median_kernel_time_ms << ","
            << median_throughput_gbps << ","
            << compression_ratio << ","
            << num_partitions << std::endl;
    }
};

// Measure pure decompression kernel time
template<typename T>
BenchmarkResult benchmarkDecompression(
    const std::vector<T>& data,
    int partition_size,
    int target_bits,
    int warmup_iters = 5,
    int measure_iters = 20)
{
    BenchmarkResult result;
    result.bit_width = target_bits;
    result.num_elements = data.size();

    // Compress data
    CompressionStats comp_stats;
    auto* compressed = compressData(data, partition_size, &comp_stats);
    result.compression_ratio = comp_stats.compression_ratio;
    result.num_partitions = compressed->num_partitions;

    // Allocate output buffer
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup_iters; i++) {
        launchDecompressWarpOpt(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measure
    std::vector<float> times;
    times.reserve(measure_iters);

    for (int i = 0; i < measure_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launchDecompressWarpOpt(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    // Calculate median
    std::sort(times.begin(), times.end());
    result.median_kernel_time_ms = times[measure_iters / 2];

    // Calculate throughput (based on uncompressed data size)
    double data_size_gb = (data.size() * sizeof(T)) / 1e9;
    result.median_throughput_gbps = data_size_gb / (result.median_kernel_time_ms / 1000.0);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return result;
}

int main(int argc, char** argv) {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  L3 Decompression Kernel - Baseline Benchmark            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Test configuration
    std::vector<int> bit_widths = {1, 2, 4, 8, 12, 16, 24, 32};
    std::vector<size_t> data_sizes = {10000000, 50000000, 200000000};  // 10M, 50M, 200M
    int partition_size = 2048;

    // CSV output
    std::ofstream csv("baseline_results.csv");
    csv << "BitWidth,NumElements,MedianKernelTime_ms,MedianThroughput_GBps,"
        << "CompressionRatio,NumPartitions\n";

    std::vector<BenchmarkResult> all_results;

    for (size_t data_size : data_sizes) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Dataset size: " << data_size << " elements ("
                  << (data_size * sizeof(uint64_t) / 1e6) << " MB)\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

        std::cout << "BitWidth | Elements      | Time     | Throughput | Ratio  | Partitions\n";
        std::cout << "─────────┼───────────────┼──────────┼────────────┼────────┼───────────\n";

        for (int bits : bit_widths) {
            auto data = generateDataForBitWidth<uint64_t>(data_size, bits);
            auto result = benchmarkDecompression(data, partition_size, bits);

            result.print();
            result.writeCSV(csv);
            all_results.push_back(result);
        }
    }

    csv.close();

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Baseline results saved to: baseline_results.csv             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    return 0;
}
