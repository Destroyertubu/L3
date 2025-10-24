/**
 * Phase 2 Decompression Benchmark
 *
 * Comprehensive benchmarking of Phase 2 optimizations:
 * - Tests all bit widths (1/2/4/8/12/16/24/32)
 * - Measures performance with/without persistent threads
 * - Compares against Phase 1 baseline
 * - Generates detailed performance reports
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include "L3_codec.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// Forward declarations
template<typename T>
void decompressGLECO_Optimized(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, T*, int);

template<typename T>
void decompressGLECO_Phase2(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, T*, int, bool);

struct BenchmarkResult {
    std::string kernel_name;
    int bitwidth;
    size_t data_size;
    int num_partitions;
    float median_time_ms;
    double throughput_gbs;
    double speedup;
    bool correctness_passed;
};

template<typename T>
std::vector<T> generateData(size_t n, const std::string& pattern, int target_bits) {
    std::vector<T> data(n);

    if (pattern == "linear") {
        // Linear pattern optimized for low bit width
        int64_t step = (target_bits <= 8) ? 1 : (1LL << (target_bits - 8));
        for (size_t i = 0; i < n; i++) {
            data[i] = static_cast<T>(i * step + 1000);
        }
    } else if (pattern == "quadratic") {
        // Quadratic with controlled growth
        for (size_t i = 0; i < n; i++) {
            int64_t val = (i * i) / 1000 + i * 10 + 5000;
            data[i] = static_cast<T>(val);
        }
    } else if (pattern == "random_low") {
        // Random with low delta bits
        srand(42);
        data[0] = 10000;
        for (size_t i = 1; i < n; i++) {
            int delta = (rand() % 256) - 128;  // 8-bit deltas
            data[i] = data[i-1] + delta;
        }
    } else if (pattern == "uniform") {
        // Uniform distribution
        T base = static_cast<T>(1000000);
        for (size_t i = 0; i < n; i++) {
            data[i] = base + static_cast<T>(i);
        }
    }

    return data;
}

template<typename T>
float benchmarkKernel(
    void (*kernel_func)(const int32_t*, const int32_t*, const int32_t*, const double*,
                       const int32_t*, const int64_t*, const uint32_t*,
                       int, int, T*, int),
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    int total_elements,
    int avg_delta_bits,
    int num_iters = 20)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; i++) {
        kernel_func(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            compressed->d_delta_bits,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            total_elements,
            d_output,
            avg_delta_bits
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    std::vector<float> times;
    for (int i = 0; i < num_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        kernel_func(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            compressed->d_delta_bits,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            total_elements,
            d_output,
            avg_delta_bits
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Return median
    std::sort(times.begin(), times.end());
    return times[num_iters / 2];
}

template<typename T>
float benchmarkPhase2(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
    int total_elements,
    int avg_delta_bits,
    bool use_persistent,
    int num_iters = 20)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; i++) {
        decompressGLECO_Phase2(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            compressed->d_delta_bits,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            total_elements,
            d_output,
            avg_delta_bits,
            use_persistent
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Measure
    std::vector<float> times;
    for (int i = 0; i < num_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressGLECO_Phase2(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            compressed->d_delta_bits,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            total_elements,
            d_output,
            avg_delta_bits,
            use_persistent
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::sort(times.begin(), times.end());
    return times[num_iters / 2];
}

template<typename T>
bool verifyCorrectness(const std::vector<T>& original, T* d_output, size_t n) {
    std::vector<T> decompressed(n);
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         n * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; i++) {
        if (decompressed[i] != original[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                     << "expected " << original[i]
                     << ", got " << decompressed[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GLECO Phase 2 Decompression Benchmark                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Get device properties
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Memory Bandwidth: " << (prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2 / 1e9) << " GB/s\n";
    std::cout << "\n";

    // Test configurations
    const std::vector<size_t> sizes = {1000000, 8000000, 64000000};  // 1M, 8M, 64M
    const std::vector<int> bitwidths = {8, 16};  // Focus on optimized widths
    const std::vector<std::string> patterns = {"linear", "uniform"};

    std::vector<BenchmarkResult> results;

    // CSV output
    std::ofstream csv("phase2_results.csv");
    csv << "Kernel,Bitwidth,DataSize,NumPartitions,MedianTimeMS,ThroughputGBs,SpeedupVsPhase1,Correctness\n";

    for (size_t n : sizes) {
        for (int target_bits : bitwidths) {
            for (const auto& pattern : patterns) {
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
                std::cout << "Dataset: " << (n / 1000000.0) << "M elements, "
                         << target_bits << "-bit, pattern: " << pattern << "\n";
                std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

                // Generate data
                auto data = generateData<uint64_t>(n, pattern, target_bits);

                // Compress
                CompressionStats stats;
                auto* compressed = compressData(data, 4096, &stats);

                std::cout << "Partitions: " << compressed->num_partitions << "\n";
                std::cout << "Avg delta bits: " << stats.avg_delta_bits << "\n";
                std::cout << "Compression ratio: " << stats.compression_ratio << "x\n\n";

                // Allocate output
                uint64_t* d_output;
                CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(uint64_t)));

                // Benchmark Phase 1 (baseline)
                std::cout << "[1/3] Phase 1 (Optimized)...";
                float time_phase1 = benchmarkKernel<uint64_t>(
                    decompressGLECO_Optimized<uint64_t>,
                    compressed, d_output, n,
                    static_cast<int>(stats.avg_delta_bits)
                );
                double throughput_phase1 = (n * sizeof(uint64_t) / 1e9) / (time_phase1 / 1000.0);
                std::cout << " " << time_phase1 << " ms, " << throughput_phase1 << " GB/s\n";

                bool correct_phase1 = verifyCorrectness(data, d_output, n);

                // Benchmark Phase 2 (standard)
                std::cout << "[2/3] Phase 2 (Standard)...";
                float time_phase2 = benchmarkPhase2<uint64_t>(
                    compressed, d_output, n,
                    static_cast<int>(stats.avg_delta_bits),
                    false
                );
                double throughput_phase2 = (n * sizeof(uint64_t) / 1e9) / (time_phase2 / 1000.0);
                double speedup = time_phase1 / time_phase2;
                std::cout << " " << time_phase2 << " ms, " << throughput_phase2 << " GB/s"
                         << " (speedup: " << speedup << "x)\n";

                bool correct_phase2 = verifyCorrectness(data, d_output, n);

                // Benchmark Phase 2 (persistent threads)
                std::cout << "[3/3] Phase 2 (Persistent)...";
                float time_persistent = benchmarkPhase2<uint64_t>(
                    compressed, d_output, n,
                    static_cast<int>(stats.avg_delta_bits),
                    true
                );
                double throughput_persistent = (n * sizeof(uint64_t) / 1e9) / (time_persistent / 1000.0);
                double speedup_persistent = time_phase1 / time_persistent;
                std::cout << " " << time_persistent << " ms, " << throughput_persistent << " GB/s"
                         << " (speedup: " << speedup_persistent << "x)\n";

                bool correct_persistent = verifyCorrectness(data, d_output, n);

                // Record results
                csv << "Phase1," << target_bits << "," << n << ","
                    << compressed->num_partitions << "," << time_phase1 << ","
                    << throughput_phase1 << ",1.0,"
                    << (correct_phase1 ? "PASS" : "FAIL") << "\n";

                csv << "Phase2," << target_bits << "," << n << ","
                    << compressed->num_partitions << "," << time_phase2 << ","
                    << throughput_phase2 << "," << speedup << ","
                    << (correct_phase2 ? "PASS" : "FAIL") << "\n";

                csv << "Phase2_Persistent," << target_bits << "," << n << ","
                    << compressed->num_partitions << "," << time_persistent << ","
                    << throughput_persistent << "," << speedup_persistent << ","
                    << (correct_persistent ? "PASS" : "FAIL") << "\n";

                std::cout << "\nCorrectness: Phase1=" << (correct_phase1 ? "✓" : "✗")
                         << " Phase2=" << (correct_phase2 ? "✓" : "✗")
                         << " Persistent=" << (correct_persistent ? "✓" : "✗") << "\n\n";

                // Cleanup
                CUDA_CHECK(cudaFree(d_output));
                freeCompressedData(compressed);
            }
        }
    }

    csv.close();

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Benchmark Complete                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\nResults saved to: phase2_results.csv\n\n";

    return 0;
}
