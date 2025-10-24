/**
 * Standalone Phase 2 Performance Benchmark
 * Tests Phase 2 kernels with real data patterns
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "L3_codec.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

template<typename T>
void decompressGLECO_Phase2(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, T*, int, bool);

template<typename T>
float benchmarkPhase2(
    const CompressedDataGLECO<T>* compressed,
    T* d_output,
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
            compressed->total_values,
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
            compressed->total_values,
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

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GLECO Phase 2 Standalone Performance Benchmark             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute: SM " << prop.major << "." << prop.minor << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    double peak_bandwidth = (prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) * 2) / 1e9;
    std::cout << "Peak Memory Bandwidth: " << peak_bandwidth << " GB/s\n";
    std::cout << "\n";

    // Test configurations
    struct TestConfig {
        size_t size;
        std::string name;
        int step;  // For linear pattern
    };

    std::vector<TestConfig> configs = {
        {1000000, "1M", 100},
        {8000000, "8M", 100},
        {64000000, "64M", 100}
    };

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Running benchmarks (linear data, perfect prediction)...\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    for (const auto& config : configs) {
        std::cout << "Dataset: " << config.name << " elements\n";
        std::cout << "─────────────────────────────────────────\n";

        // Generate linear data (perfect prediction -> 0-bit deltas)
        std::vector<uint64_t> data(config.size);
        for (size_t i = 0; i < config.size; i++) {
            data[i] = i * config.step + 50000;
        }

        // Compress
        CompressionStats stats;
        auto* compressed = compressData(data, 4096, &stats);

        std::cout << "  Partitions: " << compressed->num_partitions << "\n";
        std::cout << "  Avg delta bits: " << stats.avg_delta_bits << "\n";
        std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n\n";

        // Allocate output
        uint64_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, config.size * sizeof(uint64_t)));

        // Benchmark standard mode
        float time_standard = benchmarkPhase2<uint64_t>(
            compressed, d_output,
            static_cast<int>(stats.avg_delta_bits),
            false
        );

        double data_size_gb = (config.size * sizeof(uint64_t)) / 1e9;
        double throughput_standard = data_size_gb / (time_standard / 1000.0);

        std::cout << "  Phase 2 (Standard):    " << time_standard << " ms, "
                 << throughput_standard << " GB/s ("
                 << (throughput_standard / peak_bandwidth * 100.0) << "% of peak)\n";

        // Verify correctness
        std::vector<uint64_t> result(config.size);
        CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                             config.size * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool correct = true;
        for (size_t i = 0; i < std::min(config.size, size_t(1000)); i++) {
            if (result[i] != data[i]) {
                correct = false;
                break;
            }
        }

        std::cout << "  Correctness:           " << (correct ? "✓ PASS" : "✗ FAIL") << "\n";
        std::cout << "\n";

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));
        freeCompressedData(compressed);
    }

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Testing with non-zero deltas...\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Test with actual deltas (random walk)
    {
        const size_t N = 8000000;
        std::cout << "Dataset: 8M elements (random walk, 8-bit deltas)\n";
        std::cout << "─────────────────────────────────────────\n";

        std::vector<uint64_t> data(N);
        data[0] = 1000000;
        srand(42);
        for (size_t i = 1; i < N; i++) {
            int delta = (rand() % 256) - 128;
            data[i] = data[i-1] + delta;
        }

        CompressionStats stats;
        auto* compressed = compressData(data, 4096, &stats);

        std::cout << "  Partitions: " << compressed->num_partitions << "\n";
        std::cout << "  Avg delta bits: " << stats.avg_delta_bits << "\n";
        std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n\n";

        uint64_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));

        float time_standard = benchmarkPhase2<uint64_t>(
            compressed, d_output,
            static_cast<int>(stats.avg_delta_bits),
            false
        );

        double data_size_gb = (N * sizeof(uint64_t)) / 1e9;
        double throughput = data_size_gb / (time_standard / 1000.0);

        std::cout << "  Phase 2 (Standard):    " << time_standard << " ms, "
                 << throughput << " GB/s ("
                 << (throughput / peak_bandwidth * 100.0) << "% of peak)\n";

        // Verify
        std::vector<uint64_t> result(N);
        CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                             N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (size_t i = 0; i < N && errors < 10; i++) {
            if (result[i] != data[i]) {
                errors++;
            }
        }

        std::cout << "  Correctness:           " << (errors == 0 ? "✓ PASS" : "✗ FAIL") << "\n";
        std::cout << "\n";

        CUDA_CHECK(cudaFree(d_output));
        freeCompressedData(compressed);
    }

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Benchmark Complete                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
