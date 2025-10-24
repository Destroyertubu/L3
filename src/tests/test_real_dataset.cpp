/**
 * Test Phase 2 with real 200M linear dataset
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>
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
    for (int i = 0; i < 5; i++) {
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

std::vector<uint32_t> loadDataset(const std::string& filename, size_t max_elements = 0) {
    std::cout << "Loading dataset: " << filename << "\n";

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << "\n";
        exit(1);
    }

    std::vector<uint32_t> data;
    data.reserve(200000000);  // Reserve for 200M

    uint32_t value;
    size_t count = 0;
    while (file >> value) {
        data.push_back(value);
        count++;
        if (max_elements > 0 && count >= max_elements) break;

        if (count % 10000000 == 0) {
            std::cout << "  Loaded " << (count / 1000000) << "M elements...\n";
        }
    }

    std::cout << "✓ Loaded " << data.size() << " elements\n";
    std::cout << "  First 10: ";
    for (int i = 0; i < std::min(10, (int)data.size()); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n";
    std::cout << "  Last 10: ";
    for (int i = std::max(0, (int)data.size() - 10); i < data.size(); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n\n";

    return data;
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GLECO Phase 2 - Real Dataset Performance Test              ║\n";
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

    // Load dataset
    auto data = loadDataset("/root/autodl-tmp/test/data/linear_200M_uint32.txt");
    const size_t N = data.size();

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Compressing...\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Test with different partition sizes
    std::vector<int> partition_sizes = {2048, 4096, 8192};

    for (int part_size : partition_sizes) {
        std::cout << "Partition size: " << part_size << "\n";
        std::cout << "─────────────────────────────────────────\n";

        // Compress
        CompressionStats stats;
        auto start_compress = std::chrono::high_resolution_clock::now();
        auto* compressed = compressData(data, part_size, &stats);
        auto end_compress = std::chrono::high_resolution_clock::now();

        double compress_time_ms = std::chrono::duration<double, std::milli>(
            end_compress - start_compress).count();

        std::cout << "  Compression time: " << compress_time_ms << " ms\n";
        std::cout << "  Partitions: " << compressed->num_partitions << "\n";
        std::cout << "  Avg delta bits: " << stats.avg_delta_bits << "\n";
        std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n";
        std::cout << "  Compressed size: "
                 << (compressed->delta_array_words * 4.0 / (1024*1024)) << " MB\n";
        std::cout << "  Original size: " << (N * sizeof(uint32_t) / (1024.0*1024)) << " MB\n";
        std::cout << "\n";

        // Allocate output
        uint32_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint32_t)));

        // Benchmark Phase 2
        std::cout << "  Benchmarking Phase 2 decompression (20 iterations)...\n";

        float time_ms = benchmarkPhase2<uint32_t>(
            compressed, d_output,
            static_cast<int>(stats.avg_delta_bits),
            false,  // standard mode
            20
        );

        double data_size_gb = (N * sizeof(uint32_t)) / 1e9;
        double throughput = data_size_gb / (time_ms / 1000.0);
        double efficiency = (throughput / peak_bandwidth) * 100.0;

        std::cout << "\n";
        std::cout << "  ┌─────────────────────────────────────────┐\n";
        std::cout << "  │ Performance Results                     │\n";
        std::cout << "  ├─────────────────────────────────────────┤\n";
        printf("  │ Median Time:     %12.6f ms      │\n", time_ms);
        printf("  │ Throughput:      %12.2f GB/s    │\n", throughput);
        printf("  │ Peak Efficiency: %12.2f %%      │\n", efficiency);
        printf("  │ Elements/sec:    %12.2f B/s     │\n", (N / (time_ms / 1000.0)) / 1e9);
        std::cout << "  └─────────────────────────────────────────┘\n";
        std::cout << "\n";

        // Verify correctness (sample check)
        std::cout << "  Verifying correctness (sampling)...\n";
        std::vector<uint32_t> result(N);
        CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                             N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        int errors = 0;
        const int sample_stride = N / 1000;  // Check 1000 points
        for (size_t i = 0; i < N && errors < 10; i += sample_stride) {
            if (result[i] != data[i]) {
                if (errors < 5) {
                    std::cout << "    ERROR at " << i << ": expected "
                             << data[i] << ", got " << result[i] << "\n";
                }
                errors++;
            }
        }

        if (errors == 0) {
            std::cout << "  ✅ Correctness: PASS (sampled " << (N/sample_stride) << " points)\n";
        } else {
            std::cout << "  ❌ Correctness: FAIL (" << errors << " errors found)\n";
        }

        std::cout << "\n";

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));
        freeCompressedData(compressed);
    }

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Test Complete                                               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return 0;
}
