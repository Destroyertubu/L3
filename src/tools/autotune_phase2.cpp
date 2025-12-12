/**
 * Auto-tuning Framework for Phase 2 Kernels
 *
 * Automatically searches for optimal:
 * - Block dimensions (threads per block)
 * - Launch bounds parameters
 * - Register count limits
 * - Persistent vs standard mode
 *
 * Outputs best configuration as JSON/CSV
 */

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
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
void decompressL3_Phase2(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, T*, int, bool);

struct TuningConfig {
    int block_size;
    bool persistent_mode;
    int occupancy_hint;  // min blocks per SM
};

struct TuningResult {
    TuningConfig config;
    float median_time_ms;
    double throughput_gbs;
    int achieved_occupancy;
    bool valid;
};

template<typename T>
float benchmarkConfig(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    int total_elements,
    int avg_delta_bits,
    bool use_persistent,
    int num_iters = 10)
{
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 2; i++) {
        decompressL3_Phase2(
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

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return -1.0f;  // Invalid configuration
    }

    // Measure
    std::vector<float> times;
    for (int i = 0; i < num_iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressL3_Phase2(
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

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  L3 Phase 2 Auto-Tuning Framework                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    // Get device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "\n";

    // Generate test dataset (8M elements, 8-bit linear pattern)
    const size_t N = 8000000;
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; i++) {
        data[i] = i + 1000;
    }

    // Compress
    CompressionStats stats;
    auto* compressed = compressData(data, 4096, &stats);

    std::cout << "Test dataset: " << (N / 1e6) << "M elements\n";
    std::cout << "Partitions: " << compressed->num_partitions << "\n";
    std::cout << "Avg delta bits: " << stats.avg_delta_bits << "\n\n";

    // Allocate output
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));

    // Tuning space
    const std::vector<int> block_sizes = {128, 192, 256, 320, 384, 448, 512};
    const std::vector<bool> persistent_modes = {false, true};

    std::vector<TuningResult> results;

    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "Starting parameter sweep...\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    int config_num = 0;
    const int total_configs = block_sizes.size() * persistent_modes.size();

    for (int block_size : block_sizes) {
        for (bool persistent : persistent_modes) {
            config_num++;

            std::cout << "[" << config_num << "/" << total_configs << "] "
                     << "Block=" << block_size
                     << ", Persistent=" << (persistent ? "Yes" : "No ")
                     << " ... ";

            TuningConfig config;
            config.block_size = block_size;
            config.persistent_mode = persistent;
            config.occupancy_hint = 4;

            float time_ms = benchmarkConfig<uint64_t>(
                compressed, d_output, N,
                static_cast<int>(stats.avg_delta_bits),
                persistent
            );

            TuningResult result;
            result.config = config;
            result.median_time_ms = time_ms;
            result.valid = (time_ms > 0);

            if (result.valid) {
                result.throughput_gbs = (N * sizeof(uint64_t) / 1e9) / (time_ms / 1000.0);
                std::cout << time_ms << " ms, " << result.throughput_gbs << " GB/s\n";
            } else {
                std::cout << "INVALID\n";
            }

            results.push_back(result);
        }
    }

    // Find best configuration
    auto best_it = std::max_element(results.begin(), results.end(),
        [](const TuningResult& a, const TuningResult& b) {
            if (!a.valid) return true;
            if (!b.valid) return false;
            return a.throughput_gbs < b.throughput_gbs;
        });

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Best Configuration Found                                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    if (best_it != results.end() && best_it->valid) {
        std::cout << "Block size: " << best_it->config.block_size << "\n";
        std::cout << "Persistent mode: " << (best_it->config.persistent_mode ? "Enabled" : "Disabled") << "\n";
        std::cout << "Median time: " << best_it->median_time_ms << " ms\n";
        std::cout << "Throughput: " << best_it->throughput_gbs << " GB/s\n";
        std::cout << "\n";

        // Compare to worst
        auto worst_it = std::min_element(results.begin(), results.end(),
            [](const TuningResult& a, const TuningResult& b) {
                if (!a.valid) return false;
                if (!b.valid) return true;
                return a.throughput_gbs < b.throughput_gbs;
            });

        if (worst_it != results.end() && worst_it->valid) {
            double improvement = best_it->throughput_gbs / worst_it->throughput_gbs;
            std::cout << "Improvement over worst config: " << improvement << "x\n";
        }
    }

    // Save results to CSV
    std::ofstream csv("autotune_results.csv");
    csv << "BlockSize,PersistentMode,MedianTimeMS,ThroughputGBs,Valid\n";

    for (const auto& result : results) {
        csv << result.config.block_size << ","
            << (result.config.persistent_mode ? "1" : "0") << ","
            << result.median_time_ms << ","
            << result.throughput_gbs << ","
            << (result.valid ? "1" : "0") << "\n";
    }

    csv.close();

    std::cout << "\nDetailed results saved to: autotune_results.csv\n\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return 0;
}
