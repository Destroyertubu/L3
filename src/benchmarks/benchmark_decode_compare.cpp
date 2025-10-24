/**
 * GLECO Decompression: Baseline vs Optimized Comparison
 *
 * Compares performance of baseline and optimized kernels
 * Reports speedup and validates correctness
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>
#include "L3_codec.hpp"

// Forward declarations
template<typename T>
void launchDecompressWarpOpt(
    const CompressedDataGLECO<T>* compressed,
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

// Forward declaration of optimized decompression
template<typename T>
void decompressGLECO_Optimized(
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

template<typename T>
std::vector<T> generateLinearData(size_t num_elements, int target_bits) {
    std::vector<T> data(num_elements);
    T max_delta = (1ULL << std::min(target_bits, 32)) - 1;
    T value = 1000;

    for (size_t i = 0; i < num_elements; i++) {
        data[i] = value;
        value += (i % 7) + 1;
    }
    return data;
}

struct CompareResult {
    int bit_width;
    size_t num_elements;
    double baseline_time_ms;
    double optimized_time_ms;
    double speedup;
    bool correctness_passed;

    void print() const {
        std::cout << std::setw(6) << bit_width << " | "
                  << std::setw(12) << num_elements << " | "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << baseline_time_ms << " | "
                  << std::setw(8) << std::fixed << std::setprecision(3)
                  << optimized_time_ms << " | "
                  << std::setw(6) << std::fixed << std::setprecision(2)
                  << speedup << "x | "
                  << (correctness_passed ? "PASS" : "FAIL") << std::endl;
    }
};

template<typename T>
bool verifyResults(const T* d_baseline, const T* d_optimized, size_t num_elements) {
    std::vector<T> h_baseline(num_elements);
    std::vector<T> h_optimized(num_elements);

    CUDA_CHECK(cudaMemcpy(h_baseline.data(), d_baseline,
                         num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_optimized.data(), d_optimized,
                         num_elements * sizeof(T), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < num_elements; i++) {
        if (h_baseline[i] != h_optimized[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                     << h_baseline[i] << " != " << h_optimized[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
CompareResult compareImplementations(
    const std::vector<T>& data,
    int partition_size,
    int target_bits)
{
    CompareResult result;
    result.bit_width = target_bits;
    result.num_elements = data.size();

    // Compress once
    CompressionStats stats;
    auto* compressed = compressData(data, partition_size, &stats);

    // Allocate output buffers
    T *d_baseline_out, *d_optimized_out;
    CUDA_CHECK(cudaMalloc(&d_baseline_out, data.size() * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_optimized_out, data.size() * sizeof(T)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int warmup = 3;
    const int iters = 15;

    // Baseline benchmark
    for (int i = 0; i < warmup; i++) {
        launchDecompressWarpOpt(compressed, d_baseline_out, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    std::vector<float> baseline_times;
    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launchDecompressWarpOpt(compressed, d_baseline_out, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        baseline_times.push_back(time_ms);
    }

    std::sort(baseline_times.begin(), baseline_times.end());
    result.baseline_time_ms = baseline_times[iters / 2];

    // Optimized benchmark (only for 8/16 bit)
    if (target_bits == 8 || target_bits == 16) {
        for (int i = 0; i < warmup; i++) {
            decompressGLECO_Optimized(
                compressed->d_start_indices,
                compressed->d_end_indices,
                compressed->d_model_types,
                compressed->d_model_params,
                compressed->d_delta_bits,
                compressed->d_delta_array_bit_offsets,
                compressed->delta_array,
                compressed->num_partitions,
                compressed->total_values,
                d_optimized_out,
                static_cast<int>(stats.avg_delta_bits)
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> opt_times;
        for (int i = 0; i < iters; i++) {
            CUDA_CHECK(cudaEventRecord(start));
            decompressGLECO_Optimized(
                compressed->d_start_indices,
                compressed->d_end_indices,
                compressed->d_model_types,
                compressed->d_model_params,
                compressed->d_delta_bits,
                compressed->d_delta_array_bit_offsets,
                compressed->delta_array,
                compressed->num_partitions,
                compressed->total_values,
                d_optimized_out,
                static_cast<int>(stats.avg_delta_bits)
            );
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float time_ms;
            CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
            opt_times.push_back(time_ms);
        }

        std::sort(opt_times.begin(), opt_times.end());
        result.optimized_time_ms = opt_times[iters / 2];
        result.speedup = result.baseline_time_ms / result.optimized_time_ms;

        // Verify correctness
        result.correctness_passed = verifyResults(d_baseline_out, d_optimized_out,
                                                 data.size());
    } else {
        result.optimized_time_ms = result.baseline_time_ms;
        result.speedup = 1.0;
        result.correctness_passed = true;
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_baseline_out));
    CUDA_CHECK(cudaFree(d_optimized_out));
    freeCompressedData(compressed);

    return result;
}

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  GLECO Decompression: Baseline vs Optimized                 ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    std::vector<int> bit_widths = {8, 16};
    std::vector<size_t> sizes = {10000000, 50000000, 200000000};
    int partition_size = 2048;

    std::ofstream csv("optimization_results.csv");
    csv << "BitWidth,NumElements,BaselineTime_ms,OptimizedTime_ms,Speedup,Correctness\n";

    for (size_t size : sizes) {
        std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
        std::cout << "Size: " << size << " elements\n";
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n";
        std::cout << " Bits  | Elements      | Baseline | Optimized | Speedup | Status\n";
        std::cout << "───────┼───────────────┼──────────┼───────────┼─────────┼───────\n";

        for (int bits : bit_widths) {
            auto data = generateLinearData<uint64_t>(size, bits);
            auto result = compareImplementations(data, partition_size, bits);
            result.print();

            csv << result.bit_width << ","
                << result.num_elements << ","
                << result.baseline_time_ms << ","
                << result.optimized_time_ms << ","
                << result.speedup << ","
                << (result.correctness_passed ? "PASS" : "FAIL") << "\n";
        }
    }

    csv.close();
    std::cout << "\n✓ Results saved to: optimization_results.csv\n";

    return 0;
}
