/**
 * Quick test of baseline and optimized decompression
 * Uses small dataset for rapid verification
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
void launchDecompressWarpOpt(
    const CompressedDataL3<T>* compressed,
    T* d_output,
    cudaStream_t stream
);

int main() {
    std::cout << "Quick decompression test\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━\n\n";

    // Generate small linear dataset (1M elements)
    const size_t N = 1000000;
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; i++) {
        data[i] = i * 3 + 1000;  // Linear pattern
    }

    std::cout << "Dataset: " << N << " elements\n";

    // Compress
    CompressionStats comp_stats;
    auto* compressed = compressData(data, 2048, &comp_stats);

    std::cout << "Compressed: " << compressed->num_partitions << " partitions\n";
    std::cout << "Compression ratio: " << comp_stats.compression_ratio << "x\n";
    std::cout << "Avg delta bits: " << comp_stats.avg_delta_bits << "\n\n";

    // Allocate output buffer
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));

    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; i++) {
        launchDecompressWarpOpt(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Measure
    const int iters = 10;
    std::vector<float> times;

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        launchDecompressWarpOpt(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&time_ms, start, stop));
        times.push_back(time_ms);
    }

    // Sort to get median
    std::sort(times.begin(), times.end());
    float median_time = times[iters / 2];

    // Calculate throughput
    double data_size_gb = (N * sizeof(uint64_t)) / 1e9;
    double throughput = data_size_gb / (median_time / 1000.0);

    std::cout << "Decompression Results:\n";
    std::cout << "  Median kernel time: " << median_time << " ms\n";
    std::cout << "  Throughput: " << throughput << " GB/s\n\n";

    // Verify correctness
    std::vector<uint64_t> decompressed(N);
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    bool correct = true;
    for (size_t i = 0; i < std::min(N, size_t(1000)); i++) {
        if (decompressed[i] != data[i]) {
            std::cerr << "Mismatch at " << i << ": "
                     << decompressed[i] << " != " << data[i] << "\n";
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ Correctness check passed\n";
    } else {
        std::cout << "✗ Correctness check FAILED\n";
        return 1;
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    std::cout << "\n✓ Test completed successfully\n";

    return 0;
}
