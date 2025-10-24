/**
 * Simple diagnostic test for Phase 2
 */

#include <iostream>
#include <vector>
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

int main() {
    std::cout << "Simple Phase 2 Test\n\n";

    // Generate simple linear data with non-zero deltas
    const size_t N = 100000;
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; i++) {
        data[i] = i * 100 + 50000;  // Ensure non-zero deltas
    }

    std::cout << "Data: " << N << " elements\n";
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n\n";

    // Compress
    CompressionStats stats;
    auto* compressed = compressData(data, 4096, &stats);

    std::cout << "Compressed:\n";
    std::cout << "  Partitions: " << compressed->num_partitions << "\n";
    std::cout << "  Avg delta bits: " << stats.avg_delta_bits << "\n";
    std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n\n";

    // Check actual delta bits per partition
    std::vector<int32_t> h_delta_bits(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    std::cout << "Delta bits per partition (first 10):\n";
    for (int i = 0; i < std::min(10, compressed->num_partitions); i++) {
        std::cout << "  Partition " << i << ": " << h_delta_bits[i] << " bits\n";
    }
    std::cout << "\n";

    // Allocate output
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_output, 0, N * sizeof(uint64_t)));

    // Decompress using Phase 2
    std::cout << "Decompressing with Phase 2...\n";

    decompressGLECO_Phase2(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->delta_array,
        compressed->num_partitions,
        N,
        d_output,
        static_cast<int>(stats.avg_delta_bits),
        false
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    // Verify
    std::vector<uint64_t> result(N);
    CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                         N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    std::cout << "First 5 decompressed values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n\n";

    // Check for correctness
    int errors = 0;
    for (size_t i = 0; i < N; i++) {
        if (result[i] != data[i]) {
            if (errors < 10) {
                std::cout << "ERROR at " << i << ": expected " << data[i]
                         << ", got " << result[i] << "\n";
            }
            errors++;
        }
    }

    if (errors == 0) {
        std::cout << "✓ All values correct!\n";
    } else {
        std::cout << "✗ " << errors << " errors found\n";
    }

    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return errors == 0 ? 0 : 1;
}
