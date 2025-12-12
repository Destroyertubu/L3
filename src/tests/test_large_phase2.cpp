/**
 * Debug test for large dataset Phase 2
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
void decompressL3_Phase2(
    const int32_t*, const int32_t*, const int32_t*, const double*,
    const int32_t*, const int64_t*, const uint32_t*, int, int, T*, int, bool);

int main() {
    std::cout << "Large Dataset Phase 2 Test\n\n";

    // Test with 2M elements (same as failing test)
    const size_t N = 2000000;
    std::vector<uint64_t> data(N);
    for (size_t i = 0; i < N; i++) {
        data[i] = i * 2 + 123456;
    }

    std::cout << "Data: " << (N / 1e6) << "M elements\n";
    std::cout << "First 10: ";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << "\n\n";

    // Compress
    CompressionStats stats;
    auto* compressed = compressData(data, 4096, &stats);

    std::cout << "Compressed:\n";
    std::cout << "  Partitions: " << compressed->num_partitions << "\n";
    std::cout << "  Total elements: " << compressed->total_values << "\n";
    std::cout << "  Avg delta bits: " << stats.avg_delta_bits << "\n";
    std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n";
    std::cout << "  Delta array words: " << compressed->delta_array_words << "\n\n";

    // Check first few partitions
    std::vector<int32_t> h_start(compressed->num_partitions);
    std::vector<int32_t> h_end(compressed->num_partitions);
    std::vector<int32_t> h_delta_bits(compressed->num_partitions);

    CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    std::cout << "First 5 partitions:\n";
    for (int i = 0; i < std::min(5, compressed->num_partitions); i++) {
        std::cout << "  " << i << ": [" << h_start[i] << ", " << h_end[i]
                 << ") = " << (h_end[i] - h_start[i]) << " elements, "
                 << h_delta_bits[i] << " bits\n";
    }
    std::cout << "\n";

    // Allocate output
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_output, 0xFF, N * sizeof(uint64_t)));  // Fill with 0xFF to detect untouched

    // Decompress
    std::cout << "Decompressing with Phase 2...\n";

    decompressL3_Phase2(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->delta_array,
        compressed->num_partitions,
        compressed->total_values,  // Use actual total from compressed structure
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

    std::cout << "First 10 decompressed: ";
    for (int i = 0; i < 10; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n";

    std::cout << "Last 10 decompressed: ";
    for (int i = N-10; i < N; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << "\n\n";

    // Check for errors
    int errors = 0;
    int zeros = 0;
    int untouched = 0;
    for (size_t i = 0; i < N; i++) {
        if (result[i] == 0) zeros++;
        if (result[i] == 0xFFFFFFFFFFFFFFFF) untouched++;
        if (result[i] != data[i]) {
            if (errors < 10) {
                std::cout << "ERROR at " << i << ": expected " << data[i]
                         << ", got " << result[i] << "\n";
            }
            errors++;
        }
    }

    std::cout << "\nStatistics:\n";
    std::cout << "  Total errors: " << errors << "\n";
    std::cout << "  Zero values: " << zeros << "\n";
    std::cout << "  Untouched (0xFF...): " << untouched << "\n";

    if (errors == 0) {
        std::cout << "\n✓ All values correct!\n";
    } else {
        std::cout << "\n✗ Test failed\n";
    }

    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return errors == 0 ? 0 : 1;
}
