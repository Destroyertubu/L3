/**
 * F3 Regression Test: 8-bit Random Walk Scenario
 *
 * This test verifies that the F1 fix (weighted avg_delta_bits) correctly
 * computes ~8 bits for random walk data, instead of the buggy ~50 bits.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include "L3_codec.hpp"
#include "L3_format.hpp"

// Forward declaration
template<typename T>
void decompressGLECO_Phase2(
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
    int avg_delta_bits,
    bool use_persistent);

int main() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  F3: 8-bit Random Walk Regression Test                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    // Generate 8-bit random walk data
    const size_t N = 1000000;  // 1M elements
    std::vector<uint32_t> data(N);

    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> walk_dist(-10, 10);

    data[0] = 50000;
    for (size_t i = 1; i < N; i++) {
        int delta = walk_dist(rng);
        data[i] = static_cast<uint32_t>(static_cast<int>(data[i-1]) + delta);
    }

    std::cout << "Generated " << N << " random walk values\n";
    std::cout << "Sample: " << data[0] << ", " << data[1] << ", " << data[2] << ", ..., " << data[N-1] << "\n\n";

    // Compress with codec (uses optimized encoder internally)
    CompressionStats stats;
    auto compressed = compressData(data, 2048, &stats);

    std::cout << "Compression Stats:\n";
    std::cout << "  Partitions: " << stats.num_partitions << "\n";
    std::cout << "  avg_delta_bits (WEIGHTED): " << stats.avg_delta_bits << "\n";
    std::cout << "  Compression ratio: " << stats.compression_ratio << "x\n";
    std::cout << "  Total bits used: " << stats.total_bits_used << "\n\n";

    // F1 VERIFICATION: avg_delta_bits should be 7-9 for 8-bit random walk
    if (stats.avg_delta_bits < 7 || stats.avg_delta_bits > 9) {
        std::cout << "❌ F1 FAIL: avg_delta_bits = " << stats.avg_delta_bits
                  << " (expected 7-9 for 8-bit random walk)\n\n";

        // Debug: print actual delta_bits for each partition
        std::vector<int32_t> h_delta_bits(stats.num_partitions);
        std::vector<int32_t> h_start(stats.num_partitions);
        std::vector<int32_t> h_end(stats.num_partitions);

        cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                   stats.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_start.data(), compressed->d_start_indices,
                   stats.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end.data(), compressed->d_end_indices,
                   stats.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::cout << "First 10 partitions:\n";
        for (int p = 0; p < std::min(10, stats.num_partitions); p++) {
            int len = h_end[p] - h_start[p];
            std::cout << "  P" << p << ": bits=" << h_delta_bits[p]
                      << " len=" << len << "\n";
        }

        freeCompressedData(compressed);
        return 1;
    }

    std::cout << "✓ F1 PASS: avg_delta_bits is correct (" << stats.avg_delta_bits << " bits)\n\n";

    // F2 VERIFICATION: Decompress using Phase 2 kernel (should route to 8-bit kernel)
    uint32_t* d_output;
    cudaMalloc(&d_output, N * sizeof(uint32_t));

    std::cout << "Testing Phase 2 decompression (should use 8-bit kernel)...\n";

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
        stats.avg_delta_bits,
        false  // No persistent threads (F4)
    );

    cudaDeviceSynchronize();

    // Verify correctness
    std::vector<uint32_t> result(N);
    cudaMemcpy(result.data(), d_output, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int mismatches = 0;
    for (size_t i = 0; i < N; i++) {
        if (result[i] != data[i]) {
            if (mismatches < 5) {
                std::cout << "  Mismatch at " << i << ": expected " << data[i]
                          << ", got " << result[i] << "\n";
            }
            mismatches++;
        }
    }

    if (mismatches > 0) {
        std::cout << "❌ F2/F3 FAIL: " << mismatches << " decompression errors\n\n";
        cudaFree(d_output);
        freeCompressedData(compressed);
        return 1;
    }

    std::cout << "✓ F2/F3 PASS: All values decompressed correctly\n\n";

    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  ✓ ALL TESTS PASSED                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    cudaFree(d_output);
    freeCompressedData(compressed);
    return 0;
}
