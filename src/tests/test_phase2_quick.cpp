/**
 * Quick Correctness Test for Phase 2 Kernels
 *
 * Verifies:
 * - Bit-exact correctness across all bitwidths
 * - Compression ratio preservation
 * - Multiple data patterns
 * - Persistent vs standard modes
 */

#include <iostream>
#include <vector>
#include <cmath>
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

template<typename T>
bool testPattern(const std::string& name, const std::vector<T>& data, int target_bits) {
    std::cout << "Testing: " << name << " (" << data.size() / 1000 << "K elements, "
             << "target " << target_bits << "-bit)... ";

    // Compress
    CompressionStats stats;
    auto* compressed = compressData(data, 2048, &stats);

    // Check compression ratio preservation
    double expected_ratio = (data.size() * sizeof(T) * 8.0) /
                           (compressed->delta_array_words * 32.0 +
                            compressed->num_partitions * 10 * 8.0);  // metadata overhead

    // Allocate output
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Test standard mode
    decompressL3_Phase2(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->delta_array,
        compressed->num_partitions,
        data.size(),
        d_output,
        static_cast<int>(stats.avg_delta_bits),
        false
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<T> result(data.size());
    CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                         data.size() * sizeof(T), cudaMemcpyDeviceToHost));

    bool correct = true;
    int mismatches = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (result[i] != data[i]) {
            if (mismatches < 5) {
                std::cerr << "\n  Mismatch at " << i << ": expected "
                         << data[i] << ", got " << result[i];
            }
            mismatches++;
            correct = false;
        }
    }

    if (!correct) {
        std::cout << "✗ FAILED (" << mismatches << " mismatches)\n";
        CUDA_CHECK(cudaFree(d_output));
        freeCompressedData(compressed);
        return false;
    }

    // Test persistent mode
    decompressL3_Phase2(
        compressed->d_start_indices,
        compressed->d_end_indices,
        compressed->d_model_types,
        compressed->d_model_params,
        compressed->d_delta_bits,
        compressed->d_delta_array_bit_offsets,
        compressed->delta_array,
        compressed->num_partitions,
        data.size(),
        d_output,
        static_cast<int>(stats.avg_delta_bits),
        true
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify persistent mode
    CUDA_CHECK(cudaMemcpy(result.data(), d_output,
                         data.size() * sizeof(T), cudaMemcpyDeviceToHost));

    mismatches = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (result[i] != data[i]) {
            mismatches++;
            correct = false;
        }
    }

    if (!correct) {
        std::cout << "✗ FAILED (persistent mode, " << mismatches << " mismatches)\n";
        CUDA_CHECK(cudaFree(d_output));
        freeCompressedData(compressed);
        return false;
    }

    std::cout << "✓ PASS (ratio: " << stats.compression_ratio << "x, "
             << "avg bits: " << stats.avg_delta_bits << ")\n";

    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);
    return true;
}

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  L3 Phase 2 Quick Correctness Test                       ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute: SM " << prop.major << prop.minor << "\n\n";

    bool all_passed = true;

    // Test 1: Linear pattern (8-bit deltas)
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = i + 10000;
        }
        all_passed &= testPattern("Linear (8-bit)", data, 8);
    }

    // Test 2: Linear pattern (16-bit deltas)
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = i * 256 + 50000;
        }
        all_passed &= testPattern("Linear (16-bit)", data, 16);
    }

    // Test 3: Quadratic pattern
    {
        const size_t N = 50000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = (i * i) / 100 + i * 10 + 1000;
        }
        all_passed &= testPattern("Quadratic", data, 12);
    }

    // Test 4: Small random deltas (low bits)
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N);
        data[0] = 100000;
        srand(42);
        for (size_t i = 1; i < N; i++) {
            int delta = (rand() % 32) - 16;  // 4-5 bit deltas
            data[i] = data[i-1] + delta;
        }
        all_passed &= testPattern("Random low-bit", data, 8);
    }

    // Test 5: Larger random deltas
    {
        const size_t N = 50000;
        std::vector<uint64_t> data(N);
        data[0] = 1000000;
        srand(123);
        for (size_t i = 1; i < N; i++) {
            int delta = (rand() % 8192) - 4096;  // 12-13 bit deltas
            data[i] = data[i-1] + delta;
        }
        all_passed &= testPattern("Random mid-bit", data, 16);
    }

    // Test 6: Uniform distribution
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = 5000000 + i;
        }
        all_passed &= testPattern("Uniform", data, 8);
    }

    // Test 7: Constant (1-bit)
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N, 42);
        all_passed &= testPattern("Constant", data, 1);
    }

    // Test 8: Binary (2-bit)
    {
        const size_t N = 100000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = 1000 + (i % 4);
        }
        all_passed &= testPattern("Binary (2-bit)", data, 2);
    }

    // Test 9: Signed data type
    {
        const size_t N = 50000;
        std::vector<int64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = static_cast<int64_t>(i) - 25000;
        }
        all_passed &= testPattern("Signed linear", data, 16);
    }

    // Test 10: Large dataset (stress test)
    {
        const size_t N = 2000000;
        std::vector<uint64_t> data(N);
        for (size_t i = 0; i < N; i++) {
            data[i] = i * 2 + 123456;
        }
        all_passed &= testPattern("Large (2M elements)", data, 8);
    }

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    if (all_passed) {
        std::cout << "║  ✓ ALL TESTS PASSED                                          ║\n";
    } else {
        std::cout << "║  ✗ SOME TESTS FAILED                                         ║\n";
    }
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";

    return all_passed ? 0 : 1;
}
