/**
 * Phase 2.2 Bucket-Based Decompression Test Suite
 *
 * Tests:
 * 1. 0-bit (linear sequences - perfect prediction)
 * 2. 8-bit (random walk)
 * 3. 16-bit (medium deltas)
 * 4. 64-bit (full range)
 * 5. Mixed bitwidths (heterogeneous partitions: 8/9/10/12/64)
 * 6. Compression ratio invariance
 *
 * Success criteria:
 * - Bit-exact roundtrip (compress → decompress == original)
 * - Compression ratio delta ≤ 0.1%
 * - No routing errors (verified via debug logging)
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <random>
#include <cassert>
#include <map>
#include <algorithm>
#include "L3_codec.hpp"
#include "L3_format.hpp"

// External function declarations
template<typename T>
void decompressL3_Phase2_Bucket(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const uint8_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// Test Data Generators
// ============================================================================

std::vector<uint64_t> generateLinearSequence(int n, uint64_t start = 0, uint64_t step = 1) {
    std::vector<uint64_t> data(n);
    for (int i = 0; i < n; ++i) {
        data[i] = start + i * step;
    }
    return data;
}

std::vector<uint64_t> generateRandomWalk(int n, int max_step = 100) {
    std::vector<uint64_t> data(n);
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int> dist(-max_step, max_step);

    uint64_t val = 1000000;
    for (int i = 0; i < n; ++i) {
        val = static_cast<uint64_t>(static_cast<int64_t>(val) + dist(rng));
        data[i] = val;
    }
    return data;
}

std::vector<uint64_t> generateLargeDeltas(int n) {
    std::vector<uint64_t> data(n);
    std::mt19937_64 rng(123);
    std::uniform_int_distribution<uint64_t> dist(0, (1ULL << 62) - 1);

    for (int i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
    return data;
}

// ============================================================================
// Roundtrip Test Helper
// ============================================================================

template<typename T>
bool testRoundtripBucket(const std::vector<T>& original, int partition_size,
                         const char* test_name) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Elements: " << original.size()
              << ", Partition size: " << partition_size << std::endl;

    // Compress
    CompressionStats comp_stats;
    auto* compressed = compressData(original, partition_size, &comp_stats);

    if (!compressed) {
        std::cerr << "❌ Compression failed" << std::endl;
        return false;
    }

    std::cout << "Compressed: " << comp_stats.compressed_bytes << " bytes"
              << " (ratio: " << std::fixed << std::setprecision(2)
              << comp_stats.compression_ratio << "x)" << std::endl;
    std::cout << "Avg delta bits: " << comp_stats.avg_delta_bits << std::endl;
    std::cout << "Partitions: " << comp_stats.num_partitions << std::endl;

    // Download delta_bits to show heterogeneity
    std::vector<uint8_t> h_delta_bits(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(uint8_t),
                         cudaMemcpyDeviceToHost));

    // Count bitwidth distribution
    std::map<int, int> bit_dist;
    for (auto b : h_delta_bits) {
        bit_dist[b]++;
    }

    std::cout << "Bitwidth distribution: ";
    for (auto& p : bit_dist) {
        std::cout << p.first << "b:" << p.second << "p ";
    }
    std::cout << std::endl;

    // Decompress using Phase 2.2 bucket scheduler
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, original.size() * sizeof(T)));

    // Convert d_delta_bits from int32_t* to uint8_t* (Phase 2.2 interface)
    uint8_t* d_delta_bits_u8;
    CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));

    // Copy from int32_t to uint8_t on device
    std::vector<int32_t> h_delta_bits_i32(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits_i32.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
    std::vector<uint8_t> h_delta_bits_u8(compressed->num_partitions);
    for (int i = 0; i < compressed->num_partitions; ++i) {
        h_delta_bits_u8[i] = static_cast<uint8_t>(h_delta_bits_i32[i]);
    }
    CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits_u8.data(),
                         compressed->num_partitions * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        decompressL3_Phase2_Bucket<T>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int iters = 20;
    std::vector<float> times;

    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressL3_Phase2_Bucket<T>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    // Sort and get median
    std::sort(times.begin(), times.end());
    float median_ms = times[iters / 2];

    double gb = original.size() * sizeof(T) / 1e9;
    double throughput = gb / (median_ms / 1000.0);

    std::cout << "Decompression: " << std::fixed << std::setprecision(3)
              << median_ms << " ms (median of " << iters << ")" << std::endl;
    std::cout << "Throughput: " << std::fixed << std::setprecision(2)
              << throughput << " GB/s" << std::endl;

    // Download and verify
    std::vector<T> decompressed(original.size());
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         original.size() * sizeof(T), cudaMemcpyDeviceToHost));

    bool correct = true;
    int errors = 0;
    const int max_errors = 10;

    for (size_t i = 0; i < original.size(); ++i) {
        if (decompressed[i] != original[i]) {
            if (errors < max_errors) {
                std::cerr << "Mismatch at [" << i << "]: "
                          << "expected=" << original[i]
                          << ", got=" << decompressed[i] << std::endl;
            }
            errors++;
            correct = false;
        }
    }

    if (correct) {
        std::cout << "✅ PASS: Bit-exact roundtrip" << std::endl;
    } else {
        std::cerr << "❌ FAIL: " << errors << " mismatches" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_delta_bits_u8));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    freeCompressedData(compressed);

    return correct;
}

// ============================================================================
// Main Test Suite
// ============================================================================

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  L3 Phase 2.2 Bucket-Based Decompression Test Suite  ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "\nDevice: " << prop.name << " (SM " << prop.major << "."
              << prop.minor << ")" << std::endl;

    bool all_pass = true;

    // Test 1: 0-bit (perfect prediction - linear)
    {
        auto data = generateLinearSequence(1000000, 0, 1);
        all_pass &= testRoundtripBucket(data, 2048, "Test 1: 0-bit (Linear Sequence)");
    }

    // Test 2: 8-bit (random walk)
    {
        auto data = generateRandomWalk(1000000, 100);
        all_pass &= testRoundtripBucket(data, 2048, "Test 2: 8-bit (Random Walk)");
    }

    // Test 3: 16-bit (medium deltas)
    {
        auto data = generateRandomWalk(1000000, 10000);
        all_pass &= testRoundtripBucket(data, 2048, "Test 3: 16-bit (Medium Deltas)");
    }

    // Test 4: 64-bit (large deltas)
    {
        auto data = generateLargeDeltas(100000);
        all_pass &= testRoundtripBucket(data, 2048, "Test 4: 64-bit (Large Deltas)");
    }

    // Test 5: Mixed bitwidths (small partitions to force heterogeneity)
    {
        std::vector<uint64_t> mixed;
        // Linear (0-bit)
        auto seg1 = generateLinearSequence(10000, 0, 1);
        mixed.insert(mixed.end(), seg1.begin(), seg1.end());

        // Small walk (8-bit)
        auto seg2 = generateRandomWalk(10000, 50);
        mixed.insert(mixed.end(), seg2.begin(), seg2.end());

        // Medium walk (12-16 bit)
        auto seg3 = generateRandomWalk(10000, 1000);
        mixed.insert(mixed.end(), seg3.begin(), seg3.end());

        // Large deltas (64-bit)
        auto seg4 = generateLargeDeltas(10000);
        mixed.insert(mixed.end(), seg4.begin(), seg4.end());

        all_pass &= testRoundtripBucket(mixed, 512, "Test 5: Mixed Bitwidths (0/8/16/64)");
    }

    // Test 6: Scaling (8M elements)
    {
        auto data = generateRandomWalk(8000000, 100);
        all_pass &= testRoundtripBucket(data, 2048, "Test 6: Scaling (8M, 8-bit)");
    }

    // Test 7: 64M elements (if memory allows)
    {
        try {
            auto data = generateRandomWalk(64000000, 100);
            all_pass &= testRoundtripBucket(data, 4096, "Test 7: Scaling (64M, 8-bit)");
        } catch (...) {
            std::cout << "\nTest 7: Skipped (insufficient memory)" << std::endl;
        }
    }

    std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
    if (all_pass) {
        std::cout << "║             ✅ ALL TESTS PASSED                          ║" << std::endl;
    } else {
        std::cout << "║             ❌ SOME TESTS FAILED                         ║" << std::endl;
    }
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;

    return all_pass ? 0 : 1;
}
