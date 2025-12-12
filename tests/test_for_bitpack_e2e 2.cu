/**
 * End-to-End Test for FOR+BitPacking
 *
 * Verifies that data encoded with FOR+BitPacking decodes correctly.
 *
 * Date: 2025-12-05
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

template<typename T>
bool testFORBitPackRoundtrip(const std::string& test_name, const std::vector<T>& data) {
    std::cout << "Testing: " << test_name << " (n=" << data.size() << ")... ";

    // Encode
    VerticalConfig config;
    config.partition_size_hint = 2048;

    // Create partitions first
    auto partitions = Vertical_encoder::createFixedPartitions<T>(
        data.size(), config.partition_size_hint);

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);

    // Check that some partitions use FOR_BITPACK
    int for_count = 0;
    int linear_count = 0;
    for (const auto& p : partitions) {
        if (p.model_type == MODEL_FOR_BITPACK) {
            for_count++;
        } else if (p.model_type == MODEL_LINEAR) {
            linear_count++;
        }
    }

    // Decode
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    Vertical_decoder::decompressAll<T>(
        compressed, d_output, DecompressMode::BRANCHLESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back
    std::vector<T> output(data.size());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data.size() * sizeof(T), cudaMemcpyDeviceToHost));

    // Verify
    int errors = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != output[i]) {
            errors++;
            if (errors <= 5) {
                std::cout << "\n  Error at " << i << ": expected " << data[i]
                          << ", got " << output[i];
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    if (errors == 0) {
        std::cout << "PASSED (FOR=" << for_count << ", LINEAR=" << linear_count << ")\n";
        return true;
    } else {
        std::cout << "FAILED (" << errors << " errors)\n";
        return false;
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "FOR+BitPacking End-to-End Test\n";
    std::cout << "===========================================\n";

    int passed = 0;
    int total = 0;

    // Test 1: Constant data (should use FOR with bits=0)
    {
        std::vector<int32_t> data(10000, 12345);
        if (testFORBitPackRoundtrip("constant_i32", data)) passed++;
        total++;
    }

    // Test 2: Small range random (should use FOR)
    {
        std::vector<int32_t> data(10000);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int32_t> dist(1000, 1100);
        for (auto& v : data) v = dist(gen);
        if (testFORBitPackRoundtrip("small_range_random_i32", data)) passed++;
        total++;
    }

    // Test 3: Clustered data (should use FOR)
    {
        std::vector<int32_t> data(10000);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int32_t> dist(50000, 50200);
        for (auto& v : data) v = dist(gen);
        if (testFORBitPackRoundtrip("clustered_i32", data)) passed++;
        total++;
    }

    // Test 4: Full range random (should use FOR)
    {
        std::vector<int32_t> data(10000);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int32_t> dist(0, INT32_MAX/2);
        for (auto& v : data) v = dist(gen);
        if (testFORBitPackRoundtrip("full_range_random_i32", data)) passed++;
        total++;
    }

    // Test 5: Linear data (should use LINEAR)
    {
        std::vector<int32_t> data(10000);
        for (size_t i = 0; i < data.size(); i++) {
            data[i] = 1000 + static_cast<int32_t>(i);
        }
        if (testFORBitPackRoundtrip("linear_i32", data)) passed++;
        total++;
    }

    // Test 6: Mixed partitions (some FOR, some LINEAR)
    {
        std::vector<int32_t> data(20000);
        // First half: linear
        for (size_t i = 0; i < 10000; i++) {
            data[i] = 1000 + static_cast<int32_t>(i);
        }
        // Second half: random small range
        std::mt19937 gen(42);
        std::uniform_int_distribution<int32_t> dist(50000, 50100);
        for (size_t i = 10000; i < 20000; i++) {
            data[i] = dist(gen);
        }
        if (testFORBitPackRoundtrip("mixed_i32", data)) passed++;
        total++;
    }

    // Test 7: 64-bit constant
    {
        std::vector<int64_t> data(10000, 123456789012345LL);
        if (testFORBitPackRoundtrip("constant_i64", data)) passed++;
        total++;
    }

    // Test 8: 64-bit small range
    {
        std::vector<int64_t> data(10000);
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(1000000000000LL, 1000000001000LL);
        for (auto& v : data) v = dist(gen);
        if (testFORBitPackRoundtrip("small_range_i64", data)) passed++;
        total++;
    }

    std::cout << "\n===========================================\n";
    std::cout << "SUMMARY: " << passed << "/" << total << " tests passed\n";
    std::cout << "===========================================\n";

    return (passed == total) ? 0 : 1;
}
