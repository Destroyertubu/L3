/**
 * Simplified Test for 64-bit Compression Implementations
 *
 * Tests:
 * 1. FastLanesGPU 64-bit pack/unpack roundtrip (all bitwidths)
 * 2. Basic functionality verification
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <chrono>

// Include 64-bit headers
#include "FastLanesGPU/fastlanes/src/include/fls_gen/unpack/unpack_64.cuh"
#include "FastLanesGPU/fastlanes/src/include/fls_gen/pack/pack_64.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Test FastLanes 64-bit Pack/Unpack for all bitwidths
bool test_all_bitwidths() {
    std::cout << "\n=== Testing All Bitwidths (0-64) ===" << std::endl;

    const int num_values = 1024;  // One block
    bool all_passed = true;
    int passed_count = 0;

    for (int bw = 0; bw <= 64; bw++) {
        // Generate test data
        std::vector<uint64_t> h_input(num_values);
        uint64_t mask = (bw == 0) ? 0 : ((bw == 64) ? ~0ULL : (1ULL << bw) - 1);

        std::mt19937_64 rng(42 + bw);
        for (int i = 0; i < num_values; i++) {
            h_input[i] = (bw == 0) ? 0 : (rng() & mask);
        }

        // Calculate packed size
        size_t packed_words = (bw == 0) ? 1 : ((num_values * bw + 63) / 64);
        std::vector<uint64_t> h_output(num_values, 0xDEADBEEF);

        // Allocate device memory
        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, std::max((size_t)1, packed_words) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        // Copy input to device and initialize output
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_packed, 0, std::max((size_t)1, packed_words) * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset(d_output, 0xFF, num_values * sizeof(uint64_t)));

        // Pack
        pack_global_64<<<1, 32>>>(d_input, d_packed, bw);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Unpack
        unpack_global_64<<<1, 32>>>(d_packed, d_output, bw);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back and verify
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        int first_fail = -1;
        uint64_t expected_val = 0, got_val = 0;

        for (int i = 0; i < num_values; i++) {
            uint64_t expected = (bw == 0) ? 0 : (h_input[i] & mask);
            if (h_output[i] != expected) {
                if (first_fail < 0) {
                    first_fail = i;
                    expected_val = expected;
                    got_val = h_output[i];
                }
                passed = false;
            }
        }

        if (passed) {
            passed_count++;
        } else {
            std::cout << "  Bitwidth " << bw << ": FAILED at index " << first_fail
                      << " (expected 0x" << std::hex << expected_val
                      << ", got 0x" << got_val << std::dec << ")" << std::endl;
            all_passed = false;
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    std::cout << "  Result: " << passed_count << "/65 bitwidths passed" << std::endl;
    return all_passed;
}

// Test multiple blocks
bool test_multiple_blocks() {
    std::cout << "\n=== Testing Multiple Blocks ===" << std::endl;

    const int num_blocks = 100;
    const int num_values = num_blocks * 1024;  // 100 blocks
    const int test_bitwidths[] = {8, 16, 32, 48};
    bool all_passed = true;

    for (int bw : test_bitwidths) {
        std::vector<uint64_t> h_input(num_values);
        uint64_t mask = (1ULL << bw) - 1;

        std::mt19937_64 rng(12345);
        for (int i = 0; i < num_values; i++) {
            h_input[i] = rng() & mask;
        }

        size_t packed_words = (num_values * bw + 63) / 64;
        std::vector<uint64_t> h_output(num_values, 0);

        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_packed, 0, packed_words * sizeof(uint64_t)));

        pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bw);
        CUDA_CHECK(cudaDeviceSynchronize());

        unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bw);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < num_values; i++) {
            if (h_input[i] != h_output[i]) {
                std::cerr << "  Bitwidth " << bw << ": FAILED at index " << i << std::endl;
                passed = false;
                break;
            }
        }

        if (passed) {
            std::cout << "  Bitwidth " << bw << " (" << num_blocks << " blocks): PASSED" << std::endl;
        } else {
            all_passed = false;
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    return all_passed;
}

// Performance benchmark
void benchmark() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

    const int num_values = 1024 * 1024;  // 1M values
    const int num_blocks = num_values / 1024;
    const int num_iterations = 100;
    const int test_bitwidths[] = {8, 16, 32, 48, 64};

    for (int bw : test_bitwidths) {
        std::vector<uint64_t> h_input(num_values);
        std::mt19937_64 rng(42);
        uint64_t mask = (bw == 64) ? ~0ULL : (1ULL << bw) - 1;

        for (int i = 0; i < num_values; i++) {
            h_input[i] = rng() & mask;
        }

        uint64_t *d_input, *d_packed, *d_output;
        size_t packed_words = (num_values * bw + 63) / 64;

        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // Warmup
        pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bw);
        unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bw);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark pack
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bw);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double pack_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;

        // Benchmark unpack
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; i++) {
            unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bw);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        end = std::chrono::high_resolution_clock::now();
        double unpack_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;

        double input_size_gb = (num_values * sizeof(uint64_t)) / 1e9;
        double pack_throughput = input_size_gb / (pack_time / 1000.0);
        double unpack_throughput = input_size_gb / (unpack_time / 1000.0);

        std::cout << "  Bitwidth " << bw << ": Pack " << pack_throughput << " GB/s, Unpack " << unpack_throughput << " GB/s" << std::endl;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }
}

// Test edge cases
bool test_edge_cases() {
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;

    bool all_passed = true;
    const int num_values = 1024;

    // Test 1: All zeros
    {
        std::vector<uint64_t> h_input(num_values, 0);
        std::vector<uint64_t> h_output(num_values, 0xDEADBEEF);

        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_packed, 0, num_values * sizeof(uint64_t)));

        pack_global_64<<<1, 32>>>(d_input, d_packed, 32);
        unpack_global_64<<<1, 32>>>(d_packed, d_output, 32);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < num_values; i++) {
            if (h_output[i] != 0) {
                passed = false;
                break;
            }
        }
        std::cout << "  All zeros: " << (passed ? "PASSED" : "FAILED") << std::endl;
        all_passed &= passed;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    // Test 2: All max values
    {
        std::vector<uint64_t> h_input(num_values, ~0ULL);
        std::vector<uint64_t> h_output(num_values, 0);

        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));

        pack_global_64<<<1, 32>>>(d_input, d_packed, 64);
        unpack_global_64<<<1, 32>>>(d_packed, d_output, 64);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < num_values; i++) {
            if (h_output[i] != ~0ULL) {
                passed = false;
                break;
            }
        }
        std::cout << "  All max values: " << (passed ? "PASSED" : "FAILED") << std::endl;
        all_passed &= passed;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    // Test 3: Large 64-bit values
    {
        std::vector<uint64_t> h_input(num_values);
        for (int i = 0; i < num_values; i++) {
            h_input[i] = 0x8000000000000000ULL + i;  // Values near INT64_MAX
        }
        std::vector<uint64_t> h_output(num_values, 0);

        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));

        pack_global_64<<<1, 32>>>(d_input, d_packed, 64);
        unpack_global_64<<<1, 32>>>(d_packed, d_output, 64);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < num_values; i++) {
            if (h_output[i] != h_input[i]) {
                passed = false;
                break;
            }
        }
        std::cout << "  Large 64-bit values: " << (passed ? "PASSED" : "FAILED") << std::endl;
        all_passed &= passed;

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    return all_passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "64-bit FastLanes Pack/Unpack Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    // Check CUDA device
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "\nUsing device: " << prop.name << std::endl;

    int passed = 0;
    int failed = 0;

    // Run tests
    if (test_all_bitwidths()) passed++; else failed++;
    if (test_multiple_blocks()) passed++; else failed++;
    if (test_edge_cases()) passed++; else failed++;

    // Run benchmark
    benchmark();

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << passed << " test groups passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
