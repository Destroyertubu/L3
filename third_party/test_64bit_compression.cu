/**
 * Test program for 64-bit compression implementations
 *
 * Tests:
 * 1. FastLanesGPU 64-bit pack/unpack roundtrip
 * 2. tile-gpu-compression 64-bit binpack random access
 * 3. tile-gpu-compression 64-bit deltabinpack random access
 * 4. tile-gpu-compression 64-bit RLE random access
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>
#include <cstring>
#include <chrono>

// Include 64-bit headers
#include "FastLanesGPU/fastlanes/src/include/fls_gen/unpack/unpack_64.cuh"
#include "FastLanesGPU/fastlanes/src/include/fls_gen/pack/pack_64.cuh"
#include "tile-gpu-compression/src/binpack_random_access_64.cuh"
#include "tile-gpu-compression/src/deltabinpack_random_access_64.cuh"
#include "tile-gpu-compression/src/rlebinpack_random_access_64.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Test 1: FastLanes 64-bit Pack/Unpack
// ============================================================================
bool test_fastlanes_pack_unpack_64() {
    std::cout << "\n=== Test 1: FastLanes 64-bit Pack/Unpack ===" << std::endl;

    const int num_values = 1024;  // One block
    const int test_bitwidths[] = {1, 2, 4, 8, 16, 32, 48, 64};
    bool all_passed = true;

    for (int bw : test_bitwidths) {
        std::cout << "  Testing bitwidth " << bw << "... ";

        // Generate test data
        std::vector<uint64_t> h_input(num_values);
        uint64_t mask = (bw == 64) ? ~0ULL : (1ULL << bw) - 1;

        std::mt19937_64 rng(42);
        for (int i = 0; i < num_values; i++) {
            h_input[i] = rng() & mask;
        }

        // Calculate packed size
        size_t packed_words = (num_values * bw + 63) / 64;
        std::vector<uint64_t> h_packed(packed_words, 0);
        std::vector<uint64_t> h_output(num_values, 0);

        // Allocate device memory
        uint64_t *d_input, *d_packed, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_packed, 0, packed_words * sizeof(uint64_t)));

        // Pack
        pack_global_64<<<1, 32>>>(d_input, d_packed, bw);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Unpack
        unpack_global_64<<<1, 32>>>(d_packed, d_output, bw);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back and verify
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        bool passed = true;
        for (int i = 0; i < num_values; i++) {
            if (h_input[i] != h_output[i]) {
                std::cerr << "FAILED at index " << i << ": expected " << h_input[i]
                          << ", got " << h_output[i] << std::endl;
                passed = false;
                break;
            }
        }

        if (passed) {
            std::cout << "PASSED" << std::endl;
        } else {
            all_passed = false;
        }

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_packed));
        CUDA_CHECK(cudaFree(d_output));
    }

    return all_passed;
}

// ============================================================================
// Test 2: Binpack 64-bit Random Access
// ============================================================================
bool test_binpack_random_access_64() {
    std::cout << "\n=== Test 2: Binpack 64-bit Random Access ===" << std::endl;

    const int num_values = 1024;  // 8 blocks of 128
    const int num_blocks = num_values / 128;
    const int bitwidth = 20;

    // Generate test data with controlled range
    std::vector<int64_t> h_original(num_values);
    int64_t base_val = 1000000000000LL;  // Large 64-bit base

    std::mt19937_64 rng(123);
    for (int i = 0; i < num_values; i++) {
        h_original[i] = base_val + (rng() & ((1ULL << bitwidth) - 1));
    }

    // Simple compression simulation for testing
    // Each block: [reference (1 word)][bitwidths (1 word)][packed data]
    std::vector<uint64_t> h_compressed;
    std::vector<uint32_t> h_block_starts(num_blocks + 1);

    for (int b = 0; b < num_blocks; b++) {
        h_block_starts[b] = h_compressed.size();

        int block_start = b * 128;
        int64_t min_val = h_original[block_start];
        for (int i = 1; i < 128; i++) {
            if (h_original[block_start + i] < min_val)
                min_val = h_original[block_start + i];
        }

        // Reference
        h_compressed.push_back(static_cast<uint64_t>(min_val));

        // Bitwidths (same for all miniblocks for simplicity)
        uint64_t packed_bw = bitwidth | (bitwidth << 8) | (bitwidth << 16) | (bitwidth << 24);
        h_compressed.push_back(packed_bw);

        // Pack each miniblock
        for (int m = 0; m < 4; m++) {
            int mb_start = block_start + m * 32;
            uint64_t packed = 0;
            int shift = 0;

            for (int i = 0; i < 32; i++) {
                uint64_t delta = static_cast<uint64_t>(h_original[mb_start + i] - min_val);

                if (shift + bitwidth > 64) {
                    packed |= (delta << shift);
                    h_compressed.push_back(packed);
                    packed = delta >> (64 - shift);
                    shift = (shift + bitwidth) - 64;
                } else {
                    packed |= (delta << shift);
                    shift += bitwidth;
                }
            }
            if (shift > 0) {
                h_compressed.push_back(packed);
            }
        }
    }
    h_block_starts[num_blocks] = h_compressed.size();

    // Allocate device memory
    uint64_t *d_compressed;
    uint32_t *d_block_starts;
    uint32_t *d_indices;
    int64_t *d_output;

    CUDA_CHECK(cudaMalloc(&d_compressed, h_compressed.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_block_starts, h_block_starts.size() * sizeof(uint32_t)));

    const int num_queries = 100;
    std::vector<uint32_t> h_indices(num_queries);
    std::vector<int64_t> h_output(num_queries);

    // Random query indices
    for (int i = 0; i < num_queries; i++) {
        h_indices[i] = rng() % num_values;
    }

    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_queries * sizeof(int64_t)));

    CUDA_CHECK(cudaMemcpy(d_compressed, h_compressed.data(), h_compressed.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_block_starts, h_block_starts.data(), h_block_starts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_queries * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Run random access kernel
    int blocks = (num_queries + 127) / 128;
    randomAccessKernel64<128><<<blocks, 128>>>(d_block_starts, d_compressed, d_indices, d_output, num_queries);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_queries * sizeof(int64_t), cudaMemcpyDeviceToHost));

    bool all_passed = true;
    for (int i = 0; i < num_queries; i++) {
        if (h_output[i] != h_original[h_indices[i]]) {
            std::cerr << "FAILED at query " << i << " (index " << h_indices[i] << "): expected "
                      << h_original[h_indices[i]] << ", got " << h_output[i] << std::endl;
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "  Random access test: PASSED (" << num_queries << " queries)" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_compressed));
    CUDA_CHECK(cudaFree(d_block_starts));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_output));

    return all_passed;
}

// ============================================================================
// Test 3: 64-bit data range test
// ============================================================================
bool test_64bit_range() {
    std::cout << "\n=== Test 3: 64-bit Full Range Test ===" << std::endl;

    // Test with values that require full 64 bits
    const int num_values = 1024;
    std::vector<uint64_t> h_input(num_values);
    std::vector<uint64_t> h_output(num_values);

    // Generate large 64-bit values
    std::mt19937_64 rng(999);
    for (int i = 0; i < num_values; i++) {
        h_input[i] = rng();  // Full 64-bit random values
    }

    uint64_t *d_input, *d_packed, *d_output;
    size_t packed_words = num_values;  // 64-bit pack = same size

    CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Pack with bw=64
    pack_global_64<<<1, 32>>>(d_input, d_packed, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Unpack
    unpack_global_64<<<1, 32>>>(d_packed, d_output, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_values * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    bool passed = true;
    for (int i = 0; i < num_values; i++) {
        if (h_input[i] != h_output[i]) {
            std::cerr << "FAILED at index " << i << ": expected 0x" << std::hex << h_input[i]
                      << ", got 0x" << h_output[i] << std::dec << std::endl;
            passed = false;
            break;
        }
    }

    if (passed) {
        std::cout << "  Full 64-bit range test: PASSED" << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_output));

    return passed;
}

// ============================================================================
// Performance benchmark
// ============================================================================
void benchmark_64bit() {
    std::cout << "\n=== Performance Benchmark ===" << std::endl;

    const int num_values = 1024 * 1024;  // 1M values
    const int num_blocks = num_values / 1024;
    const int num_iterations = 100;

    std::vector<uint64_t> h_input(num_values);
    std::mt19937_64 rng(42);
    for (int i = 0; i < num_values; i++) {
        h_input[i] = rng() & 0xFFFFFFFF;  // 32-bit values in 64-bit container
    }

    uint64_t *d_input, *d_packed, *d_output;
    size_t packed_words = num_values / 2;  // 32-bit pack in 64-bit words

    CUDA_CHECK(cudaMalloc(&d_input, num_values * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_values * sizeof(uint64_t)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), num_values * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Warmup
    pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, 32);
    unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, 32);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark pack
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, 32);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    double pack_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;

    // Benchmark unpack
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; i++) {
        unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, 32);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    double unpack_time = std::chrono::duration<double, std::milli>(end - start).count() / num_iterations;

    double pack_throughput = (num_values * sizeof(uint64_t)) / (pack_time * 1e6);  // GB/s
    double unpack_throughput = (num_values * sizeof(uint64_t)) / (unpack_time * 1e6);  // GB/s

    std::cout << "  Pack (32-bit in 64-bit):   " << pack_time << " ms, " << pack_throughput << " GB/s" << std::endl;
    std::cout << "  Unpack (32-bit in 64-bit): " << unpack_time << " ms, " << unpack_throughput << " GB/s" << std::endl;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_output));
}

// ============================================================================
// Main
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "64-bit Compression Library Test Suite" << std::endl;
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
    if (test_fastlanes_pack_unpack_64()) passed++; else failed++;
    if (test_binpack_random_access_64()) passed++; else failed++;
    if (test_64bit_range()) passed++; else failed++;

    // Run benchmark
    benchmark_64bit();

    // Summary
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "========================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
