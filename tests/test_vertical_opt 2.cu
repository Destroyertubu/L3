/**
 * L3 Vertical-Optimized Compression Test Suite
 *
 * Tests for the Vertical-optimized encoder and decoder:
 * 1. Correctness verification (roundtrip encode/decode)
 * 2. Random access verification
 * 3. Performance benchmarking vs original implementation
 *
 * Datasets:
 * - Linear data (best case for L3)
 * - Random data (stress test)
 * - Real-world patterns (mixed)
 *
 * Platform: SM 8.0+
 * Date: 2025-12-04
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

// ============================================================================
// Test Utilities
// ============================================================================

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
struct TestResult {
    bool passed;
    int errors;
    double max_error;
    double encode_time_ms;
    double decode_time_ms;
    double random_access_time_us;
    double compression_ratio;
    double encode_throughput_gbps;
    double decode_throughput_gbps;
};

// ============================================================================
// Data Generators
// ============================================================================

template<typename T>
std::vector<T> generateLinearData(size_t n, T start = 1000000, T slope = 1) {
    std::vector<T> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = start + static_cast<T>(i) * slope;
    }
    return data;
}

template<typename T>
std::vector<T> generateLinearWithNoise(size_t n, T start = 1000000, T slope = 1, T noise_range = 10) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> noise(-noise_range, noise_range);

    for (size_t i = 0; i < n; i++) {
        data[i] = start + static_cast<T>(i) * slope + noise(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateRandomData(size_t n, T min_val = 0, T max_val = 1000000) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> dist(min_val, max_val);

    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generatePiecewiseLinear(size_t n, int num_segments = 10) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> base_dist(0, 1000000);
    std::uniform_int_distribution<int> slope_dist(-100, 100);

    size_t segment_size = n / num_segments;
    for (int s = 0; s < num_segments; s++) {
        T base = base_dist(gen);
        int slope = slope_dist(gen);
        size_t start = s * segment_size;
        size_t end = (s == num_segments - 1) ? n : (s + 1) * segment_size;

        for (size_t i = start; i < end; i++) {
            data[i] = base + static_cast<T>((i - start) * slope);
        }
    }
    return data;
}

// ============================================================================
// Correctness Tests
// ============================================================================

template<typename T>
TestResult<T> testRoundtrip(
    const std::vector<T>& data,
    const VerticalConfig& config,
    const std::string& test_name)
{
    TestResult<T> result;
    result.passed = true;
    result.errors = 0;
    result.max_error = 0.0;

    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Data size: " << data.size() << " elements" << std::endl;

    // Create partitions
    auto partitions = Vertical_encoder::createFixedPartitions<T>(
        data.size(), config.partition_size_hint);

    std::cout << "Partitions: " << partitions.size() << std::endl;

    // Encode
    auto encode_start = std::chrono::high_resolution_clock::now();

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);

    auto encode_end = std::chrono::high_resolution_clock::now();
    result.encode_time_ms = std::chrono::duration<double, std::milli>(
        encode_end - encode_start).count();

    // Calculate compression ratio
    size_t original_size = data.size() * sizeof(T);
    size_t compressed_size = compressed.sequential_delta_words * sizeof(uint32_t);
    if (config.enable_interleaved) {
        compressed_size += compressed.interleaved_delta_words * sizeof(uint32_t);
    }
    result.compression_ratio = static_cast<double>(original_size) / compressed_size;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << result.compression_ratio << "x" << std::endl;
    std::cout << "Encode time: " << result.encode_time_ms << " ms" << std::endl;

    // Allocate output
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Decode - Sequential path
    auto decode_start = std::chrono::high_resolution_clock::now();

    Vertical_decoder::decompressAll<T>(
        compressed, d_output, DecompressMode::SEQUENTIAL);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto decode_end = std::chrono::high_resolution_clock::now();
    result.decode_time_ms = std::chrono::duration<double, std::milli>(
        decode_end - decode_start).count();

    std::cout << "Decode time (sequential): " << result.decode_time_ms << " ms" << std::endl;

    // Copy back and verify
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data.size() * sizeof(T),
                          cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.errors++;
            double error = std::abs(static_cast<double>(data[i]) - static_cast<double>(decoded[i]));
            result.max_error = std::max(result.max_error, error);

            if (result.errors <= 5) {
                std::cout << "  Error at index " << i << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
        }
    }

    if (result.errors > 0) {
        result.passed = false;
        std::cout << "FAILED: " << result.errors << " errors, max error = "
                  << result.max_error << std::endl;
    } else {
        std::cout << "PASSED: All values match" << std::endl;
    }

    // Calculate throughput
    result.encode_throughput_gbps = (original_size / 1e9) / (result.encode_time_ms / 1000.0);
    result.decode_throughput_gbps = (original_size / 1e9) / (result.decode_time_ms / 1000.0);

    std::cout << "Encode throughput: " << std::fixed << std::setprecision(2)
              << result.encode_throughput_gbps << " GB/s" << std::endl;
    std::cout << "Decode throughput: " << std::fixed << std::setprecision(2)
              << result.decode_throughput_gbps << " GB/s" << std::endl;

    // Test interleaved path if enabled
    if (config.enable_interleaved && compressed.total_interleaved_partitions > 0) {
        CUDA_CHECK(cudaMemset(d_output, 0, data.size() * sizeof(T)));

        auto int_decode_start = std::chrono::high_resolution_clock::now();

        Vertical_decoder::decompressAll<T>(
            compressed, d_output, DecompressMode::INTERLEAVED);

        CUDA_CHECK(cudaDeviceSynchronize());
        auto int_decode_end = std::chrono::high_resolution_clock::now();

        double int_decode_time_ms = std::chrono::duration<double, std::milli>(
            int_decode_end - int_decode_start).count();

        std::cout << "Decode time (interleaved): " << int_decode_time_ms << " ms" << std::endl;
        std::cout << "Interleaved partitions: " << compressed.total_interleaved_partitions << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    return result;
}

// ============================================================================
// Random Access Test
// ============================================================================

template<typename T>
double testRandomAccess(
    const std::vector<T>& data,
    const VerticalConfig& config,
    int num_queries = 10000)
{
    std::cout << "\n=== Random Access Test ===" << std::endl;

    // Encode
    auto partitions = Vertical_encoder::createFixedPartitions<T>(
        data.size(), config.partition_size_hint);
    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);

    // Generate random query indices
    std::vector<int> h_indices(num_queries);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, data.size() - 1);
    for (int i = 0; i < num_queries; i++) {
        h_indices[i] = dist(gen);
    }

    // Copy indices to device
    int* d_indices;
    T* d_results;
    CUDA_CHECK(cudaMalloc(&d_indices, num_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, num_queries * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), num_queries * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < 3; i++) {
        Vertical_decoder::decompressIndices<T>(
            compressed, d_indices, num_queries, d_results);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_trials = 10;
    auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < num_trials; t++) {
        Vertical_decoder::decompressIndices<T>(
            compressed, d_indices, num_queries, d_results);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    double total_time_us = std::chrono::duration<double, std::micro>(end - start).count();
    double time_per_query_ns = (total_time_us * 1000.0) / (num_trials * num_queries);

    std::cout << "Random access queries: " << num_queries << std::endl;
    std::cout << "Time per query: " << std::fixed << std::setprecision(1)
              << time_per_query_ns << " ns" << std::endl;

    // Verify correctness
    std::vector<T> h_results(num_queries);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, num_queries * sizeof(T),
                          cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < num_queries; i++) {
        if (h_results[i] != data[h_indices[i]]) {
            errors++;
            if (errors <= 3) {
                std::cout << "  Error at query " << i << ": index=" << h_indices[i]
                          << ", expected " << data[h_indices[i]]
                          << ", got " << h_results[i] << std::endl;
            }
        }
    }

    if (errors > 0) {
        std::cout << "Random access FAILED: " << errors << " errors" << std::endl;
    } else {
        std::cout << "Random access PASSED" << std::endl;
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_results));
    Vertical_encoder::freeCompressedData(compressed);

    return time_per_query_ns;
}

// ============================================================================
// Benchmark Suite
// ============================================================================

template<typename T>
void runBenchmarks(size_t data_size = 10000000) {  // 10M elements
    std::cout << "\n========================================" << std::endl;
    std::cout << "L3 Vertical Optimization Benchmarks" << std::endl;
    std::cout << "Data size: " << data_size << " elements (" << (data_size * sizeof(T) / 1e6) << " MB)" << std::endl;
    std::cout << "Element type: " << (sizeof(T) == 4 ? "int32" : "int64") << std::endl;
    std::cout << "========================================" << std::endl;

    VerticalConfig config;
    config.partition_size_hint = 4096;
    config.enable_interleaved = true;
    config.enable_branchless_unpack = true;

    // Test 1: Linear data (best case)
    {
        auto data = generateLinearData<T>(data_size);
        testRoundtrip<T>(data, config, "Linear Data (Best Case)");
    }

    // Test 2: Linear with noise
    {
        auto data = generateLinearWithNoise<T>(data_size, 1000000, 1, 100);
        testRoundtrip<T>(data, config, "Linear Data with Noise");
    }

    // Test 3: Piecewise linear
    {
        auto data = generatePiecewiseLinear<T>(data_size, 100);
        testRoundtrip<T>(data, config, "Piecewise Linear (100 segments)");
    }

    // Test 4: Random data (worst case)
    {
        auto data = generateRandomData<T>(data_size, static_cast<T>(0), static_cast<T>(1000000));
        testRoundtrip<T>(data, config, "Random Data (Stress Test)");
    }

    // Test 5: Random access performance
    {
        auto data = generateLinearWithNoise<T>(data_size, 1000000, 1, 50);
        testRandomAccess<T>(data, config, 100000);
    }
}

// ============================================================================
// Compare with Sequential-Only Mode
// ============================================================================

template<typename T>
void compareWithSequentialOnly(size_t data_size = 5000000) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Comparison: Vertical vs Sequential-Only" << std::endl;
    std::cout << "========================================" << std::endl;

    auto data = generateLinearWithNoise<T>(data_size, 1000000, 1, 50);

    // Vertical config
    VerticalConfig fl_config;
    fl_config.enable_interleaved = true;
    fl_config.enable_branchless_unpack = true;

    // Sequential-only config
    VerticalConfig seq_config = VerticalConfig::sequentialOnly();

    std::cout << "\n--- Vertical Optimized ---" << std::endl;
    auto fl_result = testRoundtrip<T>(data, fl_config, "Vertical Mode");

    std::cout << "\n--- Sequential Only ---" << std::endl;
    auto seq_result = testRoundtrip<T>(data, seq_config, "Sequential Mode");

    std::cout << "\n=== Comparison Summary ===" << std::endl;
    std::cout << "Decode speedup: " << std::fixed << std::setprecision(2)
              << (seq_result.decode_time_ms / fl_result.decode_time_ms) << "x" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize CUDA
    int device = 0;
    cudaSetDevice(device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Parse arguments
    size_t data_size = 10000000;  // Default: 10M
    if (argc > 1) {
        data_size = std::atol(argv[1]);
    }

    // Run benchmarks
    std::cout << "\n*** Testing with int64_t ***" << std::endl;
    runBenchmarks<int64_t>(data_size);

    std::cout << "\n*** Testing with int32_t ***" << std::endl;
    runBenchmarks<int32_t>(data_size);

    // Compare modes
    compareWithSequentialOnly<int64_t>(data_size / 2);

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
