/**
 * Test Suite for GPU-Parallel Merge Implementation
 *
 * This file tests the GPUCostOptimalPartitioner against the original
 * CPU-based implementation to ensure correctness.
 *
 * Tests:
 * 1. Basic correctness with various data patterns
 * 2. Edge cases (single partition, all merge, no merge)
 * 3. Performance comparison
 * 4. Large scale stress test
 *
 * Author: Claude Code Assistant
 * Date: 2025-12-06
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"

// ============================================================================
// Test Utilities
// ============================================================================

template<typename T>
std::vector<T> generateSortedData(size_t size, T start = 0, T step = 1) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = start + static_cast<T>(i) * step;
    }
    return data;
}

template<typename T>
std::vector<T> generateRandomData(size_t size, T min_val, T max_val, unsigned seed = 42) {
    std::vector<T> data(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateNearSortedData(size_t size, T noise_range = 10, unsigned seed = 42) {
    std::vector<T> data(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> noise(-static_cast<int>(noise_range), static_cast<int>(noise_range));
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(i * 100 + noise(gen));
    }
    return data;
}

template<typename T>
std::vector<T> generatePeriodicData(size_t size, size_t period = 1000, T amplitude = 1000) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        double phase = 2.0 * M_PI * i / period;
        data[i] = static_cast<T>(i + amplitude * std::sin(phase));
    }
    return data;
}

template<typename T>
std::vector<T> generateSegmentedData(size_t size, size_t num_segments = 10, unsigned seed = 42) {
    std::vector<T> data(size);
    std::mt19937 gen(seed);
    std::uniform_int_distribution<T> base_dist(0, 10000);

    size_t segment_size = size / num_segments;
    for (size_t seg = 0; seg < num_segments; seg++) {
        T base = base_dist(gen);
        T slope = (gen() % 100) - 50;  // Random slope
        size_t start = seg * segment_size;
        size_t end = (seg == num_segments - 1) ? size : start + segment_size;

        for (size_t i = start; i < end; i++) {
            data[i] = base + static_cast<T>((i - start) * slope);
        }
    }
    return data;
}

bool comparePartitions(const std::vector<PartitionInfo>& gpu,
                       const std::vector<PartitionInfo>& cpu,
                       bool verbose = false) {
    if (gpu.size() != cpu.size()) {
        if (verbose) {
            std::cout << "  Partition count mismatch: GPU=" << gpu.size()
                      << " CPU=" << cpu.size() << std::endl;
        }
        return false;
    }

    for (size_t i = 0; i < gpu.size(); i++) {
        if (gpu[i].start_idx != cpu[i].start_idx ||
            gpu[i].end_idx != cpu[i].end_idx) {
            if (verbose) {
                std::cout << "  Partition " << i << " boundary mismatch:" << std::endl;
                std::cout << "    GPU: [" << gpu[i].start_idx << ", " << gpu[i].end_idx << ")" << std::endl;
                std::cout << "    CPU: [" << cpu[i].start_idx << ", " << cpu[i].end_idx << ")" << std::endl;
            }
            return false;
        }
    }

    return true;
}

// ============================================================================
// Test Cases
// ============================================================================

template<typename T>
bool testSortedData(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing sorted data (" << size << " elements)... " << std::flush;

    auto data = generateSortedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testRandomData(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing random data (" << size << " elements)... " << std::flush;

    auto data = generateRandomData<T>(size, 0, 1000000);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testNearSortedData(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing near-sorted data (" << size << " elements)... " << std::flush;

    auto data = generateNearSortedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testPeriodicData(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing periodic data (" << size << " elements)... " << std::flush;

    auto data = generatePeriodicData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testSegmentedData(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing segmented data (" << size << " elements)... " << std::flush;

    auto data = generateSegmentedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testSmallData(bool verbose = true) {
    if (verbose) std::cout << "Testing small data (256 elements)... " << std::flush;

    auto data = generateSortedData<T>(256);
    CostOptimalConfig config;
    config.min_partition_size = 64;
    config.analysis_block_size = 64;

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testNoMergeConfig(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing with merging disabled (" << size << " elements)... " << std::flush;

    auto data = generateNearSortedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::highThroughput();
    config.enable_merging = false;

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

template<typename T>
bool testHighCompressionConfig(size_t size, bool verbose = true) {
    if (verbose) std::cout << "Testing high compression config (" << size << " elements)... " << std::flush;

    auto data = generateNearSortedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::highCompression();

    GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    bool passed = comparePartitions(gpu_result, cpu_result, verbose);

    if (verbose) {
        std::cout << (passed ? "PASSED" : "FAILED");
        std::cout << " (GPU: " << gpu_result.size() << ", CPU: " << cpu_result.size() << " partitions)" << std::endl;
    }

    return passed;
}

// ============================================================================
// Performance Test
// ============================================================================

template<typename T>
void runPerformanceTest(size_t size, int iterations = 5) {
    std::cout << "\n=== Performance Test (" << size << " elements, " << iterations << " iterations) ===" << std::endl;

    auto data = generateNearSortedData<T>(size);
    CostOptimalConfig config = CostOptimalConfig::balanced();

    // Warm up
    {
        GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
        auto result = gpu_partitioner.partition();
    }
    {
        auto result = createPartitionsCostOptimal(data, config, nullptr);
    }

    // GPU timing
    double gpu_total_ms = 0;
    int gpu_partitions = 0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
        auto result = gpu_partitioner.partition();

        auto end = std::chrono::high_resolution_clock::now();
        gpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        gpu_partitions = result.size();
    }

    // CPU timing
    double cpu_total_ms = 0;
    int cpu_partitions = 0;
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = createPartitionsCostOptimal(data, config, nullptr);

        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        cpu_partitions = result.size();
    }

    double gpu_avg = gpu_total_ms / iterations;
    double cpu_avg = cpu_total_ms / iterations;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GPU Merge: " << gpu_avg << " ms (avg), " << gpu_partitions << " partitions" << std::endl;
    std::cout << "CPU Merge: " << cpu_avg << " ms (avg), " << cpu_partitions << " partitions" << std::endl;
    std::cout << "Speedup: " << (cpu_avg / gpu_avg) << "x" << std::endl;
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "=== GPU Merge Implementation Test Suite ===" << std::endl;
    std::cout << std::endl;

    int passed = 0;
    int failed = 0;

    // Basic correctness tests
    std::cout << "--- Basic Correctness Tests (uint32_t) ---" << std::endl;

    if (testSmallData<uint32_t>()) passed++; else failed++;
    if (testSortedData<uint32_t>(10000)) passed++; else failed++;
    if (testRandomData<uint32_t>(10000)) passed++; else failed++;
    if (testNearSortedData<uint32_t>(10000)) passed++; else failed++;
    if (testPeriodicData<uint32_t>(10000)) passed++; else failed++;
    if (testSegmentedData<uint32_t>(10000)) passed++; else failed++;

    std::cout << std::endl;
    std::cout << "--- Configuration Tests (uint32_t) ---" << std::endl;

    if (testNoMergeConfig<uint32_t>(10000)) passed++; else failed++;
    if (testHighCompressionConfig<uint32_t>(10000)) passed++; else failed++;

    std::cout << std::endl;
    std::cout << "--- Large Scale Tests (uint32_t) ---" << std::endl;

    if (testSortedData<uint32_t>(100000)) passed++; else failed++;
    if (testNearSortedData<uint32_t>(100000)) passed++; else failed++;
    if (testSegmentedData<uint32_t>(100000)) passed++; else failed++;

    std::cout << std::endl;
    std::cout << "--- Different Data Types ---" << std::endl;

    std::cout << "int32_t: ";
    if (testNearSortedData<int32_t>(10000, false)) {
        std::cout << "PASSED" << std::endl;
        passed++;
    } else {
        std::cout << "FAILED" << std::endl;
        failed++;
    }

    std::cout << "int64_t: ";
    if (testNearSortedData<int64_t>(10000, false)) {
        std::cout << "PASSED" << std::endl;
        passed++;
    } else {
        std::cout << "FAILED" << std::endl;
        failed++;
    }

    std::cout << "uint64_t: ";
    if (testNearSortedData<uint64_t>(10000, false)) {
        std::cout << "PASSED" << std::endl;
        passed++;
    } else {
        std::cout << "FAILED" << std::endl;
        failed++;
    }

    // Summary
    std::cout << std::endl;
    std::cout << "=== Test Summary ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total:  " << (passed + failed) << std::endl;

    // Performance tests (optional)
    if (argc > 1 && std::string(argv[1]) == "--perf") {
        runPerformanceTest<uint32_t>(100000);
        runPerformanceTest<uint32_t>(1000000);
        runPerformanceTest<uint32_t>(10000000);
    }

    return (failed == 0) ? 0 : 1;
}
