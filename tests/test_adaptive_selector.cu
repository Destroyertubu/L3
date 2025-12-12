/**
 * L3 Adaptive Model Selector Test Suite
 *
 * Tests for the adaptive model selector that chooses between LINEAR and FOR+BitPacking:
 * 1. Decision correctness for various data patterns
 * 2. Cost calculation verification
 * 3. End-to-end compression/decompression roundtrip
 *
 * Date: 2025-12-05
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

#include "L3_format.hpp"
#include "../src/kernels/compression/adaptive_selector.cuh"

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

const char* modelTypeToString(int model_type) {
    switch (model_type) {
        case MODEL_LINEAR: return "LINEAR";
        case MODEL_FOR_BITPACK: return "FOR+BP";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Data Generators
// ============================================================================

template<typename T>
std::vector<T> generateLinearData(size_t n, T start, double slope) {
    std::vector<T> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<T>(start + slope * i);
    }
    return data;
}

template<typename T>
std::vector<T> generateConstantData(size_t n, T value) {
    return std::vector<T>(n, value);
}

template<typename T>
std::vector<T> generateRandomData(size_t n, T min_val, T max_val) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateClusteredData(size_t n, T center, T range) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> dist(center - range/2, center + range/2);
    for (size_t i = 0; i < n; i++) {
        data[i] = dist(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateLinearWithNoise(size_t n, T start, double slope, T noise_range) {
    std::vector<T> data(n);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> noise(-noise_range, noise_range);
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<T>(start + slope * i) + noise(gen);
    }
    return data;
}

// ============================================================================
// Test Cases
// ============================================================================

struct TestCase {
    std::string name;
    int expected_model;  // MODEL_LINEAR or MODEL_FOR_BITPACK
    std::vector<int32_t> data;
};

std::vector<TestCase> createTestCases() {
    std::vector<TestCase> cases;
    const size_t n = 2048;

    // Test 1: Perfect linear data - should choose LINEAR
    {
        TestCase tc;
        tc.name = "perfect_linear";
        tc.expected_model = MODEL_LINEAR;
        tc.data = generateLinearData<int32_t>(n, 1000, 1.0);
        cases.push_back(tc);
    }

    // Test 2: Steep linear data - should choose LINEAR
    {
        TestCase tc;
        tc.name = "steep_linear";
        tc.expected_model = MODEL_LINEAR;
        tc.data = generateLinearData<int32_t>(n, 0, 100.0);
        cases.push_back(tc);
    }

    // Test 3: Constant data - should choose FOR (bits=0)
    {
        TestCase tc;
        tc.name = "constant";
        tc.expected_model = MODEL_FOR_BITPACK;
        tc.data = generateConstantData<int32_t>(n, 12345);
        cases.push_back(tc);
    }

    // Test 4: Random uniform full range - should choose FOR
    {
        TestCase tc;
        tc.name = "random_full_range";
        tc.expected_model = MODEL_FOR_BITPACK;
        tc.data = generateRandomData<int32_t>(n, 0, INT32_MAX/2);
        cases.push_back(tc);
    }

    // Test 5: Random small range - should choose FOR
    {
        TestCase tc;
        tc.name = "random_small_range";
        tc.expected_model = MODEL_FOR_BITPACK;
        tc.data = generateRandomData<int32_t>(n, 1000, 1100);
        cases.push_back(tc);
    }

    // Test 6: Clustered data - should choose FOR
    {
        TestCase tc;
        tc.name = "clustered";
        tc.expected_model = MODEL_FOR_BITPACK;
        tc.data = generateClusteredData<int32_t>(n, 50000, 200);
        cases.push_back(tc);
    }

    // Test 7: Linear with small noise - should choose LINEAR
    {
        TestCase tc;
        tc.name = "linear_small_noise";
        tc.expected_model = MODEL_LINEAR;
        tc.data = generateLinearWithNoise<int32_t>(n, 1000, 10.0, 5);
        cases.push_back(tc);
    }

    // Test 8: Linear with large noise - should choose FOR
    {
        TestCase tc;
        tc.name = "linear_large_noise";
        tc.expected_model = MODEL_FOR_BITPACK;
        tc.data = generateLinearWithNoise<int32_t>(n, 1000, 1.0, 1000);
        cases.push_back(tc);
    }

    return cases;
}

// ============================================================================
// Test Functions
// ============================================================================

bool testDecisionCorrectness() {
    std::cout << "\n========================================\n";
    std::cout << "Test 1: Decision Correctness\n";
    std::cout << "========================================\n";

    auto test_cases = createTestCases();
    int passed = 0;
    int total = test_cases.size();

    for (const auto& tc : test_cases) {
        // CPU decision
        auto cpu_decision = adaptive_selector::computeDecisionCPU<int32_t>(
            tc.data.data(), 0, tc.data.size());

        std::cout << std::setw(25) << tc.name << ": ";
        std::cout << "expected=" << modelTypeToString(tc.expected_model);
        std::cout << ", got=" << modelTypeToString(cpu_decision.model_type);
        std::cout << ", bits=" << cpu_decision.delta_bits;
        std::cout << ", cost=" << std::fixed << std::setprecision(1) << cpu_decision.estimated_cost;

        if (cpu_decision.model_type == tc.expected_model) {
            std::cout << " [PASS]\n";
            passed++;
        } else {
            std::cout << " [FAIL]\n";
        }
    }

    std::cout << "\nPassed: " << passed << "/" << total << "\n";
    return passed == total;
}

bool testGPUDecision() {
    std::cout << "\n========================================\n";
    std::cout << "Test 2: GPU Decision vs CPU\n";
    std::cout << "========================================\n";

    auto test_cases = createTestCases();
    int passed = 0;
    int total = test_cases.size();

    for (const auto& tc : test_cases) {
        size_t n = tc.data.size();

        // Allocate GPU memory
        int32_t* d_data;
        int32_t* d_start_indices;
        int32_t* d_end_indices;
        adaptive_selector::ModelDecision<int32_t>* d_decisions;

        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_start_indices, sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_end_indices, sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_decisions, sizeof(adaptive_selector::ModelDecision<int32_t>)));

        // Copy data
        CUDA_CHECK(cudaMemcpy(d_data, tc.data.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice));
        int32_t start = 0;
        int32_t end = static_cast<int32_t>(n);
        CUDA_CHECK(cudaMemcpy(d_start_indices, &start, sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_end_indices, &end, sizeof(int32_t), cudaMemcpyHostToDevice));

        // Run GPU kernel
        adaptive_selector::launchAdaptiveSelector<int32_t>(
            d_data, d_start_indices, d_end_indices, 1, d_decisions);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy back results
        adaptive_selector::ModelDecision<int32_t> gpu_decision;
        CUDA_CHECK(cudaMemcpy(&gpu_decision, d_decisions,
            sizeof(adaptive_selector::ModelDecision<int32_t>), cudaMemcpyDeviceToHost));

        // CPU decision for comparison
        auto cpu_decision = adaptive_selector::computeDecisionCPU<int32_t>(
            tc.data.data(), 0, n);

        std::cout << std::setw(25) << tc.name << ": ";
        std::cout << "GPU=" << modelTypeToString(gpu_decision.model_type);
        std::cout << ", CPU=" << modelTypeToString(cpu_decision.model_type);
        std::cout << ", GPU_bits=" << gpu_decision.delta_bits;
        std::cout << ", CPU_bits=" << cpu_decision.delta_bits;

        // Model type should match, but bits may differ (GPU uses estimation, CPU uses actual)
        bool match = (gpu_decision.model_type == cpu_decision.model_type);

        if (match) {
            std::cout << " [PASS]\n";
            passed++;
        } else {
            std::cout << " [FAIL]\n";
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_data));
        CUDA_CHECK(cudaFree(d_start_indices));
        CUDA_CHECK(cudaFree(d_end_indices));
        CUDA_CHECK(cudaFree(d_decisions));
    }

    std::cout << "\nPassed: " << passed << "/" << total << "\n";
    return passed == total;
}

bool testCostCalculation() {
    std::cout << "\n========================================\n";
    std::cout << "Test 3: Cost Calculation Verification\n";
    std::cout << "========================================\n";

    const size_t n = 2048;

    // Test case: constant data (range = 0)
    {
        auto data = generateConstantData<int32_t>(n, 12345);
        auto decision = adaptive_selector::computeDecisionCPU<int32_t>(
            data.data(), 0, n);

        // FOR cost = sizeof(T) + n * 0 / 8 = 4 bytes
        double expected_cost = 4.0;  // Only base, no deltas

        std::cout << "Constant data: bits=" << decision.delta_bits;
        std::cout << ", cost=" << std::fixed << std::setprecision(2) << decision.estimated_cost;
        std::cout << ", expected~=" << expected_cost;

        if (decision.delta_bits == 0 && decision.estimated_cost < 10.0) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL]\n";
            return false;
        }
    }

    // Test case: linear data
    {
        auto data = generateLinearData<int32_t>(n, 1000, 1.0);
        auto decision = adaptive_selector::computeDecisionCPU<int32_t>(
            data.data(), 0, n);

        // Should choose LINEAR with small residual bits
        std::cout << "Linear data: model=" << modelTypeToString(decision.model_type);
        std::cout << ", bits=" << decision.delta_bits;
        std::cout << ", cost=" << std::fixed << std::setprecision(2) << decision.estimated_cost;

        if (decision.model_type == MODEL_LINEAR && decision.delta_bits < 10) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL]\n";
            return false;
        }
    }

    // Test case: full range random data
    {
        auto data = generateRandomData<int32_t>(n, 0, INT32_MAX/2);
        auto decision = adaptive_selector::computeDecisionCPU<int32_t>(
            data.data(), 0, n);

        // FOR cost ≈ 4 + n * 31 / 8 ≈ 7940 bytes
        // Should choose FOR with ~31 bits
        std::cout << "Random data: model=" << modelTypeToString(decision.model_type);
        std::cout << ", bits=" << decision.delta_bits;
        std::cout << ", cost=" << std::fixed << std::setprecision(2) << decision.estimated_cost;

        if (decision.model_type == MODEL_FOR_BITPACK && decision.delta_bits >= 28) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL]\n";
            return false;
        }
    }

    return true;
}

bool testMultiplePartitions() {
    std::cout << "\n========================================\n";
    std::cout << "Test 4: Multiple Partitions\n";
    std::cout << "========================================\n";

    const int num_partitions = 4;
    const size_t partition_size = 1024;
    const size_t total_size = num_partitions * partition_size;

    // Create mixed data: linear, constant, random, clustered
    std::vector<int32_t> data(total_size);

    // Partition 0: linear
    auto linear = generateLinearData<int32_t>(partition_size, 1000, 1.0);
    std::copy(linear.begin(), linear.end(), data.begin());

    // Partition 1: constant
    auto constant = generateConstantData<int32_t>(partition_size, 50000);
    std::copy(constant.begin(), constant.end(), data.begin() + partition_size);

    // Partition 2: random
    auto random = generateRandomData<int32_t>(partition_size, 0, 1000000);
    std::copy(random.begin(), random.end(), data.begin() + 2 * partition_size);

    // Partition 3: clustered
    auto clustered = generateClusteredData<int32_t>(partition_size, 100000, 100);
    std::copy(clustered.begin(), clustered.end(), data.begin() + 3 * partition_size);

    // Allocate GPU memory
    int32_t* d_data;
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    adaptive_selector::ModelDecision<int32_t>* d_decisions;

    CUDA_CHECK(cudaMalloc(&d_data, total_size * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<int32_t>)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), total_size * sizeof(int32_t), cudaMemcpyHostToDevice));

    std::vector<int32_t> start_indices(num_partitions);
    std::vector<int32_t> end_indices(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        start_indices[i] = i * partition_size;
        end_indices[i] = (i + 1) * partition_size;
    }
    CUDA_CHECK(cudaMemcpy(d_start_indices, start_indices.data(),
        num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end_indices, end_indices.data(),
        num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));

    // Run GPU kernel
    adaptive_selector::launchAdaptiveSelector<int32_t>(
        d_data, d_start_indices, d_end_indices, num_partitions, d_decisions);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results
    std::vector<adaptive_selector::ModelDecision<int32_t>> decisions(num_partitions);
    CUDA_CHECK(cudaMemcpy(decisions.data(), d_decisions,
        num_partitions * sizeof(adaptive_selector::ModelDecision<int32_t>), cudaMemcpyDeviceToHost));

    // Expected results
    int expected_models[] = {MODEL_LINEAR, MODEL_FOR_BITPACK, MODEL_FOR_BITPACK, MODEL_FOR_BITPACK};
    const char* partition_names[] = {"linear", "constant", "random", "clustered"};

    int passed = 0;
    for (int i = 0; i < num_partitions; i++) {
        std::cout << "Partition " << i << " (" << partition_names[i] << "): ";
        std::cout << "model=" << modelTypeToString(decisions[i].model_type);
        std::cout << ", bits=" << decisions[i].delta_bits;

        if (decisions[i].model_type == expected_models[i]) {
            std::cout << " [PASS]\n";
            passed++;
        } else {
            std::cout << " [FAIL - expected " << modelTypeToString(expected_models[i]) << "]\n";
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_start_indices));
    CUDA_CHECK(cudaFree(d_end_indices));
    CUDA_CHECK(cudaFree(d_decisions));

    std::cout << "\nPassed: " << passed << "/" << num_partitions << "\n";
    return passed == num_partitions;
}

bool test64BitData() {
    std::cout << "\n========================================\n";
    std::cout << "Test 5: 64-bit Data Types\n";
    std::cout << "========================================\n";

    const size_t n = 1024;

    // Test linear 64-bit data
    std::vector<int64_t> linear_data(n);
    for (size_t i = 0; i < n; i++) {
        linear_data[i] = 1000000000000LL + static_cast<int64_t>(i) * 1000;
    }

    // Allocate GPU memory
    int64_t* d_data;
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    adaptive_selector::ModelDecision<int64_t>* d_decisions;

    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_start_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_end_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_decisions, sizeof(adaptive_selector::ModelDecision<int64_t>)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_data, linear_data.data(), n * sizeof(int64_t), cudaMemcpyHostToDevice));
    int32_t start = 0;
    int32_t end = static_cast<int32_t>(n);
    CUDA_CHECK(cudaMemcpy(d_start_indices, &start, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end_indices, &end, sizeof(int32_t), cudaMemcpyHostToDevice));

    // Run GPU kernel
    adaptive_selector::launchAdaptiveSelector<int64_t>(
        d_data, d_start_indices, d_end_indices, 1, d_decisions);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back results
    adaptive_selector::ModelDecision<int64_t> gpu_decision;
    CUDA_CHECK(cudaMemcpy(&gpu_decision, d_decisions,
        sizeof(adaptive_selector::ModelDecision<int64_t>), cudaMemcpyDeviceToHost));

    // CPU decision for comparison
    auto cpu_decision = adaptive_selector::computeDecisionCPU<int64_t>(
        linear_data.data(), 0, n);

    std::cout << "64-bit linear data:\n";
    std::cout << "  GPU: model=" << modelTypeToString(gpu_decision.model_type);
    std::cout << ", bits=" << gpu_decision.delta_bits;
    std::cout << ", params[1]=" << std::fixed << std::setprecision(2) << gpu_decision.params[1] << "\n";
    std::cout << "  CPU: model=" << modelTypeToString(cpu_decision.model_type);
    std::cout << ", bits=" << cpu_decision.delta_bits;
    std::cout << ", params[1]=" << std::fixed << std::setprecision(2) << cpu_decision.params[1] << "\n";

    // Model type and params[1] (slope) should match, bits may differ slightly
    bool passed = (gpu_decision.model_type == cpu_decision.model_type) &&
                  (std::fabs(gpu_decision.params[1] - cpu_decision.params[1]) < 1.0);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_start_indices));
    CUDA_CHECK(cudaFree(d_end_indices));
    CUDA_CHECK(cudaFree(d_decisions));

    if (passed) {
        std::cout << "[PASS]\n";
    } else {
        std::cout << "[FAIL]\n";
    }

    return passed;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "L3 Adaptive Model Selector Test Suite\n";
    std::cout << "========================================\n";

    // Check CUDA device
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    std::cout << "Device: " << prop.name << "\n";

    int total_tests = 5;
    int passed_tests = 0;

    // Run tests
    if (testDecisionCorrectness()) passed_tests++;
    if (testGPUDecision()) passed_tests++;
    if (testCostCalculation()) passed_tests++;
    if (testMultiplePartitions()) passed_tests++;
    if (test64BitData()) passed_tests++;

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "SUMMARY: " << passed_tests << "/" << total_tests << " tests passed\n";
    std::cout << "========================================\n";

    return (passed_tests == total_tests) ? 0 : 1;
}
