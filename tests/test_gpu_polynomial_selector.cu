/**
 * Test GPU Polynomial Selector
 *
 * Validates that the GPU kernel correctly selects polynomial models
 * (LINEAR, POLY2, POLY3, FOR_BITPACK) based on cost.
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "../src/kernels/compression/adaptive_selector.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Generate different types of test data
template<typename T>
std::vector<T> generateLinearData(int n, T base, T slope) {
    std::vector<T> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = base + static_cast<T>(slope * i);
    }
    return data;
}

template<typename T>
std::vector<T> generateQuadraticData(int n, double a0, double a1, double a2) {
    std::vector<T> data(n);
    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        data[i] = static_cast<T>(a0 + a1 * x + a2 * x * x);
    }
    return data;
}

template<typename T>
std::vector<T> generateCubicData(int n, double a0, double a1, double a2, double a3) {
    std::vector<T> data(n);
    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        data[i] = static_cast<T>(a0 + a1 * x + a2 * x * x + a3 * x * x * x);
    }
    return data;
}

template<typename T>
std::vector<T> generateRandomData(int n, T min_val, T max_val) {
    std::vector<T> data(n);
    for (int i = 0; i < n; i++) {
        data[i] = min_val + rand() % (max_val - min_val + 1);
    }
    return data;
}

const char* modelName(int model_type) {
    switch (model_type) {
        case MODEL_CONSTANT: return "CONSTANT";
        case MODEL_LINEAR: return "LINEAR";
        case MODEL_POLYNOMIAL2: return "POLY2";
        case MODEL_POLYNOMIAL3: return "POLY3";
        case MODEL_FOR_BITPACK: return "FOR_BITPACK";
        default: return "UNKNOWN";
    }
}

template<typename T>
void testSelector(const std::vector<T>& data, const std::string& name, int expected_model = -1) {
    int n = data.size();

    // Allocate device memory
    T* d_data;
    int32_t* d_starts;
    int32_t* d_ends;
    adaptive_selector::ModelDecision<T>* d_decisions;

    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_starts, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ends, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_decisions, sizeof(adaptive_selector::ModelDecision<T>)));

    // Copy data
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    int32_t start = 0, end = n;
    CUDA_CHECK(cudaMemcpy(d_starts, &start, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ends, &end, sizeof(int32_t), cudaMemcpyHostToDevice));

    // Launch kernel
    adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
        d_data, d_starts, d_ends, 1, d_decisions, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get result
    adaptive_selector::ModelDecision<T> decision;
    CUDA_CHECK(cudaMemcpy(&decision, d_decisions, sizeof(decision), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Test: " << name << std::endl;
    std::cout << "  Elements: " << n << std::endl;
    std::cout << "  Selected model: " << modelName(decision.model_type) << std::endl;
    std::cout << "  Delta bits: " << decision.delta_bits << std::endl;
    std::cout << "  Estimated cost: " << decision.estimated_cost << " bytes" << std::endl;
    std::cout << "  Params: [" << decision.params[0] << ", " << decision.params[1]
              << ", " << decision.params[2] << ", " << decision.params[3] << "]" << std::endl;

    if (expected_model >= 0) {
        if (decision.model_type == expected_model) {
            std::cout << "  Result: PASS (expected " << modelName(expected_model) << ")" << std::endl;
        } else {
            std::cout << "  Result: UNEXPECTED (expected " << modelName(expected_model)
                      << ", got " << modelName(decision.model_type) << ")" << std::endl;
        }
    }
    std::cout << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_decisions);
}

int main() {
    std::cout << "=== GPU Polynomial Selector Test ===" << std::endl;
    std::cout << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << std::endl;

    srand(42);

    // Test 1: Pure linear data - should select LINEAR
    {
        auto data = generateLinearData<uint64_t>(2048, 1000000, 100);
        testSelector(data, "Pure Linear (uint64)", MODEL_LINEAR);
    }

    // Test 2: Quadratic data - should select POLY2
    {
        auto data = generateQuadraticData<uint64_t>(2048, 1000000, 10, 0.5);
        testSelector(data, "Quadratic (uint64)", MODEL_POLYNOMIAL2);
    }

    // Test 3: Cubic data - should select POLY3
    {
        auto data = generateCubicData<uint64_t>(2048, 1000000, 1, 0.01, 0.0001);
        testSelector(data, "Cubic (uint64)", MODEL_POLYNOMIAL3);
    }

    // Test 4: Random data - should select FOR_BITPACK or LINEAR
    {
        auto data = generateRandomData<uint64_t>(2048, 1000000, 2000000);
        testSelector(data, "Random (uint64)");
    }

    // Test 5: Constant data - should select LINEAR with 0 bits
    {
        std::vector<uint64_t> data(2048, 12345678);
        testSelector(data, "Constant (uint64)", MODEL_LINEAR);
    }

    // Test 6: Polylog-like data (accelerating growth)
    {
        std::vector<uint64_t> data(2048);
        for (int i = 0; i < 2048; i++) {
            double x = i + 1;
            data[i] = static_cast<uint64_t>(1000000 + 100 * x * log(x));
        }
        testSelector(data, "Polylog-like (uint64)");
    }

    // Test 7: uint32 quadratic
    {
        auto data = generateQuadraticData<uint32_t>(2048, 100000, 5, 0.1);
        testSelector(data, "Quadratic (uint32)", MODEL_POLYNOMIAL2);
    }

    // Test 8: Small partition (n=50)
    {
        auto data = generateQuadraticData<uint64_t>(50, 1000000, 10, 0.5);
        testSelector(data, "Small Quadratic n=50 (uint64)");
    }

    // Test 9: Very small partition (n=15) - POLY3 should not be used
    {
        auto data = generateCubicData<uint64_t>(15, 1000000, 1, 0.1, 0.01);
        testSelector(data, "Small Cubic n=15 (uint64)");
    }

    std::cout << "=== Test Complete ===" << std::endl;

    return 0;
}
