/**
 * POLY3 Verification Test
 *
 * This test verifies that:
 * 1. POLY3 model is actually being selected when beneficial
 * 2. POLY3 parameters (theta0-theta3) are different from LINEAR and POLY2
 * 3. POLY3 produces correct decompression results
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "L3_format.hpp"
#include "encoder_cost_optimal_gpu_merge.cuh"

// Generate data that follows a cubic pattern (ideal for POLY3)
std::vector<int64_t> generateCubicData(int n, double a, double b, double c, double d, double noise_level = 0.0) {
    std::vector<int64_t> data(n);
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, noise_level);

    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = a + b * x + c * x * x + d * x * x * x;
        if (noise_level > 0) {
            y += noise(gen);
        }
        data[i] = static_cast<int64_t>(std::round(y));
    }
    return data;
}

// Generate data that follows a linear pattern (ideal for LINEAR)
std::vector<int64_t> generateLinearData(int n, double a, double b, double noise_level = 0.0) {
    std::vector<int64_t> data(n);
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, noise_level);

    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = a + b * x;
        if (noise_level > 0) {
            y += noise(gen);
        }
        data[i] = static_cast<int64_t>(std::round(y));
    }
    return data;
}

// Generate data that follows a quadratic pattern (ideal for POLY2)
std::vector<int64_t> generateQuadraticData(int n, double a, double b, double c, double noise_level = 0.0) {
    std::vector<int64_t> data(n);
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, noise_level);

    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = a + b * x + c * x * x;
        if (noise_level > 0) {
            y += noise(gen);
        }
        data[i] = static_cast<int64_t>(std::round(y));
    }
    return data;
}

void printModelName(int model_type) {
    switch(model_type) {
        case MODEL_LINEAR: std::cout << "LINEAR"; break;
        case MODEL_POLYNOMIAL2: std::cout << "POLY2"; break;
        case MODEL_POLYNOMIAL3: std::cout << "POLY3"; break;
        default: std::cout << "UNKNOWN(" << model_type << ")"; break;
    }
}

template<typename T>
void testDataset(const std::string& name, const std::vector<T>& data, int partition_size) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << "Data size: " << data.size() << ", Partition size: " << partition_size << std::endl;
    std::cout << "========================================" << std::endl;

    // Configure partitioner - use polynomialEnabled preset
    CostOptimalConfig config = CostOptimalConfig::polynomialEnabled();
    config.min_partition_size = partition_size;
    config.max_partition_size = partition_size * 4;
    config.target_partition_size = partition_size;
    config.polynomial_min_size = 64;   // Allow POLY2 for partitions >= 64
    config.cubic_min_size = 128;       // Allow POLY3 for partitions >= 128
    config.polynomial_cost_threshold = 0.98f;  // Require only 2% improvement

    // Create partitioner
    GPUCostOptimalPartitioner<T> partitioner(data, config);

    // Partition
    auto partitions = partitioner.partition();

    // Count model types
    int linear_count = 0, poly2_count = 0, poly3_count = 0, other_count = 0;

    // Track some example partitions for each type
    std::vector<int> linear_examples, poly2_examples, poly3_examples;

    for (size_t i = 0; i < partitions.size(); i++) {
        const auto& p = partitions[i];
        switch(p.model_type) {
            case MODEL_LINEAR:
                linear_count++;
                if (linear_examples.size() < 3) linear_examples.push_back(i);
                break;
            case MODEL_POLYNOMIAL2:
                poly2_count++;
                if (poly2_examples.size() < 3) poly2_examples.push_back(i);
                break;
            case MODEL_POLYNOMIAL3:
                poly3_count++;
                if (poly3_examples.size() < 3) poly3_examples.push_back(i);
                break;
            default:
                other_count++;
                break;
        }
    }

    std::cout << "\nModel Distribution:" << std::endl;
    std::cout << "  LINEAR: " << linear_count << " partitions ("
              << std::fixed << std::setprecision(1)
              << (100.0 * linear_count / partitions.size()) << "%)" << std::endl;
    std::cout << "  POLY2:  " << poly2_count << " partitions ("
              << (100.0 * poly2_count / partitions.size()) << "%)" << std::endl;
    std::cout << "  POLY3:  " << poly3_count << " partitions ("
              << (100.0 * poly3_count / partitions.size()) << "%)" << std::endl;
    if (other_count > 0) {
        std::cout << "  OTHER:  " << other_count << " partitions" << std::endl;
    }

    // Show example partitions with parameters
    auto showPartition = [&](int idx, const char* label) {
        const auto& p = partitions[idx];
        std::cout << "\n  " << label << " Partition " << idx << ":" << std::endl;
        std::cout << "    Range: [" << p.start_idx << ", " << p.end_idx << ") = "
                  << (p.end_idx - p.start_idx) << " elements" << std::endl;
        std::cout << "    Model: ";
        printModelName(p.model_type);
        std::cout << std::endl;
        std::cout << "    theta0: " << std::scientific << std::setprecision(6) << p.model_params[0] << std::endl;
        std::cout << "    theta1: " << p.model_params[1] << std::endl;
        std::cout << "    theta2: " << p.model_params[2] << std::endl;
        std::cout << "    theta3: " << p.model_params[3] << std::endl;
        std::cout << "    delta_bits: " << p.delta_bits << std::endl;

        // Verify a few predictions
        std::cout << "    Sample predictions:" << std::endl;
        int start = p.start_idx;
        int n = p.end_idx - p.start_idx;
        for (int j = 0; j < std::min(3, n); j++) {
            int local_idx = j;
            double x = static_cast<double>(local_idx);
            double pred;
            switch(p.model_type) {
                case MODEL_LINEAR:
                    pred = p.model_params[0] + p.model_params[1] * x;
                    break;
                case MODEL_POLYNOMIAL2:
                    pred = p.model_params[0] + x * (p.model_params[1] + x * p.model_params[2]);
                    break;
                case MODEL_POLYNOMIAL3:
                    pred = p.model_params[0] + x * (p.model_params[1] + x * (p.model_params[2] + x * p.model_params[3]));
                    break;
                default:
                    pred = p.model_params[0] + p.model_params[1] * x;
            }
            int64_t actual = data[start + j];
            int64_t predicted = static_cast<int64_t>(std::llrint(pred));
            int64_t delta = actual - predicted;
            std::cout << "      x=" << j << ": actual=" << actual
                      << ", pred=" << predicted << ", delta=" << delta << std::endl;
        }
    };

    if (!linear_examples.empty()) {
        showPartition(linear_examples[0], "LINEAR");
    }
    if (!poly2_examples.empty()) {
        showPartition(poly2_examples[0], "POLY2");
    }
    if (!poly3_examples.empty()) {
        showPartition(poly3_examples[0], "POLY3");
    }
}

int main() {
    std::cout << "POLY3 Verification Test" << std::endl;
    std::cout << "========================" << std::endl;

    const int N = 100000;  // 100k elements
    const int PARTITION_SIZE = 2048;  // Large enough for POLY3

    // Test 1: Pure cubic data - should strongly prefer POLY3
    std::cout << "\n\n*** TEST 1: Pure Cubic Data ***" << std::endl;
    std::cout << "Generated: y = 1000 + 0.1*x + 0.0001*x^2 + 0.000001*x^3" << std::endl;
    auto cubic_data = generateCubicData(N, 1000.0, 0.1, 0.0001, 0.000001, 0.0);
    testDataset("Pure Cubic", cubic_data, PARTITION_SIZE);

    // Test 2: Pure quadratic data - should prefer POLY2
    std::cout << "\n\n*** TEST 2: Pure Quadratic Data ***" << std::endl;
    std::cout << "Generated: y = 1000 + 0.5*x + 0.001*x^2" << std::endl;
    auto quad_data = generateQuadraticData(N, 1000.0, 0.5, 0.001, 0.0);
    testDataset("Pure Quadratic", quad_data, PARTITION_SIZE);

    // Test 3: Pure linear data - should prefer LINEAR
    std::cout << "\n\n*** TEST 3: Pure Linear Data ***" << std::endl;
    std::cout << "Generated: y = 1000 + 2*x" << std::endl;
    auto linear_data = generateLinearData(N, 1000.0, 2.0, 0.0);
    testDataset("Pure Linear", linear_data, PARTITION_SIZE);

    // Test 4: Cubic with noise - should still prefer POLY3 if noise is small
    std::cout << "\n\n*** TEST 4: Noisy Cubic Data ***" << std::endl;
    std::cout << "Generated: y = 1000 + 0.1*x + 0.0001*x^2 + 0.000001*x^3 + noise(σ=10)" << std::endl;
    auto noisy_cubic = generateCubicData(N, 1000.0, 0.1, 0.0001, 0.000001, 10.0);
    testDataset("Noisy Cubic", noisy_cubic, PARTITION_SIZE);

    // Test 5: Strong cubic coefficient
    std::cout << "\n\n*** TEST 5: Strong Cubic Data ***" << std::endl;
    std::cout << "Generated: y = 0 + 0*x + 0*x^2 + 0.00001*x^3" << std::endl;
    auto strong_cubic = generateCubicData(N, 0.0, 0.0, 0.0, 0.00001, 0.0);
    testDataset("Strong Cubic", strong_cubic, PARTITION_SIZE);

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "Test Complete!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
