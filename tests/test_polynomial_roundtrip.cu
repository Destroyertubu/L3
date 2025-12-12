/**
 * Test Polynomial Compression Roundtrip
 *
 * Tests the complete encode/decode cycle for polynomial models:
 * 1. Generate synthetic polynomial data
 * 2. Fit using Legendre polynomials
 * 3. Encode with Vertical packer
 * 4. Decode with polynomial prediction
 * 5. Verify correctness
 *
 * Date: 2025-12-05
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/polynomial_fitting.cuh"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// Include encoder and decoder
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

#undef WARP_SIZE
#undef BLOCK_SIZE
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

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
// Test Data Generation
// ============================================================================

template<typename T>
void generateLinearData(std::vector<T>& data, int n, double a0, double a1, int noise_range = 0) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise_dist(-noise_range, noise_range);

    data.resize(n);
    for (int i = 0; i < n; i++) {
        double y = a0 + a1 * i;
        if (noise_range > 0) y += noise_dist(gen);
        data[i] = static_cast<T>(std::llrint(y));
    }
}

template<typename T>
void generateQuadraticData(std::vector<T>& data, int n, double a0, double a1, double a2, int noise_range = 0) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise_dist(-noise_range, noise_range);

    data.resize(n);
    for (int i = 0; i < n; i++) {
        double x = i;
        double y = a0 + a1 * x + a2 * x * x;
        if (noise_range > 0) y += noise_dist(gen);
        data[i] = static_cast<T>(std::llrint(y));
    }
}

template<typename T>
void generateCubicData(std::vector<T>& data, int n, double a0, double a1, double a2, double a3, int noise_range = 0) {
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> noise_dist(-noise_range, noise_range);

    data.resize(n);
    for (int i = 0; i < n; i++) {
        double x = i;
        double y = a0 + a1 * x + a2 * x * x + a3 * x * x * x;
        if (noise_range > 0) y += noise_dist(gen);
        data[i] = static_cast<T>(std::llrint(y));
    }
}

// ============================================================================
// CPU Reference Implementation for Legendre Fitting
// ============================================================================

double legendreP_cpu(int k, double x) {
    switch (k) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return (3.0 * x * x - 1.0) * 0.5;
        case 3: return (5.0 * x * x * x - 3.0 * x) * 0.5;
        default: return 0.0;
    }
}

template<typename T>
void fitLegendreCPU(const std::vector<T>& data, int start, int end, int degree,
                    double* legendre_coeffs, double* std_coeffs,
                    int64_t* max_error, int* delta_bits) {
    int n = end - start;
    if (n <= 0) return;

    // Compute Σ y*P_k and Σ P_k² for each k
    std::vector<double> sum_yPk(degree + 1, 0.0);
    std::vector<double> sum_Pk2(degree + 1, 0.0);

    for (int i = start; i < end; i++) {
        int local_idx = i - start;
        double xp = (n <= 1) ? 0.0 : (2.0 * local_idx / (n - 1) - 1.0);  // x' ∈ [-1, 1]
        double y = static_cast<double>(data[i]);

        for (int k = 0; k <= degree; k++) {
            double Pk = legendreP_cpu(k, xp);
            sum_yPk[k] += y * Pk;
            sum_Pk2[k] += Pk * Pk;
        }
    }

    // Compute Legendre coefficients
    for (int k = 0; k <= degree; k++) {
        legendre_coeffs[k] = (sum_Pk2[k] > 1e-12) ? sum_yPk[k] / sum_Pk2[k] : 0.0;
    }
    for (int k = degree + 1; k < 4; k++) {
        legendre_coeffs[k] = 0.0;
    }

    // Convert to standard coefficients
    double s = (n > 1) ? 2.0 / (n - 1) : 1.0;
    double d = -1.0;
    double s2 = s * s;
    double s3 = s2 * s;
    double d2 = d * d;
    double d3 = d * d * d;

    std_coeffs[0] = std_coeffs[1] = std_coeffs[2] = std_coeffs[3] = 0.0;

    // c0 * P0 = c0
    std_coeffs[0] += legendre_coeffs[0];

    if (degree >= 1) {
        // c1 * P1 = c1 * (s*x + d)
        std_coeffs[0] += legendre_coeffs[1] * d;
        std_coeffs[1] += legendre_coeffs[1] * s;
    }

    if (degree >= 2) {
        double p2_const = (3.0 * d2 - 1.0) * 0.5;
        double p2_x1 = 3.0 * s * d;
        double p2_x2 = 1.5 * s2;

        std_coeffs[0] += legendre_coeffs[2] * p2_const;
        std_coeffs[1] += legendre_coeffs[2] * p2_x1;
        std_coeffs[2] += legendre_coeffs[2] * p2_x2;
    }

    if (degree >= 3) {
        double p3_const = (5.0 * d3 - 3.0 * d) * 0.5;
        double p3_x1 = (15.0 * s * d2 - 3.0 * s) * 0.5;
        double p3_x2 = (15.0 * s2 * d) * 0.5;
        double p3_x3 = (5.0 * s3) * 0.5;

        std_coeffs[0] += legendre_coeffs[3] * p3_const;
        std_coeffs[1] += legendre_coeffs[3] * p3_x1;
        std_coeffs[2] += legendre_coeffs[3] * p3_x2;
        std_coeffs[3] += legendre_coeffs[3] * p3_x3;
    }

    // Compute max error with standard coefficients
    *max_error = 0;
    for (int i = start; i < end; i++) {
        int local_idx = i - start;
        double x = local_idx;
        double predicted = std_coeffs[0] + x * (std_coeffs[1] + x * (std_coeffs[2] + x * std_coeffs[3]));
        T pred_val = static_cast<T>(std::llrint(predicted));

        int64_t delta;
        if (data[i] >= pred_val) {
            delta = static_cast<int64_t>(data[i] - pred_val);
        } else {
            delta = -static_cast<int64_t>(pred_val - data[i]);
        }
        *max_error = std::max(*max_error, std::abs(delta));
    }

    // Compute bits needed
    if (*max_error == 0) {
        *delta_bits = 0;
    } else {
        *delta_bits = 64 - __builtin_clzll(static_cast<unsigned long long>(*max_error)) + 1;
    }
}

// ============================================================================
// Test Functions
// ============================================================================

template<typename T>
bool testPolynomialRoundtrip(const std::string& test_name, const std::vector<T>& data,
                              int model_type, int degree) {
    std::cout << "\n=== " << test_name << " ===" << std::endl;
    std::cout << "Data size: " << data.size() << std::endl;

    int n = data.size();
    double data_bytes = n * sizeof(T);

    // 1. CPU Fitting
    double legendre_coeffs[4], std_coeffs[4];
    int64_t max_error;
    int delta_bits;

    fitLegendreCPU(data, 0, n, degree, legendre_coeffs, std_coeffs, &max_error, &delta_bits);

    std::cout << "Standard coefficients: [" << std_coeffs[0] << ", " << std_coeffs[1]
              << ", " << std_coeffs[2] << ", " << std_coeffs[3] << "]" << std::endl;
    std::cout << "Max error: " << max_error << ", Delta bits: " << delta_bits << std::endl;

    // 2. Build compressed structure
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    CompressedDataVertical<T> compressed;
    compressed.num_partitions = 1;
    compressed.total_values = n;

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, sizeof(int64_t)));

    int32_t h_start = 0, h_end = n;
    int32_t h_model_type = model_type;
    int32_t h_delta_bits = delta_bits;
    int64_t h_bit_offset = 0;

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, &h_start, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, &h_end, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, &h_model_type, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, std_coeffs, 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, &h_delta_bits, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, &h_bit_offset, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, &max_error, sizeof(int64_t), cudaMemcpyHostToDevice));

    // Allocate delta array
    int64_t total_bits = static_cast<int64_t>(n) * delta_bits;
    int64_t delta_words = (total_bits + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // 3. Encode (pack deltas)
    int blocks = std::min((int)((n + 255) / 256), 65535);
    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices, compressed.d_end_indices,
        compressed.d_model_types, compressed.d_model_params,
        compressed.d_delta_bits, compressed.d_delta_array_bit_offsets,
        1, compressed.d_sequential_deltas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Decode
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, data_bytes));

    Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Verify
    std::vector<T> decoded(n);
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] != decoded[i]) {
            if (error_count < 5) {
                std::cerr << "ERROR at index " << i << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
            error_count++;
        }
    }

    // Calculate compression ratio
    double metadata_bytes = sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2;
    double compressed_bytes = metadata_bytes + delta_words * sizeof(uint32_t);
    double ratio = data_bytes / compressed_bytes;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) << ratio << "x" << std::endl;
    std::cout << "Result: " << (error_count == 0 ? "PASS" : "FAIL") << std::endl;
    if (error_count > 0) {
        std::cout << "Total errors: " << error_count << " / " << n << std::endl;
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(compressed.d_start_indices);
    cudaFree(compressed.d_end_indices);
    cudaFree(compressed.d_model_types);
    cudaFree(compressed.d_model_params);
    cudaFree(compressed.d_delta_bits);
    cudaFree(compressed.d_delta_array_bit_offsets);
    cudaFree(compressed.d_error_bounds);
    cudaFree(compressed.d_sequential_deltas);

    return error_count == 0;
}

template<typename T>
bool testRandomAccess(const std::string& test_name, const std::vector<T>& data,
                      int model_type, int degree, int num_queries) {
    std::cout << "\n=== Random Access: " << test_name << " ===" << std::endl;

    int n = data.size();
    double data_bytes = n * sizeof(T);

    // Fit model
    double legendre_coeffs[4], std_coeffs[4];
    int64_t max_error;
    int delta_bits;
    fitLegendreCPU(data, 0, n, degree, legendre_coeffs, std_coeffs, &max_error, &delta_bits);

    // Build compressed structure
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    CompressedDataVertical<T> compressed;
    compressed.num_partitions = 1;
    compressed.total_values = n;

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, sizeof(int64_t)));

    int32_t h_start = 0, h_end = n;
    int32_t h_model_type = model_type;
    int32_t h_delta_bits = delta_bits;
    int64_t h_bit_offset = 0;

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, &h_start, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, &h_end, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, &h_model_type, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, std_coeffs, 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, &h_delta_bits, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, &h_bit_offset, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, &max_error, sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t total_bits = static_cast<int64_t>(n) * delta_bits;
    int64_t delta_words = (total_bits + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // Encode
    int blocks = std::min((int)((n + 255) / 256), 65535);
    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices, compressed.d_end_indices,
        compressed.d_model_types, compressed.d_model_params,
        compressed.d_delta_bits, compressed.d_delta_array_bit_offsets,
        1, compressed.d_sequential_deltas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Generate random query indices
    std::vector<int> query_indices(num_queries);
    std::mt19937 gen(12345);
    std::uniform_int_distribution<int> idx_dist(0, n - 1);
    for (int i = 0; i < num_queries; i++) {
        query_indices[i] = idx_dist(gen);
    }

    // Copy queries to device
    int* d_queries;
    T* d_results;
    CUDA_CHECK(cudaMalloc(&d_queries, num_queries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, num_queries * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_queries, query_indices.data(), num_queries * sizeof(int), cudaMemcpyHostToDevice));

    // Random access decompress
    Vertical_decoder::decompressIndices<T>(compressed, d_queries, num_queries, d_results, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<T> results(num_queries);
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, num_queries * sizeof(T), cudaMemcpyDeviceToHost));

    int error_count = 0;
    for (int i = 0; i < num_queries; i++) {
        int idx = query_indices[i];
        if (data[idx] != results[i]) {
            if (error_count < 5) {
                std::cerr << "ERROR: query[" << i << "] idx=" << idx << ": expected " << data[idx]
                          << ", got " << results[i] << std::endl;
            }
            error_count++;
        }
    }

    std::cout << "Queries: " << num_queries << ", Errors: " << error_count << std::endl;
    std::cout << "Result: " << (error_count == 0 ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_queries);
    cudaFree(d_results);
    cudaFree(compressed.d_start_indices);
    cudaFree(compressed.d_end_indices);
    cudaFree(compressed.d_model_types);
    cudaFree(compressed.d_model_params);
    cudaFree(compressed.d_delta_bits);
    cudaFree(compressed.d_delta_array_bit_offsets);
    cudaFree(compressed.d_error_bounds);
    cudaFree(compressed.d_sequential_deltas);

    return error_count == 0;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Polynomial Compression Roundtrip Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;

    bool all_pass = true;
    int n = 8192;  // Typical partition size

    // Test 1: Linear data with MODEL_LINEAR
    {
        std::vector<uint64_t> data;
        generateLinearData(data, n, 1000.0, 5.0, 10);  // y = 1000 + 5x + noise
        all_pass &= testPolynomialRoundtrip("Linear Data (MODEL_LINEAR)", data, MODEL_LINEAR, 1);
    }

    // Test 2: Quadratic data with MODEL_POLYNOMIAL2
    {
        std::vector<uint64_t> data;
        generateQuadraticData(data, n, 1000.0, 2.0, 0.01, 5);  // y = 1000 + 2x + 0.01x² + noise
        all_pass &= testPolynomialRoundtrip("Quadratic Data (MODEL_POLYNOMIAL2)", data, MODEL_POLYNOMIAL2, 2);
    }

    // Test 3: Cubic data with MODEL_POLYNOMIAL3
    {
        std::vector<uint64_t> data;
        generateCubicData(data, n, 1000.0, 1.0, 0.001, 0.00001, 10);  // y = 1000 + x + 0.001x² + 0.00001x³ + noise
        all_pass &= testPolynomialRoundtrip("Cubic Data (MODEL_POLYNOMIAL3)", data, MODEL_POLYNOMIAL3, 3);
    }

    // Test 4: Perfect linear data (delta_bits = 0)
    {
        std::vector<uint64_t> data;
        generateLinearData(data, n, 100.0, 3.0, 0);  // y = 100 + 3x (no noise)
        all_pass &= testPolynomialRoundtrip("Perfect Linear (delta_bits=0)", data, MODEL_LINEAR, 1);
    }

    // Test 5: Large quadratic data
    {
        std::vector<uint64_t> data;
        generateQuadraticData(data, 100000, 1000000.0, 100.0, 0.1, 100);
        all_pass &= testPolynomialRoundtrip("Large Quadratic (100K)", data, MODEL_POLYNOMIAL2, 2);
    }

    // Test 6: Random access tests
    {
        std::vector<uint64_t> data;
        generateCubicData(data, n, 5000.0, 3.0, 0.005, 0.000005, 20);
        all_pass &= testRandomAccess("Random Access Cubic", data, MODEL_POLYNOMIAL3, 3, 1000);
    }

    // Test 7: uint32_t data
    {
        std::vector<uint32_t> data;
        generateQuadraticData(data, n, 100.0, 1.5, 0.002, 5);
        all_pass &= testPolynomialRoundtrip("uint32 Quadratic", data, MODEL_POLYNOMIAL2, 2);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Overall Result: " << (all_pass ? "ALL PASS" : "SOME FAILED") << std::endl;
    std::cout << "========================================" << std::endl;

    return all_pass ? 0 : 1;
}
