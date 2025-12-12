/**
 * Benchmark High-Order Polynomial Models
 *
 * Tests datasets 14-20 with different polynomial models:
 * - Linear (degree 1) - equivalent to V3
 * - Quadratic (degree 2)
 * - Cubic (degree 3)
 * - Auto (cost-based selection)
 *
 * Date: 2025-12-05
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/polynomial_fitting.cuh"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"
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

struct DatasetInfo {
    std::string filename;
    std::string name;
    std::string type;
};

// Datasets 14-20
const std::vector<DatasetInfo> DATASETS = {
    {"14-cosmos_int32.bin", "cosmos", "int32"},
    {"15-polylog_10M_uint64.bin", "polylog_10M", "uint64"},
    {"16-exp_200M_uint64.bin", "exp_200M", "uint64"},
    {"17-poly_200M_uint64.bin", "poly_200M", "uint64"},
    {"18-site_250k_uint32.bin", "site_250k", "uint32"},
    {"19-weight_25k_uint32.bin", "weight_25k", "uint32"},
    {"20-adult_30k_uint32.bin", "adult_30k", "uint32"},
};

template<typename T>
std::vector<T> loadData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

// ============================================================================
// CPU Legendre Fitting (Correct Least Squares - mirrors GPU implementation)
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

void legendreAll_cpu(double x, int degree, double* P) {
    P[0] = 1.0;
    if (degree >= 1) P[1] = x;
    if (degree >= 2) P[2] = (3.0 * x * x - 1.0) * 0.5;
    if (degree >= 3) P[3] = (5.0 * x * x * x - 3.0 * x) * 0.5;
}

// Solve small linear system using Gaussian elimination with partial pivoting
void solveLinearSystem_cpu(double G[4][4], double b[4], int d) {
    int size = d + 1;

    // Forward elimination with partial pivoting
    for (int k = 0; k < size; k++) {
        // Find pivot
        int maxRow = k;
        double maxVal = fabs(G[k][k]);
        for (int i = k + 1; i < size; i++) {
            if (fabs(G[i][k]) > maxVal) {
                maxVal = fabs(G[i][k]);
                maxRow = i;
            }
        }

        // Swap rows
        if (maxRow != k) {
            for (int j = k; j < size; j++) {
                std::swap(G[k][j], G[maxRow][j]);
            }
            std::swap(b[k], b[maxRow]);
        }

        // Eliminate column
        if (fabs(G[k][k]) > 1e-15) {
            for (int i = k + 1; i < size; i++) {
                double factor = G[i][k] / G[k][k];
                for (int j = k; j < size; j++) {
                    G[i][j] -= factor * G[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }

    // Back substitution
    for (int k = size - 1; k >= 0; k--) {
        if (fabs(G[k][k]) > 1e-15) {
            for (int j = k + 1; j < size; j++) {
                b[k] -= G[k][j] * b[j];
            }
            b[k] /= G[k][k];
        } else {
            b[k] = 0.0;
        }
    }
}

// Convert Legendre coefficients to standard polynomial coefficients
void legendreToStandard_cpu(const double* c, int degree, int n, double* a) {
    if (n <= 1) {
        a[0] = c[0];
        a[1] = a[2] = a[3] = 0.0;
        return;
    }

    double s = 2.0 / (n - 1);
    double d = -1.0;
    double s2 = s * s;
    double s3 = s2 * s;
    double d2 = d * d;
    double d3 = d * d * d;

    a[0] = a[1] = a[2] = a[3] = 0.0;

    a[0] += c[0];

    if (degree >= 1) {
        a[0] += c[1] * d;
        a[1] += c[1] * s;
    }

    if (degree >= 2) {
        double p2_const = (3.0 * d2 - 1.0) * 0.5;
        double p2_x1 = 3.0 * s * d;
        double p2_x2 = 1.5 * s2;

        a[0] += c[2] * p2_const;
        a[1] += c[2] * p2_x1;
        a[2] += c[2] * p2_x2;
    }

    if (degree >= 3) {
        double p3_const = (5.0 * d3 - 3.0 * d) * 0.5;
        double p3_x1 = (15.0 * s * d2 - 3.0 * s) * 0.5;
        double p3_x2 = (15.0 * s2 * d) * 0.5;
        double p3_x3 = (5.0 * s3) * 0.5;

        a[0] += c[3] * p3_const;
        a[1] += c[3] * p3_x1;
        a[2] += c[3] * p3_x2;
        a[3] += c[3] * p3_x3;
    }
}

template<typename T>
void fitLegendreCPU(const std::vector<T>& data, int start, int end, int degree,
                    double* std_coeffs, int64_t* max_error, int* delta_bits) {
    int n = end - start;
    if (n <= 0) {
        for (int i = 0; i < 4; i++) std_coeffs[i] = 0.0;
        *max_error = 0;
        *delta_bits = 0;
        return;
    }

    // Build Gram matrix G and RHS vector b
    double G[4][4] = {{0}};
    double b[4] = {0};

    for (int i = start; i < end; i++) {
        int local_idx = i - start;
        double xp = (n <= 1) ? 0.0 : (2.0 * local_idx / (n - 1) - 1.0);
        double y = static_cast<double>(data[i]);

        double P[4];
        legendreAll_cpu(xp, degree, P);

        for (int j = 0; j <= degree; j++) {
            b[j] += y * P[j];
            for (int k = 0; k <= degree; k++) {
                G[j][k] += P[j] * P[k];
            }
        }
    }

    // Solve G * c = b using Gaussian elimination
    solveLinearSystem_cpu(G, b, degree);

    // b now contains the Legendre coefficients
    double legendre_coeffs[4] = {0};
    for (int k = 0; k <= degree; k++) {
        legendre_coeffs[k] = b[k];
    }

    // Convert to standard polynomial coefficients
    legendreToStandard_cpu(legendre_coeffs, degree, n, std_coeffs);

    // Compute max error using Legendre basis (more accurate)
    *max_error = 0;
    for (int i = start; i < end; i++) {
        int local_idx = i - start;
        double xp = (n <= 1) ? 0.0 : (2.0 * local_idx / (n - 1) - 1.0);

        double P[4];
        legendreAll_cpu(xp, degree, P);

        double predicted = 0.0;
        for (int k = 0; k <= degree; k++) {
            predicted += legendre_coeffs[k] * P[k];
        }
        T pred_val = static_cast<T>(std::llrint(predicted));

        int64_t delta;
        if (data[i] >= pred_val) {
            delta = static_cast<int64_t>(data[i] - pred_val);
        } else {
            delta = -static_cast<int64_t>(pred_val - data[i]);
        }
        *max_error = std::max(*max_error, std::abs(delta));
    }

    if (*max_error == 0) {
        *delta_bits = 0;
    } else {
        *delta_bits = 64 - __builtin_clzll(static_cast<unsigned long long>(*max_error)) + 1;
    }
}

// ============================================================================
// Test with specific polynomial degree
// ============================================================================

struct TestResult {
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double compression_ratio;
    double decompress_gbps;
    bool correct;
    double compressed_mb;
    double original_mb;
};

template<typename T>
TestResult runTestWithDegree(const std::vector<T>& data, const std::string& name,
                             int degree, int partition_size = 2048) {
    TestResult result = {};

    if (data.empty()) {
        return result;
    }

    double data_bytes = data.size() * sizeof(T);
    result.original_mb = data_bytes / (1024.0 * 1024.0);

    // Create fixed partitions
    auto partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), partition_size);
    int num_partitions = partitions.size();
    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;

    // Fit each partition with the specified degree
    for (auto& part : partitions) {
        int start = part.start_idx;
        int end = part.end_idx;

        double std_coeffs[4];
        int64_t max_error;
        int delta_bits;

        fitLegendreCPU(data, start, end, degree, std_coeffs, &max_error, &delta_bits);

        // Set model type based on degree
        switch (degree) {
            case 1: part.model_type = MODEL_LINEAR; break;
            case 2: part.model_type = MODEL_POLYNOMIAL2; break;
            case 3: part.model_type = MODEL_POLYNOMIAL3; break;
            default: part.model_type = MODEL_LINEAR; break;
        }

        for (int j = 0; j < 4; j++) {
            part.model_params[j] = std_coeffs[j];
        }
        part.delta_bits = delta_bits;
        part.error_bound = max_error;
    }

    // Calculate stats
    double total_bits = 0;
    int64_t total_delta_storage = 0;
    for (const auto& p : partitions) {
        total_bits += p.delta_bits;
        total_delta_storage += static_cast<int64_t>(p.end_idx - p.start_idx) * p.delta_bits;
    }
    result.avg_delta_bits = total_bits / num_partitions;

    // Copy to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // Build compressed structure
    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t)));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits_storage = 0;
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_bit_offsets[p] = total_bits_storage;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
        total_bits_storage += static_cast<int64_t>(h_end[p] - h_start[p]) * h_delta_bits[p];
    }

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t delta_words = (total_bits_storage + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // Pack deltas
    int blocks = std::min((int)((data.size() + 255) / 256), 65535);
    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices, compressed.d_end_indices,
        compressed.d_model_types, compressed.d_model_params,
        compressed.d_delta_bits, compressed.d_delta_array_bit_offsets,
        num_partitions, compressed.d_sequential_deltas);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Calculate compressed size
    double metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2);
    double delta_bytes = delta_words * sizeof(uint32_t);
    result.compressed_mb = (metadata_bytes + delta_bytes) / (1024.0 * 1024.0);
    result.compression_ratio = result.original_mb / result.compressed_mb;

    // Decompression timing
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 3; i++) {
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    const int NUM_TRIALS = 10;
    float total_decompress_ms = 0;
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float trial_ms;
        CUDA_CHECK(cudaEventElapsedTime(&trial_ms, start, stop));
        total_decompress_ms += trial_ms;
    }
    float decompress_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_gbps = (data_bytes / 1e9) / (decompress_ms / 1000.0);

    // Verify
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));
    result.correct = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.correct = false;
            break;
        }
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

// ============================================================================
// Cost-based model selection (Auto mode)
// ============================================================================

template<typename T>
TestResult runTestAuto(const std::vector<T>& data, const std::string& name,
                       int partition_size = 2048, int* model_counts = nullptr) {
    TestResult result = {};

    if (data.empty()) {
        return result;
    }

    double data_bytes = data.size() * sizeof(T);
    result.original_mb = data_bytes / (1024.0 * 1024.0);

    // Create fixed partitions
    auto partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), partition_size);
    int num_partitions = partitions.size();
    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;

    int linear_count = 0, poly2_count = 0, poly3_count = 0;

    // Fit each partition with cost-based model selection
    for (auto& part : partitions) {
        int start = part.start_idx;
        int end = part.end_idx;
        int n = end - start;

        // Try all degrees and select best by cost
        double best_cost = 1e30;
        int best_degree = 1;
        double best_coeffs[4];
        int best_bits = 0;
        int64_t best_error = 0;

        for (int degree = 1; degree <= 3; degree++) {
            double std_coeffs[4];
            int64_t max_error;
            int delta_bits;

            fitLegendreCPU(data, start, end, degree, std_coeffs, &max_error, &delta_bits);

            // Cost = metadata + delta storage
            // Metadata: (degree+1) * 8 bytes for coefficients
            double metadata = (degree + 1) * 8.0;
            double storage = static_cast<double>(n) * delta_bits / 8.0;
            double cost = metadata + storage;

            if (cost < best_cost) {
                best_cost = cost;
                best_degree = degree;
                for (int j = 0; j < 4; j++) best_coeffs[j] = std_coeffs[j];
                best_bits = delta_bits;
                best_error = max_error;
            }
        }

        // Set model type based on best degree
        switch (best_degree) {
            case 1:
                part.model_type = MODEL_LINEAR;
                linear_count++;
                break;
            case 2:
                part.model_type = MODEL_POLYNOMIAL2;
                poly2_count++;
                break;
            case 3:
                part.model_type = MODEL_POLYNOMIAL3;
                poly3_count++;
                break;
        }

        for (int j = 0; j < 4; j++) {
            part.model_params[j] = best_coeffs[j];
        }
        part.delta_bits = best_bits;
        part.error_bound = best_error;
    }

    if (model_counts) {
        model_counts[0] = linear_count;
        model_counts[1] = poly2_count;
        model_counts[2] = poly3_count;
    }

    // Same encoding/decoding as runTestWithDegree
    double total_bits = 0;
    int64_t total_delta_storage = 0;
    for (const auto& p : partitions) {
        total_bits += p.delta_bits;
        total_delta_storage += static_cast<int64_t>(p.end_idx - p.start_idx) * p.delta_bits;
    }
    result.avg_delta_bits = total_bits / num_partitions;

    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t)));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits_storage = 0;
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_bit_offsets[p] = total_bits_storage;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
        total_bits_storage += static_cast<int64_t>(h_end[p] - h_start[p]) * h_delta_bits[p];
    }

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t delta_words = (total_bits_storage + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    int blocks = std::min((int)((data.size() + 255) / 256), 65535);
    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices, compressed.d_end_indices,
        compressed.d_model_types, compressed.d_model_params,
        compressed.d_delta_bits, compressed.d_delta_array_bit_offsets,
        num_partitions, compressed.d_sequential_deltas);
    CUDA_CHECK(cudaDeviceSynchronize());

    double metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2);
    double delta_bytes = delta_words * sizeof(uint32_t);
    result.compressed_mb = (metadata_bytes + delta_bytes) / (1024.0 * 1024.0);
    result.compression_ratio = result.original_mb / result.compressed_mb;

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 3; i++) {
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const int NUM_TRIALS = 10;
    float total_decompress_ms = 0;
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float trial_ms;
        CUDA_CHECK(cudaEventElapsedTime(&trial_ms, start, stop));
        total_decompress_ms += trial_ms;
    }
    float decompress_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_gbps = (data_bytes / 1e9) / (decompress_ms / 1000.0);

    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));
    result.correct = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.correct = false;
            break;
        }
    }

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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

void printResult(const std::string& name, const std::string& method, const TestResult& r) {
    std::cout << std::setw(12) << name << " | "
              << std::setw(8) << method << " | "
              << std::setw(5) << r.num_partitions << " | "
              << std::fixed << std::setprecision(2) << std::setw(6) << r.avg_delta_bits << " | "
              << std::setprecision(3) << std::setw(8) << r.compression_ratio << " | "
              << std::setprecision(2) << std::setw(8) << r.decompress_gbps << " | "
              << (r.correct ? "PASS" : "FAIL") << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_dir>" << std::endl;
        return 1;
    }
    std::string data_dir = argv[1];

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "# High-Order Polynomial Model Benchmark" << std::endl;
    std::cout << "# GPU: " << prop.name << std::endl;
    std::cout << "#" << std::endl;
    std::cout << "# Partition Size: 2048" << std::endl;
    std::cout << "# Models: Linear (d=1), Quadratic (d=2), Cubic (d=3), Auto (cost-based)" << std::endl;
    std::cout << "#" << std::endl;
    std::cout << std::setw(12) << "Dataset" << " | "
              << std::setw(8) << "Method" << " | "
              << std::setw(5) << "Parts" << " | "
              << std::setw(6) << "AvgBit" << " | "
              << std::setw(8) << "Ratio" << " | "
              << std::setw(8) << "GB/s" << " | "
              << "Status" << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (const auto& ds : DATASETS) {
        std::string path = data_dir + "/" + ds.filename;

        if (ds.type == "int32" || ds.type == "uint32") {
            auto data = loadData<uint32_t>(path);
            if (data.empty()) {
                std::cout << ds.name << " | SKIP (file not found)" << std::endl;
                continue;
            }

            auto r1 = runTestWithDegree<uint32_t>(data, ds.name, 1);
            printResult(ds.name, "Linear", r1);

            auto r2 = runTestWithDegree<uint32_t>(data, ds.name, 2);
            printResult(ds.name, "Poly2", r2);

            auto r3 = runTestWithDegree<uint32_t>(data, ds.name, 3);
            printResult(ds.name, "Poly3", r3);

            int model_counts[3];
            auto ra = runTestAuto<uint32_t>(data, ds.name, 2048, model_counts);
            printResult(ds.name, "Auto", ra);

            std::cout << "  # Auto model distribution: Linear=" << model_counts[0]
                      << ", Poly2=" << model_counts[1]
                      << ", Poly3=" << model_counts[2] << std::endl;
            std::cout << std::endl;
        } else {
            auto data = loadData<uint64_t>(path);
            if (data.empty()) {
                std::cout << ds.name << " | SKIP (file not found)" << std::endl;
                continue;
            }

            auto r1 = runTestWithDegree<uint64_t>(data, ds.name, 1);
            printResult(ds.name, "Linear", r1);

            auto r2 = runTestWithDegree<uint64_t>(data, ds.name, 2);
            printResult(ds.name, "Poly2", r2);

            auto r3 = runTestWithDegree<uint64_t>(data, ds.name, 3);
            printResult(ds.name, "Poly3", r3);

            int model_counts[3];
            auto ra = runTestAuto<uint64_t>(data, ds.name, 2048, model_counts);
            printResult(ds.name, "Auto", ra);

            std::cout << "  # Auto model distribution: Linear=" << model_counts[0]
                      << ", Poly2=" << model_counts[1]
                      << ", Poly3=" << model_counts[2] << std::endl;
            std::cout << std::endl;
        }
    }

    return 0;
}
