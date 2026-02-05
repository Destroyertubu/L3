/**
 * Debug test to identify why Horizontal Random Access FAILS but Decompression PASSES
 * for normal, ml, exp, poly datasets
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cstdint>
#include <cuda_runtime.h>

// L3 headers
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "L3_random_access.hpp"
#include "L3_opt.h"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

// Generate synthetic "normal-like" data
std::vector<uint64_t> generateTestData(size_t n, int seed = 42) {
    std::vector<uint64_t> data(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(1e10, 1e8);

    for (size_t i = 0; i < n; i++) {
        double val = dist(rng);
        if (val < 0) val = -val;
        data[i] = static_cast<uint64_t>(val);
    }

    std::sort(data.begin(), data.end());
    return data;
}

template<typename T>
std::vector<PartitionInfo> generatePartitionsV2(const std::vector<T>& data, int partition_size) {
    CostOptimalConfig config;
    config.target_partition_size = partition_size;
    config.analysis_block_size = partition_size / 2;
    config.min_partition_size = 256;
    config.max_partition_size = partition_size * 2;
    config.breakpoint_threshold = 2;
    config.merge_benefit_threshold = 0.05f;
    config.max_merge_rounds = 4;
    config.enable_merging = true;
    config.enable_polynomial_models = true;
    config.polynomial_min_size = 10;
    config.cubic_min_size = 20;
    config.polynomial_cost_threshold = 0.95f;

    GPUCostOptimalPartitionerV2<T> partitioner(data, config, 0);
    return partitioner.partition();
}

int main() {
    std::cout << "=== Debug Random Access Failure ===" << std::endl;

    const size_t N = 100000;
    const int PARTITION_SIZE = 4096;

    std::cout << "Generating " << N << " test elements..." << std::endl;
    auto data = generateTestData(N);

    std::cout << "Partitioning data..." << std::endl;
    auto partitions = generatePartitionsV2<uint64_t>(data, PARTITION_SIZE);

    int linear_count = 0, poly2_count = 0, for_count = 0;
    for (const auto& p : partitions) {
        switch (p.model_type) {
            case MODEL_LINEAR: linear_count++; break;
            case MODEL_POLYNOMIAL2: poly2_count++; break;
            case MODEL_FOR_BITPACK: for_count++; break;
        }
    }
    std::cout << "  Partitions: " << partitions.size()
              << " (LINEAR=" << linear_count
              << ", POLY2=" << poly2_count
              << ", FOR=" << for_count << ")" << std::endl;

    std::cout << "Compressing..." << std::endl;
    auto* compressed = compressDataWithPartitions(data, partitions, nullptr);
    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }

    uint64_t* d_decomp_output;
    uint64_t* d_ra_output;
    CUDA_CHECK(cudaMalloc(&d_decomp_output, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_ra_output, N * sizeof(uint64_t)));

    std::cout << "\n--- Testing Decompression ---" << std::endl;
    launchDecompressWarpOpt<uint64_t>(compressed, d_decomp_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> decomp_result(N);
    CUDA_CHECK(cudaMemcpy(decomp_result.data(), d_decomp_output, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int decomp_errors = 0;
    for (size_t i = 0; i < N; i++) {
        if (decomp_result[i] != data[i]) {
            if (decomp_errors < 10) {
                std::cout << "  Decomp MISMATCH[" << i << "]: got " << decomp_result[i]
                          << ", expected " << data[i]
                          << " (diff=" << (int64_t)(decomp_result[i] - data[i]) << ")" << std::endl;
            }
            decomp_errors++;
        }
    }
    std::cout << "Decompression: " << (decomp_errors == 0 ? "PASS" : "FAIL")
              << " (" << decomp_errors << " errors)" << std::endl;

    std::cout << "\n--- Testing Random Access (Sequential) ---" << std::endl;
    std::vector<int> seq_indices(N);
    for (size_t i = 0; i < N; i++) seq_indices[i] = i;

    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, seq_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    RandomAccessStats stats;
    randomAccessOptimized(compressed, d_indices, N, d_ra_output, &stats, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> ra_result(N);
    CUDA_CHECK(cudaMemcpy(ra_result.data(), d_ra_output, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int ra_errors = 0;
    int ra_vs_decomp_diff = 0;
    int first_error_idx = -1;
    for (size_t i = 0; i < N; i++) {
        if (ra_result[i] != data[i]) {
            if (ra_errors < 10) {
                int partition_idx = -1;
                for (size_t p = 0; p < partitions.size(); p++) {
                    if ((int)i >= partitions[p].start_idx && (int)i < partitions[p].end_idx) {
                        partition_idx = p;
                        break;
                    }
                }

                std::cout << "  RA MISMATCH[" << i << "] (partition " << partition_idx;
                if (partition_idx >= 0) {
                    std::cout << ", model=" << partitions[partition_idx].model_type
                              << ", delta_bits=" << partitions[partition_idx].delta_bits;
                }
                std::cout << "): got " << ra_result[i]
                          << ", expected " << data[i]
                          << " (diff=" << (int64_t)(ra_result[i] - data[i]) << ")";

                if (ra_result[i] == decomp_result[i]) {
                    std::cout << " [matches decomp]";
                } else {
                    std::cout << " [decomp=" << decomp_result[i] << "]";
                }
                std::cout << std::endl;

                if (first_error_idx < 0) first_error_idx = i;
            }
            ra_errors++;

            if (ra_result[i] != decomp_result[i]) {
                ra_vs_decomp_diff++;
            }
        }
    }
    std::cout << "Random Access: " << (ra_errors == 0 ? "PASS" : "FAIL")
              << " (" << ra_errors << " errors, " << ra_vs_decomp_diff << " differ from decomp)" << std::endl;

    if (first_error_idx >= 0) {
        std::cout << "\n--- Detailed Debug for Index " << first_error_idx << " ---" << std::endl;

        int partition_idx = -1;
        for (size_t p = 0; p < partitions.size(); p++) {
            if (first_error_idx >= partitions[p].start_idx && first_error_idx < partitions[p].end_idx) {
                partition_idx = p;
                break;
            }
        }

        if (partition_idx >= 0) {
            const auto& part = partitions[partition_idx];
            std::cout << "Partition " << partition_idx << ":" << std::endl;
            std::cout << "  start_idx: " << part.start_idx << std::endl;
            std::cout << "  end_idx: " << part.end_idx << std::endl;
            std::cout << "  model_type: " << part.model_type << std::endl;
            std::cout << "  delta_bits: " << part.delta_bits << std::endl;
            std::cout << "  theta0 (model_params[0]): " << part.model_params[0] << std::endl;
            std::cout << "  theta1 (model_params[1]): " << part.model_params[1] << std::endl;
            std::cout << "  theta2 (model_params[2]): " << part.model_params[2] << std::endl;
            std::cout << "  theta3 (model_params[3]): " << part.model_params[3] << std::endl;

            int local_idx = first_error_idx - part.start_idx;
            std::cout << "  local_idx: " << local_idx << std::endl;

            double x = static_cast<double>(local_idx);
            double predicted = 0;
            switch (part.model_type) {
                case MODEL_LINEAR:
                    predicted = part.model_params[0] + part.model_params[1] * x;
                    break;
                case MODEL_POLYNOMIAL2:
                    predicted = part.model_params[0] + x * (part.model_params[1] + x * part.model_params[2]);
                    break;
                case MODEL_FOR_BITPACK:
                    predicted = part.model_params[0];
                    break;
            }
            std::cout << "  predicted (raw): " << predicted << std::endl;

            if (predicted < 0.0) {
                std::cout << "  predicted (clamped): 0.0" << std::endl;
                predicted = 0.0;
            }

            int64_t pred_int = static_cast<int64_t>(llround(predicted));
            std::cout << "  predicted (int64): " << pred_int << std::endl;

            uint64_t expected = data[first_error_idx];
            int64_t expected_delta = static_cast<int64_t>(expected) - pred_int;
            std::cout << "  expected value: " << expected << std::endl;
            std::cout << "  expected delta: " << expected_delta << std::endl;
            std::cout << "  RA returned: " << ra_result[first_error_idx] << std::endl;
            std::cout << "  Decomp returned: " << decomp_result[first_error_idx] << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_decomp_output));
    CUDA_CHECK(cudaFree(d_ra_output));
    CUDA_CHECK(cudaFree(d_indices));
    freeCompressedData(compressed);

    return (ra_errors > 0) ? 1 : 0;
}
