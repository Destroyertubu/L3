/**
 * Debug test using actual SOSD normal dataset to identify random access failure
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
#include "L3.h"
#include "sosd_loader.h"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while (0)

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

int main(int argc, char** argv) {
    std::cout << "=== Debug Random Access with Real Dataset ===" << std::endl;

    // Use normal dataset (the one that fails in benchmark)
    std::string filename = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin";

    std::cout << "Loading " << filename << "..." << std::endl;
    std::vector<uint64_t> full_data;
    if (!SOSDLoader::loadDataset(filename, full_data)) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }

    // For debugging, use a smaller subset
    size_t N = (full_data.size() < 1000000) ? full_data.size() : 1000000;
    std::cout << "Using " << N << " / " << full_data.size() << " elements" << std::endl;

    std::vector<uint64_t> data(full_data.begin(), full_data.begin() + N);

    const int PARTITION_SIZE = 4096;

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

    // Test decompression first
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

    // Test random access with RANDOM indices (matching benchmark)
    std::cout << "\n--- Testing Random Access ---" << std::endl;
    std::vector<int> h_indices(N);
    std::mt19937 rng(42);  // Same seed as benchmark
    for (size_t i = 0; i < N; i++) {
        h_indices[i] = rng() % N;
    }

    int* d_indices;
    CUDA_CHECK(cudaMalloc(&d_indices, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    RandomAccessStats stats;
    randomAccessOptimized(compressed, d_indices, N, d_ra_output, &stats, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<uint64_t> ra_result(N);
    CUDA_CHECK(cudaMemcpy(ra_result.data(), d_ra_output, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int ra_errors = 0;
    int first_error_idx = -1;
    int first_error_query_idx = -1;
    for (size_t i = 0; i < N; i++) {
        int queried_idx = h_indices[i];
        if (ra_result[i] != data[queried_idx]) {
            if (ra_errors < 15) {
                int partition_idx = -1;
                for (size_t p = 0; p < partitions.size(); p++) {
                    if (queried_idx >= partitions[p].start_idx && queried_idx < partitions[p].end_idx) {
                        partition_idx = p;
                        break;
                    }
                }

                std::cout << "  RA MISMATCH query[" << i << "] idx=" << queried_idx
                          << " (partition " << partition_idx;
                if (partition_idx >= 0) {
                    std::cout << ", model=" << partitions[partition_idx].model_type
                              << ", delta_bits=" << partitions[partition_idx].delta_bits;
                }
                std::cout << "): got " << ra_result[i]
                          << ", expected " << data[queried_idx]
                          << " (diff=" << (int64_t)(ra_result[i] - data[queried_idx]) << ")";

                // Compare with decompression result for that index
                if (ra_result[i] == decomp_result[queried_idx]) {
                    std::cout << " [matches decomp]";
                } else {
                    std::cout << " [decomp=" << decomp_result[queried_idx] << "]";
                }
                std::cout << std::endl;

                if (first_error_idx < 0) {
                    first_error_idx = queried_idx;
                    first_error_query_idx = i;
                }
            }
            ra_errors++;
        }
    }
    std::cout << "Random Access: " << (ra_errors == 0 ? "PASS" : "FAIL")
              << " (" << ra_errors << " errors)" << std::endl;

    // Print detailed debug for first error
    if (first_error_idx >= 0) {
        std::cout << "\n--- Detailed Debug for queried index " << first_error_idx << " ---" << std::endl;

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
            std::cout << "  model_type: " << part.model_type
                      << " (" << (part.model_type == MODEL_LINEAR ? "LINEAR" :
                                  part.model_type == MODEL_POLYNOMIAL2 ? "POLY2" :
                                  part.model_type == MODEL_FOR_BITPACK ? "FOR" : "???") << ")" << std::endl;
            std::cout << "  delta_bits: " << part.delta_bits << std::endl;
            std::cout << "  model_params[0]: " << part.model_params[0] << std::endl;
            std::cout << "  model_params[1]: " << part.model_params[1] << std::endl;
            std::cout << "  model_params[2]: " << part.model_params[2] << std::endl;
            std::cout << "  model_params[3]: " << part.model_params[3] << std::endl;

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

            double predicted_clamped = predicted;
            if (predicted < 0.0) {
                std::cout << "  predicted < 0, clamping to 0" << std::endl;
                predicted_clamped = 0.0;
            }

            int64_t pred_int = static_cast<int64_t>(llround(predicted_clamped));
            std::cout << "  predicted (int64): " << pred_int << std::endl;

            uint64_t expected = data[first_error_idx];
            int64_t expected_delta = static_cast<int64_t>(expected) - pred_int;
            std::cout << "  expected value: " << expected << std::endl;
            std::cout << "  expected delta: " << expected_delta << std::endl;
            std::cout << "  RA returned: " << ra_result[first_error_query_idx] << std::endl;
            std::cout << "  Decomp returned: " << decomp_result[first_error_idx] << std::endl;
        }
    }

    CUDA_CHECK(cudaFree(d_decomp_output));
    CUDA_CHECK(cudaFree(d_ra_output));
    CUDA_CHECK(cudaFree(d_indices));
    freeCompressedData(compressed);

    return (ra_errors > 0) ? 1 : 0;
}
