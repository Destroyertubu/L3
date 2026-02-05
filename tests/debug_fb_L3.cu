/**
 * Debug test for L3 decompression on fb dataset
 * Finds exactly which partition and index is causing errors
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>

// L3 headers
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "L3_opt.h"

// Partitioner header
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// External decoder
template<typename T>
void launchDecompressWarpOpt(const CompressedDataL3<T>*, T*, cudaStream_t);

// Load fb dataset
bool loadDataset(const std::string& path, std::vector<uint64_t>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    // Check for SOSD header (8 bytes = element count)
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    // Check if reasonable element count
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (count * sizeof(uint64_t) + sizeof(uint64_t) == file_size) {
        // SOSD format with header
        file.read(reinterpret_cast<char*>(&count), sizeof(count));
        data.resize(count);
        file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint64_t));
    } else {
        // Raw format without header
        count = file_size / sizeof(uint64_t);
        data.resize(count);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }
    return true;
}

int main() {
    std::cout << "=== L3 fb Dataset Debug Test ===" << std::endl;

    // Load fb dataset
    std::string path = "data/sosd/6-fb_200M_uint64.bin";
    std::vector<uint64_t> data;

    if (!loadDataset(path, data)) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Generate partitions with V2 partitioner
    CostOptimalConfig config;
    config.target_partition_size = 4096;
    config.analysis_block_size = 2048;
    config.min_partition_size = 256;
    config.max_partition_size = 8192;
    config.breakpoint_threshold = 2;
    config.merge_benefit_threshold = 0.05f;
    config.max_merge_rounds = 4;
    config.enable_merging = false;  // DISABLE merging to test initial partitioning
    config.enable_polynomial_models = false;  // DISABLE poly models
    config.polynomial_min_size = 10;
    config.cubic_min_size = 20;
    config.polynomial_cost_threshold = 0.95f;

    GPUCostOptimalPartitionerV2<uint64_t> partitioner(data, config, 0);
    std::vector<PartitionInfo> partitions = partitioner.partition();

    // DEBUG: Print last 3 partitions from the partitioner
    std::cout << "\n=== Last 3 partitions from V2 partitioner ===" << std::endl;
    std::cout << "data_size = " << data.size() << std::endl;
    for (size_t i = partitions.size() - 3; i < partitions.size(); i++) {
        const auto& p = partitions[i];
        std::cout << "Partition[" << i << "]: start=" << p.start_idx << ", end=" << p.end_idx
                  << ", model=" << p.model_type << ", theta0=" << p.model_params[0] << ", theta1=" << p.model_params[1]
                  << ", delta_bits=" << p.delta_bits << std::endl;

        // Manually compute what theta should be for last partition
        if (i == partitions.size() - 1) {
            int start = p.start_idx;
            int end = p.end_idx;
            int n = end - start;
            std::cout << "  Last partition has " << n << " elements" << std::endl;
            std::cout << "  First 5 data values: ";
            for (int k = 0; k < std::min(5, n); k++) {
                std::cout << data[start + k] << " ";
            }
            std::cout << std::endl;
            std::cout << "  Last 5 data values: ";
            for (int k = std::max(0, n-5); k < n; k++) {
                std::cout << data[start + k] << " ";
            }
            std::cout << std::endl;

            double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
            for (int j = 0; j < n; j++) {
                double x = (double)j;
                double y = (double)data[start + j];
                sum_x += x;
                sum_y += y;
                sum_xx += x * x;
                sum_xy += x * y;
            }
            double dn = (double)n;
            double det = dn * sum_xx - sum_x * sum_x;
            double theta0_correct = 0, theta1_correct = 0;
            if (std::fabs(det) > 1e-10) {
                theta1_correct = (dn * sum_xy - sum_x * sum_y) / det;
                theta0_correct = (sum_y - theta1_correct * sum_x) / dn;
            } else {
                theta0_correct = sum_y / dn;
            }
            std::cout << "  sum_y=" << sum_y << ", sum_xy=" << sum_xy << std::endl;
            std::cout << "  CORRECT theta: theta0=" << theta0_correct << ", theta1=" << theta1_correct << std::endl;
        }

        // Also verify data values at start
        if (p.start_idx < (int)data.size()) {
            std::cout << "  data[" << p.start_idx << "] = " << data[p.start_idx] << std::endl;
        }
    }
    std::cout << std::endl;

    // Count model types
    int linear = 0, poly2 = 0, poly3 = 0, for_bp = 0;
    for (const auto& p : partitions) {
        if (p.model_type == MODEL_LINEAR || p.model_type == MODEL_CONSTANT) linear++;
        else if (p.model_type == MODEL_POLYNOMIAL2) poly2++;
        else if (p.model_type == MODEL_POLYNOMIAL3) poly3++;
        else if (p.model_type == MODEL_FOR_BITPACK) for_bp++;
    }
    std::cout << "Partitions: " << partitions.size()
              << " (LINEAR=" << linear << ", POLY2=" << poly2
              << ", POLY3=" << poly3 << ", FOR=" << for_bp << ")" << std::endl;

    // Compress using L3_codec API
    auto* compressed = compressDataWithPartitions(data, partitions, nullptr);
    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }
    std::cout << "Compressed: " << compressed->num_partitions << " partitions" << std::endl;

    // Copy compressed metadata back to host for analysis
    std::vector<int> h_model_types(compressed->num_partitions);
    std::vector<double> h_model_params(compressed->num_partitions * 4);
    std::vector<int> h_start_indices(compressed->num_partitions);
    std::vector<int> h_end_indices(compressed->num_partitions);
    std::vector<int> h_delta_bits(compressed->num_partitions);

    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed->d_model_types,
        compressed->num_partitions * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_model_params.data(), compressed->d_model_params,
        compressed->num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_start_indices.data(), compressed->d_start_indices,
        compressed->num_partitions * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end_indices.data(), compressed->d_end_indices,
        compressed->num_partitions * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
        compressed->num_partitions * sizeof(int), cudaMemcpyDeviceToHost));

    // Decompress
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_output, 0, data.size() * sizeof(uint64_t)));

    launchDecompressWarpOpt<uint64_t>(compressed, d_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back output
    std::vector<uint64_t> output(data.size());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Find errors and analyze them
    int total_errors = 0;
    int first_poly2_error_partition = -1;

    for (size_t i = 0; i < data.size(); i++) {
        if (output[i] != data[i]) {
            // Find which partition this belongs to
            int partition_idx = -1;
            for (size_t p = 0; p < (size_t)compressed->num_partitions; p++) {
                int start = h_start_indices[p];
                int end = h_end_indices[p];
                if ((int)i >= start && (int)i < end) {
                    partition_idx = p;
                    break;
                }
            }

            if (total_errors < 10) {
                int local_idx = i - h_start_indices[partition_idx];
                std::cout << "\nError at global[" << i << "], partition[" << partition_idx << "], local[" << local_idx << "]:" << std::endl;
                std::cout << "  Got: " << output[i] << ", Expected: " << data[i] << std::endl;
                std::cout << "  Diff: " << (int64_t)(output[i] - data[i]) << std::endl;
                std::cout << "  Model: " << h_model_types[partition_idx] << " (";
                switch (h_model_types[partition_idx]) {
                    case 0: std::cout << "CONSTANT"; break;
                    case 1: std::cout << "LINEAR"; break;
                    case 2: std::cout << "POLY2"; break;
                    case 3: std::cout << "POLY3"; break;
                    case 5: std::cout << "FOR"; break;
                    default: std::cout << "UNKNOWN"; break;
                }
                std::cout << ")" << std::endl;
                std::cout << "  Params: theta0=" << h_model_params[partition_idx * 4]
                          << ", theta1=" << h_model_params[partition_idx * 4 + 1]
                          << ", theta2=" << h_model_params[partition_idx * 4 + 2]
                          << ", theta3=" << h_model_params[partition_idx * 4 + 3] << std::endl;
                std::cout << "  Delta bits: " << h_delta_bits[partition_idx] << std::endl;
                std::cout << "  Partition: start=" << h_start_indices[partition_idx]
                          << ", end=" << h_end_indices[partition_idx] << std::endl;

                // Compute what prediction should be
                double x = (double)local_idx;
                double theta0 = h_model_params[partition_idx * 4];
                double theta1 = h_model_params[partition_idx * 4 + 1];
                double theta2 = h_model_params[partition_idx * 4 + 2];
                double theta3 = h_model_params[partition_idx * 4 + 3];

                double pred_linear = theta0 + theta1 * x;
                double pred_poly2 = theta0 + x * (theta1 + x * theta2);
                double pred_poly3 = theta0 + x * (theta1 + x * (theta2 + x * theta3));

                std::cout << "  Predictions: linear=" << (int64_t)pred_linear
                          << ", poly2=" << (int64_t)pred_poly2
                          << ", poly3=" << (int64_t)pred_poly3 << std::endl;

                // Show original delta that should have been encoded
                int64_t expected_delta_linear = data[i] - (int64_t)pred_linear;
                int64_t expected_delta_poly2 = data[i] - (int64_t)pred_poly2;
                std::cout << "  Expected delta (linear): " << expected_delta_linear << std::endl;
                std::cout << "  Expected delta (poly2): " << expected_delta_poly2 << std::endl;

                if (h_model_types[partition_idx] == 2 && first_poly2_error_partition < 0) {
                    first_poly2_error_partition = partition_idx;
                }
            }
            total_errors++;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total errors: " << total_errors << " / " << data.size() << std::endl;

    if (first_poly2_error_partition >= 0) {
        std::cout << "First POLY2 error in partition: " << first_poly2_error_partition << std::endl;
    }

    std::cout << "Result: " << (total_errors == 0 ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_output);
    freeCompressedData(compressed);

    return total_errors > 0 ? 1 : 0;
}
