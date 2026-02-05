/**
 * Debug V2 Partitioner for Linear Dataset
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "L3_format.hpp"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

// Load raw binary dataset
std::vector<uint64_t> loadRawDataset(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t count = file_size / sizeof(uint64_t);
    if (max_elements > 0 && count > max_elements) {
        count = max_elements;
    }

    std::vector<uint64_t> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint64_t));

    return data;
}

int main(int argc, char** argv) {
    std::string filename = "data/sosd/1-linear_200M_uint64.bin";
    size_t max_elements = 10000000;  // 10M for testing

    std::cout << "Debug V2 Partitioner for Linear Dataset" << std::endl;
    std::cout << "========================================" << std::endl;

    auto data = loadRawDataset(filename, max_elements);
    if (data.empty()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Check data range
    uint64_t global_min = data[0], global_max = data[0];
    for (size_t i = 1; i < data.size(); i++) {
        global_min = std::min(global_min, data[i]);
        global_max = std::max(global_max, data[i]);
    }
    uint64_t range = global_max - global_min;
    int for_bits = (range > 0) ? (64 - __builtin_clzll(range)) : 0;

    std::cout << "\n=== Data Range ===" << std::endl;
    std::cout << "  global_min = " << global_min << std::endl;
    std::cout << "  global_max = " << global_max << std::endl;
    std::cout << "  range = " << range << std::endl;
    std::cout << "  FOR bits = " << for_bits << std::endl;

    // Configure partitioner
    CostOptimalConfig config;
    config.min_partition_size = 256;
    config.max_partition_size = 8192;
    config.target_partition_size = 2048;
    config.polynomial_cost_threshold = 0.95f;  // 5% improvement required
    config.enable_polynomial_models = true;
    config.polynomial_min_size = 64;
    config.cubic_min_size = 128;

    std::cout << "\n=== Config ===" << std::endl;
    std::cout << "  polynomial_cost_threshold = " << config.polynomial_cost_threshold << std::endl;
    std::cout << "  min_partition_size = " << config.min_partition_size << std::endl;
    std::cout << "  max_partition_size = " << config.max_partition_size << std::endl;
    std::cout << "  target_partition_size = " << config.target_partition_size << std::endl;

    std::cout << "\nRunning V2 Partitioner..." << std::endl;

    GPUCostOptimalPartitionerV2<uint64_t> partitioner(data, config);
    auto partitions = partitioner.partition();

    std::cout << "Total partitions: " << partitions.size() << std::endl;

    // Count models
    int linear_count = 0, poly2_count = 0, poly3_count = 0, for_count = 0;
    long long total_delta_bits = 0;
    for (const auto& p : partitions) {
        int size = p.end_idx - p.start_idx;
        total_delta_bits += (long long)size * p.delta_bits;

        if (p.model_type == MODEL_LINEAR) linear_count++;
        else if (p.model_type == MODEL_POLYNOMIAL2) poly2_count++;
        else if (p.model_type == MODEL_POLYNOMIAL3) poly3_count++;
        else if (p.model_type == MODEL_FOR_BITPACK) for_count++;
    }

    std::cout << "\n=== Model Distribution ===" << std::endl;
    std::cout << "LINEAR: " << linear_count << std::endl;
    std::cout << "POLY2: " << poly2_count << std::endl;
    std::cout << "POLY3: " << poly3_count << std::endl;
    std::cout << "FOR_BITPACK: " << for_count << std::endl;

    // Show first few partitions
    std::cout << "\n=== First 5 Partitions ===" << std::endl;
    const char* model_names[] = {"CONST", "LINEAR", "POLY2", "POLY3", "FOR", "DIRECT"};
    for (size_t i = 0; i < std::min((size_t)5, partitions.size()); i++) {
        const auto& p = partitions[i];
        std::cout << "Partition " << i << ": [" << p.start_idx << ", " << p.end_idx << ")"
                  << " model=" << model_names[p.model_type]
                  << " delta_bits=" << p.delta_bits
                  << " theta0=" << std::scientific << p.model_params[0]
                  << " theta1=" << p.model_params[1]
                  << std::endl;
    }

    // Calculate theoretical compression ratio
    size_t original_bytes = data.size() * sizeof(uint64_t);
    size_t delta_bytes = (total_delta_bits + 7) / 8;
    size_t metadata_bytes = partitions.size() * 44;  // Approximate metadata size
    size_t compressed_bytes = delta_bytes + metadata_bytes;

    std::cout << "\n=== Compression Analysis ===" << std::endl;
    std::cout << "Original: " << original_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Delta array: " << delta_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Metadata: " << metadata_bytes / (1024.0) << " KB" << std::endl;
    std::cout << "Total compressed: " << compressed_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Theoretical ratio: " << std::fixed << std::setprecision(2)
              << (double)original_bytes / compressed_bytes << "x" << std::endl;

    // Verify a few partitions manually
    std::cout << "\n=== Manual Verification of First Partition ===" << std::endl;
    if (!partitions.empty()) {
        const auto& p = partitions[0];
        int n = p.end_idx - p.start_idx;

        // Calculate linear fit
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_x2 = 0;
        for (int i = 0; i < n; i++) {
            double x = i;
            double y = static_cast<double>(data[p.start_idx + i]);
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }
        double dn = static_cast<double>(n);
        double det = dn * sum_x2 - sum_x * sum_x;
        double theta0_linear = (sum_x2 * sum_y - sum_x * sum_xy) / det;
        double theta1_linear = (dn * sum_xy - sum_x * sum_y) / det;

        // Calculate max error for linear
        long long linear_max_err = 0;
        for (int i = 0; i < n; i++) {
            double x = i;
            double pred = theta0_linear + theta1_linear * x;
            int64_t pv = static_cast<int64_t>(std::llrint(pred));
            int64_t err = static_cast<int64_t>(data[p.start_idx + i]) - pv;
            linear_max_err = std::max(linear_max_err, (long long)std::abs(err));
        }
        int linear_bits = (linear_max_err > 0) ? (64 - __builtin_clzll(linear_max_err)) + 1 : 0;

        // Calculate FOR range for this partition
        uint64_t part_min = data[p.start_idx], part_max = data[p.start_idx];
        for (int i = 1; i < n; i++) {
            part_min = std::min(part_min, data[p.start_idx + i]);
            part_max = std::max(part_max, data[p.start_idx + i]);
        }
        uint64_t part_range = part_max - part_min;
        int for_bits_part = (part_range > 0) ? (64 - __builtin_clzll(part_range)) : 0;

        // Calculate costs
        float linear_cost = 16.0f + n * linear_bits / 8.0f;
        float for_cost = 8.0f + n * for_bits_part / 8.0f;

        std::cout << "Partition size: " << n << std::endl;
        std::cout << "\nLINEAR model:" << std::endl;
        std::cout << "  theta0 = " << std::scientific << theta0_linear << std::endl;
        std::cout << "  theta1 = " << theta1_linear << std::endl;
        std::cout << "  max_error = " << linear_max_err << std::endl;
        std::cout << "  bits = " << linear_bits << std::endl;
        std::cout << "  cost = " << std::fixed << linear_cost << " bytes" << std::endl;

        std::cout << "\nFOR model:" << std::endl;
        std::cout << "  part_min = " << part_min << std::endl;
        std::cout << "  part_max = " << part_max << std::endl;
        std::cout << "  range = " << part_range << std::endl;
        std::cout << "  bits = " << for_bits_part << std::endl;
        std::cout << "  cost = " << for_cost << " bytes" << std::endl;

        std::cout << "\nComparison:" << std::endl;
        std::cout << "  LINEAR cost (" << linear_cost << ") < FOR cost * 0.95 (" << (for_cost * 0.95f) << ")? "
                  << (linear_cost < for_cost * 0.95f ? "YES" : "NO") << std::endl;
        std::cout << "  Actual selected model: " << model_names[p.model_type] << std::endl;
        std::cout << "  Actual delta_bits: " << p.delta_bits << std::endl;
    }

    return 0;
}
