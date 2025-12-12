/**
 * Linear Dataset Model Distribution Analysis
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <map>

#include "L3_format.hpp"
#include "encoder_cost_optimal_gpu_merge.cuh"

// Load raw binary dataset (no header)
std::vector<uint64_t> loadRawDataset(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    // Get file size
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

void analyzeModelDistribution(const std::vector<PartitionInfo>& partitions, const std::vector<uint64_t>& data) {
    std::map<int, int> model_counts;
    std::map<int, std::vector<int>> model_delta_bits;

    for (const auto& p : partitions) {
        model_counts[p.model_type]++;
        model_delta_bits[p.model_type].push_back(p.delta_bits);
    }

    std::cout << "\n=== Model Distribution ===" << std::endl;
    std::cout << "Total partitions: " << partitions.size() << std::endl;

    const char* model_names[] = {"LINEAR", "POLY2", "POLY3", "FOR_BITPACK"};
    for (int m = 0; m <= 3; m++) {
        if (model_counts.count(m) && model_counts[m] > 0) {
            double pct = 100.0 * model_counts[m] / partitions.size();
            auto& bits = model_delta_bits[m];
            double avg_bits = 0;
            int min_bits = bits[0], max_bits = bits[0];
            for (int b : bits) {
                avg_bits += b;
                min_bits = std::min(min_bits, b);
                max_bits = std::max(max_bits, b);
            }
            avg_bits /= bits.size();

            std::cout << "  " << model_names[m] << ": " << model_counts[m]
                      << " (" << std::fixed << std::setprecision(1) << pct << "%)"
                      << "  delta_bits: avg=" << avg_bits << ", min=" << min_bits << ", max=" << max_bits
                      << std::endl;
        }
    }

    // Show first few data values
    std::cout << "\n=== First 20 data values ===" << std::endl;
    for (int i = 0; i < std::min(20, (int)data.size()); i++) {
        std::cout << "  data[" << i << "] = " << data[i] << std::endl;
    }

    // Show first partition details
    if (!partitions.empty()) {
        const auto& p = partitions[0];
        std::cout << "\n=== First Partition Details ===" << std::endl;
        std::cout << "  Range: [" << p.start_idx << ", " << p.end_idx << ")" << std::endl;
        std::cout << "  Model: " << model_names[p.model_type] << std::endl;
        std::cout << "  theta0: " << std::scientific << p.model_params[0] << std::endl;
        std::cout << "  theta1: " << p.model_params[1] << std::endl;
        std::cout << "  theta2: " << p.model_params[2] << std::endl;
        std::cout << "  theta3: " << p.model_params[3] << std::endl;
        std::cout << "  delta_bits: " << p.delta_bits << std::endl;

        // Verify predictions
        std::cout << "\n  Sample predictions:" << std::endl;
        for (int i = 0; i < std::min(5, p.end_idx - p.start_idx); i++) {
            double x = i;
            double pred = p.model_params[0] + x * p.model_params[1];
            if (p.model_type >= MODEL_POLYNOMIAL2) {
                pred = p.model_params[0] + x * (p.model_params[1] + x * p.model_params[2]);
            }
            if (p.model_type >= MODEL_POLYNOMIAL3) {
                pred = p.model_params[0] + x * (p.model_params[1] + x * (p.model_params[2] + x * p.model_params[3]));
            }
            int64_t actual = data[p.start_idx + i];
            int64_t predicted = std::llrint(pred);
            int64_t delta = actual - predicted;
            std::cout << "    x=" << i << ": actual=" << actual << ", pred=" << predicted << ", delta=" << delta << std::endl;
        }
    }

    // Calculate theoretical compression ratio
    size_t original_bytes = data.size() * sizeof(uint64_t);
    size_t metadata_per_partition = 4 * sizeof(double) + 3 * sizeof(int32_t); // params + model_type + delta_bits + indices
    size_t total_delta_bits = 0;
    for (const auto& p : partitions) {
        total_delta_bits += (p.end_idx - p.start_idx) * p.delta_bits;
    }
    size_t delta_bytes = (total_delta_bits + 7) / 8;
    size_t metadata_bytes = partitions.size() * metadata_per_partition;
    size_t compressed_bytes = delta_bytes + metadata_bytes;

    std::cout << "\n=== Compression Analysis ===" << std::endl;
    std::cout << "  Original: " << original_bytes << " bytes" << std::endl;
    std::cout << "  Delta array: " << delta_bytes << " bytes" << std::endl;
    std::cout << "  Metadata: " << metadata_bytes << " bytes" << std::endl;
    std::cout << "  Total compressed: " << compressed_bytes << " bytes" << std::endl;
    std::cout << "  Theoretical ratio: " << std::fixed << std::setprecision(2)
              << (double)original_bytes / compressed_bytes << "x" << std::endl;
}

int main(int argc, char** argv) {
    std::string filename = "/root/autodl-tmp/test/data/sosd/1-linear_200M_uint64.bin";
    size_t max_elements = 10000000;  // 10M for quick test

    std::cout << "Linear Dataset Model Distribution Analysis" << std::endl;
    std::cout << "===========================================" << std::endl;

    std::cout << "\nLoading dataset: " << filename << std::endl;

    auto data = loadRawDataset(filename, max_elements);
    if (data.empty()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Configure partitioner with polynomial models enabled
    CostOptimalConfig config = CostOptimalConfig::polynomialEnabled();
    config.min_partition_size = 256;
    config.max_partition_size = 8192;
    config.target_partition_size = 2048;

    std::cout << "\nRunning partitioner..." << std::endl;

    GPUCostOptimalPartitioner<uint64_t> partitioner(data, config);
    auto partitions = partitioner.partition();

    std::cout << "Partitioning complete." << std::endl;

    analyzeModelDistribution(partitions, data);

    return 0;
}
