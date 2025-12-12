/**
 * OSM Dataset Model Distribution Analysis
 *
 * Analyzes how LINEAR, POLY2, and POLY3 models are selected on real OSM data.
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <map>

#include "L3_format.hpp"
#include "encoder_cost_optimal_gpu_merge.cuh"

// Load SOSD format dataset
std::vector<uint64_t> loadSOSDDataset(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    // Read header (SOSD format: 8-byte count)
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));

    if (max_elements > 0 && count > max_elements) {
        count = max_elements;
    }

    std::vector<uint64_t> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint64_t));

    return data;
}

void analyzeModelDistribution(const std::vector<PartitionInfo>& partitions) {
    std::map<int, int> model_counts;
    std::map<int, std::vector<int>> model_delta_bits;
    std::map<int, std::vector<int>> model_sizes;

    for (const auto& p : partitions) {
        model_counts[p.model_type]++;
        model_delta_bits[p.model_type].push_back(p.delta_bits);
        model_sizes[p.model_type].push_back(p.end_idx - p.start_idx);
    }

    auto printStats = [](const std::vector<int>& v, const char* name) {
        if (v.empty()) return;
        double sum = 0, min_v = v[0], max_v = v[0];
        for (int x : v) {
            sum += x;
            min_v = std::min(min_v, (double)x);
            max_v = std::max(max_v, (double)x);
        }
        double avg = sum / v.size();
        std::cout << "      " << name << ": avg=" << std::fixed << std::setprecision(1)
                  << avg << ", min=" << (int)min_v << ", max=" << (int)max_v << std::endl;
    };

    std::cout << "\n=== Model Distribution ===" << std::endl;
    std::cout << "Total partitions: " << partitions.size() << std::endl;

    const char* model_names[] = {"LINEAR", "POLY2", "POLY3", "FOR_BITPACK"};
    for (int m = 0; m <= 3; m++) {
        if (model_counts.count(m) && model_counts[m] > 0) {
            double pct = 100.0 * model_counts[m] / partitions.size();
            std::cout << "\n  " << model_names[m] << ": " << model_counts[m]
                      << " partitions (" << std::fixed << std::setprecision(2) << pct << "%)" << std::endl;
            printStats(model_delta_bits[m], "delta_bits");
            printStats(model_sizes[m], "partition_size");
        }
    }

    // Show some example POLY3 partitions with their parameters
    std::cout << "\n=== Example POLY3 Partitions ===" << std::endl;
    int poly3_shown = 0;
    for (size_t i = 0; i < partitions.size() && poly3_shown < 5; i++) {
        const auto& p = partitions[i];
        if (p.model_type == MODEL_POLYNOMIAL3) {
            std::cout << "\n  Partition " << i << " [" << p.start_idx << ", " << p.end_idx << "):" << std::endl;
            std::cout << "    Size: " << (p.end_idx - p.start_idx) << " elements" << std::endl;
            std::cout << "    theta0: " << std::scientific << std::setprecision(6) << p.model_params[0] << std::endl;
            std::cout << "    theta1: " << p.model_params[1] << std::endl;
            std::cout << "    theta2: " << p.model_params[2] << std::endl;
            std::cout << "    theta3: " << p.model_params[3] << std::endl;
            std::cout << "    delta_bits: " << p.delta_bits << std::endl;
            poly3_shown++;
        }
    }

    // Show some example POLY2 partitions
    std::cout << "\n=== Example POLY2 Partitions ===" << std::endl;
    int poly2_shown = 0;
    for (size_t i = 0; i < partitions.size() && poly2_shown < 5; i++) {
        const auto& p = partitions[i];
        if (p.model_type == MODEL_POLYNOMIAL2) {
            std::cout << "\n  Partition " << i << " [" << p.start_idx << ", " << p.end_idx << "):" << std::endl;
            std::cout << "    Size: " << (p.end_idx - p.start_idx) << " elements" << std::endl;
            std::cout << "    theta0: " << std::scientific << std::setprecision(6) << p.model_params[0] << std::endl;
            std::cout << "    theta1: " << p.model_params[1] << std::endl;
            std::cout << "    theta2: " << p.model_params[2] << std::endl;
            std::cout << "    delta_bits: " << p.delta_bits << std::endl;
            poly2_shown++;
        }
    }

    // Show some example LINEAR partitions
    std::cout << "\n=== Example LINEAR Partitions ===" << std::endl;
    int linear_shown = 0;
    for (size_t i = 0; i < partitions.size() && linear_shown < 5; i++) {
        const auto& p = partitions[i];
        if (p.model_type == MODEL_LINEAR) {
            std::cout << "\n  Partition " << i << " [" << p.start_idx << ", " << p.end_idx << "):" << std::endl;
            std::cout << "    Size: " << (p.end_idx - p.start_idx) << " elements" << std::endl;
            std::cout << "    theta0: " << std::scientific << std::setprecision(6) << p.model_params[0] << std::endl;
            std::cout << "    theta1: " << p.model_params[1] << std::endl;
            std::cout << "    delta_bits: " << p.delta_bits << std::endl;
            linear_shown++;
        }
    }
}

int main(int argc, char** argv) {
    std::string filename = "/root/autodl-tmp/test/data/sosd/8-osm_cellids_800M_uint64.bin";
    size_t max_elements = 100000000;  // 100M elements for faster testing

    if (argc > 1) {
        max_elements = std::stoull(argv[1]);
    }

    std::cout << "OSM Dataset Model Distribution Analysis" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\nLoading dataset: " << filename << std::endl;
    std::cout << "Max elements: " << max_elements << std::endl;

    auto data = loadSOSDDataset(filename, max_elements);
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
    config.polynomial_min_size = 64;
    config.cubic_min_size = 128;
    config.polynomial_cost_threshold = 0.95f;  // 5% improvement required

    std::cout << "\nPartitioner configuration:" << std::endl;
    std::cout << "  min_partition_size: " << config.min_partition_size << std::endl;
    std::cout << "  max_partition_size: " << config.max_partition_size << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  polynomial_min_size: " << config.polynomial_min_size << std::endl;
    std::cout << "  cubic_min_size: " << config.cubic_min_size << std::endl;
    std::cout << "  polynomial_cost_threshold: " << config.polynomial_cost_threshold << std::endl;

    std::cout << "\nRunning partitioner..." << std::endl;

    // Run partitioner
    GPUCostOptimalPartitioner<uint64_t> partitioner(data, config);
    auto partitions = partitioner.partition();

    std::cout << "Partitioning complete." << std::endl;

    // Analyze results
    analyzeModelDistribution(partitions);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Analysis Complete!" << std::endl;

    return 0;
}
