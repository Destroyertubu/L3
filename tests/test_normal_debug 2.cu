/**
 * Debug test for GPU Merge on a subset of normal data
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <cstring>

#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"

template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    uint64_t header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    size_t num_elements;
    size_t data_start;

    if (header > 0 && header * sizeof(T) <= file_size - 8) {
        num_elements = header;
        data_start = 8;
    } else {
        file.seekg(0, std::ios::beg);
        num_elements = file_size / sizeof(T);
        data_start = 0;
    }

    if (max_elements > 0 && num_elements > max_elements) {
        num_elements = max_elements;
    }

    std::vector<T> data(num_elements);
    file.seekg(data_start, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));

    return data;
}

int main() {
    std::cout << "=== Debug GPU Merge on Normal Data ===" << std::endl;

    // Load a small subset for debugging
    std::string normal_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin";
    auto data = loadBinaryFile<uint64_t>(normal_file, 50000);

    if (data.empty()) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    CostOptimalConfig config = CostOptimalConfig::balanced();
    config.max_merge_rounds = 1;  // Limit to 1 round for easier debugging

    std::cout << "\nConfig:" << std::endl;
    std::cout << "  analysis_block_size: " << config.analysis_block_size << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  max_partition_size: " << config.max_partition_size << std::endl;
    std::cout << "  merge_benefit_threshold: " << config.merge_benefit_threshold << std::endl;
    std::cout << "  max_merge_rounds: " << config.max_merge_rounds << std::endl;

    // GPU version
    std::cout << "\n--- GPU Version ---" << std::endl;
    GPUCostOptimalPartitioner<uint64_t> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    std::cout << "GPU partitions: " << gpu_result.size() << std::endl;
    if (gpu_result.size() <= 30) {
        for (size_t i = 0; i < gpu_result.size(); i++) {
            std::cout << "  [" << i << "] " << gpu_result[i].start_idx << " - "
                      << gpu_result[i].end_idx << " (bits: " << gpu_result[i].delta_bits << ")" << std::endl;
        }
    }

    // CPU version
    std::cout << "\n--- CPU Version ---" << std::endl;
    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    std::cout << "CPU partitions: " << cpu_result.size() << std::endl;
    if (cpu_result.size() <= 30) {
        for (size_t i = 0; i < cpu_result.size(); i++) {
            std::cout << "  [" << i << "] " << cpu_result[i].start_idx << " - "
                      << cpu_result[i].end_idx << " (bits: " << cpu_result[i].delta_bits << ")" << std::endl;
        }
    }

    // Compare
    std::cout << "\n--- Comparison ---" << std::endl;
    if (gpu_result.size() != cpu_result.size()) {
        std::cout << "Partition count mismatch: GPU=" << gpu_result.size()
                  << " CPU=" << cpu_result.size() << std::endl;
    } else {
        int mismatches = 0;
        for (size_t i = 0; i < gpu_result.size() && i < 10; i++) {
            if (gpu_result[i].start_idx != cpu_result[i].start_idx ||
                gpu_result[i].end_idx != cpu_result[i].end_idx) {
                std::cout << "Mismatch at " << i << ": GPU[" << gpu_result[i].start_idx
                          << ", " << gpu_result[i].end_idx << "] vs CPU["
                          << cpu_result[i].start_idx << ", " << cpu_result[i].end_idx << "]" << std::endl;
                mismatches++;
            }
        }
        if (mismatches == 0) {
            std::cout << "All partitions match!" << std::endl;
        }
    }

    return 0;
}
