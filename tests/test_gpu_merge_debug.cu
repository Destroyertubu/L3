/**
 * Debug test for GPU Merge
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"

template<typename T>
std::vector<T> generateSortedData(size_t size) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(i);
    }
    return data;
}

int main() {
    std::cout << "=== Debug GPU Merge ===" << std::endl;

    // Small sorted data - should result in 1 partition after merging
    auto data = generateSortedData<uint32_t>(10000);

    CostOptimalConfig config = CostOptimalConfig::balanced();

    std::cout << "\nConfig:" << std::endl;
    std::cout << "  analysis_block_size: " << config.analysis_block_size << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  min_partition_size: " << config.min_partition_size << std::endl;
    std::cout << "  max_partition_size: " << config.max_partition_size << std::endl;
    std::cout << "  merge_benefit_threshold: " << config.merge_benefit_threshold << std::endl;
    std::cout << "  max_merge_rounds: " << config.max_merge_rounds << std::endl;
    std::cout << "  enable_merging: " << config.enable_merging << std::endl;

    // GPU version
    std::cout << "\n--- GPU Version ---" << std::endl;
    GPUCostOptimalPartitioner<uint32_t> gpu_partitioner(data, config);
    auto gpu_result = gpu_partitioner.partition();

    std::cout << "GPU partitions: " << gpu_result.size() << std::endl;
    for (size_t i = 0; i < gpu_result.size(); i++) {
        std::cout << "  [" << i << "] " << gpu_result[i].start_idx << " - "
                  << gpu_result[i].end_idx << " (bits: " << gpu_result[i].delta_bits << ")" << std::endl;
    }

    // CPU version
    std::cout << "\n--- CPU Version ---" << std::endl;
    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    std::cout << "CPU partitions: " << cpu_result.size() << std::endl;
    for (size_t i = 0; i < cpu_result.size(); i++) {
        std::cout << "  [" << i << "] " << cpu_result[i].start_idx << " - "
                  << cpu_result[i].end_idx << " (bits: " << cpu_result[i].delta_bits << ")" << std::endl;
    }

    return 0;
}
