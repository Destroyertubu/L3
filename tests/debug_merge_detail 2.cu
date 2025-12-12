#include <iostream>
#include <vector>
#include <cstdint>
#include <fstream>
#include <cuda_runtime.h>
#include "../src/kernels/compression/encoder_cost_optimal_gpu_merge.cuh"

using namespace l3::codec;

int main() {
    // Load 10000 elements from normal dataset
    std::vector<uint64_t> data(10000);
    std::ifstream f("/root/autodl-tmp/test/data/sosd/normal_200M_uint64.bin", std::ios::binary);
    if (!f) { std::cerr << "Cannot open file\n"; return 1; }
    f.read(reinterpret_cast<char*>(data.data()), 10000 * sizeof(uint64_t));
    f.close();

    std::cout << "Loaded 10000 elements\n";
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; i++) std::cout << data[i] << " ";
    std::cout << "\n\n";

    // GPU partition
    GPUCostOptimalConfig config;
    config.target_partition_size = 2048;
    config.min_partition_size = 512;
    config.max_partition_size = 8192;
    config.enable_merging = true;
    config.max_merge_rounds = 4;
    config.benefit_threshold = 0.05f;

    GPUCostOptimalPartitioner<uint64_t> partitioner(data, config);
    auto result = partitioner.partition();

    std::cout << "=== GPU Partitioning ===\n";
    std::cout << "Number of partitions: " << result.size() << "\n\n";

    // Check first few partitions for detailed stats
    std::cout << "First 5 partitions:\n";
    for (int i = 0; i < std::min(5, (int)result.size()); i++) {
        const auto& p = result[i];
        std::cout << "  [" << i << "] range=[" << p.start_idx << ", " << p.end_idx << ") "
                  << "size=" << (p.end_idx - p.start_idx) << " "
                  << "theta0=" << p.theta0 << " theta1=" << p.theta1 << " "
                  << "bits=" << p.delta_bits << "\n";
    }

    // Check contiguity
    std::cout << "\nChecking partition contiguity:\n";
    bool all_contiguous = true;
    for (int i = 0; i < (int)result.size() - 1; i++) {
        if (result[i].end_idx != result[i+1].start_idx) {
            std::cout << "  Gap at " << i << ": end=" << result[i].end_idx
                      << " next_start=" << result[i+1].start_idx << "\n";
            all_contiguous = false;
        }
    }
    if (all_contiguous) {
        std::cout << "  All partitions are contiguous!\n";
    }

    // Now compare with CPU version
    std::cout << "\n=== CPU Partitioning ===\n";
    CostOptimalConfig cpu_config;
    cpu_config.target_partition_size = 2048;
    cpu_config.min_partition_size = 512;
    cpu_config.max_partition_size = 8192;
    cpu_config.enable_merging = true;
    cpu_config.max_merge_rounds = 4;
    cpu_config.merge_benefit_threshold = 0.05f;

    int num_cpu_partitions = 0;
    auto cpu_result = createPartitionsCostOptimal(data, cpu_config, &num_cpu_partitions);
    std::cout << "Number of partitions: " << cpu_result.size() << "\n\n";

    std::cout << "First 5 partitions:\n";
    for (int i = 0; i < std::min(5, (int)cpu_result.size()); i++) {
        const auto& p = cpu_result[i];
        std::cout << "  [" << i << "] range=[" << p.start_idx << ", " << p.end_idx << ") "
                  << "size=" << (p.end_idx - p.start_idx) << " "
                  << "theta0=" << p.theta0 << " theta1=" << p.theta1 << " "
                  << "bits=" << p.delta_bits << "\n";
    }

    return 0;
}
