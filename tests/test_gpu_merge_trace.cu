/**
 * Detailed Debug for GPU Merge
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"

// Helper to read GPU array
template<typename T>
std::vector<T> readGPUArray(T* d_arr, int n) {
    std::vector<T> h_arr(n);
    cudaMemcpy(h_arr.data(), d_arr, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h_arr;
}

template<typename T>
std::vector<T> generateSortedData(size_t size) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(i);
    }
    return data;
}

int main() {
    std::cout << "=== Detailed Debug GPU Merge ===" << std::endl;

    // Small sorted data
    auto data = generateSortedData<uint32_t>(10000);

    CostOptimalConfig config = CostOptimalConfig::balanced();

    std::cout << "\nConfig:" << std::endl;
    std::cout << "  max_partition_size: " << config.max_partition_size << std::endl;

    // Run CPU version first
    std::cout << "\n--- CPU Version ---" << std::endl;
    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);
    std::cout << "CPU result: " << cpu_result.size() << " partitions" << std::endl;
    for (const auto& p : cpu_result) {
        std::cout << "  [" << p.start_idx << ", " << p.end_idx << ") size="
                  << (p.end_idx - p.start_idx) << " bits=" << p.delta_bits << std::endl;
    }

    // Manual GPU trace
    std::cout << "\n--- GPU Trace ---" << std::endl;

    uint32_t* d_data;
    cudaMalloc(&d_data, data.size() * sizeof(uint32_t));
    cudaMemcpy(d_data, data.data(), data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int data_size = data.size();
    int num_analysis_blocks = (data_size + config.analysis_block_size - 1) / config.analysis_block_size;

    std::cout << "num_analysis_blocks: " << num_analysis_blocks << std::endl;

    // Allocate arrays for delta-bits analysis
    int* d_delta_bits;
    int* d_is_breakpoint;
    cudaMalloc(&d_delta_bits, num_analysis_blocks * sizeof(int));
    cudaMalloc(&d_is_breakpoint, num_analysis_blocks * sizeof(int));

    // Compute delta-bits (simulating the kernel)
    std::vector<int> h_delta_bits(num_analysis_blocks);
    for (int bid = 0; bid < num_analysis_blocks; bid++) {
        int start = bid * config.analysis_block_size;
        int end = std::min(start + config.analysis_block_size, data_size);
        // For sorted data starting at 0: y = x, so theta0 = 0, theta1 = 1
        // All predictions are exact, so delta_bits = 0
        h_delta_bits[bid] = 0;
    }
    std::cout << "delta_bits per block: ";
    for (int i = 0; i < num_analysis_blocks; i++) {
        std::cout << h_delta_bits[i] << " ";
    }
    std::cout << std::endl;

    // Detect breakpoints
    std::vector<int> breakpoint_positions;
    for (int i = 0; i < num_analysis_blocks; i++) {
        if (i == 0 || abs(h_delta_bits[i] - h_delta_bits[i-1]) >= config.breakpoint_threshold) {
            breakpoint_positions.push_back(i * config.analysis_block_size);
        }
    }
    std::cout << "breakpoint_positions: ";
    for (int pos : breakpoint_positions) {
        std::cout << pos << " ";
    }
    std::cout << std::endl;

    // Create partitions within segments
    int num_breakpoints = breakpoint_positions.size();
    std::vector<int> partition_starts, partition_ends;

    for (int seg_idx = 0; seg_idx < num_breakpoints; seg_idx++) {
        int seg_start = breakpoint_positions[seg_idx];
        int seg_end = (seg_idx + 1 < num_breakpoints) ?
                      breakpoint_positions[seg_idx + 1] : data_size;
        int seg_len = seg_end - seg_start;

        int part_size = config.target_partition_size;
        int num_parts = (seg_len + part_size - 1) / part_size;
        if (num_parts > 0) {
            part_size = (seg_len + num_parts - 1) / num_parts;
            part_size = ((part_size + 31) / 32) * 32;
            part_size = std::max(config.min_partition_size,
                                std::min(config.max_partition_size, part_size));
        }

        std::cout << "Segment " << seg_idx << ": [" << seg_start << ", " << seg_end
                  << ") len=" << seg_len << " part_size=" << part_size << std::endl;

        for (int pos = seg_start; pos < seg_end; pos += part_size) {
            partition_starts.push_back(pos);
            partition_ends.push_back(std::min(pos + part_size, seg_end));
        }
    }

    std::cout << "\nInitial partitions (" << partition_starts.size() << "):" << std::endl;
    for (size_t i = 0; i < partition_starts.size(); i++) {
        int size = partition_ends[i] - partition_starts[i];
        std::cout << "  [" << i << "] [" << partition_starts[i] << ", " << partition_ends[i]
                  << ") size=" << size << std::endl;
    }

    // Simulate merge evaluation
    std::cout << "\nMerge evaluation (max_partition_size=" << config.max_partition_size << "):" << std::endl;
    for (size_t i = 0; i < partition_starts.size() - 1; i++) {
        int merged_size = partition_ends[i+1] - partition_starts[i];
        std::cout << "  Pair (" << i << "," << (i+1) << "): merged_size=" << merged_size;
        if (merged_size > config.max_partition_size) {
            std::cout << " -> TOO LARGE, cannot merge";
        } else {
            std::cout << " -> can merge";
        }
        std::cout << std::endl;
    }

    cudaFree(d_data);
    cudaFree(d_delta_bits);
    cudaFree(d_is_breakpoint);

    return 0;
}
