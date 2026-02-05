/**
 * Test GPU Merge on Normal Distribution Dataset
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <iomanip>

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

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for header (8 bytes for element count)
    uint64_t header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));

    size_t num_elements;
    size_t data_start;

    // Heuristic: if header value is reasonable, use it
    if (header > 0 && header * sizeof(T) <= file_size - 8) {
        num_elements = header;
        data_start = 8;
    } else {
        // No header, raw data
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

template<typename T>
void testDataset(const std::string& name, const std::vector<T>& data, int iterations = 3) {
    std::cout << "\n=== Testing " << name << " (" << data.size() << " elements) ===" << std::endl;

    CostOptimalConfig config = CostOptimalConfig::balanced();

    // Warm up
    {
        GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
        auto result = gpu_partitioner.partition();
    }
    {
        auto result = createPartitionsCostOptimal(data, config, nullptr);
    }

    // GPU timing
    double gpu_total_ms = 0;
    int gpu_partitions = 0;
    std::vector<PartitionInfo> gpu_result;

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitioner<T> gpu_partitioner(data, config);
        gpu_result = gpu_partitioner.partition();

        auto end = std::chrono::high_resolution_clock::now();
        gpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        gpu_partitions = gpu_result.size();
    }

    // CPU timing
    double cpu_total_ms = 0;
    int cpu_partitions = 0;
    std::vector<PartitionInfo> cpu_result;

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        cpu_result = createPartitionsCostOptimal(data, config, nullptr);

        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        cpu_partitions = cpu_result.size();
    }

    double gpu_avg = gpu_total_ms / iterations;
    double cpu_avg = cpu_total_ms / iterations;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "GPU Merge: " << gpu_avg << " ms (avg), " << gpu_partitions << " partitions" << std::endl;
    std::cout << "CPU Merge: " << cpu_avg << " ms (avg), " << cpu_partitions << " partitions" << std::endl;
    std::cout << "Speedup: " << (cpu_avg / gpu_avg) << "x" << std::endl;

    // Verify correctness
    bool match = (gpu_partitions == cpu_partitions);
    if (match) {
        for (size_t i = 0; i < gpu_result.size(); i++) {
            if (gpu_result[i].start_idx != cpu_result[i].start_idx ||
                gpu_result[i].end_idx != cpu_result[i].end_idx) {
                match = false;
                break;
            }
        }
    }
    std::cout << "Correctness: " << (match ? "PASSED" : "FAILED") << std::endl;

    // Print partition statistics
    if (gpu_partitions > 0) {
        int total_bits = 0;
        for (const auto& p : gpu_result) {
            total_bits += p.delta_bits;
        }
        double avg_bits = static_cast<double>(total_bits) / gpu_partitions;
        double avg_size = static_cast<double>(data.size()) / gpu_partitions;
        std::cout << "Avg partition size: " << avg_size << ", Avg delta bits: " << avg_bits << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "=== GPU Merge Test on SOSD Datasets ===" << std::endl;

    // Test on normal_200M dataset
    std::string normal_file = "data/sosd/2-normal_200M_uint64.bin";

    // Test different sizes
    std::vector<size_t> test_sizes = {1000000, 10000000, 50000000, 100000000, 200000000};

    for (size_t size : test_sizes) {
        auto data = loadBinaryFile<uint64_t>(normal_file, size);
        if (data.empty()) {
            std::cerr << "Failed to load data" << std::endl;
            continue;
        }

        std::string name = "normal_" + std::to_string(size / 1000000) + "M";
        testDataset(name, data);
    }

    return 0;
}
