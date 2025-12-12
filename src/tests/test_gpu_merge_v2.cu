/**
 * Test and benchmark for GPU Merge V2 optimization
 */

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cstring>
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

// Generate test data with different patterns
template<typename T>
std::vector<T> generateLinearData(size_t size, T start = 1000000, T slope = 1) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = start + slope * i;
    }
    return data;
}

template<typename T>
std::vector<T> generateNoisyLinearData(size_t size, T start = 1000000, T slope = 1, T noise_range = 100) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> noise(-noise_range, noise_range);
    for (size_t i = 0; i < size; i++) {
        data[i] = start + slope * i + noise(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateRandomData(size_t size, T min_val = 0, T max_val = 1000000000) {
    std::vector<T> data(size);
    std::mt19937 gen(42);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < size; i++) {
        data[i] = dist(gen);
    }
    return data;
}

template<typename T>
std::vector<T> generateSegmentedData(size_t size, int num_segments = 10) {
    std::vector<T> data(size);
    std::mt19937 gen(42);

    size_t segment_size = size / num_segments;
    for (int seg = 0; seg < num_segments; seg++) {
        T base = gen() % 1000000000;
        T slope = (gen() % 100) - 50;
        T noise_range = gen() % 1000;

        std::uniform_int_distribution<T> noise(-noise_range, noise_range);

        size_t start = seg * segment_size;
        size_t end = (seg == num_segments - 1) ? size : (seg + 1) * segment_size;

        for (size_t i = start; i < end; i++) {
            data[i] = base + slope * (i - start) + noise(gen);
        }
    }
    return data;
}

void runBenchmark(const char* name, size_t data_size, int num_runs = 5) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark: " << name << std::endl;
    std::cout << "Data size: " << data_size / 1000000.0 << "M elements" << std::endl;
    std::cout << "========================================" << std::endl;

    // Generate test data
    auto linear_data = generateNoisyLinearData<uint64_t>(data_size, 1000000ULL, 1ULL, 100ULL);

    CostOptimalConfig config;
    config.analysis_block_size = 1024;
    config.target_partition_size = 8192;
    config.min_partition_size = 256;
    config.max_partition_size = 65536;
    config.breakpoint_threshold = 4;
    config.merge_benefit_threshold = 0.01f;
    config.enable_merging = true;
    config.max_merge_rounds = 10;

    std::cout << "\nConfig:" << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  max_merge_rounds: " << config.max_merge_rounds << std::endl;
    std::cout << "  merge_benefit_threshold: " << config.merge_benefit_threshold << std::endl;

    // Warmup
    std::cout << "\nWarmup..." << std::endl;
    {
        GPUCostOptimalPartitioner<uint64_t> v1(linear_data, config);
        v1.partition();
        GPUCostOptimalPartitionerV2<uint64_t> v2(linear_data, config);
        v2.partition();
        cudaDeviceSynchronize();
    }

    // Benchmark CPU version
    std::cout << "\nBenchmarking CPU version..." << std::endl;
    double cpu_total_ms = 0;
    size_t cpu_partitions = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();

        auto result = createPartitionsCostOptimal(linear_data, config, nullptr);

        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();

        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        cpu_partitions = result.size();
    }

    // Benchmark V1 (with init/partition breakdown)
    std::cout << "Benchmarking GPU V1..." << std::endl;
    double v1_init_ms = 0, v1_part_ms = 0;
    size_t v1_partitions = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitioner<uint64_t> v1(linear_data, config);

        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        auto result = v1.partition();

        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        v1_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        v1_part_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        v1_partitions = result.size();
    }

    // Benchmark V2 (with init/partition breakdown)
    std::cout << "Benchmarking GPU V2 (optimized)..." << std::endl;
    double v2_init_ms = 0, v2_part_ms = 0;
    size_t v2_partitions = 0;
    bool cooperative_used = false;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();

        GPUCostOptimalPartitionerV2<uint64_t> v2(linear_data, config);
        cooperative_used = v2.isCooperativeLaunchSupported();

        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();

        auto result = v2.partition();

        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();

        v2_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        v2_part_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        v2_partitions = result.size();
    }

    // Results
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << "CPU (original):     " << cpu_total_ms / num_runs << " ms, "
              << cpu_partitions << " partitions" << std::endl;
    std::cout << "GPU V1: init=" << v1_init_ms / num_runs << " ms, partition=" << v1_part_ms / num_runs
              << " ms, total=" << (v1_init_ms + v1_part_ms) / num_runs << " ms, "
              << v1_partitions << " partitions" << std::endl;
    std::cout << "GPU V2: init=" << v2_init_ms / num_runs << " ms, partition=" << v2_part_ms / num_runs
              << " ms, total=" << (v2_init_ms + v2_part_ms) / num_runs << " ms, "
              << v2_partitions << " partitions" << std::endl;
    std::cout << "\nCooperative groups: " << (cooperative_used ? "ENABLED" : "DISABLED") << std::endl;

    std::cout << "\n--- Speedups (partition only) ---" << std::endl;
    std::cout << "V2 vs V1: " << v1_part_ms / v2_part_ms << "x" << std::endl;

    std::cout << "\n--- Speedups (total) ---" << std::endl;
    std::cout << "V2 vs CPU: " << cpu_total_ms / (v2_init_ms + v2_part_ms) << "x" << std::endl;
    std::cout << "V2 vs V1:  " << (v1_init_ms + v1_part_ms) / (v2_init_ms + v2_part_ms) << "x" << std::endl;
}

void runValidation() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Validation Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    CostOptimalConfig config;
    config.analysis_block_size = 1024;
    config.target_partition_size = 8192;
    config.min_partition_size = 256;
    config.max_partition_size = 65536;
    config.breakpoint_threshold = 4;
    config.merge_benefit_threshold = 0.01f;
    config.enable_merging = true;
    config.max_merge_rounds = 10;

    // Test 1: Small linear data
    {
        std::cout << "\nTest 1: Small linear data (100K)..." << std::endl;
        auto data = generateLinearData<uint64_t>(100000);
        bool pass = validateGPUMergeV2(data, config, true);
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    }

    // Test 2: Noisy linear data
    {
        std::cout << "\nTest 2: Noisy linear data (500K)..." << std::endl;
        auto data = generateNoisyLinearData<uint64_t>(500000, 1000000ULL, 1ULL, 500ULL);
        bool pass = validateGPUMergeV2(data, config, true);
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    }

    // Test 3: Segmented data
    {
        std::cout << "\nTest 3: Segmented data (1M)..." << std::endl;
        auto data = generateSegmentedData<uint64_t>(1000000, 20);
        bool pass = validateGPUMergeV2(data, config, true);
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    }

    // Test 4: Random data
    {
        std::cout << "\nTest 4: Random data (500K)..." << std::endl;
        auto data = generateRandomData<uint64_t>(500000);
        bool pass = validateGPUMergeV2(data, config, true);
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
    }
}

void printGPUInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Cooperative launch: " << (prop.cooperativeLaunch ? "supported" : "not supported") << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024*1024.0) << " GB" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "GPU Merge V2 Optimization Test\n" << std::endl;

    printGPUInfo();

    bool run_validation = true;
    bool run_benchmark = true;
    size_t benchmark_size = 10000000;  // 10M elements
    int num_runs = 5;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--validate-only") == 0) {
            run_benchmark = false;
        } else if (strcmp(argv[i], "--benchmark-only") == 0) {
            run_validation = false;
        } else if (strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
            benchmark_size = std::stoul(argv[++i]);
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        }
    }

    if (run_validation) {
        runValidation();
    }

    if (run_benchmark) {
        runBenchmark("Linear data", benchmark_size, num_runs);
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
