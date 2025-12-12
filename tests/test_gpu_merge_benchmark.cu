/**
 * GPU Merge V1 vs V2 Performance Benchmark
 * Tests using real SOSD datasets
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <iomanip>
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

template<typename T>
std::vector<T> loadBinaryData(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return {};
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check if file has header (SOSD format has 8-byte header)
    // If file size is not divisible by sizeof(T), assume 8-byte header
    size_t header_size = 0;
    if (file_size % sizeof(T) != 0) {
        header_size = 8;  // SOSD header
    }

    size_t data_size = (file_size - header_size) / sizeof(T);
    std::vector<T> data(data_size);

    // Skip header if present
    if (header_size > 0) {
        file.seekg(header_size, std::ios::beg);
    }

    file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(T));
    file.close();

    std::cout << "Loaded " << data_size << " elements ("
              << data_size * sizeof(T) / (1024.0 * 1024.0) << " MB)" << std::endl;
    return data;
}

void printGPUInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    std::cout << "\n=== GPU Information ==="  << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SMs: " << prop.multiProcessorCount << std::endl;
    std::cout << "Global Memory: " << std::fixed << std::setprecision(2)
              << prop.totalGlobalMem / (1024.0*1024.0*1024.0) << " GB" << std::endl;
    std::cout << "Cooperative Launch: " << (prop.cooperativeLaunch ? "Yes" : "No") << std::endl;
}

template<typename T>
void runBenchmark(const std::vector<T>& data, const std::string& dataset_name, int num_runs = 5) {
    if (data.empty()) return;

    std::cout << "\n======================================================" << std::endl;
    std::cout << "Benchmark: " << dataset_name << std::endl;
    std::cout << "Data size: " << data.size() / 1000000.0 << " M elements" << std::endl;
    std::cout << "======================================================" << std::endl;

    // Configuration
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
    std::cout << "  analysis_block_size: " << config.analysis_block_size << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  max_merge_rounds: " << config.max_merge_rounds << std::endl;
    std::cout << "  merge_benefit_threshold: " << config.merge_benefit_threshold << std::endl;

    // Warmup
    std::cout << "\nWarmup..." << std::flush;
    {
        GPUCostOptimalPartitioner<T> v1(data, config);
        v1.partition();
        GPUCostOptimalPartitionerV2<T> v2(data, config);
        v2.partition();
        cudaDeviceSynchronize();
    }
    std::cout << " done" << std::endl;

    // Benchmark CPU version
    std::cout << "\nBenchmarking CPU version..." << std::flush;
    double cpu_total_ms = 0;
    size_t cpu_partitions = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        auto result = createPartitionsCostOptimal(data, config, nullptr);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total_ms += std::chrono::duration<double, std::milli>(end - start).count();
        cpu_partitions = result.size();
    }
    std::cout << " done" << std::endl;

    // Benchmark GPU V1
    std::cout << "Benchmarking GPU V1..." << std::flush;
    double v1_init_ms = 0, v1_part_ms = 0;
    size_t v1_partitions = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        GPUCostOptimalPartitioner<T> v1(data, config);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = v1.partition();
        cudaDeviceSynchronize();
        auto t2 = std::chrono::high_resolution_clock::now();
        v1_init_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
        v1_part_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        v1_partitions = result.size();
    }
    std::cout << " done" << std::endl;

    // Benchmark GPU V2
    std::cout << "Benchmarking GPU V2 (optimized)..." << std::flush;
    double v2_init_ms = 0, v2_part_ms = 0;
    size_t v2_partitions = 0;
    bool cooperative_used = false;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto t0 = std::chrono::high_resolution_clock::now();
        GPUCostOptimalPartitionerV2<T> v2(data, config);
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
    std::cout << " done" << std::endl;

    // Calculate averages
    double cpu_avg = cpu_total_ms / num_runs;
    double v1_init_avg = v1_init_ms / num_runs;
    double v1_part_avg = v1_part_ms / num_runs;
    double v1_total_avg = v1_init_avg + v1_part_avg;
    double v2_init_avg = v2_init_ms / num_runs;
    double v2_part_avg = v2_part_ms / num_runs;
    double v2_total_avg = v2_init_avg + v2_part_avg;

    // Print results
    std::cout << "\n--- Performance Results (" << num_runs << " runs avg) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);

    std::cout << "\n| Version | Init (ms) | Partition (ms) | Total (ms) | Partitions |" << std::endl;
    std::cout << "|---------|-----------|----------------|------------|------------|" << std::endl;
    std::cout << "| CPU     |    -      | " << std::setw(14) << cpu_avg << " | "
              << std::setw(10) << cpu_avg << " | " << std::setw(10) << cpu_partitions << " |" << std::endl;
    std::cout << "| GPU V1  | " << std::setw(9) << v1_init_avg << " | " << std::setw(14) << v1_part_avg
              << " | " << std::setw(10) << v1_total_avg << " | " << std::setw(10) << v1_partitions << " |" << std::endl;
    std::cout << "| GPU V2  | " << std::setw(9) << v2_init_avg << " | " << std::setw(14) << v2_part_avg
              << " | " << std::setw(10) << v2_total_avg << " | " << std::setw(10) << v2_partitions << " |" << std::endl;

    std::cout << "\nCooperative Groups: " << (cooperative_used ? "ENABLED" : "DISABLED") << std::endl;

    std::cout << "\n--- Speedup Analysis ---" << std::endl;
    std::cout << "GPU V1 vs CPU:          " << std::setw(6) << cpu_avg / v1_total_avg << "x" << std::endl;
    std::cout << "GPU V2 vs CPU:          " << std::setw(6) << cpu_avg / v2_total_avg << "x" << std::endl;
    std::cout << "GPU V2 vs V1 (total):   " << std::setw(6) << v1_total_avg / v2_total_avg << "x" << std::endl;
    std::cout << "GPU V2 vs V1 (partition):" << std::setw(5) << v1_part_avg / v2_part_avg << "x" << std::endl;

    // Throughput
    double data_mb = data.size() * sizeof(T) / (1024.0 * 1024.0);
    std::cout << "\n--- Throughput ---" << std::endl;
    std::cout << "CPU:    " << std::setw(8) << data_mb / (cpu_avg / 1000.0) << " MB/s" << std::endl;
    std::cout << "GPU V1: " << std::setw(8) << data_mb / (v1_total_avg / 1000.0) << " MB/s" << std::endl;
    std::cout << "GPU V2: " << std::setw(8) << data_mb / (v2_total_avg / 1000.0) << " MB/s" << std::endl;
}

template<typename T>
bool runValidation(const std::vector<T>& data, const std::string& dataset_name) {
    if (data.empty()) return false;

    std::cout << "\nValidation: " << dataset_name << std::endl;

    CostOptimalConfig config;
    config.analysis_block_size = 1024;
    config.target_partition_size = 8192;
    config.min_partition_size = 256;
    config.max_partition_size = 65536;
    config.breakpoint_threshold = 4;
    config.merge_benefit_threshold = 0.01f;
    config.enable_merging = true;
    config.max_merge_rounds = 10;

    // Run CPU
    auto cpu_result = createPartitionsCostOptimal(data, config, nullptr);

    // Run GPU V1
    GPUCostOptimalPartitioner<T> v1(data, config);
    auto v1_result = v1.partition();

    // Run GPU V2
    GPUCostOptimalPartitionerV2<T> v2(data, config);
    auto v2_result = v2.partition();

    // Compare partition counts
    std::cout << "  CPU partitions:    " << cpu_result.size() << std::endl;
    std::cout << "  GPU V1 partitions: " << v1_result.size() << std::endl;
    std::cout << "  GPU V2 partitions: " << v2_result.size() << std::endl;

    // Check if V1 and V2 match
    if (v1_result.size() != v2_result.size()) {
        std::cout << "  WARNING: V1 and V2 partition counts differ!" << std::endl;
        return false;
    }

    // Check boundaries match
    bool boundaries_match = true;
    for (size_t i = 0; i < v1_result.size() && i < v2_result.size(); i++) {
        if (v1_result[i].start_idx != v2_result[i].start_idx ||
            v1_result[i].end_idx != v2_result[i].end_idx) {
            if (boundaries_match) {
                std::cout << "  WARNING: Partition boundaries differ at partition " << i << std::endl;
                boundaries_match = false;
            }
        }
    }

    std::cout << "  V1 vs V2: " << (boundaries_match ? "MATCH" : "DIFFER") << std::endl;
    return boundaries_match;
}

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] <data_file>" << std::endl;
    std::cout << "\nOptions:" << std::endl;
    std::cout << "  --runs <N>        Number of benchmark runs (default: 5)" << std::endl;
    std::cout << "  --validate        Run validation only" << std::endl;
    std::cout << "  --benchmark       Run benchmark only (default)" << std::endl;
    std::cout << "  --all             Run both validation and benchmark" << std::endl;
    std::cout << "\nExample:" << std::endl;
    std::cout << "  " << prog << " /path/to/2-normal_200M_uint64.bin" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "===== GPU Merge V1 vs V2 Performance Benchmark =====\n" << std::endl;

    // Parse arguments
    std::string data_file;
    int num_runs = 5;
    bool run_validation = false;
    bool run_benchmark = true;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--validate") == 0) {
            run_validation = true;
            run_benchmark = false;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            run_validation = false;
            run_benchmark = true;
        } else if (strcmp(argv[i], "--all") == 0) {
            run_validation = true;
            run_benchmark = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (argv[i][0] != '-') {
            data_file = argv[i];
        }
    }

    if (data_file.empty()) {
        std::cerr << "Error: No data file specified\n" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    printGPUInfo();

    // Load data
    std::cout << "\nLoading data from: " << data_file << std::endl;
    auto data = loadBinaryData<uint64_t>(data_file);

    if (data.empty()) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    // Get dataset name from filename
    std::string dataset_name = data_file;
    size_t pos = dataset_name.find_last_of("/\\");
    if (pos != std::string::npos) {
        dataset_name = dataset_name.substr(pos + 1);
    }

    // Run tests
    if (run_validation) {
        runValidation(data, dataset_name);
    }

    if (run_benchmark) {
        runBenchmark(data, dataset_name, num_runs);
    }

    std::cout << "\nDone!" << std::endl;
    return 0;
}
