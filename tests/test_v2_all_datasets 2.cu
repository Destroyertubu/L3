/**
 * Test V2 partitioning strategy on all 20 SOSD datasets
 */

#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <dirent.h>
#include <algorithm>
#include "encoder_cost_optimal.cuh"
#include "encoder_cost_optimal_gpu_merge.cuh"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

template<typename T>
std::vector<T> loadBinaryData(const std::string& filepath, bool has_header = true) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // SOSD format has 8-byte header (element count)
    size_t header_size = 0;
    if (has_header && file_size % sizeof(T) != 0) {
        header_size = 8;
    }

    size_t data_size = (file_size - header_size) / sizeof(T);
    std::vector<T> data(data_size);

    if (header_size > 0) {
        file.seekg(header_size, std::ios::beg);
    }

    file.read(reinterpret_cast<char*>(data.data()), data_size * sizeof(T));
    file.close();

    return data;
}

struct DatasetResult {
    std::string name;
    std::string dtype;
    size_t elements;
    double size_mb;
    size_t cpu_partitions;
    size_t v1_partitions;
    size_t v2_partitions;
    double cpu_time_ms;
    double v1_time_ms;
    double v2_time_ms;
    double v2_speedup_vs_cpu;
    double v2_speedup_vs_v1;
    double v2_throughput_mbs;
    bool v1_v2_match;
};

template<typename T>
DatasetResult testDataset(const std::string& filepath, const std::string& dtype, int num_runs = 3) {
    DatasetResult result;

    // Extract name
    size_t pos = filepath.find_last_of("/\\");
    result.name = (pos != std::string::npos) ? filepath.substr(pos + 1) : filepath;
    result.dtype = dtype;

    // Load data
    auto data = loadBinaryData<T>(filepath);
    if (data.empty()) {
        result.elements = 0;
        return result;
    }

    result.elements = data.size();
    result.size_mb = data.size() * sizeof(T) / (1024.0 * 1024.0);

    // Config
    CostOptimalConfig config;
    config.analysis_block_size = 1024;
    config.target_partition_size = 8192;
    config.min_partition_size = 256;
    config.max_partition_size = 65536;
    config.breakpoint_threshold = 4;
    config.merge_benefit_threshold = 0.01f;
    config.enable_merging = true;
    config.max_merge_rounds = 10;

    // Warmup
    {
        GPUCostOptimalPartitionerV2<T> v2(data, config);
        v2.partition();
        cudaDeviceSynchronize();
    }

    // CPU
    double cpu_total = 0;
    for (int i = 0; i < num_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto res = createPartitionsCostOptimal(data, config, nullptr);
        auto end = std::chrono::high_resolution_clock::now();
        cpu_total += std::chrono::duration<double, std::milli>(end - start).count();
        result.cpu_partitions = res.size();
    }
    result.cpu_time_ms = cpu_total / num_runs;

    // GPU V1
    double v1_total = 0;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        GPUCostOptimalPartitioner<T> v1(data, config);
        auto res = v1.partition();
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        v1_total += std::chrono::duration<double, std::milli>(end - start).count();
        result.v1_partitions = res.size();
    }
    result.v1_time_ms = v1_total / num_runs;

    // GPU V2
    double v2_total = 0;
    std::vector<PartitionInfo> v2_result;
    for (int i = 0; i < num_runs; i++) {
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        GPUCostOptimalPartitionerV2<T> v2(data, config);
        v2_result = v2.partition();
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        v2_total += std::chrono::duration<double, std::milli>(end - start).count();
        result.v2_partitions = v2_result.size();
    }
    result.v2_time_ms = v2_total / num_runs;

    // Compute metrics
    result.v2_speedup_vs_cpu = result.cpu_time_ms / result.v2_time_ms;
    result.v2_speedup_vs_v1 = result.v1_time_ms / result.v2_time_ms;
    result.v2_throughput_mbs = result.size_mb / (result.v2_time_ms / 1000.0);
    result.v1_v2_match = (result.v1_partitions == result.v2_partitions);

    return result;
}

int main(int argc, char** argv) {
    std::cout << "===== V2 Partitioning Strategy Test on All SOSD Datasets =====\n" << std::endl;

    std::string data_dir = "/root/autodl-tmp/test/data/sosd";
    int num_runs = 3;

    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        }
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << " (" << prop.multiProcessorCount << " SMs)" << std::endl;
    std::cout << "Runs per dataset: " << num_runs << "\n" << std::endl;

    // Dataset list with types
    struct DatasetInfo {
        std::string filename;
        std::string dtype;
    };

    std::vector<DatasetInfo> datasets = {
        {"1-linear_200M_uint64.bin", "uint64"},
        {"2-normal_200M_uint64.bin", "uint64"},
        {"3-poisson_87M_uint64.bin", "uint64"},
        {"4-ml_uint64.bin", "uint64"},
        {"5-books_200M_uint32.bin", "uint32"},
        {"6-fb_200M_uint64.bin", "uint64"},
        {"7-wiki_200M_uint64.bin", "uint64"},
        {"8-osm_cellids_800M_uint64.bin", "uint64"},
        {"9-movieid_uint32.bin", "uint32"},
        {"10-house_price_uint64.bin", "uint64"},
        {"11-planet_uint64.bin", "uint64"},
        {"12-libio.bin", "uint64"},
        {"13-medicare.bin", "uint64"},
        {"14-cosmos_int32.bin", "int32"},
        {"15-polylog_10M_uint64.bin", "uint64"},
        {"16-exp_200M_uint64.bin", "uint64"},
        {"17-poly_200M_uint64.bin", "uint64"},
        {"18-site_250k_uint32.bin", "uint32"},
        {"19-weight_25k_uint32.bin", "uint32"},
        {"20-adult_30k_uint32.bin", "uint32"},
    };

    std::vector<DatasetResult> results;

    // Test each dataset
    for (size_t i = 0; i < datasets.size(); i++) {
        std::string filepath = data_dir + "/" + datasets[i].filename;
        std::cout << "[" << (i+1) << "/" << datasets.size() << "] Testing: " << datasets[i].filename << std::flush;

        DatasetResult result;
        if (datasets[i].dtype == "uint64") {
            result = testDataset<uint64_t>(filepath, datasets[i].dtype, num_runs);
        } else if (datasets[i].dtype == "uint32") {
            result = testDataset<uint32_t>(filepath, datasets[i].dtype, num_runs);
        } else if (datasets[i].dtype == "int32") {
            result = testDataset<int32_t>(filepath, datasets[i].dtype, num_runs);
        }

        if (result.elements > 0) {
            results.push_back(result);
            std::cout << " - " << std::fixed << std::setprecision(1)
                      << result.elements / 1e6 << "M elems, "
                      << result.v2_partitions << " parts, "
                      << result.v2_time_ms << " ms"
                      << (result.v1_v2_match ? "" : " [MISMATCH]") << std::endl;
        } else {
            std::cout << " - FAILED" << std::endl;
        }
    }

    // Print summary table
    std::cout << "\n" << std::string(140, '=') << std::endl;
    std::cout << "SUMMARY: V2 Partitioning Strategy Results" << std::endl;
    std::cout << std::string(140, '=') << std::endl;

    std::cout << std::left << std::setw(32) << "Dataset"
              << std::right << std::setw(8) << "Type"
              << std::setw(10) << "Elems(M)"
              << std::setw(10) << "Size(MB)"
              << std::setw(12) << "CPU Parts"
              << std::setw(10) << "V2 Parts"
              << std::setw(12) << "CPU(ms)"
              << std::setw(10) << "V1(ms)"
              << std::setw(10) << "V2(ms)"
              << std::setw(10) << "V2/CPU"
              << std::setw(10) << "V2/V1"
              << std::setw(12) << "MB/s"
              << std::setw(8) << "Match" << std::endl;
    std::cout << std::string(140, '-') << std::endl;

    double total_cpu_time = 0, total_v1_time = 0, total_v2_time = 0;
    double total_size = 0;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::setw(8) << r.dtype
                  << std::fixed << std::setprecision(1)
                  << std::setw(10) << (r.elements / 1e6)
                  << std::setw(10) << r.size_mb
                  << std::setw(12) << r.cpu_partitions
                  << std::setw(10) << r.v2_partitions
                  << std::setprecision(2)
                  << std::setw(12) << r.cpu_time_ms
                  << std::setw(10) << r.v1_time_ms
                  << std::setw(10) << r.v2_time_ms
                  << std::setw(10) << r.v2_speedup_vs_cpu
                  << std::setw(10) << r.v2_speedup_vs_v1
                  << std::setprecision(0)
                  << std::setw(12) << r.v2_throughput_mbs
                  << std::setw(8) << (r.v1_v2_match ? "Yes" : "NO")
                  << std::endl;

        total_cpu_time += r.cpu_time_ms;
        total_v1_time += r.v1_time_ms;
        total_v2_time += r.v2_time_ms;
        total_size += r.size_mb;
    }

    std::cout << std::string(140, '-') << std::endl;
    std::cout << std::left << std::setw(32) << "TOTAL/AVERAGE"
              << std::right << std::setw(8) << ""
              << std::fixed << std::setprecision(1)
              << std::setw(10) << ""
              << std::setw(10) << total_size
              << std::setw(12) << ""
              << std::setw(10) << ""
              << std::setprecision(2)
              << std::setw(12) << total_cpu_time
              << std::setw(10) << total_v1_time
              << std::setw(10) << total_v2_time
              << std::setw(10) << (total_cpu_time / total_v2_time)
              << std::setw(10) << (total_v1_time / total_v2_time)
              << std::setprecision(0)
              << std::setw(12) << (total_size / (total_v2_time / 1000.0))
              << std::endl;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "KEY METRICS:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Total data processed:     " << std::fixed << std::setprecision(2) << total_size / 1024.0 << " GB" << std::endl;
    std::cout << "Total V2 time:            " << total_v2_time << " ms" << std::endl;
    std::cout << "Avg V2 speedup vs CPU:    " << (total_cpu_time / total_v2_time) << "x" << std::endl;
    std::cout << "Avg V2 speedup vs V1:     " << (total_v1_time / total_v2_time) << "x" << std::endl;
    std::cout << "Avg V2 throughput:        " << std::setprecision(0) << (total_size / (total_v2_time / 1000.0)) << " MB/s" << std::endl;

    std::cout << "\nDone!" << std::endl;
    return 0;
}
