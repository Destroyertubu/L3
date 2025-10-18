/**
 * Accurate kernel-only benchmark for L3 compression
 * This measures ONLY the GPU kernel execution time, excluding data loading and transfer
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cuda_runtime.h>
#include "l3_codec.hpp"

// Forward declarations for optimized functions
template<typename T>
CompressedDataL3<T>* compressDataOptimized(
    const std::vector<T>& h_data,
    int partition_size,
    CompressionStats* stats);

template<typename T>
void freeCompressedDataOptimized(CompressedDataL3<T>* compressed);

// Forward declarations for original functions
template<typename T>
CompressedDataL3<T>* compressData(
    const std::vector<T>& h_data,
    int partition_size,
    CompressionStats* stats);

template<typename T>
void freeCompressedData(CompressedDataL3<T>* compressed);

// Dataset information
struct DatasetInfo {
    std::string name;
    std::string path;
    bool is_uint64;
    int partition_size;
};

// Load binary data file
template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return data;
}

// Benchmark function that measures ONLY kernel time
template<typename T>
void benchmarkDataset(const DatasetInfo& info, std::ofstream& csv_file, bool use_optimized) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << info.name << std::endl;
    std::cout << "Mode: " << (use_optimized ? "OPTIMIZED" : "ORIGINAL") << std::endl;
    std::cout << "========================================" << std::endl;

    // Load data
    std::vector<T> data;
    try {
        data = loadBinaryFile<T>(info.path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;
    std::cout << "Data size: " << (data.size() * sizeof(T) / (1024.0 * 1024.0))
              << " MB" << std::endl;

    // Warm-up GPU (important for accurate measurements)
    std::cout << "Warming up GPU..." << std::endl;
    for (int i = 0; i < 3; i++) {
        CompressionStats warmup_stats;
        if (use_optimized) {
            auto* warmup_compressed = compressDataOptimized(data, info.partition_size, &warmup_stats);
            freeCompressedDataOptimized(warmup_compressed);
        } else {
            auto* warmup_compressed = compressData(data, info.partition_size, &warmup_stats);
            freeCompressedData(warmup_compressed);
        }
    }

    // Run multiple trials for stable measurement
    const int num_trials = 10;  // More trials for accuracy
    double total_compress_time = 0;
    double compression_ratio = 0;
    double avg_delta_bits = 0;

    std::cout << "\nRunning " << num_trials << " compression trials..." << std::endl;

    std::vector<double> times;
    for (int trial = 0; trial < num_trials; trial++) {
        CompressionStats comp_stats;

        if (use_optimized) {
            auto* compressed = compressDataOptimized(data, info.partition_size, &comp_stats);
            freeCompressedDataOptimized(compressed);
        } else {
            auto* compressed = compressData(data, info.partition_size, &comp_stats);
            freeCompressedData(compressed);
        }

        times.push_back(comp_stats.compression_time_ms);
        total_compress_time += comp_stats.compression_time_ms;
        compression_ratio = comp_stats.compression_ratio;
        avg_delta_bits = comp_stats.avg_delta_bits;

        std::cout << "Trial " << (trial + 1) << ": "
                  << comp_stats.compression_time_ms << " ms" << std::endl;
    }

    // Calculate statistics
    double avg_compress_time = total_compress_time / num_trials;

    // Calculate standard deviation
    double variance = 0;
    for (double t : times) {
        variance += (t - avg_compress_time) * (t - avg_compress_time);
    }
    double std_dev = std::sqrt(variance / num_trials);

    double compress_throughput = (data.size() * sizeof(T) / 1e9) / (avg_compress_time / 1000.0);

    // Print results
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Compression Ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Avg Delta Bits: " << avg_delta_bits << std::endl;
    std::cout << "Avg Compression Time: " << avg_compress_time << " ms (±" << std_dev << ")" << std::endl;
    std::cout << "Compression Throughput: " << compress_throughput << " GB/s" << std::endl;

    // Write to CSV
    csv_file << info.name << ","
             << (use_optimized ? "Optimized" : "Original") << ","
             << data.size() << ","
             << (data.size() * sizeof(T) / (1024.0 * 1024.0)) << ","
             << compression_ratio << ","
             << avg_delta_bits << ","
             << avg_compress_time << ","
             << std_dev << ","
             << compress_throughput << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "  L3 Kernel-Only Performance Benchmark        " << std::endl;
    std::cout << "==================================================" << std::endl;

    // Configure data directory (can be set via environment variable)
    const char* data_dir_env = std::getenv("L3_DATA_DIR");
    std::string data_dir = data_dir_env ? data_dir_env : "../../test/data";

    // Define datasets (all binary files now)
    std::vector<DatasetInfo> datasets = {
        {"movieid", data_dir + "/movieid_uint32.bin", false, 4096},
        {"linear_200M", data_dir + "/linear_200M_uint32_binary.bin", false, 4096},
        {"books_200M", data_dir + "/books_200M_uint32.bin", false, 4096},
        {"normal_200M", data_dir + "/normal_200M_uint32_binary.bin", false, 4096},
        {"fb_200M", data_dir + "/fb_200M_uint64.bin", true, 4096},
        {"wiki_200M", data_dir + "/wiki_200M_uint64.bin", true, 4096}
    };

    // Open CSV file for results (relative to current directory or env variable)
    const char* output_dir_env = std::getenv("L3_OUTPUT_DIR");
    std::string output_dir = output_dir_env ? output_dir_env : ".";
    std::string csv_filename = output_dir + "/kernel_benchmark_results.csv";
    std::ofstream csv_file(csv_filename);
    csv_file << "Dataset,Version,Elements,Size_MB,Compression_Ratio,Avg_Delta_Bits,"
             << "Compress_Time_ms,Std_Dev_ms,Compress_Throughput_GBps" << std::endl;

    // Test both original and optimized versions
    std::cout << "\n=== Testing ORIGINAL Implementation ===" << std::endl;
    for (const auto& dataset : datasets) {
        if (dataset.is_uint64) {
            benchmarkDataset<uint64_t>(dataset, csv_file, false);
        } else {
            benchmarkDataset<uint32_t>(dataset, csv_file, false);
        }
    }

    std::cout << "\n=== Testing OPTIMIZED Implementation ===" << std::endl;
    for (const auto& dataset : datasets) {
        if (dataset.is_uint64) {
            benchmarkDataset<uint64_t>(dataset, csv_file, true);
        } else {
            benchmarkDataset<uint32_t>(dataset, csv_file, true);
        }
    }

    csv_file.close();

    // Print summary
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Benchmark completed!" << std::endl;
    std::cout << "Results saved to: " << csv_filename << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}