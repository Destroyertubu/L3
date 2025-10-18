/**
 * Benchmark optimized L3 compression on 6 datasets
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstring>
#include <iomanip>
#include <sstream>
#include "l3_codec.hpp"

// Forward declarations for optimized functions
template<typename T>
CompressedDataL3<T>* compressDataOptimized(
    const std::vector<T>& h_data,
    int partition_size,
    CompressionStats* stats);

template<typename T>
void freeCompressedDataOptimized(CompressedDataL3<T>* compressed);

template<typename T>
int decompressDataOptimized(
    const CompressedDataL3<T>* compressed,
    std::vector<T>& h_output,
    DecompressionStats* stats);

// Dataset information
struct DatasetInfo {
    std::string name;
    std::string path;
    bool is_uint64;
    int partition_size;
};

// Load binary data file
template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename, size_t expected_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(T);
    if (expected_elements > 0 && num_elements != expected_elements) {
        std::cerr << "Warning: Expected " << expected_elements << " elements but found "
                  << num_elements << " in " << filename << std::endl;
    }

    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return data;
}

// Load text file (for movieid dataset)
std::vector<uint32_t> loadTextFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<uint32_t> data;
    uint32_t value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();

    return data;
}

// Run benchmark for a single dataset
template<typename T>
void benchmarkDataset(const DatasetInfo& info, std::ofstream& csv_file) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << info.name << std::endl;
    std::cout << "========================================" << std::endl;

    // Load data
    std::vector<T> data;
    try {
        if (info.name == "movieid" || info.name.find(".txt") != std::string::npos) {
            // Special handling for text files
            auto data_uint32 = loadTextFile(info.path);
            // Always convert to appropriate type
            data.resize(data_uint32.size());
            for (size_t i = 0; i < data_uint32.size(); i++) {
                data[i] = static_cast<T>(data_uint32[i]);
            }
        } else {
            // Binary files
            data = loadBinaryFile<T>(info.path);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading dataset: " << e.what() << std::endl;
        return;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;
    std::cout << "Data size: " << (data.size() * sizeof(T) / (1024.0 * 1024.0))
              << " MB" << std::endl;

    // Warm-up GPU
    std::cout << "Warming up GPU..." << std::endl;
    CompressionStats warmup_stats;
    auto* warmup_compressed = compressDataOptimized(data, info.partition_size, &warmup_stats);
    freeCompressedDataOptimized(warmup_compressed);

    // Run multiple trials for stable measurement
    const int num_trials = 3;
    double total_compress_time = 0;
    double total_decompress_time = 0;
    double compression_ratio = 0;
    double avg_delta_bits = 0;

    std::cout << "\nRunning " << num_trials << " compression trials..." << std::endl;

    for (int trial = 0; trial < num_trials; trial++) {
        // Compression
        CompressionStats comp_stats;
        auto* compressed = compressDataOptimized(data, info.partition_size, &comp_stats);

        // Decompression
        std::vector<T> decompressed;
        DecompressionStats decomp_stats;
        decompressDataOptimized(compressed, decompressed, &decomp_stats);

        // Verify correctness
        bool correct = (decompressed.size() == data.size());
        if (correct) {
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != decompressed[i]) {
                    correct = false;
                    std::cerr << "Mismatch at index " << i << std::endl;
                    break;
                }
            }
        }

        if (!correct) {
            std::cerr << "ERROR: Decompression failed!" << std::endl;
        }

        // Accumulate statistics
        total_compress_time += comp_stats.compression_time_ms;
        total_decompress_time += decomp_stats.decompression_time_ms;
        compression_ratio = comp_stats.compression_ratio;
        avg_delta_bits = comp_stats.avg_delta_bits;

        std::cout << "Trial " << (trial + 1) << ": "
                  << "Compress: " << comp_stats.compression_time_ms << " ms, "
                  << "Decompress: " << decomp_stats.decompression_time_ms << " ms"
                  << std::endl;

        freeCompressedDataOptimized(compressed);
    }

    // Calculate averages
    double avg_compress_time = total_compress_time / num_trials;
    double avg_decompress_time = total_decompress_time / num_trials;
    double compress_throughput = (data.size() * sizeof(T) / 1e9) / (avg_compress_time / 1000.0);
    double decompress_throughput = (data.size() * sizeof(T) / 1e9) / (avg_decompress_time / 1000.0);

    // Print results
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Compression Ratio: " << compression_ratio << "x" << std::endl;
    std::cout << "Avg Delta Bits: " << avg_delta_bits << std::endl;
    std::cout << "Avg Compression Time: " << avg_compress_time << " ms" << std::endl;
    std::cout << "Avg Decompression Time: " << avg_decompress_time << " ms" << std::endl;
    std::cout << "Compression Throughput: " << compress_throughput << " GB/s" << std::endl;
    std::cout << "Decompression Throughput: " << decompress_throughput << " GB/s" << std::endl;

    // Write to CSV
    csv_file << info.name << ","
             << data.size() << ","
             << (data.size() * sizeof(T) / (1024.0 * 1024.0)) << ","
             << compression_ratio << ","
             << avg_delta_bits << ","
             << avg_compress_time << ","
             << compress_throughput << ","
             << avg_decompress_time << ","
             << decompress_throughput << std::endl;
}

int main() {
    std::cout << "==================================================" << std::endl;
    std::cout << "     L3 Optimized Compression Benchmark       " << std::endl;
    std::cout << "==================================================" << std::endl;

    // Configure data directory (can be set via environment variable)
    const char* data_dir_env = std::getenv("L3_DATA_DIR");
    std::string data_dir = data_dir_env ? data_dir_env : "../../test/data";

    // Define datasets
    std::vector<DatasetInfo> datasets = {
        {"movieid", data_dir + "/movieid.txt", false, 4096},
        {"linear_200M_uint32", data_dir + "/linear_200M_uint32.txt", false, 4096},
        {"books_200M_uint32", data_dir + "/books_200M_uint32.bin", false, 4096},
        {"normal_200M_uint32", data_dir + "/normal_200M_uint32.txt", false, 4096},
        {"fb_200M_uint64", data_dir + "/fb_200M_uint64.bin", true, 4096},
        {"wiki_200M_uint64", data_dir + "/wiki_200M_uint64.bin", true, 4096}
    };

    // Open CSV file for results (relative to current directory or env variable)
    const char* output_dir_env = std::getenv("L3_OUTPUT_DIR");
    std::string output_dir = output_dir_env ? output_dir_env : ".";
    std::string csv_filename = output_dir + "/l3_optimized_results.csv";
    std::ofstream csv_file(csv_filename);
    csv_file << "Dataset,Elements,Size_MB,Compression_Ratio,Avg_Delta_Bits,"
             << "Compress_Time_ms,Compress_Throughput_GBps,"
             << "Decompress_Time_ms,Decompress_Throughput_GBps" << std::endl;

    // Run benchmarks
    for (const auto& dataset : datasets) {
        if (dataset.is_uint64) {
            benchmarkDataset<uint64_t>(dataset, csv_file);
        } else {
            benchmarkDataset<uint32_t>(dataset, csv_file);
        }
    }

    csv_file.close();
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Results saved to: " << csv_filename << std::endl;
    std::cout << "==================================================" << std::endl;

    return 0;
}