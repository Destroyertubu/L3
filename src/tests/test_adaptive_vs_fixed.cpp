/**
 * Test: Adaptive Partitioning vs Fixed Partitioning Comparison
 *
 * Compares compression ratio, compression time, and decompression time
 * between adaptive and fixed partitioning strategies.
 */

#include "L3_codec.hpp"
#include "L3_format.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

struct TestResult {
    std::string dataset_name;
    std::string data_type;
    size_t num_elements;
    size_t original_bytes;

    // Fixed partitioning results
    size_t fixed_compressed_bytes;
    double fixed_compression_ratio;
    double fixed_compress_time_ms;
    double fixed_decompress_time_ms;
    int fixed_num_partitions;

    // Adaptive partitioning results
    size_t adaptive_compressed_bytes;
    double adaptive_compression_ratio;
    double adaptive_compress_time_ms;
    double adaptive_decompress_time_ms;
    int adaptive_num_partitions;

    // Improvement
    double ratio_improvement_pct;
};

template<typename T>
bool loadBinaryFile(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << std::endl;
        return false;
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(T);
    file.seekg(0, std::ios::beg);

    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    return true;
}

template<typename T>
TestResult runComparison(const std::string& filename, const std::string& dataset_name,
                         const std::string& data_type, int partition_size = 2048) {
    TestResult result;
    result.dataset_name = dataset_name;
    result.data_type = data_type;

    // Load data
    std::vector<T> data;
    if (!loadBinaryFile(filename, data)) {
        std::cerr << "Failed to load " << filename << std::endl;
        return result;
    }

    result.num_elements = data.size();
    result.original_bytes = data.size() * sizeof(T);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << " (" << data_type << ")" << std::endl;
    std::cout << "Elements: " << result.num_elements << std::endl;
    std::cout << "Original size: " << (result.original_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    // ==================== Test Fixed Partitioning ====================
    std::cout << "\n[Fixed Partitioning]" << std::endl;
    {
        L3Config config = L3Config::fixedPartitioning(partition_size);
        CompressionStats comp_stats;
        DecompressionStats decomp_stats;

        auto start = std::chrono::high_resolution_clock::now();
        CompressedDataL3<T>* compressed = compressDataWithConfig(data, config, &comp_stats);
        auto end = std::chrono::high_resolution_clock::now();

        result.fixed_compress_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.fixed_compressed_bytes = comp_stats.compressed_bytes;
        result.fixed_compression_ratio = comp_stats.compression_ratio;
        result.fixed_num_partitions = comp_stats.num_partitions;

        // Decompress and verify
        std::vector<T> decompressed;
        start = std::chrono::high_resolution_clock::now();
        decompressData(compressed, decompressed, &decomp_stats);
        end = std::chrono::high_resolution_clock::now();

        result.fixed_decompress_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Verify correctness
        bool correct = (data.size() == decompressed.size());
        if (correct) {
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != decompressed[i]) {
                    correct = false;
                    break;
                }
            }
        }

        std::cout << "  Compressed: " << (result.fixed_compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Ratio: " << std::fixed << std::setprecision(2) << result.fixed_compression_ratio << "x" << std::endl;
        std::cout << "  Partitions: " << result.fixed_num_partitions << std::endl;
        std::cout << "  Compress time: " << std::fixed << std::setprecision(1) << result.fixed_compress_time_ms << " ms" << std::endl;
        std::cout << "  Decompress time: " << std::fixed << std::setprecision(1) << result.fixed_decompress_time_ms << " ms" << std::endl;
        std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

        freeCompressedData(compressed);
    }

    // ==================== Test Adaptive Partitioning ====================
    std::cout << "\n[Adaptive Partitioning]" << std::endl;
    {
        L3Config config;  // Default is adaptive
        config.partition_size = partition_size;
        CompressionStats comp_stats;
        DecompressionStats decomp_stats;

        auto start = std::chrono::high_resolution_clock::now();
        CompressedDataL3<T>* compressed = compressDataWithConfig(data, config, &comp_stats);
        auto end = std::chrono::high_resolution_clock::now();

        result.adaptive_compress_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        result.adaptive_compressed_bytes = comp_stats.compressed_bytes;
        result.adaptive_compression_ratio = comp_stats.compression_ratio;
        result.adaptive_num_partitions = comp_stats.num_partitions;

        // Decompress and verify
        std::vector<T> decompressed;
        start = std::chrono::high_resolution_clock::now();
        decompressData(compressed, decompressed, &decomp_stats);
        end = std::chrono::high_resolution_clock::now();

        result.adaptive_decompress_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Verify correctness
        bool correct = (data.size() == decompressed.size());
        if (correct) {
            for (size_t i = 0; i < data.size(); i++) {
                if (data[i] != decompressed[i]) {
                    correct = false;
                    break;
                }
            }
        }

        std::cout << "  Compressed: " << (result.adaptive_compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Ratio: " << std::fixed << std::setprecision(2) << result.adaptive_compression_ratio << "x" << std::endl;
        std::cout << "  Partitions: " << result.adaptive_num_partitions << std::endl;
        std::cout << "  Compress time: " << std::fixed << std::setprecision(1) << result.adaptive_compress_time_ms << " ms" << std::endl;
        std::cout << "  Decompress time: " << std::fixed << std::setprecision(1) << result.adaptive_decompress_time_ms << " ms" << std::endl;
        std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

        freeCompressedData(compressed);
    }

    // Calculate improvement
    result.ratio_improvement_pct = ((result.adaptive_compression_ratio / result.fixed_compression_ratio) - 1.0) * 100.0;

    std::cout << "\n[Comparison]" << std::endl;
    std::cout << "  Ratio improvement: " << std::showpos << std::fixed << std::setprecision(1)
              << result.ratio_improvement_pct << "%" << std::noshowpos << std::endl;

    return result;
}

void printSummaryTable(const std::vector<TestResult>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "SUMMARY: Adaptive vs Fixed Partitioning Comparison" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    std::cout << std::left << std::setw(20) << "Dataset"
              << std::right << std::setw(12) << "Elements"
              << std::setw(12) << "Fixed"
              << std::setw(12) << "Adaptive"
              << std::setw(12) << "Improve"
              << std::setw(14) << "Fixed Parts"
              << std::setw(14) << "Adapt Parts"
              << std::setw(12) << "F.Comp(ms)"
              << std::setw(12) << "A.Comp(ms)"
              << std::endl;

    std::cout << std::string(120, '-') << std::endl;

    double total_fixed_ratio = 0, total_adaptive_ratio = 0;
    int count = 0;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.dataset_name
                  << std::right << std::setw(12) << r.num_elements
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.fixed_compression_ratio << "x"
                  << std::setw(11) << std::fixed << std::setprecision(2) << r.adaptive_compression_ratio << "x"
                  << std::setw(10) << std::showpos << std::fixed << std::setprecision(1) << r.ratio_improvement_pct << "%" << std::noshowpos
                  << std::setw(14) << r.fixed_num_partitions
                  << std::setw(14) << r.adaptive_num_partitions
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.fixed_compress_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(0) << r.adaptive_compress_time_ms
                  << std::endl;

        total_fixed_ratio += r.fixed_compression_ratio;
        total_adaptive_ratio += r.adaptive_compression_ratio;
        count++;
    }

    std::cout << std::string(120, '-') << std::endl;

    if (count > 0) {
        double avg_fixed = total_fixed_ratio / count;
        double avg_adaptive = total_adaptive_ratio / count;
        double avg_improvement = ((avg_adaptive / avg_fixed) - 1.0) * 100.0;

        std::cout << std::left << std::setw(20) << "AVERAGE"
                  << std::right << std::setw(12) << "-"
                  << std::setw(11) << std::fixed << std::setprecision(2) << avg_fixed << "x"
                  << std::setw(11) << std::fixed << std::setprecision(2) << avg_adaptive << "x"
                  << std::setw(10) << std::showpos << std::fixed << std::setprecision(1) << avg_improvement << "%" << std::noshowpos
                  << std::endl;
    }

    std::cout << std::string(120, '=') << std::endl;
}

int main() {
    std::cout << "======================================================================" << std::endl;
    std::cout << "L3 Compression: Adaptive vs Fixed Partitioning Comparison" << std::endl;
    std::cout << "======================================================================" << std::endl;

    std::vector<TestResult> results;

    const std::string base_path = "/root/autodl-tmp/test/data/sosd/";

    // Test all 7 datasets
    results.push_back(runComparison<uint32_t>(base_path + "books_200M_uint32.bin", "books_200M", "uint32"));
    results.push_back(runComparison<uint64_t>(base_path + "fb_200M_uint64.bin", "fb_200M", "uint64"));
    results.push_back(runComparison<uint32_t>(base_path + "linear_200M_uint32_binary.bin", "linear_200M", "uint32"));
    results.push_back(runComparison<uint32_t>(base_path + "normal_200M_uint32_binary.bin", "normal_200M", "uint32"));
    results.push_back(runComparison<uint32_t>(base_path + "movieid_uint32.bin", "movieid", "uint32"));
    results.push_back(runComparison<uint64_t>(base_path + "wiki_200M_uint64.bin", "wiki_200M", "uint64"));
    results.push_back(runComparison<uint64_t>(base_path + "osm_cellids_800M_uint64.bin", "osm_800M", "uint64"));

    // Print summary table
    printSummaryTable(results);

    return 0;
}
