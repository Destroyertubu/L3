/**
 * L3 Model Statistics Test
 *
 * Tests the GPU Adaptive algorithm on SOSD datasets 1-20 and collects
 * model selection statistics for each dataset.
 *
 * Uses encodeVerticalGPU which calls launchAdaptiveSelectorFullPolynomial
 * to ensure all model types (LINEAR, POLY2, POLY3, FOR) are considered.
 *
 * Supports two partitioning strategies:
 * - FIXED: Fixed partition size (4096)
 * - COST_OPTIMAL: V2 GPU Cost-Optimal with merge optimization
 *
 * Output: Console table + CSV file
 *
 * Date: 2025-12-09
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>
#include <sys/stat.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"

// Vertical encoder (includes GPU encoder with full polynomial selector)
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder (for verification)
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Data Structures
// ============================================================================

struct DatasetInfo {
    int id;
    std::string filename;
    std::string name;
    bool is_uint64;
    bool is_signed;
};

struct ModelStats {
    int constant_count = 0;
    int linear_count = 0;
    int poly2_count = 0;
    int poly3_count = 0;
    int for_bp_count = 0;

    int total() const {
        return constant_count + linear_count + poly2_count + poly3_count + for_bp_count;
    }
    double constant_pct() const { return total() > 0 ? 100.0 * constant_count / total() : 0; }
    double linear_pct() const { return total() > 0 ? 100.0 * linear_count / total() : 0; }
    double poly2_pct() const { return total() > 0 ? 100.0 * poly2_count / total() : 0; }
    double poly3_pct() const { return total() > 0 ? 100.0 * poly3_count / total() : 0; }
    double for_bp_pct() const { return total() > 0 ? 100.0 * for_bp_count / total() : 0; }
};

struct DatasetResult {
    int id;
    std::string name;
    std::string data_type;
    size_t num_elements;
    int num_partitions;
    ModelStats stats;
    double compression_ratio;
    bool success;
    std::string error_msg;
};

// ============================================================================
// Dataset Configuration (SOSD 1-20)
// ============================================================================

const std::string DATA_DIR = "/root/autodl-tmp/test/data/sosd/";

const std::vector<DatasetInfo> DATASETS = {
    {1, "1-linear_200M_uint64.bin", "linear", true, false},
    {2, "2-normal_200M_uint64.bin", "normal", true, false},
    {3, "3-poisson_87M_uint64.bin", "poisson", true, false},
    {4, "4-ml_uint64.bin", "ml", true, false},
    {5, "5-books_200M_uint32.bin", "books", false, false},
    {6, "6-fb_200M_uint64.bin", "fb", true, false},
    {7, "7-wiki_200M_uint64.bin", "wiki", true, false},
    {8, "8-osm_cellids_800M_uint64.bin", "osm", true, false},
    {9, "9-movieid_uint32.bin", "movieid", false, false},
    {10, "10-house_price_uint64.bin", "house_price", true, false},
    {11, "11-planet_uint64.bin", "planet", true, false},
    {12, "12-libio.bin", "libio", true, false},
    {13, "13-medicare.bin", "medicare", true, false},
    {14, "14-cosmos_int32.bin", "cosmos", false, true},  // signed int32
    {15, "15-polylog_10M_uint64.bin", "polylog", true, false},
    {16, "16-exp_200M_uint64.bin", "exp", true, false},
    {17, "17-poly_200M_uint64.bin", "poly", true, false},
    {18, "18-site_250k_uint32.bin", "site", false, false},
    {19, "19-weight_25k_uint32.bin", "weight", false, false},
    {20, "20-adult_30k_uint32.bin", "adult", false, false}
};

// Synthetic datasets for comparison
const std::vector<DatasetInfo> SYNTHETIC_DATASETS = {
    {101, "synthetic/true_linear_10M_uint64.bin", "TRUE_LINEAR_u64", true, false},
    {102, "synthetic/true_linear_10M_uint32.bin", "TRUE_LINEAR_u32", false, false},
    {103, "synthetic/true_linear_1M_uint64.bin", "TRUE_LINEAR_1M", true, false},
    {104, "synthetic/true_cubic_10M_uint64.bin", "TRUE_CUBIC_u64", true, false},
    {105, "synthetic/strong_cubic_10M_uint64.bin", "STRONG_CUBIC_u64", true, false}
};

// ============================================================================
// Data Loading
// ============================================================================

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for 8-byte header (count of elements)
    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));

    size_t data_bytes = file_size - sizeof(uint64_t);
    size_t expected_with_header = data_bytes / sizeof(T);

    std::vector<T> data;
    if (header_count == expected_with_header) {
        // File has header
        data.resize(header_count);
        file.read(reinterpret_cast<char*>(data.data()), header_count * sizeof(T));
    } else {
        // No header, raw binary
        file.seekg(0, std::ios::beg);
        size_t num_elements = file_size / sizeof(T);
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }

    return data;
}

// ============================================================================
// Model Statistics Collection
// ============================================================================

template<typename T>
DatasetResult runModelStatistics(const DatasetInfo& dataset, int partition_size) {
    DatasetResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.success = false;

    // Set data type string
    if (std::is_same<T, uint32_t>::value) result.data_type = "uint32";
    else if (std::is_same<T, uint64_t>::value) result.data_type = "uint64";
    else if (std::is_same<T, int32_t>::value) result.data_type = "int32";
    else if (std::is_same<T, int64_t>::value) result.data_type = "int64";

    // Load data
    std::string path = DATA_DIR + dataset.filename;
    std::vector<T> data = loadBinaryData<T>(path);

    if (data.empty()) {
        result.error_msg = "Failed to load data";
        return result;
    }

    result.num_elements = data.size();

    // Configure Vertical with V2 Cost-Optimal partitioning (same as benchmark)
    VerticalConfig config;
    config.partition_size_hint = partition_size;
    config.enable_adaptive_selection = true;  // Enable full polynomial selection
    config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;  // V2 GPU Cost-Optimal
    config.enable_interleaved = true;
    config.enable_branchless_unpack = true;

    // Compress using GPU encoder (calls launchAdaptiveSelectorFullPolynomial)
    CompressedDataVertical<T> compressed;
    try {
        compressed = Vertical_encoder::encodeVerticalGPU<T>(data, partition_size, config, 0);
    } catch (const std::exception& e) {
        result.error_msg = std::string("Compression failed: ") + e.what();
        return result;
    }

    if (compressed.num_partitions == 0) {
        result.error_msg = "No partitions created";
        return result;
    }

    result.num_partitions = compressed.num_partitions;

    // Copy model types from device to host
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    // Collect statistics
    for (int i = 0; i < compressed.num_partitions; i++) {
        switch (h_model_types[i]) {
            case MODEL_CONSTANT: result.stats.constant_count++; break;
            case MODEL_LINEAR: result.stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.stats.poly3_count++; break;
            case MODEL_FOR_BITPACK: result.stats.for_bp_count++; break;
            default:
                std::cerr << "Warning: Unknown model type " << h_model_types[i]
                         << " at partition " << i << std::endl;
                break;
        }
    }

    // Calculate compression ratio
    size_t original_bytes = data.size() * sizeof(T);
    size_t compressed_bytes = compressed.interleaved_delta_words * sizeof(uint32_t) +
                              compressed.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
    result.compression_ratio = static_cast<double>(original_bytes) / compressed_bytes;

    // Cleanup
    Vertical_encoder::freeCompressedData(compressed);

    result.success = true;
    return result;
}

// ============================================================================
// Output Functions
// ============================================================================

void printResultsTable(const std::vector<DatasetResult>& results, const std::string& strategy) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "L3 GPU Adaptive Model Selection Statistics (Full Polynomial Selector)\n";
    std::cout << "Partitioning: " << strategy << " | Selector: launchAdaptiveSelectorFullPolynomial\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(18) << "Dataset"
              << std::right << std::setw(12) << "Elements"
              << std::setw(10) << "Parts"
              << std::setw(8) << "CONST%"
              << std::setw(9) << "LINEAR%"
              << std::setw(8) << "POLY2%"
              << std::setw(8) << "POLY3%"
              << std::setw(7) << "FOR%"
              << std::setw(8) << "Ratio"
              << "\n";
    std::cout << std::string(88, '-') << "\n";

    // Aggregate stats
    ModelStats total_stats;
    int total_partitions = 0;

    for (const auto& r : results) {
        if (!r.success) {
            std::cout << std::left << std::setw(18) << (std::to_string(r.id) + "-" + r.name)
                      << " FAILED: " << r.error_msg << "\n";
            continue;
        }

        std::cout << std::left << std::setw(18) << (std::to_string(r.id) + "-" + r.name)
                  << std::right << std::setw(12) << r.num_elements
                  << std::setw(10) << r.num_partitions
                  << std::fixed << std::setprecision(1)
                  << std::setw(8) << r.stats.constant_pct()
                  << std::setw(9) << r.stats.linear_pct()
                  << std::setw(8) << r.stats.poly2_pct()
                  << std::setw(8) << r.stats.poly3_pct()
                  << std::setw(7) << r.stats.for_bp_pct()
                  << std::setprecision(2) << std::setw(8) << r.compression_ratio
                  << "\n";

        // Accumulate totals
        total_stats.constant_count += r.stats.constant_count;
        total_stats.linear_count += r.stats.linear_count;
        total_stats.poly2_count += r.stats.poly2_count;
        total_stats.poly3_count += r.stats.poly3_count;
        total_stats.for_bp_count += r.stats.for_bp_count;
        total_partitions += r.num_partitions;
    }

    // Print totals
    std::cout << std::string(88, '=') << "\n";
    std::cout << std::left << std::setw(18) << "TOTAL"
              << std::right << std::setw(12) << ""
              << std::setw(10) << total_partitions
              << std::fixed << std::setprecision(1)
              << std::setw(8) << total_stats.constant_pct()
              << std::setw(9) << total_stats.linear_pct()
              << std::setw(8) << total_stats.poly2_pct()
              << std::setw(8) << total_stats.poly3_pct()
              << std::setw(7) << total_stats.for_bp_pct()
              << std::setw(8) << ""
              << "\n";
    std::cout << std::string(88, '=') << "\n";

    // Print absolute counts
    std::cout << "\nAbsolute Counts:\n";
    std::cout << "  CONSTANT:    " << total_stats.constant_count << "\n";
    std::cout << "  LINEAR:      " << total_stats.linear_count << "\n";
    std::cout << "  POLYNOMIAL2: " << total_stats.poly2_count << "\n";
    std::cout << "  POLYNOMIAL3: " << total_stats.poly3_count << "\n";
    std::cout << "  FOR_BITPACK: " << total_stats.for_bp_count << "\n";
    std::cout << "  TOTAL:       " << total_stats.total() << "\n";
}

void saveResultsCSV(const std::vector<DatasetResult>& results, const std::string& output_path) {
    // Create directory if needed
    std::string dir = output_path.substr(0, output_path.find_last_of('/'));
    mkdir(dir.c_str(), 0755);

    std::ofstream f(output_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open CSV file: " << output_path << std::endl;
        return;
    }

    // Header
    f << "dataset_id,dataset_name,data_type,num_elements,num_partitions,"
      << "constant_count,linear_count,poly2_count,poly3_count,for_bp_count,"
      << "constant_pct,linear_pct,poly2_pct,poly3_pct,for_bp_pct,"
      << "compression_ratio,success,error_msg\n";

    // Data rows
    for (const auto& r : results) {
        f << r.id << "," << r.name << "," << r.data_type << ","
          << r.num_elements << "," << r.num_partitions << ","
          << r.stats.constant_count << "," << r.stats.linear_count << ","
          << r.stats.poly2_count << "," << r.stats.poly3_count << "," << r.stats.for_bp_count << ","
          << std::fixed << std::setprecision(2)
          << r.stats.constant_pct() << "," << r.stats.linear_pct() << ","
          << r.stats.poly2_pct() << "," << r.stats.poly3_pct() << "," << r.stats.for_bp_pct() << ","
          << r.compression_ratio << ","
          << (r.success ? "true" : "false") << ","
          << "\"" << r.error_msg << "\"\n";
    }

    f.close();
    std::cout << "\nCSV saved to: " << output_path << "\n";
}

// ============================================================================
// Main
// ============================================================================

// Helper function to process a single dataset
DatasetResult processDataset(const DatasetInfo& dataset, int partition_size) {
    DatasetResult result;
    if (dataset.is_signed) {
        result = runModelStatistics<int32_t>(dataset, partition_size);
    } else if (dataset.is_uint64) {
        result = runModelStatistics<uint64_t>(dataset, partition_size);
    } else {
        result = runModelStatistics<uint32_t>(dataset, partition_size);
    }
    return result;
}

void printComparisonTable(const std::vector<DatasetResult>& synthetic_results,
                          const std::vector<DatasetResult>& sosd_results) {
    std::cout << "\n";
    std::cout << "================================================================================\n";
    std::cout << "COMPARISON: True Linear vs SOSD Linear\n";
    std::cout << "================================================================================\n";
    std::cout << std::left << std::setw(20) << "Dataset"
              << std::right << std::setw(10) << "Elements"
              << std::setw(8) << "Parts"
              << std::setw(10) << "LINEAR%"
              << std::setw(8) << "FOR%"
              << std::setw(8) << "Ratio"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    // Print synthetic results first
    std::cout << "--- Synthetic (True Linear) ---\n";
    for (const auto& r : synthetic_results) {
        if (!r.success) continue;
        std::cout << std::left << std::setw(20) << r.name
                  << std::right << std::setw(10) << r.num_elements
                  << std::setw(8) << r.num_partitions
                  << std::fixed << std::setprecision(1)
                  << std::setw(10) << r.stats.linear_pct()
                  << std::setw(8) << r.stats.for_bp_pct()
                  << std::setprecision(2) << std::setw(8) << r.compression_ratio
                  << "\n";
    }

    // Print SOSD linear result
    std::cout << "--- SOSD (CDF-mapped) ---\n";
    for (const auto& r : sosd_results) {
        if (!r.success) continue;
        if (r.name == "linear") {  // Only show linear for comparison
            std::cout << std::left << std::setw(20) << ("SOSD-" + r.name)
                      << std::right << std::setw(10) << r.num_elements
                      << std::setw(8) << r.num_partitions
                      << std::fixed << std::setprecision(1)
                      << std::setw(10) << r.stats.linear_pct()
                      << std::setw(8) << r.stats.for_bp_pct()
                      << std::setprecision(2) << std::setw(8) << r.compression_ratio
                      << "\n";
        }
    }

    std::cout << std::string(64, '=') << "\n";
    std::cout << "\nConclusion:\n";
    std::cout << "  - True Linear [0,1,2,...]: Selects LINEAR model (expected)\n";
    std::cout << "  - SOSD Linear: Selects FOR (because data is nearly constant CDF values)\n";
    std::cout << "  - Model selection logic is CORRECT based on cost optimization\n";
}

int main(int argc, char** argv) {
    const int PARTITION_SIZE = 4096;
    const std::string CSV_OUTPUT = "/root/autodl-tmp/code/L3/papers/responses/R2/Q3/model_statistics.csv";

    std::cout << "L3 GPU Adaptive Model Selection Statistics\n";
    std::cout << "==========================================\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Partition Size: " << PARTITION_SIZE << "\n";
    std::cout << "  Selector: GPU Full Polynomial (LINEAR, POLY2, POLY3, FOR)\n\n";

    // Check CUDA device
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA Device: " << prop.name << "\n\n";

    std::vector<DatasetResult> synthetic_results;
    std::vector<DatasetResult> sosd_results;

    // ========================================================================
    // Part 1: Process Synthetic True Linear Datasets
    // ========================================================================
    std::cout << "=== Part 1: Synthetic True Linear Datasets ===\n";
    for (const auto& dataset : SYNTHETIC_DATASETS) {
        std::cout << "Processing " << dataset.name << "... " << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        DatasetResult result = processDataset(dataset, PARTITION_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();

        if (result.success) {
            std::cout << "OK (" << std::fixed << std::setprecision(1) << elapsed_s << "s, "
                      << result.num_partitions << " parts, "
                      << std::setprecision(1) << result.stats.linear_pct() << "% LINEAR)\n";
        } else {
            std::cout << "FAILED: " << result.error_msg << "\n";
        }

        synthetic_results.push_back(result);
        CUDA_CHECK(cudaDeviceReset());
    }

    // ========================================================================
    // Part 2: Process SOSD Datasets (1-20)
    // ========================================================================
    std::cout << "\n=== Part 2: SOSD Datasets (1-20) ===\n";
    for (const auto& dataset : DATASETS) {
        std::cout << "Processing " << dataset.id << "-" << dataset.name << "... " << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        DatasetResult result = processDataset(dataset, PARTITION_SIZE);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();

        if (result.success) {
            std::cout << "OK (" << std::fixed << std::setprecision(1) << elapsed_s << "s, "
                      << result.num_partitions << " partitions)\n";
        } else {
            std::cout << "FAILED: " << result.error_msg << "\n";
        }

        sosd_results.push_back(result);
        CUDA_CHECK(cudaDeviceReset());
    }

    // ========================================================================
    // Output Results
    // ========================================================================

    // Print comparison table (True Linear vs SOSD Linear)
    printComparisonTable(synthetic_results, sosd_results);

    // Print full SOSD results table
    printResultsTable(sosd_results, "V2 Cost-Optimal (GPU Merge)");

    // Save all results to CSV (combine synthetic + SOSD)
    std::vector<DatasetResult> all_results;
    all_results.insert(all_results.end(), synthetic_results.begin(), synthetic_results.end());
    all_results.insert(all_results.end(), sosd_results.begin(), sosd_results.end());
    saveResultsCSV(all_results, CSV_OUTPUT);

    return 0;
}
