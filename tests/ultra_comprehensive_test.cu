/**
 * L3 Ultra-Comprehensive Test Suite
 *
 * Tests ALL combinations of:
 * - 20 SOSD datasets (1-20)
 * - 3 partitioning strategies: FIXED, VARIANCE_ADAPTIVE, COST_OPTIMAL
 * - 5 model types: CONSTANT, LINEAR, POLY2, POLY3, FOR_BITPACK (auto-selected)
 * - 4 decompression modes: AUTO, SEQUENTIAL, INTERLEAVED, BRANCHLESS
 * - Random access functionality
 *
 * Generates comprehensive reports on compression ratio, throughput, correctness
 *
 * Date: 2025-12-05
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// Cost-optimal encoder (includes adaptive selector)
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// Vertical encoder
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder
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
            return false; \
        } \
    } while(0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return; \
        } \
    } while(0)

// ============================================================================
// Data Structures
// ============================================================================

struct DatasetInfo {
    int id;
    std::string name;
    std::string filename;
    std::string type;  // "uint32" or "uint64" or "int32" or "int64"
};

// Partitioning strategy enum
enum class PartitionStrategy {
    FIXED_1024,
    FIXED_2048,
    FIXED_4096,
    COST_OPTIMAL_BALANCED,
    COST_OPTIMAL_HIGH_COMPRESSION,
    COST_OPTIMAL_HIGH_THROUGHPUT
};

std::string strategyToString(PartitionStrategy s) {
    switch (s) {
        case PartitionStrategy::FIXED_1024: return "FIXED_1024";
        case PartitionStrategy::FIXED_2048: return "FIXED_2048";
        case PartitionStrategy::FIXED_4096: return "FIXED_4096";
        case PartitionStrategy::COST_OPTIMAL_BALANCED: return "COST_OPTIMAL_BALANCED";
        case PartitionStrategy::COST_OPTIMAL_HIGH_COMPRESSION: return "COST_OPTIMAL_HC";
        case PartitionStrategy::COST_OPTIMAL_HIGH_THROUGHPUT: return "COST_OPTIMAL_HT";
    }
    return "UNKNOWN";
}

std::string decompModeToString(DecompressMode m) {
    switch (m) {
        case DecompressMode::AUTO: return "AUTO";
        case DecompressMode::SEQUENTIAL: return "SEQUENTIAL";
        case DecompressMode::INTERLEAVED: return "INTERLEAVED";
        case DecompressMode::BRANCHLESS: return "BRANCHLESS";
    }
    return "UNKNOWN";
}

std::string modelTypeToString(int mt) {
    switch (mt) {
        case MODEL_CONSTANT: return "CONSTANT";
        case MODEL_LINEAR: return "LINEAR";
        case MODEL_POLYNOMIAL2: return "POLY2";
        case MODEL_POLYNOMIAL3: return "POLY3";
        case MODEL_FOR_BITPACK: return "FOR_BP";
    }
    return "UNKNOWN";
}

struct ModelStats {
    int constant_count = 0;
    int linear_count = 0;
    int poly2_count = 0;
    int poly3_count = 0;
    int for_bp_count = 0;

    int total() const { return constant_count + linear_count + poly2_count + poly3_count + for_bp_count; }
    double constant_pct() const { return total() > 0 ? 100.0 * constant_count / total() : 0; }
    double linear_pct() const { return total() > 0 ? 100.0 * linear_count / total() : 0; }
    double poly2_pct() const { return total() > 0 ? 100.0 * poly2_count / total() : 0; }
    double poly3_pct() const { return total() > 0 ? 100.0 * poly3_count / total() : 0; }
    double for_bp_pct() const { return total() > 0 ? 100.0 * for_bp_count / total() : 0; }
};

struct TestResult {
    // Dataset info
    int dataset_id;
    std::string dataset_name;
    std::string data_type;
    size_t num_elements;
    double original_size_mb;

    // Test configuration
    PartitionStrategy partition_strategy;
    DecompressMode decompress_mode;

    // Compression results
    double compressed_size_mb;
    double metadata_size_mb;
    double delta_array_size_mb;
    double compression_ratio;
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double min_delta_bits;
    double max_delta_bits;

    // Model distribution
    ModelStats model_stats;

    // Timing (ms)
    double compress_time_ms;
    double decompress_time_ms;
    double random_access_time_ms;  // For 10000 random accesses

    // Throughput (GB/s)
    double compress_throughput_gbps;
    double decompress_throughput_gbps;

    // Correctness
    bool compress_success;
    bool decompress_correct;
    bool random_access_correct;
    std::string error_msg;
};

// ============================================================================
// Data Loading
// ============================================================================

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for 8-byte header
    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));

    size_t data_bytes = file_size - sizeof(uint64_t);
    size_t expected_with_header = data_bytes / sizeof(T);

    std::vector<T> data;
    if (header_count == expected_with_header) {
        data.resize(header_count);
        file.read(reinterpret_cast<char*>(data.data()), header_count * sizeof(T));
    } else {
        file.seekg(0, std::ios::beg);
        size_t num_elements = file_size / sizeof(T);
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }

    return data;
}

// ============================================================================
// Test Implementations
// ============================================================================

template<typename T>
bool runSingleTest(
    const std::vector<T>& data,
    const DatasetInfo& dataset,
    PartitionStrategy strategy,
    DecompressMode decomp_mode,
    TestResult& result)
{
    result.dataset_id = dataset.id;
    result.dataset_name = dataset.name;
    result.data_type = dataset.type;
    result.num_elements = data.size();
    result.original_size_mb = static_cast<double>(data.size() * sizeof(T)) / (1024.0 * 1024.0);
    result.partition_strategy = strategy;
    result.decompress_mode = decomp_mode;
    result.compress_success = false;
    result.decompress_correct = false;
    result.random_access_correct = false;
    result.error_msg = "";

    if (data.empty()) {
        result.error_msg = "Empty data";
        return false;
    }

    double data_bytes = data.size() * sizeof(T);

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // ========== Compression Phase ==========
    std::vector<PartitionInfo> partitions;
    int num_partitions = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    try {
        switch (strategy) {
            case PartitionStrategy::FIXED_1024: {
                partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), 1024);
                num_partitions = partitions.size();
                break;
            }
            case PartitionStrategy::FIXED_2048: {
                partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), 2048);
                num_partitions = partitions.size();
                break;
            }
            case PartitionStrategy::FIXED_4096: {
                partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), 4096);
                num_partitions = partitions.size();
                break;
            }
            case PartitionStrategy::COST_OPTIMAL_BALANCED: {
                CostOptimalConfig co_config = CostOptimalConfig::balanced();
                partitions = createPartitionsCostOptimal<T>(data, co_config, &num_partitions, 0);
                break;
            }
            case PartitionStrategy::COST_OPTIMAL_HIGH_COMPRESSION: {
                CostOptimalConfig co_config = CostOptimalConfig::highCompression();
                partitions = createPartitionsCostOptimal<T>(data, co_config, &num_partitions, 0);
                break;
            }
            case PartitionStrategy::COST_OPTIMAL_HIGH_THROUGHPUT: {
                CostOptimalConfig co_config = CostOptimalConfig::highThroughput();
                partitions = createPartitionsCostOptimal<T>(data, co_config, &num_partitions, 0);
                break;
            }
        }
    } catch (const std::exception& e) {
        result.error_msg = std::string("Partitioning failed: ") + e.what();
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    if (partitions.empty()) {
        result.error_msg = "Partitioning produced no partitions";
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    // Use Vertical encoder with adaptive model selection
    VerticalConfig fl_config;
    fl_config.enable_adaptive_selection = true;
    fl_config.enable_interleaved = (decomp_mode == DecompressMode::INTERLEAVED);

    CompressedDataVertical<T> compressed;
    try {
        compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config, 0);
    } catch (const std::exception& e) {
        result.error_msg = std::string("Encoding failed: ") + e.what();
        cudaFree(d_data);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return false;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, stop));
    result.compress_time_ms = compress_ms;
    result.compress_throughput_gbps = (data_bytes / 1e9) / (compress_ms / 1e3);

    result.num_partitions = compressed.num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / compressed.num_partitions;

    // Analyze partitions for model stats and delta bits
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    std::vector<int32_t> h_delta_bits(compressed.num_partitions);

    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    double total_bits = 0, min_bits = 64, max_bits = 0;
    for (int i = 0; i < compressed.num_partitions; i++) {
        switch (h_model_types[i]) {
            case MODEL_CONSTANT: result.model_stats.constant_count++; break;
            case MODEL_LINEAR: result.model_stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.model_stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.model_stats.poly3_count++; break;
            case MODEL_FOR_BITPACK: result.model_stats.for_bp_count++; break;
        }
        total_bits += h_delta_bits[i];
        min_bits = std::min(min_bits, static_cast<double>(h_delta_bits[i]));
        max_bits = std::max(max_bits, static_cast<double>(h_delta_bits[i]));
    }
    result.avg_delta_bits = total_bits / compressed.num_partitions;
    result.min_delta_bits = min_bits;
    result.max_delta_bits = max_bits;

    // Calculate compressed sizes
    result.metadata_size_mb = static_cast<double>(
        compressed.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2)
    ) / (1024.0 * 1024.0);
    result.delta_array_size_mb = static_cast<double>(
        compressed.sequential_delta_words * sizeof(uint32_t)
    ) / (1024.0 * 1024.0);
    result.compressed_size_mb = result.metadata_size_mb + result.delta_array_size_mb;
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;

    result.compress_success = true;

    // ========== Decompression Phase ==========
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, data_bytes));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    Vertical_decoder::decompressAll<T>(compressed, d_output, decomp_mode, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float decompress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&decompress_ms, start, stop));
    result.decompress_time_ms = decompress_ms;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (decompress_ms / 1e3);

    // Verify decompression correctness
    std::vector<T> decompressed(data.size());
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    result.decompress_correct = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decompressed[i]) {
            result.decompress_correct = false;
            result.error_msg = "Decompression mismatch at index " + std::to_string(i) +
                              ": expected " + std::to_string(data[i]) +
                              ", got " + std::to_string(decompressed[i]);
            break;
        }
    }

    // ========== Random Access Phase ==========
    const int NUM_RANDOM_ACCESSES = 10000;
    std::vector<int> random_indices(NUM_RANDOM_ACCESSES);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::uniform_int_distribution<int> dist(0, data.size() - 1);
    for (int i = 0; i < NUM_RANDOM_ACCESSES; i++) {
        random_indices[i] = dist(gen);
    }

    int* d_indices;
    T* d_ra_output;
    CUDA_CHECK(cudaMalloc(&d_indices, NUM_RANDOM_ACCESSES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ra_output, NUM_RANDOM_ACCESSES * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_indices, random_indices.data(),
                          NUM_RANDOM_ACCESSES * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    Vertical_decoder::decompressIndices<T>(compressed, d_indices, NUM_RANDOM_ACCESSES, d_ra_output, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ra_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ra_ms, start, stop));
    result.random_access_time_ms = ra_ms;

    // Verify random access correctness
    std::vector<T> ra_results(NUM_RANDOM_ACCESSES);
    CUDA_CHECK(cudaMemcpy(ra_results.data(), d_ra_output,
                          NUM_RANDOM_ACCESSES * sizeof(T), cudaMemcpyDeviceToHost));

    result.random_access_correct = true;
    for (int i = 0; i < NUM_RANDOM_ACCESSES; i++) {
        if (ra_results[i] != data[random_indices[i]]) {
            result.random_access_correct = false;
            if (result.error_msg.empty()) {
                result.error_msg = "Random access mismatch at query " + std::to_string(i) +
                                  " (idx=" + std::to_string(random_indices[i]) +
                                  "): expected " + std::to_string(data[random_indices[i]]) +
                                  ", got " + std::to_string(ra_results[i]);
            }
            break;
        }
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_indices);
    cudaFree(d_ra_output);
    Vertical_encoder::freeCompressedData(compressed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}

// ============================================================================
// Report Generation
// ============================================================================

void generateReports(const std::vector<TestResult>& results, const std::string& output_dir) {
    // Create output directory
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());

    // 1. Summary report (markdown)
    {
        std::ofstream f(output_dir + "/summary_report.md");
        f << "# L3 Ultra-Comprehensive Test Report\n\n";
        f << "**Generated**: " << __DATE__ << " " << __TIME__ << "\n\n";
        f << "## Test Configuration\n\n";
        f << "- **Datasets**: 20 SOSD datasets\n";
        f << "- **Partitioning Strategies**: FIXED(1024/2048/4096), COST_OPTIMAL(Balanced/HC/HT)\n";
        f << "- **Decompression Modes**: AUTO, SEQUENTIAL, INTERLEAVED, BRANCHLESS\n";
        f << "- **Model Types**: CONSTANT, LINEAR, POLY2, POLY3, FOR_BITPACK (auto-selected)\n";
        f << "- **Random Access Tests**: 10,000 random queries per configuration\n\n";

        f << "## Overall Statistics\n\n";

        int total_tests = results.size();
        int compress_success = 0, decomp_correct = 0, ra_correct = 0;
        double avg_ratio = 0, avg_compress_tp = 0, avg_decomp_tp = 0;

        for (const auto& r : results) {
            if (r.compress_success) compress_success++;
            if (r.decompress_correct) decomp_correct++;
            if (r.random_access_correct) ra_correct++;
            if (r.compress_success) {
                avg_ratio += r.compression_ratio;
                avg_compress_tp += r.compress_throughput_gbps;
                avg_decomp_tp += r.decompress_throughput_gbps;
            }
        }

        f << "| Metric | Value |\n";
        f << "|--------|-------|\n";
        f << "| Total Tests | " << total_tests << " |\n";
        f << "| Compression Success | " << compress_success << " / " << total_tests
          << " (" << std::fixed << std::setprecision(1) << (100.0 * compress_success / total_tests) << "%) |\n";
        f << "| Decompression Correct | " << decomp_correct << " / " << total_tests
          << " (" << (100.0 * decomp_correct / total_tests) << "%) |\n";
        f << "| Random Access Correct | " << ra_correct << " / " << total_tests
          << " (" << (100.0 * ra_correct / total_tests) << "%) |\n";
        if (compress_success > 0) {
            f << "| Avg Compression Ratio | " << std::setprecision(2) << (avg_ratio / compress_success) << "x |\n";
            f << "| Avg Compress Throughput | " << (avg_compress_tp / compress_success) << " GB/s |\n";
            f << "| Avg Decompress Throughput | " << (avg_decomp_tp / compress_success) << " GB/s |\n\n";
        }

        f.close();
    }

    // 2. Detailed CSV for all results
    {
        std::ofstream f(output_dir + "/detailed_results.csv");
        f << "dataset_id,dataset_name,data_type,num_elements,original_size_mb,"
          << "partition_strategy,decompress_mode,"
          << "compressed_size_mb,compression_ratio,num_partitions,avg_partition_size,"
          << "avg_delta_bits,min_delta_bits,max_delta_bits,"
          << "constant_pct,linear_pct,poly2_pct,poly3_pct,for_bp_pct,"
          << "compress_time_ms,decompress_time_ms,random_access_time_ms,"
          << "compress_throughput_gbps,decompress_throughput_gbps,"
          << "compress_success,decompress_correct,random_access_correct,error_msg\n";

        for (const auto& r : results) {
            f << r.dataset_id << "," << r.dataset_name << "," << r.data_type << ","
              << r.num_elements << "," << std::fixed << std::setprecision(4) << r.original_size_mb << ","
              << strategyToString(r.partition_strategy) << "," << decompModeToString(r.decompress_mode) << ","
              << r.compressed_size_mb << "," << r.compression_ratio << ","
              << r.num_partitions << "," << r.avg_partition_size << ","
              << r.avg_delta_bits << "," << r.min_delta_bits << "," << r.max_delta_bits << ","
              << r.model_stats.constant_pct() << "," << r.model_stats.linear_pct() << ","
              << r.model_stats.poly2_pct() << "," << r.model_stats.poly3_pct() << ","
              << r.model_stats.for_bp_pct() << ","
              << r.compress_time_ms << "," << r.decompress_time_ms << "," << r.random_access_time_ms << ","
              << r.compress_throughput_gbps << "," << r.decompress_throughput_gbps << ","
              << (r.compress_success ? "true" : "false") << ","
              << (r.decompress_correct ? "true" : "false") << ","
              << (r.random_access_correct ? "true" : "false") << ","
              << "\"" << r.error_msg << "\"\n";
        }
        f.close();
    }

    // 3. Per-strategy comparison
    {
        std::ofstream f(output_dir + "/strategy_comparison.md");
        f << "# Partitioning Strategy Comparison\n\n";

        std::map<PartitionStrategy, std::vector<const TestResult*>> by_strategy;
        for (const auto& r : results) {
            by_strategy[r.partition_strategy].push_back(&r);
        }

        f << "| Strategy | Avg Ratio | Avg Decomp GB/s | Success Rate |\n";
        f << "|----------|-----------|-----------------|-------------|\n";

        for (auto& [strat, strat_results] : by_strategy) {
            double avg_ratio = 0, avg_tp = 0;
            int success = 0, valid = 0;
            for (auto* r : strat_results) {
                if (r->compress_success) {
                    avg_ratio += r->compression_ratio;
                    avg_tp += r->decompress_throughput_gbps;
                    valid++;
                }
                if (r->decompress_correct) success++;
            }
            int n = strat_results.size();
            if (valid > 0) {
                f << "| " << strategyToString(strat) << " | "
                  << std::fixed << std::setprecision(2) << (avg_ratio / valid) << "x | "
                  << (avg_tp / valid) << " | "
                  << success << "/" << n << " |\n";
            }
        }
        f << "\n";
        f.close();
    }

    // 4. Per-decompress mode comparison
    {
        std::ofstream f(output_dir + "/decompress_mode_comparison.md");
        f << "# Decompression Mode Comparison\n\n";

        std::map<DecompressMode, std::vector<const TestResult*>> by_mode;
        for (const auto& r : results) {
            by_mode[r.decompress_mode].push_back(&r);
        }

        f << "| Mode | Avg Decomp Time (ms) | Avg Throughput (GB/s) | Correct Rate |\n";
        f << "|------|---------------------|----------------------|-------------|\n";

        for (auto& [mode, mode_results] : by_mode) {
            double avg_time = 0, avg_tp = 0;
            int correct = 0, valid = 0;
            for (auto* r : mode_results) {
                if (r->compress_success) {
                    avg_time += r->decompress_time_ms;
                    avg_tp += r->decompress_throughput_gbps;
                    valid++;
                }
                if (r->decompress_correct) correct++;
            }
            int n = mode_results.size();
            if (valid > 0) {
                f << "| " << decompModeToString(mode) << " | "
                  << std::fixed << std::setprecision(2) << (avg_time / valid) << " | "
                  << (avg_tp / valid) << " | "
                  << correct << "/" << n << " |\n";
            }
        }
        f << "\n";
        f.close();
    }

    // 5. Per-dataset summary
    {
        std::ofstream f(output_dir + "/dataset_summary.md");
        f << "# Per-Dataset Summary\n\n";

        std::map<int, std::vector<const TestResult*>> by_dataset;
        for (const auto& r : results) {
            by_dataset[r.dataset_id].push_back(&r);
        }

        f << "| ID | Dataset | Type | Elements | Best Ratio | Best Strategy | Best Decomp Mode |\n";
        f << "|----|---------|------|----------|------------|---------------|------------------|\n";

        for (auto& [did, ds_results] : by_dataset) {
            if (ds_results.empty()) continue;

            const TestResult* best = nullptr;
            for (auto* r : ds_results) {
                if (r->decompress_correct) {
                    if (!best || r->compression_ratio > best->compression_ratio) {
                        best = r;
                    }
                }
            }

            if (best) {
                f << "| " << did << " | " << best->dataset_name << " | " << best->data_type << " | "
                  << best->num_elements << " | " << std::fixed << std::setprecision(2)
                  << best->compression_ratio << "x | "
                  << strategyToString(best->partition_strategy) << " | "
                  << decompModeToString(best->decompress_mode) << " |\n";
            }
        }
        f << "\n";
        f.close();
    }

    // 6. Model distribution analysis
    {
        std::ofstream f(output_dir + "/model_distribution.md");
        f << "# Model Type Distribution Analysis\n\n";

        std::map<int, std::map<PartitionStrategy, ModelStats>> by_dataset_strategy;
        for (const auto& r : results) {
            if (r.decompress_mode == DecompressMode::BRANCHLESS && r.compress_success) {
                auto& stats = by_dataset_strategy[r.dataset_id][r.partition_strategy];
                stats.constant_count += r.model_stats.constant_count;
                stats.linear_count += r.model_stats.linear_count;
                stats.poly2_count += r.model_stats.poly2_count;
                stats.poly3_count += r.model_stats.poly3_count;
                stats.for_bp_count += r.model_stats.for_bp_count;
            }
        }

        f << "| Dataset | Strategy | CONST% | LINEAR% | POLY2% | POLY3% | FOR_BP% | Total Parts |\n";
        f << "|---------|----------|--------|---------|--------|--------|---------|-------------|\n";

        for (auto& [did, strat_stats] : by_dataset_strategy) {
            for (auto& [strat, stats] : strat_stats) {
                f << "| " << did << " | " << strategyToString(strat) << " | "
                  << std::fixed << std::setprecision(1)
                  << stats.constant_pct() << " | "
                  << stats.linear_pct() << " | "
                  << stats.poly2_pct() << " | "
                  << stats.poly3_pct() << " | "
                  << stats.for_bp_pct() << " | "
                  << stats.total() << " |\n";
            }
        }
        f << "\n";
        f.close();
    }

    // 7. Errors report
    {
        std::ofstream f(output_dir + "/errors.md");
        f << "# Test Errors and Failures\n\n";

        int error_count = 0;
        for (const auto& r : results) {
            if (!r.compress_success || !r.decompress_correct || !r.random_access_correct) {
                error_count++;
                f << "## Dataset " << r.dataset_id << " - " << r.dataset_name << "\n";
                f << "- **Strategy**: " << strategyToString(r.partition_strategy) << "\n";
                f << "- **Decompress Mode**: " << decompModeToString(r.decompress_mode) << "\n";
                f << "- **Compress Success**: " << (r.compress_success ? "✓" : "✗") << "\n";
                f << "- **Decompress Correct**: " << (r.decompress_correct ? "✓" : "✗") << "\n";
                f << "- **Random Access Correct**: " << (r.random_access_correct ? "✓" : "✗") << "\n";
                if (!r.error_msg.empty()) {
                    f << "- **Error**: " << r.error_msg << "\n";
                }
                f << "\n";
            }
        }

        if (error_count == 0) {
            f << "**All tests passed successfully!** ✓\n";
        } else {
            f << "**Total failures**: " << error_count << "\n";
        }
        f.close();
    }

    // 8. Best configurations per dataset
    {
        std::ofstream f(output_dir + "/best_configurations.md");
        f << "# Best Configuration per Dataset\n\n";
        f << "## By Compression Ratio\n\n";

        std::map<int, const TestResult*> best_ratio;
        std::map<int, const TestResult*> best_throughput;

        for (const auto& r : results) {
            if (!r.decompress_correct) continue;

            if (best_ratio.find(r.dataset_id) == best_ratio.end() ||
                r.compression_ratio > best_ratio[r.dataset_id]->compression_ratio) {
                best_ratio[r.dataset_id] = &r;
            }
            if (best_throughput.find(r.dataset_id) == best_throughput.end() ||
                r.decompress_throughput_gbps > best_throughput[r.dataset_id]->decompress_throughput_gbps) {
                best_throughput[r.dataset_id] = &r;
            }
        }

        f << "| ID | Dataset | Best Ratio | Strategy | Decomp GB/s |\n";
        f << "|----|---------|------------|----------|-------------|\n";
        for (auto& [did, r] : best_ratio) {
            f << "| " << did << " | " << r->dataset_name << " | "
              << std::fixed << std::setprecision(2) << r->compression_ratio << "x | "
              << strategyToString(r->partition_strategy) << " | "
              << r->decompress_throughput_gbps << " |\n";
        }

        f << "\n## By Decompression Throughput\n\n";
        f << "| ID | Dataset | Decomp GB/s | Strategy | Ratio |\n";
        f << "|----|---------|-------------|----------|-------|\n";
        for (auto& [did, r] : best_throughput) {
            f << "| " << did << " | " << r->dataset_name << " | "
              << std::fixed << std::setprecision(2) << r->decompress_throughput_gbps << " | "
              << strategyToString(r->partition_strategy) << " | "
              << r->compression_ratio << "x |\n";
        }
        f << "\n";
        f.close();
    }

    std::cout << "\nReports generated in: " << output_dir << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/data/sosd";
    std::string output_dir = "/root/autodl-tmp/code/L3/reports/L3/All/ultra-comprehensive";

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) output_dir = argv[2];

    // Define all 20 datasets
    std::vector<DatasetInfo> datasets = {
        {1, "linear_200M", "1-linear_200M_uint64.bin", "uint64"},
        {2, "normal_200M", "2-normal_200M_uint64.bin", "uint64"},
        {3, "poisson_87M", "3-poisson_87M_uint64.bin", "uint64"},
        {4, "ml", "4-ml_uint64.bin", "uint64"},
        {5, "books_200M", "5-books_200M_uint32.bin", "uint32"},
        {6, "fb_200M", "6-fb_200M_uint64.bin", "uint64"},
        {7, "wiki_200M", "7-wiki_200M_uint64.bin", "uint64"},
        {8, "osm_800M", "8-osm_cellids_800M_uint64.bin", "uint64"},
        {9, "movieid", "9-movieid_uint32.bin", "uint32"},
        {10, "house_price", "10-house_price_uint64.bin", "uint64"},
        {11, "planet", "11-planet_uint64.bin", "uint64"},
        {12, "libio", "12-libio.bin", "uint64"},
        {13, "medicare", "13-medicare.bin", "uint64"},
        {14, "cosmos", "14-cosmos_int32.bin", "int32"},
        {15, "polylog_10M", "15-polylog_10M_uint64.bin", "uint64"},
        {16, "exp_200M", "16-exp_200M_uint64.bin", "uint64"},
        {17, "poly_200M", "17-poly_200M_uint64.bin", "uint64"},
        {18, "site_250k", "18-site_250k_uint32.bin", "uint32"},
        {19, "weight_25k", "19-weight_25k_uint32.bin", "uint32"},
        {20, "adult_30k", "20-adult_30k_uint32.bin", "uint32"},
    };

    // Test configurations - 6 strategies x 4 decomp modes = 24 configs per dataset
    std::vector<PartitionStrategy> strategies = {
        PartitionStrategy::FIXED_1024,
        PartitionStrategy::FIXED_2048,
        PartitionStrategy::FIXED_4096,
        PartitionStrategy::COST_OPTIMAL_BALANCED,
        PartitionStrategy::COST_OPTIMAL_HIGH_COMPRESSION,
        PartitionStrategy::COST_OPTIMAL_HIGH_THROUGHPUT
    };

    std::vector<DecompressMode> decomp_modes = {
        DecompressMode::AUTO,
        DecompressMode::SEQUENTIAL,
        DecompressMode::INTERLEAVED,
        DecompressMode::BRANCHLESS
    };

    std::vector<TestResult> all_results;

    std::cout << "========================================\n";
    std::cout << "  L3 ULTRA-COMPREHENSIVE TEST SUITE\n";
    std::cout << "========================================\n";
    std::cout << "Data directory: " << data_dir << "\n";
    std::cout << "Output directory: " << output_dir << "\n";
    std::cout << "Datasets: " << datasets.size() << "\n";
    std::cout << "Strategies: " << strategies.size() << "\n";
    std::cout << "Decompress modes: " << decomp_modes.size() << "\n";
    std::cout << "Total test configurations: " << (datasets.size() * strategies.size() * decomp_modes.size()) << "\n";
    std::cout << "========================================\n\n";

    int test_count = 0;
    int total_tests = datasets.size() * strategies.size() * decomp_modes.size();

    for (const auto& dataset : datasets) {
        std::cout << "\n===== Dataset " << dataset.id << "/20: " << dataset.name << " =====\n";

        // Load data based on type
        std::string path = data_dir + "/" + dataset.filename;

        if (dataset.type == "uint64") {
            auto data = loadBinaryData<uint64_t>(path);
            if (data.empty()) {
                std::cout << "SKIP: Could not load " << path << "\n";
                // Add placeholder results for skipped tests
                for (auto strategy : strategies) {
                    for (auto mode : decomp_modes) {
                        TestResult result;
                        result.dataset_id = dataset.id;
                        result.dataset_name = dataset.name;
                        result.data_type = dataset.type;
                        result.partition_strategy = strategy;
                        result.decompress_mode = mode;
                        result.error_msg = "Could not load file";
                        all_results.push_back(result);
                        test_count++;
                    }
                }
                continue;
            }

            // Limit to 200M for very large datasets to avoid OOM
            if (data.size() > 200000000) {
                std::cout << "Limiting to 200M elements (was " << data.size() << ")\n";
                data.resize(200000000);
            }

            for (auto strategy : strategies) {
                for (auto mode : decomp_modes) {
                    test_count++;
                    std::cout << "[" << test_count << "/" << total_tests << "] "
                              << strategyToString(strategy) << " + " << decompModeToString(mode) << "... ";
                    std::cout.flush();

                    TestResult result;
                    bool ok = runSingleTest<uint64_t>(data, dataset, strategy, mode, result);
                    all_results.push_back(result);

                    if (ok && result.decompress_correct) {
                        std::cout << "OK (ratio=" << std::fixed << std::setprecision(2)
                                  << result.compression_ratio << "x, "
                                  << result.decompress_throughput_gbps << " GB/s)\n";
                    } else {
                        std::cout << "FAIL: " << result.error_msg << "\n";
                    }
                }
            }
        } else if (dataset.type == "uint32") {
            auto data = loadBinaryData<uint32_t>(path);
            if (data.empty()) {
                std::cout << "SKIP: Could not load " << path << "\n";
                for (auto strategy : strategies) {
                    for (auto mode : decomp_modes) {
                        TestResult result;
                        result.dataset_id = dataset.id;
                        result.dataset_name = dataset.name;
                        result.data_type = dataset.type;
                        result.partition_strategy = strategy;
                        result.decompress_mode = mode;
                        result.error_msg = "Could not load file";
                        all_results.push_back(result);
                        test_count++;
                    }
                }
                continue;
            }

            if (data.size() > 200000000) {
                std::cout << "Limiting to 200M elements (was " << data.size() << ")\n";
                data.resize(200000000);
            }

            for (auto strategy : strategies) {
                for (auto mode : decomp_modes) {
                    test_count++;
                    std::cout << "[" << test_count << "/" << total_tests << "] "
                              << strategyToString(strategy) << " + " << decompModeToString(mode) << "... ";
                    std::cout.flush();

                    TestResult result;
                    bool ok = runSingleTest<uint32_t>(data, dataset, strategy, mode, result);
                    all_results.push_back(result);

                    if (ok && result.decompress_correct) {
                        std::cout << "OK (ratio=" << std::fixed << std::setprecision(2)
                                  << result.compression_ratio << "x, "
                                  << result.decompress_throughput_gbps << " GB/s)\n";
                    } else {
                        std::cout << "FAIL: " << result.error_msg << "\n";
                    }
                }
            }
        } else if (dataset.type == "int32") {
            auto data = loadBinaryData<int32_t>(path);
            if (data.empty()) {
                std::cout << "SKIP: Could not load " << path << "\n";
                for (auto strategy : strategies) {
                    for (auto mode : decomp_modes) {
                        TestResult result;
                        result.dataset_id = dataset.id;
                        result.dataset_name = dataset.name;
                        result.data_type = dataset.type;
                        result.partition_strategy = strategy;
                        result.decompress_mode = mode;
                        result.error_msg = "Could not load file";
                        all_results.push_back(result);
                        test_count++;
                    }
                }
                continue;
            }

            for (auto strategy : strategies) {
                for (auto mode : decomp_modes) {
                    test_count++;
                    std::cout << "[" << test_count << "/" << total_tests << "] "
                              << strategyToString(strategy) << " + " << decompModeToString(mode) << "... ";
                    std::cout.flush();

                    TestResult result;
                    bool ok = runSingleTest<int32_t>(data, dataset, strategy, mode, result);
                    all_results.push_back(result);

                    if (ok && result.decompress_correct) {
                        std::cout << "OK (ratio=" << std::fixed << std::setprecision(2)
                                  << result.compression_ratio << "x, "
                                  << result.decompress_throughput_gbps << " GB/s)\n";
                    } else {
                        std::cout << "FAIL: " << result.error_msg << "\n";
                    }
                }
            }
        } else if (dataset.type == "int64") {
            auto data = loadBinaryData<int64_t>(path);
            if (data.empty()) {
                std::cout << "SKIP: Could not load " << path << "\n";
                for (auto strategy : strategies) {
                    for (auto mode : decomp_modes) {
                        TestResult result;
                        result.dataset_id = dataset.id;
                        result.dataset_name = dataset.name;
                        result.data_type = dataset.type;
                        result.partition_strategy = strategy;
                        result.decompress_mode = mode;
                        result.error_msg = "Could not load file";
                        all_results.push_back(result);
                        test_count++;
                    }
                }
                continue;
            }

            for (auto strategy : strategies) {
                for (auto mode : decomp_modes) {
                    test_count++;
                    std::cout << "[" << test_count << "/" << total_tests << "] "
                              << strategyToString(strategy) << " + " << decompModeToString(mode) << "... ";
                    std::cout.flush();

                    TestResult result;
                    bool ok = runSingleTest<int64_t>(data, dataset, strategy, mode, result);
                    all_results.push_back(result);

                    if (ok && result.decompress_correct) {
                        std::cout << "OK (ratio=" << std::fixed << std::setprecision(2)
                                  << result.compression_ratio << "x, "
                                  << result.decompress_throughput_gbps << " GB/s)\n";
                    } else {
                        std::cout << "FAIL: " << result.error_msg << "\n";
                    }
                }
            }
        }
    }

    std::cout << "\n========================================\n";
    std::cout << "  Generating Reports...\n";
    std::cout << "========================================\n";

    generateReports(all_results, output_dir);

    std::cout << "\n========================================\n";
    std::cout << "  TESTS COMPLETE\n";
    std::cout << "========================================\n";

    return 0;
}
