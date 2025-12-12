/**
 * Benchmark: Horizontal vs Vertical Data Layout Comparison
 *
 * This benchmark compares data organization structures ONLY:
 * - Horizontal Structure: L3 encoder (sequential bit-packing)
 * - Vertical Structure: Vertical encoder (interleaved mini-vectors)
 *
 * CRITICAL: Both encoders use IDENTICAL:
 * - Partitioning strategy (V2 pure GPU partitioner)
 * - Model selection (same model_type per partition)
 * - Residual computation (same delta_bits per partition)
 *
 * The ONLY difference is how compressed deltas are organized in memory.
 *
 * Measures:
 * - Compression ratio (should be similar for both)
 * - Decompression throughput
 * - Random Access throughput
 * - Model type distribution statistics
 *
 * Author: L3 Benchmark Suite
 * Date: 2025-12-08
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <random>
#include <unordered_map>
#include <cuda_runtime.h>

// L3 headers
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "L3_random_access.hpp"
#include "L3.h"
#include "sosd_loader.h"

// Partitioner header (V2 only)
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

// ============================================================================
// Configuration
// ============================================================================

const std::string DATA_DIR = "/root/autodl-tmp/test/data/sosd";
const std::string REPORT_PATH = "/root/autodl-tmp/code/L3/papers/responses/R2/O3/benchmark_report.md";
const std::string LOG_PATH = "/root/autodl-tmp/code/L3/papers/responses/sgt.txt";

const int PARTITION_SIZE = 4096;
const int WARMUP_RUNS = 3;
const int TIMED_RUNS = 5;
const size_t MIN_BENCHMARK_SIZE = 1000;

// ============================================================================
// CUDA Error Checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Partition Generation (V2 Only - Pure GPU)
// ============================================================================

template<typename T>
std::vector<PartitionInfo> generatePartitionsV2(
    const std::vector<T>& data, int partition_size)
{
    CostOptimalConfig config;
    config.target_partition_size = partition_size;
    config.analysis_block_size = partition_size / 2;
    config.min_partition_size = 256;
    config.max_partition_size = partition_size * 2;
    config.breakpoint_threshold = 2;
    config.merge_benefit_threshold = 0.05f;
    config.max_merge_rounds = 4;
    config.enable_merging = true;
    config.enable_polynomial_models = true;  // Enable POLY2/POLY3
    config.polynomial_min_size = 10;
    config.cubic_min_size = 20;
    config.polynomial_cost_threshold = 0.95f;

    GPUCostOptimalPartitionerV2<T> partitioner(data, config, 0);
    return partitioner.partition();
}

// ============================================================================
// Data Structures
// ============================================================================

struct DetailedTiming {
    double h2d_ms = 0.0;
    double kernel_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
    double throughput_gbps = 0.0;
};

struct ModelTypeStats {
    int linear = 0;
    int poly2 = 0;
    int poly3 = 0;
    int for_bitpack = 0;

    int total() const { return linear + poly2 + poly3 + for_bitpack; }
};

struct BenchmarkResult {
    std::string dataset_name;
    std::string data_type;
    size_t num_elements;

    // Partition info (V2 only)
    int num_partitions;

    // Model type distribution
    ModelTypeStats model_stats;

    // Compression ratios (should be similar with same partitions)
    double compression_ratio_horizontal;
    double compression_ratio_vertical;

    // Decompression performance
    DetailedTiming decompression_horizontal;
    DetailedTiming decompression_vertical;
    bool decompression_correct_horizontal;
    bool decompression_correct_vertical;

    // Random Access performance
    DetailedTiming random_access_horizontal;
    DetailedTiming random_access_vertical;
    bool random_access_correct_horizontal;
    bool random_access_correct_vertical;
};

struct DatasetInfo {
    std::string filename;
    std::string name;
    bool is_uint64;
    size_t expected_elements;
};

// Dataset definitions (1-20)
const std::unordered_map<int, DatasetInfo> DATASETS = {
    {1, {"1-linear_200M_uint64.bin", "linear", true, 200000000}},
    {2, {"2-normal_200M_uint64.bin", "normal", true, 200000000}},
    {3, {"3-poisson_87M_uint64.bin", "poisson", true, 87000000}},
    {4, {"4-ml_uint64.bin", "ml", true, 0}},
    {5, {"5-books_200M_uint32.bin", "books", false, 200000000}},
    {6, {"6-fb_200M_uint64.bin", "fb", true, 200000000}},
    {7, {"7-wiki_200M_uint64.bin", "wiki", true, 200000000}},
    {8, {"8-osm_cellids_800M_uint64.bin", "osm", true, 800000000}},
    {9, {"9-movieid_uint32.bin", "movieid", false, 0}},
    {10, {"10-house_price_uint64.bin", "house_price", true, 0}},
    {11, {"11-planet_uint64.bin", "planet", true, 200000000}},
    {12, {"12-libio.bin", "libio", true, 200000000}},
    {13, {"13-medicare.bin", "medicare", true, 0}},
    {14, {"14-cosmos_int32.bin", "cosmos", false, 0}},
    {15, {"15-polylog_10M_uint64.bin", "polylog", true, 10000000}},
    {16, {"16-exp_200M_uint64.bin", "exp", true, 200000000}},
    {17, {"17-poly_200M_uint64.bin", "poly", true, 200000000}},
    {18, {"18-site_250k_uint32.bin", "site", false, 250000}},
    {19, {"19-weight_25k_uint32.bin", "weight", false, 25000}},
    {20, {"20-adult_30k_uint32.bin", "adult", false, 30000}},
};

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
std::vector<T> loadDataset(const DatasetInfo& info) {
    std::string filepath = DATA_DIR + "/" + info.filename;
    std::vector<T> data;
    bool success = SOSDLoader::loadBinary<T>(filepath, data);
    if (!success) {
        std::cerr << "Failed to load dataset: " << filepath << std::endl;
        return {};
    }
    return data;
}

std::string getGPUName() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    return std::string(prop.name);
}

double getMedian(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

void countModelTypes(const std::vector<PartitionInfo>& partitions, ModelTypeStats& stats) {
    stats = ModelTypeStats();
    for (const auto& p : partitions) {
        switch (p.model_type) {
            case MODEL_LINEAR:      stats.linear++; break;
            case MODEL_POLYNOMIAL2: stats.poly2++; break;
            case MODEL_POLYNOMIAL3: stats.poly3++; break;
            case MODEL_FOR_BITPACK: stats.for_bitpack++; break;
        }
    }
}

// ============================================================================
// Horizontal (L3) Benchmarks
// ============================================================================

template<typename T>
double benchmarkHorizontalCompression(
    const std::vector<T>& data,
    const std::vector<PartitionInfo>& partitions,
    CompressedDataL3<T>** compressed_out)
{
    size_t data_bytes = data.size() * sizeof(T);

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto* temp = compressDataWithPartitions(data, partitions, nullptr);
        if (temp) freeCompressedData(temp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get result
    auto* result = compressDataWithPartitions(data, partitions, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());

    *compressed_out = result;

    // Calculate compression ratio
    size_t compressed_bytes = result->delta_array_words * sizeof(uint32_t) +
                             result->num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t));
    return static_cast<double>(data_bytes) / compressed_bytes;
}

template<typename T>
DetailedTiming benchmarkHorizontalDecompression(
    CompressedDataL3<T>* compressed,
    const std::vector<T>& original,
    bool* correct_out)
{
    DetailedTiming timing;
    size_t n = original.size();
    size_t data_bytes = n * sizeof(T);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        launchDecompressWarpOpt<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    std::vector<double> kernel_times;
    for (int run = 0; run < TIMED_RUNS; run++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_start));
        launchDecompressWarpOpt<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms);
    }

    timing.kernel_ms = getMedian(kernel_times);
    timing.throughput_gbps = (data_bytes / 1e9) / (timing.kernel_ms / 1000.0);

    // Verify Horizontal Decompression
    std::vector<T> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    *correct_out = true;
    int error_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (output[i] != original[i]) {
            if (error_count < 5) {
                std::cerr << "  H-Decomp MISMATCH[" << i << "]: got " << output[i]
                          << ", expected " << original[i]
                          << " (diff=" << (int64_t)(output[i] - original[i]) << ")" << std::endl;
            }
            error_count++;
            *correct_out = false;
        }
    }
    if (error_count > 0) {
        std::cerr << "  H-Decomp Total errors: " << error_count << " / " << n << std::endl;
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return timing;
}

template<typename T>
DetailedTiming benchmarkHorizontalRandomAccess(
    CompressedDataL3<T>* compressed,
    const std::vector<T>& original,
    bool* correct_out)
{
    DetailedTiming timing;
    size_t n = original.size();
    size_t data_bytes = n * sizeof(T);

    std::vector<int> h_indices(n);
    std::mt19937 rng(42);
    for (size_t i = 0; i < n; i++) {
        h_indices[i] = rng() % n;
    }

    int* d_indices;
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    // Warmup
    RandomAccessStats stats;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        randomAccessOptimized(compressed, d_indices, n, d_output, &stats, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    std::vector<double> kernel_times;
    for (int run = 0; run < TIMED_RUNS; run++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_start));
        randomAccessOptimized(compressed, d_indices, n, d_output, nullptr, 0);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms);
    }

    timing.kernel_ms = getMedian(kernel_times);
    timing.throughput_gbps = (data_bytes / 1e9) / (timing.kernel_ms / 1000.0);

    // Verify
    std::vector<T> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    *correct_out = true;
    for (size_t i = 0; i < n; i++) {
        if (output[i] != original[h_indices[i]]) {
            *correct_out = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return timing;
}

// ============================================================================
// Vertical (Vertical) Benchmarks
// ============================================================================

template<typename T>
double benchmarkVerticalCompression(
    const std::vector<T>& data,
    std::vector<PartitionInfo>& partitions,
    CompressedDataVertical<T>* compressed_out)
{
    size_t data_bytes = data.size() * sizeof(T);

    // Configure Vertical - SKIP metadata recompute to use partition values
    VerticalConfig config;
    config.partition_size_hint = PARTITION_SIZE;
    config.enable_adaptive_selection = true;
    config.enable_interleaved = true;
    config.enable_branchless_unpack = true;
    config.skip_metadata_recompute = true;  // KEY: Use partition values directly!

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        auto temp = Vertical_encoder::encodeVertical<T>(data, partitions, config, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        Vertical_encoder::freeCompressedData(temp);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get result
    auto result = Vertical_encoder::encodeVertical<T>(data, partitions, config, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    *compressed_out = result;

    // Calculate compression ratio
    size_t compressed_bytes = result.interleaved_delta_words * sizeof(uint32_t) +
                             result.num_partitions * (sizeof(int32_t) * 6 + sizeof(double) * 4 + sizeof(int64_t) * 2);
    return static_cast<double>(data_bytes) / compressed_bytes;
}

template<typename T>
DetailedTiming benchmarkVerticalDecompression(
    const CompressedDataVertical<T>& compressed,
    const std::vector<T>& original,
    bool* correct_out)
{
    DetailedTiming timing;
    size_t n = original.size();
    size_t data_bytes = n * sizeof(T);

    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::AUTO, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    std::vector<double> kernel_times;
    for (int run = 0; run < TIMED_RUNS; run++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_start));
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::AUTO, 0);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms);
    }

    timing.kernel_ms = getMedian(kernel_times);
    timing.throughput_gbps = (data_bytes / 1e9) / (timing.kernel_ms / 1000.0);

    // Verify
    std::vector<T> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    *correct_out = true;
    for (size_t i = 0; i < n; i++) {
        if (output[i] != original[i]) {
            *correct_out = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return timing;
}

template<typename T>
DetailedTiming benchmarkVerticalRandomAccess(
    const CompressedDataVertical<T>& compressed,
    const std::vector<T>& original,
    bool* correct_out)
{
    DetailedTiming timing;
    size_t n = original.size();
    size_t data_bytes = n * sizeof(T);

    std::vector<int> h_indices(n);
    std::mt19937 rng(42);  // Same seed as horizontal
    for (size_t i = 0; i < n; i++) {
        h_indices[i] = rng() % n;
    }

    int* d_indices;
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_indices, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    cudaEvent_t kernel_start, kernel_stop;
    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressIndices<T>(compressed, d_indices, n, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    std::vector<double> kernel_times;
    for (int run = 0; run < TIMED_RUNS; run++) {
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(kernel_start));
        Vertical_decoder::decompressIndices<T>(compressed, d_indices, n, d_output, 0);
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float kernel_ms;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, kernel_start, kernel_stop));
        kernel_times.push_back(kernel_ms);
    }

    timing.kernel_ms = getMedian(kernel_times);
    timing.throughput_gbps = (data_bytes / 1e9) / (timing.kernel_ms / 1000.0);

    // Verify
    std::vector<T> output(n);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    *correct_out = true;
    for (size_t i = 0; i < n; i++) {
        if (output[i] != original[h_indices[i]]) {
            *correct_out = false;
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    return timing;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

template<typename T>
BenchmarkResult runDatasetBenchmark(const DatasetInfo& info, std::ostream& log) {
    BenchmarkResult result;
    result.dataset_name = info.name;
    result.data_type = info.is_uint64 ? "uint64" : "uint32";

    log << "\n========================================\n";
    log << "Testing Dataset: " << info.name << " (" << info.filename << ")\n";
    log << "========================================\n";

    // Load data
    log << "Loading dataset..." << std::endl;
    std::vector<T> data = loadDataset<T>(info);
    if (data.empty()) {
        log << "Failed to load dataset: " << info.name << std::endl;
        result.num_elements = 0;
        return result;
    }

    result.num_elements = data.size();
    log << "Loaded " << data.size() << " elements ("
        << (data.size() * sizeof(T) / (1024.0 * 1024.0)) << " MB)\n";

    if (data.size() < MIN_BENCHMARK_SIZE) {
        log << "\n*** SKIPPED: Dataset too small ***\n";
        result.compression_ratio_horizontal = 0.0;
        result.compression_ratio_vertical = 0.0;
        result.decompression_correct_horizontal = true;
        result.decompression_correct_vertical = true;
        result.random_access_correct_horizontal = true;
        result.random_access_correct_vertical = true;
        return result;
    }

    // ========== Step 1: Generate Partitions (V2 Only) ==========
    log << "\n--- Step 1: Generating Partitions (V2 Pure GPU) ---\n";
    std::vector<PartitionInfo> partitions = generatePartitionsV2<T>(data, PARTITION_SIZE);
    result.num_partitions = partitions.size();
    log << "  Partitions: " << result.num_partitions << std::endl;

    // ========== Step 2: Count Model Types ==========
    countModelTypes(partitions, result.model_stats);
    log << "  Model distribution: LINEAR=" << result.model_stats.linear
        << ", POLY2=" << result.model_stats.poly2
        << ", POLY3=" << result.model_stats.poly3
        << ", FOR=" << result.model_stats.for_bitpack << std::endl;

    // ========== Step 3: Horizontal (L3) Compression ==========
    log << "\n--- Step 3: Horizontal (L3) Compression ---\n";
    CompressedDataL3<T>* compressed_h = nullptr;
    result.compression_ratio_horizontal = benchmarkHorizontalCompression(data, partitions, &compressed_h);
    log << "  Compression ratio: " << std::fixed << std::setprecision(2)
        << result.compression_ratio_horizontal << "x" << std::endl;

    // ========== Step 4: Vertical (Vertical) Compression ==========
    log << "\n--- Step 4: Vertical (Vertical) Compression ---\n";
    CompressedDataVertical<T> compressed_v;
    result.compression_ratio_vertical = benchmarkVerticalCompression(data, partitions, &compressed_v);
    log << "  Compression ratio: " << std::fixed << std::setprecision(2)
        << result.compression_ratio_vertical << "x" << std::endl;

    // ========== Step 5: Decompression Tests ==========
    log << "\n--- Step 5: Decompression Tests ---\n";

    log << "Horizontal Decompression..." << std::endl;
    result.decompression_horizontal = benchmarkHorizontalDecompression(
        compressed_h, data, &result.decompression_correct_horizontal);
    log << "  Kernel: " << result.decompression_horizontal.kernel_ms << " ms, "
        << result.decompression_horizontal.throughput_gbps << " GB/s, "
        << (result.decompression_correct_horizontal ? "PASS" : "FAIL") << std::endl;

    log << "Vertical Decompression..." << std::endl;
    result.decompression_vertical = benchmarkVerticalDecompression(
        compressed_v, data, &result.decompression_correct_vertical);
    log << "  Kernel: " << result.decompression_vertical.kernel_ms << " ms, "
        << result.decompression_vertical.throughput_gbps << " GB/s, "
        << (result.decompression_correct_vertical ? "PASS" : "FAIL") << std::endl;

    // ========== Step 6: Random Access Tests ==========
    log << "\n--- Step 6: Random Access Tests (" << data.size() << " accesses) ---\n";

    log << "Horizontal Random Access..." << std::endl;
    result.random_access_horizontal = benchmarkHorizontalRandomAccess(
        compressed_h, data, &result.random_access_correct_horizontal);
    log << "  Kernel: " << result.random_access_horizontal.kernel_ms << " ms, "
        << result.random_access_horizontal.throughput_gbps << " GB/s, "
        << (result.random_access_correct_horizontal ? "PASS" : "FAIL") << std::endl;

    log << "Vertical Random Access..." << std::endl;
    result.random_access_vertical = benchmarkVerticalRandomAccess(
        compressed_v, data, &result.random_access_correct_vertical);
    log << "  Kernel: " << result.random_access_vertical.kernel_ms << " ms, "
        << result.random_access_vertical.throughput_gbps << " GB/s, "
        << (result.random_access_correct_vertical ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    freeCompressedData(compressed_h);
    Vertical_encoder::freeCompressedData(compressed_v);

    return result;
}

// ============================================================================
// Report Generation
// ============================================================================

void generateMarkdownReport(const std::vector<BenchmarkResult>& results, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open report file: " << path << std::endl;
        return;
    }

    std::string gpu_name = getGPUName();

    file << "# Horizontal vs Vertical Data Layout Benchmark\n\n";
    file << "**Date**: " << __DATE__ << "\n";
    file << "**Platform**: " << gpu_name << "\n";
    file << "**Partition Size**: " << PARTITION_SIZE << "\n";
    file << "**Partitioner**: V2 (Pure GPU, Cost-Optimal with Polynomial Models)\n";
    file << "**Random Access**: N accesses where N = dataset element count\n\n";

    file << "## Key Point: Fair Comparison\n\n";
    file << "Both encoders use **IDENTICAL**:\n";
    file << "- Partition boundaries (from V2 partitioner)\n";
    file << "- Model types per partition (LINEAR, POLY2, POLY3, FOR)\n";
    file << "- Model parameters (theta0, theta1, theta2, theta3)\n";
    file << "- Delta bits per partition\n\n";
    file << "The **ONLY** difference is data layout:\n";
    file << "- **Horizontal (L3)**: Sequential bit-packing\n";
    file << "- **Vertical (Vertical)**: Interleaved mini-vectors (256 values)\n\n";

    // Summary Table
    file << "## 1. Summary Results\n\n";
    file << "| # | Dataset | Elements | Parts | H Ratio | V Ratio | Decomp H/V (GB/s) | RA H/V (GB/s) | Correct |\n";
    file << "|---|---------|----------|-------|---------|---------|-------------------|---------------|---------|\n";

    for (size_t i = 0; i < results.size(); i++) {
        const auto& r = results[i];
        if (r.num_elements == 0) continue;

        bool skipped = (r.num_elements < MIN_BENCHMARK_SIZE);

        file << "| " << (i + 1) << " | " << r.dataset_name << " | " << r.num_elements << " | ";

        if (skipped) {
            file << "- | - | - | - | - | - |\n";
        } else {
            file << r.num_partitions << " | "
                 << std::fixed << std::setprecision(2) << r.compression_ratio_horizontal << "x | "
                 << r.compression_ratio_vertical << "x | "
                 << std::setprecision(1) << r.decompression_horizontal.throughput_gbps << " / "
                 << r.decompression_vertical.throughput_gbps << " | "
                 << r.random_access_horizontal.throughput_gbps << " / "
                 << r.random_access_vertical.throughput_gbps << " | "
                 << ((r.decompression_correct_horizontal && r.decompression_correct_vertical &&
                      r.random_access_correct_horizontal && r.random_access_correct_vertical) ? "PASS" : "FAIL")
                 << " |\n";
        }
    }

    // Model Type Distribution
    file << "\n## 2. Model Type Distribution\n\n";
    file << "| Dataset | LINEAR | POLY2 | POLY3 | FOR | Total |\n";
    file << "|---------|--------|-------|-------|-----|-------|\n";

    for (const auto& r : results) {
        if (r.num_elements == 0 || r.num_elements < MIN_BENCHMARK_SIZE) continue;
        file << "| " << r.dataset_name << " | "
             << r.model_stats.linear << " | "
             << r.model_stats.poly2 << " | "
             << r.model_stats.poly3 << " | "
             << r.model_stats.for_bitpack << " | "
             << r.model_stats.total() << " |\n";
    }

    // Decompression Details
    file << "\n## 3. Decompression Performance\n\n";
    file << "| Dataset | H Kernel (ms) | H Throughput | V Kernel (ms) | V Throughput | V/H Speedup |\n";
    file << "|---------|---------------|--------------|---------------|--------------|-------------|\n";

    for (const auto& r : results) {
        if (r.num_elements == 0 || r.num_elements < MIN_BENCHMARK_SIZE) continue;
        double speedup = r.decompression_vertical.throughput_gbps / r.decompression_horizontal.throughput_gbps;
        file << "| " << r.dataset_name << " | "
             << std::fixed << std::setprecision(3) << r.decompression_horizontal.kernel_ms << " | "
             << std::setprecision(1) << r.decompression_horizontal.throughput_gbps << " GB/s | "
             << std::setprecision(3) << r.decompression_vertical.kernel_ms << " | "
             << std::setprecision(1) << r.decompression_vertical.throughput_gbps << " GB/s | "
             << std::setprecision(2) << speedup << "x |\n";
    }

    // Random Access Details
    file << "\n## 4. Random Access Performance\n\n";
    file << "| Dataset | N | H Kernel (ms) | H Throughput | V Kernel (ms) | V Throughput | V/H Speedup |\n";
    file << "|---------|---|---------------|--------------|---------------|--------------|-------------|\n";

    for (const auto& r : results) {
        if (r.num_elements == 0 || r.num_elements < MIN_BENCHMARK_SIZE) continue;
        double speedup = r.random_access_vertical.throughput_gbps / r.random_access_horizontal.throughput_gbps;
        file << "| " << r.dataset_name << " | " << r.num_elements << " | "
             << std::fixed << std::setprecision(1) << r.random_access_horizontal.kernel_ms << " | "
             << r.random_access_horizontal.throughput_gbps << " GB/s | "
             << r.random_access_vertical.kernel_ms << " | "
             << r.random_access_vertical.throughput_gbps << " GB/s | "
             << std::setprecision(2) << speedup << "x |\n";
    }

    // Analysis
    file << "\n## 5. Analysis\n\n";

    // Calculate averages
    double avg_decomp_speedup = 0, avg_ra_speedup = 0;
    double avg_ratio_diff = 0;
    int valid_count = 0;
    for (const auto& r : results) {
        if (r.num_elements == 0 || r.num_elements < MIN_BENCHMARK_SIZE ||
            r.decompression_horizontal.throughput_gbps == 0) continue;
        avg_decomp_speedup += r.decompression_vertical.throughput_gbps / r.decompression_horizontal.throughput_gbps;
        avg_ra_speedup += r.random_access_vertical.throughput_gbps / r.random_access_horizontal.throughput_gbps;
        avg_ratio_diff += (r.compression_ratio_vertical - r.compression_ratio_horizontal) / r.compression_ratio_horizontal * 100.0;
        valid_count++;
    }
    if (valid_count > 0) {
        avg_decomp_speedup /= valid_count;
        avg_ra_speedup /= valid_count;
        avg_ratio_diff /= valid_count;
    }

    file << "### Key Findings\n\n";
    file << "1. **Compression Ratio Difference**: " << std::fixed << std::setprecision(1)
         << (avg_ratio_diff >= 0 ? "+" : "") << avg_ratio_diff << "% (V vs H)\n";
    file << "   - With identical partitions and models, ratio difference is from metadata overhead only\n";
    file << "2. **Decompression Speedup (V/H)**: " << std::setprecision(2) << avg_decomp_speedup << "x\n";
    file << "3. **Random Access Speedup (V/H)**: " << avg_ra_speedup << "x\n";
    file << "4. **Correctness**: All results verified against original data\n\n";

    file << "### Data Layout Comparison\n\n";
    file << "| Aspect | Horizontal (L3) | Vertical (Vertical) |\n";
    file << "|--------|-------------------|---------------------|\n";
    file << "| Data Layout | Sequential bit-packing | Interleaved mini-vectors (256 values) |\n";
    file << "| Decoder | WarpOpt (cp.async) | Interleaved Vertical |\n";
    file << "| Memory Access | Sequential within partition | Coalesced across lanes |\n";
    file << "| Random Access | Binary search + sequential extract | Coordinate mapping + interleaved extract |\n";

    file << "\n---\n\n";
    file << "*Report generated by benchmark_vertical_horizontal*\n";
    file << "*L3 Compression System v3.2*\n";

    file.close();
    std::cout << "Report saved to: " << path << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Open log file
    std::ofstream log_file(LOG_PATH);
    if (!log_file.is_open()) {
        std::cerr << "Failed to open log file: " << LOG_PATH << std::endl;
        return 1;
    }

    // Tee output to both console and log file
    auto log = [&log_file](const std::string& msg) {
        std::cout << msg;
        log_file << msg;
    };

    log_file << "========================================\n";
    log_file << "Horizontal vs Vertical Layout Benchmark\n";
    log_file << "========================================\n";
    log_file << "GPU: " << getGPUName() << "\n";
    log_file << "Partition Size: " << PARTITION_SIZE << "\n";
    log_file << "Partitioner: V2 (Pure GPU, Cost-Optimal)\n";
    log_file << "Random Access: N accesses (N = dataset size)\n";
    log_file << "========================================\n";

    std::cout << "========================================\n";
    std::cout << "Horizontal vs Vertical Layout Benchmark\n";
    std::cout << "========================================\n";
    std::cout << "GPU: " << getGPUName() << "\n";
    std::cout << "Partition Size: " << PARTITION_SIZE << "\n";
    std::cout << "Log file: " << LOG_PATH << "\n";
    std::cout << "========================================\n";

    std::vector<BenchmarkResult> results;

    // Test datasets 1-20
    for (int id = 1; id <= 20; id++) {
        auto it = DATASETS.find(id);
        if (it == DATASETS.end()) continue;

        const DatasetInfo& info = it->second;

        BenchmarkResult result;
        if (info.is_uint64) {
            result = runDatasetBenchmark<uint64_t>(info, log_file);
        } else {
            result = runDatasetBenchmark<uint32_t>(info, log_file);
        }

        results.push_back(result);
    }

    // Generate report
    generateMarkdownReport(results, REPORT_PATH);

    log_file << "\n========================================\n";
    log_file << "Benchmark Complete!\n";
    log_file << "========================================\n";

    std::cout << "\n========================================\n";
    std::cout << "Benchmark Complete!\n";
    std::cout << "Report: " << REPORT_PATH << "\n";
    std::cout << "Log: " << LOG_PATH << "\n";
    std::cout << "========================================\n";

    log_file.close();
    return 0;
}
