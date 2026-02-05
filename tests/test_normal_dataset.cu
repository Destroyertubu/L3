/**
 * L3 Normal Dataset Test
 * Tests normal_200M_uint64 with all partitioning strategies and decompression modes.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>
#include <algorithm>
#include <sstream>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

enum class PartitionStrategy {
    FIXED_2048,
    VARIANCE_ADAPTIVE,
    COST_OPTIMAL_BALANCED
};

std::string strategyToString(PartitionStrategy s) {
    switch (s) {
        case PartitionStrategy::FIXED_2048: return "FIXED_2048";
        case PartitionStrategy::VARIANCE_ADAPTIVE: return "VARIANCE_ADAPTIVE";
        case PartitionStrategy::COST_OPTIMAL_BALANCED: return "COST_OPTIMAL_BALANCED";
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

template<typename T>
std::vector<PartitionInfo> createVarianceAdaptivePartitions(const std::vector<T>& data, int base_size = 2048) {
    std::vector<PartitionInfo> partitions;
    size_t n = data.size();
    if (n == 0) return partitions;

    int analysis_block_size = base_size * 4;
    int num_analysis_blocks = (n + analysis_block_size - 1) / analysis_block_size;

    std::vector<double> block_variances(num_analysis_blocks);
    for (int b = 0; b < num_analysis_blocks; b++) {
        size_t start = b * analysis_block_size;
        size_t end = std::min(start + (size_t)analysis_block_size, n);
        size_t block_n = end - start;

        double sum = 0, sum_sq = 0;
        for (size_t i = start; i < end; i++) {
            double val = static_cast<double>(data[i]);
            sum += val;
            sum_sq += val * val;
        }
        double mean = sum / block_n;
        block_variances[b] = (sum_sq / block_n) - mean * mean;
    }

    std::vector<double> sorted_vars = block_variances;
    std::sort(sorted_vars.begin(), sorted_vars.end());

    double t1 = sorted_vars[sorted_vars.size() * 25 / 100];
    double t2 = sorted_vars[sorted_vars.size() * 50 / 100];
    double t3 = sorted_vars[sorted_vars.size() * 75 / 100];

    int sizes[] = {base_size * 4, base_size * 2, base_size, base_size / 2};

    for (int b = 0; b < num_analysis_blocks; b++) {
        size_t block_start = b * analysis_block_size;
        size_t block_end = std::min(block_start + (size_t)analysis_block_size, n);

        int part_size;
        if (block_variances[b] < t1) part_size = sizes[0];
        else if (block_variances[b] < t2) part_size = sizes[1];
        else if (block_variances[b] < t3) part_size = sizes[2];
        else part_size = sizes[3];

        for (size_t pos = block_start; pos < block_end; pos += part_size) {
            PartitionInfo p;
            p.start_idx = pos;
            p.end_idx = std::min(pos + (size_t)part_size, block_end);
            p.model_type = MODEL_LINEAR;
            p.delta_bits = 0;
            p.delta_array_bit_offset = 0;
            partitions.push_back(p);
        }
    }
    return partitions;
}

struct TestResult {
    std::string strategy;
    std::string mode;
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double compressed_size_mb;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double decompress_throughput_gbps;
    double random_access_time_ms;
    bool decompress_correct;
    bool random_access_correct;
    int constant_count, linear_count, poly2_count, poly3_count, for_bp_count;
};

template<typename T>
TestResult runTest(const std::vector<T>& data, PartitionStrategy strategy, DecompressMode mode) {
    TestResult result;
    result.strategy = strategyToString(strategy);
    result.mode = decompModeToString(mode);
    result.decompress_correct = false;
    result.random_access_correct = false;

    size_t n = data.size();
    double data_bytes = n * sizeof(T);
    double data_mb = data_bytes / (1024.0 * 1024.0);

    // Create partitions
    std::vector<PartitionInfo> partitions;
    int num_partitions = 0;

    switch (strategy) {
        case PartitionStrategy::FIXED_2048:
            partitions = Vertical_encoder::createFixedPartitions<T>(n, 2048);
            num_partitions = partitions.size();
            break;
        case PartitionStrategy::VARIANCE_ADAPTIVE:
            partitions = createVarianceAdaptivePartitions<T>(data, 2048);
            num_partitions = partitions.size();
            break;
        case PartitionStrategy::COST_OPTIMAL_BALANCED: {
            CostOptimalConfig co_config = CostOptimalConfig::balanced();
            partitions = createPartitionsCostOptimal<T>(data, co_config, &num_partitions, 0);
            break;
        }
    }

    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(n) / num_partitions;

    // Compress
    VerticalConfig fl_config;
    fl_config.enable_adaptive_selection = true;
    fl_config.enable_interleaved = (mode == DecompressMode::INTERLEAVED);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, stop));
    result.compress_time_ms = compress_ms;

    // Analyze model stats
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    CUDA_CHECK(cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
                          compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    result.constant_count = result.linear_count = result.poly2_count = result.poly3_count = result.for_bp_count = 0;
    double total_bits = 0;
    for (int i = 0; i < compressed.num_partitions; i++) {
        switch (h_model_types[i]) {
            case MODEL_CONSTANT: result.constant_count++; break;
            case MODEL_LINEAR: result.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.poly3_count++; break;
            case MODEL_FOR_BITPACK: result.for_bp_count++; break;
        }
        total_bits += h_delta_bits[i];
    }
    result.avg_delta_bits = total_bits / compressed.num_partitions;

    double metadata_size = compressed.num_partitions * 64.0;
    double delta_size = compressed.sequential_delta_words * sizeof(uint32_t);
    result.compressed_size_mb = (metadata_size + delta_size) / (1024.0 * 1024.0);
    result.compression_ratio = data_mb / result.compressed_size_mb;

    // Decompress
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));
    CUDA_CHECK(cudaMemset(d_output, 0, data_bytes));

    // Warmup
    Vertical_decoder::decompressAll<T>(compressed, d_output, mode, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed run
    CUDA_CHECK(cudaEventRecord(start));
    Vertical_decoder::decompressAll<T>(compressed, d_output, mode, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float decompress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&decompress_ms, start, stop));
    result.decompress_time_ms = decompress_ms;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (decompress_ms / 1e3);

    // Verify decompression
    std::vector<T> decompressed(n);
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    result.decompress_correct = true;
    int error_count = 0;
    for (size_t i = 0; i < n && error_count < 5; i++) {
        if (data[i] != decompressed[i]) {
            result.decompress_correct = false;
            std::cerr << "  Mismatch at " << i << ": expected " << data[i] << ", got " << decompressed[i] << std::endl;
            error_count++;
        }
    }

    // Random access test
    const int NUM_QUERIES = 10000;
    std::vector<int> queries(NUM_QUERIES);
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < NUM_QUERIES; i++) {
        queries[i] = dist(gen);
    }

    int* d_queries;
    T* d_ra_output;
    CUDA_CHECK(cudaMalloc(&d_queries, NUM_QUERIES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ra_output, NUM_QUERIES * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_queries, queries.data(), NUM_QUERIES * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    Vertical_decoder::decompressIndices<T>(compressed, d_queries, NUM_QUERIES, d_ra_output, 0);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ra_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ra_ms, start, stop));
    result.random_access_time_ms = ra_ms;

    std::vector<T> ra_results(NUM_QUERIES);
    CUDA_CHECK(cudaMemcpy(ra_results.data(), d_ra_output, NUM_QUERIES * sizeof(T), cudaMemcpyDeviceToHost));

    result.random_access_correct = true;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (ra_results[i] != data[queries[i]]) {
            result.random_access_correct = false;
            break;
        }
    }

    // Cleanup
    cudaFree(d_output);
    cudaFree(d_queries);
    cudaFree(d_ra_output);
    Vertical_encoder::freeCompressedData(compressed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main(int argc, char* argv[]) {
    std::string data_file = "data/sosd/2-normal_200M_uint64.bin";
    std::string output_dir = "reports/L3/datasets/2-normal";

    if (argc >= 2) data_file = argv[1];
    if (argc >= 3) output_dir = argv[2];

    std::cout << "========================================" << std::endl;
    std::cout << "  L3 Normal Dataset Comprehensive Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Data file: " << data_file << std::endl;
    std::cout << "Output dir: " << output_dir << std::endl;

    // Read data
    std::ifstream file(data_file, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file: " << data_file << std::endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t n = file_size / sizeof(uint64_t);
    std::vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    double data_mb = n * sizeof(uint64_t) / (1024.0 * 1024.0);
    std::cout << "Elements: " << n << std::endl;
    std::cout << "Size: " << std::fixed << std::setprecision(2) << data_mb << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    // Create output directory
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());

    // Run all tests
    std::vector<PartitionStrategy> strategies = {
        PartitionStrategy::FIXED_2048,
        PartitionStrategy::VARIANCE_ADAPTIVE,
        PartitionStrategy::COST_OPTIMAL_BALANCED
    };

    std::vector<DecompressMode> modes = {
        DecompressMode::AUTO,
        DecompressMode::SEQUENTIAL,
        DecompressMode::INTERLEAVED,
        DecompressMode::BRANCHLESS
    };

    std::vector<TestResult> results;
    int total_tests = strategies.size() * modes.size();
    int test_num = 0;

    for (auto strategy : strategies) {
        for (auto mode : modes) {
            test_num++;
            std::cout << "[" << test_num << "/" << total_tests << "] "
                      << strategyToString(strategy) << " + " << decompModeToString(mode) << "... " << std::flush;

            auto result = runTest<uint64_t>(data, strategy, mode);
            results.push_back(result);

            if (result.decompress_correct && result.random_access_correct) {
                std::cout << "OK (ratio=" << std::fixed << std::setprecision(2) << result.compression_ratio
                          << "x, " << std::setprecision(2) << result.decompress_throughput_gbps << " GB/s)" << std::endl;
            } else {
                std::cout << "FAIL" << std::endl;
            }
        }
    }

    // Generate report
    std::cout << std::endl << "Generating report..." << std::endl;

    std::ofstream report(output_dir + "/test_report.md");
    report << "# L3 Normal Dataset Test Report\n\n";
    report << "**Dataset**: " << data_file << "\n";
    report << "**Elements**: " << n << "\n";
    report << "**Original Size**: " << std::fixed << std::setprecision(2) << data_mb << " MB\n";
    report << "**Generated**: " << __DATE__ << " " << __TIME__ << "\n\n";

    report << "## Test Results\n\n";
    report << "| Strategy | Mode | Partitions | Avg Size | Delta Bits | Ratio | Decomp Time (ms) | Throughput (GB/s) | Decomp | RA |\n";
    report << "|----------|------|------------|----------|------------|-------|------------------|-------------------|--------|----|\n";

    int decomp_pass = 0, ra_pass = 0;
    double best_ratio = 0, best_throughput = 0;
    std::string best_ratio_config, best_throughput_config;

    for (const auto& r : results) {
        report << "| " << r.strategy << " | " << r.mode << " | " << r.num_partitions << " | "
               << std::fixed << std::setprecision(0) << r.avg_partition_size << " | "
               << std::setprecision(1) << r.avg_delta_bits << " | "
               << std::setprecision(2) << r.compression_ratio << "x | "
               << std::setprecision(2) << r.decompress_time_ms << " | "
               << std::setprecision(2) << r.decompress_throughput_gbps << " | "
               << (r.decompress_correct ? "PASS" : "FAIL") << " | "
               << (r.random_access_correct ? "PASS" : "FAIL") << " |\n";

        if (r.decompress_correct) decomp_pass++;
        if (r.random_access_correct) ra_pass++;

        if (r.compression_ratio > best_ratio) {
            best_ratio = r.compression_ratio;
            best_ratio_config = r.strategy + " + " + r.mode;
        }
        if (r.decompress_throughput_gbps > best_throughput && r.decompress_correct) {
            best_throughput = r.decompress_throughput_gbps;
            best_throughput_config = r.strategy + " + " + r.mode;
        }
    }

    report << "\n## Summary\n\n";
    report << "| Metric | Value |\n";
    report << "|--------|-------|\n";
    report << "| Total Tests | " << results.size() << " |\n";
    report << "| Decompression Passed | " << decomp_pass << " / " << results.size() << " |\n";
    report << "| Random Access Passed | " << ra_pass << " / " << results.size() << " |\n";
    report << "| Best Compression Ratio | " << std::fixed << std::setprecision(2) << best_ratio << "x (" << best_ratio_config << ") |\n";
    report << "| Best Throughput | " << best_throughput << " GB/s (" << best_throughput_config << ") |\n";

    report << "\n## Model Distribution\n\n";
    report << "| Strategy | Mode | CONSTANT | LINEAR | POLY2 | POLY3 | FOR_BP |\n";
    report << "|----------|------|----------|--------|-------|-------|--------|\n";

    for (const auto& r : results) {
        int total = r.constant_count + r.linear_count + r.poly2_count + r.poly3_count + r.for_bp_count;
        report << "| " << r.strategy << " | " << r.mode << " | "
               << std::fixed << std::setprecision(1) << (100.0 * r.constant_count / total) << "% | "
               << (100.0 * r.linear_count / total) << "% | "
               << (100.0 * r.poly2_count / total) << "% | "
               << (100.0 * r.poly3_count / total) << "% | "
               << (100.0 * r.for_bp_count / total) << "% |\n";
    }

    report.close();

    // CSV output
    std::ofstream csv(output_dir + "/detailed_results.csv");
    csv << "strategy,mode,num_partitions,avg_partition_size,avg_delta_bits,compressed_size_mb,compression_ratio,"
        << "compress_time_ms,decompress_time_ms,decompress_throughput_gbps,random_access_time_ms,"
        << "decompress_correct,random_access_correct,constant_pct,linear_pct,poly2_pct,poly3_pct,for_bp_pct\n";

    for (const auto& r : results) {
        int total = r.constant_count + r.linear_count + r.poly2_count + r.poly3_count + r.for_bp_count;
        csv << r.strategy << "," << r.mode << "," << r.num_partitions << ","
            << r.avg_partition_size << "," << r.avg_delta_bits << "," << r.compressed_size_mb << ","
            << r.compression_ratio << "," << r.compress_time_ms << "," << r.decompress_time_ms << ","
            << r.decompress_throughput_gbps << "," << r.random_access_time_ms << ","
            << (r.decompress_correct ? "true" : "false") << "," << (r.random_access_correct ? "true" : "false") << ","
            << (100.0 * r.constant_count / total) << "," << (100.0 * r.linear_count / total) << ","
            << (100.0 * r.poly2_count / total) << "," << (100.0 * r.poly3_count / total) << ","
            << (100.0 * r.for_bp_count / total) << "\n";
    }
    csv.close();

    std::cout << "Report saved to: " << output_dir << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << decomp_pass << "/" << results.size() << " decompression tests passed" << std::endl;
    std::cout << "         " << ra_pass << "/" << results.size() << " random access tests passed" << std::endl;

    return (decomp_pass == results.size() && ra_pass == results.size()) ? 0 : 1;
}
