/**
 * L3 Comprehensive Benchmark - All 20 SOSD Datasets
 * Detailed metrics including model selection statistics
 *
 * Collects:
 * - Compression ratio and compressed size
 * - Partition statistics (count, avg size, avg delta bits)
 * - Model selection distribution (Linear, Poly2, Poly3, DirectCopy)
 * - Compression time breakdown (partitioning + packing)
 * - Decompression time and throughput
 * - Correctness verification
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// V3 Cost-Optimal encoder
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// Vertical encoder
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return result; \
        } \
    } while(0)

struct DatasetInfo {
    int id;
    std::string name;
    std::string filename;
    std::string type;
};

struct ModelStats {
    int constant_count = 0;
    int linear_count = 0;
    int poly2_count = 0;
    int poly3_count = 0;
    int direct_copy_count = 0;

    double constant_pct() const { return total() > 0 ? 100.0 * constant_count / total() : 0; }
    double linear_pct() const { return total() > 0 ? 100.0 * linear_count / total() : 0; }
    double poly2_pct() const { return total() > 0 ? 100.0 * poly2_count / total() : 0; }
    double poly3_pct() const { return total() > 0 ? 100.0 * poly3_count / total() : 0; }
    double direct_copy_pct() const { return total() > 0 ? 100.0 * direct_copy_count / total() : 0; }
    int total() const { return constant_count + linear_count + poly2_count + poly3_count + direct_copy_count; }
};

struct DetailedResult {
    int id;
    std::string name;
    std::string type;

    // Size metrics
    size_t num_elements;
    double original_size_mb;
    double compressed_size_mb;
    double metadata_size_mb;
    double delta_array_size_mb;
    double compression_ratio;

    // Partition metrics
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double min_delta_bits;
    double max_delta_bits;

    // Model selection
    ModelStats model_stats;

    // Timing (milliseconds)
    double partition_time_ms;
    double pack_time_ms;
    double total_compress_time_ms;
    double decompress_time_ms;

    // Throughput (GB/s)
    double compress_throughput_gbps;
    double decompress_throughput_gbps;

    // Correctness
    bool correctness;
    std::string error_msg;
};

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

template<typename T>
DetailedResult runDetailedTest(const DatasetInfo& dataset, const std::string& data_dir) {
    DetailedResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.type = dataset.type;
    result.correctness = false;
    result.error_msg = "";

    std::string path = data_dir + "/" + dataset.filename;
    std::cout << "\n=== [" << dataset.id << "/20] " << dataset.name << " ===" << std::endl;

    // Load data
    std::vector<T> data = loadBinaryData<T>(path);
    if (data.empty()) {
        result.error_msg = "File not found";
        std::cout << "SKIP: " << result.error_msg << std::endl;
        return result;
    }

    result.num_elements = data.size();
    result.original_size_mb = static_cast<double>(data.size() * sizeof(T)) / (1024.0 * 1024.0);
    double data_bytes = data.size() * sizeof(T);

    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Original size: " << std::fixed << std::setprecision(2) << result.original_size_mb << " MB" << std::endl;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // ========== Partitioning Phase ==========
    CostOptimalConfig config = CostOptimalConfig::balanced();
    int num_partitions = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    auto partitions = createPartitionsCostOptimal<T>(data, config, &num_partitions, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float partition_ms;
    CUDA_CHECK(cudaEventElapsedTime(&partition_ms, start, stop));
    result.partition_time_ms = partition_ms;

    if (partitions.empty()) {
        result.error_msg = "Partitioning failed";
        std::cout << "FAIL: " << result.error_msg << std::endl;
        cudaFree(d_data);
        return result;
    }

    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;

    // Analyze model selection and delta bits
    double total_delta_bits = 0;
    result.min_delta_bits = 65;
    result.max_delta_bits = 0;

    for (int p = 0; p < num_partitions; p++) {
        total_delta_bits += partitions[p].delta_bits;
        if (partitions[p].delta_bits < result.min_delta_bits) result.min_delta_bits = partitions[p].delta_bits;
        if (partitions[p].delta_bits > result.max_delta_bits) result.max_delta_bits = partitions[p].delta_bits;

        switch (partitions[p].model_type) {
            case MODEL_CONSTANT: result.model_stats.constant_count++; break;
            case MODEL_LINEAR: result.model_stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.model_stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.model_stats.poly3_count++; break;
            case MODEL_FOR_BITPACK: // == MODEL_DIRECT_COPY
                result.model_stats.direct_copy_count++; break;
            default: break;
        }
    }
    result.avg_delta_bits = total_delta_bits / num_partitions;

    // Recompute delta bits for Vertical consistency
    for (int p = 0; p < num_partitions; p++) {
        if (partitions[p].model_type == MODEL_DIRECT_COPY || partitions[p].model_type == MODEL_FOR_BITPACK) continue;

        int start_idx = partitions[p].start_idx;
        int end_idx = partitions[p].end_idx;
        int n = end_idx - start_idx;

        double theta0 = partitions[p].model_params[0];
        double theta1 = partitions[p].model_params[1];

        int64_t max_error = 0;
        for (int i = 0; i < n; i++) {
            double predicted = theta0 + theta1 * i;
            T pred_val = static_cast<T>(std::llrint(predicted));
            int64_t delta;
            if (data[start_idx + i] >= pred_val) {
                delta = static_cast<int64_t>(data[start_idx + i] - pred_val);
            } else {
                delta = -static_cast<int64_t>(pred_val - data[start_idx + i]);
            }
            int64_t abs_delta = (delta < 0) ? -delta : delta;
            if (abs_delta > max_error) max_error = abs_delta;
        }

        int bits = 0;
        if (max_error > 0) {
            bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
        }
        partitions[p].delta_bits = bits;
        partitions[p].error_bound = max_error;
    }

    // Build compressed structure
    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t)));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits = 0;
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_bit_offsets[p] = total_bits;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
        total_bits += static_cast<int64_t>(h_end[p] - h_start[p]) * h_delta_bits[p];
    }

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t delta_words = (total_bits + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // ========== Packing Phase ==========
    int blocks = std::min((int)((data.size() + 255) / 256), 65535);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_delta_array_bit_offsets,
        num_partitions,
        compressed.d_sequential_deltas
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float pack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pack_ms, start, stop));
    result.pack_time_ms = pack_ms;

    result.total_compress_time_ms = result.partition_time_ms + result.pack_time_ms;
    result.compress_throughput_gbps = (data_bytes / 1e9) / (result.total_compress_time_ms / 1000.0);

    // Calculate compressed size
    result.metadata_size_mb = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2) / (1024.0 * 1024.0);
    result.delta_array_size_mb = delta_words * sizeof(uint32_t) / (1024.0 * 1024.0);
    result.compressed_size_mb = result.metadata_size_mb + result.delta_array_size_mb;
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(3) << result.compression_ratio << "x" << std::endl;
    std::cout << "Partitions: " << num_partitions << " (avg size: " << std::setprecision(1) << result.avg_partition_size << ")" << std::endl;
    std::cout << "Avg delta bits: " << std::setprecision(2) << result.avg_delta_bits << " (min: " << result.min_delta_bits << ", max: " << result.max_delta_bits << ")" << std::endl;
    std::cout << "Model selection: Linear=" << result.model_stats.linear_count
              << " (" << std::setprecision(1) << result.model_stats.linear_pct() << "%)";
    if (result.model_stats.constant_count > 0) std::cout << ", Const=" << result.model_stats.constant_count;
    if (result.model_stats.poly2_count > 0) std::cout << ", Poly2=" << result.model_stats.poly2_count;
    if (result.model_stats.poly3_count > 0) std::cout << ", Poly3=" << result.model_stats.poly3_count;
    if (result.model_stats.direct_copy_count > 0) std::cout << ", DirectCopy=" << result.model_stats.direct_copy_count;
    std::cout << std::endl;
    std::cout << "Compress: " << std::setprecision(2) << result.total_compress_time_ms << " ms (partition: "
              << result.partition_time_ms << " ms, pack: " << result.pack_time_ms << " ms)" << std::endl;

    // ========== Decompression Phase ==========
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    // Warmup
    for (int i = 0; i < 3; i++) {
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    const int NUM_TRIALS = 10;
    float total_decompress_ms = 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float trial_ms;
        CUDA_CHECK(cudaEventElapsedTime(&trial_ms, start, stop));
        total_decompress_ms += trial_ms;
    }

    result.decompress_time_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    std::cout << "Decompress: " << std::setprecision(2) << result.decompress_time_ms << " ms ("
              << result.decompress_throughput_gbps << " GB/s)" << std::endl;

    // Verify correctness
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    result.correctness = true;
    int error_count = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.correctness = false;
            error_count++;
        }
    }
    if (error_count > 0) {
        result.error_msg = std::to_string(error_count) + " mismatches";
    }
    std::cout << "Correctness: " << (result.correctness ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(compressed.d_start_indices);
    cudaFree(compressed.d_end_indices);
    cudaFree(compressed.d_model_types);
    cudaFree(compressed.d_model_params);
    cudaFree(compressed.d_delta_bits);
    cudaFree(compressed.d_delta_array_bit_offsets);
    cudaFree(compressed.d_error_bounds);
    cudaFree(compressed.d_sequential_deltas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

void printDetailedReport(const std::vector<DetailedResult>& results, const std::string& gpu_name) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "                         L3 COMPREHENSIVE BENCHMARK REPORT" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    // Summary statistics
    int pass_count = 0;
    double total_ratio = 0, total_decomp_throughput = 0, total_comp_throughput = 0;
    int valid_count = 0;

    for (const auto& r : results) {
        if (r.correctness) {
            pass_count++;
            total_ratio += r.compression_ratio;
            total_decomp_throughput += r.decompress_throughput_gbps;
            total_comp_throughput += r.compress_throughput_gbps;
            valid_count++;
        }
    }

    std::cout << "\n## Test Environment\n" << std::endl;
    std::cout << "| Item | Value |" << std::endl;
    std::cout << "|------|-------|" << std::endl;
    std::cout << "| GPU | " << gpu_name << " |" << std::endl;
    std::cout << "| Method | V3 (Cost-Optimal) + Vertical Branchless |" << std::endl;
    std::cout << "| Datasets | 20 (SOSD benchmark suite) |" << std::endl;
    std::cout << "| Timing | CUDA Events (kernel-only) |" << std::endl;

    std::cout << "\n## Summary Statistics\n" << std::endl;
    std::cout << "| Metric | Value |" << std::endl;
    std::cout << "|--------|-------|" << std::endl;
    std::cout << "| Total Datasets | 20 |" << std::endl;
    std::cout << "| Passed | " << pass_count << " |" << std::endl;
    std::cout << "| Failed/Skipped | " << (20 - pass_count) << " |" << std::endl;
    if (valid_count > 0) {
        std::cout << "| Avg Compression Ratio | " << std::fixed << std::setprecision(2) << (total_ratio / valid_count) << "x |" << std::endl;
        std::cout << "| Avg Decompress Throughput | " << std::setprecision(1) << (total_decomp_throughput / valid_count) << " GB/s |" << std::endl;
        std::cout << "| Avg Compress Throughput | " << std::setprecision(2) << (total_comp_throughput / valid_count) << " GB/s |" << std::endl;
    }

    // Main results table
    std::cout << "\n## Full Results Table\n" << std::endl;
    std::cout << "| # | Dataset | Type | Elements | Size(MB) | Parts | AvgSize | AvgBits | Ratio | CompMB | Decomp GB/s | Compress GB/s | Status |" << std::endl;
    std::cout << "|---|---------|------|----------|----------|-------|---------|---------|-------|--------|-------------|---------------|--------|" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg == "File not found") {
            std::cout << "| " << r.id << " | " << r.name << " | " << r.type << " | - | - | - | - | - | - | - | - | - | SKIP |" << std::endl;
        } else {
            std::cout << "| " << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << r.type
                      << " | " << std::right << r.num_elements
                      << " | " << std::fixed << std::setprecision(1) << r.original_size_mb
                      << " | " << r.num_partitions
                      << " | " << std::setprecision(0) << r.avg_partition_size
                      << " | " << std::setprecision(1) << r.avg_delta_bits
                      << " | " << std::setprecision(2) << r.compression_ratio << "x"
                      << " | " << std::setprecision(2) << r.compressed_size_mb
                      << " | " << std::setprecision(1) << r.decompress_throughput_gbps
                      << " | " << std::setprecision(2) << r.compress_throughput_gbps
                      << " | " << (r.correctness ? "PASS" : "FAIL") << " |" << std::endl;
        }
    }

    // Timing breakdown
    std::cout << "\n## Timing Breakdown (ms)\n" << std::endl;
    std::cout << "| # | Dataset | Partition | Pack | Total Compress | Decompress |" << std::endl;
    std::cout << "|---|---------|-----------|------|----------------|------------|" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg != "File not found") {
            std::cout << "| " << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::right << std::fixed << std::setprecision(2) << r.partition_time_ms
                      << " | " << r.pack_time_ms
                      << " | " << r.total_compress_time_ms
                      << " | " << r.decompress_time_ms << " |" << std::endl;
        }
    }

    // Model selection statistics
    std::cout << "\n## Model Selection Distribution\n" << std::endl;
    std::cout << "| # | Dataset | Total | Linear | Linear% | Const | Poly2 | Poly3 | DirectCopy |" << std::endl;
    std::cout << "|---|---------|-------|--------|---------|-------|-------|-------|------------|" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg != "File not found") {
            std::cout << "| " << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::right << r.model_stats.total()
                      << " | " << r.model_stats.linear_count
                      << " | " << std::fixed << std::setprecision(1) << r.model_stats.linear_pct() << "%"
                      << " | " << r.model_stats.constant_count
                      << " | " << r.model_stats.poly2_count
                      << " | " << r.model_stats.poly3_count
                      << " | " << r.model_stats.direct_copy_count << " |" << std::endl;
        }
    }

    // Delta bits distribution
    std::cout << "\n## Delta Bits Distribution\n" << std::endl;
    std::cout << "| # | Dataset | Avg Bits | Min Bits | Max Bits |" << std::endl;
    std::cout << "|---|---------|----------|----------|----------|" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg != "File not found") {
            std::cout << "| " << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::right << std::fixed << std::setprecision(2) << r.avg_delta_bits
                      << " | " << (int)r.min_delta_bits
                      << " | " << (int)r.max_delta_bits << " |" << std::endl;
        }
    }

    // Compressed size breakdown
    std::cout << "\n## Compressed Size Breakdown (MB)\n" << std::endl;
    std::cout << "| # | Dataset | Original | Metadata | Delta Array | Total Compressed | Ratio |" << std::endl;
    std::cout << "|---|---------|----------|----------|-------------|------------------|-------|" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg != "File not found") {
            std::cout << "| " << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::right << std::fixed << std::setprecision(2) << r.original_size_mb
                      << " | " << r.metadata_size_mb
                      << " | " << r.delta_array_size_mb
                      << " | " << r.compressed_size_mb
                      << " | " << r.compression_ratio << "x |" << std::endl;
        }
    }

    // Per-dataset detailed results
    std::cout << "\n## Detailed Results Per Dataset\n" << std::endl;

    for (const auto& r : results) {
        if (r.error_msg == "File not found") continue;

        std::cout << "\n### " << r.id << ". " << r.name << "\n" << std::endl;
        std::cout << "| Metric | Value |" << std::endl;
        std::cout << "|--------|-------|" << std::endl;
        std::cout << "| Type | " << r.type << " |" << std::endl;
        std::cout << "| Elements | " << r.num_elements << " |" << std::endl;
        std::cout << "| Original Size | " << std::fixed << std::setprecision(2) << r.original_size_mb << " MB |" << std::endl;
        std::cout << "| Compressed Size | " << r.compressed_size_mb << " MB |" << std::endl;
        std::cout << "| Compression Ratio | " << r.compression_ratio << "x |" << std::endl;
        std::cout << "| Partitions | " << r.num_partitions << " |" << std::endl;
        std::cout << "| Avg Partition Size | " << std::setprecision(1) << r.avg_partition_size << " |" << std::endl;
        std::cout << "| Avg Delta Bits | " << std::setprecision(2) << r.avg_delta_bits << " |" << std::endl;
        std::cout << "| Delta Bits Range | " << (int)r.min_delta_bits << " - " << (int)r.max_delta_bits << " |" << std::endl;
        std::cout << "| Model: Linear | " << r.model_stats.linear_count << " (" << std::setprecision(1) << r.model_stats.linear_pct() << "%) |" << std::endl;
        if (r.model_stats.constant_count > 0) std::cout << "| Model: Constant | " << r.model_stats.constant_count << " (" << r.model_stats.constant_pct() << "%) |" << std::endl;
        if (r.model_stats.poly2_count > 0) std::cout << "| Model: Poly2 | " << r.model_stats.poly2_count << " (" << r.model_stats.poly2_pct() << "%) |" << std::endl;
        if (r.model_stats.poly3_count > 0) std::cout << "| Model: Poly3 | " << r.model_stats.poly3_count << " (" << r.model_stats.poly3_pct() << "%) |" << std::endl;
        if (r.model_stats.direct_copy_count > 0) std::cout << "| Model: DirectCopy | " << r.model_stats.direct_copy_count << " (" << r.model_stats.direct_copy_pct() << "%) |" << std::endl;
        std::cout << "| Partition Time | " << r.partition_time_ms << " ms |" << std::endl;
        std::cout << "| Pack Time | " << r.pack_time_ms << " ms |" << std::endl;
        std::cout << "| Total Compress Time | " << r.total_compress_time_ms << " ms |" << std::endl;
        std::cout << "| Compress Throughput | " << r.compress_throughput_gbps << " GB/s |" << std::endl;
        std::cout << "| Decompress Time | " << r.decompress_time_ms << " ms |" << std::endl;
        std::cout << "| Decompress Throughput | " << std::setprecision(1) << r.decompress_throughput_gbps << " GB/s |" << std::endl;
        std::cout << "| Correctness | " << (r.correctness ? "PASS" : "FAIL") << " |" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sosd_data_dir>" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::string gpu_name = prop.name;

    std::cout << "===========================================" << std::endl;
    std::cout << "L3 Comprehensive Benchmark - All 20 Datasets" << std::endl;
    std::cout << "===========================================" << std::endl;
    std::cout << "GPU: " << gpu_name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    // All 20 datasets
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
        {14, "cosmos", "14-cosmos_int32.bin", "uint32"},
        {15, "polylog_10M", "15-polylog_10M_uint64.bin", "uint64"},
        {16, "exp_200M", "16-exp_200M_uint64.bin", "uint64"},
        {17, "poly_200M", "17-poly_200M_uint64.bin", "uint64"},
        {18, "site_250k", "18-site_250k_uint32.bin", "uint32"},
        {19, "weight_25k", "19-weight_25k_uint32.bin", "uint32"},
        {20, "adult_30k", "20-adult_30k_uint32.bin", "uint32"},
    };

    std::vector<DetailedResult> results;

    for (const auto& ds : datasets) {
        DetailedResult result;
        if (ds.type == "uint32") {
            result = runDetailedTest<uint32_t>(ds, data_dir);
        } else {
            result = runDetailedTest<uint64_t>(ds, data_dir);
        }
        results.push_back(result);
    }

    // Print detailed report
    printDetailedReport(results, gpu_name);

    return 0;
}
