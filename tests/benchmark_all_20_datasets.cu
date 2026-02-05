/**
 * Comprehensive L3 Benchmark: All 20 SOSD Datasets
 *
 * Tests all 4 compression methods with Vertical branchless decompression:
 * - Fixed (2048): Fixed-size partitions
 * - V1 (Variance): Variance-aware adaptive partitions
 * - V2 (Cost-Aware): Cost-aware partitions considering metadata overhead
 * - V3 (Cost-Optimal): Delta-bits driven partitioning with parallel merging
 *
 * Date: December 5, 2025
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// V1 Variance encoder
#include "../src/kernels/compression/encoder_variable_length.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// V2 Cost-Aware encoder
#include "../src/kernels/compression/encoder_variable_length_v2.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// V3 Cost-Optimal encoder
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// Vertical encoder (for packing and fixed partitions)
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            return result; \
        } \
    } while(0)

struct DatasetInfo {
    std::string filename;
    std::string name;
    std::string type;  // "uint32" or "uint64"
};

// All 20 SOSD datasets
const std::vector<DatasetInfo> DATASETS = {
    {"1-linear_200M_uint64.bin", "1-linear_200M", "uint64"},
    {"2-normal_200M_uint64.bin", "2-normal_200M", "uint64"},
    {"3-poisson_87M_uint64.bin", "3-poisson_87M", "uint64"},
    {"4-ml_uint64.bin", "4-ml", "uint64"},
    {"5-books_200M_uint32.bin", "5-books_200M", "uint32"},
    {"6-fb_200M_uint64.bin", "6-fb_200M", "uint64"},
    {"7-wiki_200M_uint64.bin", "7-wiki_200M", "uint64"},
    {"8-osm_cellids_800M_uint64.bin", "8-osm_800M", "uint64"},
    {"9-movieid_uint32.bin", "9-movieid", "uint32"},
    {"10-house_price_uint64.bin", "10-house_price", "uint64"},
    {"11-planet_uint64.bin", "11-planet", "uint64"},
    {"12-libio.bin", "12-libio", "uint64"},
    {"13-medicare.bin", "13-medicare", "uint64"},
    {"14-cosmos_int32.bin", "14-cosmos", "int32"},
    {"15-polylog_10M_uint64.bin", "15-polylog_10M", "uint64"},
    {"16-exp_200M_uint64.bin", "16-exp_200M", "uint64"},
    {"17-poly_200M_uint64.bin", "17-poly_200M", "uint64"},
    {"18-site_250k_uint32.bin", "18-site_250k", "uint32"},
    {"19-weight_25k_uint32.bin", "19-weight_25k", "uint32"},
    {"20-adult_30k_uint32.bin", "20-adult_30k", "uint32"},
};

struct BenchmarkResult {
    std::string dataset;
    std::string method;
    size_t num_elements;
    double original_size_mb;
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double compressed_size_mb;
    double compression_ratio;
    double compress_kernel_ms;
    double compress_throughput_gbps;
    double decompress_kernel_ms;
    double decompress_throughput_gbps;
    bool correctness;
    std::string error_msg;
};

template<typename T>
std::vector<T> loadData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for header (8 bytes count)
    uint64_t count = 0;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));

    size_t data_size = file_size - sizeof(uint64_t);
    size_t num_elements = data_size / sizeof(T);

    // If header count doesn't match, assume no header
    if (count != num_elements && count != 0) {
        file.seekg(0, std::ios::beg);
        num_elements = file_size / sizeof(T);
    }

    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    return data;
}

// Recompute delta_bits for consistency with Vertical rounding
template<typename T>
void recomputeDeltaBits(const std::vector<T>& data, std::vector<PartitionInfo>& partitions) {
    for (auto& part : partitions) {
        // Skip direct copy partitions
        if (part.model_type == MODEL_DIRECT_COPY) continue;

        int start = part.start_idx;
        int end = part.end_idx;
        int n = end - start;
        if (n <= 0) continue;

        double theta0 = part.model_params[0];
        double theta1 = part.model_params[1];

        int64_t max_error = 0;
        for (int i = 0; i < n; i++) {
            double predicted = theta0 + theta1 * i;
            T pred_val = static_cast<T>(std::llrint(predicted));
            int64_t delta;
            if (data[start + i] >= pred_val) {
                delta = static_cast<int64_t>(data[start + i] - pred_val);
            } else {
                delta = -static_cast<int64_t>(pred_val - data[start + i]);
            }
            int64_t abs_delta = (delta < 0) ? -delta : delta;
            if (abs_delta > max_error) max_error = abs_delta;
        }

        int bits = 0;
        if (max_error > 0) {
            bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
        }
        part.delta_bits = bits;
        part.error_bound = max_error;
    }
}

template<typename T>
BenchmarkResult runBenchmark(
    const std::vector<T>& data,
    const std::string& dataset_name,
    const std::string& method)
{
    BenchmarkResult result;
    result.dataset = dataset_name;
    result.method = method;
    result.num_elements = data.size();
    result.original_size_mb = static_cast<double>(data.size() * sizeof(T)) / (1024.0 * 1024.0);
    result.correctness = false;
    result.error_msg = "";

    double data_bytes = data.size() * sizeof(T);

    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Create partitions based on method
    std::vector<PartitionInfo> partitions;
    int num_partitions = 0;

    cudaDeviceSynchronize();
    cudaEventRecord(start);

    if (method == "Fixed") {
        partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), 2048);
        Vertical_encoder::computePartitionMetadata(data, partitions);
    }
    else if (method == "V1") {
        VariableLengthConfig cfg;
        cfg.base_partition_size = 2048;
        cfg.min_partition_size = 256;
        partitions = createPartitionsVariableLength<T>(data, cfg, &num_partitions, 0);
    }
    else if (method == "V2") {
        CostAwareConfig cfg;
        cfg.base_partition_size = 2048;
        partitions = createPartitionsCostAware<T>(data, cfg, &num_partitions, 0);
    }
    else if (method == "V3") {
        CostOptimalConfig cfg = CostOptimalConfig::balanced();
        partitions = createPartitionsCostOptimal<T>(data, cfg, &num_partitions, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float partition_ms;
    cudaEventElapsedTime(&partition_ms, start, stop);

    if (partitions.empty()) {
        result.error_msg = "Empty partitions";
        cudaFree(d_data);
        return result;
    }

    num_partitions = partitions.size();
    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;

    // Recompute delta_bits for consistency
    recomputeDeltaBits(data, partitions);

    // Calculate avg delta bits
    double total_bits = 0;
    int64_t total_delta_bits_storage = 0;
    for (const auto& p : partitions) {
        total_bits += p.delta_bits;
        total_delta_bits_storage += static_cast<int64_t>(p.end_idx - p.start_idx) * p.delta_bits;
    }
    result.avg_delta_bits = total_bits / num_partitions;

    // Build compressed structure
    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t));
    cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double));
    cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t));
    cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t));
    cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits_storage = 0;
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_bit_offsets[p] = total_bits_storage;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
        total_bits_storage += static_cast<int64_t>(h_end[p] - h_start[p]) * h_delta_bits[p];
    }

    cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t delta_words = (total_bits_storage + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t));
    cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t));

    // Time packing kernel
    int blocks = std::min((int)((data.size() + 255) / 256), 65535);

    cudaDeviceSynchronize();
    cudaEventRecord(start);
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
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float pack_ms;
    cudaEventElapsedTime(&pack_ms, start, stop);
    result.compress_kernel_ms = partition_ms + pack_ms;
    result.compress_throughput_gbps = (data_bytes / 1e9) / (result.compress_kernel_ms / 1000.0);

    // Calculate compressed size
    double metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2);
    double delta_bytes = delta_words * sizeof(uint32_t);
    result.compressed_size_mb = (metadata_bytes + delta_bytes) / (1024.0 * 1024.0);
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;

    // Decompression
    T* d_output;
    cudaMalloc(&d_output, data_bytes);

    // Warmup
    for (int i = 0; i < 3; i++) {
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        cudaDeviceSynchronize();
    }

    // Timed decompression
    const int NUM_TRIALS = 10;
    float total_decompress_ms = 0;

    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        cudaEventRecord(start);
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_ms;
        cudaEventElapsedTime(&trial_ms, start, stop);
        total_decompress_ms += trial_ms;
    }

    result.decompress_kernel_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (result.decompress_kernel_ms / 1000.0);

    // Verify correctness
    std::vector<T> decoded(data.size());
    cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);

    result.correctness = true;
    int error_count = 0;
    size_t first_error_idx = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            if (error_count == 0) first_error_idx = i;
            error_count++;
            if (error_count >= 10) break;
        }
    }
    if (error_count > 0) {
        result.correctness = false;
        result.error_msg = "Mismatch at idx " + std::to_string(first_error_idx) + " (total: " + std::to_string(error_count) + "+)";
    }

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

int main(int argc, char** argv) {
    std::string base_path = "data/sosd";
    if (argc > 1) base_path = argv[1];

    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;

    std::vector<std::string> methods = {"Fixed", "V1", "V2", "V3"};
    std::vector<BenchmarkResult> all_results;

    for (const auto& ds : DATASETS) {
        std::string path = base_path + "/" + ds.filename;
        std::cout << "\n========== " << ds.name << " ==========" << std::endl;

        for (const auto& method : methods) {
            std::cout << "  " << method << "... " << std::flush;

            BenchmarkResult result;
            if (ds.type == "uint32") {
                auto data = loadData<uint32_t>(path);
                if (data.empty()) {
                    std::cout << "SKIP (file not found)" << std::endl;
                    result.dataset = ds.name;
                    result.method = method;
                    result.error_msg = "File not found";
                    all_results.push_back(result);
                    continue;
                }
                result = runBenchmark<uint32_t>(data, ds.name, method);
            } else if (ds.type == "int32") {
                // Handle int32 as uint32 (reinterpret)
                auto data = loadData<uint32_t>(path);
                if (data.empty()) {
                    std::cout << "SKIP (file not found)" << std::endl;
                    result.dataset = ds.name;
                    result.method = method;
                    result.error_msg = "File not found";
                    all_results.push_back(result);
                    continue;
                }
                result = runBenchmark<uint32_t>(data, ds.name, method);
            } else {
                auto data = loadData<uint64_t>(path);
                if (data.empty()) {
                    std::cout << "SKIP (file not found)" << std::endl;
                    result.dataset = ds.name;
                    result.method = method;
                    result.error_msg = "File not found";
                    all_results.push_back(result);
                    continue;
                }
                result = runBenchmark<uint64_t>(data, ds.name, method);
            }

            std::cout << std::fixed << std::setprecision(2)
                      << result.compression_ratio << "x, "
                      << result.decompress_throughput_gbps << " GB/s, "
                      << (result.correctness ? "PASS" : "FAIL");
            if (!result.correctness && !result.error_msg.empty()) {
                std::cout << " (" << result.error_msg << ")";
            }
            std::cout << std::endl;

            all_results.push_back(result);
        }
    }

    // Print detailed summary
    std::cout << "\n\n" << std::string(120, '=') << std::endl;
    std::cout << "                           L3 COMPREHENSIVE BENCHMARK SUMMARY" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    std::cout << "\n| Dataset        | Method | Elements    | Orig(MB) | Parts  | AvgSize  | AvgBits | Comp(MB) | Ratio  | Decomp GB/s | Status |" << std::endl;
    std::cout << "|----------------|--------|-------------|----------|--------|----------|---------|----------|--------|-------------|--------|" << std::endl;

    for (const auto& r : all_results) {
        if (r.error_msg == "File not found") {
            std::cout << "| " << std::left << std::setw(14) << r.dataset
                      << " | " << std::setw(6) << r.method
                      << " | " << std::right << std::setw(11) << "-"
                      << " | " << std::setw(8) << "-"
                      << " | " << std::setw(6) << "-"
                      << " | " << std::setw(8) << "-"
                      << " | " << std::setw(7) << "-"
                      << " | " << std::setw(8) << "-"
                      << " | " << std::setw(6) << "-"
                      << " | " << std::setw(11) << "-"
                      << " | SKIP   |" << std::endl;
        } else {
            std::cout << "| " << std::left << std::setw(14) << r.dataset
                      << " | " << std::setw(6) << r.method
                      << " | " << std::right << std::setw(11) << r.num_elements
                      << " | " << std::fixed << std::setprecision(2) << std::setw(8) << r.original_size_mb
                      << " | " << std::setw(6) << r.num_partitions
                      << " | " << std::setprecision(1) << std::setw(8) << r.avg_partition_size
                      << " | " << std::setprecision(2) << std::setw(7) << r.avg_delta_bits
                      << " | " << std::setw(8) << r.compressed_size_mb
                      << " | " << std::setprecision(2) << std::setw(6) << r.compression_ratio
                      << " | " << std::setw(11) << r.decompress_throughput_gbps
                      << " | " << (r.correctness ? "PASS  " : "FAIL  ") << " |" << std::endl;
        }
    }

    // Print method comparison
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "                    METHOD COMPARISON (Average across datasets)" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    for (const auto& method : methods) {
        double total_ratio = 0, total_throughput = 0;
        int count = 0, pass_count = 0;
        for (const auto& r : all_results) {
            if (r.method == method && r.error_msg != "File not found" && r.compression_ratio > 0) {
                total_ratio += r.compression_ratio;
                total_throughput += r.decompress_throughput_gbps;
                count++;
                if (r.correctness) pass_count++;
            }
        }
        if (count > 0) {
            std::cout << method << ":" << std::endl;
            std::cout << "  Avg Compression Ratio: " << std::fixed << std::setprecision(2) << (total_ratio / count) << "x" << std::endl;
            std::cout << "  Avg Decompression:     " << std::setprecision(2) << (total_throughput / count) << " GB/s" << std::endl;
            std::cout << "  Correctness:           " << pass_count << "/" << count << " passed" << std::endl;
        }
    }

    return 0;
}
