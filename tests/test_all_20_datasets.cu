/**
 * Test V3 (Cost-Optimal) Compression + Vertical Branchless Decompression
 * Extended to all 20 SOSD datasets
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// V3 Cost-Optimal encoder (defines createPartitionsCostOptimal)
#include "../src/kernels/compression/encoder_cost_optimal.cu"

// Undefine macros from cost_optimal to avoid conflicts
#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// Vertical encoder (for packing kernel)
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
    std::string type;  // "uint32" or "uint64"
};

struct TestResult {
    int id;
    std::string name;
    std::string type;
    size_t num_elements;
    double original_size_mb;
    double compressed_size_mb;
    double compression_ratio;
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double compress_kernel_ms;
    double compress_throughput_gbps;
    double decompress_kernel_ms;
    double decompress_throughput_gbps;
    bool correctness;
    std::string error;
};

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for 8-byte header (element count)
    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));

    size_t data_bytes = file_size - sizeof(uint64_t);
    size_t expected_with_header = data_bytes / sizeof(T);

    std::vector<T> data;
    if (header_count == expected_with_header) {
        // Has header
        data.resize(header_count);
        file.read(reinterpret_cast<char*>(data.data()), header_count * sizeof(T));
    } else {
        // No header, read entire file
        file.seekg(0, std::ios::beg);
        size_t num_elements = file_size / sizeof(T);
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }

    return data;
}

template<typename T>
TestResult runTest(const DatasetInfo& dataset, const std::string& data_dir) {
    TestResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.type = dataset.type;
    result.correctness = false;
    result.error = "";

    std::string path = data_dir + "/" + dataset.filename;
    std::cout << "\n=== [" << dataset.id << "/20] Testing: " << dataset.name << " ===" << std::endl;

    // Load data
    std::vector<T> data = loadBinaryData<T>(path);
    if (data.empty()) {
        result.error = "File not found or empty";
        std::cout << "SKIP: " << result.error << std::endl;
        return result;
    }

    result.num_elements = data.size();
    result.original_size_mb = static_cast<double>(data.size() * sizeof(T)) / (1024.0 * 1024.0);
    double data_bytes = data.size() * sizeof(T);

    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Original size: " << std::fixed << std::setprecision(2)
              << result.original_size_mb << " MB" << std::endl;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // Run V3 partitioning
    CostOptimalConfig config = CostOptimalConfig::balanced();
    int num_partitions = 0;

    CUDA_CHECK(cudaEventRecord(start));
    auto partitions = createPartitionsCostOptimal<T>(data, config, &num_partitions, 0);

    if (partitions.empty()) {
        result.error = "Partitioning failed";
        std::cout << "FAIL: " << result.error << std::endl;
        cudaFree(d_data);
        return result;
    }

    // Build compressed structure
    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    // Allocate device arrays
    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t)));

    // Prepare host arrays
    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits = 0;
    double total_delta_bits = 0;

    // Recompute delta_bits to ensure consistency with Vertical rounding
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;

        int start_idx = h_start[p];
        int end_idx = h_end[p];
        int n = end_idx - start_idx;

        // For MODEL_DIRECT_COPY, use original delta_bits
        if (partitions[p].model_type == MODEL_DIRECT_COPY) {
            h_delta_bits[p] = partitions[p].delta_bits;
            h_bit_offsets[p] = total_bits;
            h_error_bounds[p] = 0;
            for (int j = 0; j < 4; j++) {
                h_model_params[p * 4 + j] = partitions[p].model_params[j];
            }
            total_bits += static_cast<int64_t>(n) * h_delta_bits[p];
            total_delta_bits += h_delta_bits[p];
            continue;
        }

        double theta0 = partitions[p].model_params[0];
        double theta1 = partitions[p].model_params[1];

        // Recompute max_error with llrint
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

        // Compute bits needed
        int bits = 0;
        if (max_error > 0) {
            bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
        }
        h_delta_bits[p] = bits;

        h_bit_offsets[p] = total_bits;
        h_error_bounds[p] = max_error;

        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }

        total_bits += static_cast<int64_t>(n) * h_delta_bits[p];
        total_delta_bits += h_delta_bits[p];
    }

    // Copy to device
    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(),
                          num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(),
                          num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(),
                          num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(),
                          num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(),
                          num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(),
                          num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(),
                          num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    // Allocate and pack delta array
    int64_t delta_words = (total_bits + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // Pack deltas
    int blocks = (data.size() + 255) / 256;
    blocks = std::min(blocks, 65535);

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

    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, stop));
    result.compress_kernel_ms = compress_ms;
    result.compress_throughput_gbps = (data_bytes / 1e9) / (compress_ms / 1000.0);

    // Calculate compression stats
    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;
    result.avg_delta_bits = total_delta_bits / num_partitions;

    double metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 +
                                              sizeof(int64_t) * 2);
    double delta_bytes = delta_words * sizeof(uint32_t);
    result.compressed_size_mb = (metadata_bytes + delta_bytes) / (1024.0 * 1024.0);
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(3)
              << result.compression_ratio << "x" << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    std::cout << "Avg partition size: " << std::fixed << std::setprecision(1)
              << result.avg_partition_size << std::endl;
    std::cout << "Avg delta bits: " << std::fixed << std::setprecision(2)
              << result.avg_delta_bits << std::endl;

    // Decompression
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

    result.decompress_kernel_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (result.decompress_kernel_ms / 1000.0);

    std::cout << "Decompress: " << std::fixed << std::setprecision(2)
              << result.decompress_kernel_ms << " ms ("
              << result.decompress_throughput_gbps << " GB/s)" << std::endl;

    // Verify correctness
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    result.correctness = true;
    int error_count = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.correctness = false;
            if (error_count < 3) {
                std::cerr << "ERROR at index " << i << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
            error_count++;
        }
    }
    if (error_count > 0) {
        result.error = std::to_string(error_count) + " mismatches";
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

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <sosd_data_dir>" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "==========================================" << std::endl;
    std::cout << "L3 Comprehensive Test - All 20 Datasets" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
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
        {14, "cosmos", "14-cosmos_int32.bin", "uint32"},  // treat int32 as uint32
        {15, "polylog_10M", "15-polylog_10M_uint64.bin", "uint64"},
        {16, "exp_200M", "16-exp_200M_uint64.bin", "uint64"},
        {17, "poly_200M", "17-poly_200M_uint64.bin", "uint64"},
        {18, "site_250k", "18-site_250k_uint32.bin", "uint32"},
        {19, "weight_25k", "19-weight_25k_uint32.bin", "uint32"},
        {20, "adult_30k", "20-adult_30k_uint32.bin", "uint32"},
    };

    std::vector<TestResult> results;

    for (const auto& ds : datasets) {
        TestResult result;
        if (ds.type == "uint32") {
            result = runTest<uint32_t>(ds, data_dir);
        } else {
            result = runTest<uint64_t>(ds, data_dir);
        }
        results.push_back(result);
    }

    // Print comprehensive summary
    std::cout << "\n==========================================" << std::endl;
    std::cout << "COMPREHENSIVE SUMMARY" << std::endl;
    std::cout << "==========================================" << std::endl;

    std::cout << "\n| # | Dataset | Type | Elements | Size(MB) | Parts | AvgSize | Bits | Ratio | Decomp GB/s | Status |" << std::endl;
    std::cout << "|---|---------|------|----------|----------|-------|---------|------|-------|-------------|--------|" << std::endl;

    int pass_count = 0;
    double total_ratio = 0, total_throughput = 0;
    int valid_count = 0;

    for (const auto& r : results) {
        if (r.error.empty() || r.error.find("mismatch") != std::string::npos) {
            std::cout << "| " << std::setw(2) << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::setw(6) << r.type
                      << " | " << std::right << std::setw(10) << r.num_elements
                      << " | " << std::fixed << std::setprecision(1) << std::setw(8) << r.original_size_mb
                      << " | " << std::setw(5) << r.num_partitions
                      << " | " << std::setprecision(0) << std::setw(7) << r.avg_partition_size
                      << " | " << std::setprecision(1) << std::setw(4) << r.avg_delta_bits
                      << " | " << std::setprecision(2) << std::setw(5) << r.compression_ratio << "x"
                      << " | " << std::setprecision(1) << std::setw(11) << r.decompress_throughput_gbps
                      << " | " << (r.correctness ? "PASS  " : "FAIL  ") << " |" << std::endl;

            if (r.correctness) {
                pass_count++;
                total_ratio += r.compression_ratio;
                total_throughput += r.decompress_throughput_gbps;
                valid_count++;
            }
        } else {
            std::cout << "| " << std::setw(2) << r.id
                      << " | " << std::left << std::setw(12) << r.name
                      << " | " << std::setw(6) << r.type
                      << " | " << std::right << std::setw(10) << "-"
                      << " | " << std::setw(8) << "-"
                      << " | " << std::setw(5) << "-"
                      << " | " << std::setw(7) << "-"
                      << " | " << std::setw(4) << "-"
                      << " | " << std::setw(6) << "-"
                      << " | " << std::setw(11) << "-"
                      << " | SKIP   |" << std::endl;
        }
    }

    std::cout << "\n==========================================" << std::endl;
    std::cout << "STATISTICS" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Total datasets: 20" << std::endl;
    std::cout << "Passed: " << pass_count << std::endl;
    std::cout << "Failed/Skipped: " << (20 - pass_count) << std::endl;
    if (valid_count > 0) {
        std::cout << "Avg Compression Ratio: " << std::fixed << std::setprecision(2)
                  << (total_ratio / valid_count) << "x" << std::endl;
        std::cout << "Avg Decompression Throughput: " << std::fixed << std::setprecision(1)
                  << (total_throughput / valid_count) << " GB/s" << std::endl;
    }

    return 0;
}
