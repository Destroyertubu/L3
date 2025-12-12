/**
 * Test V3 (Cost-Optimal) Compression + Vertical Branchless Decompression
 *
 * This test combines:
 * 1. V3 Cost-Optimal partitioning (best compression ratio)
 * 2. Vertical branchless per-partition decompression (best throughput)
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
            exit(1); \
        } \
    } while(0)

struct DatasetInfo {
    std::string name;
    std::string path;
    std::string type;  // "uint32" or "uint64"
};

struct TestResult {
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
};

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << path << std::endl;
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    return data;
}

template<typename T>
TestResult runTest(const DatasetInfo& dataset) {
    TestResult result;
    result.name = dataset.name;
    result.type = dataset.type;

    std::cout << "\n=== Testing: " << dataset.name << " ===" << std::endl;

    // Load data
    std::vector<T> data = loadBinaryData<T>(dataset.path);
    if (data.empty()) {
        result.correctness = false;
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

    // ========== V3 Cost-Optimal Compression ==========
    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // Run V3 partitioning
    CostOptimalConfig config = CostOptimalConfig::balanced();
    int num_partitions = 0;

    CUDA_CHECK(cudaEventRecord(start));
    auto partitions = createPartitionsCostOptimal<T>(data, config, &num_partitions, 0);

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
    // EXCEPT for MODEL_DIRECT_COPY partitions which store raw values
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;

        int start = h_start[p];
        int end = h_end[p];
        int n = end - start;

        // For MODEL_DIRECT_COPY, use original delta_bits (64 for uint64, 32 for uint32)
        if (partitions[p].model_type == MODEL_DIRECT_COPY) {
            h_delta_bits[p] = partitions[p].delta_bits;  // Should be sizeof(T)*8
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

        // Recompute max_error with llrint (consistent with __double2ll_rn)
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

        int partition_size = n;
        total_bits += static_cast<int64_t>(partition_size) * h_delta_bits[p];
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

    // Pack deltas using Vertical branchless packing kernel
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

    // Compressed size = metadata + delta array
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
    std::cout << "Compress time: " << std::fixed << std::setprecision(2)
              << compress_ms << " ms (" << result.compress_throughput_gbps << " GB/s)" << std::endl;

    // ========== Vertical Branchless Decompression ==========
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

    std::cout << "Decompress kernel: " << std::fixed << std::setprecision(2)
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
            if (error_count < 5) {
                std::cerr << "ERROR at index " << i << ": expected " << data[i]
                          << ", got " << decoded[i] << std::endl;
            }
            error_count++;
        }
    }
    if (error_count > 5) {
        std::cerr << "... and " << (error_count - 5) << " more errors" << std::endl;
    }
    std::cout << "Correctness: " << (result.correctness ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(compressed.d_start_indices));
    CUDA_CHECK(cudaFree(compressed.d_end_indices));
    CUDA_CHECK(cudaFree(compressed.d_model_types));
    CUDA_CHECK(cudaFree(compressed.d_model_params));
    CUDA_CHECK(cudaFree(compressed.d_delta_bits));
    CUDA_CHECK(cudaFree(compressed.d_delta_array_bit_offsets));
    CUDA_CHECK(cudaFree(compressed.d_error_bounds));
    CUDA_CHECK(cudaFree(compressed.d_sequential_deltas));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

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
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

    // Define datasets
    std::vector<DatasetInfo> datasets = {
        {"linear_200M", data_dir + "/1-linear_200M_uint64.bin", "uint64"},
        {"normal_200M", data_dir + "/2-normal_200M_uint64.bin", "uint64"},
        {"poisson_87M", data_dir + "/3-poisson_87M_uint64.bin", "uint64"},
        {"ml", data_dir + "/4-ml_uint64.bin", "uint64"},
        {"books_200M", data_dir + "/5-books_200M_uint32.bin", "uint32"},
        {"fb_200M", data_dir + "/6-fb_200M_uint64.bin", "uint64"},
        {"wiki_200M", data_dir + "/7-wiki_200M_uint64.bin", "uint64"},
        {"osm_800M", data_dir + "/8-osm_cellids_800M_uint64.bin", "uint64"},
        {"movieid", data_dir + "/9-movieid_uint32.bin", "uint32"},
        {"house_price", data_dir + "/10-house_price_uint64.bin", "uint64"},
        {"planet", data_dir + "/11-planet_uint64.bin", "uint64"},
        {"libio", data_dir + "/12-libio.bin", "uint64"},
    };

    std::vector<TestResult> results;

    for (const auto& ds : datasets) {
        TestResult result;
        if (ds.type == "uint32") {
            result = runTest<uint32_t>(ds);
        } else {
            result = runTest<uint64_t>(ds);
        }
        results.push_back(result);
    }

    // Print summary table
    std::cout << "\n========================================" << std::endl;
    std::cout << "V3 Cost-Optimal + Vertical Summary" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << std::left << std::setw(15) << "Dataset"
              << std::setw(12) << "Ratio"
              << std::setw(15) << "Decomp GB/s"
              << std::setw(10) << "Status" << std::endl;
    std::cout << std::string(52, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(15) << r.name
                  << std::fixed << std::setprecision(2) << std::setw(12) << r.compression_ratio
                  << std::fixed << std::setprecision(2) << std::setw(15) << r.decompress_throughput_gbps
                  << std::setw(10) << (r.correctness ? "PASS" : "FAIL") << std::endl;
    }

    // Print detailed results for report
    std::cout << "\n========================================" << std::endl;
    std::cout << "Detailed Results (for report)" << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << "\n| Dataset | Partitions | Avg Size | Ratio | Decomp GB/s |" << std::endl;
    std::cout << "|---------|------------|----------|-------|-------------|" << std::endl;
    for (const auto& r : results) {
        std::cout << "| " << std::left << std::setw(14) << r.name
                  << "| " << std::setw(10) << r.num_partitions
                  << " | " << std::fixed << std::setprecision(1) << std::setw(8) << r.avg_partition_size
                  << " | " << std::setprecision(2) << std::setw(5) << r.compression_ratio
                  << " | " << std::setprecision(2) << std::setw(11) << r.decompress_throughput_gbps
                  << " |" << std::endl;
    }

    return 0;
}
