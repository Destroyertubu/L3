/**
 * Benchmark V1 (Variance) Partitioning + Vertical Decompression
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

// V1 encoder
#include "../src/kernels/compression/encoder_variable_length.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES
#undef MIN_PARTITION_SIZE

// Vertical encoder (packing kernel only)
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

struct DatasetInfo {
    std::string filename;
    std::string name;
    std::string type;
};

const std::vector<DatasetInfo> DATASETS = {
    {"1-linear_200M_uint64.bin", "linear_200M", "uint64"},
    {"2-normal_200M_uint64.bin", "normal_200M", "uint64"},
    {"3-poisson_87M_uint64.bin", "poisson_87M", "uint64"},
    {"4-ml_uint64.bin", "ml", "uint64"},
    {"5-books_200M_uint32.bin", "books_200M", "uint32"},
    {"6-fb_200M_uint64.bin", "fb_200M", "uint64"},
    {"7-wiki_200M_uint64.bin", "wiki_200M", "uint64"},
    {"8-osm_cellids_800M_uint64.bin", "osm_800M", "uint64"},
    {"9-movieid_uint32.bin", "movieid", "uint32"},
    {"10-house_price_uint64.bin", "house_price", "uint64"},
    {"11-planet_uint64.bin", "planet", "uint64"},
    {"12-libio.bin", "libio", "uint64"},
};

template<typename T>
std::vector<T> loadData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

template<typename T>
void runTest(const std::vector<T>& data, const std::string& name) {
    if (data.empty()) {
        std::cout << name << "|SKIP|0|0|0|0|0|SKIP" << std::endl;
        return;
    }

    double data_bytes = data.size() * sizeof(T);
    double original_mb = data_bytes / (1024.0 * 1024.0);

    // Create V1 partitions
    int num_partitions = 0;
    auto partitions = createPartitionsVariableLength<T>(data, 2048, &num_partitions, 0, 1, 5);

    if (partitions.empty()) {
        std::cout << name << "|V1_FAIL|0|0|0|0|0|FAIL" << std::endl;
        return;
    }

    num_partitions = partitions.size();

    // Recompute delta_bits for consistency
    // EXCEPT for MODEL_DIRECT_COPY partitions which store raw values
    for (auto& part : partitions) {
        // For MODEL_DIRECT_COPY, keep original delta_bits
        if (part.model_type == MODEL_DIRECT_COPY) {
            continue;
        }

        int start = part.start_idx;
        int end = part.end_idx;
        int n = end - start;
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

    // Calculate stats
    double avg_partition_size = static_cast<double>(data.size()) / num_partitions;
    double total_bits = 0;
    int64_t total_delta_storage = 0;
    for (const auto& p : partitions) {
        total_bits += p.delta_bits;
        total_delta_storage += static_cast<int64_t>(p.end_idx - p.start_idx) * p.delta_bits;
    }
    double avg_delta_bits = total_bits / num_partitions;

    // Copy to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

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

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t delta_words = (total_bits_storage + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // Time compression
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blocks = std::min((int)((data.size() + 255) / 256), 65535);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices, compressed.d_end_indices,
        compressed.d_model_types, compressed.d_model_params,
        compressed.d_delta_bits, compressed.d_delta_array_bit_offsets,
        num_partitions, compressed.d_sequential_deltas);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float compress_ms;
    CUDA_CHECK(cudaEventElapsedTime(&compress_ms, start, stop));

    // Calculate compressed size
    double metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2);
    double delta_bytes = delta_words * sizeof(uint32_t);
    double compressed_mb = (metadata_bytes + delta_bytes) / (1024.0 * 1024.0);
    double compression_ratio = original_mb / compressed_mb;

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
    float decompress_ms = total_decompress_ms / NUM_TRIALS;
    double decompress_gbps = (data_bytes / 1e9) / (decompress_ms / 1000.0);

    // Verify
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));
    bool correct = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) { correct = false; break; }
    }

    // Output: name|partitions|avg_size|avg_bits|ratio|decomp_gbps|status
    std::cout << name << "|" << num_partitions << "|"
              << std::fixed << std::setprecision(1) << avg_partition_size << "|"
              << std::setprecision(2) << avg_delta_bits << "|"
              << std::setprecision(3) << compression_ratio << "|"
              << std::setprecision(2) << decompress_gbps << "|"
              << (correct ? "PASS" : "FAIL") << std::endl;

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
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <data_dir>" << std::endl;
        return 1;
    }
    std::string data_dir = argv[1];

    std::cout << "# V1 (Variance) + Vertical Results" << std::endl;
    for (const auto& ds : DATASETS) {
        std::string path = data_dir + "/" + ds.filename;
        if (ds.type == "uint32") {
            auto data = loadData<uint32_t>(path);
            runTest<uint32_t>(data, ds.name);
        } else {
            auto data = loadData<uint64_t>(path);
            runTest<uint64_t>(data, ds.name);
        }
    }
    return 0;
}
