#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include "l3_opt.h"

// CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple compressed data generator for benchmarking
template<typename T>
class SyntheticCompressedData {
public:
    std::vector<int32_t> h_start_indices;
    std::vector<int32_t> h_end_indices;
    std::vector<int32_t> h_model_types;
    std::vector<double> h_model_params;
    std::vector<int32_t> h_delta_bits;
    std::vector<int64_t> h_bit_offsets;
    std::vector<uint32_t> h_delta_array;

    int32_t* d_start_indices = nullptr;
    int32_t* d_end_indices = nullptr;
    int32_t* d_model_types = nullptr;
    double* d_model_params = nullptr;
    int32_t* d_delta_bits = nullptr;
    int64_t* d_bit_offsets = nullptr;
    uint32_t* d_delta_array = nullptr;

    int num_partitions;
    int total_elements;

    SyntheticCompressedData(int num_elems, int num_parts, int bitwidth, int model_type = 1) {
        total_elements = num_elems;
        num_partitions = num_parts;

        int elements_per_partition = (num_elems + num_parts - 1) / num_parts;

        int64_t bit_offset = 0;

        for (int p = 0; p < num_parts; ++p) {
            int start = p * elements_per_partition;
            int end = std::min(start + elements_per_partition, num_elems);

            h_start_indices.push_back(start);
            h_end_indices.push_back(end);
            h_model_types.push_back(model_type);
            h_delta_bits.push_back(bitwidth);
            h_bit_offsets.push_back(bit_offset);

            // Model params (theta0, theta1, theta2, theta3)
            h_model_params.push_back(1000.0);  // theta0
            h_model_params.push_back(10.0);    // theta1
            h_model_params.push_back(0.0);     // theta2
            h_model_params.push_back(0.0);     // theta3

            // Update bit offset
            int partition_size = end - start;
            bit_offset += static_cast<int64_t>(partition_size) * bitwidth;
        }

        // Generate random delta array
        int64_t total_bits = bit_offset;
        int64_t total_words = (total_bits + 31) / 32;
        h_delta_array.resize(total_words + 32);  // Extra padding

        std::mt19937 rng(42);
        std::uniform_int_distribution<uint32_t> dist;
        for (size_t i = 0; i < h_delta_array.size(); ++i) {
            h_delta_array[i] = dist(rng);
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_start_indices, num_partitions * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_end_indices, num_partitions * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_model_types, num_partitions * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_model_params, num_partitions * 4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_delta_bits, num_partitions * sizeof(int32_t)));
        CUDA_CHECK(cudaMalloc(&d_bit_offsets, num_partitions * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_delta_array, h_delta_array.size() * sizeof(uint32_t)));

        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_start_indices, h_start_indices.data(),
                             num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_end_indices, h_end_indices.data(),
                             num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_model_types, h_model_types.data(),
                             num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_model_params, h_model_params.data(),
                             num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_delta_bits, h_delta_bits.data(),
                             num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bit_offsets, h_bit_offsets.data(),
                             num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_delta_array, h_delta_array.data(),
                             h_delta_array.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    ~SyntheticCompressedData() {
        if (d_start_indices) cudaFree(d_start_indices);
        if (d_end_indices) cudaFree(d_end_indices);
        if (d_model_types) cudaFree(d_model_types);
        if (d_model_params) cudaFree(d_model_params);
        if (d_delta_bits) cudaFree(d_delta_bits);
        if (d_bit_offsets) cudaFree(d_bit_offsets);
        if (d_delta_array) cudaFree(d_delta_array);
    }
};

// Benchmark configuration
struct BenchConfig {
    int elements;
    int partitions;
    int bitwidth;
    int model_type;
    std::string name;
};

int main(int argc, char** argv) {
    std::cout << "L3 Optimized Decompression Benchmark" << std::endl;
    std::cout << "=======================================" << std::endl;

    // Print GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << std::endl;

    // Benchmark configurations
    std::vector<BenchConfig> configs = {
        {1024 * 1024,     256,  8, 1, "1M_8bit"},
        {4 * 1024 * 1024, 512,  8, 1, "4M_8bit"},
        {16 * 1024 * 1024, 1024, 8, 1, "16M_8bit"},
        {1024 * 1024,     256, 16, 1, "1M_16bit"},
        {4 * 1024 * 1024, 512, 16, 1, "4M_16bit"},
        {1024 * 1024,     256,  4, 1, "1M_4bit"},
        {4 * 1024 * 1024, 512,  4, 1, "4M_4bit"},
    };

    const std::string csv_file = "results/bench_log.csv";
    bool first_result = true;

    for (const auto& config : configs) {
        std::cout << "Running: " << config.name << std::endl;

        // Generate synthetic data
        using T = int32_t;
        SyntheticCompressedData<T> synth_data(
            config.elements,
            config.partitions,
            config.bitwidth,
            config.model_type
        );

        // Allocate output
        T* d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_output, config.elements * sizeof(T)));

        // Create compressed data structure for kernel
        CompressedDataOpt<T> compressed_data;
        compressed_data.d_start_indices = synth_data.d_start_indices;
        compressed_data.d_end_indices = synth_data.d_end_indices;
        compressed_data.d_model_types = synth_data.d_model_types;
        compressed_data.d_model_params = synth_data.d_model_params;
        compressed_data.d_delta_bits = synth_data.d_delta_bits;
        compressed_data.d_delta_array_bit_offsets = synth_data.d_bit_offsets;
        compressed_data.delta_array = synth_data.d_delta_array;
        compressed_data.d_plain_deltas = nullptr;  // Test bit-packed path
        compressed_data.num_partitions = config.partitions;
        compressed_data.total_elements = config.elements;

        // Warmup
        const int warmup_iters = 5;
        for (int i = 0; i < warmup_iters; ++i) {
            launchDecompressOptimized<T>(compressed_data, d_output, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed runs
        const int timed_iters = 20;
        std::vector<float> times;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < timed_iters; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            launchDecompressOptimized<T>(compressed_data, d_output, 0);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            times.push_back(ms);
        }

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // Compute median time
        std::sort(times.begin(), times.end());
        float median_time = times[timed_iters / 2];

        // Calculate throughput
        double bytes = static_cast<double>(config.elements) * sizeof(T);
        double gbps = (bytes / (median_time * 1e-3)) / 1e9;

        // Create result
        BenchResult result;
        result.config_name = config.name;
        result.elements = config.elements;
        result.bitwidth = config.bitwidth;
        result.num_partitions = config.partitions;
        result.kernel_time_ms = median_time;
        result.throughput_gbps = gbps;
        result.warmup_iters = warmup_iters;
        result.timed_iters = timed_iters;

        printBenchResult(result);
        writeBenchResultToCSV(csv_file, result, first_result);
        first_result = false;

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));
    }

    std::cout << "Benchmark complete. Results written to: " << csv_file << std::endl;

    return 0;
}
