/**
 * L3 Phase 2.2 Benchmark: All 6 SOSD Datasets
 *
 * Tests the optimized Phase 2.2 bucket decoder on:
 * 1. books_200M_uint32.bin
 * 2. fb_200M_uint64.bin
 * 3. linear_200M_uint32_binary.bin
 * 4. movieid_uint32.bin
 * 5. normal_200M_uint32_binary.bin
 * 6. wiki_200M_uint64.bin
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <map>
#include <chrono>
#include <cstring>
#include "L3_codec.hpp"
#include "L3_format.hpp"

// External function declarations for Phase 2.2
template<typename T>
void decompressL3_Phase2_Bucket(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const uint8_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Dataset information structure
struct DatasetInfo {
    std::string name;
    std::string filepath;
    std::string data_type;  // "uint32" or "uint64"
    size_t file_size_mb;
    size_t num_elements;
};

// Load uint32 binary data
std::vector<uint32_t> loadBinaryUint32(const char* filename, size_t& num_elements) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    num_elements = file_size / sizeof(uint32_t);

    std::vector<uint32_t> data(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint32_t));

    return data;
}

// Load uint64 binary data
std::vector<uint64_t> loadBinaryUint64(const char* filename, size_t& num_elements) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    num_elements = file_size / sizeof(uint64_t);

    std::vector<uint64_t> data(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint64_t));

    return data;
}

// Benchmark template for any data type
template<typename T>
void benchmarkDataset(const std::vector<T>& data, const std::string& dataset_name,
                      const std::string& data_type, std::ofstream& csv_out) {

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Dataset: " << dataset_name << " (" << data_type << ")" << std::endl;
    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Size: " << (data.size() * sizeof(T) / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Compress with partition size 4096
    const int PARTITION_SIZE = 4096;
    std::cout << "Compressing with partition size " << PARTITION_SIZE << "..." << std::endl;

    CompressedDataL3<T>* compressed = compressData(data, PARTITION_SIZE);
    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return;
    }

    std::cout << "  Partitions: " << compressed->num_partitions << std::endl;

    // Calculate compression ratio
    size_t original_size = data.size() * sizeof(T);
    size_t compressed_size = 0;

    // Download delta_bits to analyze
    std::vector<int32_t> h_delta_bits(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    // Estimate compressed size
    std::vector<int32_t> h_start(compressed->num_partitions);
    std::vector<int32_t> h_end(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    int64_t total_bits = 0;
    for (int i = 0; i < compressed->num_partitions; i++) {
        int len = h_end[i] - h_start[i];
        total_bits += (int64_t)h_delta_bits[i] * len;
    }

    compressed_size = (total_bits + 7) / 8; // bits to bytes
    compressed_size += compressed->num_partitions * (8 + 4 * 8); // metadata per partition

    double compression_ratio = (double)original_size / compressed_size;
    double avg_bits_per_value = (double)total_bits / data.size();

    std::cout << "  Avg delta bits: " << std::fixed << std::setprecision(2)
              << avg_bits_per_value << std::endl;
    std::cout << "  Compression ratio: " << std::setprecision(2)
              << compression_ratio << "x" << std::endl;

    // Prepare delta_bits as uint8_t for Phase 2.2
    std::vector<uint8_t> h_delta_bits_u8(compressed->num_partitions);
    for (int i = 0; i < compressed->num_partitions; i++) {
        h_delta_bits_u8[i] = static_cast<uint8_t>(h_delta_bits[i]);
    }

    uint8_t* d_delta_bits_u8;
    CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits_u8.data(),
                         compressed->num_partitions * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    // Allocate output buffer
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(T)));

    // Warmup runs
    const int WARMUP_RUNS = 3;
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < WARMUP_RUNS; i++) {
        decompressL3_Phase2_Bucket<T>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark runs
    const int BENCHMARK_RUNS = 10;
    std::cout << "Benchmarking (" << BENCHMARK_RUNS << " runs)..." << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times;
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressL3_Phase2_Bucket<T>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }

    // Calculate statistics
    std::sort(times.begin(), times.end());
    float median_ms = times[times.size() / 2];
    float min_ms = times[0];
    float max_ms = times[times.size() - 1];
    float avg_ms = 0;
    for (float t : times) avg_ms += t;
    avg_ms /= times.size();

    double bytes = data.size() * sizeof(T);
    double gb = bytes / 1e9;
    double throughput_median = gb / (median_ms / 1000.0);
    double throughput_best = gb / (min_ms / 1000.0);

    // Print results
    std::cout << "\n" << "Results:" << std::endl;
    std::cout << "  Time (median): " << std::fixed << std::setprecision(3)
              << median_ms << " ms" << std::endl;
    std::cout << "  Time (best):   " << min_ms << " ms" << std::endl;
    std::cout << "  Time (worst):  " << max_ms << " ms" << std::endl;
    std::cout << "  Time (avg):    " << avg_ms << " ms" << std::endl;
    std::cout << "  Throughput (median): " << std::setprecision(2)
              << throughput_median << " GB/s" << std::endl;
    std::cout << "  Throughput (best):   " << throughput_best << " GB/s" << std::endl;

    // Write to CSV
    csv_out << dataset_name << ","
            << data_type << ","
            << data.size() << ","
            << compressed->num_partitions << ","
            << std::fixed << std::setprecision(2) << compression_ratio << ","
            << std::setprecision(2) << avg_bits_per_value << ","
            << std::setprecision(3) << median_ms << ","
            << std::setprecision(3) << min_ms << ","
            << std::setprecision(3) << max_ms << ","
            << std::setprecision(2) << throughput_median << ","
            << std::setprecision(2) << throughput_best << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_delta_bits_u8));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    freeCompressedData(compressed);

    std::cout << "✓ Completed" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  L3 Phase 2.2 Benchmark: All 6 SOSD Datasets                  ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Get device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << " (SM " << prop.major << "."
              << prop.minor << ")" << std::endl;
    std::cout << "Global Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0)
              << " GB" << std::endl;
    std::cout << std::endl;

    // Prepare CSV output
    const char* csv_filename = "benchmark_results_all_datasets.csv";
    std::ofstream csv_out(csv_filename);
    csv_out << "dataset,data_type,num_elements,num_partitions,compression_ratio,"
            << "avg_delta_bits,time_median_ms,time_best_ms,time_worst_ms,"
            << "throughput_median_gbps,throughput_best_gbps" << std::endl;

    // Define all 6 datasets
    std::vector<DatasetInfo> datasets = {
        {"books_200M", "data/books_200M_uint32.bin", "uint32", 0, 0},
        {"fb_200M", "data/fb_200M_uint64.bin", "uint64", 0, 0},
        {"linear_200M", "data/linear_200M_uint32_binary.bin", "uint32", 0, 0},
        {"movieid", "data/movieid_uint32.bin", "uint32", 0, 0},
        {"normal_200M", "data/normal_200M_uint32_binary.bin", "uint32", 0, 0},
        {"wiki_200M", "data/wiki_200M_uint64.bin", "uint64", 0, 0}
    };

    // Process each dataset
    int completed = 0;
    for (const auto& ds : datasets) {
        std::cout << "\n" << std::string(70, '━') << std::endl;
        std::cout << "Processing dataset " << (completed + 1) << "/6: " << ds.name << std::endl;
        std::cout << std::string(70, '━') << std::endl;

        try {
            if (ds.data_type == "uint32") {
                size_t num_elements = 0;
                auto data = loadBinaryUint32(ds.filepath.c_str(), num_elements);
                benchmarkDataset(data, ds.name, ds.data_type, csv_out);
            } else {
                size_t num_elements = 0;
                auto data = loadBinaryUint64(ds.filepath.c_str(), num_elements);
                benchmarkDataset(data, ds.name, ds.data_type, csv_out);
            }
            completed++;
        } catch (const std::exception& e) {
            std::cerr << "Error processing " << ds.name << ": " << e.what() << std::endl;
        }
    }

    csv_out.close();

    std::cout << "\n" << std::string(70, '═') << std::endl;
    std::cout << "Benchmark Complete!" << std::endl;
    std::cout << "Processed " << completed << "/6 datasets successfully" << std::endl;
    std::cout << "Results saved to: " << csv_filename << std::endl;
    std::cout << std::string(70, '═') << std::endl;

    return 0;
}
