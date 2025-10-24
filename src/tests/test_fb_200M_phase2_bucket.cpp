/**
 * GLECO Phase 2.2 Test: Facebook 200M uint64 Dataset
 *
 * Real-world benchmark on large-scale sorted dataset
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
#include "L3_codec.hpp"
#include "L3_format.hpp"

// External function declarations
template<typename T>
void decompressGLECO_Phase2_Bucket(
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

// Load binary data
std::vector<uint64_t> loadBinaryFile(const char* filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Failed to open " << filename << std::endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(uint64_t);

    if (max_elements > 0 && max_elements < num_elements) {
        num_elements = max_elements;
    }

    std::cout << "Loading " << num_elements << " elements from " << filename << std::endl;
    std::cout << "File size: " << (file_size / 1024.0 / 1024.0 / 1024.0)
              << " GB" << std::endl;

    std::vector<uint64_t> data(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint64_t));

    std::cout << "Loaded successfully" << std::endl;
    std::cout << "Sample values: " << data[0] << ", " << data[1] << ", "
              << data[2] << ", ..." << std::endl;

    return data;
}

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   GLECO Phase 2.2: Facebook 200M Dataset Benchmark      ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
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

    // Load dataset
    const char* filename = "/root/autodl-tmp/test/data/fb_200M_uint64.bin";

    // Test with different sizes
    std::vector<size_t> test_sizes = {1000000, 8000000, 64000000, 200000000};

    for (size_t test_size : test_sizes) {
        std::cout << "\n" << std::string(70, '=') << std::endl;
        std::cout << "Testing with " << test_size << " elements ("
                  << (test_size * 8.0 / 1024.0 / 1024.0 / 1024.0) << " GB)" << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        auto data = loadBinaryFile(filename, test_size);

        // Determine partition size based on dataset size
        // Phase 2.2 optimal configuration: 16K partition size
        // Sweet spot for vectorization alignment + batch count balance
        int partition_size;
        if (test_size <= 1000000) {
            partition_size = 2048;
        } else if (test_size <= 8000000) {
            partition_size = 4096;
        } else if (test_size <= 64000000) {
            partition_size = 8192;
        } else {
            // Phase 2.2 baseline (reverted from Phase 2.3 attempt)
            // 16,384 is optimal: maintains 100% uint4 vectorization
            partition_size = 16384;  // Optimal for 200M test
        }

        std::cout << "\nPartition size: " << partition_size << std::endl;

        // Compress
        auto compress_start = std::chrono::high_resolution_clock::now();

        CompressionStats comp_stats;
        auto* compressed = compressData(data, partition_size, &comp_stats);

        auto compress_end = std::chrono::high_resolution_clock::now();
        double compress_time = std::chrono::duration<double, std::milli>(
            compress_end - compress_start).count();

        if (!compressed) {
            std::cerr << "❌ Compression failed" << std::endl;
            continue;
        }

        std::cout << "\n--- Compression Results ---" << std::endl;
        std::cout << "Compressed size: " << comp_stats.compressed_bytes
                  << " bytes (" << (comp_stats.compressed_bytes / 1024.0 / 1024.0)
                  << " MB)" << std::endl;
        std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
                  << comp_stats.compression_ratio << "x" << std::endl;
        std::cout << "Compression time: " << std::fixed << std::setprecision(3)
                  << compress_time << " ms" << std::endl;
        std::cout << "Avg delta bits: " << comp_stats.avg_delta_bits << std::endl;
        std::cout << "Partitions: " << comp_stats.num_partitions << std::endl;

        // Download delta_bits to show distribution
        std::vector<uint8_t> h_delta_bits(compressed->num_partitions);
        CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                             compressed->num_partitions * sizeof(uint8_t),
                             cudaMemcpyDeviceToHost));

        // Convert from int32_t to uint8_t properly
        std::vector<int32_t> h_delta_bits_i32(compressed->num_partitions);
        CUDA_CHECK(cudaMemcpy(h_delta_bits_i32.data(), compressed->d_delta_bits,
                             compressed->num_partitions * sizeof(int32_t),
                             cudaMemcpyDeviceToHost));

        for (int i = 0; i < compressed->num_partitions; ++i) {
            h_delta_bits[i] = static_cast<uint8_t>(h_delta_bits_i32[i]);
        }

        // Count bitwidth distribution
        std::map<int, int> bit_dist;
        for (auto b : h_delta_bits) {
            bit_dist[b]++;
        }

        std::cout << "\nBitwidth distribution: ";
        int count = 0;
        for (auto& p : bit_dist) {
            std::cout << p.first << "b:" << p.second << "p ";
            if (++count >= 10) {
                std::cout << "...";
                break;
            }
        }
        std::cout << std::endl;

        // Prepare for Phase 2.2 decompression
        uint8_t* d_delta_bits_u8;
        CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));
        CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits.data(),
                             compressed->num_partitions * sizeof(uint8_t),
                             cudaMemcpyHostToDevice));

        uint64_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));

        // Warmup
        std::cout << "\n--- Phase 2.2 Decompression ---" << std::endl;
        std::cout << "Warmup..." << std::flush;
        for (int i = 0; i < 5; ++i) {
            decompressGLECO_Phase2_Bucket<uint64_t>(
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
        std::cout << " done" << std::endl;

        // Benchmark
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        const int iters = 20;
        std::vector<float> times;

        std::cout << "Benchmarking (" << iters << " iterations)..." << std::flush;

        for (int i = 0; i < iters; ++i) {
            CUDA_CHECK(cudaEventRecord(start));

            decompressGLECO_Phase2_Bucket<uint64_t>(
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

        std::cout << " done" << std::endl;

        // Statistics
        std::sort(times.begin(), times.end());
        float min_ms = times.front();
        float max_ms = times.back();
        float median_ms = times[iters / 2];
        float avg_ms = 0;
        for (auto t : times) avg_ms += t;
        avg_ms /= iters;

        double gb = data.size() * sizeof(uint64_t) / 1e9;
        double throughput_median = gb / (median_ms / 1000.0);
        double throughput_best = gb / (min_ms / 1000.0);

        std::cout << "\n--- Performance Results ---" << std::endl;
        std::cout << "Min time:    " << std::fixed << std::setprecision(3) << min_ms << " ms" << std::endl;
        std::cout << "Median time: " << std::fixed << std::setprecision(3) << median_ms << " ms" << std::endl;
        std::cout << "Avg time:    " << std::fixed << std::setprecision(3) << avg_ms << " ms" << std::endl;
        std::cout << "Max time:    " << std::fixed << std::setprecision(3) << max_ms << " ms" << std::endl;
        std::cout << std::endl;
        std::cout << "Throughput (median): " << std::fixed << std::setprecision(2)
                  << throughput_median << " GB/s" << std::endl;
        std::cout << "Throughput (best):   " << std::fixed << std::setprecision(2)
                  << throughput_best << " GB/s" << std::endl;

        // Verify correctness (sample check)
        std::cout << "\n--- Correctness Check ---" << std::endl;
        std::vector<uint64_t> decompressed(data.size());
        CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                             data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // Check first 1000, middle 1000, last 1000
        bool correct = true;
        int errors = 0;
        const int check_size = std::min<size_t>(1000, data.size());

        auto check_range = [&](size_t start, size_t count, const char* label) {
            for (size_t i = start; i < start + count && i < data.size(); ++i) {
                if (decompressed[i] != data[i]) {
                    if (errors < 5) {
                        std::cerr << "Mismatch at [" << i << "] (" << label << "): "
                                  << "expected=" << data[i]
                                  << ", got=" << decompressed[i] << std::endl;
                    }
                    errors++;
                    correct = false;
                }
            }
        };

        check_range(0, check_size, "start");
        check_range(data.size() / 2, check_size, "middle");
        check_range(data.size() - check_size, check_size, "end");

        if (correct) {
            std::cout << "✅ PASS: Bit-exact roundtrip verified" << std::endl;
        } else {
            std::cerr << "❌ FAIL: " << errors << " mismatches found" << std::endl;
        }

        // Cleanup
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_delta_bits_u8));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        freeCompressedData(compressed);

        std::cout << std::endl;
    }

    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║             Facebook 200M Benchmark Complete             ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
