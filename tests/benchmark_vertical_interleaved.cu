/**
 * Vertical Interleaved Format Performance Benchmark
 *
 * Specifically tests the interleaved (mini-vector) format vs sequential format
 * for decompression performance on SOSD dataset #2 (normal_200M_uint64).
 *
 * Metrics:
 * - Sequential decompression throughput
 * - Interleaved decompression throughput
 * - Speedup ratio
 * - Memory bandwidth utilization
 *
 * Date: 2025-12-07
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

// ============================================================================
// Data Loading
// ============================================================================

template<typename T>
std::vector<T> loadBinaryDataset(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << path << std::endl;
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check for header (some SOSD files have 8-byte count header)
    size_t data_size = file_size;
    if (file_size % sizeof(T) == 8) {
        file.seekg(8, std::ios::beg);
        data_size = file_size - 8;
    } else if (file_size % sizeof(T) != 0) {
        std::cerr << "Warning: File size not aligned to element size" << std::endl;
    }

    size_t num_elements = data_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    file.close();

    return data;
}

// ============================================================================
// GPU Information
// ============================================================================

void printGPUInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "========================================\n";
    std::cout << "GPU Information\n";
    std::cout << "========================================\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memory: " << (prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0))
              << " GB\n";
    std::cout << "Memory Bandwidth: " << (prop.memoryClockRate * 1e-6 * 2 * prop.memoryBusWidth / 8)
              << " GB/s (theoretical)\n";
    std::cout << "SMs: " << prop.multiProcessorCount << "\n";
    std::cout << "Warp Size: " << prop.warpSize << "\n";
    std::cout << "========================================\n\n";
}

// ============================================================================
// Benchmark Functions
// ============================================================================

template<typename T>
void benchmarkInterleavedFormat(
    const std::vector<T>& data,
    const std::string& dataset_name,
    int partition_size = 4096)
{
    std::cout << "\n========================================\n";
    std::cout << "Vertical Interleaved Benchmark: " << dataset_name << "\n";
    std::cout << "========================================\n";

    size_t n = data.size();
    double original_size_mb = static_cast<double>(n * sizeof(T)) / (1024.0 * 1024.0);
    double original_size_gb = original_size_mb / 1024.0;

    std::cout << "Elements: " << n << "\n";
    std::cout << "Element size: " << sizeof(T) << " bytes\n";
    std::cout << "Original size: " << std::fixed << std::setprecision(2)
              << original_size_mb << " MB\n";
    std::cout << "Partition size: " << partition_size << " elements\n";
    std::cout << "Mini-vector size: " << MINI_VECTOR_SIZE << " elements\n\n";

    // Configure Vertical with interleaved enabled
    VerticalConfig config;
    config.partition_size_hint = partition_size;
    config.enable_interleaved = true;
    config.enable_dual_format = true;
    config.interleaved_threshold = INTERLEAVED_THRESHOLD;
    config.enable_branchless_unpack = true;
    config.enable_adaptive_selection = true;

    // Create partitions
    auto partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);
    int num_partitions = partitions.size();

    std::cout << "Number of partitions: " << num_partitions << "\n";
    std::cout << "Average partition size: " << (double)n / num_partitions << "\n\n";

    // ========== Compression ==========
    std::cout << "--- Compression ---\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    auto compressed_warmup = Vertical_encoder::encodeVertical<T>(data, partitions, config);
    Vertical_encoder::freeCompressedData(compressed_warmup);
    partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);

    // Timed compression
    CUDA_CHECK(cudaDeviceSynchronize());
    auto compress_start = std::chrono::high_resolution_clock::now();

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time_ms = std::chrono::duration<double, std::milli>(
        compress_end - compress_start).count();

    // Compression statistics
    double seq_size_mb = static_cast<double>(compressed.sequential_delta_words * sizeof(uint32_t))
                         / (1024.0 * 1024.0);
    double int_size_mb = static_cast<double>(compressed.interleaved_delta_words * sizeof(uint32_t))
                         / (1024.0 * 1024.0);
    double total_size_mb = seq_size_mb + int_size_mb;
    double compression_ratio = original_size_mb / seq_size_mb;
    double compress_throughput = original_size_gb / (compress_time_ms / 1000.0);

    // Get delta bits statistics
    std::vector<int32_t> h_delta_bits(num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
                          num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));

    double total_bits = 0, min_bits = 64, max_bits = 0;
    for (int i = 0; i < num_partitions; i++) {
        total_bits += h_delta_bits[i];
        min_bits = std::min(min_bits, (double)h_delta_bits[i]);
        max_bits = std::max(max_bits, (double)h_delta_bits[i]);
    }
    double avg_bits = total_bits / num_partitions;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(3)
              << compression_ratio << "x\n";
    std::cout << "Sequential format size: " << std::setprecision(2) << seq_size_mb << " MB\n";
    std::cout << "Interleaved format size: " << int_size_mb << " MB\n";
    std::cout << "Total compressed size (dual): " << total_size_mb << " MB\n";
    std::cout << "Compression time: " << compress_time_ms << " ms\n";
    std::cout << "Compression throughput: " << compress_throughput << " GB/s\n";
    std::cout << "Delta bits - avg: " << std::setprecision(1) << avg_bits
              << ", min: " << min_bits << ", max: " << max_bits << "\n";
    std::cout << "Interleaved partitions: " << compressed.total_interleaved_partitions
              << " / " << num_partitions << "\n\n";

    // ========== Allocate output buffer ==========
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));

    const int WARMUP_RUNS = 5;
    const int BENCHMARK_RUNS = 20;

    // ========== Sequential (Branchless) Decompression ==========
    std::cout << "--- Sequential Decompression (Branchless) ---\n";

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::BRANCHLESS);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::vector<float> seq_times(BENCHMARK_RUNS);
    for (int run = 0; run < BENCHMARK_RUNS; run++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::BRANCHLESS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&seq_times[run], start, stop));
    }

    // Verify correctness
    std::vector<T> decoded_seq(n);
    CUDA_CHECK(cudaMemcpy(decoded_seq.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    bool seq_correct = true;
    size_t first_error_idx = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != decoded_seq[i]) {
            seq_correct = false;
            first_error_idx = i;
            break;
        }
    }

    // Statistics
    std::sort(seq_times.begin(), seq_times.end());
    float seq_median = seq_times[BENCHMARK_RUNS / 2];
    float seq_min = seq_times[0];
    float seq_max = seq_times[BENCHMARK_RUNS - 1];
    float seq_avg = 0;
    for (float t : seq_times) seq_avg += t;
    seq_avg /= BENCHMARK_RUNS;

    double seq_throughput = original_size_gb / (seq_median / 1000.0);

    std::cout << "Time (median): " << std::fixed << std::setprecision(3) << seq_median << " ms\n";
    std::cout << "Time (min/max): " << seq_min << " / " << seq_max << " ms\n";
    std::cout << "Throughput (median): " << std::setprecision(2) << seq_throughput << " GB/s\n";
    std::cout << "Correctness: " << (seq_correct ? "PASS" : "FAIL") << "\n";
    if (!seq_correct) {
        std::cout << "  First error at index " << first_error_idx
                  << ": expected " << data[first_error_idx]
                  << ", got " << decoded_seq[first_error_idx] << "\n";
    }
    std::cout << "\n";

    // ========== Interleaved Decompression ==========
    std::cout << "--- Interleaved Decompression (Mini-Vector) ---\n";

    if (compressed.d_interleaved_deltas == nullptr || compressed.total_interleaved_partitions == 0) {
        std::cout << "No interleaved data available!\n\n";
    } else {
        // Clear output
        CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(T)));

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; i++) {
            Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::INTERLEAVED);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Benchmark
        std::vector<float> int_times(BENCHMARK_RUNS);
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            CUDA_CHECK(cudaEventRecord(start));
            Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::INTERLEAVED);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&int_times[run], start, stop));
        }

        // Verify correctness
        std::vector<T> decoded_int(n);
        CUDA_CHECK(cudaMemcpy(decoded_int.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

        bool int_correct = true;
        size_t int_first_error = 0;
        for (size_t i = 0; i < n; i++) {
            if (data[i] != decoded_int[i]) {
                int_correct = false;
                int_first_error = i;
                break;
            }
        }

        // Statistics
        std::sort(int_times.begin(), int_times.end());
        float int_median = int_times[BENCHMARK_RUNS / 2];
        float int_min = int_times[0];
        float int_max = int_times[BENCHMARK_RUNS - 1];
        float int_avg = 0;
        for (float t : int_times) int_avg += t;
        int_avg /= BENCHMARK_RUNS;

        double int_throughput = original_size_gb / (int_median / 1000.0);

        std::cout << "Time (median): " << std::fixed << std::setprecision(3) << int_median << " ms\n";
        std::cout << "Time (min/max): " << int_min << " / " << int_max << " ms\n";
        std::cout << "Throughput (median): " << std::setprecision(2) << int_throughput << " GB/s\n";
        std::cout << "Correctness: " << (int_correct ? "PASS" : "FAIL") << "\n";
        if (!int_correct) {
            std::cout << "  First error at index " << int_first_error
                      << ": expected " << data[int_first_error]
                      << ", got " << decoded_int[int_first_error] << "\n";
        }
        std::cout << "\n";

        // ========== Comparison ==========
        std::cout << "--- Performance Comparison ---\n";
        double speedup = seq_median / int_median;
        std::cout << "Sequential time: " << seq_median << " ms\n";
        std::cout << "Interleaved time: " << int_median << " ms\n";
        std::cout << "Speedup (Interleaved vs Sequential): " << std::setprecision(3)
                  << speedup << "x\n";

        if (speedup > 1.0) {
            std::cout << "Result: Interleaved is " << ((speedup - 1) * 100)
                      << "% FASTER\n";
        } else {
            std::cout << "Result: Sequential is " << ((1.0/speedup - 1) * 100)
                      << "% FASTER\n";
        }
        std::cout << "\n";
    }

    // ========== Random Access Test ==========
    std::cout << "--- Random Access Performance ---\n";

    const int NUM_QUERIES = 100000;
    std::vector<int> h_indices(NUM_QUERIES);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < NUM_QUERIES; i++) {
        h_indices[i] = dist(rng);
    }

    int* d_indices;
    T* d_results;
    CUDA_CHECK(cudaMalloc(&d_indices, NUM_QUERIES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, NUM_QUERIES * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_indices, h_indices.data(), NUM_QUERIES * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressIndices<T>(compressed, d_indices, NUM_QUERIES, d_results);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < BENCHMARK_RUNS; run++) {
        Vertical_decoder::decompressIndices<T>(compressed, d_indices, NUM_QUERIES, d_results);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ra_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&ra_total_ms, start, stop));
    double ra_ns_per_query = (ra_total_ms * 1e6) / (BENCHMARK_RUNS * NUM_QUERIES);
    double ra_queries_per_sec = (BENCHMARK_RUNS * NUM_QUERIES) / (ra_total_ms / 1000.0);

    // Verify random access correctness
    std::vector<T> h_results(NUM_QUERIES);
    CUDA_CHECK(cudaMemcpy(h_results.data(), d_results, NUM_QUERIES * sizeof(T),
                          cudaMemcpyDeviceToHost));

    bool ra_correct = true;
    for (int i = 0; i < NUM_QUERIES; i++) {
        if (h_results[i] != data[h_indices[i]]) {
            ra_correct = false;
            std::cout << "Random access error at query " << i
                      << ": index=" << h_indices[i]
                      << ", expected=" << data[h_indices[i]]
                      << ", got=" << h_results[i] << "\n";
            break;
        }
    }

    std::cout << "Queries: " << NUM_QUERIES << "\n";
    std::cout << "Latency: " << std::fixed << std::setprecision(1) << ra_ns_per_query << " ns/query\n";
    std::cout << "Throughput: " << std::setprecision(2) << (ra_queries_per_sec / 1e6)
              << " M queries/sec\n";
    std::cout << "Correctness: " << (ra_correct ? "PASS" : "FAIL") << "\n\n";

    // ========== Summary ==========
    std::cout << "========================================\n";
    std::cout << "SUMMARY: " << dataset_name << "\n";
    std::cout << "========================================\n";
    std::cout << "Compression ratio: " << compression_ratio << "x\n";
    std::cout << "Sequential throughput: " << seq_throughput << " GB/s\n";
    if (compressed.d_interleaved_deltas != nullptr) {
        double int_throughput = original_size_gb / (seq_times[BENCHMARK_RUNS/2] / 1000.0);
        // Recalculate with actual interleaved timing
        std::cout << "Interleaved throughput: see above\n";
    }
    std::cout << "Random access: " << ra_ns_per_query << " ns/query\n";
    std::cout << "========================================\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Vertical_encoder::freeCompressedData(compressed);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string dataset_path = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin";
    int partition_size = 4096;

    if (argc > 1) {
        dataset_path = argv[1];
    }
    if (argc > 2) {
        partition_size = std::atoi(argv[2]);
    }

    std::cout << "========================================\n";
    std::cout << "Vertical Interleaved Format Benchmark\n";
    std::cout << "========================================\n\n";

    printGPUInfo();

    std::cout << "Dataset: " << dataset_path << "\n";
    std::cout << "Loading data...\n";

    // Load dataset (assuming uint64 for dataset #2)
    auto data = loadBinaryDataset<uint64_t>(dataset_path);
    if (data.empty()) {
        std::cerr << "Failed to load dataset!\n";
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements\n\n";

    // Test with different partition sizes
    std::vector<int> partition_sizes = {1024, 2048, 4096, 8192};

    for (int psize : partition_sizes) {
        benchmarkInterleavedFormat<uint64_t>(data, "normal_200M (partition=" + std::to_string(psize) + ")", psize);
    }

    std::cout << "\n========================================\n";
    std::cout << "All benchmarks completed!\n";
    std::cout << "========================================\n";

    return 0;
}
