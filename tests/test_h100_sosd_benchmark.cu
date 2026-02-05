/**
 * Test: Transposed vs Vertical on H100 with SOSD Dataset
 *
 * Compares Vertical (Lane-Major) vs Transposed (Word-Interleaved) layouts
 * on the 5-books_200M_uint32 SOSD dataset.
 *
 * Date: 2025-12-16
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <cstdint>

// Include both Vertical and Transposed implementations
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Transposed_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"
#include "../src/kernels/compression/encoder_Transposed.cu"
#include "../src/kernels/decompression/decoder_Transposed.cu"

// ============================================================================
// SOSD Data Loader
// ============================================================================

template<typename T>
bool loadSOSDDataset(const std::string& filepath, std::vector<T>& data) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file: " << filepath << std::endl;
        return false;
    }

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(T);
    data.resize(num_elements);

    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return true;
}

// ============================================================================
// Timer
// ============================================================================

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }
    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }
private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Verification
// ============================================================================

template<typename T>
bool verifyResults(const T* h_original, const T* h_decoded, int64_t n) {
    int errors = 0;
    for (int64_t i = 0; i < n && errors < 10; i++) {
        if (h_original[i] != h_decoded[i]) {
            std::cout << "  Mismatch at index " << i
                      << ": expected " << h_original[i]
                      << ", got " << h_decoded[i] << std::endl;
            errors++;
        }
    }
    return errors == 0;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Configuration
    std::string dataset_path = "data/sosd/5-books_200M_uint32.bin";
    int gpu_id = 1;  // H100
    int partition_size = 4096;
    int num_warmup = 5;
    int num_runs = 20;

    // Parse args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--gpu" && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        } else if (arg == "--partition" && i + 1 < argc) {
            partition_size = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            num_runs = std::stoi(argv[++i]);
        } else if (arg == "--data" && i + 1 < argc) {
            dataset_path = argv[++i];
        }
    }

    // Set GPU
    cudaSetDevice(gpu_id);

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_id);
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Vertical vs Transposed on H100 - SOSD Dataset Benchmark        ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << "\nGPU " << gpu_id << ": " << prop.name << std::endl;
    std::cout << "L2 Cache: " << (prop.l2CacheSize / 1024 / 1024) << " MB" << std::endl;
    std::cout << "Memory Bandwidth: " << (prop.memoryBusWidth * prop.memoryClockRate * 2 / 1e6) << " GB/s (theoretical)" << std::endl;

    // Load dataset
    std::cout << "\nLoading dataset: " << dataset_path << std::endl;
    std::vector<uint32_t> h_data;
    if (!loadSOSDDataset(dataset_path, h_data)) {
        return 1;
    }

    int64_t num_values = h_data.size();
    double uncompressed_mb = num_values * sizeof(uint32_t) / 1024.0 / 1024.0;
    std::cout << "  Values: " << num_values << std::endl;
    std::cout << "  Size: " << std::fixed << std::setprecision(2) << uncompressed_mb << " MB" << std::endl;
    std::cout << "  Partition size: " << partition_size << std::endl;
    std::cout << "  Warmup runs: " << num_warmup << std::endl;
    std::cout << "  Benchmark runs: " << num_runs << std::endl;

    // ========== Vertical Encoding ==========
    std::cout << "\n=== Encoding with Vertical (Lane-Major) ===" << std::endl;

    VerticalConfig vertical_config;
    vertical_config.partition_size_hint = partition_size;
    vertical_config.enable_adaptive_selection = true;

    auto vertical_compressed = Vertical_encoder::encodeVerticalGPU<uint32_t>(
        h_data, partition_size, vertical_config);

    double vertical_compressed_mb = vertical_compressed.interleaved_delta_words * sizeof(uint32_t) / 1024.0 / 1024.0;
    std::cout << "  Compressed size: " << vertical_compressed_mb << " MB" << std::endl;
    std::cout << "  Compression ratio: " << (uncompressed_mb / vertical_compressed_mb) << "x" << std::endl;
    std::cout << "  Encoding time: " << vertical_compressed.kernel_time_ms << " ms" << std::endl;
    std::cout << "  Partitions: " << vertical_compressed.num_partitions << std::endl;

    // ========== Transposed Encoding ==========
    std::cout << "\n=== Encoding with Transposed (Word-Interleaved) ===" << std::endl;

    TransposedConfig transposed_config;
    transposed_config.partition_size_hint = partition_size;
    transposed_config.enable_adaptive_selection = true;

    auto transposed_compressed = Transposed_encoder::encodeTransposedGPU<uint32_t>(
        h_data, partition_size, transposed_config);

    double transposed_compressed_mb = transposed_compressed.transposed_delta_words * sizeof(uint32_t) / 1024.0 / 1024.0;
    std::cout << "  Compressed size: " << transposed_compressed_mb << " MB" << std::endl;
    std::cout << "  Compression ratio: " << (uncompressed_mb / transposed_compressed_mb) << "x" << std::endl;
    std::cout << "  Encoding time: " << transposed_compressed.kernel_time_ms << " ms" << std::endl;
    std::cout << "  Partitions: " << transposed_compressed.num_partitions << std::endl;

    // Allocate output buffers
    uint32_t* d_output_vertical;
    uint32_t* d_output_transposed;
    cudaMalloc(&d_output_vertical, num_values * sizeof(uint32_t));
    cudaMalloc(&d_output_transposed, num_values * sizeof(uint32_t));

    CudaTimer timer;

    // ========== Vertical Decode Benchmark ==========
    std::cout << "\n=== Benchmarking Vertical Decode ===" << std::endl;

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        Vertical_decoder::decompressAll<uint32_t>(vertical_compressed, d_output_vertical,
                                                   DecompressMode::INTERLEAVED);
    }
    cudaDeviceSynchronize();

    // Benchmark
    float vertical_total_ms = 0;
    for (int i = 0; i < num_runs; i++) {
        timer.start();
        Vertical_decoder::decompressAll<uint32_t>(vertical_compressed, d_output_vertical,
                                                   DecompressMode::INTERLEAVED);
        vertical_total_ms += timer.stop();
    }
    float vertical_decode_ms = vertical_total_ms / num_runs;

    double vertical_throughput = (num_values * sizeof(uint32_t) / 1e9) / (vertical_decode_ms / 1000.0);
    std::cout << "  Decode time: " << std::fixed << std::setprecision(4) << vertical_decode_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << vertical_throughput << " GB/s" << std::endl;

    // ========== Transposed Decode Benchmark ==========
    std::cout << "\n=== Benchmarking Transposed Decode ===" << std::endl;

    // Warmup
    for (int i = 0; i < num_warmup; i++) {
        Transposed_decoder::decompressAll<uint32_t>(transposed_compressed, d_output_transposed);
    }
    cudaDeviceSynchronize();

    // Benchmark
    float transposed_total_ms = 0;
    for (int i = 0; i < num_runs; i++) {
        timer.start();
        Transposed_decoder::decompressAll<uint32_t>(transposed_compressed, d_output_transposed);
        transposed_total_ms += timer.stop();
    }
    float transposed_decode_ms = transposed_total_ms / num_runs;

    double transposed_throughput = (num_values * sizeof(uint32_t) / 1e9) / (transposed_decode_ms / 1000.0);
    std::cout << "  Decode time: " << std::fixed << std::setprecision(4) << transposed_decode_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << transposed_throughput << " GB/s" << std::endl;

    // ========== Verification ==========
    std::cout << "\n=== Verifying Correctness ===" << std::endl;

    std::vector<uint32_t> h_output_vertical(num_values);
    std::vector<uint32_t> h_output_transposed(num_values);

    cudaMemcpy(h_output_vertical.data(), d_output_vertical,
               num_values * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_transposed.data(), d_output_transposed,
               num_values * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    bool vertical_ok = verifyResults(h_data.data(), h_output_vertical.data(), num_values);
    bool transposed_ok = verifyResults(h_data.data(), h_output_transposed.data(), num_values);

    std::cout << "  Vertical:   " << (vertical_ok ? "PASS" : "FAIL") << std::endl;
    std::cout << "  Transposed: " << (transposed_ok ? "PASS" : "FAIL") << std::endl;

    // ========== Summary ==========
    float speedup = vertical_decode_ms / transposed_decode_ms;
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                      PERFORMANCE SUMMARY                          ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Layout              │ Decode (ms) │ Throughput   │ Speedup       ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Vertical (Strided)  │ " << std::setw(11) << std::fixed << std::setprecision(4) << vertical_decode_ms
              << " │ " << std::setw(7) << std::setprecision(2) << vertical_throughput << " GB/s"
              << " │   1.00x       ║" << std::endl;
    std::cout << "║  Transposed (Coalesce│ " << std::setw(11) << std::fixed << std::setprecision(4) << transposed_decode_ms
              << " │ " << std::setw(7) << std::setprecision(2) << transposed_throughput << " GB/s"
              << " │ " << std::setw(6) << std::setprecision(2) << speedup << "x       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                        TEST SUMMARY                             │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│  Dataset: 5-books_200M_uint32                                   │" << std::endl;
    std::cout << "│  GPU: " << std::setw(57) << std::left << prop.name << "│" << std::endl;
    std::cout << "│  Data Size: " << std::setw(51) << std::left << (std::to_string(num_values) + " values (" + std::to_string(int(uncompressed_mb)) + " MB)") << "│" << std::endl;
    std::cout << "│  Speedup: " << std::setw(53) << std::left << (std::to_string(speedup).substr(0, 4) + "x (Transposed vs Vertical)") << "│" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    // Cleanup
    cudaFree(d_output_vertical);
    cudaFree(d_output_transposed);
    Vertical_encoder::freeCompressedData(vertical_compressed);
    Transposed_encoder::freeCompressedData(transposed_compressed);

    return (vertical_ok && transposed_ok) ? 0 : 1;
}
