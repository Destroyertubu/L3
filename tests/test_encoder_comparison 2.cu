/**
 * Encoder Comparison Test
 *
 * Compares three encoder implementations:
 * 1. encodeVertical (CPU version) - uses CPU for metadata computation
 * 2. encodeVerticalGPU (GPU dynamic) - GPU-only with dynamic buffer allocation
 * 3. encodeVerticalGPU_ZeroSync (GPU zero-sync) - GPU-only with pre-allocation
 *
 * Measures:
 * - Compression time
 * - Memory usage
 * - Correctness (all encoders should produce identical decompressed results)
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

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

// Generate test data with different patterns
template<typename T>
std::vector<T> generateLinearData(size_t n, T start = 1000000, T step = 1) {
    std::vector<T> data(n);
    for (size_t i = 0; i < n; i++) {
        data[i] = start + static_cast<T>(i) * step;
    }
    return data;
}

template<typename T>
std::vector<T> generateRandomData(size_t n, T max_val = 1000000000) {
    std::vector<T> data(n);
    srand(42);
    for (size_t i = 0; i < n; i++) {
        data[i] = static_cast<T>(rand()) % max_val;
    }
    return data;
}

// Measure memory usage
struct MemoryStats {
    size_t sequential_bytes;
    size_t interleaved_bytes;
    size_t metadata_bytes;
    size_t total_bytes;
};

template<typename T>
MemoryStats getMemoryStats(const CompressedDataVertical<T>& compressed, int num_partitions) {
    MemoryStats stats;
    stats.sequential_bytes = compressed.sequential_delta_words * sizeof(uint32_t);
    stats.interleaved_bytes = compressed.interleaved_delta_words * sizeof(uint32_t);
    stats.metadata_bytes = num_partitions * (
        sizeof(int32_t) * 2 +    // start/end indices
        sizeof(int32_t) +        // model_types
        sizeof(double) * 4 +     // model_params
        sizeof(int32_t) +        // delta_bits
        sizeof(int64_t) * 2      // bit_offsets, error_bounds
    );
    stats.total_bytes = stats.sequential_bytes + stats.interleaved_bytes + stats.metadata_bytes;
    return stats;
}

// Test a single encoder
template<typename T>
struct EncoderResult {
    double time_ms;
    MemoryStats memory;
    bool correct;
    std::vector<T> decompressed;
};

template<typename T>
EncoderResult<T> testCPUEncoder(const std::vector<T>& data, int partition_size, const VerticalConfig& config) {
    EncoderResult<T> result;

    auto partitions = Vertical_encoder::createFixedPartitions<T>(static_cast<int>(data.size()), partition_size);

    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.memory = getMemoryStats(compressed, partitions.size());

    // Decompress for verification
    size_t n = data.size();
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));

    Vertical_decoder::decompressAll(compressed, d_output, DecompressMode::BRANCHLESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    result.decompressed.resize(n);
    CUDA_CHECK(cudaMemcpy(result.decompressed.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    result.correct = (result.decompressed == data);

    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    return result;
}

template<typename T>
EncoderResult<T> testGPUEncoder(const std::vector<T>& data, int partition_size, const VerticalConfig& config) {
    EncoderResult<T> result;

    int num_partitions = (data.size() + partition_size - 1) / partition_size;

    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = Vertical_encoder::encodeVerticalGPU<T>(data, partition_size, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.memory = getMemoryStats(compressed, num_partitions);

    // Decompress for verification
    size_t n = data.size();
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));

    Vertical_decoder::decompressAll(compressed, d_output, DecompressMode::BRANCHLESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    result.decompressed.resize(n);
    CUDA_CHECK(cudaMemcpy(result.decompressed.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    result.correct = (result.decompressed == data);

    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    return result;
}

template<typename T>
EncoderResult<T> testGPUZeroSyncEncoder(const std::vector<T>& data, int partition_size, const VerticalConfig& config) {
    EncoderResult<T> result;

    int num_partitions = (data.size() + partition_size - 1) / partition_size;

    auto start = std::chrono::high_resolution_clock::now();
    auto compressed = Vertical_encoder::encodeVerticalGPU_ZeroSync<T>(data, partition_size, config);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    result.time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.memory = getMemoryStats(compressed, num_partitions);

    // Decompress for verification
    size_t n = data.size();
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));

    Vertical_decoder::decompressAll(compressed, d_output, DecompressMode::BRANCHLESS);
    CUDA_CHECK(cudaDeviceSynchronize());

    result.decompressed.resize(n);
    CUDA_CHECK(cudaMemcpy(result.decompressed.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    result.correct = (result.decompressed == data);

    CUDA_CHECK(cudaFree(d_output));
    Vertical_encoder::freeCompressedData(compressed);

    return result;
}

void printHeader() {
    std::cout << "\n";
    std::cout << "=========================================================================\n";
    std::cout << "                    Encoder Comparison Test\n";
    std::cout << "=========================================================================\n";
    std::cout << std::setw(15) << "Encoder"
              << std::setw(12) << "Time(ms)"
              << std::setw(15) << "SeqMB"
              << std::setw(15) << "IntMB"
              << std::setw(15) << "TotalMB"
              << std::setw(10) << "Status"
              << "\n";
    std::cout << "-------------------------------------------------------------------------\n";
}

template<typename T>
void runComparison(const std::vector<T>& data, const std::string& data_name, int partition_size) {
    std::cout << "\n=== " << data_name << " (n=" << data.size() << ", partition=" << partition_size << ") ===\n";

    // v3.0: Interleaved-only config
    VerticalConfig config;
    config.partition_size_hint = partition_size;
    config.enable_interleaved = true;
    config.enable_branchless_unpack = true;
    config.enable_adaptive_selection = true;

    printHeader();

    // Test CPU encoder
    auto cpu_result = testCPUEncoder<T>(data, partition_size, config);
    std::cout << std::setw(15) << "CPU"
              << std::setw(12) << std::fixed << std::setprecision(2) << cpu_result.time_ms
              << std::setw(15) << std::setprecision(2) << cpu_result.memory.sequential_bytes / 1024.0 / 1024.0
              << std::setw(15) << cpu_result.memory.interleaved_bytes / 1024.0 / 1024.0
              << std::setw(15) << cpu_result.memory.total_bytes / 1024.0 / 1024.0
              << std::setw(10) << (cpu_result.correct ? "PASS" : "FAIL")
              << "\n";

    // Test GPU encoder
    auto gpu_result = testGPUEncoder<T>(data, partition_size, config);
    std::cout << std::setw(15) << "GPU"
              << std::setw(12) << std::fixed << std::setprecision(2) << gpu_result.time_ms
              << std::setw(15) << std::setprecision(2) << gpu_result.memory.sequential_bytes / 1024.0 / 1024.0
              << std::setw(15) << gpu_result.memory.interleaved_bytes / 1024.0 / 1024.0
              << std::setw(15) << gpu_result.memory.total_bytes / 1024.0 / 1024.0
              << std::setw(10) << (gpu_result.correct ? "PASS" : "FAIL")
              << "\n";

    // Test GPU ZeroSync encoder
    auto zerosync_result = testGPUZeroSyncEncoder<T>(data, partition_size, config);
    std::cout << std::setw(15) << "GPU_ZEROSYNC"
              << std::setw(12) << std::fixed << std::setprecision(2) << zerosync_result.time_ms
              << std::setw(15) << std::setprecision(2) << zerosync_result.memory.sequential_bytes / 1024.0 / 1024.0
              << std::setw(15) << zerosync_result.memory.interleaved_bytes / 1024.0 / 1024.0
              << std::setw(15) << zerosync_result.memory.total_bytes / 1024.0 / 1024.0
              << std::setw(10) << (zerosync_result.correct ? "PASS" : "FAIL")
              << "\n";

    std::cout << "-------------------------------------------------------------------------\n";

    // Compare decompression results
    bool all_match = true;
    if (cpu_result.decompressed != gpu_result.decompressed) {
        std::cout << "WARNING: CPU and GPU results differ!\n";
        all_match = false;
    }
    if (cpu_result.decompressed != zerosync_result.decompressed) {
        std::cout << "WARNING: CPU and GPU_ZEROSYNC results differ!\n";
        all_match = false;
    }

    if (all_match && cpu_result.correct && gpu_result.correct && zerosync_result.correct) {
        std::cout << "All encoders produce identical, correct results.\n";
    }

    // Speedup analysis
    double gpu_speedup = cpu_result.time_ms / gpu_result.time_ms;
    double zerosync_speedup = cpu_result.time_ms / zerosync_result.time_ms;
    double zerosync_vs_gpu = gpu_result.time_ms / zerosync_result.time_ms;

    std::cout << "\nSpeedup Analysis:\n";
    std::cout << "  GPU vs CPU:          " << std::fixed << std::setprecision(2) << gpu_speedup << "x\n";
    std::cout << "  GPU_ZEROSYNC vs CPU: " << zerosync_speedup << "x\n";
    std::cout << "  GPU_ZEROSYNC vs GPU: " << zerosync_vs_gpu << "x\n";

    // Memory analysis
    double zerosync_memory_overhead =
        (zerosync_result.memory.total_bytes - cpu_result.memory.total_bytes) * 100.0 / cpu_result.memory.total_bytes;
    std::cout << "\nMemory Analysis:\n";
    std::cout << "  GPU_ZEROSYNC memory overhead: " << std::fixed << std::setprecision(1)
              << zerosync_memory_overhead << "%\n";
}

int main(int argc, char* argv[]) {
    std::cout << "=========================================================================\n";
    std::cout << "        L3 Encoder Comparison: CPU vs GPU vs GPU_ZEROSYNC\n";
    std::cout << "=========================================================================\n";

    // Test with different data sizes
    std::vector<size_t> sizes = {1000000, 10000000, 50000000};
    std::vector<int> partition_sizes = {1024, 2048, 4096};

    // Allow command-line override for quick testing
    if (argc > 1) {
        sizes = {static_cast<size_t>(std::atol(argv[1]))};
    }
    if (argc > 2) {
        partition_sizes = {std::atoi(argv[2])};
    }

    for (size_t n : sizes) {
        for (int ps : partition_sizes) {
            // Test with linear data
            auto linear_data = generateLinearData<uint64_t>(n);
            runComparison(linear_data, "Linear uint64", ps);

            // Test with random data
            auto random_data = generateRandomData<uint64_t>(n);
            runComparison(random_data, "Random uint64", ps);
        }
    }

    std::cout << "\n=========================================================================\n";
    std::cout << "                         Test Complete\n";
    std::cout << "=========================================================================\n";

    return 0;
}
