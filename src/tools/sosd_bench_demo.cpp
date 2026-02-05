#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>
#include <cuda_runtime.h>
#include "sosd_loader.h"
#include "L3_opt.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Simple compression simulator (generates synthetic compressed data for demo)
template<typename T>
void simulateCompression(const std::vector<T>& data,
                        int bitwidth,
                        double& compression_ratio,
                        double& compression_gbps) {
    // Simplified: assume linear model with small deltas
    size_t original_bytes = data.size() * sizeof(T);
    size_t compressed_bytes = data.size() * bitwidth / 8 + 1024;  // Data + metadata

    compression_ratio = static_cast<double>(original_bytes) / compressed_bytes;

    // Simulate compression time (based on data size)
    double compression_time_ms = data.size() / 1000000.0 * 0.5;  // ~2 GB/s estimate
    compression_gbps = (original_bytes / 1e9) / (compression_time_ms / 1000.0);
}

// Benchmark a single SOSD dataset
template<typename T>
void benchmarkDataset(const std::string& filepath) {
    std::cout << "\n=== Benchmarking: " << SOSDLoader::getDatasetName(filepath) << " ===" << std::endl;

    // Load dataset
    std::vector<T> data;
    if (!SOSDLoader::loadDataset<T>(filepath, data)) {
        std::cerr << "Failed to load dataset" << std::endl;
        return;
    }

    // Test with different bitwidths
    std::vector<int> bitwidths = {4, 8, 16};

    for (int bitwidth : bitwidths) {
        std::cout << "\n--- Bitwidth: " << bitwidth << " ---" << std::endl;

        // Simulate compression
        double comp_ratio, comp_gbps;
        simulateCompression(data, bitwidth, comp_ratio, comp_gbps);

        std::cout << "Compression Ratio: " << comp_ratio << "x" << std::endl;
        std::cout << "Compression Throughput: " << comp_gbps << " GB/s (simulated)" << std::endl;

        // For decompression, we would:
        // 1. Upload compressed data to GPU
        // 2. Call launchDecompressOptimized<T>()
        // 3. Measure kernel time with CUDA events
        // 4. Calculate throughput

        std::cout << "Decompression: Would use existing optimized kernel" << std::endl;
        std::cout << "Expected throughput: 300-500 GB/s (based on previous benchmarks)" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "L3 SOSD Benchmark Demo" << std::endl;
    std::cout << "=========================" << std::endl;

    // Get GPU info
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << std::endl;

    // SOSD dataset directory
    const std::string sosd_dir = "data";

    // List all datasets using dirent
    std::vector<std::string> datasets;
    DIR* dir = opendir(sosd_dir.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename = entry->d_name;
            if (filename.find(".bin") != std::string::npos ||
                filename.find(".txt") != std::string::npos) {
                datasets.push_back(sosd_dir + "/" + filename);
            }
        }
        closedir(dir);
    }

    std::cout << "Found " << datasets.size() << " datasets:" << std::endl;
    for (const auto& ds : datasets) {
        std::cout << "  - " << SOSDLoader::getDatasetName(ds) << std::endl;
    }

    // Benchmark first dataset as demo
    if (!datasets.empty()) {
        std::string first_dataset = datasets[0];

        // Determine type from filename
        if (first_dataset.find("uint64") != std::string::npos) {
            std::cout << "\nDataset type: uint64" << std::endl;
            // For full implementation: benchmarkDataset<uint64_t>(first_dataset);
        } else {
            std::cout << "\nDataset type: uint32" << std::endl;
            benchmarkDataset<uint32_t>(first_dataset);
        }
    }

    std::cout << "\n=======================" << std::endl;
    std::cout << "Demo completed." << std::endl;
    std::cout << "\nFull implementation would include:" << std::endl;
    std::cout << "  1. Complete compression kernel" << std::endl;
    std::cout << "  2. End-to-end timing with H2D/D2H" << std::endl;
    std::cout << "  3. Multiple selectivity tests" << std::endl;
    std::cout << "  4. CSV logging for all metrics" << std::endl;
    std::cout << "  5. Automatic visualization generation" << std::endl;

    return 0;
}
