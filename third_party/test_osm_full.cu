/**
 * Full benchmark of FastLanesGPU 64-bit on OSM 800M dataset
 * Processes entire dataset with proper memory management
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cstdint>

#include "FastLanesGPU/fastlanes/src/include/fls_gen/unpack/unpack_64.cuh"
#include "FastLanesGPU/fastlanes/src/include/fls_gen/pack/pack_64.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

int main() {
    std::cout << "=== FastLanesGPU 64-bit Full Benchmark ===" << std::endl;

    // Check GPU
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem / (1024*1024)) << " MB)" << std::endl;

    // Load data
    const char* filename = "/root/autodl-tmp/test/data/sosd/osm_cellids_800M_uint64.bin";
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open file" << std::endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(uint64_t);
    // Align to 1024 (FastLanes block size)
    num_elements = (num_elements / 1024) * 1024;
    size_t num_blocks = num_elements / 1024;

    std::cout << "Elements: " << num_elements << " (" << (num_elements * 8.0 / 1e9) << " GB)" << std::endl;
    std::cout << "Blocks: " << num_blocks << std::endl;

    // Allocate host memory
    std::vector<uint64_t> h_data(num_elements);
    file.read(reinterpret_cast<char*>(h_data.data()), num_elements * sizeof(uint64_t));
    file.close();

    // Allocate GPU memory
    const uint8_t bitwidth = 64;  // Full 64-bit for high-entropy OSM data
    size_t packed_words = (num_elements * bitwidth + 63) / 64;

    uint64_t *d_input, *d_packed, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_packed, packed_words * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, num_elements * sizeof(uint64_t)));

    // Copy to GPU
    std::cout << "Copying to GPU..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_input, h_data.data(), num_elements * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_packed, 0, packed_words * sizeof(uint64_t)));

    // Warmup
    pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bitwidth);
    unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bitwidth);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark with CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iterations = 10;

    // Pack benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        pack_global_64<<<num_blocks, 32>>>(d_input, d_packed, bitwidth);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float pack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pack_ms, start, stop));
    pack_ms /= iterations;

    // Unpack benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        unpack_global_64<<<num_blocks, 32>>>(d_packed, d_output, bitwidth);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float unpack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&unpack_ms, start, stop));
    unpack_ms /= iterations;

    // Verify
    std::cout << "Verifying..." << std::endl;
    std::vector<uint64_t> h_output(num_elements);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, num_elements * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    bool correct = true;
    size_t error_count = 0;
    for (size_t i = 0; i < num_elements && error_count < 10; i++) {
        if (h_output[i] != h_data[i]) {
            if (error_count == 0) {
                std::cerr << "First error at index " << i << ": expected " << h_data[i]
                          << ", got " << h_output[i] << std::endl;
            }
            correct = false;
            error_count++;
        }
    }

    // Results
    double data_gb = (num_elements * sizeof(uint64_t)) / 1e9;
    double pack_throughput = data_gb / (pack_ms / 1000.0);
    double unpack_throughput = data_gb / (unpack_ms / 1000.0);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Dataset: OSM 800M uint64" << std::endl;
    std::cout << "Original size: " << (data_gb * 1000) << " MB" << std::endl;
    std::cout << "Bitwidth: " << (int)bitwidth << std::endl;
    std::cout << "Compression ratio: " << 1.0 << "x (no compression for random data)" << std::endl;
    std::cout << "\nPack (compression):" << std::endl;
    std::cout << "  Time: " << pack_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << pack_throughput << " GB/s" << std::endl;
    std::cout << "\nUnpack (decompression):" << std::endl;
    std::cout << "  Time: " << unpack_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << unpack_throughput << " GB/s" << std::endl;
    std::cout << "\nCorrectness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_packed));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return correct ? 0 : 1;
}
