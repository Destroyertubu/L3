/**
 * GLECO Compression Test Program
 *
 * Purpose: Test compression and decompression using decompression_kernels_phase2_bucket.cu
 * Dataset: /root/autodl-tmp/test/data/fb_200M_uint64.bin
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include "L3_codec.hpp"
#include "L3_format.hpp"

// External function declaration for Phase 2 Bucket decompression
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

// Load binary uint64 data from file
std::vector<uint64_t> loadBinaryFile(const char* filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(uint64_t);
    if (max_elements > 0 && num_elements > max_elements) {
        num_elements = max_elements;
    }

    std::vector<uint64_t> data(num_elements);
    if (!file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint64_t))) {
        std::cerr << "Error: Failed to read data from file" << std::endl;
        exit(1);
    }

    return data;
}

// Verify decompression correctness
bool verifyDecompression(const std::vector<uint64_t>& original,
                        const std::vector<uint64_t>& decompressed) {
    if (original.size() != decompressed.size()) {
        std::cerr << "Size mismatch: original=" << original.size()
                  << ", decompressed=" << decompressed.size() << std::endl;
        return false;
    }

    size_t errors = 0;
    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != decompressed[i]) {
            if (errors < 10) {
                std::cerr << "Mismatch at index " << i << ": original="
                          << original[i] << ", decompressed=" << decompressed[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << original.size() << std::endl;
        return false;
    }

    return true;
}

void printCompressionStats(const std::vector<uint64_t>& data,
                          const GLECOCompressedData<uint64_t>& compressed) {
    size_t original_bytes = data.size() * sizeof(uint64_t);
    size_t compressed_bytes = compressed.calculateCompressedSize();
    double ratio = static_cast<double>(original_bytes) / compressed_bytes;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Compression Statistics" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Original size:     " << std::setw(12) << original_bytes
              << " bytes (" << data.size() << " elements)" << std::endl;
    std::cout << "Compressed size:   " << std::setw(12) << compressed_bytes
              << " bytes" << std::endl;
    std::cout << "Compression ratio: " << std::setw(12) << std::fixed
              << std::setprecision(2) << ratio << "x" << std::endl;
    std::cout << "Space savings:     " << std::setw(12) << std::fixed
              << std::setprecision(2) << (1.0 - 1.0/ratio) * 100 << "%" << std::endl;
    std::cout << "Partitions:        " << std::setw(12) << compressed.num_partitions << std::endl;
    std::cout << std::string(70, '=') << std::endl;
}

int main(int argc, char** argv) {
    const char* default_file = "/root/autodl-tmp/test/data/fb_200M_uint64.bin";
    const char* input_file = (argc > 1) ? argv[1] : default_file;
    size_t max_elements = (argc > 2) ? std::stoull(argv[2]) : 0;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "GLECO Compression Test (Phase 2 Bucket Decompression)" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    std::cout << "Input file: " << input_file << std::endl;
    if (max_elements > 0) {
        std::cout << "Max elements: " << max_elements << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;

    // Step 1: Load data
    std::cout << "\n[1/4] Loading data..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<uint64_t> data = loadBinaryFile(input_file, max_elements);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_time = end - start;

    std::cout << "  Loaded " << data.size() << " elements in "
              << std::fixed << std::setprecision(3) << load_time.count() << " seconds" << std::endl;

    // Verify data is sorted
    bool is_sorted = std::is_sorted(data.begin(), data.end());
    std::cout << "  Data is " << (is_sorted ? "sorted" : "NOT sorted") << std::endl;

    // Step 2: Compress data
    std::cout << "\n[2/4] Compressing data..." << std::endl;
    start = std::chrono::high_resolution_clock::now();
    GLECOCompressedData<uint64_t> compressed = compressGLECO(data);
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_time = end - start;

    std::cout << "  Compressed in " << std::fixed << std::setprecision(3)
              << compress_time.count() << " seconds" << std::endl;
    printCompressionStats(data, compressed);

    // Step 3: Decompress data using Phase 2 Bucket kernel
    std::cout << "\n[3/4] Decompressing data (Phase 2 Bucket)..." << std::endl;

    std::vector<uint64_t> decompressed(data.size());
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));

    start = std::chrono::high_resolution_clock::now();
    decompressGLECO_Phase2_Bucket<uint64_t>(
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_delta_array_bit_offsets,
        compressed.d_delta_array,
        compressed.num_partitions,
        data.size(),
        d_output
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_time = end - start;

    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_output));

    std::cout << "  Decompressed in " << std::fixed << std::setprecision(3)
              << decompress_time.count() << " seconds" << std::endl;

    // Step 4: Verify correctness
    std::cout << "\n[4/4] Verifying correctness..." << std::endl;
    bool correct = verifyDecompression(data, decompressed);

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Test Result: " << (correct ? "PASS ✓" : "FAIL ✗") << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Performance summary
    if (correct) {
        double throughput = (data.size() * sizeof(uint64_t) / 1e9) / decompress_time.count();
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  Compression time:   " << std::setw(10) << std::fixed
                  << std::setprecision(3) << compress_time.count() << " s" << std::endl;
        std::cout << "  Decompression time: " << std::setw(10) << std::fixed
                  << std::setprecision(3) << decompress_time.count() << " s" << std::endl;
        std::cout << "  Decompression throughput: " << std::setw(10) << std::fixed
                  << std::setprecision(2) << throughput << " GB/s" << std::endl;
    }

    std::cout << std::endl;

    return correct ? 0 : 1;
}
