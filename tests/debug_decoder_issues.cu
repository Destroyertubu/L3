/**
 * Debug decoder issues
 *
 * Analyzes specific failure patterns:
 * 1. AUTO/BRANCHLESS "got 0" errors
 * 2. SEQUENTIAL off-by-one errors
 * 3. INTERLEAVED mismatches
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

template<typename T>
std::vector<T> loadBinaryData(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return {};

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));

    size_t data_bytes = file_size - sizeof(uint64_t);
    size_t expected_with_header = data_bytes / sizeof(T);

    std::vector<T> data;
    if (header_count == expected_with_header) {
        data.resize(header_count);
        file.read(reinterpret_cast<char*>(data.data()), header_count * sizeof(T));
    } else {
        file.seekg(0, std::ios::beg);
        size_t num_elements = file_size / sizeof(T);
        data.resize(num_elements);
        file.read(reinterpret_cast<char*>(data.data()), file_size);
    }

    return data;
}

template<typename T>
void debugPartitionDetails(
    const CompressedDataVertical<T>& compressed,
    int partition_id)
{
    std::cout << "\n=== Partition " << partition_id << " Details ===" << std::endl;

    int32_t start_idx, end_idx, model_type, delta_bits;
    int64_t bit_offset;
    double params[4];

    CUDA_CHECK(cudaMemcpy(&start_idx, compressed.d_start_indices + partition_id, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&end_idx, compressed.d_end_indices + partition_id, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&model_type, compressed.d_model_types + partition_id, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&delta_bits, compressed.d_delta_bits + partition_id, sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&bit_offset, compressed.d_delta_array_bit_offsets + partition_id, sizeof(int64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(params, compressed.d_model_params + partition_id * 4, 4 * sizeof(double), cudaMemcpyDeviceToHost));

    std::cout << "  start_idx: " << start_idx << std::endl;
    std::cout << "  end_idx: " << end_idx << std::endl;
    std::cout << "  size: " << (end_idx - start_idx) << std::endl;
    std::cout << "  model_type: " << model_type << " (";
    switch (model_type) {
        case MODEL_CONSTANT: std::cout << "CONSTANT"; break;
        case MODEL_LINEAR: std::cout << "LINEAR"; break;
        case MODEL_POLYNOMIAL2: std::cout << "POLY2"; break;
        case MODEL_POLYNOMIAL3: std::cout << "POLY3"; break;
        case MODEL_FOR_BITPACK: std::cout << "FOR_BITPACK"; break;
        default: std::cout << "UNKNOWN"; break;
    }
    std::cout << ")" << std::endl;
    std::cout << "  delta_bits: " << delta_bits << std::endl;
    std::cout << "  bit_offset: " << bit_offset << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  params[0] (theta0): " << params[0] << std::endl;
    std::cout << "  params[1] (theta1): " << params[1] << std::endl;
    std::cout << "  params[2]: " << params[2] << std::endl;
    std::cout << "  params[3]: " << params[3] << std::endl;

    // For FOR_BITPACK, show the base value interpretation
    if (model_type == MODEL_FOR_BITPACK && sizeof(T) == 8) {
        // Use memcpy for host-side bit reinterpretation
        long long base_as_ll;
        memcpy(&base_as_ll, &params[0], sizeof(long long));
        std::cout << "  FOR base (as int64): " << base_as_ll << std::endl;
    }
}

template<typename T>
void testSingleDataset(const std::string& path, const std::string& name, size_t max_elements = 10000000) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Testing: " << name << std::endl;
    std::cout << "========================================" << std::endl;

    auto data = loadBinaryData<T>(path);
    if (data.empty()) {
        std::cout << "SKIP: Could not load " << path << std::endl;
        return;
    }

    if (data.size() > max_elements) {
        std::cout << "Limiting to " << max_elements << " elements (was " << data.size() << ")" << std::endl;
        data.resize(max_elements);
    }

    std::cout << "Data size: " << data.size() << " elements" << std::endl;
    std::cout << "First few values: ";
    for (int i = 0; i < std::min(5, (int)data.size()); i++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    // Create partitions
    auto partitions = Vertical_encoder::createFixedPartitions<T>(data.size(), 2048);
    std::cout << "Created " << partitions.size() << " partitions" << std::endl;

    // Encode
    VerticalConfig fl_config;
    fl_config.enable_adaptive_selection = true;
    fl_config.enable_interleaved = false;

    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config, 0);
    std::cout << "Compression done. Total partitions: " << compressed.num_partitions << std::endl;

    // Show first few partition details
    for (int p = 0; p < std::min(3, compressed.num_partitions); p++) {
        debugPartitionDetails(compressed, p);
    }

    // Allocate output
    size_t data_bytes = data.size() * sizeof(T);
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    // Test each decompression mode
    std::vector<DecompressMode> modes = {
        DecompressMode::AUTO,
        DecompressMode::BRANCHLESS,
        DecompressMode::SEQUENTIAL
    };
    std::vector<std::string> mode_names = {"AUTO", "BRANCHLESS", "SEQUENTIAL"};

    for (int m = 0; m < modes.size(); m++) {
        CUDA_CHECK(cudaMemset(d_output, 0, data_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());

        Vertical_decoder::decompressAll<T>(compressed, d_output, modes[m], 0);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<T> decompressed(data.size());
        CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

        int errors = 0;
        int first_error_idx = -1;
        T first_error_expected = 0, first_error_got = 0;

        for (size_t i = 0; i < data.size() && errors < 10; i++) {
            if (data[i] != decompressed[i]) {
                if (first_error_idx < 0) {
                    first_error_idx = i;
                    first_error_expected = data[i];
                    first_error_got = decompressed[i];
                }
                errors++;
            }
        }

        std::cout << "\n" << mode_names[m] << " mode: ";
        if (errors == 0) {
            std::cout << "PASS" << std::endl;
        } else {
            std::cout << "FAIL (" << errors << "+ errors)" << std::endl;
            std::cout << "  First error at index " << first_error_idx << ":" << std::endl;
            std::cout << "    Expected: " << first_error_expected << std::endl;
            std::cout << "    Got:      " << first_error_got << std::endl;

            // Find which partition this index belongs to
            int pid = first_error_idx / 2048;  // Approximation for fixed partitions
            if (pid < compressed.num_partitions) {
                debugPartitionDetails(compressed, pid);

                // Check adjacent partitions too
                if (pid > 0) {
                    int32_t prev_end;
                    CUDA_CHECK(cudaMemcpy(&prev_end, compressed.d_end_indices + pid - 1, sizeof(int32_t), cudaMemcpyDeviceToHost));
                    std::cout << "  Previous partition ends at: " << prev_end << std::endl;
                }
            }

            // Show more context around the error
            std::cout << "  Values around error (original vs decompressed):" << std::endl;
            int start = std::max(0, first_error_idx - 3);
            int end = std::min((int)data.size(), first_error_idx + 4);
            for (int i = start; i < end; i++) {
                std::cout << "    [" << i << "]: " << data[i] << " vs " << decompressed[i];
                if (data[i] != decompressed[i]) std::cout << " <-- MISMATCH";
                std::cout << std::endl;
            }
        }
    }

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/data/sosd";

    if (argc > 1) data_dir = argv[1];

    std::cout << "=== L3 Decoder Debug Test ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;

    // Test the problematic datasets

    // 1. linear_200M - AUTO/BRANCHLESS fail with "got 0"
    testSingleDataset<uint64_t>(data_dir + "/1-linear_200M_uint64.bin", "linear_200M (uint64)", 1000000);

    // 2. normal_200M - SEQUENTIAL fails with large diff
    testSingleDataset<uint64_t>(data_dir + "/2-normal_200M_uint64.bin", "normal_200M (uint64)", 1000000);

    // 3. wiki_200M - All modes fail
    testSingleDataset<uint64_t>(data_dir + "/7-wiki_200M_uint64.bin", "wiki_200M (uint64)", 1000000);

    // 4. books_200M (uint32) - SEQUENTIAL off-by-one
    testSingleDataset<uint32_t>(data_dir + "/5-books_200M_uint32.bin", "books_200M (uint32)", 1000000);

    std::cout << "\n=== Debug Test Complete ===" << std::endl;

    return 0;
}
