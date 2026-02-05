#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include "../src/kernels/compression/adaptive_selector.cuh"
#include "../include/L3_format.hpp"
#include "../include/L3_Vertical_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

int main() {
    std::string filepath = "data/sosd/2-normal_200M_uint64.bin";
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Cannot open\n"; return 1; }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Test full dataset
    size_t n = file_size / sizeof(uint64_t);
    std::vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    
    std::cout << "Testing " << n << " elements\n";
    
    // Configure Vertical with LINEAR model only
    VerticalConfig fl_config;
    fl_config.partition_size_hint = 4096;
    fl_config.enable_interleaved = true;
    fl_config.enable_dual_format = true;
    fl_config.enable_adaptive_selection = false;
    fl_config.fixed_model_type = MODEL_LINEAR;
    
    // Create partitions
    auto partitions = Vertical_encoder::createFixedPartitions<uint64_t>(n, 4096);
    std::cout << "Created " << partitions.size() << " partitions\n";
    
    // Compress
    auto compressed = Vertical_encoder::encodeVertical<uint64_t>(data, partitions, fl_config);
    std::cout << "Compressed: " << compressed.sequential_delta_words << " words (seq), "
              << compressed.interleaved_delta_words << " words (int)\n";
    
    // Copy model info to host to inspect
    std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    
    int max_bits = 0;
    for (int i = 0; i < compressed.num_partitions; i++) {
        if (h_delta_bits[i] > max_bits) max_bits = h_delta_bits[i];
    }
    std::cout << "Max delta_bits: " << max_bits << "\n";
    
    // Decompress using BRANCHLESS
    uint64_t* d_output;
    cudaMalloc(&d_output, n * sizeof(uint64_t));
    cudaMemset(d_output, 0, n * sizeof(uint64_t));
    
    std::cout << "\nTesting BRANCHLESS mode...\n";
    Vertical_decoder::decompressAll<uint64_t>(compressed, d_output, DecompressMode::BRANCHLESS);
    cudaDeviceSynchronize();
    
    std::vector<uint64_t> decompressed(n);
    cudaMemcpy(decompressed.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    // Verify
    int errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != decompressed[i]) {
            if (errors < 10) {
                int pid = i / 4096;
                std::cout << "Error at " << i << " (partition " << pid << ", bits=" << h_delta_bits[pid] 
                          << "): expected " << data[i] 
                          << " got " << decompressed[i] << "\n";
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "BRANCHLESS: SUCCESS - All values match!\n";
    } else {
        std::cout << "BRANCHLESS: FAILED - " << errors << " errors\n";
    }
    
    // Decompress using INTERLEAVED
    cudaMemset(d_output, 0, n * sizeof(uint64_t));
    
    std::cout << "\nTesting INTERLEAVED mode...\n";
    Vertical_decoder::decompressAll<uint64_t>(compressed, d_output, DecompressMode::INTERLEAVED);
    cudaDeviceSynchronize();
    
    cudaMemcpy(decompressed.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    errors = 0;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != decompressed[i]) {
            if (errors < 10) {
                int pid = i / 4096;
                std::cout << "Error at " << i << " (partition " << pid << ", bits=" << h_delta_bits[pid]
                          << "): expected " << data[i] 
                          << " got " << decompressed[i] << "\n";
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        std::cout << "INTERLEAVED: SUCCESS - All values match!\n";
    } else {
        std::cout << "INTERLEAVED: FAILED - " << errors << " errors\n";
    }
    
    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);
    return 0;
}
