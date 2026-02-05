/**
 * Quick test for L3 POLY2 decompression on fb dataset
 */
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "L3_opt.h"
#include "sosd_loader.h"
#include "encoder_cost_optimal_gpu_merge_v2.cuh"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

// External declaration for decoder
template<typename T>
void launchDecompressWarpOpt(const CompressedDataL3<T>*, T*, cudaStream_t);

int main() {
    std::cout << "=== L3 POLY2 Debug Test ===" << std::endl;

    // Load fb dataset
    std::string path = "data/sosd/6-fb_200M_uint64.bin";
    std::vector<uint64_t> data;

    if (!loadSOSDDataset(path, data)) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Generate partitions with V2 partitioner
    CostOptimalConfig config;
    config.target_partition_size = 4096;
    config.analysis_block_size = 2048;
    config.min_partition_size = 256;
    config.max_partition_size = 8192;
    config.breakpoint_threshold = 2;
    config.merge_benefit_threshold = 0.05f;
    config.max_merge_rounds = 4;
    config.enable_merging = true;
    config.enable_polynomial_models = true;
    config.polynomial_min_size = 10;
    config.cubic_min_size = 20;
    config.polynomial_cost_threshold = 0.95f;

    GPUCostOptimalPartitionerV2<uint64_t> partitioner(data, config, 0);
    std::vector<PartitionInfo> partitions = partitioner.partition();

    // Count model types
    int linear = 0, poly2 = 0, poly3 = 0;
    for (const auto& p : partitions) {
        if (p.model_type == MODEL_LINEAR || p.model_type == MODEL_CONSTANT) linear++;
        else if (p.model_type == MODEL_POLYNOMIAL2) poly2++;
        else if (p.model_type == MODEL_POLYNOMIAL3) poly3++;
    }
    std::cout << "Partitions: " << partitions.size()
              << " (LINEAR=" << linear << ", POLY2=" << poly2 << ", POLY3=" << poly3 << ")" << std::endl;

    // Create L3 encoder
    L3Encoder<uint64_t> encoder;
    encoder.setPartitions(partitions);

    // Copy data to device
    uint64_t* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, data.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_input, data.data(), data.size() * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Compress
    auto* compressed = encoder.encode(d_input, data.size());
    std::cout << "Compressed: " << compressed->num_partitions << " partitions, "
              << compressed->total_values << " total values" << std::endl;

    // Decompress
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_output, 0, data.size() * sizeof(uint64_t)));

    launchDecompressWarpOpt<uint64_t>(compressed, d_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back and verify
    std::vector<uint64_t> output(data.size());
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // Check errors - detailed diagnostics
    int errors = 0;
    int first_error_idx = -1;
    int first_poly2_error = -1;

    for (size_t i = 0; i < data.size(); i++) {
        if (output[i] != data[i]) {
            if (errors < 10) {
                std::cout << "Error at index " << i << ": got " << output[i]
                          << ", expected " << data[i] << std::endl;

                // Find which partition this belongs to
                for (size_t p = 0; p < partitions.size(); p++) {
                    if (i >= partitions[p].start_idx && i < partitions[p].end_idx) {
                        std::cout << "  Partition " << p << ": type=" << partitions[p].model_type
                                  << " start=" << partitions[p].start_idx
                                  << " end=" << partitions[p].end_idx
                                  << " delta_bits=" << partitions[p].delta_bits << std::endl;
                        if (partitions[p].model_type == MODEL_POLYNOMIAL2 && first_poly2_error < 0) {
                            first_poly2_error = p;
                        }
                        break;
                    }
                }
            }
            if (first_error_idx < 0) first_error_idx = i;
            errors++;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total errors: " << errors << " / " << data.size() << std::endl;
    std::cout << "Result: " << (errors == 0 ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    encoder.freeCompressedData(compressed);

    return errors > 0 ? 1 : 0;
}
