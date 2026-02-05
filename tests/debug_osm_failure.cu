/**
 * Debug osm_800M failure
 * Find which partition and model type causes the mismatch
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/compression/adaptive_selector.cuh"
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
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

int main() {
    std::string path = "data/sosd/8-osm_cellids_800M_uint64.bin";

    std::cout << "Loading osm_800M data..." << std::endl;
    std::vector<uint64_t> data = loadBinaryData<uint64_t>(path);
    std::cout << "Elements: " << data.size() << std::endl;

    // Check the value range
    uint64_t min_val = data[0], max_val = data[0];
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
    }
    std::cout << "Value range: [" << min_val << ", " << max_val << "]" << std::endl;
    std::cout << "Max value bits: " << (64 - __builtin_clzll(max_val)) << std::endl;
    std::cout << "Double precision limit (2^53): " << (1ULL << 53) << std::endl;
    std::cout << "Max value > 2^53: " << (max_val > (1ULL << 53) ? "YES (precision issue!)" : "NO") << std::endl;

    // Check around the failure index
    size_t fail_idx = 637472768;
    std::cout << "\nValues around failure index " << fail_idx << ":" << std::endl;
    for (size_t i = fail_idx - 5; i <= fail_idx + 5 && i < data.size(); i++) {
        std::cout << "  data[" << i << "] = " << data[i] << std::endl;
    }

    // Allocate and copy to GPU
    double data_bytes = data.size() * sizeof(uint64_t);
    uint64_t* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // Create partitions
    CostOptimalConfig config = CostOptimalConfig::balanced();
    int num_partitions = 0;
    auto partitions = createPartitionsCostOptimal<uint64_t>(data, config, &num_partitions, 0);

    std::cout << "\nPartitions: " << num_partitions << std::endl;

    // Find which partition contains the failure index
    int fail_partition = -1;
    for (int p = 0; p < num_partitions; p++) {
        if (partitions[p].start_idx <= (int)fail_idx && (int)fail_idx < partitions[p].end_idx) {
            fail_partition = p;
            break;
        }
    }

    std::cout << "Failure in partition: " << fail_partition << std::endl;
    if (fail_partition >= 0) {
        std::cout << "  start_idx: " << partitions[fail_partition].start_idx << std::endl;
        std::cout << "  end_idx: " << partitions[fail_partition].end_idx << std::endl;
        std::cout << "  size: " << (partitions[fail_partition].end_idx - partitions[fail_partition].start_idx) << std::endl;
    }

    // Use GPU polynomial selector
    int32_t* d_starts;
    int32_t* d_ends;
    adaptive_selector::ModelDecision<uint64_t>* d_decisions;

    CUDA_CHECK(cudaMalloc(&d_starts, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ends, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>)));

    std::vector<int32_t> h_starts(num_partitions), h_ends(num_partitions);
    for (int p = 0; p < num_partitions; p++) {
        h_starts[p] = partitions[p].start_idx;
        h_ends[p] = partitions[p].end_idx;
    }
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ends, h_ends.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));

    adaptive_selector::launchAdaptiveSelectorFullPolynomial<uint64_t>(
        d_data, d_starts, d_ends, num_partitions, d_decisions, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<adaptive_selector::ModelDecision<uint64_t>> decisions(num_partitions);
    CUDA_CHECK(cudaMemcpy(decisions.data(), d_decisions,
                          num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>),
                          cudaMemcpyDeviceToHost));

    // Count model types and find POLY2/POLY3 partitions
    int linear_count = 0, poly2_count = 0, poly3_count = 0, for_count = 0;
    std::vector<int> poly2_partitions, poly3_partitions;

    for (int p = 0; p < num_partitions; p++) {
        switch (decisions[p].model_type) {
            case MODEL_LINEAR: linear_count++; break;
            case MODEL_POLYNOMIAL2:
                poly2_count++;
                poly2_partitions.push_back(p);
                break;
            case MODEL_POLYNOMIAL3:
                poly3_count++;
                poly3_partitions.push_back(p);
                break;
            case MODEL_FOR_BITPACK: for_count++; break;
        }
    }

    std::cout << "\nModel selection:" << std::endl;
    std::cout << "  LINEAR: " << linear_count << std::endl;
    std::cout << "  POLY2: " << poly2_count << std::endl;
    std::cout << "  POLY3: " << poly3_count << std::endl;
    std::cout << "  FOR: " << for_count << std::endl;

    if (fail_partition >= 0) {
        auto& dec = decisions[fail_partition];
        const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR"};
        std::cout << "\nFailing partition model: " << model_names[dec.model_type] << std::endl;
        std::cout << "  params: [" << dec.params[0] << ", " << dec.params[1]
                  << ", " << dec.params[2] << ", " << dec.params[3] << "]" << std::endl;
        std::cout << "  delta_bits: " << dec.delta_bits << std::endl;
        std::cout << "  min_val: " << dec.min_val << std::endl;
        std::cout << "  max_val: " << dec.max_val << std::endl;

        // Manually verify the prediction for the failing element
        int local_idx = fail_idx - partitions[fail_partition].start_idx;
        double x = static_cast<double>(local_idx);
        double predicted;

        switch (dec.model_type) {
            case MODEL_LINEAR:
                predicted = dec.params[0] + dec.params[1] * x;
                break;
            case MODEL_POLYNOMIAL2:
                predicted = dec.params[0] + x * (dec.params[1] + x * dec.params[2]);
                break;
            case MODEL_POLYNOMIAL3:
                predicted = dec.params[0] + x * (dec.params[1] + x * (dec.params[2] + x * dec.params[3]));
                break;
            default:
                predicted = dec.min_val;
                break;
        }

        uint64_t actual_val = data[fail_idx];
        int64_t pred_rounded = static_cast<int64_t>(llrint(predicted));
        int64_t residual = static_cast<int64_t>(actual_val) - pred_rounded;

        std::cout << "\nManual verification at fail_idx=" << fail_idx << ":" << std::endl;
        std::cout << "  local_idx: " << local_idx << std::endl;
        std::cout << "  actual value: " << actual_val << std::endl;
        std::cout << "  predicted (double): " << std::fixed << predicted << std::endl;
        std::cout << "  predicted (rounded): " << pred_rounded << std::endl;
        std::cout << "  residual: " << residual << std::endl;

        // Check if value exceeds double precision
        std::cout << "\n  actual_val > 2^53: " << (actual_val > (1ULL << 53) ? "YES" : "NO") << std::endl;
        std::cout << "  predicted > 2^53: " << (predicted > (double)(1ULL << 53) ? "YES" : "NO") << std::endl;
    }

    // Also check a few POLY2/POLY3 partitions
    std::cout << "\n=== Checking POLY2 partitions ===" << std::endl;
    for (int i = 0; i < std::min(5, (int)poly2_partitions.size()); i++) {
        int p = poly2_partitions[i];
        auto& dec = decisions[p];
        std::cout << "Partition " << p << ": start=" << partitions[p].start_idx
                  << ", end=" << partitions[p].end_idx
                  << ", params=[" << dec.params[0] << "," << dec.params[1] << "," << dec.params[2] << "]"
                  << ", bits=" << dec.delta_bits << std::endl;
    }

    if (!poly3_partitions.empty()) {
        std::cout << "\n=== Checking POLY3 partitions ===" << std::endl;
        for (int i = 0; i < std::min(5, (int)poly3_partitions.size()); i++) {
            int p = poly3_partitions[i];
            auto& dec = decisions[p];
            std::cout << "Partition " << p << ": start=" << partitions[p].start_idx
                      << ", end=" << partitions[p].end_idx
                      << ", params=[" << dec.params[0] << "," << dec.params[1] << ","
                      << dec.params[2] << "," << dec.params[3] << "]"
                      << ", bits=" << dec.delta_bits << std::endl;
        }
    }

    cudaFree(d_data);
    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_decisions);

    return 0;
}
