/**
 * Debug: Full compression pipeline to find why ratio is 2.13x instead of 7.57x
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <iomanip>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

// Load raw binary dataset
std::vector<uint64_t> loadRawDataset(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t count = file_size / sizeof(uint64_t);
    if (max_elements > 0 && count > max_elements) {
        count = max_elements;
    }

    std::vector<uint64_t> data(count);
    file.read(reinterpret_cast<char*>(data.data()), count * sizeof(uint64_t));

    return data;
}

int main(int argc, char** argv) {
    std::string filename = "data/sosd/1-linear_200M_uint64.bin";
    size_t max_elements = 10000000;  // 10M for testing

    std::cout << "Debug: Full Compression Pipeline" << std::endl;
    std::cout << "=================================" << std::endl;

    auto data = loadRawDataset(filename, max_elements);
    if (data.empty()) {
        std::cerr << "Failed to load dataset" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << data.size() << " elements" << std::endl;

    // Configure encoder
    VerticalConfig config;
    config.partition_size_hint = 2048;
    config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
    config.enable_adaptive_selection = true;
    config.cost_min_partition_size = 256;
    config.cost_max_partition_size = 8192;
    config.cost_target_partition_size = 2048;

    std::cout << "\nRunning encodeVerticalGPU..." << std::endl;

    auto compressed = Vertical_encoder::encodeVerticalGPU<uint64_t>(data, 2048, config);

    std::cout << "\n=== Compression Results ===" << std::endl;
    std::cout << "num_partitions: " << compressed.num_partitions << std::endl;
    std::cout << "total_values: " << compressed.total_values << std::endl;
    std::cout << "interleaved_delta_words: " << compressed.interleaved_delta_words << std::endl;

    // Calculate sizes
    int64_t original_size = data.size() * sizeof(uint64_t);
    int64_t metadata_size = compressed.num_partitions * (
        sizeof(int32_t) +     // start_indices
        sizeof(int32_t) +     // end_indices
        sizeof(int32_t) +     // model_types
        sizeof(double) * 4 +  // model_params (4 doubles per partition)
        sizeof(int32_t) +     // delta_bits
        sizeof(int32_t) +     // num_mini_vectors
        sizeof(int32_t) +     // tail_sizes
        sizeof(int64_t)       // interleaved_offsets
    );
    int64_t delta_array_size = compressed.interleaved_delta_words * sizeof(uint32_t);
    int64_t compressed_size = metadata_size + delta_array_size;

    std::cout << "\n=== Size Analysis ===" << std::endl;
    std::cout << "Original size: " << original_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Metadata size: " << metadata_size / 1024.0 << " KB" << std::endl;
    std::cout << "Delta array size: " << delta_array_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Total compressed: " << compressed_size / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << (double)original_size / compressed_size << "x" << std::endl;

    // Download and analyze partition metadata
    std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    std::vector<int32_t> h_start_indices(compressed.num_partitions);
    std::vector<int32_t> h_end_indices(compressed.num_partitions);

    cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_model_types.data(), compressed.d_model_types,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_start_indices.data(), compressed.d_start_indices,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end_indices.data(), compressed.d_end_indices,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Count models and analyze delta_bits
    int linear_count = 0, poly2_count = 0, poly3_count = 0, for_count = 0;
    int64_t total_bits = 0;
    int max_bits = 0, min_bits = 64;

    for (int i = 0; i < compressed.num_partitions; i++) {
        int size = h_end_indices[i] - h_start_indices[i];
        total_bits += (int64_t)size * h_delta_bits[i];
        max_bits = std::max(max_bits, h_delta_bits[i]);
        if (h_delta_bits[i] > 0) min_bits = std::min(min_bits, h_delta_bits[i]);

        if (h_model_types[i] == MODEL_LINEAR) linear_count++;
        else if (h_model_types[i] == MODEL_POLYNOMIAL2) poly2_count++;
        else if (h_model_types[i] == MODEL_POLYNOMIAL3) poly3_count++;
        else if (h_model_types[i] == MODEL_FOR_BITPACK) for_count++;
    }

    std::cout << "\n=== Model Distribution ===" << std::endl;
    std::cout << "LINEAR: " << linear_count << std::endl;
    std::cout << "POLY2: " << poly2_count << std::endl;
    std::cout << "POLY3: " << poly3_count << std::endl;
    std::cout << "FOR_BITPACK: " << for_count << std::endl;

    std::cout << "\n=== Delta Bits Analysis ===" << std::endl;
    std::cout << "Total bits: " << total_bits << std::endl;
    std::cout << "Expected delta array bytes: " << (total_bits + 7) / 8 << std::endl;
    std::cout << "Actual delta array bytes: " << delta_array_size << std::endl;
    std::cout << "Min delta_bits: " << (min_bits == 64 ? 0 : min_bits) << std::endl;
    std::cout << "Max delta_bits: " << max_bits << std::endl;

    // Show first 10 partitions
    std::cout << "\n=== First 10 Partitions ===" << std::endl;
    const char* model_names[] = {"CONST", "LINEAR", "POLY2", "POLY3", "FOR", "DIRECT"};
    for (int i = 0; i < std::min(10, compressed.num_partitions); i++) {
        std::cout << "Partition " << i << ": [" << h_start_indices[i] << ", " << h_end_indices[i] << ")"
                  << " model=" << model_names[h_model_types[i]]
                  << " delta_bits=" << h_delta_bits[i]
                  << std::endl;
    }

    // Count partitions with 0 delta_bits
    int zero_bits_count = 0;
    int nonzero_bits_count = 0;
    for (int i = 0; i < compressed.num_partitions; i++) {
        if (h_delta_bits[i] == 0) zero_bits_count++;
        else nonzero_bits_count++;
    }
    std::cout << "\nPartitions with delta_bits=0: " << zero_bits_count << std::endl;
    std::cout << "Partitions with delta_bits>0: " << nonzero_bits_count << std::endl;

    // Show some non-zero delta_bits partitions
    std::cout << "\n=== First 10 Non-Zero Delta Bits Partitions ===" << std::endl;
    int shown = 0;
    for (int i = 0; i < compressed.num_partitions && shown < 10; i++) {
        if (h_delta_bits[i] > 0) {
            int size = h_end_indices[i] - h_start_indices[i];
            std::cout << "Partition " << i << ": [" << h_start_indices[i] << ", " << h_end_indices[i] << ")"
                      << " size=" << size
                      << " model=" << model_names[h_model_types[i]]
                      << " delta_bits=" << h_delta_bits[i]
                      << std::endl;
            shown++;
        }
    }

    // Download model params
    std::vector<double> h_model_params(compressed.num_partitions * 4);
    cudaMemcpy(h_model_params.data(), compressed.d_model_params,
               compressed.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);

    // Check if data is truly linear in the non-zero region
    std::cout << "\n=== Data Linearity Check at Boundary ===" << std::endl;
    // Check around index 24576 where delta_bits changes from 0 to 2
    std::cout << "Data around index 24576:" << std::endl;
    for (int i = 24570; i < 24585 && i < (int)data.size(); i++) {
        int64_t diff = (i > 0) ? (int64_t)(data[i] - data[i-1]) : 0;
        std::cout << "  data[" << i << "] = " << data[i] << " (diff=" << diff << ")" << std::endl;
    }

    // Check first few values
    std::cout << "\nExpected diff (theta1): ";
    if (data.size() > 1) {
        std::cout << (data[1] - data[0]) << std::endl;
    }

    // Check partition 12 (the first one with delta_bits=2)
    std::cout << "\n=== Partition 12 Analysis ===" << std::endl;
    int pid = 12;
    int p_start = h_start_indices[pid];
    int p_end = h_end_indices[pid];
    double theta0 = h_model_params[pid * 4 + 0];
    double theta1 = h_model_params[pid * 4 + 1];

    std::cout << "Partition range: [" << p_start << ", " << p_end << ")" << std::endl;
    std::cout << "theta0 = " << std::setprecision(20) << theta0 << std::endl;
    std::cout << "theta1 = " << std::setprecision(20) << theta1 << std::endl;

    // Manually compute max error
    long long max_err = 0;
    for (int i = p_start; i < p_end && i < (int)data.size(); i++) {
        int local_idx = i - p_start;
        double pred = theta0 + theta1 * static_cast<double>(local_idx);
        int64_t pred_int = static_cast<int64_t>(std::llrint(pred));
        int64_t actual = static_cast<int64_t>(data[i]);
        int64_t err = std::abs(actual - pred_int);
        if (err > max_err) {
            max_err = err;
            if (local_idx < 10 || err > 1) {  // Show first few or any with error > 1
                std::cout << "  local_idx=" << local_idx << " actual=" << actual
                          << " pred=" << pred_int << " err=" << err << std::endl;
            }
        }
    }
    std::cout << "Max error in partition 12: " << max_err << std::endl;
    std::cout << "This requires " << (max_err > 0 ? (64 - __builtin_clzll(max_err)) + 1 : 0) << " bits" << std::endl;

    // Free compressed data
    Vertical_encoder::freeCompressedData(compressed);

    return 0;
}
