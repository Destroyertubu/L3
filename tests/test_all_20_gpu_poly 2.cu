/**
 * L3 Comprehensive Benchmark - All 20 SOSD Datasets
 * Using GPU Full Polynomial Model Selector (LINEAR, POLY2, POLY3, FOR)
 *
 * This version uses the GPU-accelerated adaptive selector that can choose
 * from all polynomial models based on cost optimization.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <map>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// GPU Full Polynomial Selector
#include "../src/kernels/compression/adaptive_selector.cuh"

// V3 Cost-Optimal encoder (for partitioning only)
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

// Vertical encoder
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Vertical decoder
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return result; \
        } \
    } while(0)

struct DatasetInfo {
    int id;
    std::string name;
    std::string filename;
    std::string type;
};

struct ModelStats {
    int constant_count = 0;
    int linear_count = 0;
    int poly2_count = 0;
    int poly3_count = 0;
    int direct_copy_count = 0;

    double constant_pct() const { return total() > 0 ? 100.0 * constant_count / total() : 0; }
    double linear_pct() const { return total() > 0 ? 100.0 * linear_count / total() : 0; }
    double poly2_pct() const { return total() > 0 ? 100.0 * poly2_count / total() : 0; }
    double poly3_pct() const { return total() > 0 ? 100.0 * poly3_count / total() : 0; }
    double direct_copy_pct() const { return total() > 0 ? 100.0 * direct_copy_count / total() : 0; }
    int total() const { return constant_count + linear_count + poly2_count + poly3_count + direct_copy_count; }
};

struct DetailedResult {
    int id;
    std::string name;
    std::string type;

    // Size metrics
    size_t num_elements;
    double original_size_mb;
    double compressed_size_mb;
    double metadata_size_mb;
    double delta_array_size_mb;
    double compression_ratio;

    // Partition metrics
    int num_partitions;
    double avg_partition_size;
    double avg_delta_bits;
    double min_delta_bits;
    double max_delta_bits;

    // Model selection
    ModelStats model_stats;

    // Timing (milliseconds)
    double partition_time_ms;
    double selector_time_ms;  // GPU model selection time
    double pack_time_ms;
    double total_compress_time_ms;
    double decompress_time_ms;

    // Throughput (GB/s)
    double compress_throughput_gbps;
    double decompress_throughput_gbps;

    // Correctness
    bool correctness;
    std::string error_msg;
};

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
DetailedResult runDetailedTest(const DatasetInfo& dataset, const std::string& data_dir) {
    DetailedResult result;
    result.id = dataset.id;
    result.name = dataset.name;
    result.type = dataset.type;
    result.correctness = false;
    result.error_msg = "";

    std::string path = data_dir + "/" + dataset.filename;
    std::cout << "\n=== [" << dataset.id << "/20] Testing: " << dataset.name << " ===" << std::endl;

    // Load data
    std::vector<T> data = loadBinaryData<T>(path);
    if (data.empty()) {
        result.error_msg = "File not found";
        std::cout << "SKIP: " << result.error_msg << std::endl;
        return result;
    }

    result.num_elements = data.size();
    result.original_size_mb = static_cast<double>(data.size() * sizeof(T)) / (1024.0 * 1024.0);
    double data_bytes = data.size() * sizeof(T);

    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Original size: " << std::fixed << std::setprecision(2) << result.original_size_mb << " MB" << std::endl;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Copy data to device
    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), data_bytes, cudaMemcpyHostToDevice));

    // ========== Phase 1: Partitioning (using Cost-Optimal) ==========
    CostOptimalConfig config = CostOptimalConfig::balanced();
    int num_partitions = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    auto partitions = createPartitionsCostOptimal<T>(data, config, &num_partitions, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float partition_ms;
    CUDA_CHECK(cudaEventElapsedTime(&partition_ms, start, stop));
    result.partition_time_ms = partition_ms;

    if (partitions.empty()) {
        result.error_msg = "Partitioning failed";
        std::cout << "FAIL: " << result.error_msg << std::endl;
        cudaFree(d_data);
        return result;
    }

    result.num_partitions = num_partitions;
    result.avg_partition_size = static_cast<double>(data.size()) / num_partitions;

    // ========== Phase 2: GPU Model Selection (Full Polynomial) ==========
    // Prepare partition indices on device
    int32_t* d_starts;
    int32_t* d_ends;
    adaptive_selector::ModelDecision<T>* d_decisions;

    CUDA_CHECK(cudaMalloc(&d_starts, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_ends, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<T>)));

    std::vector<int32_t> h_starts(num_partitions), h_ends(num_partitions);
    for (int p = 0; p < num_partitions; p++) {
        h_starts[p] = partitions[p].start_idx;
        h_ends[p] = partitions[p].end_idx;
    }
    CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ends, h_ends.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    // Launch GPU polynomial selector
    adaptive_selector::launchAdaptiveSelectorFullPolynomial<T>(
        d_data, d_starts, d_ends, num_partitions, d_decisions, 0);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float selector_ms;
    CUDA_CHECK(cudaEventElapsedTime(&selector_ms, start, stop));
    result.selector_time_ms = selector_ms;

    // Copy decisions back and update partitions
    std::vector<adaptive_selector::ModelDecision<T>> decisions(num_partitions);
    CUDA_CHECK(cudaMemcpy(decisions.data(), d_decisions,
                          num_partitions * sizeof(adaptive_selector::ModelDecision<T>),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_decisions);

    // Update partitions with GPU selector results
    double total_delta_bits = 0;
    result.min_delta_bits = 65;
    result.max_delta_bits = 0;

    for (int p = 0; p < num_partitions; p++) {
        partitions[p].model_type = decisions[p].model_type;
        for (int j = 0; j < 4; j++) {
            partitions[p].model_params[j] = decisions[p].params[j];
        }
        partitions[p].delta_bits = decisions[p].delta_bits;
        partitions[p].error_bound = 0;  // Will be recomputed if needed

        total_delta_bits += decisions[p].delta_bits;
        if (decisions[p].delta_bits < result.min_delta_bits) result.min_delta_bits = decisions[p].delta_bits;
        if (decisions[p].delta_bits > result.max_delta_bits) result.max_delta_bits = decisions[p].delta_bits;

        // Count model selection
        switch (decisions[p].model_type) {
            case MODEL_CONSTANT: result.model_stats.constant_count++; break;
            case MODEL_LINEAR: result.model_stats.linear_count++; break;
            case MODEL_POLYNOMIAL2: result.model_stats.poly2_count++; break;
            case MODEL_POLYNOMIAL3: result.model_stats.poly3_count++; break;
            case MODEL_FOR_BITPACK:
                result.model_stats.direct_copy_count++; break;
            default: break;
        }
    }
    result.avg_delta_bits = total_delta_bits / num_partitions;

    // ========== Phase 3: Build Compressed Structure ==========
    CompressedDataVertical<T> compressed;
    compressed.num_partitions = num_partitions;
    compressed.total_values = data.size();

    CUDA_CHECK(cudaMalloc(&compressed.d_start_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_end_indices, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_types, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_model_params, num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_bits, num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&compressed.d_error_bounds, num_partitions * sizeof(int64_t)));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_type(num_partitions), h_delta_bits(num_partitions);
    std::vector<double> h_model_params(num_partitions * 4);
    std::vector<int64_t> h_bit_offsets(num_partitions), h_error_bounds(num_partitions);

    int64_t total_bits = 0;
    for (int p = 0; p < num_partitions; p++) {
        h_start[p] = partitions[p].start_idx;
        h_end[p] = partitions[p].end_idx;
        h_model_type[p] = partitions[p].model_type;
        h_delta_bits[p] = partitions[p].delta_bits;
        h_bit_offsets[p] = total_bits;
        h_error_bounds[p] = partitions[p].error_bound;
        for (int j = 0; j < 4; j++) {
            h_model_params[p * 4 + j] = partitions[p].model_params[j];
        }
        total_bits += static_cast<int64_t>(h_end[p] - h_start[p]) * h_delta_bits[p];
    }

    CUDA_CHECK(cudaMemcpy(compressed.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_types, h_model_type.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_model_params, h_model_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(compressed.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice));

    int64_t delta_words = (total_bits + 31) / 32 + 4;
    compressed.sequential_delta_words = delta_words;
    CUDA_CHECK(cudaMalloc(&compressed.d_sequential_deltas, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(compressed.d_sequential_deltas, 0, delta_words * sizeof(uint32_t)));

    // ========== Phase 4: Delta Packing ==========
    int blocks = std::min((int)((data.size() + 255) / 256), 65535);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    Vertical_encoder::packDeltasSequentialBranchless<T><<<blocks, 256>>>(
        d_data,
        compressed.d_start_indices,
        compressed.d_end_indices,
        compressed.d_model_types,
        compressed.d_model_params,
        compressed.d_delta_bits,
        compressed.d_delta_array_bit_offsets,
        num_partitions,
        compressed.d_sequential_deltas
    );

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float pack_ms;
    CUDA_CHECK(cudaEventElapsedTime(&pack_ms, start, stop));
    result.pack_time_ms = pack_ms;

    result.total_compress_time_ms = result.partition_time_ms + result.selector_time_ms + result.pack_time_ms;
    result.compress_throughput_gbps = (data_bytes / 1e9) / (result.total_compress_time_ms / 1000.0);

    // Calculate compressed size
    result.metadata_size_mb = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t) * 2) / (1024.0 * 1024.0);
    result.delta_array_size_mb = delta_words * sizeof(uint32_t) / (1024.0 * 1024.0);
    result.compressed_size_mb = result.metadata_size_mb + result.delta_array_size_mb;
    result.compression_ratio = result.original_size_mb / result.compressed_size_mb;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(3) << result.compression_ratio << "x" << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    std::cout << "Avg partition size: " << std::setprecision(1) << result.avg_partition_size << std::endl;
    std::cout << "Avg delta bits: " << std::setprecision(2) << result.avg_delta_bits << std::endl;

    // Model selection summary
    std::cout << "Model selection:" << std::endl;
    std::cout << "  LINEAR:     " << std::setw(6) << result.model_stats.linear_count
              << " (" << std::setprecision(1) << result.model_stats.linear_pct() << "%)" << std::endl;
    if (result.model_stats.poly2_count > 0) {
        std::cout << "  POLY2:      " << std::setw(6) << result.model_stats.poly2_count
                  << " (" << result.model_stats.poly2_pct() << "%)" << std::endl;
    }
    if (result.model_stats.poly3_count > 0) {
        std::cout << "  POLY3:      " << std::setw(6) << result.model_stats.poly3_count
                  << " (" << result.model_stats.poly3_pct() << "%)" << std::endl;
    }
    if (result.model_stats.direct_copy_count > 0) {
        std::cout << "  FOR_BITPACK:" << std::setw(6) << result.model_stats.direct_copy_count
                  << " (" << result.model_stats.direct_copy_pct() << "%)" << std::endl;
    }

    std::cout << "Timing: partition=" << std::setprecision(2) << result.partition_time_ms
              << "ms, selector=" << result.selector_time_ms
              << "ms, pack=" << result.pack_time_ms << "ms" << std::endl;

    // ========== Phase 5: Decompression ==========
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data_bytes));

    // Warmup
    for (int i = 0; i < 3; i++) {
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Timed runs
    const int NUM_TRIALS = 10;
    float total_decompress_ms = 0;
    for (int trial = 0; trial < NUM_TRIALS; trial++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::launchDecompressPerPartitionBranchless<T>(compressed, d_output, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float trial_ms;
        CUDA_CHECK(cudaEventElapsedTime(&trial_ms, start, stop));
        total_decompress_ms += trial_ms;
    }
    result.decompress_time_ms = total_decompress_ms / NUM_TRIALS;
    result.decompress_throughput_gbps = (data_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    std::cout << "Decompress: " << std::setprecision(2) << result.decompress_time_ms
              << " ms (" << std::setprecision(2) << result.decompress_throughput_gbps << " GB/s)" << std::endl;

    // ========== Phase 6: Verification ==========
    std::vector<T> decoded(data.size());
    CUDA_CHECK(cudaMemcpy(decoded.data(), d_output, data_bytes, cudaMemcpyDeviceToHost));

    result.correctness = true;
    for (size_t i = 0; i < data.size(); i++) {
        if (data[i] != decoded[i]) {
            result.correctness = false;
            result.error_msg = "Mismatch at index " + std::to_string(i);
            break;
        }
    }

    std::cout << "Correctness: " << (result.correctness ? "PASS" : "FAIL") << std::endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(compressed.d_start_indices);
    cudaFree(compressed.d_end_indices);
    cudaFree(compressed.d_model_types);
    cudaFree(compressed.d_model_params);
    cudaFree(compressed.d_delta_bits);
    cudaFree(compressed.d_delta_array_bit_offsets);
    cudaFree(compressed.d_error_bounds);
    cudaFree(compressed.d_sequential_deltas);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return result;
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/data/sosd";
    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "==========================================\n";
    std::cout << "L3 + GPU Polynomial Selector Test\n";
    std::cout << "==========================================\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n\n";

    // All 20 datasets
    std::vector<DatasetInfo> datasets = {
        {1, "linear_200M", "1-linear_200M_uint64.bin", "uint64"},
        {2, "normal_200M", "2-normal_200M_uint64.bin", "uint64"},
        {3, "poisson_87M", "3-poisson_87M_uint64.bin", "uint64"},
        {4, "ml", "4-ml_uint64.bin", "uint64"},
        {5, "books_200M", "5-books_200M_uint32.bin", "uint32"},
        {6, "fb_200M", "6-fb_200M_uint64.bin", "uint64"},
        {7, "wiki_200M", "7-wiki_200M_uint64.bin", "uint64"},
        {8, "osm_800M", "8-osm_cellids_800M_uint64.bin", "uint64"},
        {9, "movieid", "9-movieid_uint32.bin", "uint32"},
        {10, "house_price", "10-house_price_uint64.bin", "uint64"},
        {11, "planet", "11-planet_uint64.bin", "uint64"},
        {12, "libio", "12-libio.bin", "uint64"},
        {13, "medicare", "13-medicare.bin", "uint64"},
        {14, "cosmos", "14-cosmos_int32.bin", "int32"},
        {15, "polylog_10M", "15-polylog_10M_uint64.bin", "uint64"},
        {16, "exp_200M", "16-exp_200M_uint64.bin", "uint64"},
        {17, "poly_200M", "17-poly_200M_uint64.bin", "uint64"},
        {18, "site_250k", "18-site_250k_uint32.bin", "uint32"},
        {19, "weight_25k", "19-weight_25k_uint32.bin", "uint32"},
        {20, "adult_30k", "20-adult_30k_uint32.bin", "uint32"},
    };

    std::vector<DetailedResult> results;
    int passed = 0, failed = 0, skipped = 0;

    // Aggregated model stats
    ModelStats total_model_stats;

    for (const auto& ds : datasets) {
        DetailedResult result;
        if (ds.type == "uint64") {
            result = runDetailedTest<uint64_t>(ds, data_dir);
        } else if (ds.type == "uint32" || ds.type == "int32") {
            result = runDetailedTest<uint32_t>(ds, data_dir);
        }

        results.push_back(result);

        if (result.error_msg.empty() && result.correctness) {
            passed++;
            total_model_stats.linear_count += result.model_stats.linear_count;
            total_model_stats.poly2_count += result.model_stats.poly2_count;
            total_model_stats.poly3_count += result.model_stats.poly3_count;
            total_model_stats.direct_copy_count += result.model_stats.direct_copy_count;
        } else if (!result.error_msg.empty()) {
            skipped++;
        } else {
            failed++;
        }
    }

    // Print summary
    std::cout << "\n==========================================\n";
    std::cout << "SUMMARY\n";
    std::cout << "==========================================\n\n";

    std::cout << "| # | Dataset | Elements | Ratio | Parts | Linear | Poly2 | Poly3 | FOR | Decomp GB/s | Status |\n";
    std::cout << "|---|---------|----------|-------|-------|--------|-------|-------|-----|-------------|--------|\n";

    double total_ratio = 0;
    double total_throughput = 0;
    int count = 0;

    for (const auto& r : results) {
        if (r.error_msg.empty()) {
            std::cout << "| " << std::setw(2) << r.id << " | "
                      << std::setw(12) << r.name << " | "
                      << std::setw(10) << r.num_elements << " | "
                      << std::fixed << std::setprecision(2) << std::setw(6) << r.compression_ratio << "x | "
                      << std::setw(6) << r.num_partitions << " | "
                      << std::setw(6) << r.model_stats.linear_count << " | "
                      << std::setw(5) << r.model_stats.poly2_count << " | "
                      << std::setw(5) << r.model_stats.poly3_count << " | "
                      << std::setw(3) << r.model_stats.direct_copy_count << " | "
                      << std::setw(11) << std::setprecision(1) << r.decompress_throughput_gbps << " | "
                      << (r.correctness ? "PASS" : "FAIL") << " |\n";

            if (r.correctness) {
                total_ratio += r.compression_ratio;
                total_throughput += r.decompress_throughput_gbps;
                count++;
            }
        } else {
            std::cout << "| " << std::setw(2) << r.id << " | "
                      << std::setw(12) << r.name << " | SKIP | " << r.error_msg << " |\n";
        }
    }

    std::cout << "\n==========================================\n";
    std::cout << "OVERALL STATISTICS\n";
    std::cout << "==========================================\n";
    std::cout << "Passed: " << passed << "/" << datasets.size() << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Skipped: " << skipped << std::endl;

    if (count > 0) {
        std::cout << "\nAvg Compression Ratio: " << std::fixed << std::setprecision(2) << total_ratio / count << "x\n";
        std::cout << "Avg Decompression Throughput: " << std::setprecision(1) << total_throughput / count << " GB/s\n";
    }

    std::cout << "\n==========================================\n";
    std::cout << "GLOBAL MODEL SELECTION (All Partitions)\n";
    std::cout << "==========================================\n";
    std::cout << "Total Partitions: " << total_model_stats.total() << std::endl;
    std::cout << "  LINEAR:      " << std::setw(8) << total_model_stats.linear_count
              << " (" << std::setprecision(1) << total_model_stats.linear_pct() << "%)" << std::endl;
    std::cout << "  POLY2:       " << std::setw(8) << total_model_stats.poly2_count
              << " (" << total_model_stats.poly2_pct() << "%)" << std::endl;
    std::cout << "  POLY3:       " << std::setw(8) << total_model_stats.poly3_count
              << " (" << total_model_stats.poly3_pct() << "%)" << std::endl;
    std::cout << "  FOR_BITPACK: " << std::setw(8) << total_model_stats.direct_copy_count
              << " (" << total_model_stats.direct_copy_pct() << "%)" << std::endl;

    return (failed > 0) ? 1 : 0;
}
