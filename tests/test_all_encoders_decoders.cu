/**
 * L3 All Encoders and Decoders Test
 *
 * Tests both CompressedDataL3 and CompressedDataVertical structures
 * with all available encoder/decoder combinations on normal_200M dataset.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>
#include <random>
#include <cuda_runtime.h>

// Format headers
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_opt.h"

// Utils - must come before decompression_kernels.cu
#include "../src/kernels/utils/bitpack_utils.cuh"

// Decoder headers - Original L3
#include "../src/kernels/decompression/decompression_kernels.cu"

// Decoder headers - Warp Optimized
#include "../src/kernels/decompression/decoder_warp_opt.cu"

// Encoder headers
#include "../src/kernels/compression/encoder_cost_optimal.cu"

#undef WARP_SIZE
#undef MODEL_OVERHEAD_BYTES

#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Decoder headers - Vertical
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            return; \
        } \
    } while(0)

// ============================================================================
// Test Results
// ============================================================================

struct TestResult {
    std::string encoder_name;
    std::string decoder_name;
    std::string data_structure;
    int num_partitions;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double decompress_throughput_gbps;
    bool correct;
    std::string model_distribution;
};

std::vector<TestResult> g_results;

// ============================================================================
// L3 Encoder (Original - LINEAR only)
// ============================================================================

template<typename T>
CompressedDataL3<T> encodeL3_Linear(const std::vector<T>& data, int partition_size = 2048) {
    CompressedDataL3<T> result;
    size_t n = data.size();

    // Create fixed partitions
    int num_partitions = (n + partition_size - 1) / partition_size;
    result.num_partitions = num_partitions;
    result.total_values = n;

    // Allocate metadata
    cudaMalloc(&result.d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.d_end_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.d_model_types, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.d_model_params, num_partitions * 4 * sizeof(double));
    cudaMalloc(&result.d_delta_bits, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t));
    cudaMalloc(&result.d_error_bounds, num_partitions * sizeof(int64_t));

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    std::vector<int32_t> h_model_types(num_partitions);
    std::vector<double> h_params(num_partitions * 4);
    std::vector<int32_t> h_delta_bits(num_partitions);
    std::vector<int64_t> h_bit_offsets(num_partitions);
    std::vector<int64_t> h_error_bounds(num_partitions);

    int64_t total_bits = 0;

    for (int p = 0; p < num_partitions; p++) {
        int start = p * partition_size;
        int end = std::min((int)((p + 1) * partition_size), (int)n);
        h_start[p] = start;
        h_end[p] = end;

        int psize = end - start;

        // Linear regression
        double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
        for (int i = 0; i < psize; i++) {
            double x = i;
            double y = static_cast<double>(data[start + i]);
            sum_x += x;
            sum_y += y;
            sum_xx += x * x;
            sum_xy += x * y;
        }
        double det = psize * sum_xx - sum_x * sum_x;
        double theta1 = (psize * sum_xy - sum_x * sum_y) / det;
        double theta0 = (sum_y - theta1 * sum_x) / psize;

        // Compute max error
        int64_t max_err = 0;
        for (int i = 0; i < psize; i++) {
            double pred = theta0 + theta1 * i;
            int64_t pred_int = static_cast<int64_t>(std::llrint(pred));
            int64_t err = std::abs(static_cast<int64_t>(data[start + i]) - pred_int);
            max_err = std::max(max_err, err);
        }

        int bits = (max_err > 0) ? (64 - __builtin_clzll(max_err) + 1) : 0;

        h_model_types[p] = MODEL_LINEAR;
        h_params[p * 4 + 0] = theta0;
        h_params[p * 4 + 1] = theta1;
        h_params[p * 4 + 2] = 0;
        h_params[p * 4 + 3] = 0;
        h_delta_bits[p] = bits;
        h_bit_offsets[p] = total_bits;
        h_error_bounds[p] = max_err;
        total_bits += static_cast<int64_t>(psize) * bits;
    }

    // Allocate delta array
    result.delta_array_words = (total_bits + 31) / 32;
    cudaMalloc(&result.delta_array, result.delta_array_words * sizeof(uint32_t));
    cudaMemset(result.delta_array, 0, result.delta_array_words * sizeof(uint32_t));

    // Pack deltas on CPU
    std::vector<uint32_t> h_delta_array(result.delta_array_words, 0);

    for (int p = 0; p < num_partitions; p++) {
        int start = h_start[p];
        int end = h_end[p];
        int bits = h_delta_bits[p];
        int64_t bit_offset = h_bit_offsets[p];
        double theta0 = h_params[p * 4];
        double theta1 = h_params[p * 4 + 1];

        if (bits == 0) continue;

        for (int i = 0; i < end - start; i++) {
            double pred = theta0 + theta1 * i;
            int64_t pred_int = static_cast<int64_t>(std::llrint(pred));
            int64_t delta = static_cast<int64_t>(data[start + i]) - pred_int;
            uint64_t packed = (bits == 64) ? static_cast<uint64_t>(delta)
                             : (static_cast<uint64_t>(delta) & ((1ULL << bits) - 1));

            int64_t cur_bit = bit_offset + static_cast<int64_t>(i) * bits;

            // Multi-word packing for up to 64 bits
            int bits_remaining = bits;
            int current_word = cur_bit / 32;
            int bit_in_word = cur_bit % 32;
            uint64_t value_to_pack = packed;

            while (bits_remaining > 0 && current_word < (int)result.delta_array_words) {
                int bits_in_this_word = std::min(bits_remaining, 32 - bit_in_word);
                uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
                uint32_t word_part = static_cast<uint32_t>(value_to_pack & mask) << bit_in_word;
                h_delta_array[current_word] |= word_part;

                value_to_pack >>= bits_in_this_word;
                bits_remaining -= bits_in_this_word;
                current_word++;
                bit_in_word = 0;
            }
        }
    }

    // Copy to device
    cudaMemcpy(result.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_model_types, h_model_types.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_model_params, h_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.d_error_bounds, h_error_bounds.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(result.delta_array, h_delta_array.data(), result.delta_array_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

    result.d_plain_deltas = nullptr;
    result.d_partition_min_values = nullptr;
    result.d_partition_max_values = nullptr;
    result.d_self = nullptr;

    return result;
}

template<typename T>
void freeL3(CompressedDataL3<T>& data) {
    cudaFree(data.d_start_indices);
    cudaFree(data.d_end_indices);
    cudaFree(data.d_model_types);
    cudaFree(data.d_model_params);
    cudaFree(data.d_delta_bits);
    cudaFree(data.d_delta_array_bit_offsets);
    cudaFree(data.d_error_bounds);
    cudaFree(data.delta_array);
    data = CompressedDataL3<T>();
}

// ============================================================================
// Convert L3 to CompressedDataOpt
// ============================================================================

template<typename T>
CompressedDataOpt<T> L3ToOpt(const CompressedDataL3<T>& L3) {
    CompressedDataOpt<T> opt;
    opt.d_start_indices = L3.d_start_indices;
    opt.d_end_indices = L3.d_end_indices;
    opt.d_model_types = L3.d_model_types;
    opt.d_model_params = L3.d_model_params;
    opt.d_delta_bits = L3.d_delta_bits;
    opt.d_delta_array_bit_offsets = L3.d_delta_array_bit_offsets;
    opt.delta_array = L3.delta_array;
    opt.d_plain_deltas = nullptr;
    opt.num_partitions = L3.num_partitions;
    opt.total_elements = L3.total_values;
    return opt;
}

// ============================================================================
// Test Functions
// ============================================================================

template<typename T>
void testVerticalEncoder(const std::vector<T>& data, const std::string& partition_strategy) {
    size_t n = data.size();
    double data_bytes = n * sizeof(T);
    double data_mb = data_bytes / (1024.0 * 1024.0);

    // Create partitions based on strategy
    std::vector<PartitionInfo> partitions;
    int num_partitions = 0;

    if (partition_strategy == "FIXED_2048") {
        partitions = Vertical_encoder::createFixedPartitions<T>(n, 2048);
        num_partitions = partitions.size();
    } else if (partition_strategy == "COST_OPTIMAL") {
        CostOptimalConfig cfg = CostOptimalConfig::balanced();
        partitions = createPartitionsCostOptimal<T>(data, cfg, &num_partitions, 0);
    }

    VerticalConfig fl_config;
    fl_config.enable_adaptive_selection = true;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Compress
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    auto compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config, 0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float compress_ms;
    cudaEventElapsedTime(&compress_ms, start, stop);

    // Calculate compression ratio
    double metadata_size = compressed.num_partitions * 64.0;
    double delta_size = compressed.sequential_delta_words * sizeof(uint32_t);
    double compressed_mb = (metadata_size + delta_size) / (1024.0 * 1024.0);
    double ratio = data_mb / compressed_mb;

    // Get model distribution
    std::vector<int32_t> h_model_types(compressed.num_partitions);
    cudaMemcpy(h_model_types.data(), compressed.d_model_types,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int linear = 0, poly2 = 0, poly3 = 0, for_bp = 0;
    for (int mt : h_model_types) {
        switch (mt) {
            case MODEL_LINEAR: linear++; break;
            case MODEL_POLYNOMIAL2: poly2++; break;
            case MODEL_POLYNOMIAL3: poly3++; break;
            case MODEL_FOR_BITPACK: for_bp++; break;
        }
    }

    std::string model_dist;
    if (linear > 0) model_dist += "L:" + std::to_string(100*linear/compressed.num_partitions) + "% ";
    if (poly2 > 0) model_dist += "P2:" + std::to_string(100*poly2/compressed.num_partitions) + "% ";
    if (poly3 > 0) model_dist += "P3:" + std::to_string(100*poly3/compressed.num_partitions) + "% ";
    if (for_bp > 0) model_dist += "FOR:" + std::to_string(100*for_bp/compressed.num_partitions) + "%";

    // Test each decompression mode
    DecompressMode modes[] = {DecompressMode::AUTO, DecompressMode::BRANCHLESS,
                              DecompressMode::SEQUENTIAL, DecompressMode::INTERLEAVED};
    const char* mode_names[] = {"AUTO", "BRANCHLESS", "SEQUENTIAL", "INTERLEAVED"};

    T* d_output;
    cudaMalloc(&d_output, data_bytes);

    for (int m = 0; m < 4; m++) {
        cudaMemset(d_output, 0, data_bytes);

        // Warmup
        Vertical_decoder::decompressAll<T>(compressed, d_output, modes[m], 0);
        cudaDeviceSynchronize();

        // Timed run
        cudaEventRecord(start);
        Vertical_decoder::decompressAll<T>(compressed, d_output, modes[m], 0);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float decompress_ms;
        cudaEventElapsedTime(&decompress_ms, start, stop);
        double throughput = (data_bytes / 1e9) / (decompress_ms / 1e3);

        // Verify
        std::vector<T> output(n);
        cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < n && correct; i++) {
            if (output[i] != data[i]) correct = false;
        }

        TestResult result;
        result.encoder_name = "Vertical_" + partition_strategy;
        result.decoder_name = std::string("Vertical_") + mode_names[m];
        result.data_structure = "CompressedDataVertical";
        result.num_partitions = compressed.num_partitions;
        result.compression_ratio = ratio;
        result.compress_time_ms = compress_ms;
        result.decompress_time_ms = decompress_ms;
        result.decompress_throughput_gbps = throughput;
        result.correct = correct;
        result.model_distribution = model_dist;

        g_results.push_back(result);

        std::cout << "  " << partition_strategy << " + " << mode_names[m] << ": "
                  << (correct ? "PASS" : "FAIL") << " ratio=" << std::fixed << std::setprecision(2)
                  << ratio << "x, " << throughput << " GB/s" << std::endl;
    }

    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(compressed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

template<typename T>
void testL3Encoder(const std::vector<T>& data) {
    size_t n = data.size();
    double data_bytes = n * sizeof(T);
    double data_mb = data_bytes / (1024.0 * 1024.0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Compress using L3 LINEAR encoder
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    auto compressed = encodeL3_Linear<T>(data, 2048);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float compress_ms;
    cudaEventElapsedTime(&compress_ms, start, stop);

    // Calculate compression ratio
    double metadata_size = compressed.num_partitions * 64.0;
    double delta_size = compressed.delta_array_words * sizeof(uint32_t);
    double compressed_mb = (metadata_size + delta_size) / (1024.0 * 1024.0);
    double ratio = data_mb / compressed_mb;

    // Convert to CompressedDataOpt for old decoders
    auto opt = L3ToOpt(compressed);

    T* d_output;
    cudaMalloc(&d_output, data_bytes);

    // ========== Test 1: decompressPartitionsOptimized ==========
    {
        cudaMemset(d_output, 0, data_bytes);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        decompressPartitionsOptimized<T><<<compressed.num_partitions, 256>>>(opt, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float decompress_ms;
        cudaEventElapsedTime(&decompress_ms, start, stop);
        double throughput = (data_bytes / 1e9) / (decompress_ms / 1e3);

        std::vector<T> output(n);
        cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < n && correct; i++) {
            if (output[i] != data[i]) correct = false;
        }

        TestResult result;
        result.encoder_name = "L3_LINEAR";
        result.decoder_name = "decompressPartitionsOptimized";
        result.data_structure = "CompressedDataL3->Opt";
        result.num_partitions = compressed.num_partitions;
        result.compression_ratio = ratio;
        result.compress_time_ms = compress_ms;
        result.decompress_time_ms = decompress_ms;
        result.decompress_throughput_gbps = throughput;
        result.correct = correct;
        result.model_distribution = "LINEAR: 100%";

        g_results.push_back(result);

        std::cout << "  L3_LINEAR + PartitionsOptimized: " << (correct ? "PASS" : "FAIL")
                  << " ratio=" << std::fixed << std::setprecision(2)
                  << ratio << "x, " << throughput << " GB/s" << std::endl;
    }

    // ========== Test 2: decompressWarpOptimized ==========
    {
        cudaMemset(d_output, 0, data_bytes);

        cudaDeviceSynchronize();
        cudaEventRecord(start);
        // decompressWarpOptimized uses 4 warps per block = 128 threads
        decompressWarpOptimized<T><<<compressed.num_partitions, 128>>>(opt, d_output);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float decompress_ms;
        cudaEventElapsedTime(&decompress_ms, start, stop);
        double throughput = (data_bytes / 1e9) / (decompress_ms / 1e3);

        std::vector<T> output(n);
        cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);

        bool correct = true;
        for (size_t i = 0; i < n && correct; i++) {
            if (output[i] != data[i]) correct = false;
        }

        TestResult result;
        result.encoder_name = "L3_LINEAR";
        result.decoder_name = "decompressWarpOptimized";
        result.data_structure = "CompressedDataL3->Opt";
        result.num_partitions = compressed.num_partitions;
        result.compression_ratio = ratio;
        result.compress_time_ms = compress_ms;
        result.decompress_time_ms = decompress_ms;
        result.decompress_throughput_gbps = throughput;
        result.correct = correct;
        result.model_distribution = "LINEAR: 100%";

        g_results.push_back(result);

        std::cout << "  L3_LINEAR + WarpOptimized: " << (correct ? "PASS" : "FAIL")
                  << " ratio=" << ratio << "x, " << throughput << " GB/s" << std::endl;
    }

    // ========== Test 3: Convert to Vertical and test with Vertical decoders ==========
    {
        auto Vertical = CompressedDataVertical<T>::fromBase(compressed);

        DecompressMode modes[] = {DecompressMode::AUTO, DecompressMode::BRANCHLESS};
        const char* mode_names[] = {"AUTO", "BRANCHLESS"};

        for (int m = 0; m < 2; m++) {
            cudaMemset(d_output, 0, data_bytes);

            cudaDeviceSynchronize();
            cudaEventRecord(start);
            Vertical_decoder::decompressAll<T>(Vertical, d_output, modes[m], 0);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float decompress_ms;
            cudaEventElapsedTime(&decompress_ms, start, stop);
            double throughput = (data_bytes / 1e9) / (decompress_ms / 1e3);

            std::vector<T> output(n);
            cudaMemcpy(output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);

            bool correct = true;
            for (size_t i = 0; i < n && correct; i++) {
                if (output[i] != data[i]) correct = false;
            }

            TestResult res;
            res.encoder_name = "L3_LINEAR";
            res.decoder_name = std::string("Vertical_") + mode_names[m];
            res.data_structure = "L3->Vertical";
            res.num_partitions = compressed.num_partitions;
            res.compression_ratio = ratio;
            res.compress_time_ms = compress_ms;
            res.decompress_time_ms = decompress_ms;
            res.decompress_throughput_gbps = throughput;
            res.correct = correct;
            res.model_distribution = "LINEAR: 100%";

            g_results.push_back(res);

            std::cout << "  L3_LINEAR + Vertical_" << mode_names[m] << ": "
                      << (correct ? "PASS" : "FAIL")
                      << " ratio=" << ratio << "x, " << throughput << " GB/s" << std::endl;
        }
    }

    cudaFree(d_output);
    freeL3(compressed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string data_file = "data/sosd/2-normal_200M_uint64.bin";
    std::string output_dir = "reports/L3/datasets/2-normal";

    std::cout << "============================================" << std::endl;
    std::cout << "  L3 All Encoders/Decoders Test" << std::endl;
    std::cout << "  Testing both data structures" << std::endl;
    std::cout << "============================================" << std::endl;

    // Read data
    std::ifstream file(data_file, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open: " << data_file << std::endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t n = file_size / sizeof(uint64_t);
    std::vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    double data_mb = n * sizeof(uint64_t) / (1024.0 * 1024.0);
    std::cout << "Dataset: " << data_file << std::endl;
    std::cout << "Elements: " << n << std::endl;
    std::cout << "Size: " << std::fixed << std::setprecision(2) << data_mb << " MB" << std::endl;
    std::cout << "============================================" << std::endl;

    // Test Vertical encoder with different partitioning strategies
    std::cout << "\n[1] Vertical Encoder + Vertical Decoder (CompressedDataVertical)" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    testVerticalEncoder<uint64_t>(data, "FIXED_2048");
    testVerticalEncoder<uint64_t>(data, "COST_OPTIMAL");

    // Test L3 encoder with both old and new decoders
    std::cout << "\n[2] L3 Encoder + Various Decoders (CompressedDataL3)" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    testL3Encoder<uint64_t>(data);

    // Generate report
    std::cout << "\n============================================" << std::endl;
    std::cout << "  Summary Report" << std::endl;
    std::cout << "============================================" << std::endl;

    system(("mkdir -p " + output_dir).c_str());

    std::ofstream report(output_dir + "/all_encoders_decoders_report.md");
    report << "# L3 All Encoders and Decoders Test Report\n\n";
    report << "**Dataset**: " << data_file << "\n";
    report << "**Elements**: " << n << "\n";
    report << "**Size**: " << data_mb << " MB\n\n";

    report << "## Test Results\n\n";
    report << "| Encoder | Decoder | Data Structure | Partitions | Ratio | Decomp (ms) | Throughput (GB/s) | Status |\n";
    report << "|---------|---------|----------------|------------|-------|-------------|-------------------|--------|\n";

    int pass_count = 0;
    for (const auto& r : g_results) {
        report << "| " << r.encoder_name << " | " << r.decoder_name << " | "
               << r.data_structure << " | " << r.num_partitions << " | "
               << std::fixed << std::setprecision(2) << r.compression_ratio << "x | "
               << r.decompress_time_ms << " | " << r.decompress_throughput_gbps << " | "
               << (r.correct ? "PASS" : "FAIL") << " |\n";

        if (r.correct) pass_count++;

        std::cout << r.encoder_name << " + " << r.decoder_name << ": "
                  << (r.correct ? "PASS" : "FAIL") << std::endl;
    }

    report << "\n## Summary\n\n";
    report << "- Total tests: " << g_results.size() << "\n";
    report << "- Passed: " << pass_count << "\n";
    report << "- Failed: " << (g_results.size() - pass_count) << "\n";
    report.close();

    std::cout << "\nTotal: " << pass_count << "/" << g_results.size() << " tests passed" << std::endl;
    std::cout << "Report saved to: " << output_dir << "/all_encoders_decoders_report.md" << std::endl;

    return (pass_count == (int)g_results.size()) ? 0 : 1;
}
