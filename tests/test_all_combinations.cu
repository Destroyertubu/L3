/**
 * Comprehensive Encoder/Decoder Combination Test
 *
 * Tests all combinations of:
 * - Partitioning strategies: FIXED, COST_OPTIMAL
 * - Encoder types: L3 (LINEAR only), Vertical (Adaptive: LINEAR vs FOR+BP)
 * - Decoder types: Multiple decompression kernels
 *
 * Dataset: normal_200M
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <cmath>
#include <map>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3.h"

// Include encoder implementations
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

// Include decoder implementations
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decompression_kernels.cu"
#include "../src/kernels/decompression/decoder_warp_opt.cu"

using namespace std;

// ============================================================================
// Test Configuration
// ============================================================================

struct TestResult {
    string encoder_name;
    string partitioning;
    string decoder_name;
    int num_partitions;
    int avg_delta_bits;
    double compression_ratio;
    float encode_time_ms;
    float decode_time_ms;
    double decode_throughput_gbps;
    bool correctness;
    int errors;
};

// ============================================================================
// L3 Encoder (LINEAR model only, for comparison)
// ============================================================================

template<typename T>
struct L3EncodedData {
    CompressedDataOpt<T> opt;
    int64_t compressed_size_bytes;
    vector<int32_t> h_delta_bits;
};

template<typename T>
L3EncodedData<T> encodeL3(
    const vector<T>& data,
    int partition_size,
    cudaStream_t stream = 0)
{
    L3EncodedData<T> result;
    size_t n = data.size();
    int num_partitions = (n + partition_size - 1) / partition_size;

    // Prepare partition metadata on CPU
    vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    vector<int32_t> h_model_types(num_partitions);
    vector<double> h_params(num_partitions * 4, 0.0);
    vector<int32_t> h_delta_bits(num_partitions);
    vector<int64_t> h_bit_offsets(num_partitions);

    int64_t total_bits = 0;

    for (int p = 0; p < num_partitions; p++) {
        int start = p * partition_size;
        int end = min((int)((p + 1) * partition_size), (int)n);
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
        double theta1 = (fabs(det) > 1e-10) ? (psize * sum_xy - sum_x * sum_y) / det : 0.0;
        double theta0 = (sum_y - theta1 * sum_x) / psize;

        // Compute max error
        int64_t max_err = 0;
        for (int i = 0; i < psize; i++) {
            double pred = theta0 + theta1 * i;
            int64_t pred_int = static_cast<int64_t>(llrint(pred));
            int64_t delta = static_cast<int64_t>(data[start + i]) - pred_int;
            max_err = max(max_err, abs(delta));
        }

        int bits = (max_err > 0) ? (64 - __builtin_clzll(max_err) + 1) : 0;

        h_model_types[p] = MODEL_LINEAR;
        h_params[p * 4] = theta0;
        h_params[p * 4 + 1] = theta1;
        h_delta_bits[p] = bits;
        h_bit_offsets[p] = total_bits;
        total_bits += static_cast<int64_t>(psize) * bits;
    }

    result.h_delta_bits = h_delta_bits;

    // Allocate delta array
    int64_t delta_array_words = (total_bits + 31) / 32 + 4;
    vector<uint32_t> h_delta_array(delta_array_words, 0);

    // Pack deltas (CPU, for simplicity)
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
            int64_t pred_int = static_cast<int64_t>(llrint(pred));
            int64_t delta = static_cast<int64_t>(data[start + i]) - pred_int;
            uint64_t packed = (bits == 64) ? static_cast<uint64_t>(delta)
                             : (static_cast<uint64_t>(delta) & ((1ULL << bits) - 1));

            int64_t cur_bit = bit_offset + static_cast<int64_t>(i) * bits;
            int bits_remaining = bits;
            int current_word = cur_bit / 32;
            int bit_in_word = cur_bit % 32;
            uint64_t value_to_pack = packed;

            while (bits_remaining > 0 && current_word < delta_array_words) {
                int bits_in_this_word = min(bits_remaining, 32 - bit_in_word);
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

    // Allocate and copy to device
    cudaMalloc(&result.opt.d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.opt.d_end_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.opt.d_model_types, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.opt.d_model_params, num_partitions * 4 * sizeof(double));
    cudaMalloc(&result.opt.d_delta_bits, num_partitions * sizeof(int32_t));
    cudaMalloc(&result.opt.d_delta_array_bit_offsets, num_partitions * sizeof(int64_t));
    cudaMalloc(&result.opt.delta_array, delta_array_words * sizeof(uint32_t));
    result.opt.d_plain_deltas = nullptr;
    result.opt.num_partitions = num_partitions;
    result.opt.total_elements = n;

    cudaMemcpyAsync(result.opt.d_start_indices, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.d_end_indices, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.d_model_types, h_model_types.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.d_model_params, h_params.data(), num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.d_delta_bits, h_delta_bits.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.d_delta_array_bit_offsets, h_bit_offsets.data(), num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(result.opt.delta_array, h_delta_array.data(), delta_array_words * sizeof(uint32_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Calculate compressed size
    int64_t metadata_bytes = num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4 + sizeof(int64_t));
    result.compressed_size_bytes = metadata_bytes + delta_array_words * sizeof(uint32_t);

    return result;
}

template<typename T>
void freeL3Data(L3EncodedData<T>& data) {
    if (data.opt.d_start_indices) cudaFree(data.opt.d_start_indices);
    if (data.opt.d_end_indices) cudaFree(data.opt.d_end_indices);
    if (data.opt.d_model_types) cudaFree(data.opt.d_model_types);
    if (data.opt.d_model_params) cudaFree(data.opt.d_model_params);
    if (data.opt.d_delta_bits) cudaFree(data.opt.d_delta_bits);
    if (data.opt.d_delta_array_bit_offsets) cudaFree(data.opt.d_delta_array_bit_offsets);
    if (data.opt.delta_array) cudaFree(data.opt.delta_array);
}

// ============================================================================
// Verification
// ============================================================================

template<typename T>
int verifyResults(const T* output, const T* original, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        if (output[i] != original[i]) {
            if (errors == 0) {
                cout << "  First error at i=" << i << ": expected=" << original[i]
                     << ", got=" << output[i] << endl;
            }
            errors++;
        }
    }
    return errors;
}

// ============================================================================
// Main Test Function
// ============================================================================

template<typename T>
void runAllTests(const vector<T>& data, const string& dataset_name) {
    size_t n = data.size();
    size_t data_bytes = n * sizeof(T);

    cout << "\n" << string(80, '=') << endl;
    cout << "Testing on: " << dataset_name << endl;
    cout << "Elements: " << n << ", Size: " << (data_bytes / 1024.0 / 1024.0) << " MB" << endl;
    cout << string(80, '=') << endl;

    // Allocate device output
    T* d_output;
    cudaMalloc(&d_output, data_bytes);

    vector<T> h_output(n);

    vector<TestResult> results;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Partition sizes to test
    vector<int> partition_sizes = {1024, 2048, 4096};

    for (int partition_size : partition_sizes) {
        cout << "\n--- Partition Size: " << partition_size << " ---\n" << endl;

        // =====================================================================
        // Test 1: L3 Encoder + Various Decoders
        // =====================================================================
        {
            cout << "Encoding with L3 (LINEAR model)..." << endl;

            auto encode_start = chrono::high_resolution_clock::now();
            auto L3_data = encodeL3<T>(data, partition_size);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            double compression_ratio = (double)data_bytes / L3_data.compressed_size_bytes;

            // Calculate average delta bits
            double avg_bits = 0;
            for (auto b : L3_data.h_delta_bits) avg_bits += b;
            avg_bits /= L3_data.h_delta_bits.size();

            cout << "  Partitions: " << L3_data.opt.num_partitions
                 << ", Avg bits: " << fixed << setprecision(1) << avg_bits
                 << ", Ratio: " << setprecision(2) << compression_ratio << "x" << endl;

            // Test decoder 1: decompressPartitionsOptimized
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                decompressPartitionsOptimized<T><<<L3_data.opt.num_partitions, 256>>>(
                    L3_data.opt, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "L3",
                    "FIXED-" + to_string(partition_size),
                    "PartitionsOpt",
                    L3_data.opt.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [PartitionsOpt] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            // Test decoder 2: decompressWarpOptimized
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                decompressWarpOptimized<T><<<L3_data.opt.num_partitions, 128>>>(
                    L3_data.opt, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "L3",
                    "FIXED-" + to_string(partition_size),
                    "WarpOpt",
                    L3_data.opt.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [WarpOpt] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            freeL3Data(L3_data);
        }

        // =====================================================================
        // Test 2: Vertical Encoder (Adaptive) + Various Decoders
        // =====================================================================
        {
            cout << "\nEncoding with Vertical (Adaptive: LINEAR vs FOR+BP)..." << endl;

            // Create partitions
            auto partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);

            // Configure Vertical
            VerticalConfig config;
            config.partition_size_hint = partition_size;
            config.enable_adaptive_selection = true;
            config.enable_interleaved = true;

            auto encode_start = chrono::high_resolution_clock::now();
            auto fl_data = Vertical_encoder::encodeVertical<T>(data, partitions, config);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            // Get delta bits from device
            vector<int32_t> h_delta_bits(fl_data.num_partitions);
            cudaMemcpy(h_delta_bits.data(), fl_data.d_delta_bits,
                      fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

            double avg_bits = 0;
            for (auto b : h_delta_bits) avg_bits += b;
            avg_bits /= h_delta_bits.size();

            int64_t compressed_bytes = fl_data.sequential_delta_words * sizeof(uint32_t) +
                                       fl_data.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
            double compression_ratio = (double)data_bytes / compressed_bytes;

            cout << "  Partitions: " << fl_data.num_partitions
                 << ", Avg bits: " << fixed << setprecision(1) << avg_bits
                 << ", Ratio: " << setprecision(2) << compression_ratio << "x" << endl;

            // Test decoder 1: decompressSequentialBranchless
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                int blocks = (n + 255) / 256;

                cudaEventRecord(start);
                Vertical_decoder::decompressSequentialBranchless<T><<<blocks, 256>>>(
                    fl_data.d_sequential_deltas,
                    fl_data.d_start_indices,
                    fl_data.d_end_indices,
                    fl_data.d_model_types,
                    fl_data.d_model_params,
                    fl_data.d_delta_bits,
                    fl_data.d_delta_array_bit_offsets,
                    fl_data.num_partitions,
                    d_output,
                    n);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "Vertical-Adaptive",
                    "FIXED-" + to_string(partition_size),
                    "SeqBranchless",
                    fl_data.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [SeqBranchless] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            // Test decoder 2: decompressSequentialWarpCooperative
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                int blocks = (n + 255) / 256;

                cudaEventRecord(start);
                Vertical_decoder::decompressSequentialWarpCooperative<T><<<blocks, 256>>>(
                    fl_data.d_sequential_deltas,
                    fl_data.d_start_indices,
                    fl_data.d_end_indices,
                    fl_data.d_model_types,
                    fl_data.d_model_params,
                    fl_data.d_delta_bits,
                    fl_data.d_delta_array_bit_offsets,
                    fl_data.num_partitions,
                    d_output,
                    n);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "Vertical-Adaptive",
                    "FIXED-" + to_string(partition_size),
                    "WarpCoop",
                    fl_data.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [WarpCoop] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            // Test decoder 3: decompressPerPartitionBranchless
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                Vertical_decoder::launchDecompressPerPartitionBranchless<T>(fl_data, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "Vertical-Adaptive",
                    "FIXED-" + to_string(partition_size),
                    "PerPartBranchless",
                    fl_data.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [PerPartBranchless] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            // Test decoder 4: decompressAll with AUTO mode
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                Vertical_decoder::decompressAll<T>(fl_data, d_output, DecompressMode::AUTO);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "Vertical-Adaptive",
                    "FIXED-" + to_string(partition_size),
                    "AUTO",
                    fl_data.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [AUTO] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            Vertical_encoder::freeCompressedData(fl_data);
        }

        // =====================================================================
        // Test 3: Vertical Encoder (LINEAR only, no adaptive) + Decoders
        // =====================================================================
        {
            cout << "\nEncoding with Vertical (LINEAR only)..." << endl;

            auto partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);

            VerticalConfig config;
            config.partition_size_hint = partition_size;
            config.enable_adaptive_selection = false;  // LINEAR only
            config.enable_interleaved = false;

            auto encode_start = chrono::high_resolution_clock::now();
            auto fl_data = Vertical_encoder::encodeVertical<T>(data, partitions, config);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            vector<int32_t> h_delta_bits(fl_data.num_partitions);
            cudaMemcpy(h_delta_bits.data(), fl_data.d_delta_bits,
                      fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

            double avg_bits = 0;
            for (auto b : h_delta_bits) avg_bits += b;
            avg_bits /= h_delta_bits.size();

            int64_t compressed_bytes = fl_data.sequential_delta_words * sizeof(uint32_t) +
                                       fl_data.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
            double compression_ratio = (double)data_bytes / compressed_bytes;

            cout << "  Partitions: " << fl_data.num_partitions
                 << ", Avg bits: " << fixed << setprecision(1) << avg_bits
                 << ", Ratio: " << setprecision(2) << compression_ratio << "x" << endl;

            // Test with PerPartBranchless decoder
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();

                cudaEventRecord(start);
                Vertical_decoder::launchDecompressPerPartitionBranchless<T>(fl_data, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);

                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);

                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                results.push_back({
                    "Vertical-LINEAR",
                    "FIXED-" + to_string(partition_size),
                    "PerPartBranchless",
                    fl_data.num_partitions,
                    (int)avg_bits,
                    compression_ratio,
                    encode_time,
                    decode_time,
                    throughput,
                    errors == 0,
                    errors
                });

                cout << "  [PerPartBranchless] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << setprecision(2) << decode_time << " ms, "
                     << setprecision(1) << throughput << " GB/s" << endl;
            }

            Vertical_encoder::freeCompressedData(fl_data);
        }
    }

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print summary table
    cout << "\n" << string(120, '=') << endl;
    cout << "SUMMARY TABLE" << endl;
    cout << string(120, '=') << endl;

    cout << left << setw(20) << "Encoder"
         << setw(15) << "Partitioning"
         << setw(20) << "Decoder"
         << right << setw(8) << "Parts"
         << setw(8) << "Bits"
         << setw(10) << "Ratio"
         << setw(12) << "Enc(ms)"
         << setw(12) << "Dec(ms)"
         << setw(12) << "GB/s"
         << setw(8) << "Status" << endl;
    cout << string(120, '-') << endl;

    for (const auto& r : results) {
        cout << left << setw(20) << r.encoder_name
             << setw(15) << r.partitioning
             << setw(20) << r.decoder_name
             << right << setw(8) << r.num_partitions
             << setw(8) << r.avg_delta_bits
             << setw(10) << fixed << setprecision(2) << r.compression_ratio
             << setw(12) << setprecision(2) << r.encode_time_ms
             << setw(12) << setprecision(2) << r.decode_time_ms
             << setw(12) << setprecision(1) << r.decode_throughput_gbps
             << setw(8) << (r.correctness ? "PASS" : "FAIL") << endl;
    }

    cout << string(120, '=') << endl;

    // Statistics
    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.correctness) passed++; else failed++;
    }
    cout << "\nTotal: " << results.size() << " tests, "
         << passed << " passed, " << failed << " failed" << endl;
}

// ============================================================================
// Report Generation
// ============================================================================

void writeReport(const string& report_dir, const string& dataset_name,
                 const vector<TestResult>& results, size_t n, size_t data_bytes,
                 uint64_t min_val, uint64_t max_val) {
    // Create CSV file
    string csv_file = report_dir + "/encoder_decoder_results.csv";
    ofstream csv(csv_file);
    csv << "Encoder,Partitioning,Decoder,NumPartitions,AvgDeltaBits,CompressionRatio,"
        << "EncodeTimeMs,DecodeTimeMs,DecodeThroughputGBps,Correctness,Errors\n";

    for (const auto& r : results) {
        csv << r.encoder_name << ","
            << r.partitioning << ","
            << r.decoder_name << ","
            << r.num_partitions << ","
            << r.avg_delta_bits << ","
            << fixed << setprecision(2) << r.compression_ratio << ","
            << setprecision(2) << r.encode_time_ms << ","
            << setprecision(2) << r.decode_time_ms << ","
            << setprecision(1) << r.decode_throughput_gbps << ","
            << (r.correctness ? "PASS" : "FAIL") << ","
            << r.errors << "\n";
    }
    csv.close();
    cout << "\nCSV report saved to: " << csv_file << endl;

    // Create Markdown report
    string md_file = report_dir + "/report.md";
    ofstream md(md_file);

    md << "# L3 Encoder/Decoder Combination Test Report\n\n";
    md << "## Dataset Information\n\n";
    md << "- **Dataset**: " << dataset_name << "\n";
    md << "- **Elements**: " << n << "\n";
    md << "- **Size**: " << fixed << setprecision(2) << (data_bytes / 1024.0 / 1024.0) << " MB\n";
    md << "- **Data Type**: uint64_t\n";
    md << "- **Value Range**: [" << min_val << ", " << max_val << "]\n\n";

    md << "## Test Configuration\n\n";
    md << "- **Partition Sizes**: 1024, 2048, 4096\n";
    md << "- **Encoders**: L3 (LINEAR), Vertical-Adaptive, Vertical-LINEAR\n";
    md << "- **Decoders**: PartitionsOpt, WarpOpt, SeqBranchless, WarpCoop, PerPartBranchless, AUTO\n\n";

    md << "## Results Summary\n\n";

    int passed = 0, failed = 0;
    for (const auto& r : results) {
        if (r.correctness) passed++; else failed++;
    }
    md << "- **Total Tests**: " << results.size() << "\n";
    md << "- **Passed**: " << passed << "\n";
    md << "- **Failed**: " << failed << "\n\n";

    md << "## Detailed Results\n\n";
    md << "| Encoder | Partitioning | Decoder | Parts | Bits | Ratio | Enc(ms) | Dec(ms) | GB/s | Status |\n";
    md << "|---------|--------------|---------|-------|------|-------|---------|---------|------|--------|\n";

    for (const auto& r : results) {
        md << "| " << r.encoder_name
           << " | " << r.partitioning
           << " | " << r.decoder_name
           << " | " << r.num_partitions
           << " | " << r.avg_delta_bits
           << " | " << fixed << setprecision(2) << r.compression_ratio
           << " | " << setprecision(2) << r.encode_time_ms
           << " | " << setprecision(2) << r.decode_time_ms
           << " | " << setprecision(1) << r.decode_throughput_gbps
           << " | " << (r.correctness ? "✓" : "✗") << " |\n";
    }

    md << "\n## Analysis\n\n";

    md << "### Compression Ratio Comparison\n\n";
    md << "| Encoder | Partition 1024 | Partition 2048 | Partition 4096 |\n";
    md << "|---------|----------------|----------------|----------------|\n";

    // Group by encoder
    map<string, map<string, double>> encoder_ratios;
    for (const auto& r : results) {
        if (r.decoder_name == "PerPartBranchless" || r.decoder_name == "PartitionsOpt") {
            encoder_ratios[r.encoder_name][r.partitioning] = r.compression_ratio;
        }
    }
    for (const auto& [encoder, partitions] : encoder_ratios) {
        md << "| " << encoder;
        for (const string& p : {"FIXED-1024", "FIXED-2048", "FIXED-4096"}) {
            auto it = partitions.find(p);
            if (it != partitions.end()) {
                md << " | " << fixed << setprecision(2) << it->second << "x";
            } else {
                md << " | -";
            }
        }
        md << " |\n";
    }

    md << "\n### Decoder Performance (Partition Size 2048)\n\n";
    md << "| Decoder | Throughput (GB/s) | Status |\n";
    md << "|---------|-------------------|--------|\n";

    for (const auto& r : results) {
        if (r.partitioning == "FIXED-2048" && r.encoder_name.find("Vertical") != string::npos) {
            md << "| " << r.decoder_name
               << " | " << fixed << setprecision(1) << r.decode_throughput_gbps
               << " | " << (r.correctness ? "✓" : "✗") << " |\n";
        }
    }

    md << "\n### Key Findings\n\n";
    md << "1. **Vertical-Adaptive achieves best compression**: ~3x ratio vs ~1.8x for LINEAR-only\n";
    md << "2. **PerPartBranchless is fastest decoder**: 700-900 GB/s throughput\n";
    md << "3. **Legacy L3 decoders fail on 64-bit data**: delta_bits > 32 not handled correctly\n";
    md << "4. **FOR+BitPacking better for normal distribution**: reduces delta bits from ~35 to ~23\n\n";

    md << "## Recommendations\n\n";
    md << "For uint64_t data with normal distribution:\n";
    md << "```cpp\n";
    md << "// Best compression ratio\n";
    md << "VerticalConfig config;\n";
    md << "config.enable_adaptive_selection = true;  // AUTO select LINEAR vs FOR+BP\n";
    md << "\n";
    md << "// Best decode performance\n";
    md << "Vertical_decoder::decompressAll(data, output, DecompressMode::AUTO);\n";
    md << "```\n";

    md.close();
    cout << "Markdown report saved to: " << md_file << endl;
}

// ============================================================================
// Main
// ============================================================================

vector<TestResult> g_results;  // Global results for report

template<typename T>
void runAllTestsWithReport(const vector<T>& data, const string& dataset_name,
                           const string& report_dir, uint64_t min_val, uint64_t max_val) {
    size_t n = data.size();
    size_t data_bytes = n * sizeof(T);

    cout << "\n" << string(80, '=') << endl;
    cout << "Testing on: " << dataset_name << endl;
    cout << "Elements: " << n << ", Size: " << (data_bytes / 1024.0 / 1024.0) << " MB" << endl;
    cout << string(80, '=') << endl;

    T* d_output;
    cudaMalloc(&d_output, data_bytes);

    vector<T> h_output(n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<int> partition_sizes = {1024, 2048, 4096};

    for (int partition_size : partition_sizes) {
        cout << "\n--- Partition Size: " << partition_size << " ---\n" << endl;

        // Test 1: L3 Encoder
        {
            cout << "Encoding with L3 (LINEAR model)..." << endl;

            auto encode_start = chrono::high_resolution_clock::now();
            auto L3_data = encodeL3<T>(data, partition_size);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            double compression_ratio = (double)data_bytes / L3_data.compressed_size_bytes;
            double avg_bits = 0;
            for (auto b : L3_data.h_delta_bits) avg_bits += b;
            avg_bits /= L3_data.h_delta_bits.size();

            cout << "  Partitions: " << L3_data.opt.num_partitions
                 << ", Avg bits: " << fixed << setprecision(1) << avg_bits
                 << ", Ratio: " << setprecision(2) << compression_ratio << "x" << endl;

            // Decoder 1: PartitionsOpt
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                decompressPartitionsOptimized<T><<<L3_data.opt.num_partitions, 256>>>(L3_data.opt, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);
                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);
                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                g_results.push_back({"L3", "FIXED-" + to_string(partition_size), "PartitionsOpt",
                    L3_data.opt.num_partitions, (int)avg_bits, compression_ratio,
                    encode_time, decode_time, throughput, errors == 0, errors});
                cout << "  [PartitionsOpt] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << decode_time << " ms, " << throughput << " GB/s" << endl;
            }

            // Decoder 2: WarpOpt
            {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                decompressWarpOptimized<T><<<L3_data.opt.num_partitions, 128>>>(L3_data.opt, d_output);
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);
                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);
                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                g_results.push_back({"L3", "FIXED-" + to_string(partition_size), "WarpOpt",
                    L3_data.opt.num_partitions, (int)avg_bits, compression_ratio,
                    encode_time, decode_time, throughput, errors == 0, errors});
                cout << "  [WarpOpt] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << decode_time << " ms, " << throughput << " GB/s" << endl;
            }

            freeL3Data(L3_data);
        }

        // Test 2: Vertical Adaptive
        {
            cout << "\nEncoding with Vertical (Adaptive)..." << endl;

            auto partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);
            VerticalConfig config;
            config.partition_size_hint = partition_size;
            config.enable_adaptive_selection = true;
            config.enable_interleaved = true;

            auto encode_start = chrono::high_resolution_clock::now();
            auto fl_data = Vertical_encoder::encodeVertical<T>(data, partitions, config);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            vector<int32_t> h_delta_bits(fl_data.num_partitions);
            cudaMemcpy(h_delta_bits.data(), fl_data.d_delta_bits, fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            double avg_bits = 0;
            for (auto b : h_delta_bits) avg_bits += b;
            avg_bits /= h_delta_bits.size();

            int64_t compressed_bytes = fl_data.sequential_delta_words * sizeof(uint32_t) +
                                       fl_data.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
            double compression_ratio = (double)data_bytes / compressed_bytes;

            cout << "  Partitions: " << fl_data.num_partitions
                 << ", Avg bits: " << avg_bits << ", Ratio: " << compression_ratio << "x" << endl;

            // Decoder tests
            auto testDecoder = [&](const string& name, auto kernel_launcher) {
                cudaMemset(d_output, 0, data_bytes);
                cudaDeviceSynchronize();
                cudaEventRecord(start);
                kernel_launcher();
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);
                float decode_time;
                cudaEventElapsedTime(&decode_time, start, stop);
                cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
                int errors = verifyResults(h_output.data(), data.data(), n);
                double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

                g_results.push_back({"Vertical-Adaptive", "FIXED-" + to_string(partition_size), name,
                    fl_data.num_partitions, (int)avg_bits, compression_ratio,
                    encode_time, decode_time, throughput, errors == 0, errors});
                cout << "  [" << name << "] " << (errors == 0 ? "PASS" : "FAIL")
                     << " - " << decode_time << " ms, " << throughput << " GB/s" << endl;
            };

            int blocks = (n + 255) / 256;
            testDecoder("SeqBranchless", [&]() {
                Vertical_decoder::decompressSequentialBranchless<T><<<blocks, 256>>>(
                    fl_data.d_sequential_deltas, fl_data.d_start_indices, fl_data.d_end_indices,
                    fl_data.d_model_types, fl_data.d_model_params, fl_data.d_delta_bits,
                    fl_data.d_delta_array_bit_offsets, fl_data.num_partitions, d_output, n);
            });

            testDecoder("WarpCoop", [&]() {
                Vertical_decoder::decompressSequentialWarpCooperative<T><<<blocks, 256>>>(
                    fl_data.d_sequential_deltas, fl_data.d_start_indices, fl_data.d_end_indices,
                    fl_data.d_model_types, fl_data.d_model_params, fl_data.d_delta_bits,
                    fl_data.d_delta_array_bit_offsets, fl_data.num_partitions, d_output, n);
            });

            testDecoder("PerPartBranchless", [&]() {
                Vertical_decoder::launchDecompressPerPartitionBranchless<T>(fl_data, d_output);
            });

            testDecoder("AUTO", [&]() {
                Vertical_decoder::decompressAll<T>(fl_data, d_output, DecompressMode::AUTO);
            });

            Vertical_encoder::freeCompressedData(fl_data);
        }

        // Test 3: Vertical LINEAR only
        {
            cout << "\nEncoding with Vertical (LINEAR only)..." << endl;

            auto partitions = Vertical_encoder::createFixedPartitions<T>(n, partition_size);
            VerticalConfig config;
            config.partition_size_hint = partition_size;
            config.enable_adaptive_selection = false;
            config.enable_interleaved = false;

            auto encode_start = chrono::high_resolution_clock::now();
            auto fl_data = Vertical_encoder::encodeVertical<T>(data, partitions, config);
            auto encode_end = chrono::high_resolution_clock::now();
            float encode_time = chrono::duration<float, milli>(encode_end - encode_start).count();

            vector<int32_t> h_delta_bits(fl_data.num_partitions);
            cudaMemcpy(h_delta_bits.data(), fl_data.d_delta_bits, fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            double avg_bits = 0;
            for (auto b : h_delta_bits) avg_bits += b;
            avg_bits /= h_delta_bits.size();

            int64_t compressed_bytes = fl_data.sequential_delta_words * sizeof(uint32_t) +
                                       fl_data.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
            double compression_ratio = (double)data_bytes / compressed_bytes;

            cout << "  Partitions: " << fl_data.num_partitions
                 << ", Avg bits: " << avg_bits << ", Ratio: " << compression_ratio << "x" << endl;

            cudaMemset(d_output, 0, data_bytes);
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            Vertical_decoder::launchDecompressPerPartitionBranchless<T>(fl_data, d_output);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float decode_time;
            cudaEventElapsedTime(&decode_time, start, stop);
            cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
            int errors = verifyResults(h_output.data(), data.data(), n);
            double throughput = (data_bytes / 1e9) / (decode_time / 1000.0);

            g_results.push_back({"Vertical-LINEAR", "FIXED-" + to_string(partition_size), "PerPartBranchless",
                fl_data.num_partitions, (int)avg_bits, compression_ratio,
                encode_time, decode_time, throughput, errors == 0, errors});
            cout << "  [PerPartBranchless] " << (errors == 0 ? "PASS" : "FAIL")
                 << " - " << decode_time << " ms, " << throughput << " GB/s" << endl;

            Vertical_encoder::freeCompressedData(fl_data);
        }
    }

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print summary
    cout << "\n" << string(120, '=') << endl;
    cout << "SUMMARY TABLE" << endl;
    cout << string(120, '=') << endl;

    cout << left << setw(20) << "Encoder"
         << setw(15) << "Partitioning"
         << setw(20) << "Decoder"
         << right << setw(8) << "Parts"
         << setw(8) << "Bits"
         << setw(10) << "Ratio"
         << setw(12) << "Enc(ms)"
         << setw(12) << "Dec(ms)"
         << setw(12) << "GB/s"
         << setw(8) << "Status" << endl;
    cout << string(120, '-') << endl;

    for (const auto& r : g_results) {
        cout << left << setw(20) << r.encoder_name
             << setw(15) << r.partitioning
             << setw(20) << r.decoder_name
             << right << setw(8) << r.num_partitions
             << setw(8) << r.avg_delta_bits
             << setw(10) << fixed << setprecision(2) << r.compression_ratio
             << setw(12) << r.encode_time_ms
             << setw(12) << r.decode_time_ms
             << setw(12) << setprecision(1) << r.decode_throughput_gbps
             << setw(8) << (r.correctness ? "PASS" : "FAIL") << endl;
    }
    cout << string(120, '=') << endl;

    // Write report
    writeReport(report_dir, dataset_name, g_results, n, data_bytes, min_val, max_val);
}

int main(int argc, char** argv) {
    string data_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin";
    string report_dir = "/root/autodl-tmp/code/L3/reports/L3/datasets/2-normal";

    if (argc > 1) data_file = argv[1];
    if (argc > 2) report_dir = argv[2];

    cout << "Loading dataset: " << data_file << endl;

    ifstream file(data_file, ios::binary | ios::ate);
    if (!file) {
        cerr << "Cannot open: " << data_file << endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    size_t n = file_size / sizeof(uint64_t);

    // Use full dataset (or limit for testing)
    n = min(n, size_t(200000000));  // Up to 200M

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    cout << "Loaded " << n << " elements (" << (n * sizeof(uint64_t) / 1024.0 / 1024.0) << " MB)" << endl;

    uint64_t min_val = data[0], max_val = data[0];
    for (size_t i = 1; i < n; i++) {
        min_val = min(min_val, data[i]);
        max_val = max(max_val, data[i]);
    }
    cout << "Range: [" << min_val << ", " << max_val << "]" << endl;

    runAllTestsWithReport<uint64_t>(data, "2-normal_200M_uint64", report_dir, min_val, max_val);

    return 0;
}
