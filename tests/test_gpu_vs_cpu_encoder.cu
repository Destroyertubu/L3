/**
 * Test: Compare CPU vs GPU encoder performance and correctness
 *
 * Tests the new encodeVerticalGPU function against the original
 * encodeVertical function.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>
#include <map>

#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"

// Include encoder and decoder implementations
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

using namespace std;

template<typename T>
int verifyResults(const T* output, const T* original, size_t n) {
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        if (output[i] != original[i]) {
            if (errors == 0) {
                cout << "First error at i=" << i << ": expected=" << original[i]
                     << ", got=" << output[i] << endl;
            }
            errors++;
        }
    }
    return errors;
}

int main(int argc, char** argv) {
    string data_file = "data/sosd/2-normal_200M_uint64.bin";
    size_t max_elements = 200000000;  // Full 200M

    if (argc > 1) data_file = argv[1];
    if (argc > 2) max_elements = atol(argv[2]);

    cout << "Loading dataset: " << data_file << endl;

    ifstream file(data_file, ios::binary | ios::ate);
    if (!file) {
        cerr << "Cannot open: " << data_file << endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    size_t n = min(file_size / sizeof(uint64_t), max_elements);

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    size_t data_bytes = n * sizeof(uint64_t);
    cout << "Loaded " << n << " elements (" << (data_bytes / 1024.0 / 1024.0) << " MB)" << endl;

    int partition_size = 1024;
    cout << "Partition size: " << partition_size << endl;

    // Allocate output buffer
    uint64_t* d_output;
    cudaMalloc(&d_output, data_bytes);
    vector<uint64_t> h_output(n);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ========================================================================
    // Test 1: Original CPU-based encoder
    // ========================================================================
    cout << "\n" << string(80, '=') << endl;
    cout << "Test 1: Original CPU-based Encoder (encodeVertical)" << endl;
    cout << string(80, '=') << endl;

    VerticalConfig config_cpu;
    config_cpu.partition_size_hint = partition_size;
    config_cpu.enable_adaptive_selection = true;
    config_cpu.enable_interleaved = false;

    auto partitions = Vertical_encoder::createFixedPartitions<uint64_t>(n, partition_size);

    auto cpu_start = chrono::high_resolution_clock::now();
    auto fl_cpu = Vertical_encoder::encodeVertical<uint64_t>(data, partitions, config_cpu);
    auto cpu_end = chrono::high_resolution_clock::now();
    float cpu_encode_time = chrono::duration<float, milli>(cpu_end - cpu_start).count();

    // Get model distribution
    vector<int32_t> cpu_model_types(fl_cpu.num_partitions);
    vector<int32_t> cpu_delta_bits(fl_cpu.num_partitions);
    cudaMemcpy(cpu_model_types.data(), fl_cpu.d_model_types,
               fl_cpu.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_delta_bits.data(), fl_cpu.d_delta_bits,
               fl_cpu.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    map<int, int> cpu_model_counts;
    double cpu_total_bits = 0;
    for (int i = 0; i < fl_cpu.num_partitions; i++) {
        cpu_model_counts[cpu_model_types[i]]++;
        cpu_total_bits += cpu_delta_bits[i];
    }
    double cpu_avg_bits = cpu_total_bits / fl_cpu.num_partitions;

    cout << "Encode time: " << cpu_encode_time << " ms" << endl;
    cout << "Model distribution:" << endl;
    const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR+BP"};
    for (auto& kv : cpu_model_counts) {
        cout << "  " << setw(10) << model_names[kv.first] << ": "
             << setw(6) << kv.second << " ("
             << fixed << setprecision(1) << (100.0 * kv.second / fl_cpu.num_partitions) << "%)" << endl;
    }
    cout << "Average bits: " << setprecision(1) << cpu_avg_bits << endl;

    // Decode and verify
    cudaMemset(d_output, 0, data_bytes);
    cudaEventRecord(start);
    Vertical_decoder::decompressAll<uint64_t>(fl_cpu, d_output, DecompressMode::AUTO);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cpu_decode_time;
    cudaEventElapsedTime(&cpu_decode_time, start, stop);

    cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
    int cpu_errors = verifyResults(h_output.data(), data.data(), n);

    size_t cpu_compressed_bytes = fl_cpu.sequential_delta_words * sizeof(uint32_t) +
                                  fl_cpu.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
    double cpu_ratio = (double)data_bytes / cpu_compressed_bytes;
    double cpu_throughput = (data_bytes / 1e9) / (cpu_decode_time / 1000.0);

    cout << "Decode time: " << cpu_decode_time << " ms" << endl;
    cout << "Throughput: " << setprecision(1) << cpu_throughput << " GB/s" << endl;
    cout << "Compression ratio: " << setprecision(2) << cpu_ratio << "x" << endl;
    cout << "Correctness: " << (cpu_errors == 0 ? "PASS" : "FAIL") << endl;

    // ========================================================================
    // Test 2: New GPU-based encoder
    // ========================================================================
    cout << "\n" << string(80, '=') << endl;
    cout << "Test 2: New GPU-based Encoder (encodeVerticalGPU)" << endl;
    cout << string(80, '=') << endl;

    VerticalConfig config_gpu;
    config_gpu.partition_size_hint = partition_size;
    config_gpu.enable_adaptive_selection = true;
    config_gpu.enable_interleaved = false;

    auto gpu_start = chrono::high_resolution_clock::now();
    auto fl_gpu = Vertical_encoder::encodeVerticalGPU<uint64_t>(data, partition_size, config_gpu);
    auto gpu_end = chrono::high_resolution_clock::now();
    float gpu_encode_time = chrono::duration<float, milli>(gpu_end - gpu_start).count();

    // Get model distribution
    vector<int32_t> gpu_model_types(fl_gpu.num_partitions);
    vector<int32_t> gpu_delta_bits(fl_gpu.num_partitions);
    cudaMemcpy(gpu_model_types.data(), fl_gpu.d_model_types,
               fl_gpu.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_delta_bits.data(), fl_gpu.d_delta_bits,
               fl_gpu.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    map<int, int> gpu_model_counts;
    double gpu_total_bits = 0;
    for (int i = 0; i < fl_gpu.num_partitions; i++) {
        gpu_model_counts[gpu_model_types[i]]++;
        gpu_total_bits += gpu_delta_bits[i];
    }
    double gpu_avg_bits = gpu_total_bits / fl_gpu.num_partitions;

    cout << "Encode time: " << gpu_encode_time << " ms" << endl;
    cout << "Model distribution:" << endl;
    for (auto& kv : gpu_model_counts) {
        cout << "  " << setw(10) << model_names[kv.first] << ": "
             << setw(6) << kv.second << " ("
             << fixed << setprecision(1) << (100.0 * kv.second / fl_gpu.num_partitions) << "%)" << endl;
    }
    cout << "Average bits: " << setprecision(1) << gpu_avg_bits << endl;

    // Decode and verify
    cudaMemset(d_output, 0, data_bytes);
    cudaEventRecord(start);
    Vertical_decoder::decompressAll<uint64_t>(fl_gpu, d_output, DecompressMode::AUTO);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_decode_time;
    cudaEventElapsedTime(&gpu_decode_time, start, stop);

    cudaMemcpy(h_output.data(), d_output, data_bytes, cudaMemcpyDeviceToHost);
    int gpu_errors = verifyResults(h_output.data(), data.data(), n);

    size_t gpu_compressed_bytes = fl_gpu.sequential_delta_words * sizeof(uint32_t) +
                                  fl_gpu.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
    double gpu_ratio = (double)data_bytes / gpu_compressed_bytes;
    double gpu_throughput = (data_bytes / 1e9) / (gpu_decode_time / 1000.0);

    cout << "Decode time: " << gpu_decode_time << " ms" << endl;
    cout << "Throughput: " << setprecision(1) << gpu_throughput << " GB/s" << endl;
    cout << "Compression ratio: " << setprecision(2) << gpu_ratio << "x" << endl;
    cout << "Correctness: " << (gpu_errors == 0 ? "PASS" : "FAIL") << endl;

    // ========================================================================
    // Summary
    // ========================================================================
    cout << "\n" << string(80, '=') << endl;
    cout << "SUMMARY" << endl;
    cout << string(80, '=') << endl;

    cout << "\n" << setw(25) << "Metric" << " | "
         << setw(15) << "CPU Encoder" << " | "
         << setw(15) << "GPU Encoder" << " | "
         << setw(15) << "Speedup" << endl;
    cout << string(80, '-') << endl;

    cout << setw(25) << "Encode Time (ms)" << " | "
         << setw(15) << fixed << setprecision(2) << cpu_encode_time << " | "
         << setw(15) << gpu_encode_time << " | "
         << setw(14) << setprecision(1) << (cpu_encode_time / gpu_encode_time) << "x" << endl;

    cout << setw(25) << "Average Bits" << " | "
         << setw(15) << setprecision(1) << cpu_avg_bits << " | "
         << setw(15) << gpu_avg_bits << " | "
         << setw(15) << "-" << endl;

    cout << setw(25) << "Compression Ratio" << " | "
         << setw(15) << setprecision(2) << cpu_ratio << "x | "
         << setw(14) << gpu_ratio << "x | "
         << setw(15) << "-" << endl;

    cout << setw(25) << "Decode Throughput (GB/s)" << " | "
         << setw(15) << setprecision(1) << cpu_throughput << " | "
         << setw(15) << gpu_throughput << " | "
         << setw(15) << "-" << endl;

    cout << setw(25) << "Correctness" << " | "
         << setw(15) << (cpu_errors == 0 ? "PASS" : "FAIL") << " | "
         << setw(15) << (gpu_errors == 0 ? "PASS" : "FAIL") << " | "
         << setw(15) << "-" << endl;

    // Compare model distributions
    cout << "\nModel Distribution Comparison:" << endl;
    cout << setw(12) << "Model" << " | "
         << setw(12) << "CPU" << " | "
         << setw(12) << "GPU" << " | "
         << setw(10) << "Match" << endl;
    cout << string(50, '-') << endl;

    bool models_match = true;
    for (int m = 0; m <= 4; m++) {
        int cpu_count = cpu_model_counts.count(m) ? cpu_model_counts[m] : 0;
        int gpu_count = gpu_model_counts.count(m) ? gpu_model_counts[m] : 0;
        bool match = (cpu_count == gpu_count);
        if (!match) models_match = false;

        if (cpu_count > 0 || gpu_count > 0) {
            cout << setw(12) << model_names[m] << " | "
                 << setw(12) << cpu_count << " | "
                 << setw(12) << gpu_count << " | "
                 << setw(10) << (match ? "YES" : "NO") << endl;
        }
    }

    cout << "\n" << string(80, '=') << endl;
    if (cpu_errors == 0 && gpu_errors == 0 && models_match) {
        cout << "RESULT: GPU encoder is " << setprecision(1) << (cpu_encode_time / gpu_encode_time)
             << "x faster and produces identical results!" << endl;
    } else {
        cout << "RESULT: There are differences between CPU and GPU encoders." << endl;
        if (!models_match) {
            cout << "  - Model distributions differ (this may be expected due to floating-point differences)" << endl;
        }
        if (cpu_errors > 0) cout << "  - CPU encoder has " << cpu_errors << " errors" << endl;
        if (gpu_errors > 0) cout << "  - GPU encoder has " << gpu_errors << " errors" << endl;
    }
    cout << string(80, '=') << endl;

    // Cleanup
    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(fl_cpu);
    Vertical_encoder::freeCompressedData(fl_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (cpu_errors == 0 && gpu_errors == 0) ? 0 : 1;
}
