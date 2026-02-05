/**
 * End-to-end test: Verify model selection and compression on normal dataset
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

int main() {
    string data_file = "data/sosd/2-normal_200M_uint64.bin";

    cout << "Loading dataset: " << data_file << endl;

    ifstream file(data_file, ios::binary | ios::ate);
    if (!file) {
        cerr << "Cannot open: " << data_file << endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    size_t n = file_size / sizeof(uint64_t);
    n = min(n, size_t(10000000));  // First 10M for testing

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    cout << "Loaded " << n << " elements (" << (n * 8 / 1024.0 / 1024.0) << " MB)" << endl;

    // Create fixed partitions
    int partition_size = 1024;
    auto partitions = Vertical_encoder::createFixedPartitions<uint64_t>(n, partition_size);
    cout << "Created " << partitions.size() << " partitions of size " << partition_size << endl;

    // Configure with adaptive selection
    VerticalConfig config;
    config.partition_size_hint = partition_size;
    config.enable_adaptive_selection = true;
    config.enable_interleaved = false;

    // Encode
    cout << "\nEncoding with adaptive selection..." << endl;
    auto start_time = chrono::high_resolution_clock::now();
    auto fl_data = Vertical_encoder::encodeVertical<uint64_t>(data, partitions, config);
    auto end_time = chrono::high_resolution_clock::now();
    float encode_time = chrono::duration<float, milli>(end_time - start_time).count();

    // Get model type distribution
    vector<int32_t> h_model_types(fl_data.num_partitions);
    vector<int32_t> h_delta_bits(fl_data.num_partitions);
    cudaMemcpy(h_model_types.data(), fl_data.d_model_types,
               fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits.data(), fl_data.d_delta_bits,
               fl_data.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    map<int, int> model_counts;
    map<int, double> model_bits;
    double total_bits = 0;

    for (int i = 0; i < fl_data.num_partitions; i++) {
        model_counts[h_model_types[i]]++;
        model_bits[h_model_types[i]] += h_delta_bits[i];
        total_bits += h_delta_bits[i];
    }

    cout << "\nModel selection distribution:" << endl;
    const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR+BP"};
    for (auto& kv : model_counts) {
        double avg_bits = model_bits[kv.first] / kv.second;
        cout << "  " << setw(10) << model_names[kv.first] << ": "
             << setw(6) << kv.second << " ("
             << fixed << setprecision(1) << (100.0 * kv.second / fl_data.num_partitions) << "%)"
             << " avg bits: " << setprecision(1) << avg_bits << endl;
    }

    double avg_bits = total_bits / fl_data.num_partitions;
    cout << "\nOverall average bits: " << setprecision(1) << avg_bits << endl;
    cout << "Theoretical compression: " << (64.0 / avg_bits) << "x" << endl;

    // Calculate actual compression ratio
    size_t original_bytes = n * sizeof(uint64_t);
    size_t compressed_bytes = fl_data.sequential_delta_words * sizeof(uint32_t) +
                              fl_data.num_partitions * (sizeof(int32_t) * 4 + sizeof(double) * 4);
    double compression_ratio = (double)original_bytes / compressed_bytes;

    cout << "\nActual compression:" << endl;
    cout << "  Original: " << (original_bytes / 1024.0 / 1024.0) << " MB" << endl;
    cout << "  Compressed: " << (compressed_bytes / 1024.0 / 1024.0) << " MB" << endl;
    cout << "  Ratio: " << setprecision(2) << compression_ratio << "x" << endl;
    cout << "  Encode time: " << encode_time << " ms" << endl;

    // Verify correctness by decoding
    cout << "\nDecoding and verifying..." << endl;

    uint64_t* d_output;
    cudaMalloc(&d_output, n * sizeof(uint64_t));
    cudaMemset(d_output, 0, n * sizeof(uint64_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    Vertical_decoder::decompressAll<uint64_t>(fl_data, d_output, DecompressMode::AUTO);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float decode_time;
    cudaEventElapsedTime(&decode_time, start, stop);

    vector<uint64_t> h_output(n);
    cudaMemcpy(h_output.data(), d_output, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    // Verify
    int errors = 0;
    for (size_t i = 0; i < n && errors < 10; i++) {
        if (h_output[i] != data[i]) {
            if (errors == 0) {
                cout << "First error at i=" << i << ": expected=" << data[i]
                     << ", got=" << h_output[i] << endl;
            }
            errors++;
        }
    }

    double throughput = (original_bytes / 1e9) / (decode_time / 1000.0);

    cout << "\nResults:" << endl;
    cout << "  Decode time: " << decode_time << " ms" << endl;
    cout << "  Throughput: " << setprecision(1) << throughput << " GB/s" << endl;
    cout << "  Correctness: " << (errors == 0 ? "PASS" : "FAIL") << endl;

    // Print first 10 partition details
    cout << "\nFirst 10 partitions:" << endl;
    cout << setw(5) << "Part" << " | "
         << setw(8) << "Model" << " | "
         << setw(6) << "Bits" << endl;
    cout << string(30, '-') << endl;

    for (int i = 0; i < min(10, (int)fl_data.num_partitions); i++) {
        cout << setw(5) << i << " | "
             << setw(8) << model_names[h_model_types[i]] << " | "
             << setw(6) << h_delta_bits[i] << endl;
    }

    // Cleanup
    cudaFree(d_output);
    Vertical_encoder::freeCompressedData(fl_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
