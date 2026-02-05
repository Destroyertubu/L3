/**
 * Debug test: Check what models are actually being selected by L3
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
#include "adaptive_selector.cuh"

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
    n = min(n, size_t(10000000));  // Just first 10M for quick test

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    cout << "Loaded " << n << " elements" << endl;

    // Create partitions
    int partition_size = 1024;
    int num_partitions = (n + partition_size - 1) / partition_size;

    cout << "Number of partitions: " << num_partitions << endl;

    // Allocate partition indices
    vector<int32_t> h_start_indices(num_partitions);
    vector<int32_t> h_end_indices(num_partitions);

    for (int i = 0; i < num_partitions; i++) {
        h_start_indices[i] = i * partition_size;
        h_end_indices[i] = min((i + 1) * partition_size, (int)n);
    }

    // Upload to GPU
    uint64_t* d_data;
    int32_t* d_start_indices;
    int32_t* d_end_indices;
    adaptive_selector::ModelDecision<uint64_t>* d_decisions;

    cudaMalloc(&d_data, n * sizeof(uint64_t));
    cudaMalloc(&d_start_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&d_end_indices, num_partitions * sizeof(int32_t));
    cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>));

    cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_indices, h_start_indices.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end_indices, h_end_indices.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Run adaptive selector (simplified version - LINEAR and FOR only)
    cout << "\n=== Testing SIMPLIFIED selector (LINEAR + FOR only) ===" << endl;
    adaptive_selector::launchAdaptiveSelector<uint64_t>(
        d_data, d_start_indices, d_end_indices, num_partitions, d_decisions
    );
    cudaDeviceSynchronize();

    vector<adaptive_selector::ModelDecision<uint64_t>> h_decisions_simple(num_partitions);
    cudaMemcpy(h_decisions_simple.data(), d_decisions,
               num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>),
               cudaMemcpyDeviceToHost);

    map<int, int> model_counts_simple;
    double total_bits_simple = 0;
    for (int i = 0; i < num_partitions; i++) {
        model_counts_simple[h_decisions_simple[i].model_type]++;
        total_bits_simple += h_decisions_simple[i].delta_bits;
    }

    cout << "Model selection distribution:" << endl;
    for (auto& kv : model_counts_simple) {
        cout << "  " << adaptive_selector::modelTypeName(kv.first) << ": "
             << kv.second << " (" << fixed << setprecision(1)
             << (100.0 * kv.second / num_partitions) << "%)" << endl;
    }
    cout << "Average delta_bits: " << (total_bits_simple / num_partitions) << endl;
    cout << "Theoretical compression: " << (64.0 / (total_bits_simple / num_partitions)) << "x" << endl;

    // Run FULL polynomial selector
    cout << "\n=== Testing FULL POLYNOMIAL selector (LINEAR + POLY2 + POLY3 + FOR) ===" << endl;
    adaptive_selector::launchAdaptiveSelectorFullPolynomial<uint64_t>(
        d_data, d_start_indices, d_end_indices, num_partitions, d_decisions
    );
    cudaDeviceSynchronize();

    vector<adaptive_selector::ModelDecision<uint64_t>> h_decisions_full(num_partitions);
    cudaMemcpy(h_decisions_full.data(), d_decisions,
               num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>),
               cudaMemcpyDeviceToHost);

    map<int, int> model_counts_full;
    double total_bits_full = 0;
    for (int i = 0; i < num_partitions; i++) {
        model_counts_full[h_decisions_full[i].model_type]++;
        total_bits_full += h_decisions_full[i].delta_bits;
    }

    cout << "Model selection distribution:" << endl;
    for (auto& kv : model_counts_full) {
        cout << "  " << adaptive_selector::modelTypeName(kv.first) << ": "
             << kv.second << " (" << fixed << setprecision(1)
             << (100.0 * kv.second / num_partitions) << "%)" << endl;
    }
    cout << "Average delta_bits: " << (total_bits_full / num_partitions) << endl;
    cout << "Theoretical compression: " << (64.0 / (total_bits_full / num_partitions)) << "x" << endl;

    // Print first 20 decisions for comparison
    cout << "\n=== First 20 partition decisions ===" << endl;
    cout << setw(6) << "Part" << " | "
         << setw(12) << "Simple" << " | "
         << setw(12) << "Full" << " | "
         << setw(8) << "Bits(S)" << " | "
         << setw(8) << "Bits(F)" << endl;
    cout << string(60, '-') << endl;

    for (int i = 0; i < min(20, num_partitions); i++) {
        cout << setw(6) << i << " | "
             << setw(12) << adaptive_selector::modelTypeName(h_decisions_simple[i].model_type) << " | "
             << setw(12) << adaptive_selector::modelTypeName(h_decisions_full[i].model_type) << " | "
             << setw(8) << h_decisions_simple[i].delta_bits << " | "
             << setw(8) << h_decisions_full[i].delta_bits << endl;
    }

    // Summary
    cout << "\n=== SUMMARY ===" << endl;
    cout << "Simple selector (LINEAR+FOR): " << (64.0 / (total_bits_simple / num_partitions)) << "x compression" << endl;
    cout << "Full selector (with POLY):    " << (64.0 / (total_bits_full / num_partitions)) << "x compression" << endl;
    cout << "Improvement:                  " << ((64.0 / (total_bits_full / num_partitions)) / (64.0 / (total_bits_simple / num_partitions))) << "x better" << endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_start_indices);
    cudaFree(d_end_indices);
    cudaFree(d_decisions);

    return 0;
}
