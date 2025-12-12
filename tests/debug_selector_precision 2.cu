/**
 * Debug: Compare CPU vs GPU selector on specific partitions
 * to identify precision differences
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "adaptive_selector.cuh"

using namespace std;

int main() {
    string data_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin";

    cout << "Loading dataset: " << data_file << endl;

    ifstream file(data_file, ios::binary | ios::ate);
    if (!file) {
        cerr << "Cannot open: " << data_file << endl;
        return 1;
    }

    size_t file_size = file.tellg();
    file.seekg(0, ios::beg);
    size_t n = file_size / sizeof(uint64_t);

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    cout << "Loaded " << n << " elements" << endl;

    int partition_size = 1024;
    int num_partitions = n / partition_size;

    // Upload data to GPU
    uint64_t* d_data;
    cudaMalloc(&d_data, n * sizeof(uint64_t));
    cudaMemcpy(d_data, data.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // Prepare partition indices
    vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        h_start[i] = i * partition_size;
        h_end[i] = (i + 1) * partition_size;
    }

    int32_t* d_start;
    int32_t* d_end;
    cudaMalloc(&d_start, num_partitions * sizeof(int32_t));
    cudaMalloc(&d_end, num_partitions * sizeof(int32_t));
    cudaMemcpy(d_start, h_start.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, h_end.data(), num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Allocate decisions
    adaptive_selector::ModelDecision<uint64_t>* d_decisions;
    cudaMalloc(&d_decisions, num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>));

    // Run GPU selector
    adaptive_selector::launchAdaptiveSelectorFullPolynomial<uint64_t>(
        d_data, d_start, d_end, num_partitions, d_decisions, 0);
    cudaDeviceSynchronize();

    // Copy GPU decisions back
    vector<adaptive_selector::ModelDecision<uint64_t>> gpu_decisions(num_partitions);
    cudaMemcpy(gpu_decisions.data(), d_decisions,
               num_partitions * sizeof(adaptive_selector::ModelDecision<uint64_t>),
               cudaMemcpyDeviceToHost);

    // Compare specific partitions
    cout << "\n=== Comparing CPU vs GPU selector on sample partitions ===" << endl;
    cout << setw(8) << "Pid" << " | "
         << setw(10) << "CPU Model" << " | " << setw(6) << "CPUBit" << " | "
         << setw(10) << "GPU Model" << " | " << setw(6) << "GPUBit" << " | "
         << setw(6) << "Match" << endl;
    cout << string(70, '-') << endl;

    const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR+BP"};

    // Sample partitions at different positions
    vector<int> test_pids;
    for (int i = 0; i < 10; i++) test_pids.push_back(i);  // First 10
    for (int i = 1000; i < 1010; i++) test_pids.push_back(i);  // Around 1000
    for (int i = 50000; i < 50010; i++) test_pids.push_back(i);  // Middle
    for (int i = 100000; i < 100010; i++) test_pids.push_back(i);  // After middle

    int mismatch_count = 0;
    int bits_diff_count = 0;
    double total_cpu_bits = 0;
    double total_gpu_bits = 0;

    for (int pid : test_pids) {
        if (pid >= num_partitions) continue;

        // CPU decision
        auto cpu_dec = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), h_start[pid], h_end[pid]);

        // GPU decision (already computed)
        auto& gpu_dec = gpu_decisions[pid];

        bool model_match = (cpu_dec.model_type == gpu_dec.model_type);
        bool bits_match = (cpu_dec.delta_bits == gpu_dec.delta_bits);

        if (!model_match) mismatch_count++;
        if (!bits_match) bits_diff_count++;

        total_cpu_bits += cpu_dec.delta_bits;
        total_gpu_bits += gpu_dec.delta_bits;

        cout << setw(8) << pid << " | "
             << setw(10) << model_names[cpu_dec.model_type] << " | "
             << setw(6) << cpu_dec.delta_bits << " | "
             << setw(10) << model_names[gpu_dec.model_type] << " | "
             << setw(6) << gpu_dec.delta_bits << " | "
             << setw(6) << (model_match && bits_match ? "OK" : "DIFF") << endl;

        // If there's a mismatch, print detailed debug info
        if (!model_match || !bits_match) {
            cout << "    CPU params: ["
                 << scientific << setprecision(6)
                 << cpu_dec.params[0] << ", " << cpu_dec.params[1] << ", "
                 << cpu_dec.params[2] << ", " << cpu_dec.params[3] << "]" << endl;
            cout << "    GPU params: ["
                 << gpu_dec.params[0] << ", " << gpu_dec.params[1] << ", "
                 << gpu_dec.params[2] << ", " << gpu_dec.params[3] << "]" << endl;
            cout << "    CPU cost: " << fixed << setprecision(1) << cpu_dec.estimated_cost
                 << ", GPU cost: " << gpu_dec.estimated_cost << endl;
        }
    }

    int samples = test_pids.size();
    cout << "\n=== Summary ===" << endl;
    cout << "Model mismatches: " << mismatch_count << " / " << samples << endl;
    cout << "Bits differences: " << bits_diff_count << " / " << samples << endl;
    cout << "Avg CPU bits: " << fixed << setprecision(1) << (total_cpu_bits / samples) << endl;
    cout << "Avg GPU bits: " << (total_gpu_bits / samples) << endl;

    // Full dataset comparison statistics
    cout << "\n=== Full dataset statistics (sampling every 100th) ===" << endl;

    int for_match = 0;
    int poly2_cpu = 0, poly2_gpu = 0;
    int poly3_cpu = 0, poly3_gpu = 0;
    double bits_cpu_total = 0, bits_gpu_total = 0;
    int samples_full = 0;

    for (int pid = 0; pid < num_partitions; pid += 100) {
        auto cpu_dec = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), h_start[pid], h_end[pid]);
        auto& gpu_dec = gpu_decisions[pid];

        samples_full++;
        bits_cpu_total += cpu_dec.delta_bits;
        bits_gpu_total += gpu_dec.delta_bits;

        if (cpu_dec.model_type == MODEL_FOR_BITPACK && gpu_dec.model_type == MODEL_FOR_BITPACK)
            for_match++;
        if (cpu_dec.model_type == MODEL_POLYNOMIAL2) poly2_cpu++;
        if (gpu_dec.model_type == MODEL_POLYNOMIAL2) poly2_gpu++;
        if (cpu_dec.model_type == MODEL_POLYNOMIAL3) poly3_cpu++;
        if (gpu_dec.model_type == MODEL_POLYNOMIAL3) poly3_gpu++;
    }

    cout << "Samples: " << samples_full << endl;
    cout << "FOR match: " << for_match << endl;
    cout << "POLY2 - CPU: " << poly2_cpu << ", GPU: " << poly2_gpu << endl;
    cout << "POLY3 - CPU: " << poly3_cpu << ", GPU: " << poly3_gpu << endl;
    cout << "Avg bits - CPU: " << (bits_cpu_total / samples_full)
         << ", GPU: " << (bits_gpu_total / samples_full) << endl;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_decisions);

    return 0;
}
