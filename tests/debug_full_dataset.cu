/**
 * Debug: Check what model is selected for full 200M dataset
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <iomanip>
#include <map>

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

    // Test CPU version on different segments
    int partition_size = 1024;

    // Test partitions at different positions
    vector<int> test_positions = {0, 1000, 10000, 50000, 100000, 150000, 195000};

    cout << "\n=== Testing partitions at different positions ===" << endl;
    cout << setw(8) << "Part#" << " | "
         << setw(10) << "Model" << " | "
         << setw(8) << "Bits" << " | "
         << setw(12) << "Cost" << endl;
    cout << string(50, '-') << endl;

    for (int pid : test_positions) {
        int start = pid * partition_size;
        int end = min(start + partition_size, (int)n);

        if (end > (int)n) break;

        auto decision = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), start, end);

        const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR+BP"};
        cout << setw(8) << pid << " | "
             << setw(10) << model_names[decision.model_type] << " | "
             << setw(8) << decision.delta_bits << " | "
             << setw(12) << fixed << setprecision(1) << decision.estimated_cost << endl;
    }

    // Full statistics
    cout << "\n=== Full dataset statistics (every 100th partition) ===" << endl;

    int num_partitions = n / partition_size;
    map<int, int> model_counts;
    map<int, double> model_bits;
    double total_bits = 0;

    for (int i = 0; i < num_partitions; i += 100) {
        int start = i * partition_size;
        int end = min(start + partition_size, (int)n);

        auto decision = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), start, end);

        model_counts[decision.model_type]++;
        model_bits[decision.model_type] += decision.delta_bits;
        total_bits += decision.delta_bits;
    }

    int samples = 0;
    for (auto& kv : model_counts) samples += kv.second;

    cout << "Sampled partitions: " << samples << endl;
    cout << "Model distribution:" << endl;

    const char* model_names[] = {"CONSTANT", "LINEAR", "POLY2", "POLY3", "FOR+BP"};
    for (auto& kv : model_counts) {
        double avg_bits = model_bits[kv.first] / kv.second;
        cout << "  " << setw(10) << model_names[kv.first] << ": "
             << setw(6) << kv.second << " ("
             << fixed << setprecision(1) << (100.0 * kv.second / samples) << "%)"
             << " avg bits: " << setprecision(1) << avg_bits << endl;
    }

    double avg_bits = total_bits / samples;
    cout << "\nOverall average bits: " << avg_bits << endl;
    cout << "Theoretical compression: " << (64.0 / avg_bits) << "x" << endl;

    return 0;
}
