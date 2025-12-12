/**
 * Debug test: Check CPU version of adaptive selector
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
    n = min(n, size_t(1000000));  // First 1M for quick test

    vector<uint64_t> data(n);
    file.read(reinterpret_cast<char*>(data.data()), n * sizeof(uint64_t));
    file.close();

    cout << "Loaded " << n << " elements" << endl;

    // Test CPU version on a few partitions
    int partition_size = 1024;
    int num_test = 20;

    cout << "\n=== Testing CPU Version (computeDecisionCPU) ===" << endl;
    cout << setw(6) << "Part" << " | "
         << setw(10) << "Model" << " | "
         << setw(8) << "Bits" << " | "
         << setw(12) << "Cost" << endl;
    cout << string(50, '-') << endl;

    map<int, int> model_counts;
    double total_bits = 0;

    for (int i = 0; i < min(num_test, (int)(n / partition_size)); i++) {
        int start = i * partition_size;
        int end = min(start + partition_size, (int)n);

        auto decision = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), start, end);

        model_counts[decision.model_type]++;
        total_bits += decision.delta_bits;

        if (i < 20) {
            cout << setw(6) << i << " | "
                 << setw(10) << adaptive_selector::modelTypeName(decision.model_type) << " | "
                 << setw(8) << decision.delta_bits << " | "
                 << setw(12) << fixed << setprecision(1) << decision.estimated_cost << endl;
        }
    }

    cout << "\nModel distribution:" << endl;
    for (auto& kv : model_counts) {
        cout << "  " << adaptive_selector::modelTypeName(kv.first) << ": " << kv.second << endl;
    }
    cout << "Average bits: " << (total_bits / num_test) << endl;

    // Now test full dataset
    cout << "\n=== Testing full dataset (first 1M elements) ===" << endl;

    int total_partitions = n / partition_size;
    model_counts.clear();
    total_bits = 0;

    for (int i = 0; i < total_partitions; i++) {
        int start = i * partition_size;
        int end = min(start + partition_size, (int)n);

        auto decision = adaptive_selector::computeDecisionCPU<uint64_t>(
            data.data(), start, end);

        model_counts[decision.model_type]++;
        total_bits += decision.delta_bits;
    }

    cout << "Total partitions: " << total_partitions << endl;
    cout << "Model distribution:" << endl;
    for (auto& kv : model_counts) {
        cout << "  " << setw(10) << adaptive_selector::modelTypeName(kv.first) << ": "
             << setw(6) << kv.second << " ("
             << fixed << setprecision(1) << (100.0 * kv.second / total_partitions) << "%)" << endl;
    }
    cout << "Average bits: " << fixed << setprecision(1) << (total_bits / total_partitions) << endl;
    cout << "Theoretical compression: " << (64.0 / (total_bits / total_partitions)) << "x" << endl;

    return 0;
}
