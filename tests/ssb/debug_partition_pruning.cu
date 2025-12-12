// Debug program to understand partition pruning
#include "L3_codec.hpp"
#include "ssb_utils.h"
#include "ssb_L3_utils.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Load full lo_orderdate column
    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);

    cout << "Loaded " << LO_LEN << " elements" << endl;

    // Compress with same partition size as SSB query
    CompressedDataL3<uint32_t>* compressed = compressData(lo_orderdate, 4096);

    cout << "Compression info:" << endl;
    cout << "  Num partitions: " << compressed->num_partitions << endl;

    // Download partition bounds
    vector<uint32_t> h_min(compressed->num_partitions);
    vector<uint32_t> h_max(compressed->num_partitions);

    CUDA_CHECK(cudaMemcpy(h_min.data(), compressed->d_partition_min_values,
                          compressed->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(), compressed->d_partition_max_values,
                          compressed->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Test predicate: 19930000 <= value < 19940000
    uint32_t pred_min = 19930000;
    uint32_t pred_max = 19940000;

    cout << "\nPredicate: " << pred_min << " <= value < " << pred_max << endl;
    cout << "\nAnalyzing partitions:" << endl;

    int prunable = 0;
    int not_prunable = 0;
    int count_sample = 0;

    for (int i = 0; i < compressed->num_partitions; i++) {
        bool can_prune = (h_max[i] < pred_min || h_min[i] >= pred_max);

        if (can_prune) {
            prunable++;
        } else {
            not_prunable++;
        }

        // Show first 20 partitions as samples
        if (i < 20 || can_prune) {
            if (count_sample < 30) {
                cout << "  Partition " << i << ": [" << h_min[i] << ", " << h_max[i] << "] ";
                if (can_prune) {
                    cout << "✓ PRUNABLE";
                } else {
                    cout << "✗ NOT PRUNABLE";
                }
                cout << endl;
                count_sample++;
            }
        }
    }

    cout << "\nSummary:" << endl;
    cout << "  Total partitions: " << compressed->num_partitions << endl;
    cout << "  Prunable: " << prunable << " (" << (100.0 * prunable / compressed->num_partitions) << "%)" << endl;
    cout << "  Not prunable: " << not_prunable << " (" << (100.0 * not_prunable / compressed->num_partitions) << "%)" << endl;

    freeCompressedData(compressed);

    return 0;
}
