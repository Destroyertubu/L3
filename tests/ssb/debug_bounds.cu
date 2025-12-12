// Debug program to verify partition bounds computation
#include "L3_codec.hpp"
#include "ssb_utils.h"
#include "ssb_L3_utils.cuh"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Load a small sample of lo_orderdate
    int sample_size = 100000;
    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", sample_size);

    cout << "Loaded " << sample_size << " elements" << endl;
    cout << "First 10 values: ";
    for (int i = 0; i < 10; i++) {
        cout << lo_orderdate[i] << " ";
    }
    cout << endl;

    // Compress with small partitions for easier debugging
    CompressedDataL3<uint32_t>* compressed = compressData(lo_orderdate, 4096);

    cout << "\nCompression info:" << endl;
    cout << "  Num partitions: " << compressed->num_partitions << endl;

    // Download partition bounds to host
    vector<uint32_t> h_min(compressed->num_partitions);
    vector<uint32_t> h_max(compressed->num_partitions);

    CUDA_CHECK(cudaMemcpy(h_min.data(), compressed->d_partition_min_values,
                          compressed->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(), compressed->d_partition_max_values,
                          compressed->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    cout << "\nFirst 10 partition bounds:" << endl;
    for (int i = 0; i < min(10, compressed->num_partitions); i++) {
        cout << "  Partition " << i << ": min=" << h_min[i] << ", max=" << h_max[i] << endl;
    }

    // Test predicate: 19930000 <= value < 19940000
    uint32_t pred_min = 19930000;
    uint32_t pred_max = 19940000;

    cout << "\nPredicate test: " << pred_min << " <= value < " << pred_max << endl;

    int prunable = 0;
    int not_prunable = 0;

    for (int i = 0; i < compressed->num_partitions; i++) {
        // Can prune if partition max < pred_min OR partition min >= pred_max
        if (h_max[i] < pred_min || h_min[i] >= pred_max) {
            prunable++;
        } else {
            not_prunable++;
        }
    }

    cout << "  Prunable partitions: " << prunable << " / " << compressed->num_partitions << endl;
    cout << "  Non-prunable partitions: " << not_prunable << " / " << compressed->num_partitions << endl;

    freeCompressedData(compressed);

    return 0;
}
