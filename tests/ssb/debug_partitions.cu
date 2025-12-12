#include "L3_codec.hpp"
#include "ssb_utils.h"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    // Load lo_orderdate column
    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    
    // Compress with partition size 1024
    CompressedDataL3<uint32_t>* c_data = compressData(lo_orderdate, 1024);
    
    // Download partition min/max values
    vector<uint32_t> h_min(c_data->num_partitions);
    vector<uint32_t> h_max(c_data->num_partitions);
    
    cudaMemcpy(h_min.data(), c_data->d_partition_min_values,
               c_data->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max.data(), c_data->d_partition_max_values,
               c_data->num_partitions * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    
    cout << "Total partitions: " << c_data->num_partitions << endl;
    cout << "First 20 partitions min/max:" << endl;
    
    for (int i = 0; i < min(20, c_data->num_partitions); i++) {
        cout << "Partition " << i << ": min=" << h_min[i] << ", max=" << h_max[i] << endl;
    }
    
    // Check for Q1.1 filter (19930000 - 19940000)
    int pruned_count = 0;
    for (int i = 0; i < c_data->num_partitions; i++) {
        if (h_max[i] < 19930000 || h_min[i] >= 19940000) {
            pruned_count++;
        }
    }
    
    cout << "\nFor Q1.1 filter [19930000, 19940000):" << endl;
    cout << "Partitions that could be pruned: " << pruned_count << " / " << c_data->num_partitions << endl;
    cout << "Pruning rate: " << (100.0 * pruned_count / c_data->num_partitions) << "%" << endl;
    
    freeCompressedData(c_data);
    return 0;
}
