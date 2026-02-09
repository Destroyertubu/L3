#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define L3_VERTICAL_1024_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"

int main() {
    ssb::SSBDataCompressedVertical data;
    data.loadOrCompress("/root/autodl-tmp/code/test/ssb_data",
                       "/root/autodl-tmp/code/test/ssb_data/compressed_cache_v14_2048", 2048);

    int num_partitions = data.lo_orderdate.num_partitions;
    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    cudaMemcpy(h_start.data(), data.lo_orderdate.d_start_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end.data(), data.lo_orderdate.d_end_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int count_2048 = 0, count_gt_2048 = 0, count_lt_2048 = 0;
    int max_size = 0, min_size = 9999999;
    for (int i = 0; i < num_partitions; i++) {
        int size = h_end[i] - h_start[i];
        if (size == 2048) count_2048++;
        else if (size > 2048) count_gt_2048++;
        else count_lt_2048++;
        if (size > max_size) max_size = size;
        if (size < min_size) min_size = size;
    }
    std::cout << "Total partitions: " << num_partitions << std::endl;
    std::cout << "  Size = 2048: " << count_2048 << std::endl;
    std::cout << "  Size > 2048: " << count_gt_2048 << std::endl;
    std::cout << "  Size < 2048: " << count_lt_2048 << std::endl;
    std::cout << "  Min size: " << min_size << std::endl;
    std::cout << "  Max size: " << max_size << std::endl;
    return 0;
}
