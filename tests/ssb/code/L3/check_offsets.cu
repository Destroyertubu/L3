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
    std::vector<int64_t> h_offsets(num_partitions);
    cudaMemcpy(h_offsets.data(), data.lo_orderdate.d_interleaved_offsets,
               num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Check if offsets are as expected (partition_idx * WORDS_PER_PARTITION)
    constexpr int BIT_WIDTH = 16;  // BW_ORDERDATE
    constexpr int MINI_VECTOR_SIZE = 2048;
    constexpr int WORDS_PER_PARTITION = (MINI_VECTOR_SIZE * BIT_WIDTH + 31) / 32;  // = 1024

    std::cout << "Expected WORDS_PER_PARTITION: " << WORDS_PER_PARTITION << std::endl;
    std::cout << "First 10 offsets:" << std::endl;
    for (int i = 0; i < 10; i++) {
        int64_t expected = static_cast<int64_t>(i) * WORDS_PER_PARTITION;
        std::cout << "  [" << i << "] actual=" << h_offsets[i] << ", expected=" << expected
                  << (h_offsets[i] == expected ? " OK" : " MISMATCH") << std::endl;
    }
    std::cout << "Last partition offset:" << std::endl;
    int64_t expected_last = static_cast<int64_t>(num_partitions - 1) * WORDS_PER_PARTITION;
    std::cout << "  [" << (num_partitions-1) << "] actual=" << h_offsets[num_partitions-1]
              << ", expected=" << expected_last << std::endl;
    return 0;
}
