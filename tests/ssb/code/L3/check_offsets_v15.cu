/**
 * Check actual offsets vs fixed offsets for SSB data
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define L3_VERTICAL_2048_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"

void checkOffsets(const char* name, int bw, const CompressedDataVertical<uint32_t>& col) {
    int np = col.num_partitions;
    std::vector<int64_t> h_offsets(np);
    std::vector<int32_t> h_start(np), h_end(np);
    cudaMemcpy(h_offsets.data(), col.d_interleaved_offsets, np * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_start.data(), col.d_start_indices, np * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end.data(), col.d_end_indices, np * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int words_per_partition = (2048 * bw + 31) / 32;
    int mismatches = 0;
    std::cout << name << " (BW=" << bw << ", words_per_partition=" << words_per_partition << "):" << std::endl;
    std::cout << "  First 5 offsets: ";
    for (int i = 0; i < 5 && i < np; i++) {
        int64_t expected = (int64_t)i * words_per_partition;
        std::cout << h_offsets[i] << "(exp:" << expected << ",sz:" << (h_end[i]-h_start[i]) << ") ";
        if (h_offsets[i] != expected) mismatches++;
    }
    std::cout << std::endl;

    // Check all offsets
    for (int i = 5; i < np; i++) {
        int64_t expected = (int64_t)i * words_per_partition;
        if (h_offsets[i] != expected) mismatches++;
    }
    std::cout << "  Total mismatches: " << mismatches << " / " << np << std::endl;
    std::cout << "  Total delta words: " << col.interleaved_delta_words << std::endl;

    // Also check model params for first few partitions
    std::vector<double> h_params(np * 4);
    cudaMemcpy(h_params.data(), col.d_model_params, np * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "  First 3 base_values (params[0]): ";
    for (int i = 0; i < 3; i++) std::cout << h_params[i*4] << " ";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: " << argv[0] << " <data_path> [cache_dir]" << std::endl; return 1; }
    ssb::SSBDataCompressedVertical data;
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (std::string(argv[1]) + "/compressed_cache_v15");
    data.loadOrCompress(argv[1], cache_dir, 2048);

    checkOffsets("lo_orderdate", 16, data.lo_orderdate);
    checkOffsets("lo_quantity", 6, data.lo_quantity);
    checkOffsets("lo_discount", 4, data.lo_discount);
    checkOffsets("lo_extendedprice", 16, data.lo_extendedprice);
    return 0;
}
