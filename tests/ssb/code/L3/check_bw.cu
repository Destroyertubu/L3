/**
 * Check actual bit-widths of compressed SSB data
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <map>
#define L3_VERTICAL_2048_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"

void checkColumn(const char* name, const CompressedDataVertical<uint32_t>& col) {
    int np = col.num_partitions;
    std::vector<int32_t> h_bits(np), h_models(np);
    cudaMemcpy(h_bits.data(), col.d_delta_bits, np * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_models.data(), col.d_model_types, np * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::map<int, int> bw_counts;
    std::map<int, int> model_counts;
    for (int i = 0; i < np; i++) {
        bw_counts[h_bits[i]]++;
        model_counts[h_models[i]]++;
    }

    std::cout << name << " (" << np << " partitions):" << std::endl;
    std::cout << "  Bit-widths: ";
    for (auto& [bw, cnt] : bw_counts)
        std::cout << bw << ":" << cnt << " ";
    std::cout << std::endl;
    std::cout << "  Models: ";
    for (auto& [m, cnt] : model_counts)
        std::cout << m << ":" << cnt << " ";
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: " << argv[0] << " <data_path> [cache_dir]" << std::endl; return 1; }
    ssb::SSBDataCompressedVertical data;
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (std::string(argv[1]) + "/compressed_cache_v15");
    data.loadOrCompress(argv[1], cache_dir, 2048);

    checkColumn("lo_orderdate", data.lo_orderdate);
    checkColumn("lo_quantity", data.lo_quantity);
    checkColumn("lo_discount", data.lo_discount);
    checkColumn("lo_extendedprice", data.lo_extendedprice);
    checkColumn("lo_revenue", data.lo_revenue);
    checkColumn("lo_supplycost", data.lo_supplycost);
    checkColumn("lo_custkey", data.lo_custkey);
    checkColumn("lo_partkey", data.lo_partkey);
    checkColumn("lo_suppkey", data.lo_suppkey);
    return 0;
}
