/**
 * Check raw model params bytes for SSB data
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#define L3_VERTICAL_2048_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"

void checkParamsRaw(const char* name, const CompressedDataVertical<uint32_t>& col) {
    int np = col.num_partitions;
    // Read first 3 partitions' params (each 4 doubles = 32 bytes)
    std::vector<double> h_params(12);
    cudaMemcpy(h_params.data(), col.d_model_params, 12 * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << name << " params (first 3 partitions):" << std::endl;
    for (int p = 0; p < 3; p++) {
        std::cout << "  P" << p << ": ";
        for (int i = 0; i < 4; i++) {
            double val = h_params[p * 4 + i];
            uint64_t raw;
            memcpy(&raw, &val, sizeof(raw));
            std::cout << "p[" << i << "]=" << val
                     << " (0x" << std::hex << raw << std::dec << ") ";
        }
        std::cout << std::endl;
    }

    // Also check delta_bits and model_types
    std::vector<int32_t> h_bits(3), h_models(3);
    cudaMemcpy(h_bits.data(), col.d_delta_bits, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_models.data(), col.d_model_types, 3 * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::cout << "  delta_bits: " << h_bits[0] << ", " << h_bits[1] << ", " << h_bits[2] << std::endl;
    std::cout << "  model_types: " << h_models[0] << ", " << h_models[1] << ", " << h_models[2] << std::endl;

    // Check first few raw delta words
    std::vector<uint32_t> h_deltas(10);
    cudaMemcpy(h_deltas.data(), col.d_interleaved_deltas, 10 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    std::cout << "  First 10 delta words: ";
    for (int i = 0; i < 10; i++) std::cout << h_deltas[i] << " ";
    std::cout << std::endl;

    // Check interleaved_offsets
    std::vector<int64_t> h_off(3);
    cudaMemcpy(h_off.data(), col.d_interleaved_offsets, 3 * sizeof(int64_t), cudaMemcpyDeviceToHost);
    std::cout << "  First 3 offsets: " << h_off[0] << ", " << h_off[1] << ", " << h_off[2] << std::endl;

    // Also test the official library decoder
    // Decode first partition to compare
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: " << argv[0] << " <data_path> [cache_dir]" << std::endl; return 1; }
    ssb::SSBDataCompressedVertical data;
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (std::string(argv[1]) + "/compressed_cache_v15");
    data.loadOrCompress(argv[1], cache_dir, 2048);

    checkParamsRaw("lo_orderdate", data.lo_orderdate);
    checkParamsRaw("lo_discount", data.lo_discount);
    checkParamsRaw("lo_quantity", data.lo_quantity);

    // Also read raw data for first few values
    std::cout << "\nRaw LINEORDER5 (orderdate) first 5 values:" << std::endl;
    auto raw = ssb::loadColumnFromFile<uint32_t>(std::string(argv[1]) + "/LINEORDER5.bin", 5);
    for (int i = 0; i < 5; i++) std::cout << "  [" << i << "] = " << raw[i] << std::endl;

    return 0;
}
