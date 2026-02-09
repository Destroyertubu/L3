/**
 * Check params immediately after compression (no cache)
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstring>
#define L3_VERTICAL_2048_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: " << argv[0] << " <data_path>" << std::endl; return 1; }
    ssb::SSBDataCompressedVertical data;
    // Force fresh compression (no cache)
    data.loadAndCompress(argv[1], 2048);

    int np = data.lo_orderdate.num_partitions;
    std::vector<double> h_params(12);
    cudaMemcpy(h_params.data(), data.lo_orderdate.d_model_params, 12 * sizeof(double), cudaMemcpyDeviceToHost);

    std::cout << "\nlo_orderdate params AFTER FRESH COMPRESSION:" << std::endl;
    for (int p = 0; p < 3; p++) {
        std::cout << "  P" << p << ": ";
        for (int i = 0; i < 4; i++) {
            double val = h_params[p * 4 + i];
            uint64_t raw;
            memcpy(&raw, &val, sizeof(raw));
            std::cout << "p[" << i << "]=" << val << "(0x" << std::hex << raw << std::dec << ") ";
        }
        std::cout << std::endl;
    }

    // No library decoder call - just check params
    return 0;
}
