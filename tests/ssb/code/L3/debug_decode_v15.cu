/**
 * Decode first few values of each column and print them
 */
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#define L3_VERTICAL_2048_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "v15_common.cuh"

// Simple kernel to decode first partition using dynamic decode
__global__ void decode_first_partition(
    const CompressedDataVertical<uint32_t> col,
    uint32_t* output)
{
    int lane = threadIdx.x;
    uint32_t vals[v15::V15_VALUES_PER_THREAD];
    v15::decode_column_v15(col, 0, lane, vals);

    // Write first few values (lane 0 writes its values)
    if (lane == 0) {
        for (int v = 0; v < 10 && v < v15::V15_VALUES_PER_THREAD; v++) {
            output[v] = vals[v];
        }
    }
    // Also write lane 0's first value using the opt function
    if (lane == 0) {
        uint32_t opt_vals[v15::V15_VALUES_PER_THREAD];
        v15::decode_column_v15_opt<16>(col, 0, 0, opt_vals);
        for (int v = 0; v < 10 && v < v15::V15_VALUES_PER_THREAD; v++) {
            output[10 + v] = opt_vals[v];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::cout << "Usage: " << argv[0] << " <data_path> [cache_dir]" << std::endl; return 1; }
    ssb::SSBDataCompressedVertical data;
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (std::string(argv[1]) + "/compressed_cache_v15");
    data.loadOrCompress(argv[1], cache_dir, 2048);

    // Print model params for first partition
    int np = data.lo_orderdate.num_partitions;
    std::vector<double> h_params(np * 4);
    cudaMemcpy(h_params.data(), data.lo_orderdate.d_model_params, np * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "lo_orderdate partition 0 params: "
              << h_params[0] << ", " << h_params[1] << ", " << h_params[2] << ", " << h_params[3] << std::endl;

    // Decode first partition
    uint32_t* d_output;
    cudaMalloc(&d_output, 20 * sizeof(uint32_t));
    decode_first_partition<<<1, 32>>>(data.lo_orderdate, d_output);
    cudaDeviceSynchronize();

    uint32_t h_output[20];
    cudaMemcpy(h_output, d_output, 20 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << "lo_orderdate first 10 values (dynamic decode):" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << "  [" << i << "] = " << h_output[i] << std::endl;

    std::cout << "lo_orderdate first 10 values (opt decode BW=16):" << std::endl;
    for (int i = 0; i < 10; i++) std::cout << "  [" << i << "] = " << h_output[10 + i] << std::endl;

    // Also print params for extendedprice
    std::vector<double> h_params2(np * 4);
    cudaMemcpy(h_params2.data(), data.lo_extendedprice.d_model_params, np * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "lo_extendedprice partition 0 params: "
              << h_params2[0] << ", " << h_params2[1] << ", " << h_params2[2] << ", " << h_params2[3] << std::endl;

    // Also check raw data for discount
    std::vector<double> h_params3(np * 4);
    cudaMemcpy(h_params3.data(), data.lo_discount.d_model_params, np * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "lo_discount partition 0 params: "
              << h_params3[0] << ", " << h_params3[1] << ", " << h_params3[2] << ", " << h_params3[3] << std::endl;

    cudaFree(d_output);
    return 0;
}
