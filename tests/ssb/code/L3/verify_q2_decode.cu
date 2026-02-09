#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "ssb_data_loader.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "v15_common.cuh"
#include "v15_padding.cuh"

using namespace ssb;
using namespace v15;

// Decode kernel for a single column
template<int BW>
__global__ void decode_column_kernel(
    const CompressedDataVertical<uint32_t> col,
    uint32_t* __restrict__ out,
    int num_partitions,
    int total_rows)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;

    int partition_start = col.d_start_indices[partition_idx];
    int partition_end = col.d_end_indices[partition_idx];
    int partition_size = partition_end - partition_start;

    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;
    if (num_mv == 0) return;

    uint32_t vals[V15_VALUES_PER_THREAD];
    decode_column_v15_opt<BW>(col, partition_idx, lane_id, vals);

    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int local_idx = v * V15_WARP_SIZE + lane_id;
        int global_idx = partition_start + local_idx;
        if (global_idx < total_rows) {
            out[global_idx] = vals[v];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 1;
    std::string data_path = argv[1];
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (data_path + "/compressed_cache_v15");

    // Load raw data - CORRECT file mapping based on ssb_data_loader.hpp:
    // LO_PARTKEY = "LINEORDER3.bin", LO_SUPPKEY = "LINEORDER4.bin"
    auto h_partkey_raw = loadColumnFromFile<uint32_t>(data_path + "/LINEORDER3.bin", LO_LEN);
    auto h_suppkey_raw = loadColumnFromFile<uint32_t>(data_path + "/LINEORDER4.bin", LO_LEN);
    auto h_revenue_raw = loadColumnFromFile<uint32_t>(data_path + "/LINEORDER12.bin", LO_LEN);

    std::cout << "Loaded raw data: " << h_suppkey_raw.size() << " rows\n";
    std::cout << "partkey[0]=" << h_partkey_raw[0] << " (from LINEORDER3)\n";
    std::cout << "suppkey[0]=" << h_suppkey_raw[0] << " (from LINEORDER4)\n";

    // Load compressed data
    SSBDataCompressedVertical data;
    data.loadOrCompress(data_path.c_str(), cache_dir, MINI_VECTOR_SIZE);

    // Pad columns
    v15::padColumnToFixedOffset<BW_SUPPKEY>(data.lo_suppkey);
    v15::padColumnToFixedOffset<BW_PARTKEY>(data.lo_partkey);
    v15::padColumnToFixedOffset<BW_REVENUE>(data.lo_revenue);

    int total_rows = data.lo_suppkey.total_values;
    int num_partitions = data.lo_suppkey.num_partitions;

    std::cout << "Compressed: " << total_rows << " rows, " << num_partitions << " partitions\n";

    // Allocate output buffers
    uint32_t* d_decoded_suppkey;
    uint32_t* d_decoded_partkey;
    uint32_t* d_decoded_revenue;
    cudaMalloc(&d_decoded_suppkey, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_decoded_partkey, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_decoded_revenue, total_rows * sizeof(uint32_t));

    // Decode
    decode_column_kernel<BW_SUPPKEY><<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_suppkey, d_decoded_suppkey, num_partitions, total_rows);
    decode_column_kernel<BW_PARTKEY><<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_partkey, d_decoded_partkey, num_partitions, total_rows);
    decode_column_kernel<BW_REVENUE><<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_revenue, d_decoded_revenue, num_partitions, total_rows);
    cudaDeviceSynchronize();

    // Copy back to host
    std::vector<uint32_t> h_suppkey(total_rows), h_partkey(total_rows), h_revenue(total_rows);
    cudaMemcpy(h_suppkey.data(), d_decoded_suppkey, total_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_partkey.data(), d_decoded_partkey, total_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_revenue.data(), d_decoded_revenue, total_rows * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Verify
    int suppkey_errors = 0, partkey_errors = 0, revenue_errors = 0;
    for (int i = 0; i < total_rows && i < 120000000; i++) {
        if (h_suppkey[i] != h_suppkey_raw[i]) suppkey_errors++;
        if (h_partkey[i] != h_partkey_raw[i]) partkey_errors++;
        if (h_revenue[i] != h_revenue_raw[i]) revenue_errors++;
    }

    std::cout << "\nDecoding verification:\n";
    std::cout << "  lo_suppkey errors: " << suppkey_errors << " / " << total_rows << "\n";
    std::cout << "  lo_partkey errors: " << partkey_errors << " / " << total_rows << "\n";
    std::cout << "  lo_revenue errors: " << revenue_errors << " / " << total_rows << "\n";

    if (suppkey_errors > 0 || partkey_errors > 0 || revenue_errors > 0) {
        std::cout << "\nFirst 10 errors:\n";
        int shown = 0;
        for (int i = 0; i < total_rows && shown < 10; i++) {
            if (h_suppkey[i] != h_suppkey_raw[i] ||
                h_partkey[i] != h_partkey_raw[i] ||
                h_revenue[i] != h_revenue_raw[i]) {
                std::cout << "  [" << i << "] suppkey: " << h_suppkey[i] << " vs " << h_suppkey_raw[i]
                          << ", partkey: " << h_partkey[i] << " vs " << h_partkey_raw[i]
                          << ", revenue: " << h_revenue[i] << " vs " << h_revenue_raw[i] << "\n";
                shown++;
            }
        }
    } else {
        std::cout << "\nAll Q2.x columns decode correctly!\n";
    }

    cudaFree(d_decoded_suppkey);
    cudaFree(d_decoded_partkey);
    cudaFree(d_decoded_revenue);

    return 0;
}
