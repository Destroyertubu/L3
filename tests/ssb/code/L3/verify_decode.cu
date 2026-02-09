/**
 * Verify decode correctness for MV_SIZE=256 configuration
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "v15_common.cuh"
#include "v15_padding.cuh"

using namespace ssb;
using namespace v15;

template<int BIT_WIDTH>
__global__ void verify_decode_kernel(
    const CompressedDataVertical<uint32_t> compressed,
    const uint32_t* __restrict__ original,
    int total_rows,
    int* __restrict__ error_count,
    int* __restrict__ first_error_idx,
    uint32_t* __restrict__ first_error_expected,
    uint32_t* __restrict__ first_error_got)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= compressed.num_partitions) return;

    int lane_id = threadIdx.x;
    int partition_start = compressed.d_start_indices[partition_idx];
    int partition_size = compressed.d_end_indices[partition_idx] - partition_start;

    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;

    if (num_mv > 0) {
        uint32_t decoded[V15_VALUES_PER_THREAD];
        decode_column_v15_opt<BIT_WIDTH>(compressed, partition_idx, lane_id, decoded);

        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            int local_idx = v * V15_WARP_SIZE + lane_id;
            int global_idx = partition_start + local_idx;

            if (global_idx < total_rows) {
                uint32_t expected = original[global_idx];
                if (decoded[v] != expected) {
                    int old = atomicAdd(error_count, 1);
                    if (old == 0) {
                        *first_error_idx = global_idx;
                        *first_error_expected = expected;
                        *first_error_got = decoded[v];
                    }
                }
            }
        }
    }
}

template<int BIT_WIDTH>
int verify_column(const CompressedDataVertical<uint32_t>& compressed,
                  const std::vector<uint32_t>& h_original,
                  const char* col_name) {
    uint32_t* d_original;
    cudaMalloc(&d_original, h_original.size() * sizeof(uint32_t));
    cudaMemcpy(d_original, h_original.data(), h_original.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    int* d_error_count;
    int* d_first_error_idx;
    uint32_t* d_first_error_expected;
    uint32_t* d_first_error_got;

    cudaMalloc(&d_error_count, sizeof(int));
    cudaMalloc(&d_first_error_idx, sizeof(int));
    cudaMalloc(&d_first_error_expected, sizeof(uint32_t));
    cudaMalloc(&d_first_error_got, sizeof(uint32_t));
    cudaMemset(d_error_count, 0, sizeof(int));

    int num_partitions = compressed.num_partitions;
    verify_decode_kernel<BIT_WIDTH><<<num_partitions, 32>>>(
        compressed,
        d_original,
        h_original.size(),
        d_error_count,
        d_first_error_idx,
        d_first_error_expected,
        d_first_error_got);
    cudaDeviceSynchronize();

    int error_count;
    int first_error_idx;
    uint32_t first_error_expected, first_error_got;

    cudaMemcpy(&error_count, d_error_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&first_error_idx, d_first_error_idx, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&first_error_expected, d_first_error_expected, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&first_error_got, d_first_error_got, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::cout << col_name << " (BW=" << BIT_WIDTH << "): ";
    if (error_count > 0) {
        std::cout << "FAILED - " << error_count << " errors. First at idx "
                  << first_error_idx << ": expected " << first_error_expected
                  << ", got " << first_error_got << std::endl;
    } else {
        std::cout << "OK" << std::endl;
    }

    cudaFree(d_original);
    cudaFree(d_error_count);
    cudaFree(d_first_error_idx);
    cudaFree(d_first_error_expected);
    cudaFree(d_first_error_got);

    return error_count;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path>" << std::endl;
        return 1;
    }

    std::cout << "MINI_VECTOR_SIZE = " << MINI_VECTOR_SIZE << std::endl;
    std::cout << "VALUES_PER_THREAD = " << VALUES_PER_THREAD << std::endl;
    std::cout << std::endl;

    // Load original data
    std::cout << "Loading original columns..." << std::endl;
    auto h_orderdate = loadColumnFromFile<uint32_t>(std::string(argv[1]) + "/LINEORDER5.bin", LO_LEN);
    auto h_quantity = loadColumnFromFile<uint32_t>(std::string(argv[1]) + "/LINEORDER8.bin", LO_LEN);
    auto h_discount = loadColumnFromFile<uint32_t>(std::string(argv[1]) + "/LINEORDER11.bin", LO_LEN);
    auto h_extendedprice = loadColumnFromFile<uint32_t>(std::string(argv[1]) + "/LINEORDER9.bin", LO_LEN);

    // Load compressed data
    std::cout << "Loading compressed data..." << std::endl;
    SSBDataCompressedVertical data;
    std::string cache_dir = std::string(argv[1]) + "/compressed_cache_v15";
    data.loadOrCompress(argv[1], cache_dir, MINI_VECTOR_SIZE);

    // Pad to fixed offset
    std::cout << "Padding to fixed offset..." << std::endl;
    v15::padColumnToFixedOffset<BW_ORDERDATE>(data.lo_orderdate);
    v15::padColumnToFixedOffset<BW_QUANTITY>(data.lo_quantity);
    v15::padColumnToFixedOffset<BW_DISCOUNT>(data.lo_discount);
    v15::padColumnToFixedOffset<BW_EXTENDEDPRICE>(data.lo_extendedprice);

    // Verify each column
    std::cout << std::endl << "Verifying decode..." << std::endl;
    int total_errors = 0;
    total_errors += verify_column<BW_ORDERDATE>(data.lo_orderdate, h_orderdate, "lo_orderdate");
    total_errors += verify_column<BW_QUANTITY>(data.lo_quantity, h_quantity, "lo_quantity");
    total_errors += verify_column<BW_DISCOUNT>(data.lo_discount, h_discount, "lo_discount");
    total_errors += verify_column<BW_EXTENDEDPRICE>(data.lo_extendedprice, h_extendedprice, "lo_extendedprice");

    std::cout << std::endl;
    if (total_errors == 0) {
        std::cout << "All columns decoded correctly!" << std::endl;
    } else {
        std::cout << "Total decode errors: " << total_errors << std::endl;
    }

    return total_errors > 0 ? 1 : 0;
}
