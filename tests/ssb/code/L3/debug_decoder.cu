/**
 * Debug decoder - prints sample decoded values and metadata
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#define L3_VERTICAL_1024_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "v15_common.cuh"

using namespace ssb;
using namespace v15;

__global__ void debug_print_metadata(
    const CompressedDataVertical<uint32_t> col,
    int partition_idx)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Partition %d metadata:\n", partition_idx);
        printf("  start_idx: %d\n", col.d_start_indices[partition_idx]);
        printf("  end_idx: %d\n", col.d_end_indices[partition_idx]);
        printf("  model_type: %d\n", col.d_model_types[partition_idx]);
        printf("  delta_bits: %d\n", col.d_delta_bits[partition_idx]);
        printf("  interleaved_offset: %lld\n", (long long)col.d_interleaved_offsets[partition_idx]);
        printf("  num_mini_vectors: %d\n", col.d_num_mini_vectors[partition_idx]);
        printf("  params[0]: %f\n", col.d_model_params[partition_idx * 4 + 0]);
        printf("  params[1]: %f\n", col.d_model_params[partition_idx * 4 + 1]);
        printf("  params[2]: %f\n", col.d_model_params[partition_idx * 4 + 2]);
        printf("  params[3]: %f\n", col.d_model_params[partition_idx * 4 + 3]);
    }
}

__global__ void debug_decode_values(
    const CompressedDataVertical<uint32_t> col,
    int partition_idx,
    uint32_t* out_values,
    int num_values)
{
    int lane_id = threadIdx.x;
    if (lane_id >= 32) return;

    uint32_t values[V15_VALUES_PER_THREAD];
    decode_column_v15(col, partition_idx, lane_id, values);

    // Copy first few values to output
    for (int v = 0; v < V15_VALUES_PER_THREAD && (v * 32 + lane_id) < num_values; v++) {
        int idx = v * 32 + lane_id;
        out_values[idx] = values[v];
    }
}

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    std::cout << "V13 Decoder Debug Tool" << std::endl;

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];

    SSBDataCompressedVertical data;
    std::string cache_dir = data_path + "/compressed_cache_v14_2048";
    data.loadOrCompress(data_path, cache_dir, 2048);

    // Check for any CUDA errors so far
    CHECK_CUDA(cudaGetLastError());

    std::cout << "\n=== Debug: lo_orderdate ===" << std::endl;
    std::cout << "Total partitions: " << data.lo_orderdate.num_partitions << std::endl;
    std::cout << "Total values: " << data.lo_orderdate.total_values << std::endl;

    // Read metadata from device
    int num_parts = data.lo_orderdate.num_partitions;
    std::vector<int32_t> h_start(num_parts), h_end(num_parts);
    std::vector<int32_t> h_model_types(num_parts);
    std::vector<int32_t> h_delta_bits(num_parts);
    std::vector<int64_t> h_offsets(num_parts);
    std::vector<int32_t> h_num_mvs(num_parts);
    std::vector<double> h_params(num_parts * 4);

    CHECK_CUDA(cudaMemcpy(h_start.data(), data.lo_orderdate.d_start_indices,
               num_parts * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_end.data(), data.lo_orderdate.d_end_indices,
               num_parts * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_model_types.data(), data.lo_orderdate.d_model_types,
               num_parts * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_delta_bits.data(), data.lo_orderdate.d_delta_bits,
               num_parts * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_offsets.data(), data.lo_orderdate.d_interleaved_offsets,
               num_parts * sizeof(int64_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_num_mvs.data(), data.lo_orderdate.d_num_mini_vectors,
               num_parts * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_params.data(), data.lo_orderdate.d_model_params,
               num_parts * 4 * sizeof(double), cudaMemcpyDeviceToHost));

    // Print first 5 partitions metadata
    for (int p = 0; p < 5 && p < num_parts; p++) {
        std::cout << "\n--- Partition " << p << " (host-side) ---" << std::endl;
        std::cout << "  start_idx: " << h_start[p] << std::endl;
        std::cout << "  end_idx: " << h_end[p] << std::endl;
        std::cout << "  model_type: " << h_model_types[p] << std::endl;
        std::cout << "  delta_bits: " << h_delta_bits[p] << std::endl;
        std::cout << "  interleaved_offset: " << h_offsets[p] << std::endl;
        std::cout << "  num_mini_vectors: " << h_num_mvs[p] << std::endl;
        std::cout << "  params[0..3]: " << h_params[p*4] << ", " << h_params[p*4+1]
                  << ", " << h_params[p*4+2] << ", " << h_params[p*4+3] << std::endl;
    }

    // Decode first partition
    int num_values_to_decode = 64;  // First 64 values
    uint32_t* d_values;
    CHECK_CUDA(cudaMalloc(&d_values, num_values_to_decode * sizeof(uint32_t)));
    CHECK_CUDA(cudaMemset(d_values, 0xFF, num_values_to_decode * sizeof(uint32_t)));  // Fill with 0xFFFFFFFF

    debug_decode_values<<<1, 32>>>(data.lo_orderdate, 0, d_values, num_values_to_decode);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint32_t> h_values(num_values_to_decode);
    CHECK_CUDA(cudaMemcpy(h_values.data(), d_values, num_values_to_decode * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    std::cout << "\n=== First 64 decoded lo_orderdate values ===" << std::endl;
    for (int i = 0; i < num_values_to_decode; i++) {
        if (i % 8 == 0) std::cout << "  [" << i << "]: ";
        std::cout << h_values[i] << " ";
        if (i % 8 == 7) std::cout << std::endl;
    }

    CHECK_CUDA(cudaFree(d_values));

    return 0;
}
