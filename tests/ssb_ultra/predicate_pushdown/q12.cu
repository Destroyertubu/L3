/**
 * @file q12.cu
 * @brief SSB Q1.2 Implementation - Predicate Pushdown Strategy
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t DATE_199401_MIN = 19940101;
constexpr uint32_t DATE_199401_MAX = 19940131;

__global__ void q12_predicate_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    const uint32_t* __restrict__ lo_extendedprice,
    int num_rows, unsigned long long* revenue)
{
    __shared__ unsigned long long s_sum[BLOCK_THREADS];
    uint32_t orderdate[ITEMS_PER_THREAD], discount[ITEMS_PER_THREAD];
    uint32_t quantity[ITEMS_PER_THREAD], extendedprice[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);
    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockPredAndGTE(orderdate, DATE_199401_MIN, selection_flags, block_items);
    BlockPredAndLTE(orderdate, DATE_199401_MAX, selection_flags, block_items);

    BlockPredLoad(lo_discount + block_start, discount, block_items, selection_flags);
    BlockPredAndGTE(discount, static_cast<uint32_t>(4), selection_flags, block_items);
    BlockPredAndLTE(discount, static_cast<uint32_t>(6), selection_flags, block_items);

    BlockPredLoad(lo_quantity + block_start, quantity, block_items, selection_flags);
    BlockPredAndGTE(quantity, static_cast<uint32_t>(26), selection_flags, block_items);
    BlockPredAndLTE(quantity, static_cast<uint32_t>(35), selection_flags, block_items);

    BlockPredLoad(lo_extendedprice + block_start, extendedprice, block_items, selection_flags);

    unsigned long long local_sum = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (selection_flags[i]) {
            local_sum += static_cast<unsigned long long>(extendedprice[i]) *
                        static_cast<unsigned long long>(discount[i]);
        }
    }

    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_sum[0] > 0) atomicAdd(revenue, s_sum[0]);
}

void runQ12PredicatePushdown(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    timer.start();
    acc_orderdate.cachePartitionBounds();
    acc_discount.cachePartitionBounds();
    acc_quantity.cachePartitionBounds();
    acc_extendedprice.cachePartitionBounds();
    timer.stop();
    float bounds_cache_ms = timer.elapsed_ms();

    std::vector<int> candidate_partitions;
    acc_orderdate.getCandidatePartitions(DATE_199401_MIN, DATE_199401_MAX, candidate_partitions);

    std::cout << "Predicate pushdown: " << candidate_partitions.size() << "/" << acc_orderdate.getNumPartitions()
              << " partitions" << std::endl;

    int total_elements = acc_orderdate.getTotalElementsForPartitions(candidate_partitions);
    if (total_elements == 0) {
        timing.data_load_ms = bounds_cache_ms;
        timing.kernel_ms = 0;
        timing.hash_build_ms = 0;
        timing.total_ms = bounds_cache_ms;
        std::cout << "\n=== Q1.2 Results (Predicate Pushdown) ===" << std::endl;
        std::cout << "Revenue: 0" << std::endl;
        return;
    }

    uint32_t *d_orderdate, *d_discount, *d_quantity, *d_extendedprice;
    cudaMalloc(&d_orderdate, total_elements * sizeof(uint32_t));
    cudaMalloc(&d_discount, total_elements * sizeof(uint32_t));
    cudaMalloc(&d_quantity, total_elements * sizeof(uint32_t));
    cudaMalloc(&d_extendedprice, total_elements * sizeof(uint32_t));

    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    timer.start();
    acc_orderdate.decompressPartitionsDirect(candidate_partitions, d_orderdate);
    acc_discount.decompressPartitionsDirect(candidate_partitions, d_discount);
    acc_quantity.decompressPartitionsDirect(candidate_partitions, d_quantity);
    acc_extendedprice.decompressPartitionsDirect(candidate_partitions, d_extendedprice);

    int grid_size = (total_elements + TILE_SIZE - 1) / TILE_SIZE;
    q12_predicate_kernel<<<grid_size, BLOCK_THREADS>>>(d_orderdate, d_discount, d_quantity, d_extendedprice, total_elements, d_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.data_load_ms = bounds_cache_ms;
    timing.hash_build_ms = 0;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.2 Results (Predicate Pushdown) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    timing.print("Q1.2");

    cudaFree(d_orderdate); cudaFree(d_discount); cudaFree(d_quantity); cudaFree(d_extendedprice); cudaFree(d_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.2 - Predicate Pushdown Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ12PredicatePushdown(data, timing);
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ12PredicatePushdown(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
