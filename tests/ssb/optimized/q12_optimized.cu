/**
 * @file q12_optimized.cu
 * @brief SSB Q1.2 Implementation - OPTIMIZED with multi-stream parallel decompression
 *
 * Query: SUM(lo_extendedprice * lo_discount) WHERE date in Jan 1994, discount 4-6, quantity 26-35
 *
 * Note: Even though Q1.2 has low selectivity (~1.2% for Jan 1994), parallel decompress
 * is still faster than random access due to sequential memory access patterns.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t Q12_DATE_MIN = 19940101;
constexpr uint32_t Q12_DATE_MAX = 19940131;
constexpr uint32_t DISCOUNT_MIN = 4;
constexpr uint32_t DISCOUNT_MAX = 6;
constexpr uint32_t QUANTITY_MIN = 26;
constexpr uint32_t QUANTITY_MAX = 35;

__global__ void q12_kernel_optimized(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    const uint32_t* __restrict__ lo_extendedprice,
    int num_rows,
    unsigned long long* revenue)
{
    __shared__ unsigned long long s_sum[BLOCK_THREADS];

    uint32_t orderdate[ITEMS_PER_THREAD];
    uint32_t discount[ITEMS_PER_THREAD];
    uint32_t quantity[ITEMS_PER_THREAD];
    uint32_t extendedprice[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);

    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);

    // Load orderdate and filter by Jan 1994
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockPredAndGTE(orderdate, Q12_DATE_MIN, selection_flags, block_items);
    BlockPredAndLTE(orderdate, Q12_DATE_MAX, selection_flags, block_items);

    // Load and filter by discount (4-6)
    BlockPredLoad(lo_discount + block_start, discount, block_items, selection_flags);
    BlockPredAndGTE(discount, DISCOUNT_MIN, selection_flags, block_items);
    BlockPredAndLTE(discount, DISCOUNT_MAX, selection_flags, block_items);

    // Load and filter by quantity (26-35)
    BlockPredLoad(lo_quantity + block_start, quantity, block_items, selection_flags);
    BlockPredAndGTE(quantity, QUANTITY_MIN, selection_flags, block_items);
    BlockPredAndLTE(quantity, QUANTITY_MAX, selection_flags, block_items);

    // Load extendedprice for passing rows
    BlockPredLoad(lo_extendedprice + block_start, extendedprice, block_items, selection_flags);

    // Compute local sum: extendedprice * discount
    unsigned long long local_sum = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        if (selection_flags[i]) {
            local_sum += static_cast<unsigned long long>(extendedprice[i]) *
                        static_cast<unsigned long long>(discount[i]);
        }
    }

    // Block reduction
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_sum[0] > 0) {
        atomicAdd(revenue, s_sum[0]);
    }
}

void runQ12Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    // Create 4 streams for parallel decompression
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    int total_rows = acc_orderdate.getTotalElements();

    // Allocate decompressed arrays
    uint32_t *d_orderdate, *d_discount, *d_quantity, *d_extendedprice;
    cudaMalloc(&d_orderdate, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_discount, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_quantity, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_extendedprice, total_rows * sizeof(uint32_t));

    // Result
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    // =========================================================================
    // Step 1: PARALLEL decompress all 4 columns
    // =========================================================================
    timer.start();

    acc_orderdate.decompressAll(d_orderdate, streams[0]);
    acc_discount.decompressAll(d_discount, streams[1]);
    acc_quantity.decompressAll(d_quantity, streams[2]);
    acc_extendedprice.decompressAll(d_extendedprice, streams[3]);

    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // =========================================================================
    // Step 2: Execute query kernel
    // =========================================================================
    timer.start();

    int grid_size = (total_rows + TILE_SIZE - 1) / TILE_SIZE;
    q12_kernel_optimized<<<grid_size, BLOCK_THREADS>>>(
        d_orderdate, d_discount, d_quantity, d_extendedprice,
        total_rows, d_revenue);

    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();

    timing.hash_build_ms = 0;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    // Get result
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.2 Results (OPTIMIZED) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Parallel decompress 4 cols: " << timing.data_load_ms << " ms" << std::endl;
    std::cout << "  Query kernel:               " << timing.kernel_ms << " ms" << std::endl;
    timing.print("Q1.2");

    // Cleanup
    cudaFree(d_orderdate);
    cudaFree(d_discount);
    cudaFree(d_quantity);
    cudaFree(d_extendedprice);
    cudaFree(d_revenue);

    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.2 - OPTIMIZED (Parallel Decompress) ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ12Optimized(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ12Optimized(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
