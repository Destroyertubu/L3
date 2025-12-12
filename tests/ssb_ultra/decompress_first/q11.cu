/**
 * @file q11.cu
 * @brief SSB Q1.1 Implementation - Decompress First Strategy
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey
 *     AND d_year = 1993
 *     AND lo_discount BETWEEN 1 AND 3
 *     AND lo_quantity < 25;
 *
 * Strategy: Decompress all required columns first, then run Crystal-style query kernel
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

// SSB common headers
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"

// L3 headers
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

// ============================================================================
// Q1.1 Query Kernel
// ============================================================================

/**
 * @brief Q1.1 query kernel with Crystal-style block processing
 *
 * Processes LINEORDER table with predicates:
 * - lo_orderdate in year 1993 (date between 19930101 and 19931231)
 * - lo_discount between 1 and 3
 * - lo_quantity < 25
 *
 * Computes: SUM(lo_extendedprice * lo_discount)
 */
__global__ void q11_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    const uint32_t* __restrict__ lo_extendedprice,
    int num_rows,
    unsigned long long* revenue)
{
    // Shared memory for block reduction
    __shared__ unsigned long long s_sum[BLOCK_THREADS];

    // Items processed by this thread
    uint32_t orderdate[ITEMS_PER_THREAD];
    uint32_t discount[ITEMS_PER_THREAD];
    uint32_t quantity[ITEMS_PER_THREAD];
    uint32_t extendedprice[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    // Calculate block range
    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);

    if (block_items <= 0) return;

    // Initialize selection flags
    InitFlags(selection_flags, block_items);

    // Load orderdate and filter by year 1993
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockPredAndGTE(orderdate, static_cast<uint32_t>(19930101), selection_flags, block_items);
    BlockPredAndLTE(orderdate, static_cast<uint32_t>(19931231), selection_flags, block_items);

    // Load and filter by discount (1-3)
    BlockPredLoad(lo_discount + block_start, discount, block_items, selection_flags);
    BlockPredAndGTE(discount, static_cast<uint32_t>(1), selection_flags, block_items);
    BlockPredAndLTE(discount, static_cast<uint32_t>(3), selection_flags, block_items);

    // Load and filter by quantity (<25)
    BlockPredLoad(lo_quantity + block_start, quantity, block_items, selection_flags);
    BlockPredAndLT(quantity, static_cast<uint32_t>(25), selection_flags, block_items);

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

    // Atomic add to global result
    if (threadIdx.x == 0 && s_sum[0] > 0) {
        atomicAdd(revenue, s_sum[0]);
    }
}

// ============================================================================
// Q1.1 Execution Function
// ============================================================================

/**
 * @brief Execute Q1.1 with decompress-first strategy
 */
void runQ11DecompressFirst(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    // Allocate decompressed arrays
    uint32_t* d_lo_orderdate;
    uint32_t* d_lo_discount;
    uint32_t* d_lo_quantity;
    uint32_t* d_lo_extendedprice;

    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_discount, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_quantity, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_extendedprice, LO_LEN * sizeof(uint32_t));

    // Result
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Step 1: Decompress all required columns
    // -------------------------------------------------------------------------
    timer.start();

    // Create accessors for compressed columns
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    // Decompress all columns
    acc_orderdate.decompressAll(d_lo_orderdate);
    acc_discount.decompressAll(d_lo_discount);
    acc_quantity.decompressAll(d_lo_quantity);
    acc_extendedprice.decompressAll(d_lo_extendedprice);

    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // -------------------------------------------------------------------------
    // Step 2: Execute query kernel
    // -------------------------------------------------------------------------
    timer.start();

    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q11_kernel<<<grid_size, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice,
        LO_LEN, d_revenue
    );

    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();

    // No hash build for Q1
    timing.hash_build_ms = 0.0f;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    // -------------------------------------------------------------------------
    // Step 3: Get result
    // -------------------------------------------------------------------------
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.1 Results (Decompress-First) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    timing.print("Q1.1");

    // Cleanup
    cudaFree(d_lo_orderdate);
    cudaFree(d_lo_discount);
    cudaFree(d_lo_quantity);
    cudaFree(d_lo_extendedprice);
    cudaFree(d_revenue);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";

    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "=== SSB Q1.1 - Decompress First Strategy ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;

    // Load and compress SSB data
    SSBDataCompressedVertical data;

    auto start = std::chrono::high_resolution_clock::now();
    data.loadAndCompress(data_dir);
    auto end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\nData loading + compression time: " << load_time << " ms" << std::endl;
    std::cout << "Compression ratio: " << data.getCompressionRatio() << "x" << std::endl;

    // Run Q1.1
    QueryTiming timing;
    runQ11DecompressFirst(data, timing);

    // Warmup and benchmark
    std::cout << "\n=== Benchmark (3 runs) ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ11DecompressFirst(data, t);
        std::cout << "Run " << (i+1) << ": Total=" << t.total_ms << " ms"
                  << " (Decompress=" << t.data_load_ms << " ms, Kernel=" << t.kernel_ms << " ms)"
                  << std::endl;
    }

    // Cleanup
    data.free();

    return 0;
}
