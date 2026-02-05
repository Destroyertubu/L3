/**
 * @file q11_fls.cu
 * @brief SSB Q1.1 Implementation - Vertical-style FOR+BitPack
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder
 *   WHERE lo_orderdate >= 19930101 AND lo_orderdate <= 19931231
 *     AND lo_discount >= 1 AND lo_discount <= 3
 *     AND lo_quantity < 25;
 *
 * This implementation matches Vertical-GPU structure:
 * - 32 threads x 32 items = 1024 elements per tile
 * - FOR+BitPack encoding (delta = value - min)
 * - Compile-time specialized unpack functions
 * - Crystal-style predicates and reduction
 *
 * Key difference from Vertical: predicates use original values
 * (we add back base after unpack, so no predicate adjustment needed)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

// L3 FLS headers
#include "common/fls_constants.cuh"
#include "common/fls_unpack.cuh"
#include "common/fls_block_load.cuh"
#include "common/crystal_fls.cuh"
#include "common/ssb_data_fls.hpp"

using namespace l3_fls;

// ============================================================================
// Q1.1 Query Constants (local to avoid conflicts)
// ============================================================================

namespace {
constexpr int ORDERDATE_LO = 19930101;  // Year 1993 start
constexpr int ORDERDATE_HI = 19931231;  // Year 1993 end
constexpr int DISCOUNT_LO = 1;
constexpr int DISCOUNT_HI = 3;
constexpr int QUANTITY_HI = 25;         // quantity < 25
}  // anonymous namespace

// Kernel configuration: 32 threads x 32 items = 1024 per tile
constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;    // 32
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;  // 32
constexpr int TILE_SIZE = FLS_TILE_SIZE;            // 1024

// ============================================================================
// Q1.1 Kernel - Vertical Style
// ============================================================================

/**
 * @brief Q1.1 kernel matching Vertical-GPU structure
 *
 * Each block processes one 1024-element tile.
 * Thread i unpacks values at strided positions [i, i+32, i+64, ...]
 */
__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q11_fls_kernel(
    const uint32_t* __restrict__ enc_lo_orderdate,
    const uint32_t* __restrict__ enc_lo_discount,
    const uint32_t* __restrict__ enc_lo_quantity,
    const uint32_t* __restrict__ enc_lo_extendedprice,
    int32_t lo_orderdate_min,
    int32_t lo_discount_min,
    int32_t lo_quantity_min,
    int32_t lo_extendedprice_min,
    uint8_t lo_orderdate_bw,
    uint8_t lo_discount_bw,
    uint8_t lo_quantity_bw,
    uint8_t lo_extendedprice_bw,
    int64_t n_tup_lineorder,
    unsigned long long* __restrict__ revenue)
{
    // Thread-local arrays
    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    // Shared memory for reduction
    __shared__ long long buffer[32];

    long long sum = 0;

    // Tile offset calculation
    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    // Initialize selection flags
    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // ========================================
    // Column 1: lo_orderdate
    // Predicate: >= 19930101 AND <= 19931231
    // ========================================
    {
        int64_t orderdate_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_orderdate_bw * 32;
        unpack_for(enc_lo_orderdate + orderdate_tile_offset, items, lo_orderdate_min, lo_orderdate_bw);

        // Note: items already contains original values (base + delta)
        // So we use actual predicate values directly
        BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, ORDERDATE_LO, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, ORDERDATE_HI, selection_flags, num_tile_items);
    }

    // Early termination check
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Column 2: lo_quantity
    // Predicate: < 25
    // ========================================
    {
        int64_t quantity_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_quantity_bw * 32;
        unpack_for(enc_lo_quantity + quantity_tile_offset, items, lo_quantity_min, lo_quantity_bw);

        BlockPredAndLT<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, QUANTITY_HI, selection_flags, num_tile_items);
    }

    // Early termination check
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Column 3: lo_discount
    // Predicate: >= 1 AND <= 3
    // ========================================
    {
        int64_t discount_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_discount_bw * 32;
        unpack_for(enc_lo_discount + discount_tile_offset, items, lo_discount_min, lo_discount_bw);

        BlockPredAndGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, DISCOUNT_LO, selection_flags, num_tile_items);
        BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, DISCOUNT_HI, selection_flags, num_tile_items);
    }

    // Early termination check
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Column 4: lo_extendedprice
    // (loaded for selected rows only)
    // ========================================
    {
        int64_t extprice_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_extendedprice_bw * 32;
        unpack_for(enc_lo_extendedprice + extprice_tile_offset, items2, lo_extendedprice_min, lo_extendedprice_bw);
    }

    // ========================================
    // Aggregation: SUM(extendedprice * discount)
    // ========================================
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            // items still has discount values, items2 has extendedprice
            sum += static_cast<long long>(items[ITEM]) * static_cast<long long>(items2[ITEM]);
        }
    }

    // ========================================
    // Block-level reduction
    // ========================================
    __syncthreads();
    unsigned long long aggregate = BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, buffer);
    __syncthreads();

    // Thread 0 does atomic add
    if (threadIdx.x == 0 && aggregate > 0) {
        atomicAdd(revenue, aggregate);
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ11FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate result
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    // Calculate grid size
    int num_blocks = static_cast<int>(data.n_tiles);

    std::cout << "Q1.1 FLS Kernel Configuration:" << std::endl;
    std::cout << "  Blocks: " << num_blocks << std::endl;
    std::cout << "  Threads per block: " << BLOCK_THREADS << std::endl;
    std::cout << "  Items per thread: " << ITEMS_PER_THREAD << std::endl;
    std::cout << "  Tile size: " << TILE_SIZE << std::endl;

    // Warmup run
    q11_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_orderdate.d_encoded,
        data.lo_discount.d_encoded,
        data.lo_quantity.d_encoded,
        data.lo_extendedprice.d_encoded,
        data.lo_orderdate.min_value,
        data.lo_discount.min_value,
        data.lo_quantity.min_value,
        data.lo_extendedprice.min_value,
        data.lo_orderdate.bitwidth,
        data.lo_discount.bitwidth,
        data.lo_quantity.bitwidth,
        data.lo_extendedprice.bitwidth,
        data.n_tup_lineorder,
        d_revenue
    );
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (warmup): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Timed runs
    float total_time = 0.0f;
    const int NUM_RUNS = 3;

    for (int run = 0; run < NUM_RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));

        cudaEventRecord(start);

        q11_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_orderdate.d_encoded,
            data.lo_discount.d_encoded,
            data.lo_quantity.d_encoded,
            data.lo_extendedprice.d_encoded,
            data.lo_orderdate.min_value,
            data.lo_discount.min_value,
            data.lo_quantity.min_value,
            data.lo_extendedprice.min_value,
            data.lo_orderdate.bitwidth,
            data.lo_discount.bitwidth,
            data.lo_quantity.bitwidth,
            data.lo_extendedprice.bitwidth,
            data.n_tup_lineorder,
            d_revenue
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / NUM_RUNS;
    timing.total_ms = timing.kernel_ms;

    // Get result
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.1 Results (FLS) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Average kernel time: " << timing.kernel_ms << " ms" << std::endl;

    // Cleanup
    cudaFree(d_revenue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.1 - Vertical Style FOR+BitPack ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;

    // Load and encode data
    SSBDataFLS data;
    data.loadAndEncode(data_dir);
    data.printStats();

    // Run query
    std::cout << "\n--- Running Q1.1 ---" << std::endl;
    FLSQueryTiming timing;
    runQ11FLS(data, timing);

    // Cleanup
    data.free();

    return 0;
}
