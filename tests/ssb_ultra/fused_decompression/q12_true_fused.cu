/**
 * @file q12_true_fused.cu
 * @brief SSB Q1.2 Implementation - OPTIMIZED True Fused Decompression
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey AND d_yearmonthnum = 199401
 *     AND lo_discount BETWEEN 4 AND 6 AND lo_quantity BETWEEN 26 AND 35;
 *
 * OPTIMIZATIONS APPLIED:
 * - Thread coarsening: 4 items per thread
 * - Shared memory metadata cache
 * - Warp shuffle reduction
 * - Selection flags (no warp divergence from continue)
 * - Register buffering for delta extraction
 * - Tile-based processing
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

// SSB common headers
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"

// L3 headers
#include "L3_format.hpp"
#include "L3_codec.hpp"

// Optimized fused decompression infrastructure
#include "fused_kernel_common.cuh"
#include "fused_decompress_tile.cuh"
#include "l3_decompress_device.cuh"

using namespace ssb;
using namespace l3_fused;

// ============================================================================
// Optimized Q1.2 Kernel with Thread Coarsening
// ============================================================================

__global__ __launch_bounds__(FUSED_BLOCK_SIZE, 4)
void q12_optimized_fused_kernel(
    // Orderdate column compressed metadata
    const uint32_t* __restrict__ delta_array_orderdate,
    const int32_t* __restrict__ model_types_orderdate,
    const double* __restrict__ model_params_orderdate,
    const int32_t* __restrict__ delta_bits_orderdate,
    const int64_t* __restrict__ bit_offsets_orderdate,
    // Discount column compressed metadata
    const uint32_t* __restrict__ delta_array_discount,
    const int32_t* __restrict__ model_types_discount,
    const double* __restrict__ model_params_discount,
    const int32_t* __restrict__ delta_bits_discount,
    const int64_t* __restrict__ bit_offsets_discount,
    // Quantity column compressed metadata
    const uint32_t* __restrict__ delta_array_quantity,
    const int32_t* __restrict__ model_types_quantity,
    const double* __restrict__ model_params_quantity,
    const int32_t* __restrict__ delta_bits_quantity,
    const int64_t* __restrict__ bit_offsets_quantity,
    // Extendedprice column compressed metadata
    const uint32_t* __restrict__ delta_array_extprice,
    const int32_t* __restrict__ model_types_extprice,
    const double* __restrict__ model_params_extprice,
    const int32_t* __restrict__ delta_bits_extprice,
    const int64_t* __restrict__ bit_offsets_extprice,
    // Partition boundaries (shared across columns)
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    // Output
    unsigned long long* __restrict__ revenue)
{
    // Shared memory for metadata cache and warp reduction
    __shared__ FusedKernelSharedMem smem;

    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;

    // ========================================
    // PHASE 1: Load metadata to shared memory (thread 0 only)
    // ========================================
    if (threadIdx.x == 0) {
        smem.partition_size = end_indices[partition_id] - start_indices[partition_id];
        smem.start_idx = start_indices[partition_id];

        // Column 0: orderdate
        loadColumnMetadata(smem.columns[0], partition_id,
            model_types_orderdate, model_params_orderdate,
            delta_bits_orderdate, bit_offsets_orderdate);

        // Column 1: discount
        loadColumnMetadata(smem.columns[1], partition_id,
            model_types_discount, model_params_discount,
            delta_bits_discount, bit_offsets_discount);

        // Column 2: quantity
        loadColumnMetadata(smem.columns[2], partition_id,
            model_types_quantity, model_params_quantity,
            delta_bits_quantity, bit_offsets_quantity);

        // Column 3: extendedprice
        loadColumnMetadata(smem.columns[3], partition_id,
            model_types_extprice, model_params_extprice,
            delta_bits_extprice, bit_offsets_extprice);
    }
    __syncthreads();

    int partition_size = smem.partition_size;

    // ========================================
    // PHASE 2: Tile-based processing with thread coarsening
    // ========================================
    unsigned long long local_sum = 0;

    // Process tiles of FUSED_TILE_SIZE (512) elements
    for (int tile_start = 0; tile_start < partition_size; tile_start += FUSED_TILE_SIZE) {
        int tile_size = min(FUSED_TILE_SIZE, partition_size - tile_start);

        // Thread-local arrays (4 items per thread)
        int selection_flags[FUSED_ITEMS_PER_THREAD];
        uint32_t orderdate[FUSED_ITEMS_PER_THREAD];
        uint32_t discount[FUSED_ITEMS_PER_THREAD];
        uint32_t quantity[FUSED_ITEMS_PER_THREAD];
        uint32_t extprice[FUSED_ITEMS_PER_THREAD];

        // Initialize selection flags (explicitly use l3_fused namespace)
        l3_fused::InitFlags<FUSED_ITEMS_PER_THREAD>(selection_flags, tile_size);

        // ---- Column 1: Orderdate ----
        decompressTileCoarsened<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_orderdate, smem.columns[0], tile_start,
            orderdate, selection_flags, partition_size);

        // Q1.2 Predicate: January 1994 (19940101 <= orderdate <= 19940131)
        BlockPredAndBetween<uint32_t, FUSED_ITEMS_PER_THREAD>(
            orderdate, 19940101u, 19940131u, selection_flags);

        // Early termination
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 2: Discount (conditional on orderdate passing) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_discount, smem.columns[1], tile_start,
            discount, selection_flags, partition_size);

        // Q1.2 Predicate: discount BETWEEN 4 AND 6
        BlockPredAndBetween<uint32_t, FUSED_ITEMS_PER_THREAD>(
            discount, 4u, 6u, selection_flags);

        // Early termination
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 3: Quantity (conditional on discount passing) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_quantity, smem.columns[2], tile_start,
            quantity, selection_flags, partition_size);

        // Q1.2 Predicate: quantity BETWEEN 26 AND 35
        BlockPredAndBetween<uint32_t, FUSED_ITEMS_PER_THREAD>(
            quantity, 26u, 35u, selection_flags);

        // Early termination
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 4: Extendedprice (conditional on quantity passing) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_extprice, smem.columns[3], tile_start,
            extprice, selection_flags, partition_size);

        // ---- Aggregation ----
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                local_sum += static_cast<unsigned long long>(extprice[i]) *
                             static_cast<unsigned long long>(discount[i]);
            }
        }
    }

    // ========================================
    // PHASE 3: Warp shuffle reduction
    // ========================================
    local_sum = blockReduceSum(local_sum, smem.warp_sums);

    // Thread 0 does final atomic add
    if (threadIdx.x == 0 && local_sum > 0) {
        atomicAdd(revenue, local_sum);
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ12TrueFused(SSBDataCompressed& data, QueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Allocate result
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    // Get number of partitions
    int num_partitions = data.lo_orderdate->num_partitions;

    // Warmup run
    q12_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        // Orderdate
        data.lo_orderdate->delta_array,
        data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params,
        data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        // Discount
        data.lo_discount->delta_array,
        data.lo_discount->d_model_types,
        data.lo_discount->d_model_params,
        data.lo_discount->d_delta_bits,
        data.lo_discount->d_delta_array_bit_offsets,
        // Quantity
        data.lo_quantity->delta_array,
        data.lo_quantity->d_model_types,
        data.lo_quantity->d_model_params,
        data.lo_quantity->d_delta_bits,
        data.lo_quantity->d_delta_array_bit_offsets,
        // Extendedprice
        data.lo_extendedprice->delta_array,
        data.lo_extendedprice->d_model_types,
        data.lo_extendedprice->d_model_params,
        data.lo_extendedprice->d_delta_bits,
        data.lo_extendedprice->d_delta_array_bit_offsets,
        // Partition boundaries
        data.lo_orderdate->d_start_indices,
        data.lo_orderdate->d_end_indices,
        num_partitions,
        d_revenue
    );
    cudaDeviceSynchronize();

    // Timed run
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    cudaEventRecord(start);

    q12_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        // Orderdate
        data.lo_orderdate->delta_array,
        data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params,
        data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        // Discount
        data.lo_discount->delta_array,
        data.lo_discount->d_model_types,
        data.lo_discount->d_model_params,
        data.lo_discount->d_delta_bits,
        data.lo_discount->d_delta_array_bit_offsets,
        // Quantity
        data.lo_quantity->delta_array,
        data.lo_quantity->d_model_types,
        data.lo_quantity->d_model_params,
        data.lo_quantity->d_delta_bits,
        data.lo_quantity->d_delta_array_bit_offsets,
        // Extendedprice
        data.lo_extendedprice->delta_array,
        data.lo_extendedprice->d_model_types,
        data.lo_extendedprice->d_model_params,
        data.lo_extendedprice->d_delta_bits,
        data.lo_extendedprice->d_delta_array_bit_offsets,
        // Partition boundaries
        data.lo_orderdate->d_start_indices,
        data.lo_orderdate->d_end_indices,
        num_partitions,
        d_revenue
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    timing.kernel_ms = ms;
    timing.hash_build_ms = 0;
    timing.data_load_ms = 0;
    timing.total_ms = ms;

    // Get result
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.2 Results (Optimized True Fused) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    timing.print("Q1.2 Optimized True Fused");

    // Cleanup
    cudaFree(d_revenue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.2 - Optimized True Fused Decompression ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Thread coarsening: " << FUSED_ITEMS_PER_THREAD << " items/thread" << std::endl;
    std::cout << "Block size: " << FUSED_BLOCK_SIZE << " threads" << std::endl;
    std::cout << "Tile size: " << FUSED_TILE_SIZE << " elements" << std::endl;

    // Load and compress data using FIXED config
    L3Config config = L3Config::fixedPartitioning(2048);
    SSBDataCompressed data;
    data.loadAndCompress(data_dir, config);

    // Run query multiple times
    std::cout << "\n--- Benchmark Runs ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        QueryTiming timing;
        runQ12TrueFused(data, timing);
        std::cout << "Run " << (i+1) << ": " << timing.total_ms << " ms" << std::endl;
    }

    data.free();
    return 0;
}
