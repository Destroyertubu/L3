/**
 * @file q11.cu
 * @brief SSB Q1.1 - Crystal-opt Style L3 Fused Decompression
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey
 *     AND d_year = 1993
 *     AND lo_discount BETWEEN 1 AND 3
 *     AND lo_quantity < 25
 *
 * Key optimizations:
 * - Crystal-opt block operations (BLOCK_THREADS=128, ITEMS_PER_THREAD=4)
 * - L3 fused decompression (BlockLoadL3/BlockPredLoadL3)
 * - Warp shuffle reduction
 * - Early termination with IsTerm
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

// Common infrastructure
#include "common/crystal_ssb.cuh"
#include "common/l3_block_load.cuh"
#include "common/ssb_data_new.hpp"

using namespace ssb_new;

// ============================================================================
// Q1.1 Kernel (Crystal-opt Style with L3 Decompression)
// ============================================================================

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__global__ void q11_kernel(
    // L3 compressed orderdate column
    const uint32_t* __restrict__ delta_orderdate,
    const int32_t* __restrict__ model_types_orderdate,
    const double* __restrict__ model_params_orderdate,
    const int32_t* __restrict__ delta_bits_orderdate,
    const int64_t* __restrict__ bit_offsets_orderdate,
    // L3 compressed quantity column
    const uint32_t* __restrict__ delta_quantity,
    const int32_t* __restrict__ model_types_quantity,
    const double* __restrict__ model_params_quantity,
    const int32_t* __restrict__ delta_bits_quantity,
    const int64_t* __restrict__ bit_offsets_quantity,
    // L3 compressed discount column
    const uint32_t* __restrict__ delta_discount,
    const int32_t* __restrict__ model_types_discount,
    const double* __restrict__ model_params_discount,
    const int32_t* __restrict__ delta_bits_discount,
    const int64_t* __restrict__ bit_offsets_discount,
    // L3 compressed extendedprice column
    const uint32_t* __restrict__ delta_extprice,
    const int32_t* __restrict__ model_types_extprice,
    const double* __restrict__ model_params_extprice,
    const int32_t* __restrict__ delta_bits_extprice,
    const int64_t* __restrict__ bit_offsets_extprice,
    // Partition boundaries
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    // Output
    unsigned long long* __restrict__ revenue)
{
    constexpr int TILE_SIZE = _BLOCK_THREADS * _ITEMS_PER_THREAD;

    // Shared memory for metadata cache and reduction buffer
    __shared__ L3PartitionMeta meta_orderdate;
    __shared__ L3PartitionMeta meta_quantity;
    __shared__ L3PartitionMeta meta_discount;
    __shared__ L3PartitionMeta meta_extprice;
    __shared__ long long buffer[32];

    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;

    // Load partition metadata (thread 0 only)
    if (threadIdx.x == 0) {
        loadL3Meta(meta_orderdate, partition_id,
            model_types_orderdate, model_params_orderdate,
            delta_bits_orderdate, bit_offsets_orderdate);
        loadL3Meta(meta_quantity, partition_id,
            model_types_quantity, model_params_quantity,
            delta_bits_quantity, bit_offsets_quantity);
        loadL3Meta(meta_discount, partition_id,
            model_types_discount, model_params_discount,
            delta_bits_discount, bit_offsets_discount);
        loadL3Meta(meta_extprice, partition_id,
            model_types_extprice, model_params_extprice,
            delta_bits_extprice, bit_offsets_extprice);
    }
    __syncthreads();

    int partition_start = start_indices[partition_id];
    int partition_size = end_indices[partition_id] - partition_start;
    long long local_sum = 0;

    // Process tiles within partition (512 elements per tile)
    for (int tile_start = 0; tile_start < partition_size; tile_start += TILE_SIZE) {
        int num_tile_items = min(TILE_SIZE, partition_size - tile_start);

        // Thread-local arrays
        int items[_ITEMS_PER_THREAD];
        int selection_flags[_ITEMS_PER_THREAD];
        int discount[_ITEMS_PER_THREAD];
        int extprice[_ITEMS_PER_THREAD];

        // Initialize all flags to 1
        InitFlags<_BLOCK_THREADS, _ITEMS_PER_THREAD>(selection_flags);

        // ---- Column 1: lo_orderdate ----
        // Decompress and filter: d_year = 1993 (19930101 <= orderdate < 19940101)
        BlockLoadL3<_BLOCK_THREADS, _ITEMS_PER_THREAD>(
            delta_orderdate, meta_orderdate, tile_start, items, num_tile_items);

        BlockPredAndGT<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            items, 19930000, selection_flags, num_tile_items);
        BlockPredAndLT<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            items, 19940000, selection_flags, num_tile_items);

        // Early termination check
        if (IsTerm<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 2: lo_quantity ----
        // Decompress (only selected) and filter: quantity < 25
        BlockPredLoadL3<_BLOCK_THREADS, _ITEMS_PER_THREAD>(
            delta_quantity, meta_quantity, tile_start, items, num_tile_items, selection_flags);

        BlockPredAndLT<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            items, 25, selection_flags, num_tile_items);

        if (IsTerm<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 3: lo_discount ----
        // Decompress (only selected) and filter: 1 <= discount <= 3
        BlockPredLoadL3<_BLOCK_THREADS, _ITEMS_PER_THREAD>(
            delta_discount, meta_discount, tile_start, discount, num_tile_items, selection_flags);

        BlockPredAndGTE<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            discount, 1, selection_flags, num_tile_items);
        BlockPredAndLTE<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            discount, 3, selection_flags, num_tile_items);

        if (IsTerm<int, _BLOCK_THREADS, _ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 4: lo_extendedprice ----
        // Decompress (only selected)
        BlockPredLoadL3<_BLOCK_THREADS, _ITEMS_PER_THREAD>(
            delta_extprice, meta_extprice, tile_start, extprice, num_tile_items, selection_flags);

        // ---- Aggregation ----
        #pragma unroll
        for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ++ITEM) {
            if (threadIdx.x + (_BLOCK_THREADS * ITEM) < num_tile_items) {
                if (selection_flags[ITEM]) {
                    local_sum += static_cast<long long>(extprice[ITEM]) * discount[ITEM];
                }
            }
        }
    }

    // Warp shuffle reduction
    __syncthreads();
    unsigned long long aggregate = BlockSum<long long, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
        local_sum, buffer);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(revenue, aggregate);
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ11(SSBData& data, QueryTiming& timing) {
    CudaTimer timer;

    int num_partitions = data.numPartitions();

    // Allocate output
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    // Warmup run
    q11_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_partitions, BLOCK_THREADS>>>(
        data.lo_orderdate.d_delta_array, data.lo_orderdate.d_model_types,
        data.lo_orderdate.d_model_params, data.lo_orderdate.d_delta_bits,
        data.lo_orderdate.d_bit_offsets,
        data.lo_quantity.d_delta_array, data.lo_quantity.d_model_types,
        data.lo_quantity.d_model_params, data.lo_quantity.d_delta_bits,
        data.lo_quantity.d_bit_offsets,
        data.lo_discount.d_delta_array, data.lo_discount.d_model_types,
        data.lo_discount.d_model_params, data.lo_discount.d_delta_bits,
        data.lo_discount.d_bit_offsets,
        data.lo_extendedprice.d_delta_array, data.lo_extendedprice.d_model_types,
        data.lo_extendedprice.d_model_params, data.lo_extendedprice.d_delta_bits,
        data.lo_extendedprice.d_bit_offsets,
        data.lo_orderdate.d_start_indices, data.lo_orderdate.d_end_indices,
        num_partitions,
        d_revenue
    );
    cudaDeviceSynchronize();

    // Timed run
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    timer.start();

    q11_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_partitions, BLOCK_THREADS>>>(
        data.lo_orderdate.d_delta_array, data.lo_orderdate.d_model_types,
        data.lo_orderdate.d_model_params, data.lo_orderdate.d_delta_bits,
        data.lo_orderdate.d_bit_offsets,
        data.lo_quantity.d_delta_array, data.lo_quantity.d_model_types,
        data.lo_quantity.d_model_params, data.lo_quantity.d_delta_bits,
        data.lo_quantity.d_bit_offsets,
        data.lo_discount.d_delta_array, data.lo_discount.d_model_types,
        data.lo_discount.d_model_params, data.lo_discount.d_delta_bits,
        data.lo_discount.d_bit_offsets,
        data.lo_extendedprice.d_delta_array, data.lo_extendedprice.d_model_types,
        data.lo_extendedprice.d_model_params, data.lo_extendedprice.d_delta_bits,
        data.lo_extendedprice.d_bit_offsets,
        data.lo_orderdate.d_start_indices, data.lo_orderdate.d_end_indices,
        num_partitions,
        d_revenue
    );

    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.hash_build_ms = 0;  // Q1.1 doesn't use hash tables
    timing.total_ms = timing.kernel_ms;

    // Get result
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.1 Results ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    timing.print("Q1.1 Crystal-opt L3 Fused");

    cudaFree(d_revenue);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    int partition_size = 2048;

    if (argc > 1) data_dir = argv[1];
    if (argc > 2) partition_size = std::atoi(argv[2]);

    std::cout << "=== SSB Q1.1 - Crystal-opt Style L3 Fused ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Partition size: " << partition_size << std::endl;
    std::cout << "Block threads: " << BLOCK_THREADS << std::endl;
    std::cout << "Items per thread: " << ITEMS_PER_THREAD << std::endl;
    std::cout << "Tile size: " << TILE_SIZE << std::endl;

    // Load and compress data
    SSBData data;
    data.load(data_dir, partition_size);

    // Benchmark runs
    std::cout << "\n--- Benchmark Runs ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        QueryTiming timing;
        runQ11(data, timing);
        std::cout << "Run " << (i + 1) << ": " << timing.kernel_ms << " ms" << std::endl;
    }

    data.free();
    return 0;
}
