/**
 * @file q43_true_fused.cu
 * @brief SSB Q4.3 Implementation - OPTIMIZED True Fused Decompression
 *
 * Query:
 *   SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM date, supplier, part, lineorder
 *   WHERE lo_suppkey = s_suppkey AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND s_nation = 'UNITED STATES'
 *     AND (d_year = 1997 OR d_year = 1998)
 *     AND p_category = 'MFGR#14'
 *   GROUP BY d_year, s_city, p_brand1
 *
 * Note: Q4.3 has NO customer join (only 3 joins: Date, Supplier, Part)
 *
 * OPTIMIZATIONS APPLIED:
 * - Thread coarsening: 4 items per thread
 * - Shared memory metadata cache
 * - Two-level hash probing
 * - Selection flags (no warp divergence)
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
#include "fused_hash_table.cuh"

using namespace ssb;
using namespace l3_fused;

// Q4.3 Filter constants
constexpr uint32_t S_NATION_US = 24;  // UNITED STATES
constexpr uint32_t P_CATEGORY_MFGR14 = 14;
constexpr int AGG_NUM_YEARS = 2;  // 1997, 1998
constexpr int AGG_NUM_CITIES = 250;
constexpr int AGG_NUM_BRANDS = 1000;
constexpr int AGG_SIZE = AGG_NUM_YEARS * AGG_NUM_CITIES * AGG_NUM_BRANDS;

// ============================================================================
// Optimized Q4.3 Kernel with Thread Coarsening and Two-Level Hash
// ============================================================================

__global__ __launch_bounds__(FUSED_BLOCK_SIZE, 4)
void q43_optimized_fused_kernel(
    // Partkey column (filter first - most selective)
    const uint32_t* __restrict__ delta_array_partkey,
    const int32_t* __restrict__ model_types_partkey,
    const double* __restrict__ model_params_partkey,
    const int32_t* __restrict__ delta_bits_partkey,
    const int64_t* __restrict__ bit_offsets_partkey,
    // Orderdate column (year filter)
    const uint32_t* __restrict__ delta_array_orderdate,
    const int32_t* __restrict__ model_types_orderdate,
    const double* __restrict__ model_params_orderdate,
    const int32_t* __restrict__ delta_bits_orderdate,
    const int64_t* __restrict__ bit_offsets_orderdate,
    // Suppkey column
    const uint32_t* __restrict__ delta_array_suppkey,
    const int32_t* __restrict__ model_types_suppkey,
    const double* __restrict__ model_params_suppkey,
    const int32_t* __restrict__ delta_bits_suppkey,
    const int64_t* __restrict__ bit_offsets_suppkey,
    // Revenue column
    const uint32_t* __restrict__ delta_array_revenue,
    const int32_t* __restrict__ model_types_revenue,
    const double* __restrict__ model_params_revenue,
    const int32_t* __restrict__ delta_bits_revenue,
    const int64_t* __restrict__ bit_offsets_revenue,
    // Supplycost column
    const uint32_t* __restrict__ delta_array_supplycost,
    const int32_t* __restrict__ model_types_supplycost,
    const double* __restrict__ model_params_supplycost,
    const int32_t* __restrict__ delta_bits_supplycost,
    const int64_t* __restrict__ bit_offsets_supplycost,
    // Partition boundaries
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    // Hash tables (3 joins - no customer)
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    // Output
    long long* __restrict__ agg_profit)
{
    // Shared memory for metadata cache
    __shared__ FusedKernelSharedMem smem;

    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;

    // ========================================
    // PHASE 1: Load metadata to shared memory
    // ========================================
    if (threadIdx.x == 0) {
        smem.partition_size = end_indices[partition_id] - start_indices[partition_id];
        smem.start_idx = start_indices[partition_id];

        // Column 0: partkey
        loadColumnMetadata(smem.columns[0], partition_id,
            model_types_partkey, model_params_partkey,
            delta_bits_partkey, bit_offsets_partkey);

        // Column 1: orderdate
        loadColumnMetadata(smem.columns[1], partition_id,
            model_types_orderdate, model_params_orderdate,
            delta_bits_orderdate, bit_offsets_orderdate);

        // Column 2: suppkey
        loadColumnMetadata(smem.columns[2], partition_id,
            model_types_suppkey, model_params_suppkey,
            delta_bits_suppkey, bit_offsets_suppkey);

        // Column 3: revenue
        loadColumnMetadata(smem.columns[3], partition_id,
            model_types_revenue, model_params_revenue,
            delta_bits_revenue, bit_offsets_revenue);

        // Column 4: supplycost
        loadColumnMetadata(smem.columns[4], partition_id,
            model_types_supplycost, model_params_supplycost,
            delta_bits_supplycost, bit_offsets_supplycost);
    }
    __syncthreads();

    int partition_size = smem.partition_size;

    // ========================================
    // PHASE 2: Tile-based processing
    // ========================================

    for (int tile_start = 0; tile_start < partition_size; tile_start += FUSED_TILE_SIZE) {
        int tile_size = min(FUSED_TILE_SIZE, partition_size - tile_start);

        // Thread-local arrays
        int selection_flags[FUSED_ITEMS_PER_THREAD];
        uint32_t partkey[FUSED_ITEMS_PER_THREAD];
        uint32_t orderdate[FUSED_ITEMS_PER_THREAD];
        uint32_t suppkey[FUSED_ITEMS_PER_THREAD];
        uint32_t revenue[FUSED_ITEMS_PER_THREAD];
        uint32_t supplycost[FUSED_ITEMS_PER_THREAD];
        uint32_t p_brand[FUSED_ITEMS_PER_THREAD];
        uint32_t s_city[FUSED_ITEMS_PER_THREAD];
        uint32_t year[FUSED_ITEMS_PER_THREAD];

        // Initialize selection flags
        l3_fused::InitFlags<FUSED_ITEMS_PER_THREAD>(selection_flags, tile_size);

        // ---- Column 1: Partkey (filter category = MFGR#14, get brand) ----
        decompressTileCoarsened<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_partkey, smem.columns[0], tile_start,
            partkey, selection_flags, partition_size);

        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_get_fast(ht_p_keys, ht_p_values, ht_p_size, partkey[i], p_brand[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after part probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 2: Orderdate (filter year 1997 or 1998) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_orderdate, smem.columns[1], tile_start,
            orderdate, selection_flags, partition_size);

        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_get_fast(ht_d_keys, ht_d_values, ht_d_size, orderdate[i], year[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after date probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 3: Suppkey (filter s_nation = US, get city) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_suppkey, smem.columns[2], tile_start,
            suppkey, selection_flags, partition_size);

        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_get_fast(ht_s_keys, ht_s_values, ht_s_size, suppkey[i], s_city[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after supplier probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 4: Revenue (conditional) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_revenue, smem.columns[3], tile_start,
            revenue, selection_flags, partition_size);

        // ---- Column 5: Supplycost (conditional) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_supplycost, smem.columns[4], tile_start,
            supplycost, selection_flags, partition_size);

        // ---- Aggregation by (year, s_city, p_brand) ----
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                int year_idx = (year[i] == 1997) ? 0 : 1;
                if (s_city[i] < AGG_NUM_CITIES && p_brand[i] < AGG_NUM_BRANDS) {
                    int agg_idx = year_idx * AGG_NUM_CITIES * AGG_NUM_BRANDS +
                                  s_city[i] * AGG_NUM_BRANDS + p_brand[i];
                    long long profit = static_cast<long long>(revenue[i]) - static_cast<long long>(supplycost[i]);
                    atomicAdd_int64(&agg_profit[agg_idx], profit);
                }
            }
        }
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ43TrueFused(SSBDataCompressed& data, QueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_partitions = data.lo_orderdate->num_partitions;

    // -------------------------------------------------------------------------
    // Step 1: Build hash tables
    // -------------------------------------------------------------------------
    cudaEvent_t ht_start, ht_stop;
    cudaEventCreate(&ht_start);
    cudaEventCreate(&ht_stop);
    cudaEventRecord(ht_start);

    int block_size = 256;

    // Part hash table (category = MFGR#14, partkey -> brand)
    HashTable ht_part;
    ht_part.allocate(P_LEN, true);
    build_part_ht_category_brand_kernel<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        P_CATEGORY_MFGR14,
        ht_part.d_keys, ht_part.d_values, ht_part.size);

    // Date hash table (1997 or 1998 only, datekey -> year)
    HashTable ht_date;
    ht_date.allocate(D_LEN, true);
    build_date_ht_year_range_kernel<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN,
        1997, 1998,
        ht_date.d_keys, ht_date.d_values, ht_date.size);

    // Supplier hash table (s_nation = US, suppkey -> city)
    HashTable ht_supplier;
    ht_supplier.allocate(S_LEN, true);
    build_supplier_ht_nation_city_kernel<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN,
        S_NATION_US,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size);

    cudaEventRecord(ht_stop);
    cudaEventSynchronize(ht_stop);
    float ht_ms;
    cudaEventElapsedTime(&ht_ms, ht_start, ht_stop);
    timing.hash_build_ms = ht_ms;

    // -------------------------------------------------------------------------
    // Step 2: Allocate aggregation arrays
    // -------------------------------------------------------------------------
    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    // -------------------------------------------------------------------------
    // Step 3: Warmup run
    // -------------------------------------------------------------------------
    q43_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        data.lo_partkey->delta_array, data.lo_partkey->d_model_types,
        data.lo_partkey->d_model_params, data.lo_partkey->d_delta_bits,
        data.lo_partkey->d_delta_array_bit_offsets,
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        data.lo_supplycost->delta_array, data.lo_supplycost->d_model_types,
        data.lo_supplycost->d_model_params, data.lo_supplycost->d_delta_bits,
        data.lo_supplycost->d_delta_array_bit_offsets,
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        ht_part.d_keys, ht_part.d_values, ht_part.size,
        ht_date.d_keys, ht_date.d_values, ht_date.size,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
        d_agg_profit
    );
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Step 4: Timed run
    // -------------------------------------------------------------------------
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));
    cudaEventRecord(start);

    q43_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        data.lo_partkey->delta_array, data.lo_partkey->d_model_types,
        data.lo_partkey->d_model_params, data.lo_partkey->d_delta_bits,
        data.lo_partkey->d_delta_array_bit_offsets,
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        data.lo_supplycost->delta_array, data.lo_supplycost->d_model_types,
        data.lo_supplycost->d_model_params, data.lo_supplycost->d_delta_bits,
        data.lo_supplycost->d_delta_array_bit_offsets,
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        ht_part.d_keys, ht_part.d_values, ht_part.size,
        ht_date.d_keys, ht_date.d_values, ht_date.size,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
        d_agg_profit
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    timing.kernel_ms = ms;
    timing.data_load_ms = 0;
    timing.total_ms = timing.hash_build_ms + timing.kernel_ms;

    // -------------------------------------------------------------------------
    // Step 5: Collect results
    // -------------------------------------------------------------------------
    std::vector<long long> h_agg_profit(AGG_SIZE);
    cudaMemcpy(h_agg_profit.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q4.3 Results (Optimized True Fused) ===" << std::endl;
    int num_results = 0;
    long long total_profit = 0;
    for (int y = 0; y < AGG_NUM_YEARS; ++y) {
        for (int c = 0; c < AGG_NUM_CITIES; ++c) {
            for (int b = 0; b < AGG_NUM_BRANDS; ++b) {
                int idx = y * AGG_NUM_CITIES * AGG_NUM_BRANDS + c * AGG_NUM_BRANDS + b;
                if (h_agg_profit[idx] != 0) {
                    num_results++;
                    total_profit += h_agg_profit[idx];
                }
            }
        }
    }
    std::cout << "Total groups: " << num_results << ", Total profit: " << total_profit << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    timing.print("Q4.3 Optimized True Fused");

    // Cleanup
    cudaFree(d_agg_profit);
    ht_date.free();
    ht_supplier.free();
    ht_part.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(ht_start);
    cudaEventDestroy(ht_stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q4.3 - Optimized True Fused Decompression ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Thread coarsening: " << FUSED_ITEMS_PER_THREAD << " items/thread" << std::endl;
    std::cout << "Block size: " << FUSED_BLOCK_SIZE << " threads" << std::endl;
    std::cout << "Tile size: " << FUSED_TILE_SIZE << " elements" << std::endl;

    L3Config config = L3Config::fixedPartitioning(2048);
    SSBDataCompressed data;
    data.loadAndCompress(data_dir, config);

    std::cout << "\n--- Benchmark Runs ---" << std::endl;
    for (int i = 0; i < 5; ++i) {
        QueryTiming timing;
        runQ43TrueFused(data, timing);
        std::cout << "Run " << (i+1) << ": " << timing.total_ms << " ms"
                  << " (HashBuild=" << timing.hash_build_ms << " ms, Kernel=" << timing.kernel_ms << " ms)" << std::endl;
    }

    data.free();
    return 0;
}
