/**
 * @file q22_true_fused.cu
 * @brief SSB Q2.2 Implementation - OPTIMIZED True Fused Decompression
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, date, part, supplier
 *   WHERE lo_orderdate = d_datekey
 *     AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'
 *     AND s_region = 'ASIA'
 *   GROUP BY d_year, p_brand1
 *   ORDER BY d_year, p_brand1;
 *
 * OPTIMIZATIONS APPLIED:
 * - Thread coarsening: 4 items per thread
 * - Shared memory metadata cache
 * - Two-level hash probing (fast XOR + MurmurHash3)
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

// Filter constants
constexpr uint32_t P_BRAND_MIN = 221;   // MFGR#2221
constexpr uint32_t P_BRAND_MAX = 228;   // MFGR#2228
constexpr uint32_t S_REGION_ASIA = 2;   // ASIA region code

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;   // 1992-1998
constexpr int AGG_NUM_BRANDS = 1000;

// ============================================================================
// Optimized Q2.2 Kernel with Thread Coarsening and Two-Level Hash
// ============================================================================

__global__ __launch_bounds__(FUSED_BLOCK_SIZE, 4)
void q22_optimized_fused_kernel(
    // Suppkey column compressed metadata
    const uint32_t* __restrict__ delta_array_suppkey,
    const int32_t* __restrict__ model_types_suppkey,
    const double* __restrict__ model_params_suppkey,
    const int32_t* __restrict__ delta_bits_suppkey,
    const int64_t* __restrict__ bit_offsets_suppkey,
    // Partkey column compressed metadata
    const uint32_t* __restrict__ delta_array_partkey,
    const int32_t* __restrict__ model_types_partkey,
    const double* __restrict__ model_params_partkey,
    const int32_t* __restrict__ delta_bits_partkey,
    const int64_t* __restrict__ bit_offsets_partkey,
    // Orderdate column compressed metadata
    const uint32_t* __restrict__ delta_array_orderdate,
    const int32_t* __restrict__ model_types_orderdate,
    const double* __restrict__ model_params_orderdate,
    const int32_t* __restrict__ delta_bits_orderdate,
    const int64_t* __restrict__ bit_offsets_orderdate,
    // Revenue column compressed metadata
    const uint32_t* __restrict__ delta_array_revenue,
    const int32_t* __restrict__ model_types_revenue,
    const double* __restrict__ model_params_revenue,
    const int32_t* __restrict__ delta_bits_revenue,
    const int64_t* __restrict__ bit_offsets_revenue,
    // Partition boundaries (shared across columns)
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    // Hash tables (pre-built from dimension tables)
    const uint32_t* __restrict__ ht_s_keys,
    int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    // Output: aggregation results
    unsigned long long* __restrict__ agg_revenue)
{
    // Shared memory for metadata cache
    __shared__ FusedKernelSharedMem smem;

    int partition_id = blockIdx.x;
    if (partition_id >= num_partitions) return;

    // ========================================
    // PHASE 1: Load metadata to shared memory (thread 0 only)
    // ========================================
    if (threadIdx.x == 0) {
        smem.partition_size = end_indices[partition_id] - start_indices[partition_id];
        smem.start_idx = start_indices[partition_id];

        // Column 0: suppkey
        loadColumnMetadata(smem.columns[0], partition_id,
            model_types_suppkey, model_params_suppkey,
            delta_bits_suppkey, bit_offsets_suppkey);

        // Column 1: partkey
        loadColumnMetadata(smem.columns[1], partition_id,
            model_types_partkey, model_params_partkey,
            delta_bits_partkey, bit_offsets_partkey);

        // Column 2: orderdate
        loadColumnMetadata(smem.columns[2], partition_id,
            model_types_orderdate, model_params_orderdate,
            delta_bits_orderdate, bit_offsets_orderdate);

        // Column 3: revenue
        loadColumnMetadata(smem.columns[3], partition_id,
            model_types_revenue, model_params_revenue,
            delta_bits_revenue, bit_offsets_revenue);
    }
    __syncthreads();

    int partition_size = smem.partition_size;

    // ========================================
    // PHASE 2: Tile-based processing with thread coarsening
    // ========================================

    // Process tiles of FUSED_TILE_SIZE (512) elements
    for (int tile_start = 0; tile_start < partition_size; tile_start += FUSED_TILE_SIZE) {
        int tile_size = min(FUSED_TILE_SIZE, partition_size - tile_start);

        // Thread-local arrays (4 items per thread)
        int selection_flags[FUSED_ITEMS_PER_THREAD];
        uint32_t suppkey[FUSED_ITEMS_PER_THREAD];
        uint32_t partkey[FUSED_ITEMS_PER_THREAD];
        uint32_t orderdate[FUSED_ITEMS_PER_THREAD];
        uint32_t revenue[FUSED_ITEMS_PER_THREAD];
        uint32_t brand[FUSED_ITEMS_PER_THREAD];
        uint32_t year[FUSED_ITEMS_PER_THREAD];

        // Initialize selection flags
        l3_fused::InitFlags<FUSED_ITEMS_PER_THREAD>(selection_flags, tile_size);

        // ---- Column 1: Suppkey (check supplier filter first - early exit) ----
        decompressTileCoarsened<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_suppkey, smem.columns[0], tile_start,
            suppkey, selection_flags, partition_size);

        // Probe supplier hash table (two-level hash)
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_exists_fast(ht_s_keys, ht_s_size, suppkey[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after supplier probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 2: Partkey (conditional, get brand) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_partkey, smem.columns[1], tile_start,
            partkey, selection_flags, partition_size);

        // Probe part hash table (two-level hash, get brand)
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_get_fast(ht_p_keys, ht_p_values, ht_p_size, partkey[i], brand[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after part probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 3: Orderdate (conditional, get year) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_orderdate, smem.columns[2], tile_start,
            orderdate, selection_flags, partition_size);

        // Probe date hash table (two-level hash, get year)
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

        // ---- Column 4: Revenue (conditional) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_revenue, smem.columns[3], tile_start,
            revenue, selection_flags, partition_size);

        // ---- Aggregation by (year, brand) ----
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                int year_idx = year[i] - 1992;
                if (year_idx >= 0 && year_idx < AGG_NUM_YEARS && brand[i] < AGG_NUM_BRANDS) {
                    int agg_idx = year_idx * AGG_NUM_BRANDS + brand[i];
                    atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
                }
            }
        }
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ22TrueFused(SSBDataCompressed& data, QueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int num_partitions = data.lo_orderdate->num_partitions;

    // -------------------------------------------------------------------------
    // Step 1: Build hash tables for dimension tables
    // -------------------------------------------------------------------------
    cudaEvent_t ht_start, ht_stop;
    cudaEventCreate(&ht_start);
    cudaEventCreate(&ht_stop);
    cudaEventRecord(ht_start);

    // Date hash table (all dates, datekey -> year)
    HashTable ht_date;
    ht_date.allocate(D_LEN, true);
    int block_size = 256;
    int grid_d = (D_LEN + block_size - 1) / block_size;
    build_date_ht_kernel<<<grid_d, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN,
        ht_date.d_keys, ht_date.d_values, ht_date.size);

    // Part hash table (filtered by brand range, partkey -> brand1)
    HashTable ht_part;
    ht_part.allocate(P_LEN, true);
    int grid_p = (P_LEN + block_size - 1) / block_size;
    build_part_ht_brand_range_kernel<<<grid_p, block_size>>>(
        data.d_p_partkey, data.d_p_brand1, P_LEN,
        P_BRAND_MIN, P_BRAND_MAX,
        ht_part.d_keys, ht_part.d_values, ht_part.size);

    // Supplier hash table (filtered by region, keys only)
    HashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    int grid_s = (S_LEN + block_size - 1) / block_size;
    build_supplier_ht_region_kernel<<<grid_s, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN,
        S_REGION_ASIA,
        ht_supplier.d_keys, ht_supplier.size);

    cudaEventRecord(ht_stop);
    cudaEventSynchronize(ht_stop);
    float ht_ms;
    cudaEventElapsedTime(&ht_ms, ht_start, ht_stop);
    timing.hash_build_ms = ht_ms;

    // -------------------------------------------------------------------------
    // Step 2: Allocate aggregation arrays
    // -------------------------------------------------------------------------
    int agg_size = AGG_NUM_YEARS * AGG_NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

    // -------------------------------------------------------------------------
    // Step 3: Warmup run
    // -------------------------------------------------------------------------
    q22_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        // Suppkey
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        // Partkey
        data.lo_partkey->delta_array, data.lo_partkey->d_model_types,
        data.lo_partkey->d_model_params, data.lo_partkey->d_delta_bits,
        data.lo_partkey->d_delta_array_bit_offsets,
        // Orderdate
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        // Revenue
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        // Partition boundaries
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        // Hash tables
        ht_supplier.d_keys, ht_supplier.size,
        ht_part.d_keys, ht_part.d_values, ht_part.size,
        ht_date.d_keys, ht_date.d_values, ht_date.size,
        d_agg_revenue
    );
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Step 4: Timed run
    // -------------------------------------------------------------------------
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
    cudaEventRecord(start);

    q22_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        // Suppkey
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        // Partkey
        data.lo_partkey->delta_array, data.lo_partkey->d_model_types,
        data.lo_partkey->d_model_params, data.lo_partkey->d_delta_bits,
        data.lo_partkey->d_delta_array_bit_offsets,
        // Orderdate
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        // Revenue
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        // Partition boundaries
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        // Hash tables
        ht_supplier.d_keys, ht_supplier.size,
        ht_part.d_keys, ht_part.d_values, ht_part.size,
        ht_date.d_keys, ht_date.d_values, ht_date.size,
        d_agg_revenue
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
    std::vector<unsigned long long> h_agg_revenue(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.2 Results (Optimized True Fused) ===" << std::endl;
    std::cout << "Year\tBrand\tRevenue" << std::endl;

    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int y = 0; y < AGG_NUM_YEARS; ++y) {
        for (int b = 0; b < AGG_NUM_BRANDS; ++b) {
            int idx = y * AGG_NUM_BRANDS + b;
            if (h_agg_revenue[idx] > 0) {
                num_results++;
                total_revenue += h_agg_revenue[idx];
                if (num_results <= 10) {
                    std::cout << (1992 + y) << "\t" << b << "\t" << h_agg_revenue[idx] << std::endl;
                }
            }
        }
    }
    if (num_results > 10) {
        std::cout << "... and " << (num_results - 10) << " more results" << std::endl;
    }
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    timing.print("Q2.2 Optimized True Fused");

    // Cleanup
    cudaFree(d_agg_revenue);
    ht_date.free();
    ht_part.free();
    ht_supplier.free();
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

    std::cout << "=== SSB Q2.2 - Optimized True Fused Decompression ===" << std::endl;
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
        runQ22TrueFused(data, timing);
        std::cout << "Run " << (i+1) << ": " << timing.total_ms << " ms"
                  << " (HashBuild=" << timing.hash_build_ms << " ms, Kernel=" << timing.kernel_ms << " ms)" << std::endl;
    }

    data.free();
    return 0;
}
