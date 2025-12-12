/**
 * @file q34_true_fused.cu
 * @brief SSB Q3.4 Implementation - OPTIMIZED True Fused Decompression
 *
 * Query:
 *   SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey AND lo_orderdate = d_datekey
 *     AND (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
 *     AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
 *     AND d_yearmonth = 'Dec1997'
 *   GROUP BY c_city, s_city, d_year
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

// Q3.4 Filter constants
constexpr uint32_t CITY_UK1 = 231;
constexpr uint32_t CITY_UK5 = 235;
constexpr uint32_t DATE_DEC1997_START = 19971201;
constexpr uint32_t DATE_DEC1997_END = 19971231;
constexpr int AGG_NUM_CITIES = 250;
constexpr int AGG_SIZE = AGG_NUM_CITIES * AGG_NUM_CITIES;  // Single month = single year

// ============================================================================
// Optimized Q3.4 Kernel with Thread Coarsening and Two-Level Hash
// ============================================================================

__global__ __launch_bounds__(FUSED_BLOCK_SIZE, 4)
void q34_optimized_fused_kernel(
    // Orderdate column compressed metadata (filter first for Dec1997)
    const uint32_t* __restrict__ delta_array_orderdate,
    const int32_t* __restrict__ model_types_orderdate,
    const double* __restrict__ model_params_orderdate,
    const int32_t* __restrict__ delta_bits_orderdate,
    const int64_t* __restrict__ bit_offsets_orderdate,
    // Custkey column compressed metadata
    const uint32_t* __restrict__ delta_array_custkey,
    const int32_t* __restrict__ model_types_custkey,
    const double* __restrict__ model_params_custkey,
    const int32_t* __restrict__ delta_bits_custkey,
    const int64_t* __restrict__ bit_offsets_custkey,
    // Suppkey column compressed metadata
    const uint32_t* __restrict__ delta_array_suppkey,
    const int32_t* __restrict__ model_types_suppkey,
    const double* __restrict__ model_params_suppkey,
    const int32_t* __restrict__ delta_bits_suppkey,
    const int64_t* __restrict__ bit_offsets_suppkey,
    // Revenue column compressed metadata
    const uint32_t* __restrict__ delta_array_revenue,
    const int32_t* __restrict__ model_types_revenue,
    const double* __restrict__ model_params_revenue,
    const int32_t* __restrict__ delta_bits_revenue,
    const int64_t* __restrict__ bit_offsets_revenue,
    // Partition boundaries
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    // Hash tables
    const uint32_t* __restrict__ ht_d_keys,  // Keys only for Dec1997
    int ht_d_size,
    const uint32_t* __restrict__ ht_c_keys,
    const uint32_t* __restrict__ ht_c_values,
    int ht_c_size,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    // Output: aggregation by (c_city, s_city)
    unsigned long long* __restrict__ agg_revenue)
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

        // Column 0: orderdate (filter first)
        loadColumnMetadata(smem.columns[0], partition_id,
            model_types_orderdate, model_params_orderdate,
            delta_bits_orderdate, bit_offsets_orderdate);

        // Column 1: custkey
        loadColumnMetadata(smem.columns[1], partition_id,
            model_types_custkey, model_params_custkey,
            delta_bits_custkey, bit_offsets_custkey);

        // Column 2: suppkey
        loadColumnMetadata(smem.columns[2], partition_id,
            model_types_suppkey, model_params_suppkey,
            delta_bits_suppkey, bit_offsets_suppkey);

        // Column 3: revenue
        loadColumnMetadata(smem.columns[3], partition_id,
            model_types_revenue, model_params_revenue,
            delta_bits_revenue, bit_offsets_revenue);
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
        uint32_t orderdate[FUSED_ITEMS_PER_THREAD];
        uint32_t custkey[FUSED_ITEMS_PER_THREAD];
        uint32_t suppkey[FUSED_ITEMS_PER_THREAD];
        uint32_t revenue[FUSED_ITEMS_PER_THREAD];
        uint32_t c_city[FUSED_ITEMS_PER_THREAD];
        uint32_t s_city[FUSED_ITEMS_PER_THREAD];

        // Initialize selection flags
        l3_fused::InitFlags<FUSED_ITEMS_PER_THREAD>(selection_flags, tile_size);

        // ---- Column 1: Orderdate (filter Dec1997 first - most selective) ----
        decompressTileCoarsened<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_orderdate, smem.columns[0], tile_start,
            orderdate, selection_flags, partition_size);

        // Check if in Dec1997 using hash table (keys only)
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_exists_fast(ht_d_keys, ht_d_size, orderdate[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after date probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 2: Custkey (conditional, get c_city) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_custkey, smem.columns[1], tile_start,
            custkey, selection_flags, partition_size);

        // Probe customer hash table (c_city = UK1 or UK5)
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (!ht_probe_get_fast(ht_c_keys, ht_c_values, ht_c_size, custkey[i], c_city[i])) {
                    selection_flags[i] = 0;
                }
            }
        }

        // Early termination after customer probe
        if (l3_fused::IsTerm<FUSED_ITEMS_PER_THREAD>(selection_flags)) continue;

        // ---- Column 3: Suppkey (conditional, get s_city) ----
        decompressTileConditional<uint32_t, FUSED_ITEMS_PER_THREAD>(
            delta_array_suppkey, smem.columns[2], tile_start,
            suppkey, selection_flags, partition_size);

        // Probe supplier hash table (s_city = UK1 or UK5)
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

        // ---- Aggregation by (c_city, s_city) ----
        #pragma unroll
        for (int i = 0; i < FUSED_ITEMS_PER_THREAD; ++i) {
            if (selection_flags[i]) {
                if (c_city[i] < AGG_NUM_CITIES && s_city[i] < AGG_NUM_CITIES) {
                    int agg_idx = c_city[i] * AGG_NUM_CITIES + s_city[i];
                    atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
                }
            }
        }
    }
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ34TrueFused(SSBDataCompressed& data, QueryTiming& timing) {
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

    // Date hash table (Dec1997 only, keys only)
    HashTable ht_date;
    ht_date.allocate(D_LEN, false);
    build_date_ht_month_kernel<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, D_LEN,
        DATE_DEC1997_START, DATE_DEC1997_END,
        ht_date.d_keys, ht_date.size);

    // Customer hash table (c_city = UK1 or UK5, custkey -> city)
    HashTable ht_customer;
    ht_customer.allocate(C_LEN, true);
    build_customer_ht_city_filter_kernel<<<(C_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_c_custkey, data.d_c_city, C_LEN,
        CITY_UK1, CITY_UK5,
        ht_customer.d_keys, ht_customer.d_values, ht_customer.size);

    // Supplier hash table (s_city = UK1 or UK5, suppkey -> city)
    HashTable ht_supplier;
    ht_supplier.allocate(S_LEN, true);
    build_supplier_ht_city_filter_kernel<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_city, S_LEN,
        CITY_UK1, CITY_UK5,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size);

    cudaEventRecord(ht_stop);
    cudaEventSynchronize(ht_stop);
    float ht_ms;
    cudaEventElapsedTime(&ht_ms, ht_start, ht_stop);
    timing.hash_build_ms = ht_ms;

    // -------------------------------------------------------------------------
    // Step 2: Allocate aggregation arrays
    // -------------------------------------------------------------------------
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    // -------------------------------------------------------------------------
    // Step 3: Warmup run
    // -------------------------------------------------------------------------
    q34_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        data.lo_custkey->delta_array, data.lo_custkey->d_model_types,
        data.lo_custkey->d_model_params, data.lo_custkey->d_delta_bits,
        data.lo_custkey->d_delta_array_bit_offsets,
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        ht_date.d_keys, ht_date.size,
        ht_customer.d_keys, ht_customer.d_values, ht_customer.size,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
        d_agg_revenue
    );
    cudaDeviceSynchronize();

    // -------------------------------------------------------------------------
    // Step 4: Timed run
    // -------------------------------------------------------------------------
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));
    cudaEventRecord(start);

    q34_optimized_fused_kernel<<<num_partitions, FUSED_BLOCK_SIZE>>>(
        data.lo_orderdate->delta_array, data.lo_orderdate->d_model_types,
        data.lo_orderdate->d_model_params, data.lo_orderdate->d_delta_bits,
        data.lo_orderdate->d_delta_array_bit_offsets,
        data.lo_custkey->delta_array, data.lo_custkey->d_model_types,
        data.lo_custkey->d_model_params, data.lo_custkey->d_delta_bits,
        data.lo_custkey->d_delta_array_bit_offsets,
        data.lo_suppkey->delta_array, data.lo_suppkey->d_model_types,
        data.lo_suppkey->d_model_params, data.lo_suppkey->d_delta_bits,
        data.lo_suppkey->d_delta_array_bit_offsets,
        data.lo_revenue->delta_array, data.lo_revenue->d_model_types,
        data.lo_revenue->d_model_params, data.lo_revenue->d_delta_bits,
        data.lo_revenue->d_delta_array_bit_offsets,
        data.lo_orderdate->d_start_indices, data.lo_orderdate->d_end_indices,
        num_partitions,
        ht_date.d_keys, ht_date.size,
        ht_customer.d_keys, ht_customer.d_values, ht_customer.size,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
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
    std::vector<unsigned long long> h_agg_revenue(AGG_SIZE);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q3.4 Results (Optimized True Fused) ===" << std::endl;
    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int cc = 0; cc < AGG_NUM_CITIES; ++cc) {
        for (int sc = 0; sc < AGG_NUM_CITIES; ++sc) {
            int idx = cc * AGG_NUM_CITIES + sc;
            if (h_agg_revenue[idx] > 0) {
                num_results++;
                total_revenue += h_agg_revenue[idx];
            }
        }
    }
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    std::cout << "Partitions: " << num_partitions << std::endl;
    timing.print("Q3.4 Optimized True Fused");

    // Cleanup
    cudaFree(d_agg_revenue);
    ht_date.free();
    ht_customer.free();
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

    std::cout << "=== SSB Q3.4 - Optimized True Fused Decompression ===" << std::endl;
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
        runQ34TrueFused(data, timing);
        std::cout << "Run " << (i+1) << ": " << timing.total_ms << " ms"
                  << " (HashBuild=" << timing.hash_build_ms << " ms, Kernel=" << timing.kernel_ms << " ms)" << std::endl;
    }

    data.free();
    return 0;
}
