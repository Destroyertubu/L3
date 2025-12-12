/**
 * @file q43_fused_v5.cu
 * @brief SSB Q4.3 with L3 Native Format + Prefetch Strategy
 *
 * Query:
 *   SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM lineorder, customer, supplier, part, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND s_nation = 'UNITED STATES'
 *     AND (d_year = 1997 OR d_year = 1998)
 *     AND p_category = 'MFGR#14'
 *   GROUP BY d_year, s_city, p_brand1
 *
 * Differences from Q4.2:
 *   - Customer: no filter (just exists check via custkey probe)
 *   - Supplier: filter by nation=US, get city value
 *   - Part: filter by category=MFGR#14, get brand value
 *   - Group by: year, s_city, p_brand1
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "tile_metadata.cuh"
#include "prefetch_helpers_v5.cuh"
#include "fused_hash_table.cuh"

using namespace l3_fused;
using namespace l3_v5;
using namespace ssb;

constexpr int AGG_NUM_YEARS = 2;       // 1997-1998
constexpr int AGG_NUM_CITIES = 250;
constexpr int AGG_NUM_BRANDS = 1000;
constexpr uint32_t NATION_US = 24;     // UNITED STATES
constexpr uint32_t CATEGORY_MFGR14 = 4; // MFGR#14 (category 4 in MFGR#1 range)

// ============================================================================
// Main Fused Kernel V5 - Q4.3
// ============================================================================

__global__ void q43_fused_v5_kernel(
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    const uint32_t* __restrict__ cost_deltas,
    const Q4xTileMetadata* __restrict__ tile_meta,
    int num_tiles,
    int total_rows,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    long long* __restrict__ agg_profit)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int lane_id = threadIdx.x;
    int tile_start = tile_idx * TILE_SIZE;
    int tile_size = min(TILE_SIZE, total_rows - tile_start);

    Q4xTileMetadata meta = tile_meta[tile_idx];

    uint32_t supp_base = static_cast<uint32_t>(__double2ll_rn(meta.suppkey.params[0]));
    uint32_t part_base = static_cast<uint32_t>(__double2ll_rn(meta.partkey.params[0]));
    uint32_t od_base = static_cast<uint32_t>(__double2ll_rn(meta.orderdate.params[0]));
    uint32_t rev_base = static_cast<uint32_t>(__double2ll_rn(meta.revenue.params[0]));
    uint32_t cost_base = static_cast<uint32_t>(__double2ll_rn(meta.supplycost.params[0]));

    #pragma unroll 4
    for (int mv = 0; mv < 4; mv++) {
        int mv_start_in_tile = mv * MINI_VEC_SIZE;
        int mv_valid = min(MINI_VEC_SIZE, tile_size - mv_start_in_tile);

        int flags[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            int idx_in_mv = v * WARP_SZ + lane_id;
            flags[v] = (idx_in_mv < mv_valid) ? 1 : 0;
        }

        int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // Q4.3 does not filter on customer, so skip custkey decompression
        // Just go directly to suppkey

        // ========== Column 1: Suppkey (filter by nation=US, get city) ==========
        uint32_t suppkey[8];
        uint32_t s_city[8];
        {
            int local_mv_idx = (meta.suppkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.suppkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.suppkey.delta_bits;
            prefetchExtract8FOR(supp_deltas, mv_bit_base, lane_id,
                meta.suppkey.delta_bits, supp_base, suppkey);
        }

        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                if (!ht_probe_get_fast(ht_s_keys, ht_s_values, ht_s_size, suppkey[v], s_city[v])) {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 2: Partkey (filter by category=MFGR#14, get brand) ==========
        uint32_t partkey[8];
        uint32_t p_brand[8];
        {
            int local_mv_idx = (meta.partkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.partkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.partkey.delta_bits;
            prefetchExtract8FORConditional(part_deltas, mv_bit_base, lane_id,
                meta.partkey.delta_bits, part_base, flags, partkey);
        }

        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                if (!ht_probe_get_fast(ht_p_keys, ht_p_values, ht_p_size, partkey[v], p_brand[v])) {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 3: Orderdate (filter 1997/1998, get year) ==========
        uint32_t orderdate[8];
        uint32_t year[8];
        {
            int local_mv_idx = (meta.orderdate.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.orderdate.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;
            prefetchExtract8FORConditional(od_deltas, mv_bit_base, lane_id,
                meta.orderdate.delta_bits, od_base, flags, orderdate);
        }

        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                if (!ht_probe_get_fast(ht_d_keys, ht_d_values, ht_d_size, orderdate[v], year[v])) {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 4: Revenue ==========
        uint32_t revenue[8];
        {
            int local_mv_idx = (meta.revenue.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.revenue.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.revenue.delta_bits;
            prefetchExtract8FORConditional(rev_deltas, mv_bit_base, lane_id,
                meta.revenue.delta_bits, rev_base, flags, revenue);
        }

        // ========== Column 5: Supplycost ==========
        uint32_t supplycost[8];
        {
            int local_mv_idx = (meta.supplycost.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.supplycost.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.supplycost.delta_bits;
            prefetchExtract8FORConditional(cost_deltas, mv_bit_base, lane_id,
                meta.supplycost.delta_bits, cost_base, flags, supplycost);
        }

        // ========== Aggregate profit by (d_year, s_city, p_brand1) ==========
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int year_idx = year[v] - 1997;  // 0 or 1
                if (year_idx >= 0 && year_idx < AGG_NUM_YEARS &&
                    s_city[v] < AGG_NUM_CITIES &&
                    p_brand[v] < AGG_NUM_BRANDS) {
                    int agg_idx = year_idx * AGG_NUM_CITIES * AGG_NUM_BRANDS +
                                  s_city[v] * AGG_NUM_BRANDS + p_brand[v];
                    long long profit = static_cast<long long>(revenue[v]) -
                                       static_cast<long long>(supplycost[v]);
                    // atomicAdd doesn't support long long directly, use unsigned long long
                    atomicAdd(reinterpret_cast<unsigned long long*>(&agg_profit[agg_idx]),
                              static_cast<unsigned long long>(profit));
                }
            }
        }
    }
}

// ============================================================================
// Host Code
// ============================================================================

void runQ43FusedV5(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    auto tile_meta = TileMetadataBuilder<uint32_t>::buildQ4xMetadata(
        data.lo_custkey, data.lo_suppkey, data.lo_partkey,
        data.lo_orderdate, data.lo_revenue, data.lo_supplycost, total_rows);
    Q4xTileMetadataGPU gpu_meta;
    gpu_meta.allocate(tile_meta.size());
    gpu_meta.upload(tile_meta);

    int block_size = 256;

    // Date: year 1997 or 1998
    HashTable ht_date;
    ht_date.allocate(D_LEN, true);
    build_date_ht_year_range_kernel<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN,
        1997, 1998,
        ht_date.d_keys, ht_date.d_values, ht_date.size);

    // Supplier: nation=UNITED STATES -> city
    HashTable ht_supplier;
    ht_supplier.allocate(S_LEN, true);
    build_supplier_ht_nation_city_kernel<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN,
        NATION_US,
        ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size);

    // Part: category=MFGR#14 -> brand1
    HashTable ht_part;
    ht_part.allocate(P_LEN, true);
    build_part_ht_category_brand_kernel<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        CATEGORY_MFGR14,
        ht_part.d_keys, ht_part.d_values, ht_part.size);
    cudaDeviceSynchronize();

    int agg_size = AGG_NUM_YEARS * AGG_NUM_CITIES * AGG_NUM_BRANDS;
    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, agg_size * sizeof(long long));

    int num_tiles = gpu_meta.num_tiles;
    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
        q43_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
            ht_part.d_keys, ht_part.d_values, ht_part.size,
            ht_date.d_keys, ht_date.d_values, ht_date.size,
            d_agg_profit);
    }
    cudaDeviceSynchronize();

    // Benchmark
    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
        cudaEventRecord(start);
        q43_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_supplier.d_keys, ht_supplier.d_values, ht_supplier.size,
            ht_part.d_keys, ht_part.d_values, ht_part.size,
            ht_date.d_keys, ht_date.d_values, ht_date.size,
            d_agg_profit);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average kernel: " << (total_time / RUNS) << " ms" << std::endl;

    cudaFree(d_agg_profit);
    gpu_meta.free();
    ht_date.free();
    ht_supplier.free();
    ht_part.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q4.3 - L3 Native Format + Prefetch Strategy (V5)" << std::endl;
    std::cout << "======================================================" << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    runQ43FusedV5(data);
    data.free();
    return 0;
}
