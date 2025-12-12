/**
 * @file q41_fused_v7.cu
 * @brief SSB Q4.1 with Crystal-Style Perfect Hash (Vertical Compatible)
 *
 * V7 Key Change: Use Vertical-style hash tables with SINGLE memory access per probe.
 *
 * Query:
 *   SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM lineorder, customer, supplier, part, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND c_region = 'AMERICA' AND s_region = 'AMERICA'
 *     AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 *   GROUP BY d_year, c_nation
 *
 * 4 hash tables: Customer, Supplier, Part, Date
 * 6 columns: custkey, suppkey, partkey, orderdate, revenue, supplycost
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

// V7 Crystal-style hash headers
#include "v7/crystal_hash_v7.cuh"
#include "v7/crystal_hash_build_v7.cuh"

using namespace l3_fused;
using namespace l3_v5;
using namespace l3_crystal;
using namespace ssb;

constexpr int AGG_NUM_YEARS = 7;
constexpr int AGG_NUM_NATIONS = 25;
constexpr uint32_t REGION_AMERICA = 1;
constexpr uint32_t MFGR_1 = 1;
constexpr uint32_t MFGR_2 = 2;

// ============================================================================
// Main Fused Kernel V7 - Q4.1 with Crystal-Style Hash
// ============================================================================

__global__ void q41_fused_v7_kernel(
    // Compressed columns
    const uint32_t* __restrict__ cust_deltas,
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    const uint32_t* __restrict__ cost_deltas,
    const Q4xTileMetadata* __restrict__ tile_meta,
    int num_tiles,
    int total_rows,
    // Crystal-style hash tables
    const uint32_t* __restrict__ ht_c,     // Customer: custkey -> nation
    int ht_c_len,                           // C_LEN
    const uint32_t* __restrict__ ht_s,     // Supplier: keys only
    int ht_s_len,                           // S_LEN
    const uint32_t* __restrict__ ht_p,     // Part: keys only
    int ht_p_len,                           // P_LEN
    const uint32_t* __restrict__ ht_d,     // Date: datekey -> year
    int ht_d_len,                           // DATE_HT_LEN
    // Output
    long long* __restrict__ agg_profit)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int lane_id = threadIdx.x;
    int tile_start = tile_idx * TILE_SIZE;
    int tile_size = min(TILE_SIZE, total_rows - tile_start);

    Q4xTileMetadata meta = tile_meta[tile_idx];

    uint32_t cust_base = static_cast<uint32_t>(__double2ll_rn(meta.custkey.params[0]));
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

        // ========== Column 1: Custkey (get nation from Customer HT) ==========
        uint32_t custkey[8];
        uint32_t c_nation[8];
        {
            int local_mv_idx = (meta.custkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.custkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.custkey.delta_bits;
            prefetchExtract8FOR(cust_deltas, mv_bit_base, lane_id,
                meta.custkey.delta_bits, cust_base, custkey);
        }

        // V7: Crystal-style single-probe with value retrieval
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int slot = custkey[v] % ht_c_len;
                uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht_c[slot << 1]);
                if (kv != 0) {
                    c_nation[v] = static_cast<uint32_t>(kv >> 32);
                } else {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 2: Suppkey (filter via Supplier HT) ==========
        uint32_t suppkey[8];
        {
            int local_mv_idx = (meta.suppkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.suppkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.suppkey.delta_bits;
            prefetchExtract8FORConditional(supp_deltas, mv_bit_base, lane_id,
                meta.suppkey.delta_bits, supp_base, flags, suppkey);
        }

        // V7: Crystal-style single-probe (keys only)
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int slot = suppkey[v] % ht_s_len;
                if (__ldg(&ht_s[slot]) == 0) {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 3: Partkey (filter via Part HT) ==========
        uint32_t partkey[8];
        {
            int local_mv_idx = (meta.partkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.partkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.partkey.delta_bits;
            prefetchExtract8FORConditional(part_deltas, mv_bit_base, lane_id,
                meta.partkey.delta_bits, part_base, flags, partkey);
        }

        // V7: Crystal-style single-probe (keys only)
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int slot = partkey[v] % ht_p_len;
                if (__ldg(&ht_p[slot]) == 0) {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 4: Orderdate (get year from Date HT) ==========
        uint32_t orderdate[8];
        uint32_t year[8];
        {
            int local_mv_idx = (meta.orderdate.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.orderdate.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;
            prefetchExtract8FORConditional(od_deltas, mv_bit_base, lane_id,
                meta.orderdate.delta_bits, od_base, flags, orderdate);
        }

        // V7: Crystal-style date probe with direct indexing
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int slot = CRYSTAL_HASH(orderdate[v], ht_d_len, DATE_KEY_MIN);
                uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht_d[slot << 1]);
                if (kv != 0) {
                    year[v] = static_cast<uint32_t>(kv >> 32);
                } else {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 5: Revenue ==========
        uint32_t revenue[8];
        {
            int local_mv_idx = (meta.revenue.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.revenue.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.revenue.delta_bits;
            prefetchExtract8FORConditional(rev_deltas, mv_bit_base, lane_id,
                meta.revenue.delta_bits, rev_base, flags, revenue);
        }

        // ========== Column 6: Supplycost ==========
        uint32_t supplycost[8];
        {
            int local_mv_idx = (meta.supplycost.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.supplycost.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.supplycost.delta_bits;
            prefetchExtract8FORConditional(cost_deltas, mv_bit_base, lane_id,
                meta.supplycost.delta_bits, cost_base, flags, supplycost);
        }

        // ========== Aggregate profit by (d_year, c_nation) ==========
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                int year_idx = year[v] - 1992;
                if (year_idx >= 0 && year_idx < AGG_NUM_YEARS &&
                    c_nation[v] < AGG_NUM_NATIONS) {
                    int agg_idx = year_idx * AGG_NUM_NATIONS + c_nation[v];
                    long long profit = static_cast<long long>(revenue[v]) -
                                       static_cast<long long>(supplycost[v]);
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

void runQ41FusedV7(SSBDataCompressedVertical& data) {
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

    // Date hash table: Crystal-style with direct indexing
    CrystalHashTable ht_date;
    ht_date.allocate(DATE_HT_LEN, true);
    build_date_ht_crystal<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN,
        ht_date.d_data, DATE_HT_LEN);

    // Customer hash table: Crystal-style (custkey -> nation, filtered by AMERICA)
    CrystalHashTable ht_customer;
    ht_customer.allocate(C_LEN, true);
    build_customer_ht_region_crystal<<<(C_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN,
        REGION_AMERICA,
        ht_customer.d_data, C_LEN);

    // Supplier hash table: Crystal-style (keys only, filtered by AMERICA)
    CrystalHashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    build_supplier_ht_region_crystal<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN,
        REGION_AMERICA,
        ht_supplier.d_data, S_LEN);

    // Part hash table: Crystal-style (keys only, filtered by MFGR#1 or MFGR#2)
    CrystalHashTable ht_part;
    ht_part.allocate(P_LEN, false);
    build_part_ht_mfgr_crystal<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_mfgr, P_LEN,
        MFGR_1, MFGR_2,
        ht_part.d_data, P_LEN);
    cudaDeviceSynchronize();

    int agg_size = AGG_NUM_YEARS * AGG_NUM_NATIONS;
    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, agg_size * sizeof(long long));

    int num_tiles = gpu_meta.num_tiles;
    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;
    std::cout << "Hash table sizes: Date=" << DATE_HT_LEN << ", Customer=" << C_LEN
              << ", Supplier=" << S_LEN << ", Part=" << P_LEN << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
        q41_fused_v7_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_customer.d_data, C_LEN,
            ht_supplier.d_data, S_LEN,
            ht_part.d_data, P_LEN,
            ht_date.d_data, DATE_HT_LEN,
            d_agg_profit);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Verify
    cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
    q41_fused_v7_kernel<<<num_tiles, WARP_SZ>>>(
        data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
        data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
        data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
        gpu_meta.d_tiles, num_tiles, total_rows,
        ht_customer.d_data, C_LEN,
        ht_supplier.d_data, S_LEN,
        ht_part.d_data, P_LEN,
        ht_date.d_data, DATE_HT_LEN,
        d_agg_profit);
    cudaDeviceSynchronize();

    std::vector<long long> h_agg(agg_size);
    cudaMemcpy(h_agg.data(), d_agg_profit, agg_size * sizeof(long long), cudaMemcpyDeviceToHost);

    int num_results = 0;
    long long total_profit = 0;
    for (int i = 0; i < agg_size; i++) {
        if (h_agg[i] != 0) {
            num_results++;
            total_profit += h_agg[i];
        }
    }
    std::cout << "\n=== Q4.1 V7 Verification ===" << std::endl;
    std::cout << "Total groups: " << num_results << " (expected: ~35)" << std::endl;
    std::cout << "Total profit: " << total_profit << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V7 Crystal Hash) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
        cudaEventRecord(start);
        q41_fused_v7_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_customer.d_data, C_LEN,
            ht_supplier.d_data, S_LEN,
            ht_part.d_data, P_LEN,
            ht_date.d_data, DATE_HT_LEN,
            d_agg_profit);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average kernel (V7): " << (total_time / RUNS) << " ms" << std::endl;
    std::cout << "V5 baseline: 8.51 ms" << std::endl;
    std::cout << "Vertical target: 2.73 ms" << std::endl;
    std::cout << "Speedup vs V5: " << (8.51 / (total_time / RUNS)) << "x" << std::endl;

    cudaFree(d_agg_profit);
    gpu_meta.free();
    ht_date.free();
    ht_customer.free();
    ht_supplier.free();
    ht_part.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q4.1 - V7 Crystal-Style Perfect Hash" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key optimization: Vertical-compatible single-probe hash" << std::endl;
    std::cout << "  - 4 hash tables: Customer, Supplier, Part, Date" << std::endl;
    std::cout << "  - 6 columns: custkey, suppkey, partkey, orderdate, revenue, supplycost" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    runQ41FusedV7(data);
    data.free();
    return 0;
}
