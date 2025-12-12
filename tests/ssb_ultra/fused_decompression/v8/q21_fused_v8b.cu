/**
 * @file q21_fused_v8b.cu
 * @brief SSB Q2.1 with V8b Optimizations - Warp Key Matching
 *
 * V8b Key Optimization: Use __match_any_sync for warp-level key deduplication
 *
 * This avoids the shared memory hash table overhead from V8.
 * Instead, use warp shuffle to find threads with same aggregation key.
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

// L3 headers
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

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;    // 1992-1998
constexpr int AGG_NUM_BRANDS = 1000;

// Filter constants
constexpr uint32_t P_CATEGORY_MFGR12 = 12;
constexpr uint32_t S_REGION_AMERICA = 1;

// ============================================================================
// Warp-Level Key Matching Aggregation Helper
// ============================================================================

__device__ __forceinline__
void warpAggregateByKey(
    int year, int brand, unsigned long long revenue,
    unsigned long long* global_agg,
    bool valid)
{
    if (!valid || revenue == 0) return;

    // Pack (year, brand) into single key for matching
    uint32_t key = (static_cast<uint32_t>(year) << 16) | static_cast<uint32_t>(brand);

    int lane_id = threadIdx.x & 31;

    // Find all threads in warp with same key
    unsigned int match_mask = __match_any_sync(0xFFFFFFFF, key);

    // Find lowest lane with same key (the "leader")
    int leader = __ffs(match_mask) - 1;

    if (leader == lane_id) {
        // I'm the leader for this key - sum all matching lanes' revenues
        unsigned long long total = 0;
        unsigned int mask = match_mask;
        while (mask) {
            int src = __ffs(mask) - 1;
            total += __shfl_sync(match_mask, revenue, src);
            mask &= (mask - 1);  // Clear lowest bit
        }

        // Single atomic per unique key in warp
        int global_idx = year * AGG_NUM_BRANDS + brand;
        atomicAdd(&global_agg[global_idx], total);
    }
}

// ============================================================================
// V8b Kernel with Warp Key Matching
// ============================================================================

__global__ void q21_fused_v8b_kernel(
    // Compressed columns
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    const Q2xTileMetadata* __restrict__ tile_meta,
    int num_tiles,
    int total_rows,
    // Hash tables (Crystal-style)
    const uint32_t* __restrict__ ht_s,     // Supplier: keys only
    int ht_s_len,
    const uint32_t* __restrict__ ht_p,     // Part: interleaved [key,brand,...]
    int ht_p_len,
    const uint32_t* __restrict__ ht_d,     // Date: interleaved [datekey,year,...]
    int ht_d_len,
    // Output
    unsigned long long* __restrict__ agg_revenue)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int lane_id = threadIdx.x;
    int tile_start = tile_idx * TILE_SIZE;
    int tile_size = min(TILE_SIZE, total_rows - tile_start);

    // Load tile metadata
    Q2xTileMetadata meta = tile_meta[tile_idx];

    // Get FOR base values
    uint32_t supp_base = static_cast<uint32_t>(__double2ll_rn(meta.suppkey.params[0]));
    uint32_t part_base = static_cast<uint32_t>(__double2ll_rn(meta.partkey.params[0]));
    uint32_t od_base = static_cast<uint32_t>(__double2ll_rn(meta.orderdate.params[0]));
    uint32_t rev_base = static_cast<uint32_t>(__double2ll_rn(meta.revenue.params[0]));

    // Process 4 mini-vectors per tile
    #pragma unroll 4
    for (int mv = 0; mv < 4; mv++) {
        int mv_start_in_tile = mv * MINI_VEC_SIZE;
        int mv_valid = min(MINI_VEC_SIZE, tile_size - mv_start_in_tile);

        // Initialize validity flags
        int flags[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            int idx_in_mv = v * WARP_SZ + lane_id;
            flags[v] = (idx_in_mv < mv_valid) ? 1 : 0;
        }

        int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 1: Suppkey (filter via supplier HT) ==========
        uint32_t suppkey[8];
        {
            int local_mv_idx = (meta.suppkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.suppkey.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.suppkey.delta_bits;

            prefetchExtract8FOR(supp_deltas, mv_bit_base, lane_id,
                meta.suppkey.delta_bits, supp_base, suppkey);
        }

        // V7 Crystal-style single-probe
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

        // ========== Column 2: Partkey (filter + get brand) ==========
        uint32_t partkey[8];
        uint32_t brand[8];
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
                int slot = partkey[v] % ht_p_len;
                uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht_p[slot << 1]);
                if (kv != 0) {
                    brand[v] = static_cast<uint32_t>(kv >> 32);
                } else {
                    flags[v] = 0;
                }
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 3: Orderdate (get year) ==========
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

        // ========== Column 4: Revenue ==========
        uint32_t revenue[8];
        {
            int local_mv_idx = (meta.revenue.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.revenue.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.revenue.delta_bits;

            prefetchExtract8FORConditional(rev_deltas, mv_bit_base, lane_id,
                meta.revenue.delta_bits, rev_base, flags, revenue);
        }

        // ========== V8b: Warp-Level Key Matching Aggregation ==========
        // Process each value position separately to enable warp matching
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            bool valid = flags[v];
            int year_idx = valid ? (year[v] - 1992) : -1;
            int brand_val = valid ? brand[v] : -1;
            unsigned long long rev_val = valid ? static_cast<unsigned long long>(revenue[v]) : 0;

            // Validate bounds
            if (valid && (year_idx < 0 || year_idx >= AGG_NUM_YEARS || brand_val >= AGG_NUM_BRANDS)) {
                valid = false;
                rev_val = 0;
            }

            warpAggregateByKey(year_idx, brand_val, rev_val, agg_revenue, valid);
        }
    }
}

// ============================================================================
// Host Code
// ============================================================================

void runQ21FusedV8b(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Build tile metadata
    auto tile_meta = TileMetadataBuilder<uint32_t>::buildQ2xMetadata(
        data.lo_suppkey, data.lo_partkey, data.lo_orderdate, data.lo_revenue, total_rows);

    Q2xTileMetadataGPU gpu_meta;
    gpu_meta.allocate(tile_meta.size());
    gpu_meta.upload(tile_meta);

    int block_size = 256;

    // Build Crystal-style hash tables (same as V7)
    CrystalHashTable ht_date;
    ht_date.allocate(DATE_HT_LEN, true);
    build_date_ht_crystal<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_date.d_data, DATE_HT_LEN);

    CrystalHashTable ht_part;
    ht_part.allocate(P_LEN, true);
    build_part_ht_category_crystal<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        P_CATEGORY_MFGR12, ht_part.d_data, P_LEN);

    CrystalHashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    build_supplier_ht_region_crystal<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_AMERICA,
        ht_supplier.d_data, S_LEN);
    cudaDeviceSynchronize();

    int agg_size = AGG_NUM_YEARS * AGG_NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;
    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;
    std::cout << "V8b Optimization: Warp-level key matching (__match_any_sync)" << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
        q21_fused_v8b_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_supplier.d_data, S_LEN,
            ht_part.d_data, P_LEN,
            ht_date.d_data, DATE_HT_LEN,
            d_agg_revenue);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Verify correctness
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
    q21_fused_v8b_kernel<<<num_tiles, WARP_SZ>>>(
        data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
        data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
        gpu_meta.d_tiles, num_tiles, total_rows,
        ht_supplier.d_data, S_LEN,
        ht_part.d_data, P_LEN,
        ht_date.d_data, DATE_HT_LEN,
        d_agg_revenue);
    cudaDeviceSynchronize();

    std::vector<unsigned long long> h_agg_revenue(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 V8b Verification ===" << std::endl;
    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int y = 0; y < AGG_NUM_YEARS; ++y) {
        for (int b = 0; b < AGG_NUM_BRANDS; ++b) {
            int idx = y * AGG_NUM_BRANDS + b;
            if (h_agg_revenue[idx] > 0) {
                num_results++;
                total_revenue += h_agg_revenue[idx];
            }
        }
    }
    std::cout << "Total groups: " << num_results << " (expected: 280)" << std::endl;
    std::cout << "Total revenue: " << total_revenue << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V8b Warp Key Matching) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

        cudaEventRecord(start);
        q21_fused_v8b_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
            gpu_meta.d_tiles, num_tiles, total_rows,
            ht_supplier.d_data, S_LEN,
            ht_part.d_data, P_LEN,
            ht_date.d_data, DATE_HT_LEN,
            d_agg_revenue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    float avg_time = total_time / RUNS;
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average kernel (V8b): " << avg_time << " ms" << std::endl;
    std::cout << "V7 baseline: 1.67 ms" << std::endl;
    std::cout << "Vertical target: 0.89 ms" << std::endl;
    std::cout << "Speedup vs V7: " << (1.67 / avg_time) << "x" << std::endl;
    std::cout << "Gap to Vertical: " << (avg_time / 0.89) << "x" << std::endl;

    cudaFree(d_agg_revenue);
    gpu_meta.free();
    ht_date.free();
    ht_part.free();
    ht_supplier.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q2.1 - V8b Warp Key Matching" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key optimization: __match_any_sync for warp-level key deduplication" << std::endl;
    std::cout << "  - V8: Shared memory hash (too many collisions)" << std::endl;
    std::cout << "  - V8b: Warp shuffle key matching (no shared memory)" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ21FusedV8b(data);

    data.free();
    return 0;
}
