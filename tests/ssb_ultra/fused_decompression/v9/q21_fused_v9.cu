/**
 * @file q21_fused_v9.cu
 * @brief SSB Q2.1 with V9 - Vertical-Compatible Architecture
 *
 * V9 Key Changes: Match Vertical' parallelism model
 *
 * Vertical approach:
 * - 32 threads, 8 items per thread = 256 values per block
 * - ~234K blocks for 60M rows (vs L3's 58K blocks with 1024 values/tile)
 * - Direct global atomics (no warp shuffle overhead)
 *
 * This version uses L3's 1024-value partitions but processes each
 * as 4 separate 256-value blocks for better parallelism.
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

// V9: Vertical-compatible - process 256 values per block
constexpr int V9_BLOCK_SIZE = 256;
constexpr int V9_ITEMS_PER_THREAD = 8;
constexpr int V9_BLOCK_THREADS = 32;

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;
constexpr int AGG_NUM_BRANDS = 1000;

// Filter constants
constexpr uint32_t P_CATEGORY_MFGR12 = 12;
constexpr uint32_t S_REGION_AMERICA = 1;

// ============================================================================
// V9 Kernel - Process 256-value blocks (4 blocks per 1024-value partition)
// ============================================================================

__global__ void q21_fused_v9_kernel(
    // Compressed columns
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    const Q2xTileMetadata* __restrict__ tile_meta,
    int num_tiles,  // Original L3 tile count (1024 values each)
    int total_rows,
    // Hash tables (Crystal-style)
    const uint32_t* __restrict__ ht_s,
    int ht_s_len,
    const uint32_t* __restrict__ ht_p,
    int ht_p_len,
    const uint32_t* __restrict__ ht_d,
    int ht_d_len,
    // Output
    unsigned long long* __restrict__ agg_revenue)
{
    // V9: Each block processes 256 values (1 mini-vector)
    // 4 blocks per original L3 tile
    int block_idx = blockIdx.x;
    int partition_idx = block_idx / 4;  // Which 1024-value partition
    int mv_idx = block_idx % 4;         // Which mini-vector (0-3)

    if (partition_idx >= num_tiles) return;

    int lane = threadIdx.x;
    int partition_start = partition_idx * TILE_SIZE;
    int partition_size = min(TILE_SIZE, total_rows - partition_start);
    int mv_start = mv_idx * MINI_VEC_SIZE;
    int mv_valid = min(MINI_VEC_SIZE, partition_size - mv_start);

    if (mv_valid <= 0) return;

    // Load tile metadata
    Q2xTileMetadata meta = tile_meta[partition_idx];

    // Get FOR base values
    uint32_t supp_base = static_cast<uint32_t>(__double2ll_rn(meta.suppkey.params[0]));
    uint32_t partkey_base = static_cast<uint32_t>(__double2ll_rn(meta.partkey.params[0]));
    uint32_t od_base = static_cast<uint32_t>(__double2ll_rn(meta.orderdate.params[0]));
    uint32_t rev_base = static_cast<uint32_t>(__double2ll_rn(meta.revenue.params[0]));

    // Initialize flags
    int flags[V9_ITEMS_PER_THREAD];
    #pragma unroll
    for (int v = 0; v < V9_ITEMS_PER_THREAD; v++) {
        int idx_in_mv = v * V9_BLOCK_THREADS + lane;
        flags[v] = (idx_in_mv < mv_valid) ? 1 : 0;
    }

    int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 1: Suppkey ==========
    uint32_t suppkey[V9_ITEMS_PER_THREAD];
    {
        int local_mv_idx = (meta.suppkey.local_offset + mv_start) / MINI_VEC_SIZE;
        int64_t mv_bit_base = meta.suppkey.partition_bit_base +
            static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.suppkey.delta_bits;

        prefetchExtract8FOR(supp_deltas, mv_bit_base, lane,
            meta.suppkey.delta_bits, supp_base, suppkey);
    }

    #pragma unroll
    for (int v = 0; v < V9_ITEMS_PER_THREAD; v++) {
        if (flags[v]) {
            int slot = suppkey[v] % ht_s_len;
            if (__ldg(&ht_s[slot]) == 0) {
                flags[v] = 0;
            }
        }
    }

    any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 2: Partkey ==========
    uint32_t partkey[V9_ITEMS_PER_THREAD];
    uint32_t brand[V9_ITEMS_PER_THREAD];
    {
        int local_mv_idx = (meta.partkey.local_offset + mv_start) / MINI_VEC_SIZE;
        int64_t mv_bit_base = meta.partkey.partition_bit_base +
            static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.partkey.delta_bits;

        prefetchExtract8FORConditional(part_deltas, mv_bit_base, lane,
            meta.partkey.delta_bits, partkey_base, flags, partkey);
    }

    #pragma unroll
    for (int v = 0; v < V9_ITEMS_PER_THREAD; v++) {
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
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 3: Orderdate ==========
    uint32_t orderdate[V9_ITEMS_PER_THREAD];
    uint32_t year[V9_ITEMS_PER_THREAD];
    {
        int local_mv_idx = (meta.orderdate.local_offset + mv_start) / MINI_VEC_SIZE;
        int64_t mv_bit_base = meta.orderdate.partition_bit_base +
            static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;

        prefetchExtract8FORConditional(od_deltas, mv_bit_base, lane,
            meta.orderdate.delta_bits, od_base, flags, orderdate);
    }

    #pragma unroll
    for (int v = 0; v < V9_ITEMS_PER_THREAD; v++) {
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
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 4: Revenue ==========
    uint32_t revenue[V9_ITEMS_PER_THREAD];
    {
        int local_mv_idx = (meta.revenue.local_offset + mv_start) / MINI_VEC_SIZE;
        int64_t mv_bit_base = meta.revenue.partition_bit_base +
            static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.revenue.delta_bits;

        prefetchExtract8FORConditional(rev_deltas, mv_bit_base, lane,
            meta.revenue.delta_bits, rev_base, flags, revenue);
    }

    // ========== Aggregation (Vertical-style direct atomics) ==========
    #pragma unroll
    for (int v = 0; v < V9_ITEMS_PER_THREAD; v++) {
        if (flags[v]) {
            int year_idx = year[v] - 1992;
            if (year_idx >= 0 && year_idx < AGG_NUM_YEARS && brand[v] < AGG_NUM_BRANDS) {
                int agg_idx = year_idx * AGG_NUM_BRANDS + brand[v];
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[v]));
            }
        }
    }
}

// ============================================================================
// Host Code
// ============================================================================

void runQ21FusedV9(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Build tile metadata (same as V7)
    auto tile_meta = TileMetadataBuilder<uint32_t>::buildQ2xMetadata(
        data.lo_suppkey, data.lo_partkey, data.lo_orderdate, data.lo_revenue, total_rows);

    Q2xTileMetadataGPU gpu_meta;
    gpu_meta.allocate(tile_meta.size());
    gpu_meta.upload(tile_meta);

    int block_size = 256;

    // Build Crystal-style hash tables
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
    // V9: 4 blocks per tile (256 values each = 1024 total)
    int num_blocks = num_tiles * 4;

    std::cout << "\nV9 Configuration:" << std::endl;
    std::cout << "  Block size: 256 values (Vertical-compatible)" << std::endl;
    std::cout << "  Original tiles: " << num_tiles << " (1024 values each)" << std::endl;
    std::cout << "  V9 blocks: " << num_blocks << " (4x more parallelism)" << std::endl;
    std::cout << "  Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
        q21_fused_v9_kernel<<<num_blocks, V9_BLOCK_THREADS>>>(
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
    q21_fused_v9_kernel<<<num_blocks, V9_BLOCK_THREADS>>>(
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

    std::cout << "\n=== Q2.1 V9 Verification ===" << std::endl;
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
    std::cout << "\n=== Benchmark Runs (V9 Vertical-Compatible) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

        cudaEventRecord(start);
        q21_fused_v9_kernel<<<num_blocks, V9_BLOCK_THREADS>>>(
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
    std::cout << "Average kernel (V9): " << avg_time << " ms" << std::endl;
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
    std::cout << "SSB Q2.1 - V9 Vertical-Compatible Architecture" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key changes from V7:" << std::endl;
    std::cout << "  - 4 blocks per partition (256 values each)" << std::endl;
    std::cout << "  - 4x more GPU blocks = better parallelism" << std::endl;
    std::cout << "  - Same decompression, same hash tables" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ21FusedV9(data);

    data.free();
    return 0;
}
