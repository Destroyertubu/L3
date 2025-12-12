/**
 * @file q21_fused_v6.cu
 * @brief SSB Q2.1 with V6 Hash Prefetch + Shared Memory Date HT
 *
 * V6 Optimizations:
 * 1. Date hash table in shared memory (20KB) - zero global memory latency
 * 2. Batch prefetch for supplier/part hash probes - hide memory latency
 * 3. Interleaved decompression + probing - overlap operations
 *
 * Expected: 4.38ms (V5) → ~1.5ms (V6)
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
#include "fused_hash_table.cuh"

// V6 headers
#include "v6/prefetch_hash_v6.cuh"
#include "v6/shared_hash_tables_v6.cuh"

using namespace l3_fused;
using namespace l3_v5;
using namespace l3_v6;
using namespace ssb;

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;    // 1992-1998
constexpr int AGG_NUM_BRANDS = 1000;

// Filter constants
constexpr uint32_t P_CATEGORY_MFGR12 = 12;
constexpr uint32_t S_REGION_AMERICA = 1;

// V6 block config - use fewer, larger blocks to amortize shared memory loading
constexpr int V6_BLOCK_SIZE = 128;  // 4 warps per block
constexpr int TILES_PER_WARP = 16;  // Each warp processes multiple tiles

// ============================================================================
// V6 Main Kernel - Q2.1 with Prefetch Hash + Shared Memory Date HT
// ============================================================================

__global__ __launch_bounds__(V6_BLOCK_SIZE)
void q21_fused_v6_kernel(
    // Compressed columns
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    // Tile metadata
    const Q2xTileMetadata* __restrict__ tile_meta,
    int num_tiles,
    int total_rows,
    // Hash tables (supplier, part in global memory)
    const uint32_t* __restrict__ ht_s_keys,
    int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    // Date dimension for shared memory loading
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int d_len,
    // Output
    unsigned long long* __restrict__ agg_revenue)
{
    // ========== Shared Memory Date Hash Table ==========
    __shared__ uint64_t ht_date_packed[DATE_HT_SIZE];  // ~20KB

    // Cooperative load to shared memory
    loadDateHTPacked(d_datekey, d_year, d_len, ht_date_packed);

    // Process tiles - each warp handles one tile
    int warp_id = threadIdx.x / WARP_SZ;
    int lane_id = threadIdx.x % WARP_SZ;
    int warps_per_block = V6_BLOCK_SIZE / WARP_SZ;

    // Calculate which tiles this block processes
    int tiles_per_block = warps_per_block;
    int block_tile_start = blockIdx.x * tiles_per_block;

    for (int t = 0; t < tiles_per_block; t++) {
        int tile_idx = block_tile_start + t;
        if (tile_idx >= num_tiles) return;

        // Only warp 0 processes tile 0, warp 1 processes tile 1, etc.
        if (warp_id != t % warps_per_block) continue;

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

            // ========== PHASE 1: Decompress suppkey + partkey ==========
            uint32_t suppkey[8], partkey[8];

            // Decompress suppkey
            {
                int local_mv_idx = (meta.suppkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
                int64_t mv_bit_base = meta.suppkey.partition_bit_base +
                    static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.suppkey.delta_bits;
                prefetchExtract8FOR(supp_deltas, mv_bit_base, lane_id,
                    meta.suppkey.delta_bits, supp_base, suppkey);
            }

            // Decompress partkey (overlap with supplier probe latency)
            {
                int local_mv_idx = (meta.partkey.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
                int64_t mv_bit_base = meta.partkey.partition_bit_base +
                    static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.partkey.delta_bits;
                prefetchExtract8FOR(part_deltas, mv_bit_base, lane_id,
                    meta.partkey.delta_bits, part_base, partkey);
            }

            // ========== PHASE 2: Batch probe supplier + part HTs ==========
            // Use prefetch-based probing for both tables
            uint32_t brand[8];

            // Probe supplier HT (existence check only)
            prefetchProbe8Exists(ht_s_keys, ht_s_size, suppkey, flags);

            any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
            if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

            // Probe part HT (get brand)
            prefetchProbe8Get(ht_p_keys, ht_p_values, ht_p_size, partkey, flags, brand);

            any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
            if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

            // ========== PHASE 3: Conditional decompress orderdate + revenue ==========
            uint32_t orderdate[8], revenue[8];

            // Decompress orderdate
            {
                int local_mv_idx = (meta.orderdate.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
                int64_t mv_bit_base = meta.orderdate.partition_bit_base +
                    static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;
                prefetchExtract8FORConditional(od_deltas, mv_bit_base, lane_id,
                    meta.orderdate.delta_bits, od_base, flags, orderdate);
            }

            // Decompress revenue
            {
                int local_mv_idx = (meta.revenue.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
                int64_t mv_bit_base = meta.revenue.partition_bit_base +
                    static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.revenue.delta_bits;
                prefetchExtract8FORConditional(rev_deltas, mv_bit_base, lane_id,
                    meta.revenue.delta_bits, rev_base, flags, revenue);
            }

            // ========== PHASE 4: Probe date HT from shared memory (FAST!) ==========
            uint32_t year[8];
            probeDateHTPacked8(ht_date_packed, orderdate, flags, year);

            any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
            if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

            // ========== PHASE 5: Aggregate by (year, brand) ==========
            #pragma unroll
            for (int v = 0; v < 8; v++) {
                if (flags[v]) {
                    int year_idx = year[v] - 1992;
                    if (year_idx >= 0 && year_idx < AGG_NUM_YEARS && brand[v] < AGG_NUM_BRANDS) {
                        int agg_idx = year_idx * AGG_NUM_BRANDS + brand[v];
                        atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[v]));
                    }
                }
            }
        }
    }
}

// ============================================================================
// Host Code
// ============================================================================

void runQ21FusedV6(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Build tile metadata
    float metadata_ms;
    cudaEventRecord(start);

    auto tile_meta = TileMetadataBuilder<uint32_t>::buildQ2xMetadata(
        data.lo_suppkey,
        data.lo_partkey,
        data.lo_orderdate,
        data.lo_revenue,
        total_rows);

    Q2xTileMetadataGPU gpu_meta;
    gpu_meta.allocate(tile_meta.size());
    gpu_meta.upload(tile_meta);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metadata_ms, start, stop);

    // Build hash tables (supplier and part only - date goes to shared memory)
    float ht_ms;
    cudaEventRecord(start);

    // Part hash table (filtered by category, partkey -> brand1)
    HashTable ht_part;
    ht_part.allocate(P_LEN, true);
    int block_size = 256;
    int grid_p = (P_LEN + block_size - 1) / block_size;
    build_part_ht_filtered_kernel<<<grid_p, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        P_CATEGORY_MFGR12,
        ht_part.d_keys, ht_part.d_values, ht_part.size);

    // Supplier hash table (filtered by region, keys only)
    HashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    int grid_s = (S_LEN + block_size - 1) / block_size;
    build_supplier_ht_region_kernel<<<grid_s, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN,
        S_REGION_AMERICA,
        ht_supplier.d_keys, ht_supplier.size);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ht_ms, start, stop);

    // Allocate aggregation array
    int agg_size = AGG_NUM_YEARS * AGG_NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;

    // V6 launch config: 4 warps per block, each warp handles 1 tile
    int warps_per_block = V6_BLOCK_SIZE / WARP_SZ;
    int blocks_needed = (num_tiles + warps_per_block - 1) / warps_per_block;

    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;
    std::cout << "V6 config: " << blocks_needed << " blocks × " << V6_BLOCK_SIZE << " threads" << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
        q21_fused_v6_kernel<<<blocks_needed, V6_BLOCK_SIZE>>>(
            data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas,
            gpu_meta.d_tiles,
            num_tiles,
            total_rows,
            ht_supplier.d_keys, ht_supplier.size,
            ht_part.d_keys, ht_part.d_values, ht_part.size,
            data.d_d_datekey, data.d_d_year, D_LEN,
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
    q21_fused_v6_kernel<<<blocks_needed, V6_BLOCK_SIZE>>>(
        data.lo_suppkey.d_interleaved_deltas,
        data.lo_partkey.d_interleaved_deltas,
        data.lo_orderdate.d_interleaved_deltas,
        data.lo_revenue.d_interleaved_deltas,
        gpu_meta.d_tiles,
        num_tiles,
        total_rows,
        ht_supplier.d_keys, ht_supplier.size,
        ht_part.d_keys, ht_part.d_values, ht_part.size,
        data.d_d_datekey, data.d_d_year, D_LEN,
        d_agg_revenue);
    cudaDeviceSynchronize();

    // Collect results
    std::vector<unsigned long long> h_agg_revenue(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 V6 Verification ===" << std::endl;
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
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V6) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

        cudaEventRecord(start);
        q21_fused_v6_kernel<<<blocks_needed, V6_BLOCK_SIZE>>>(
            data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas,
            gpu_meta.d_tiles,
            num_tiles,
            total_rows,
            ht_supplier.d_keys, ht_supplier.size,
            ht_part.d_keys, ht_part.d_values, ht_part.size,
            data.d_d_datekey, data.d_d_year, D_LEN,
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
    std::cout << "Metadata build: " << metadata_ms << " ms (one-time)" << std::endl;
    std::cout << "Hash table build: " << ht_ms << " ms (one-time)" << std::endl;
    std::cout << "Average kernel (V6): " << avg_time << " ms" << std::endl;
    std::cout << "Target: < 2.0 ms (vs V5: 4.38 ms)" << std::endl;

    cudaFree(d_agg_revenue);
    gpu_meta.free();
    ht_part.free();
    ht_supplier.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q2.1 - V6 Hash Prefetch + Shared Memory Date HT" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Optimizations:" << std::endl;
    std::cout << "  1. Date HT in shared memory (20KB)" << std::endl;
    std::cout << "  2. Batch prefetch for supplier/part probes" << std::endl;
    std::cout << "  3. Interleaved decompress + probe" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ21FusedV6(data);

    data.free();
    return 0;
}
