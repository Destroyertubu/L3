/**
 * @file q12_fused_v5.cu
 * @brief SSB Q1.2 with L3 Native Format + Prefetch Strategy
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey AND d_yearmonthnum = 199401
 *     AND lo_discount BETWEEN 4 AND 6 AND lo_quantity BETWEEN 26 AND 35;
 *
 * Filter differences from Q1.1:
 *   - Year: January 1994 (19940101-19940131)
 *   - Discount: 4-6
 *   - Quantity: 26-35
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

using namespace l3_fused;
using namespace l3_v5;

// ============================================================================
// Main Fused Kernel V5 - Q1.2
// ============================================================================

__global__ void q12_fused_v5_kernel(
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ disc_deltas,
    const uint32_t* __restrict__ qty_deltas,
    const uint32_t* __restrict__ price_deltas,
    const Q11TileMetadata* __restrict__ tile_meta,
    int num_tiles,
    int total_rows,
    unsigned long long* __restrict__ revenue)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= num_tiles) return;

    int lane_id = threadIdx.x;
    int tile_start = tile_idx * TILE_SIZE;
    int tile_size = min(TILE_SIZE, total_rows - tile_start);

    // Load tile metadata
    Q11TileMetadata meta = tile_meta[tile_idx];

    // Get FOR base values
    uint32_t od_base = static_cast<uint32_t>(__double2ll_rn(meta.orderdate.params[0]));
    uint32_t disc_base = static_cast<uint32_t>(__double2ll_rn(meta.discount.params[0]));
    uint32_t qty_base = static_cast<uint32_t>(__double2ll_rn(meta.quantity.params[0]));
    uint32_t price_base = static_cast<uint32_t>(__double2ll_rn(meta.extendedprice.params[0]));

    unsigned long long local_sum = 0;

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

        // ========== Column 1: Orderdate (filter: Jan 1994) ==========
        uint32_t orderdate[8];
        {
            int local_mv_idx = (meta.orderdate.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.orderdate.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;

            prefetchExtract8FOR(od_deltas, mv_bit_base, lane_id,
                meta.orderdate.delta_bits, od_base, orderdate);
        }

        // Filter: January 1994 (19940101 <= orderdate <= 19940131)
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                flags[v] = (orderdate[v] >= 19940101u && orderdate[v] <= 19940131u);
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 2: Discount (filter: 4-6) ==========
        uint32_t discount[8];
        {
            int local_mv_idx = (meta.discount.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.discount.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.discount.delta_bits;

            prefetchExtract8FORConditional(disc_deltas, mv_bit_base, lane_id,
                meta.discount.delta_bits, disc_base, flags, discount);
        }

        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                flags[v] = (discount[v] >= 4u && discount[v] <= 6u);
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 3: Quantity (filter: 26-35) ==========
        uint32_t quantity[8];
        {
            int local_mv_idx = (meta.quantity.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.quantity.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.quantity.delta_bits;

            prefetchExtract8FORConditional(qty_deltas, mv_bit_base, lane_id,
                meta.quantity.delta_bits, qty_base, flags, quantity);
        }

        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                flags[v] = (quantity[v] >= 26u && quantity[v] <= 35u);
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 4: Extendedprice ==========
        uint32_t extendedprice[8];
        {
            int local_mv_idx = (meta.extendedprice.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.extendedprice.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.extendedprice.delta_bits;

            prefetchExtract8FORConditional(price_deltas, mv_bit_base, lane_id,
                meta.extendedprice.delta_bits, price_base, flags, extendedprice);
        }

        // ========== Aggregate ==========
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                local_sum += static_cast<unsigned long long>(extendedprice[v]) *
                            static_cast<unsigned long long>(discount[v]);
            }
        }
    }

    // Warp reduction
    local_sum = warpReduceSum(local_sum);

    if (lane_id == 0 && local_sum > 0) {
        atomicAdd(revenue, local_sum);
    }
}

// ============================================================================
// Host Code
// ============================================================================

void runQ12FusedV5(ssb::SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Build tile metadata
    float metadata_ms;
    cudaEventRecord(start);

    auto tile_meta = TileMetadataBuilder<uint32_t>::buildQ11Metadata(
        data.lo_orderdate,
        data.lo_discount,
        data.lo_quantity,
        data.lo_extendedprice,
        total_rows);

    Q11TileMetadataGPU gpu_meta;
    gpu_meta.allocate(tile_meta.size());
    gpu_meta.upload(tile_meta);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&metadata_ms, start, stop);

    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;

    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q12_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_orderdate.d_interleaved_deltas,
            data.lo_discount.d_interleaved_deltas,
            data.lo_quantity.d_interleaved_deltas,
            data.lo_extendedprice.d_interleaved_deltas,
            gpu_meta.d_tiles,
            num_tiles,
            total_rows,
            d_revenue);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Verify correctness
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q12_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
        data.lo_orderdate.d_interleaved_deltas,
        data.lo_discount.d_interleaved_deltas,
        data.lo_quantity.d_interleaved_deltas,
        data.lo_extendedprice.d_interleaved_deltas,
        gpu_meta.d_tiles,
        num_tiles,
        total_rows,
        d_revenue);
    cudaDeviceSynchronize();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    const unsigned long long EXPECTED = 8571565529ULL;
    std::cout << "\n=== Q1.2 Verification ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Expected: " << EXPECTED << std::endl;
    std::cout << "Status: " << (h_revenue == EXPECTED ? "PASSED" : "FAILED") << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));

        cudaEventRecord(start);
        q12_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
            data.lo_orderdate.d_interleaved_deltas,
            data.lo_discount.d_interleaved_deltas,
            data.lo_quantity.d_interleaved_deltas,
            data.lo_extendedprice.d_interleaved_deltas,
            gpu_meta.d_tiles,
            num_tiles,
            total_rows,
            d_revenue);
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
    std::cout << "Average kernel: " << avg_time << " ms" << std::endl;

    cudaFree(d_revenue);
    gpu_meta.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q1.2 - L3 Native Format + Prefetch Strategy (V5)" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Filters: Jan 1994, discount 4-6, quantity 26-35" << std::endl;
    std::cout << std::endl;

    ssb::SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ12FusedV5(data);

    data.free();
    return 0;
}
