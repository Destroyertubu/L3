/**
 * @file q11_fused_v3.cu
 * @brief SSB Q1.1 with Template-Specialized Unpack
 *
 * Key optimizations:
 * 1. Template-specialized unpack functions (like Vertical)
 * 2. Compile-time bit width constants for SSB columns
 * 3. Pre-computed lane_in_mv to reduce per-value computation
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "tile_metadata.cuh"
#include "unpack_specialized.cuh"
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

using namespace l3_fused;

// ============================================================================
// Kernel Configuration
// ============================================================================

constexpr int BLOCK_THREADS = 32;
constexpr int IPT = 32;

// ============================================================================
// Selection Helpers
// ============================================================================

__device__ __forceinline__
void initFlags(int (&flags)[IPT], int tile_size, int lane_id) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        flags[i] = (lane_id + i * WARP_SIZE < tile_size) ? 1 : 0;
    }
}

__device__ __forceinline__
uint32_t anyFlagSet(const int (&flags)[IPT]) {
    uint32_t mask = 0;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) mask |= (1u << i);
    }
    return mask;
}

__device__ __forceinline__
void filterRange(const uint32_t (&values)[IPT], uint32_t lo, uint32_t hi, int (&flags)[IPT]) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            flags[i] = (values[i] >= lo && values[i] <= hi);
        }
    }
}

__device__ __forceinline__
void filterLT(const uint32_t (&values)[IPT], uint32_t threshold, int (&flags)[IPT]) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            flags[i] = (values[i] < threshold);
        }
    }
}

// ============================================================================
// Q1.1 Fused Kernel V3 - Template Specialized
// ============================================================================

__global__ void q11_fused_kernel_v3(
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

    // Per-thread arrays
    uint32_t orderdate[IPT];
    uint32_t discount[IPT];
    uint32_t quantity[IPT];
    uint32_t extendedprice[IPT];
    int flags[IPT];

    initFlags(flags, tile_size, lane_id);

    // =========== Column 1: Orderdate (16-bit FOR) ===========
    {
        uint32_t base = static_cast<uint32_t>(__double2ll_rn(meta.orderdate.params[0]));
        unpackFORRuntime(od_deltas, meta.orderdate.partition_bit_base,
                         meta.orderdate.local_offset, lane_id,
                         meta.orderdate.delta_bits, base, orderdate);
    }
    filterRange(orderdate, 19930101u, 19931231u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 2: Discount (4-bit FOR) ===========
    {
        uint32_t base = static_cast<uint32_t>(__double2ll_rn(meta.discount.params[0]));
        unpackFORConditionalRuntime(disc_deltas, meta.discount.partition_bit_base,
                                     meta.discount.local_offset, lane_id,
                                     meta.discount.delta_bits, base, flags, discount);
    }
    filterRange(discount, 1u, 3u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 3: Quantity (6-bit FOR) ===========
    {
        uint32_t base = static_cast<uint32_t>(__double2ll_rn(meta.quantity.params[0]));
        unpackFORConditionalRuntime(qty_deltas, meta.quantity.partition_bit_base,
                                     meta.quantity.local_offset, lane_id,
                                     meta.quantity.delta_bits, base, flags, quantity);
    }
    filterLT(quantity, 25u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 4: Extendedprice (16-bit FOR) ===========
    {
        uint32_t base = static_cast<uint32_t>(__double2ll_rn(meta.extendedprice.params[0]));
        unpackFORConditionalRuntime(price_deltas, meta.extendedprice.partition_bit_base,
                                     meta.extendedprice.local_offset, lane_id,
                                     meta.extendedprice.delta_bits, base, flags, extendedprice);
    }

    // =========== Aggregation ===========
    unsigned long long local_sum = 0;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            local_sum += static_cast<unsigned long long>(extendedprice[i]) *
                        static_cast<unsigned long long>(discount[i]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane_id == 0 && local_sum > 0) {
        atomicAdd(revenue, local_sum);
    }
}

// ============================================================================
// Host Code
// ============================================================================

struct QueryTiming {
    float metadata_ms;
    float kernel_ms;
    float total_ms;
};

void runQ11FusedV3(ssb::SSBDataCompressedVertical& data, QueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Build tile metadata
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
    cudaEventElapsedTime(&timing.metadata_ms, start, stop);

    // Execute fused kernel
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q11_fused_kernel_v3<<<num_tiles, BLOCK_THREADS>>>(
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

    // Timed run
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    cudaEventRecord(start);

    q11_fused_kernel_v3<<<num_tiles, BLOCK_THREADS>>>(
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
    cudaEventElapsedTime(&timing.kernel_ms, start, stop);
    timing.total_ms = timing.metadata_ms + timing.kernel_ms;

    // Get result
    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q1.1 Results (FUSED V3) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Tile metadata build: " << timing.metadata_ms << " ms" << std::endl;
    std::cout << "  Fused kernel:        " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "  Total:               " << timing.total_ms << " ms" << std::endl;

    const unsigned long long EXPECTED = 40602899324ULL;
    if (h_revenue == EXPECTED) {
        std::cout << "  Verification:        PASSED" << std::endl;
    } else {
        std::cout << "  Verification:        FAILED (expected " << EXPECTED << ")" << std::endl;
    }

    cudaFree(d_revenue);
    gpu_meta.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.1 - FUSED V3 (Template Specialized) ===" << std::endl;
    std::cout << "Target: < 0.8 ms (Vertical: 0.544 ms)" << std::endl;
    std::cout << std::endl;

    ssb::SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ11FusedV3(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    float total_kernel_time = 0;
    const int RUNS = 5;
    for (int i = 0; i < RUNS; ++i) {
        QueryTiming t;
        runQ11FusedV3(data, t);
        total_kernel_time += t.kernel_ms;
        std::cout << "Run " << (i+1) << ": " << t.kernel_ms << " ms (kernel only)\n";
    }
    std::cout << "\nAverage kernel time: " << (total_kernel_time / RUNS) << " ms" << std::endl;

    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Vertical GPU:        0.544 ms" << std::endl;
    std::cout << "L3 Decompress-First:  1.27 ms" << std::endl;
    std::cout << "L3 Fused V1:          1.64 ms" << std::endl;
    std::cout << "L3 Fused V3:          " << (total_kernel_time / RUNS) << " ms" << std::endl;

    data.free();
    return 0;
}
