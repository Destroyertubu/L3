/**
 * @file q11_fused_v5.cu
 * @brief SSB Q1.1 with L3 Native Format + Prefetch Strategy
 *
 * Key innovations (learning from L3 standalone decoder):
 * 1. Use L3 interleaved format directly (no format conversion)
 * 2. Prefetch all needed words to registers BEFORE extraction
 * 3. Extract from registers (zero latency)
 * 4. One warp per tile, each thread handles 32 values
 *
 * L3 Interleaved Format (256-value mini-vectors):
 *   Lane L has values at indices: L, L+32, L+64, L+96, L+128, L+160, L+192, L+224
 *   Bits are stored: Lane 0's 8 values, Lane 1's 8 values, ..., Lane 31's 8 values
 *
 * Target: < 0.7 ms
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

using namespace l3_fused;

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SZ = 32;
constexpr int VALUES_PER_LANE = 8;  // L3 interleaved: 8 values per lane in mini-vector
constexpr int MINI_VEC_SIZE = 256;

// ============================================================================
// Prefetch-based Delta Extraction (L3 Native Format)
// ============================================================================

/**
 * Prefetch all words needed for this lane's 8 values from a mini-vector,
 * then extract from registers. This matches L3 standalone decoder's strategy.
 *
 * @param data        L3 interleaved delta array
 * @param mv_bit_base Bit offset to start of this mini-vector in partition
 * @param lane_id     Thread's lane ID (0-31)
 * @param delta_bits  Bit width for deltas
 * @param base        FOR base value
 * @param output      Output array (8 values)
 */
__device__ __forceinline__
void prefetchExtract8FOR(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane_id,
    int delta_bits,
    uint32_t base,
    uint32_t (&output)[8])
{
    if (delta_bits == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            output[i] = base;
        }
        return;
    }

    // Lane's bit position within mini-vector
    // Bits layout: Lane0[8*bw], Lane1[8*bw], ..., Lane31[8*bw]
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_LANE * delta_bits;

    // Calculate words needed (same as L3 standalone decoder)
    int64_t lane_word_start = lane_bit_start >> 5;
    int bits_per_lane = VALUES_PER_LANE * delta_bits;
    int words_needed = (bits_per_lane + 31 + 32) / 32;  // +32 for alignment padding
    words_needed = min(words_needed, 20);

    // Prefetch to registers (KEY OPTIMIZATION!)
    uint32_t lane_words[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        lane_words[i] = (i < words_needed) ? __ldg(&data[lane_word_start + i]) : 0;
    }

    // Extract from registers (zero latency)
    int local_bit = lane_bit_start & 31;
    uint64_t mask = (1ULL << delta_bits) - 1;

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint32_t delta = (combined >> bit_in_word) & mask;

        output[v] = base + delta;
        local_bit += delta_bits;
    }
}

/**
 * Conditional prefetch - only extract values where flags are set.
 */
__device__ __forceinline__
void prefetchExtract8FORConditional(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane_id,
    int delta_bits,
    uint32_t base,
    const int (&flags)[8],
    uint32_t (&output)[8])
{
    // Check if any flag is set (branchless OR)
    int any_set = flags[0] | flags[1] | flags[2] | flags[3] |
                  flags[4] | flags[5] | flags[6] | flags[7];
    if (!any_set) return;

    if (delta_bits == 0) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (flags[i]) output[i] = base;
        }
        return;
    }

    // Lane's bit position within mini-vector
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * VALUES_PER_LANE * delta_bits;

    // Prefetch to registers (always do for uniform execution)
    int64_t lane_word_start = lane_bit_start >> 5;
    int bits_per_lane = VALUES_PER_LANE * delta_bits;
    int words_needed = (bits_per_lane + 31 + 32) / 32;
    words_needed = min(words_needed, 20);

    uint32_t lane_words[20];
    #pragma unroll
    for (int i = 0; i < 20; i++) {
        lane_words[i] = (i < words_needed) ? __ldg(&data[lane_word_start + i]) : 0;
    }

    // Extract only where needed
    int local_bit = lane_bit_start & 31;
    uint64_t mask = (1ULL << delta_bits) - 1;

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int word_idx = local_bit >> 5;
            int bit_in_word = local_bit & 31;

            uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                               lane_words[word_idx];
            output[v] = base + ((combined >> bit_in_word) & mask);
        }
        local_bit += delta_bits;
    }
}

// ============================================================================
// Main Fused Kernel V5 - L3 Native with Prefetch
// ============================================================================

/**
 * Process one 1024-element tile per warp.
 * Each tile has 4 mini-vectors (256 values each).
 * Each thread handles 8 values per mini-vector × 4 mini-vectors = 32 values.
 */
__global__ void q11_fused_v5_kernel(
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

    // Load tile metadata (once per tile, cached in registers)
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

        // Initialize validity flags for this lane's 8 values
        int flags[8];
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            int idx_in_mv = v * WARP_SZ + lane_id;
            flags[v] = (idx_in_mv < mv_valid) ? 1 : 0;
        }

        // Check if any thread has valid data
        int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                        flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 1: Orderdate (filter: 1993) ==========
        uint32_t orderdate[8];
        {
            // Mini-vector bit offset within partition
            int local_mv_idx = (meta.orderdate.local_offset + mv_start_in_tile) / MINI_VEC_SIZE;
            int64_t mv_bit_base = meta.orderdate.partition_bit_base +
                static_cast<int64_t>(local_mv_idx) * MINI_VEC_SIZE * meta.orderdate.delta_bits;

            prefetchExtract8FOR(od_deltas, mv_bit_base, lane_id,
                meta.orderdate.delta_bits, od_base, orderdate);
        }

        // Filter: 19930101 <= orderdate <= 19931231
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            if (flags[v]) {
                flags[v] = (orderdate[v] >= 19930101u && orderdate[v] <= 19931231u);
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 2: Discount (filter: 1-3) ==========
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
                flags[v] = (discount[v] >= 1u && discount[v] <= 3u);
            }
        }

        any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
        if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) continue;

        // ========== Column 3: Quantity (filter: < 25) ==========
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
                flags[v] = (quantity[v] < 25u);
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

void runQ11FusedV5(ssb::SSBDataCompressedVertical& data) {
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

    // Execute kernel
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;

    std::cout << "\nTiles: " << num_tiles << ", Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q11_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
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

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Verify correctness
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q11_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
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

    const unsigned long long EXPECTED = 40602899324ULL;
    std::cout << "\n=== Q1.1 Verification ===" << std::endl;
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
        q11_fused_v5_kernel<<<num_tiles, WARP_SZ>>>(
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

    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Vertical GPU:   0.544 ms" << std::endl;
    std::cout << "L3 Fused V3:     1.35 ms" << std::endl;
    std::cout << "L3 Fused V5:     " << avg_time << " ms" << std::endl;

    float speedup_vs_v3 = 1.35f / avg_time;
    std::cout << "Speedup vs V3:   " << speedup_vs_v3 << "x" << std::endl;

    if (avg_time < 0.7f) {
        std::cout << "\nTARGET ACHIEVED: < 0.7 ms!" << std::endl;
    }

    cudaFree(d_revenue);
    gpu_meta.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q1.1 - L3 Native Format + Prefetch Strategy (V5)" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key: Prefetch to registers, extract with zero latency" << std::endl;
    std::cout << "Target: < 0.7 ms (Vertical: 0.544 ms)" << std::endl;
    std::cout << std::endl;

    ssb::SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ11FusedV5(data);

    data.free();
    return 0;
}
