/**
 * @file q11_fused_optimized.cu
 * @brief SSB Q1.1 with TRUE Fused Decompression + Query
 *
 * Query: SUM(lo_extendedprice * lo_discount)
 *        WHERE lo_orderdate IN [19930101, 19931231]
 *          AND lo_discount IN [1, 3]
 *          AND lo_quantity < 25
 *
 * Architecture:
 * - 32 threads per block (single warp)
 * - 32 items per thread
 * - 1024 elements per tile (matches Vertical)
 * - Pre-computed tile metadata to avoid per-element partition lookup
 * - Warp ballot for early termination
 * - Warp shuffle for reduction
 *
 * Target: < 0.8 ms (vs Vertical 0.544 ms)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "tile_metadata.cuh"
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

using namespace l3_fused;

// ============================================================================
// Kernel Configuration
// ============================================================================

constexpr int BLOCK_THREADS = 32;  // Single warp
constexpr int IPT = 32;            // Items per thread
// TILE_SIZE = 1024 (defined in tile_metadata.cuh)

// ============================================================================
// Device Functions: Tile Decompression
// ============================================================================

/**
 * @brief Decompress a tile using pre-computed metadata (INTERLEAVED format)
 *        Simplified version using per-value offset computation.
 *
 * @param delta_array Compressed delta data
 * @param meta Column tile metadata
 * @param lane_id Thread index (0-31)
 * @param tile_start Global index of tile start
 * @param output Output array (32 values per thread)
 */
__device__ __forceinline__
void decompressTile(const uint32_t* __restrict__ delta_array,
                    const ColumnTileMeta& meta,
                    int lane_id,
                    int tile_start,
                    uint32_t (&output)[IPT])
{
    int delta_bits = meta.delta_bits;

    if (meta.is_FOR) {
        // Fast path: FOR model
        int64_t base = __double2ll_rn(meta.params[0]);

        if (delta_bits == 0) {
            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                output[i] = static_cast<uint32_t>(base);
            }
        } else {
            // FOR: base + unsigned delta
            // Thread L's values: local_offset + L, local_offset + L + 32, ...
            // Each i maps to a specific (mv, v) pair in interleaved format
            const int64_t bits_per_mv = static_cast<int64_t>(MV_SIZE) * delta_bits;
            const int64_t bits_per_lane = static_cast<int64_t>(MV_VALUES_PER_LANE) * delta_bits;

            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                // For output[i], we need value at local_offset + lane_id + i*32
                int local_idx = meta.local_offset + lane_id + i * WARP_SIZE;
                int mv = local_idx / MV_SIZE;
                int local_in_mv = local_idx % MV_SIZE;
                int lane_in_mv = local_in_mv % WARP_SIZE;
                int v_in_lane = local_in_mv / WARP_SIZE;

                int64_t bit_offset = meta.partition_bit_base +
                    static_cast<int64_t>(mv) * bits_per_mv +
                    static_cast<int64_t>(lane_in_mv) * bits_per_lane +
                    static_cast<int64_t>(v_in_lane) * delta_bits;

                uint64_t delta = extractBits64(delta_array, bit_offset, delta_bits);
                output[i] = static_cast<uint32_t>(base + delta);
            }
        }
    } else {
        // General path: Polynomial model with signed delta
        const int64_t bits_per_mv = static_cast<int64_t>(MV_SIZE) * delta_bits;
        const int64_t bits_per_lane = static_cast<int64_t>(MV_VALUES_PER_LANE) * delta_bits;

        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            int local_idx = meta.local_offset + lane_id + i * WARP_SIZE;
            int64_t predicted = computePoly(meta.model_type, meta.params, local_idx);

            if (delta_bits == 0) {
                output[i] = static_cast<uint32_t>(predicted);
            } else {
                int mv = local_idx / MV_SIZE;
                int local_in_mv = local_idx % MV_SIZE;
                int lane_in_mv = local_in_mv % WARP_SIZE;
                int v_in_lane = local_in_mv / WARP_SIZE;

                int64_t bit_offset = meta.partition_bit_base +
                    static_cast<int64_t>(mv) * bits_per_mv +
                    static_cast<int64_t>(lane_in_mv) * bits_per_lane +
                    static_cast<int64_t>(v_in_lane) * delta_bits;

                uint64_t extracted = extractBits64(delta_array, bit_offset, delta_bits);
                int64_t delta = signExtend64(extracted, delta_bits);
                output[i] = static_cast<uint32_t>(predicted + delta);
            }
        }
    }
}

/**
 * @brief Decompress only values that pass selection (INTERLEAVED format)
 */
__device__ __forceinline__
void decompressTileConditional(const uint32_t* __restrict__ delta_array,
                               const ColumnTileMeta& meta,
                               int lane_id,
                               int tile_start,
                               const int (&flags)[IPT],
                               uint32_t (&output)[IPT])
{
    int delta_bits = meta.delta_bits;

    if (meta.is_FOR) {
        int64_t base = __double2ll_rn(meta.params[0]);

        if (delta_bits == 0) {
            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                if (flags[i]) {
                    output[i] = static_cast<uint32_t>(base);
                }
            }
        } else {
            const int64_t bits_per_mv = static_cast<int64_t>(MV_SIZE) * delta_bits;
            const int64_t bits_per_lane = static_cast<int64_t>(MV_VALUES_PER_LANE) * delta_bits;

            #pragma unroll
            for (int i = 0; i < IPT; i++) {
                if (flags[i]) {
                    int local_idx = meta.local_offset + lane_id + i * WARP_SIZE;
                    int mv = local_idx / MV_SIZE;
                    int local_in_mv = local_idx % MV_SIZE;
                    int lane_in_mv = local_in_mv % WARP_SIZE;
                    int v_in_lane = local_in_mv / WARP_SIZE;

                    int64_t bit_offset = meta.partition_bit_base +
                        static_cast<int64_t>(mv) * bits_per_mv +
                        static_cast<int64_t>(lane_in_mv) * bits_per_lane +
                        static_cast<int64_t>(v_in_lane) * delta_bits;

                    uint64_t delta = extractBits64(delta_array, bit_offset, delta_bits);
                    output[i] = static_cast<uint32_t>(base + delta);
                }
            }
        }
    } else {
        const int64_t bits_per_mv = static_cast<int64_t>(MV_SIZE) * delta_bits;
        const int64_t bits_per_lane = static_cast<int64_t>(MV_VALUES_PER_LANE) * delta_bits;

        #pragma unroll
        for (int i = 0; i < IPT; i++) {
            if (flags[i]) {
                int local_idx = meta.local_offset + lane_id + i * WARP_SIZE;
                int64_t predicted = computePoly(meta.model_type, meta.params, local_idx);

                if (delta_bits == 0) {
                    output[i] = static_cast<uint32_t>(predicted);
                } else {
                    int mv = local_idx / MV_SIZE;
                    int local_in_mv = local_idx % MV_SIZE;
                    int lane_in_mv = local_in_mv % WARP_SIZE;
                    int v_in_lane = local_in_mv / WARP_SIZE;

                    int64_t bit_offset = meta.partition_bit_base +
                        static_cast<int64_t>(mv) * bits_per_mv +
                        static_cast<int64_t>(lane_in_mv) * bits_per_lane +
                        static_cast<int64_t>(v_in_lane) * delta_bits;

                    uint64_t extracted = extractBits64(delta_array, bit_offset, delta_bits);
                    int64_t delta = signExtend64(extracted, delta_bits);
                    output[i] = static_cast<uint32_t>(predicted + delta);
                }
            }
        }
    }
}

// ============================================================================
// Selection Flag Helpers
// ============================================================================

__device__ __forceinline__
void initFlags(int (&flags)[IPT], int tile_size, int lane_id) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        int idx = lane_id + i * WARP_SIZE;
        flags[i] = (idx < tile_size) ? 1 : 0;
    }
}

__device__ __forceinline__
bool anyFlagSet(const int (&flags)[IPT]) {
    bool any = false;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        any = any || flags[i];
    }
    return any;
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
// Q1.1 Fused Kernel
// ============================================================================

__global__ void q11_fused_kernel(
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

    // Load tile metadata (cached in L1)
    Q11TileMetadata meta = tile_meta[tile_idx];

    // Per-thread arrays
    uint32_t orderdate[IPT];
    uint32_t discount[IPT];
    uint32_t quantity[IPT];
    uint32_t extendedprice[IPT];
    int flags[IPT];

    // Initialize flags
    initFlags(flags, tile_size, lane_id);

    // =========== Column 1: Orderdate ===========
    decompressTile(od_deltas, meta.orderdate, lane_id, tile_start, orderdate);

    // Filter: 19930101 <= orderdate <= 19931231
    filterRange(orderdate, 19930101u, 19931231u, flags);

    // Early termination via warp ballot
    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 2: Discount ===========
    decompressTileConditional(disc_deltas, meta.discount, lane_id, tile_start, flags, discount);

    // Filter: 1 <= discount <= 3
    filterRange(discount, 1u, 3u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 3: Quantity ===========
    decompressTileConditional(qty_deltas, meta.quantity, lane_id, tile_start, flags, quantity);

    // Filter: quantity < 25
    filterLT(quantity, 25u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // =========== Column 4: Extendedprice ===========
    decompressTileConditional(price_deltas, meta.extendedprice, lane_id, tile_start, flags, extendedprice);

    // =========== Aggregation ===========
    unsigned long long local_sum = 0;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            local_sum += static_cast<unsigned long long>(extendedprice[i]) *
                        static_cast<unsigned long long>(discount[i]);
        }
    }

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Lane 0 does atomic add
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

void runQ11FusedOptimized(ssb::SSBDataCompressedVertical& data, QueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;

    // Step 1: Build tile metadata
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

    // Step 2: Execute fused kernel
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    int num_tiles = gpu_meta.num_tiles;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q11_fused_kernel<<<num_tiles, BLOCK_THREADS>>>(
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

    q11_fused_kernel<<<num_tiles, BLOCK_THREADS>>>(
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

    std::cout << "\n=== Q1.1 Results (FUSED OPTIMIZED) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Tile metadata build: " << timing.metadata_ms << " ms" << std::endl;
    std::cout << "  Fused kernel:        " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "  Total:               " << timing.total_ms << " ms" << std::endl;
    std::cout << "  Num tiles:           " << num_tiles << std::endl;

    // Verify correctness
    const unsigned long long EXPECTED = 40602899324ULL;
    if (h_revenue == EXPECTED) {
        std::cout << "  Verification:        PASSED" << std::endl;
    } else {
        std::cout << "  Verification:        FAILED (expected " << EXPECTED << ")" << std::endl;
    }

    // Cleanup
    cudaFree(d_revenue);
    gpu_meta.free();
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.1 - FUSED OPTIMIZED (32x32 Tile) ===" << std::endl;
    std::cout << "Target: < 0.8 ms (Vertical: 0.544 ms)" << std::endl;
    std::cout << std::endl;

    ssb::SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ11FusedOptimized(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    float total_kernel_time = 0;
    const int RUNS = 5;
    for (int i = 0; i < RUNS; ++i) {
        QueryTiming t;
        runQ11FusedOptimized(data, t);
        total_kernel_time += t.kernel_ms;
        std::cout << "Run " << (i+1) << ": " << t.kernel_ms << " ms (kernel only)\n";
    }
    std::cout << "\nAverage kernel time: " << (total_kernel_time / RUNS) << " ms" << std::endl;

    // Comparison
    std::cout << "\n=== Comparison ===" << std::endl;
    std::cout << "Vertical GPU:        0.544 ms" << std::endl;
    std::cout << "L3 Decompress-First:  1.27 ms" << std::endl;
    std::cout << "L3 Fused Optimized:   " << (total_kernel_time / RUNS) << " ms" << std::endl;

    data.free();
    return 0;
}
