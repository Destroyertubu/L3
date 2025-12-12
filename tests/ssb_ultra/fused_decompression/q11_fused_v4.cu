/**
 * @file q11_fused_v4.cu
 * @brief SSB Q1.1 with Vertical-Compatible Tile Layout
 *
 * Key innovations:
 * 1. Vertical tile layout: 1024 values, offset = tile_idx * bitwidth * 32
 * 2. Reuse Vertical pre-generated unpack functions - NO division/modulo
 * 3. Register-based intermediate storage (32 values per thread)
 * 4. Simple tile offset calculation: O(1) instead of O(n) divisions
 *
 * Target: < 0.7 ms (Vertical: 0.544 ms)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

// Include Vertical generated unpack functions
#include "fls_gen/unpack/unpack_fused.cuh"

// Include tile encoder
#include "Vertical_tile_encoder.cuh"

// SSB data loading
#include "ssb_data_loader.hpp"

using namespace Vertical_tile;

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE_Q11 = 32;
constexpr int IPT = 32;  // Items per thread

// ============================================================================
// Q1.1 Query Metadata
// ============================================================================

struct Q11VerticalQuery {
    // Per-column metadata
    uint8_t od_bw;
    uint32_t od_base;

    uint8_t disc_bw;
    uint32_t disc_base;

    uint8_t qty_bw;
    uint32_t qty_base;

    uint8_t price_bw;
    uint32_t price_base;

    // Common
    int32_t num_tiles;
    int32_t num_elements;
};

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__
void initFlags(int (&flags)[IPT], int valid_count) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        flags[i] = (threadIdx.x + i * WARP_SIZE_Q11 < valid_count) ? 1 : 0;
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
void filterRange(const uint32_t (&vals)[IPT], uint32_t lo, uint32_t hi, int (&flags)[IPT]) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            flags[i] = (vals[i] >= lo && vals[i] <= hi);
        }
    }
}

__device__ __forceinline__
void filterLT(const uint32_t (&vals)[IPT], uint32_t threshold, int (&flags)[IPT]) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            flags[i] = (vals[i] < threshold);
        }
    }
}

__device__ __forceinline__
void addBase(uint32_t (&vals)[IPT], uint32_t base) {
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        vals[i] += base;
    }
}

// ============================================================================
// Main Fused Kernel - Vertical Style
// ============================================================================

__global__ void q11_Vertical_fused(
    const uint32_t* __restrict__ enc_orderdate,
    const uint32_t* __restrict__ enc_discount,
    const uint32_t* __restrict__ enc_quantity,
    const uint32_t* __restrict__ enc_extendedprice,
    Q11VerticalQuery query,
    unsigned long long* __restrict__ revenue)
{
    int tile_idx = blockIdx.x;
    if (tile_idx >= query.num_tiles) return;

    // Calculate valid elements in this tile
    int tile_start = tile_idx * TILE_SIZE;
    int valid_count = min(TILE_SIZE, query.num_elements - tile_start);

    // Per-thread arrays (registers)
    uint32_t items[IPT];
    int flags[IPT];

    initFlags(flags, valid_count);

    // ============ Column 1: Orderdate (filter: 1993) ============
    // Vertical tile offset: tile_idx * bitwidth * 32 (words)
    {
        const uint32_t* tile_ptr = enc_orderdate + tile_idx * query.od_bw * 32;
        generated::unpack::cuda::fused::unpack(tile_ptr, items, query.od_bw);
        addBase(items, query.od_base);
    }

    // Filter: lo_orderdate >= 19930101 AND lo_orderdate <= 19931231
    filterRange(items, 19930101u, 19931231u, flags);

    // Early exit if no matches in this tile
    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // ============ Column 2: Discount (filter: 1-3) ============
    uint32_t discount[IPT];
    {
        const uint32_t* tile_ptr = enc_discount + tile_idx * query.disc_bw * 32;
        generated::unpack::cuda::fused::unpack(tile_ptr, discount, query.disc_bw);
        addBase(discount, query.disc_base);
    }

    // Filter: lo_discount >= 1 AND lo_discount <= 3
    filterRange(discount, 1u, 3u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // ============ Column 3: Quantity (filter: < 25) ============
    uint32_t quantity[IPT];
    {
        const uint32_t* tile_ptr = enc_quantity + tile_idx * query.qty_bw * 32;
        generated::unpack::cuda::fused::unpack(tile_ptr, quantity, query.qty_bw);
        addBase(quantity, query.qty_base);
    }

    // Filter: lo_quantity < 25
    filterLT(quantity, 25u, flags);

    if (__ballot_sync(0xFFFFFFFF, anyFlagSet(flags)) == 0) return;

    // ============ Column 4: Extendedprice ============
    uint32_t extendedprice[IPT];
    {
        const uint32_t* tile_ptr = enc_extendedprice + tile_idx * query.price_bw * 32;
        generated::unpack::cuda::fused::unpack(tile_ptr, extendedprice, query.price_bw);
        addBase(extendedprice, query.price_base);
    }

    // ============ Aggregation ============
    unsigned long long local_sum = 0;
    #pragma unroll
    for (int i = 0; i < IPT; i++) {
        if (flags[i]) {
            local_sum += static_cast<unsigned long long>(extendedprice[i]) *
                        static_cast<unsigned long long>(discount[i]);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (threadIdx.x == 0 && local_sum > 0) {
        atomicAdd(revenue, local_sum);
    }
}

// ============================================================================
// SSB Data in Vertical Tile Format
// ============================================================================

struct SSBVerticalTileData {
    // Device pointers to packed columns
    uint32_t* d_enc_orderdate;
    uint32_t* d_enc_discount;
    uint32_t* d_enc_quantity;
    uint32_t* d_enc_extendedprice;

    // Column metadata
    ColumnMeta meta_orderdate;
    ColumnMeta meta_discount;
    ColumnMeta meta_quantity;
    ColumnMeta meta_extendedprice;

    // Host packed data (for upload)
    std::vector<uint32_t> h_enc_orderdate;
    std::vector<uint32_t> h_enc_discount;
    std::vector<uint32_t> h_enc_quantity;
    std::vector<uint32_t> h_enc_extendedprice;

    void loadAndPack(const std::string& data_dir) {
        // Load raw SSB data
        std::cout << "Loading SSB data from: " << data_dir << std::endl;

        auto lo_orderdate = ssb::loadColumnFromFile<uint32_t>(data_dir + "/" + ssb::LO_ORDERDATE, ssb::LO_LEN);
        auto lo_discount = ssb::loadColumnFromFile<uint32_t>(data_dir + "/" + ssb::LO_DISCOUNT, ssb::LO_LEN);
        auto lo_quantity = ssb::loadColumnFromFile<uint32_t>(data_dir + "/" + ssb::LO_QUANTITY, ssb::LO_LEN);
        auto lo_extendedprice = ssb::loadColumnFromFile<uint32_t>(data_dir + "/" + ssb::LO_EXTENDEDPRICE, ssb::LO_LEN);

        int num_elements = lo_orderdate.size();
        std::cout << "Loaded " << num_elements << " rows" << std::endl;

        // Pack each column in Vertical tile format
        std::cout << "Packing columns in Vertical tile format..." << std::endl;

        // Orderdate
        h_enc_orderdate.resize(calcPackedSize(num_elements, 32)); // Max size
        encodeColumn(lo_orderdate.data(), num_elements, h_enc_orderdate.data(), meta_orderdate);
        h_enc_orderdate.resize(meta_orderdate.packed_words);
        std::cout << "  orderdate: " << (int)meta_orderdate.bitwidth << "-bit, base="
                  << meta_orderdate.base_value << std::endl;

        // Discount
        h_enc_discount.resize(calcPackedSize(num_elements, 32));
        encodeColumn(lo_discount.data(), num_elements, h_enc_discount.data(), meta_discount);
        h_enc_discount.resize(meta_discount.packed_words);
        std::cout << "  discount:  " << (int)meta_discount.bitwidth << "-bit, base="
                  << meta_discount.base_value << std::endl;

        // Quantity
        h_enc_quantity.resize(calcPackedSize(num_elements, 32));
        encodeColumn(lo_quantity.data(), num_elements, h_enc_quantity.data(), meta_quantity);
        h_enc_quantity.resize(meta_quantity.packed_words);
        std::cout << "  quantity:  " << (int)meta_quantity.bitwidth << "-bit, base="
                  << meta_quantity.base_value << std::endl;

        // Extendedprice
        h_enc_extendedprice.resize(calcPackedSize(num_elements, 32));
        encodeColumn(lo_extendedprice.data(), num_elements, h_enc_extendedprice.data(), meta_extendedprice);
        h_enc_extendedprice.resize(meta_extendedprice.packed_words);
        std::cout << "  extprice:  " << (int)meta_extendedprice.bitwidth << "-bit, base="
                  << meta_extendedprice.base_value << std::endl;

        // Allocate and upload to GPU
        std::cout << "Uploading to GPU..." << std::endl;

        cudaMalloc(&d_enc_orderdate, h_enc_orderdate.size() * sizeof(uint32_t));
        cudaMemcpy(d_enc_orderdate, h_enc_orderdate.data(),
                   h_enc_orderdate.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaMalloc(&d_enc_discount, h_enc_discount.size() * sizeof(uint32_t));
        cudaMemcpy(d_enc_discount, h_enc_discount.data(),
                   h_enc_discount.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaMalloc(&d_enc_quantity, h_enc_quantity.size() * sizeof(uint32_t));
        cudaMemcpy(d_enc_quantity, h_enc_quantity.data(),
                   h_enc_quantity.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaMalloc(&d_enc_extendedprice, h_enc_extendedprice.size() * sizeof(uint32_t));
        cudaMemcpy(d_enc_extendedprice, h_enc_extendedprice.data(),
                   h_enc_extendedprice.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

        std::cout << "Data ready on GPU" << std::endl;
    }

    Q11VerticalQuery buildQuery() const {
        Q11VerticalQuery q;
        q.od_bw = meta_orderdate.bitwidth;
        q.od_base = meta_orderdate.base_value;
        q.disc_bw = meta_discount.bitwidth;
        q.disc_base = meta_discount.base_value;
        q.qty_bw = meta_quantity.bitwidth;
        q.qty_base = meta_quantity.base_value;
        q.price_bw = meta_extendedprice.bitwidth;
        q.price_base = meta_extendedprice.base_value;
        q.num_tiles = meta_orderdate.num_tiles;
        q.num_elements = meta_orderdate.num_elements;
        return q;
    }

    void free() {
        if (d_enc_orderdate) cudaFree(d_enc_orderdate);
        if (d_enc_discount) cudaFree(d_enc_discount);
        if (d_enc_quantity) cudaFree(d_enc_quantity);
        if (d_enc_extendedprice) cudaFree(d_enc_extendedprice);
        d_enc_orderdate = d_enc_discount = d_enc_quantity = d_enc_extendedprice = nullptr;
    }
};

// ============================================================================
// Benchmark
// ============================================================================

void runBenchmark(SSBVerticalTileData& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Q11VerticalQuery query = data.buildQuery();

    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    int num_tiles = query.num_tiles;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q11_Vertical_fused<<<num_tiles, WARP_SIZE_Q11>>>(
            data.d_enc_orderdate,
            data.d_enc_discount,
            data.d_enc_quantity,
            data.d_enc_extendedprice,
            query,
            d_revenue);
    }
    cudaDeviceSynchronize();

    // Verify correctness
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q11_Vertical_fused<<<num_tiles, WARP_SIZE_Q11>>>(
        data.d_enc_orderdate,
        data.d_enc_discount,
        data.d_enc_quantity,
        data.d_enc_extendedprice,
        query,
        d_revenue);
    cudaDeviceSynchronize();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    const unsigned long long EXPECTED = 40602899324ULL;
    std::cout << "\n=== Q1.1 Verification ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Expected: " << EXPECTED << std::endl;
    if (h_revenue == EXPECTED) {
        std::cout << "Status: PASSED" << std::endl;
    } else {
        std::cout << "Status: FAILED" << std::endl;
    }

    // Benchmark runs
    std::cout << "\n=== Benchmark Runs ===" << std::endl;

    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));

        cudaEventRecord(start);
        q11_Vertical_fused<<<num_tiles, WARP_SIZE_Q11>>>(
            data.d_enc_orderdate,
            data.d_enc_discount,
            data.d_enc_quantity,
            data.d_enc_extendedprice,
            query,
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
    std::cout << "Average kernel time: " << avg_time << " ms" << std::endl;
    std::cout << "Tiles: " << num_tiles << std::endl;
    std::cout << "Elements: " << query.num_elements << std::endl;

    // Comparison
    std::cout << "\n=== Performance Comparison ===" << std::endl;
    std::cout << "Vertical GPU:   0.544 ms (reference)" << std::endl;
    std::cout << "L3 Fused V3:     1.35 ms (previous)" << std::endl;
    std::cout << "L3 Fused V4:     " << avg_time << " ms (this)" << std::endl;

    float speedup_vs_v3 = 1.35f / avg_time;
    float gap_to_fls = avg_time / 0.544f;
    std::cout << "\nSpeedup vs V3:   " << speedup_vs_v3 << "x" << std::endl;
    std::cout << "Gap to Vertical: " << gap_to_fls << "x" << std::endl;

    if (avg_time < 0.7f) {
        std::cout << "\nTARGET ACHIEVED: < 0.7 ms!" << std::endl;
    } else {
        std::cout << "\nTarget: < 0.7 ms (need more optimization)" << std::endl;
    }

    cudaFree(d_revenue);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q1.1 - Vertical-Compatible Fused Kernel V4" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Target: < 0.7 ms (Vertical: 0.544 ms)" << std::endl;
    std::cout << std::endl;

    // Load and pack data in Vertical tile format
    SSBVerticalTileData data;
    data.loadAndPack(data_dir);

    // Run benchmark
    runBenchmark(data);

    data.free();
    return 0;
}
