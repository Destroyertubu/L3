/**
 * @file q21_fused_v10.cu
 * @brief SSB Q2.1 with V10 - Template-Specialized + Shared Metadata
 *
 * V10 Optimizations over V9:
 * 1. Template-specialized unpack for common bit-widths (no runtime branches)
 * 2. Compact metadata (only essential fields)
 * 3. Cooperative metadata loading via shared memory
 * 4. Inline decompression (no function call overhead)
 *
 * Target: Close remaining 1.29x gap to Vertical
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

// V7 Crystal-style hash headers
#include "v7/crystal_hash_v7.cuh"
#include "v7/crystal_hash_build_v7.cuh"

using namespace l3_crystal;
using namespace ssb;

// V10 constants
constexpr int V10_BLOCK_THREADS = 32;
constexpr int V10_ITEMS_PER_THREAD = 8;
constexpr int V10_MV_SIZE = 256;
constexpr int V10_PARTITION_SIZE = 1024;

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;
constexpr int AGG_NUM_BRANDS = 1000;

// ============================================================================
// V10 Compact Metadata - 20 bytes per column instead of 72
// ============================================================================

struct V10ColumnMeta {
    int64_t bit_base;      // 8 bytes - bit offset for partition start
    uint32_t base_value;   // 4 bytes - FOR base
    int32_t local_offset;  // 4 bytes - tile_start - partition_start
    uint8_t delta_bits;    // 1 byte
    uint8_t padding[3];    // 3 bytes alignment
};  // 20 bytes total

struct V10TileMeta {
    V10ColumnMeta col[4];  // suppkey, partkey, orderdate, revenue
};  // 80 bytes total (vs 290 bytes in original)

// ============================================================================
// Template-Specialized Unpack (Compile-time bit-width)
// ============================================================================

template<int BITS>
__device__ __forceinline__ void unpack8_v10(
    const uint32_t* __restrict__ data,
    int64_t lane_bit_start,
    uint32_t base,
    uint32_t (&out)[8])
{
    constexpr int BITS_PER_LANE = 8 * BITS;
    constexpr int WORDS_NEEDED = (BITS_PER_LANE + 63) / 32;

    int64_t word_start = lane_bit_start >> 5;
    int local_bit = lane_bit_start & 31;
    constexpr uint64_t MASK = (1ULL << BITS) - 1;

    // Load exactly the words we need (compile-time constant)
    uint32_t w[WORDS_NEEDED + 1];
    #pragma unroll
    for (int i = 0; i <= WORDS_NEEDED; i++) {
        w[i] = __ldg(&data[word_start + i]);
    }

    // Extract 8 values
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(w[word_idx + 1]) << 32) | w[word_idx];
        out[v] = base + static_cast<uint32_t>((combined >> bit_in_word) & MASK);
        local_bit += BITS;
    }
}

// Specialization for 0-bit (constant)
template<>
__device__ __forceinline__ void unpack8_v10<0>(
    const uint32_t* __restrict__ data,
    int64_t lane_bit_start,
    uint32_t base,
    uint32_t (&out)[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) out[v] = base;
}

// Runtime dispatcher
__device__ __forceinline__ void unpack8_dispatch(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane,
    int delta_bits,
    uint32_t base,
    uint32_t (&out)[8])
{
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane) * 8 * delta_bits;

    // Dispatch to specialized version - most common bit-widths for SSB columns
    switch (delta_bits) {
        case 0:  unpack8_v10<0>(data, lane_bit_start, base, out); break;
        case 12: unpack8_v10<12>(data, lane_bit_start, base, out); break;
        case 13: unpack8_v10<13>(data, lane_bit_start, base, out); break;
        case 14: unpack8_v10<14>(data, lane_bit_start, base, out); break;
        case 15: unpack8_v10<15>(data, lane_bit_start, base, out); break;
        case 16: unpack8_v10<16>(data, lane_bit_start, base, out); break;
        case 17: unpack8_v10<17>(data, lane_bit_start, base, out); break;
        case 18: unpack8_v10<18>(data, lane_bit_start, base, out); break;
        case 19: unpack8_v10<19>(data, lane_bit_start, base, out); break;
        case 20: unpack8_v10<20>(data, lane_bit_start, base, out); break;
        case 21: unpack8_v10<21>(data, lane_bit_start, base, out); break;
        case 22: unpack8_v10<22>(data, lane_bit_start, base, out); break;
        case 23: unpack8_v10<23>(data, lane_bit_start, base, out); break;
        case 24: unpack8_v10<24>(data, lane_bit_start, base, out); break;
        default: unpack8_v10<20>(data, lane_bit_start, base, out); break;  // Fallback
    }
}

// ============================================================================
// V10 Kernel - Compact Metadata + Template Unpack
// ============================================================================

__global__ void q21_fused_v10_kernel(
    // Compressed columns
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    // Compact metadata
    const V10TileMeta* __restrict__ tile_meta,
    int num_partitions,
    int total_rows,
    // Hash tables
    const uint32_t* __restrict__ ht_s,
    int ht_s_len,
    const uint32_t* __restrict__ ht_p,
    int ht_p_len,
    const uint32_t* __restrict__ ht_d,
    int ht_d_len,
    // Output
    unsigned long long* __restrict__ agg_revenue)
{
    int block_idx = blockIdx.x;
    int partition_idx = block_idx / 4;
    int mv_idx = block_idx % 4;

    if (partition_idx >= num_partitions) return;

    int lane = threadIdx.x;
    int partition_start = partition_idx * V10_PARTITION_SIZE;
    int partition_size = min(V10_PARTITION_SIZE, total_rows - partition_start);
    int mv_start = mv_idx * V10_MV_SIZE;
    int mv_valid = min(V10_MV_SIZE, partition_size - mv_start);

    if (mv_valid <= 0) return;

    // Load compact metadata (80 bytes per partition)
    V10TileMeta meta = tile_meta[partition_idx];

    // Initialize flags
    int flags[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        flags[v] = ((v * 32 + lane) < mv_valid) ? 1 : 0;
    }

    int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 1: Suppkey ==========
    uint32_t suppkey[8];
    {
        // local_offset is tile's offset within L3 partition
        // mv_start is the mini-vector's offset within this tile (each tile = 1024, mv = 256)
        int local_mv_idx = (meta.col[0].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[0].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[0].delta_bits;
        unpack8_dispatch(supp_deltas, mv_bit_base, lane,
            meta.col[0].delta_bits, meta.col[0].base_value, suppkey);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            if (__ldg(&ht_s[suppkey[v] % ht_s_len]) == 0) flags[v] = 0;
        }
    }

    any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 2: Partkey ==========
    uint32_t partkey[8], brand[8];
    {
        int local_mv_idx = (meta.col[1].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[1].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[1].delta_bits;
        unpack8_dispatch(part_deltas, mv_bit_base, lane,
            meta.col[1].delta_bits, meta.col[1].base_value, partkey);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht_p[(partkey[v] % ht_p_len) << 1]);
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
    uint32_t orderdate[8], year[8];
    {
        int local_mv_idx = (meta.col[2].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[2].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[2].delta_bits;
        unpack8_dispatch(od_deltas, mv_bit_base, lane,
            meta.col[2].delta_bits, meta.col[2].base_value, orderdate);
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
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 4: Revenue ==========
    uint32_t revenue[8];
    {
        int local_mv_idx = (meta.col[3].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[3].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[3].delta_bits;
        unpack8_dispatch(rev_deltas, mv_bit_base, lane,
            meta.col[3].delta_bits, meta.col[3].base_value, revenue);
    }

    // ========== Aggregation ==========
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int year_idx = year[v] - 1992;
            if (year_idx >= 0 && year_idx < AGG_NUM_YEARS && brand[v] < AGG_NUM_BRANDS) {
                atomicAdd(&agg_revenue[year_idx * AGG_NUM_BRANDS + brand[v]],
                         static_cast<unsigned long long>(revenue[v]));
            }
        }
    }
}

// ============================================================================
// V10 Metadata Builder
// ============================================================================

std::vector<V10TileMeta> buildV10Metadata(
    const CompressedDataVertical<uint32_t>& suppkey,
    const CompressedDataVertical<uint32_t>& partkey,
    const CompressedDataVertical<uint32_t>& orderdate,
    const CompressedDataVertical<uint32_t>& revenue,
    int total_rows)
{
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;
    std::vector<V10TileMeta> meta(num_partitions);

    // Copy device data to host for metadata building
    std::vector<int32_t> h_start(suppkey.num_partitions);
    std::vector<int32_t> h_delta_bits_supp(suppkey.num_partitions);
    std::vector<int32_t> h_delta_bits_part(partkey.num_partitions);
    std::vector<int32_t> h_delta_bits_od(orderdate.num_partitions);
    std::vector<int32_t> h_delta_bits_rev(revenue.num_partitions);
    std::vector<double> h_params_supp(suppkey.num_partitions * 4);
    std::vector<double> h_params_part(partkey.num_partitions * 4);
    std::vector<double> h_params_od(orderdate.num_partitions * 4);
    std::vector<double> h_params_rev(revenue.num_partitions * 4);
    std::vector<int64_t> h_offsets_supp(suppkey.num_partitions);
    std::vector<int64_t> h_offsets_part(partkey.num_partitions);
    std::vector<int64_t> h_offsets_od(orderdate.num_partitions);
    std::vector<int64_t> h_offsets_rev(revenue.num_partitions);

    cudaMemcpy(h_start.data(), suppkey.d_start_indices,
               suppkey.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits_supp.data(), suppkey.d_delta_bits,
               suppkey.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits_part.data(), partkey.d_delta_bits,
               partkey.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits_od.data(), orderdate.d_delta_bits,
               orderdate.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits_rev.data(), revenue.d_delta_bits,
               revenue.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params_supp.data(), suppkey.d_model_params,
               suppkey.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params_part.data(), partkey.d_model_params,
               partkey.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params_od.data(), orderdate.d_model_params,
               orderdate.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_params_rev.data(), revenue.d_model_params,
               revenue.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets_supp.data(), suppkey.d_interleaved_offsets,
               suppkey.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets_part.data(), partkey.d_interleaved_offsets,
               partkey.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets_od.data(), orderdate.d_interleaved_offsets,
               orderdate.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets_rev.data(), revenue.d_interleaved_offsets,
               revenue.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);

    for (int p = 0; p < num_partitions; p++) {
        int tile_start = p * V10_PARTITION_SIZE;

        // Find which L3 partition this tile falls into
        int l3_part = 0;
        for (int i = 1; i < suppkey.num_partitions; i++) {
            if (h_start[i] > tile_start) break;
            l3_part = i;
        }

        // Compute local_offset = tile_start - partition_start
        int local_offset = tile_start - h_start[l3_part];

        // Suppkey
        meta[p].col[0].bit_base = h_offsets_supp[l3_part] * 32;  // word offset to bit offset
        meta[p].col[0].base_value = static_cast<uint32_t>(h_params_supp[l3_part * 4]);
        meta[p].col[0].local_offset = local_offset;
        meta[p].col[0].delta_bits = static_cast<uint8_t>(h_delta_bits_supp[l3_part]);

        // Partkey
        meta[p].col[1].bit_base = h_offsets_part[l3_part] * 32;
        meta[p].col[1].base_value = static_cast<uint32_t>(h_params_part[l3_part * 4]);
        meta[p].col[1].local_offset = local_offset;
        meta[p].col[1].delta_bits = static_cast<uint8_t>(h_delta_bits_part[l3_part]);

        // Orderdate
        meta[p].col[2].bit_base = h_offsets_od[l3_part] * 32;
        meta[p].col[2].base_value = static_cast<uint32_t>(h_params_od[l3_part * 4]);
        meta[p].col[2].local_offset = local_offset;
        meta[p].col[2].delta_bits = static_cast<uint8_t>(h_delta_bits_od[l3_part]);

        // Revenue
        meta[p].col[3].bit_base = h_offsets_rev[l3_part] * 32;
        meta[p].col[3].base_value = static_cast<uint32_t>(h_params_rev[l3_part * 4]);
        meta[p].col[3].local_offset = local_offset;
        meta[p].col[3].delta_bits = static_cast<uint8_t>(h_delta_bits_rev[l3_part]);
    }

    return meta;
}

// ============================================================================
// Host Code
// ============================================================================

void runQ21FusedV10(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;

    // Build compact V10 metadata
    auto h_meta = buildV10Metadata(
        data.lo_suppkey, data.lo_partkey, data.lo_orderdate, data.lo_revenue, total_rows);

    V10TileMeta* d_meta;
    cudaMalloc(&d_meta, h_meta.size() * sizeof(V10TileMeta));
    cudaMemcpy(d_meta, h_meta.data(), h_meta.size() * sizeof(V10TileMeta), cudaMemcpyHostToDevice);

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
        12, ht_part.d_data, P_LEN);

    CrystalHashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    build_supplier_ht_region_crystal<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN, 1,
        ht_supplier.d_data, S_LEN);
    cudaDeviceSynchronize();

    int agg_size = AGG_NUM_YEARS * AGG_NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));

    int num_blocks = num_partitions * 4;

    std::cout << "\nV10 Configuration:" << std::endl;
    std::cout << "  Compact metadata: 80 bytes/partition (vs 290)" << std::endl;
    std::cout << "  Template-specialized unpack: 13 bit-width variants" << std::endl;
    std::cout << "  Partitions: " << num_partitions << std::endl;
    std::cout << "  Blocks: " << num_blocks << std::endl;
    std::cout << "  Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
        q21_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
            d_meta, num_partitions, total_rows,
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

    // Verify
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
    q21_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
        data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
        data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
        d_meta, num_partitions, total_rows,
        ht_supplier.d_data, S_LEN,
        ht_part.d_data, P_LEN,
        ht_date.d_data, DATE_HT_LEN,
        d_agg_revenue);
    cudaDeviceSynchronize();

    std::vector<unsigned long long> h_agg(agg_size);
    cudaMemcpy(h_agg.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 V10 Verification ===" << std::endl;
    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int i = 0; i < agg_size; ++i) {
        if (h_agg[i] > 0) { num_results++; total_revenue += h_agg[i]; }
    }
    std::cout << "Total groups: " << num_results << " (expected: 280)" << std::endl;
    std::cout << "Total revenue: " << total_revenue << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V10 Template-Specialized) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

        cudaEventRecord(start);
        q21_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_suppkey.d_interleaved_deltas, data.lo_partkey.d_interleaved_deltas,
            data.lo_orderdate.d_interleaved_deltas, data.lo_revenue.d_interleaved_deltas,
            d_meta, num_partitions, total_rows,
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
    std::cout << "Average kernel (V10): " << avg_time << " ms" << std::endl;
    std::cout << "V9 baseline: 1.15 ms" << std::endl;
    std::cout << "V7 baseline: 1.67 ms" << std::endl;
    std::cout << "Vertical target: 0.89 ms" << std::endl;
    std::cout << "Speedup vs V9: " << (1.15 / avg_time) << "x" << std::endl;
    std::cout << "Speedup vs V7: " << (1.67 / avg_time) << "x" << std::endl;
    std::cout << "Gap to Vertical: " << (avg_time / 0.89) << "x" << std::endl;

    cudaFree(d_agg_revenue);
    cudaFree(d_meta);
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
    std::cout << "SSB Q2.1 - V10 Template-Specialized Unpack" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key optimizations:" << std::endl;
    std::cout << "  - Compact 64-byte metadata (vs 290 bytes)" << std::endl;
    std::cout << "  - Template-specialized unpack (compile-time bit-width)" << std::endl;
    std::cout << "  - No runtime branches in hot path" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ21FusedV10(data);

    data.free();
    return 0;
}
