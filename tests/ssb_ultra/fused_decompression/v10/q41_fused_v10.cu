/**
 * @file q41_fused_v10.cu
 * @brief SSB Q4.1 with V10 Optimizations - Compact Metadata + 4x Parallelism
 *
 * V10 Key Optimizations:
 * 1. Compact 120-byte metadata per partition (6 columns × 20 bytes)
 * 2. 4x parallelism: 4 blocks per partition (256 values each)
 * 3. Template-specialized unpack for common bit-widths
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
constexpr int V10_MV_SIZE = 256;
constexpr int V10_PARTITION_SIZE = 1024;

// Aggregation dimensions
constexpr int AGG_NUM_YEARS = 7;
constexpr int AGG_NUM_NATIONS = 25;
constexpr uint32_t REGION_AMERICA = 1;
constexpr uint32_t MFGR_1 = 1;
constexpr uint32_t MFGR_2 = 2;

// ============================================================================
// V10 Compact Metadata - 20 bytes per column, 6 columns = 120 bytes
// ============================================================================

struct V10ColumnMeta {
    int64_t bit_base;      // 8 bytes - bit offset for partition start
    uint32_t base_value;   // 4 bytes - FOR base
    int32_t local_offset;  // 4 bytes - tile_start - partition_start
    uint8_t delta_bits;    // 1 byte
    uint8_t padding[3];    // 3 bytes alignment
};  // 20 bytes total

struct V10TileMetaQ41 {
    V10ColumnMeta col[6];  // custkey, suppkey, partkey, orderdate, revenue, supplycost
};  // 120 bytes total

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

    uint32_t w[WORDS_NEEDED + 1];
    #pragma unroll
    for (int i = 0; i <= WORDS_NEEDED; i++) {
        w[i] = __ldg(&data[word_start + i]);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;
        uint64_t combined = (static_cast<uint64_t>(w[word_idx + 1]) << 32) | w[word_idx];
        out[v] = base + static_cast<uint32_t>((combined >> bit_in_word) & MASK);
        local_bit += BITS;
    }
}

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

__device__ __forceinline__ void unpack8_dispatch(
    const uint32_t* __restrict__ data,
    int64_t mv_bit_base,
    int lane,
    int delta_bits,
    uint32_t base,
    uint32_t (&out)[8])
{
    int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane) * 8 * delta_bits;

    switch (delta_bits) {
        case 0:  unpack8_v10<0>(data, lane_bit_start, base, out); break;
        case 1:  unpack8_v10<1>(data, lane_bit_start, base, out); break;
        case 2:  unpack8_v10<2>(data, lane_bit_start, base, out); break;
        case 3:  unpack8_v10<3>(data, lane_bit_start, base, out); break;
        case 4:  unpack8_v10<4>(data, lane_bit_start, base, out); break;
        case 5:  unpack8_v10<5>(data, lane_bit_start, base, out); break;
        case 6:  unpack8_v10<6>(data, lane_bit_start, base, out); break;
        case 7:  unpack8_v10<7>(data, lane_bit_start, base, out); break;
        case 8:  unpack8_v10<8>(data, lane_bit_start, base, out); break;
        case 9:  unpack8_v10<9>(data, lane_bit_start, base, out); break;
        case 10: unpack8_v10<10>(data, lane_bit_start, base, out); break;
        case 11: unpack8_v10<11>(data, lane_bit_start, base, out); break;
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
        case 25: unpack8_v10<25>(data, lane_bit_start, base, out); break;
        case 26: unpack8_v10<26>(data, lane_bit_start, base, out); break;
        case 27: unpack8_v10<27>(data, lane_bit_start, base, out); break;
        case 28: unpack8_v10<28>(data, lane_bit_start, base, out); break;
        case 29: unpack8_v10<29>(data, lane_bit_start, base, out); break;
        case 30: unpack8_v10<30>(data, lane_bit_start, base, out); break;
        case 31: unpack8_v10<31>(data, lane_bit_start, base, out); break;
        case 32: unpack8_v10<32>(data, lane_bit_start, base, out); break;
        default: unpack8_v10<20>(data, lane_bit_start, base, out); break;
    }
}

// ============================================================================
// V10 Kernel - Q4.1 with Compact Metadata + 4x Parallelism
// ============================================================================

__global__ void q41_fused_v10_kernel(
    // Compressed columns (6 columns)
    const uint32_t* __restrict__ cust_deltas,
    const uint32_t* __restrict__ supp_deltas,
    const uint32_t* __restrict__ part_deltas,
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ rev_deltas,
    const uint32_t* __restrict__ cost_deltas,
    // Compact metadata
    const V10TileMetaQ41* __restrict__ tile_meta,
    int num_partitions,
    int total_rows,
    // Hash tables (Crystal-style)
    const uint32_t* __restrict__ ht_c,     // Customer: custkey -> nation
    int ht_c_len,
    const uint32_t* __restrict__ ht_s,     // Supplier: keys only
    int ht_s_len,
    const uint32_t* __restrict__ ht_p,     // Part: keys only
    int ht_p_len,
    const uint32_t* __restrict__ ht_d,     // Date: datekey -> year
    int ht_d_len,
    // Output
    long long* __restrict__ agg_profit)
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

    // Load compact metadata (120 bytes per partition)
    V10TileMetaQ41 meta = tile_meta[partition_idx];

    // Initialize flags
    int flags[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        flags[v] = ((v * 32 + lane) < mv_valid) ? 1 : 0;
    }

    int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 1: Custkey (get nation) ==========
    uint32_t custkey[8], c_nation[8];
    {
        int local_mv_idx = (meta.col[0].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[0].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[0].delta_bits;
        unpack8_dispatch(cust_deltas, mv_bit_base, lane,
            meta.col[0].delta_bits, meta.col[0].base_value, custkey);
    }

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
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 2: Suppkey (filter only) ==========
    uint32_t suppkey[8];
    {
        int local_mv_idx = (meta.col[1].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[1].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[1].delta_bits;
        unpack8_dispatch(supp_deltas, mv_bit_base, lane,
            meta.col[1].delta_bits, meta.col[1].base_value, suppkey);
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

    // ========== Column 3: Partkey (filter only) ==========
    uint32_t partkey[8];
    {
        int local_mv_idx = (meta.col[2].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[2].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[2].delta_bits;
        unpack8_dispatch(part_deltas, mv_bit_base, lane,
            meta.col[2].delta_bits, meta.col[2].base_value, partkey);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            if (__ldg(&ht_p[partkey[v] % ht_p_len]) == 0) flags[v] = 0;
        }
    }

    any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 4: Orderdate (get year) ==========
    uint32_t orderdate[8], year[8];
    {
        int local_mv_idx = (meta.col[3].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[3].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[3].delta_bits;
        unpack8_dispatch(od_deltas, mv_bit_base, lane,
            meta.col[3].delta_bits, meta.col[3].base_value, orderdate);
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

    // ========== Column 5: Revenue ==========
    uint32_t revenue[8];
    {
        int local_mv_idx = (meta.col[4].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[4].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[4].delta_bits;
        unpack8_dispatch(rev_deltas, mv_bit_base, lane,
            meta.col[4].delta_bits, meta.col[4].base_value, revenue);
    }

    // ========== Column 6: Supplycost ==========
    uint32_t supplycost[8];
    {
        int local_mv_idx = (meta.col[5].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[5].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[5].delta_bits;
        unpack8_dispatch(cost_deltas, mv_bit_base, lane,
            meta.col[5].delta_bits, meta.col[5].base_value, supplycost);
    }

    // ========== Aggregation: profit = revenue - supplycost ==========
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

// ============================================================================
// V10 Metadata Builder for Q4.1
// ============================================================================

std::vector<V10TileMetaQ41> buildV10MetadataQ41(
    const CompressedDataVertical<uint32_t>& custkey,
    const CompressedDataVertical<uint32_t>& suppkey,
    const CompressedDataVertical<uint32_t>& partkey,
    const CompressedDataVertical<uint32_t>& orderdate,
    const CompressedDataVertical<uint32_t>& revenue,
    const CompressedDataVertical<uint32_t>& supplycost,
    int total_rows)
{
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;
    std::vector<V10TileMetaQ41> meta(num_partitions);

    // Copy device data to host
    std::vector<int32_t> h_start(custkey.num_partitions);
    std::vector<int32_t> h_delta_bits[6];
    std::vector<double> h_params[6];
    std::vector<int64_t> h_offsets[6];

    const CompressedDataVertical<uint32_t>* cols[6] = {
        &custkey, &suppkey, &partkey, &orderdate, &revenue, &supplycost
    };

    cudaMemcpy(h_start.data(), custkey.d_start_indices,
               custkey.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int c = 0; c < 6; c++) {
        h_delta_bits[c].resize(cols[c]->num_partitions);
        h_params[c].resize(cols[c]->num_partitions * 4);
        h_offsets[c].resize(cols[c]->num_partitions);

        cudaMemcpy(h_delta_bits[c].data(), cols[c]->d_delta_bits,
                   cols[c]->num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_params[c].data(), cols[c]->d_model_params,
                   cols[c]->num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_offsets[c].data(), cols[c]->d_interleaved_offsets,
                   cols[c]->num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }

    for (int p = 0; p < num_partitions; p++) {
        int tile_start = p * V10_PARTITION_SIZE;

        // Find L3 partition
        int l3_part = 0;
        for (int i = 1; i < custkey.num_partitions; i++) {
            if (h_start[i] > tile_start) break;
            l3_part = i;
        }

        int local_offset = tile_start - h_start[l3_part];

        for (int c = 0; c < 6; c++) {
            meta[p].col[c].bit_base = h_offsets[c][l3_part] * 32;
            meta[p].col[c].base_value = static_cast<uint32_t>(h_params[c][l3_part * 4]);
            meta[p].col[c].local_offset = local_offset;
            meta[p].col[c].delta_bits = static_cast<uint8_t>(h_delta_bits[c][l3_part]);
        }
    }

    return meta;
}

// ============================================================================
// Host Code
// ============================================================================

void runQ41FusedV10(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;

    // Build compact V10 metadata
    auto h_meta = buildV10MetadataQ41(
        data.lo_custkey, data.lo_suppkey, data.lo_partkey,
        data.lo_orderdate, data.lo_revenue, data.lo_supplycost, total_rows);

    V10TileMetaQ41* d_meta;
    cudaMalloc(&d_meta, h_meta.size() * sizeof(V10TileMetaQ41));
    cudaMemcpy(d_meta, h_meta.data(), h_meta.size() * sizeof(V10TileMetaQ41), cudaMemcpyHostToDevice);

    int block_size = 256;

    // Build Crystal-style hash tables
    CrystalHashTable ht_date;
    ht_date.allocate(DATE_HT_LEN, true);
    build_date_ht_crystal<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_date.d_data, DATE_HT_LEN);

    CrystalHashTable ht_customer;
    ht_customer.allocate(C_LEN, true);
    build_customer_ht_region_crystal<<<(C_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN,
        REGION_AMERICA, ht_customer.d_data, C_LEN);

    CrystalHashTable ht_supplier;
    ht_supplier.allocate(S_LEN, false);
    build_supplier_ht_region_crystal<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN,
        REGION_AMERICA, ht_supplier.d_data, S_LEN);

    CrystalHashTable ht_part;
    ht_part.allocate(P_LEN, false);
    build_part_ht_mfgr_crystal<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_mfgr, P_LEN,
        MFGR_1, MFGR_2, ht_part.d_data, P_LEN);
    cudaDeviceSynchronize();

    int agg_size = AGG_NUM_YEARS * AGG_NUM_NATIONS;
    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, agg_size * sizeof(long long));

    int num_blocks = num_partitions * 4;

    std::cout << "\nV10 Configuration:" << std::endl;
    std::cout << "  Compact metadata: 120 bytes/partition (6 columns)" << std::endl;
    std::cout << "  4x parallelism: " << num_blocks << " blocks" << std::endl;
    std::cout << "  Partitions: " << num_partitions << std::endl;
    std::cout << "  Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));
        q41_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            d_meta, num_partitions, total_rows,
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
    q41_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
        data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
        data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
        data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
        d_meta, num_partitions, total_rows,
        ht_customer.d_data, C_LEN,
        ht_supplier.d_data, S_LEN,
        ht_part.d_data, P_LEN,
        ht_date.d_data, DATE_HT_LEN,
        d_agg_profit);
    cudaDeviceSynchronize();

    std::vector<long long> h_agg(agg_size);
    cudaMemcpy(h_agg.data(), d_agg_profit, agg_size * sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q4.1 V10 Verification ===" << std::endl;
    int num_results = 0;
    long long total_profit = 0;
    for (int i = 0; i < agg_size; ++i) {
        if (h_agg[i] != 0) { num_results++; total_profit += h_agg[i]; }
    }
    std::cout << "Total groups: " << num_results << " (expected: ~35)" << std::endl;
    std::cout << "Total profit: " << total_profit << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V10) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg_profit, 0, agg_size * sizeof(long long));

        cudaEventRecord(start);
        q41_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_custkey.d_interleaved_deltas, data.lo_suppkey.d_interleaved_deltas,
            data.lo_partkey.d_interleaved_deltas, data.lo_orderdate.d_interleaved_deltas,
            data.lo_revenue.d_interleaved_deltas, data.lo_supplycost.d_interleaved_deltas,
            d_meta, num_partitions, total_rows,
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

    float avg_time = total_time / RUNS;
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Average kernel (V10): " << avg_time << " ms" << std::endl;
    std::cout << "V7 baseline: 2.92 ms" << std::endl;
    std::cout << "Vertical target: 2.73 ms" << std::endl;
    std::cout << "Speedup vs V7: " << (2.92 / avg_time) << "x" << std::endl;
    std::cout << "Gap to Vertical: " << (avg_time / 2.73) << "x" << std::endl;

    cudaFree(d_agg_profit);
    cudaFree(d_meta);
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
    std::cout << "SSB Q4.1 - V10 Compact Metadata + 4x Parallelism" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Key optimizations:" << std::endl;
    std::cout << "  - Compact 120-byte metadata (6 columns × 20 bytes)" << std::endl;
    std::cout << "  - 4x parallelism (4 blocks per partition)" << std::endl;
    std::cout << "  - Template-specialized unpack" << std::endl;
    std::cout << std::endl;

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    runQ41FusedV10(data);

    data.free();
    return 0;
}
