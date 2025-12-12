/**
 * @file q12_fused_v10.cu
 * @brief SSB Q1.2 - V10 Compact Metadata + 4x Parallelism
 *
 * Q1.2: Simple aggregation query (no joins)
 * SELECT SUM(lo_extendedprice * lo_discount) as revenue
 * FROM lineorder, date
 * WHERE lo_orderdate = d_datekey
 *   AND d_yearmonthnum = 199401  (January 1994)
 *   AND lo_discount BETWEEN 4 AND 6
 *   AND lo_quantity BETWEEN 26 AND 35
 *
 * Differences from Q1.1:
 *   - Q1.1: year 1993, discount 1-3, quantity < 25
 *   - Q1.2: yearmonth 199401 (Jan 1994), discount 4-6, quantity 26-35
 *
 * 4 columns: orderdate, quantity, discount, extendedprice
 * Bit widths: 16, 6, 4, 16
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

using namespace ssb;

// V10 constants
constexpr int V10_BLOCK_THREADS = 32;
constexpr int V10_MV_SIZE = 256;
constexpr int V10_PARTITION_SIZE = 1024;

// Q1.2 filter constants
constexpr uint32_t JAN_1994_START = 19940101;
constexpr uint32_t JAN_1994_END = 19940131;
constexpr uint32_t DISCOUNT_MIN = 4;
constexpr uint32_t DISCOUNT_MAX = 6;
constexpr uint32_t QUANTITY_MIN = 26;
constexpr uint32_t QUANTITY_MAX = 35;

// ============================================================================
// V10 Compact Metadata - 20 bytes per column, 4 columns = 80 bytes
// ============================================================================

struct V10ColumnMeta {
    int64_t bit_base;      // 8 bytes - bit offset for partition start
    uint32_t base_value;   // 4 bytes - FOR base
    int32_t local_offset;  // 4 bytes - tile_start - partition_start
    uint8_t delta_bits;    // 1 byte
    uint8_t padding[3];    // 3 bytes alignment
};  // 20 bytes total

struct V10TileMetaQ12 {
    V10ColumnMeta col[4];  // orderdate, quantity, discount, extendedprice
};  // 80 bytes total

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
        default: unpack8_v10<16>(data, lane_bit_start, base, out); break;
    }
}

// ============================================================================
// V10 Kernel - Q1.2 with Compact Metadata + 4x Parallelism
// ============================================================================

__global__ void q12_fused_v10_kernel(
    // Compressed columns (4 columns)
    const uint32_t* __restrict__ od_deltas,
    const uint32_t* __restrict__ qty_deltas,
    const uint32_t* __restrict__ disc_deltas,
    const uint32_t* __restrict__ price_deltas,
    // Compact metadata
    const V10TileMetaQ12* __restrict__ tile_meta,
    int num_partitions,
    int total_rows,
    // Output
    unsigned long long* __restrict__ global_revenue)
{
    // 4x parallelism: 4 blocks per partition
    int partition_id = blockIdx.x / 4;
    int sub_block = blockIdx.x % 4;

    if (partition_id >= num_partitions) return;

    int lane = threadIdx.x;

    // Each sub-block processes 256 values (one mini-vector)
    int mv_start = sub_block * V10_MV_SIZE;
    int tile_start = partition_id * V10_PARTITION_SIZE + mv_start;

    if (tile_start >= total_rows) return;

    int tile_end = min(tile_start + V10_MV_SIZE, total_rows);
    int valid_in_tile = tile_end - tile_start;

    // Load compact metadata (80 bytes - fits in registers)
    V10TileMetaQ12 meta = tile_meta[partition_id];

    // Warp-level reduction accumulator
    unsigned long long local_revenue = 0;

    // Each thread processes 8 values
    int values_per_thread = 8;
    int my_start = lane * values_per_thread;

    if (my_start >= valid_in_tile) return;

    // ========== Column 1: Orderdate (filter: January 1994) ==========
    uint32_t orderdate[8];
    int flags[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    {
        int local_mv_idx = (meta.col[0].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[0].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[0].delta_bits;
        unpack8_dispatch(od_deltas, mv_bit_base, lane,
            meta.col[0].delta_bits, meta.col[0].base_value, orderdate);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        int global_idx = my_start + v;
        if (global_idx >= valid_in_tile) {
            flags[v] = 0;
        } else if (orderdate[v] < JAN_1994_START || orderdate[v] > JAN_1994_END) {
            flags[v] = 0;
        }
    }

    int any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                    flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 2: Quantity (filter: 26-35) ==========
    uint32_t quantity[8];
    {
        int local_mv_idx = (meta.col[1].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[1].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[1].delta_bits;
        unpack8_dispatch(qty_deltas, mv_bit_base, lane,
            meta.col[1].delta_bits, meta.col[1].base_value, quantity);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v] && (quantity[v] < QUANTITY_MIN || quantity[v] > QUANTITY_MAX)) flags[v] = 0;
    }

    any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 3: Discount (filter: 4-6, compute) ==========
    uint32_t discount[8];
    {
        int local_mv_idx = (meta.col[2].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[2].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[2].delta_bits;
        unpack8_dispatch(disc_deltas, mv_bit_base, lane,
            meta.col[2].delta_bits, meta.col[2].base_value, discount);
    }

    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v] && (discount[v] < DISCOUNT_MIN || discount[v] > DISCOUNT_MAX)) flags[v] = 0;
    }

    any_valid = flags[0] | flags[1] | flags[2] | flags[3] |
                flags[4] | flags[5] | flags[6] | flags[7];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Column 4: Extendedprice (compute) ==========
    uint32_t extendedprice[8];
    {
        int local_mv_idx = (meta.col[3].local_offset + mv_start) / V10_MV_SIZE;
        int64_t mv_bit_base = meta.col[3].bit_base +
            static_cast<int64_t>(local_mv_idx) * V10_MV_SIZE * meta.col[3].delta_bits;
        unpack8_dispatch(price_deltas, mv_bit_base, lane,
            meta.col[3].delta_bits, meta.col[3].base_value, extendedprice);
    }

    // ========== Aggregation: revenue = extendedprice * discount ==========
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            local_revenue += static_cast<unsigned long long>(extendedprice[v]) *
                             static_cast<unsigned long long>(discount[v]);
        }
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_revenue += __shfl_down_sync(0xFFFFFFFF, local_revenue, offset);
    }

    // Lane 0 writes to global
    if (lane == 0 && local_revenue > 0) {
        atomicAdd(global_revenue, local_revenue);
    }
}

// ============================================================================
// V10 Metadata Builder for Q1.2
// ============================================================================

std::vector<V10TileMetaQ12> buildV10MetadataQ12(
    const CompressedDataVertical<uint32_t>& orderdate,
    const CompressedDataVertical<uint32_t>& quantity,
    const CompressedDataVertical<uint32_t>& discount,
    const CompressedDataVertical<uint32_t>& extendedprice,
    int total_rows)
{
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;
    std::vector<V10TileMetaQ12> meta(num_partitions);

    // Copy device data to host
    std::vector<int32_t> h_start(orderdate.num_partitions);
    std::vector<int32_t> h_delta_bits[4];
    std::vector<double> h_params[4];
    std::vector<int64_t> h_offsets[4];

    const CompressedDataVertical<uint32_t>* cols[4] = {
        &orderdate, &quantity, &discount, &extendedprice
    };

    cudaMemcpy(h_start.data(), orderdate.d_start_indices,
               orderdate.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int c = 0; c < 4; c++) {
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
        for (int i = 1; i < orderdate.num_partitions; i++) {
            if (h_start[i] > tile_start) break;
            l3_part = i;
        }

        int local_offset = tile_start - h_start[l3_part];

        for (int c = 0; c < 4; c++) {
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

void runQ12FusedV10(SSBDataCompressedVertical& data) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = (total_rows + V10_PARTITION_SIZE - 1) / V10_PARTITION_SIZE;

    // Build compact V10 metadata
    auto h_meta = buildV10MetadataQ12(
        data.lo_orderdate, data.lo_quantity, data.lo_discount,
        data.lo_extendedprice, total_rows);

    V10TileMetaQ12* d_meta;
    cudaMalloc(&d_meta, h_meta.size() * sizeof(V10TileMetaQ12));
    cudaMemcpy(d_meta, h_meta.data(), h_meta.size() * sizeof(V10TileMetaQ12), cudaMemcpyHostToDevice);

    // 4x parallelism: 4 blocks per partition
    int num_blocks = num_partitions * 4;

    // Allocate output
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    std::cout << "\nV10 Configuration:" << std::endl;
    std::cout << "  Compact metadata: 80 bytes/partition (4 columns)" << std::endl;
    std::cout << "  4x parallelism: " << num_blocks << " blocks" << std::endl;
    std::cout << "  Partitions: " << num_partitions << std::endl;
    std::cout << "  Rows: " << total_rows << std::endl;

    // Warmup
    for (int i = 0; i < 5; i++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        q12_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_orderdate.d_interleaved_deltas, data.lo_quantity.d_interleaved_deltas,
            data.lo_discount.d_interleaved_deltas, data.lo_extendedprice.d_interleaved_deltas,
            d_meta, num_partitions, total_rows, d_revenue);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Verify
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q12_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
        data.lo_orderdate.d_interleaved_deltas, data.lo_quantity.d_interleaved_deltas,
        data.lo_discount.d_interleaved_deltas, data.lo_extendedprice.d_interleaved_deltas,
        d_meta, num_partitions, total_rows, d_revenue);
    cudaDeviceSynchronize();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    const unsigned long long EXPECTED = 8571565529ULL;
    std::cout << "\n=== Q1.2 V10 Verification ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "Expected: " << EXPECTED << std::endl;
    std::cout << "Status: " << (h_revenue == EXPECTED ? "PASSED" : "FAILED") << std::endl;

    // Benchmark
    std::cout << "\n=== Benchmark Runs (V10) ===" << std::endl;
    const int RUNS = 10;
    float total_time = 0;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));

        cudaEventRecord(start);
        q12_fused_v10_kernel<<<num_blocks, V10_BLOCK_THREADS>>>(
            data.lo_orderdate.d_interleaved_deltas, data.lo_quantity.d_interleaved_deltas,
            data.lo_discount.d_interleaved_deltas, data.lo_extendedprice.d_interleaved_deltas,
            d_meta, num_partitions, total_rows, d_revenue);
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
    std::cout << "Speedup vs V5 baseline: " << (0.71 / avg_time) << "x" << std::endl;

    cudaFree(d_revenue);
    cudaFree(d_meta);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::cout << "======================================================" << std::endl;
    std::cout << "SSB Q1.2 - V10 Compact Metadata + 4x Parallelism" << std::endl;
    std::cout << "======================================================" << std::endl;
    std::cout << "Query: SELECT SUM(lo_extendedprice * lo_discount)" << std::endl;
    std::cout << "Filters:" << std::endl;
    std::cout << "  - d_yearmonthnum = 199401 (January 1994)" << std::endl;
    std::cout << "  - lo_discount BETWEEN 4 AND 6" << std::endl;
    std::cout << "  - lo_quantity BETWEEN 26 AND 35" << std::endl;
    std::cout << std::endl;
    std::cout << "Key optimizations:" << std::endl;
    std::cout << "  - Compact 80-byte metadata (4 columns x 20 bytes)" << std::endl;
    std::cout << "  - 4x parallelism (4 blocks per partition)" << std::endl;
    std::cout << "  - Template-specialized unpack" << std::endl;
    std::cout << "  - Warp-level reduction for aggregation" << std::endl;
    std::cout << std::endl;

    std::string data_path = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_path = argv[1];

    SSBDataCompressedVertical data;
    data.loadAndCompress(data_path);

    runQ12FusedV10(data);

    return 0;
}
