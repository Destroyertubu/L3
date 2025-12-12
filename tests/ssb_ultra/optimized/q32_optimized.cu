/**
 * @file q32_optimized.cu
 * @brief SSB Q3.2 - OPTIMIZED: Decompress-first + Two-Level Fast Hash
 * Query: c_nation='US', s_nation='US', d_year 1992-1997
 * Strategy: Decompress-first (high selectivity query) with Two-Level Fast Hash
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t C_NATION_US = 24;
constexpr uint32_t S_NATION_US = 24;
constexpr int NUM_YEARS = 6;
constexpr int NUM_CITIES = 250;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_CITIES;

__global__ void q32KernelOpt(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_custkey,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_revenue,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_values, int ht_c_size,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_values, int ht_s_size,
    unsigned long long* agg_revenue)
{
    uint32_t orderdate[ITEMS_PER_THREAD], custkey[ITEMS_PER_THREAD], suppkey[ITEMS_PER_THREAD], revenue[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);
    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockLoad(lo_custkey + block_start, custkey, block_items);
    BlockLoad(lo_suppkey + block_start, suppkey, block_items);
    BlockLoad(lo_revenue + block_start, revenue, block_items);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx >= block_items || !selection_flags[i]) continue;

        // Date probe using Two-Level Fast Hash
        uint32_t d_year = 0;
        if (!twoLevelProbeWithValue(orderdate[i], ht_d_keys, ht_d_values, ht_d_size, d_year) || d_year < 1992 || d_year > 1997) {
            selection_flags[i] = 0; continue;
        }

        // Customer probe using Two-Level Fast Hash
        uint32_t c_city = 0;
        if (!twoLevelProbeWithValue(custkey[i], ht_c_keys, ht_c_values, ht_c_size, c_city)) {
            selection_flags[i] = 0; continue;
        }

        // Supplier probe using Two-Level Fast Hash
        uint32_t s_city = 0;
        if (!twoLevelProbeWithValue(suppkey[i], ht_s_keys, ht_s_values, ht_s_size, s_city)) {
            selection_flags[i] = 0; continue;
        }

        int year_idx = d_year - 1992;
        int agg_idx = year_idx * NUM_CITIES * NUM_CITIES + c_city * NUM_CITIES + s_city;
        atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
    }
}

__global__ void build_date_ht(const uint32_t* dk, const uint32_t* dy, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; twoLevelInsert(dk[i], dy[i], k, v, s);
}
__global__ void build_cust_ht(const uint32_t* ck, const uint32_t* cn, const uint32_t* cc, int n, uint32_t fn,
    uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || cn[i] != fn) return; twoLevelInsert(ck[i], cc[i], k, v, s);
}
__global__ void build_supp_ht(const uint32_t* sk, const uint32_t* sn, const uint32_t* sc, int n, uint32_t fn,
    uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n || sn[i] != fn) return; twoLevelInsert(sk[i], sc[i], k, v, s);
}

void runQ32Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    // Create 4 streams for parallel decompression
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_orderdate.getTotalElements();

    uint32_t *d_lo_orderdate, *d_lo_custkey, *d_lo_suppkey, *d_lo_revenue;
    cudaMalloc(&d_lo_orderdate, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_lo_custkey, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, total_rows * sizeof(uint32_t));

    // PARALLEL decompress all 4 columns using multiple streams
    timer.start();
    acc_orderdate.decompressAll(d_lo_orderdate, streams[0]);
    acc_custkey.decompressAll(d_lo_custkey, streams[1]);
    acc_suppkey.decompressAll(d_lo_suppkey, streams[2]);
    acc_revenue.decompressAll(d_lo_revenue, streams[3]);

    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // Build hash tables with Two-Level Fast Hash
    timer.start();
    int ht_d_size = D_LEN * 2, ht_c_size = C_LEN * 2, ht_s_size = S_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_c_keys, *ht_c_values, *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t)); cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    int bs = 256;
    build_date_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_cust_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_nation, data.d_c_city, C_LEN, C_NATION_US, ht_c_keys, ht_c_values, ht_c_size);
    build_supp_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN, S_NATION_US, ht_s_keys, ht_s_values, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    q32KernelOpt<<<(total_rows+TILE_SIZE-1)/TILE_SIZE, BLOCK_THREADS>>>(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, total_rows,
        ht_d_keys, ht_d_values, ht_d_size, ht_c_keys, ht_c_values, ht_c_size, ht_s_keys, ht_s_values, ht_s_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q3.2 (OPTIMIZED - Parallel Decompress) ===" << std::endl;
    unsigned long long total = 0; int groups = 0;
    for (size_t i = 0; i < AGG_SIZE; ++i) if (h_agg[i] > 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Total: " << total << std::endl;
    std::cout << "  Parallel decompress 4 cols: " << timing.data_load_ms << " ms" << std::endl;
    timing.print("Q3.2");

    cudaFree(d_lo_orderdate); cudaFree(d_lo_custkey); cudaFree(d_lo_suppkey); cudaFree(d_lo_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_agg_revenue);

    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }
}

int main(int argc, char** argv) {
    std::string dir = "/root/autodl-tmp/test/ssb_data"; if (argc > 1) dir = argv[1];
    std::cout << "=== SSB Q3.2 - OPTIMIZED ===" << std::endl;
    SSBDataCompressedVertical data; data.loadAndCompress(dir);
    QueryTiming t; runQ32Optimized(data, t);
    std::cout << "\n=== Benchmark ===" << std::endl;
    for (int i = 0; i < 3; i++) { QueryTiming x; runQ32Optimized(data, x); std::cout << "Run " << i+1 << ": " << x.total_ms << " ms\n"; }
    data.free(); return 0;
}
