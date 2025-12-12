/**
 * @file q42_optimized.cu
 * @brief SSB Q4.2 - OPTIMIZED: Decompress-first + Two-Level Fast Hash
 * Query: c_region='AMERICA', s_region='AMERICA', d_year 1997-1998, p_mfgr in ('MFGR#1','MFGR#2')
 * Strategy: Decompress-first (high selectivity from part filter) with Two-Level Fast Hash
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

constexpr uint32_t C_REGION_AMERICA = 1;
constexpr uint32_t S_REGION_AMERICA = 1;
constexpr uint32_t P_MFGR_1 = 1;
constexpr uint32_t P_MFGR_2 = 2;
constexpr int NUM_YEARS = 2;  // 1997-1998
constexpr int NUM_NATIONS = 25;
constexpr int NUM_CATEGORIES = 50;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS * NUM_CATEGORIES;

__global__ void q42KernelOpt(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_custkey,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_partkey,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ lo_supplycost,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    const uint32_t* __restrict__ ht_c_keys, int ht_c_size,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_values, int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_values, int ht_p_size,
    long long* agg_profit)
{
    uint32_t orderdate[ITEMS_PER_THREAD], custkey[ITEMS_PER_THREAD], suppkey[ITEMS_PER_THREAD];
    uint32_t partkey[ITEMS_PER_THREAD], revenue[ITEMS_PER_THREAD], supplycost[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);
    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockLoad(lo_custkey + block_start, custkey, block_items);
    BlockLoad(lo_suppkey + block_start, suppkey, block_items);
    BlockLoad(lo_partkey + block_start, partkey, block_items);
    BlockLoad(lo_revenue + block_start, revenue, block_items);
    BlockLoad(lo_supplycost + block_start, supplycost, block_items);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx >= block_items || !selection_flags[i]) continue;

        // Part probe (with category value)
        uint32_t p_category = 0;
        if (!twoLevelProbeWithValue(partkey[i], ht_p_keys, ht_p_values, ht_p_size, p_category)) {
            selection_flags[i] = 0; continue;
        }

        // Supplier probe (with nation value)
        uint32_t s_nation = 0;
        if (!twoLevelProbeWithValue(suppkey[i], ht_s_keys, ht_s_values, ht_s_size, s_nation)) {
            selection_flags[i] = 0; continue;
        }

        // Customer probe (region filter only)
        int c_slot;
        if (!twoLevelProbe(custkey[i], ht_c_keys, ht_c_size, c_slot)) {
            selection_flags[i] = 0; continue;
        }

        // Date probe using Two-Level Fast Hash
        uint32_t d_year = 0;
        if (!twoLevelProbeWithValue(orderdate[i], ht_d_keys, ht_d_values, ht_d_size, d_year) || d_year < 1997 || d_year > 1998) {
            selection_flags[i] = 0; continue;
        }

        int year_idx = d_year - 1997;
        int agg_idx = year_idx * NUM_NATIONS * NUM_CATEGORIES + s_nation * NUM_CATEGORIES + p_category;
        long long profit = (long long)revenue[i] - (long long)supplycost[i];
        atomicAdd((unsigned long long*)&agg_profit[agg_idx], (unsigned long long)profit);
    }
}

__global__ void build_date_ht(const uint32_t* dk, const uint32_t* dy, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; twoLevelInsert(dk[i], dy[i], k, v, s);
}
__global__ void build_part_mfgr_ht(const uint32_t* pk, const uint32_t* pm, const uint32_t* pc, int n,
    uint32_t m1, uint32_t m2, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (pm[i] != m1 && pm[i] != m2)) return;
    twoLevelInsert(pk[i], pc[i], k, v, s);
}
__global__ void build_cust_region_ht(const uint32_t* ck, const uint32_t* cr, int n, uint32_t fr, uint32_t* k, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || cr[i] != fr) return;
    int slot = hash_fast(ck[i], s);
    uint32_t old = atomicCAS(&k[slot], (uint32_t)HT_EMPTY, ck[i]);
    if (old == (uint32_t)HT_EMPTY || old == ck[i]) return;
    slot = hash_murmur3(ck[i], s);
    for (int p = 0; p < s; p++) { int sl = (slot + p) % s; old = atomicCAS(&k[sl], (uint32_t)HT_EMPTY, ck[i]); if (old == (uint32_t)HT_EMPTY || old == ck[i]) return; }
}
__global__ void build_supp_region_ht(const uint32_t* sk, const uint32_t* sr, const uint32_t* sn, int n,
    uint32_t fr, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || sr[i] != fr) return;
    twoLevelInsert(sk[i], sn[i], k, v, s);
}

void runQ42Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    uint32_t *d_lo_orderdate, *d_lo_custkey, *d_lo_suppkey, *d_lo_partkey, *d_lo_revenue, *d_lo_supplycost;
    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_custkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_supplycost, LO_LEN * sizeof(uint32_t));

    // Decompress all columns first
    timer.start();
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_orderdate).decompressAll(d_lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_custkey).decompressAll(d_lo_custkey);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_suppkey).decompressAll(d_lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_partkey).decompressAll(d_lo_partkey);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_revenue).decompressAll(d_lo_revenue);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_supplycost).decompressAll(d_lo_supplycost);
    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // Build hash tables with Two-Level Fast Hash
    timer.start();
    int bs = 256;
    int ht_d_size = D_LEN * 2; uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    build_date_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_p_size = P_LEN * 2; uint32_t *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t)); cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));
    build_part_mfgr_ht<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_mfgr, data.d_p_category, P_LEN, P_MFGR_1, P_MFGR_2, ht_p_keys, ht_p_values, ht_p_size);

    int ht_c_size = C_LEN * 2; uint32_t *ht_c_keys;
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    build_cust_region_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_region, C_LEN, C_REGION_AMERICA, ht_c_keys, ht_c_size);

    int ht_s_size = S_LEN * 2; uint32_t *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    build_supp_region_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_values, ht_s_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    timer.start();
    q42KernelOpt<<<(LO_LEN+TILE_SIZE-1)/TILE_SIZE, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_partkey, d_lo_revenue, d_lo_supplycost, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_c_keys, ht_c_size, ht_s_keys, ht_s_values, ht_s_size,
        ht_p_keys, ht_p_values, ht_p_size, d_agg_profit);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::vector<long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q4.2 (OPTIMIZED) ===" << std::endl;
    long long total = 0; int groups = 0;
    for (size_t i = 0; i < AGG_SIZE; ++i) if (h_agg[i] != 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Profit: " << total << std::endl;
    timing.print("Q4.2");

    cudaFree(d_lo_orderdate); cudaFree(d_lo_custkey); cudaFree(d_lo_suppkey); cudaFree(d_lo_partkey);
    cudaFree(d_lo_revenue); cudaFree(d_lo_supplycost);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(ht_c_keys); cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_agg_profit);
}

int main(int argc, char** argv) {
    std::string dir = "/root/autodl-tmp/test/ssb_data"; if (argc > 1) dir = argv[1];
    std::cout << "=== SSB Q4.2 - OPTIMIZED ===" << std::endl;
    SSBDataCompressedVertical data; data.loadAndCompress(dir);
    QueryTiming t; runQ42Optimized(data, t);
    std::cout << "\n=== Benchmark ===" << std::endl;
    for (int i = 0; i < 3; i++) { QueryTiming x; runQ42Optimized(data, x); std::cout << "Run " << i+1 << ": " << x.total_ms << " ms\n"; }
    data.free(); return 0;
}
