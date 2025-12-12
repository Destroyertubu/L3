/**
 * @file q43.cu
 * @brief SSB Q4.3 Implementation - Decompress First Strategy
 *
 * Query:
 *   SELECT d_year, s_city, p_brand1, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM date, customer, supplier, part, lineorder
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND s_nation = 'UNITED STATES'
 *     AND (d_year = 1997 OR d_year = 1998)
 *     AND p_category = 'MFGR#14'
 *   GROUP BY d_year, s_city, p_brand1 ORDER BY d_year, s_city, p_brand1;
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t S_NATION_US = 24;  // UNITED STATES
constexpr uint32_t P_CATEGORY_MFGR14 = 14;  // MFGR#14
constexpr int NUM_YEARS = 2;  // 1997, 1998
constexpr int NUM_CITIES = 250;
constexpr int NUM_BRANDS = 1000;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_BRANDS;

__global__ void q43_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_partkey,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ lo_supplycost,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_values, int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_values, int ht_p_size,
    long long* agg_profit)
{
    uint32_t orderdate[ITEMS_PER_THREAD], suppkey[ITEMS_PER_THREAD];
    uint32_t partkey[ITEMS_PER_THREAD], revenue[ITEMS_PER_THREAD], supplycost[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);
    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockLoad(lo_suppkey + block_start, suppkey, block_items);
    BlockLoad(lo_partkey + block_start, partkey, block_items);
    BlockLoad(lo_revenue + block_start, revenue, block_items);
    BlockLoad(lo_supplycost + block_start, supplycost, block_items);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx >= block_items || !selection_flags[i]) continue;

        // Probe date (1997 or 1998)
        uint32_t od = orderdate[i];
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool d_found = false;
        for (int p = 0; p < ht_d_size && !d_found; ++p) {
            int slot = (d_slot + p) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; d_found = true; }
            else if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!d_found) { selection_flags[i] = 0; continue; }

        // Probe supplier (UNITED STATES, get city)
        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        uint32_t s_city = 0;
        bool s_found = false;
        for (int p = 0; p < ht_s_size && !s_found; ++p) {
            int slot = (s_slot + p) % ht_s_size;
            if (ht_s_keys[slot] == sk) { s_city = ht_s_values[slot]; s_found = true; }
            else if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) { selection_flags[i] = 0; continue; }

        // Probe part (MFGR#14, get brand1)
        uint32_t pk = partkey[i];
        int p_slot = hash_murmur3(pk, ht_p_size);
        uint32_t p_brand = 0;
        bool p_found = false;
        for (int p = 0; p < ht_p_size && !p_found; ++p) {
            int slot = (p_slot + p) % ht_p_size;
            if (ht_p_keys[slot] == pk) { p_brand = ht_p_values[slot]; p_found = true; }
            else if (ht_p_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!p_found) { selection_flags[i] = 0; continue; }

        int year_idx = d_year - 1997;
        if (year_idx >= 0 && year_idx < NUM_YEARS && s_city < NUM_CITIES && p_brand < NUM_BRANDS) {
            int agg_idx = year_idx * NUM_CITIES * NUM_BRANDS + s_city * NUM_BRANDS + p_brand;
            long long profit = static_cast<long long>(revenue[i]) - static_cast<long long>(supplycost[i]);
            atomicAddLL(&agg_profit[agg_idx], profit);
        }
    }
}

__global__ void build_date_year_filter_ht(const uint32_t* d_datekey, const uint32_t* d_year, int n,
                                           uint32_t year1, uint32_t year2, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t year = d_year[idx];
    if (year != year1 && year != year2) return;
    uint32_t key = d_datekey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = year; return; }
    }
}

__global__ void build_supplier_nation_city_ht(const uint32_t* s_suppkey, const uint32_t* s_nation, const uint32_t* s_city,
                                               int n, uint32_t filter_nation, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (s_nation[idx] != filter_nation) return;
    uint32_t key = s_suppkey[idx], val = s_city[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_part_category_brand_ht(const uint32_t* p_partkey, const uint32_t* p_category, const uint32_t* p_brand1,
                                              int n, uint32_t filter_category, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (p_category[idx] != filter_category) return;
    uint32_t key = p_partkey[idx], val = p_brand1[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

void runQ43DecompressFirst(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    uint32_t *d_lo_orderdate, *d_lo_suppkey, *d_lo_partkey, *d_lo_revenue, *d_lo_supplycost;
    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_supplycost, LO_LEN * sizeof(uint32_t));

    timer.start();
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_orderdate).decompressAll(d_lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_suppkey).decompressAll(d_lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_partkey).decompressAll(d_lo_partkey);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_revenue).decompressAll(d_lo_revenue);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_supplycost).decompressAll(d_lo_supplycost);
    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    timer.start();
    int ht_d_size = D_LEN * 2, ht_s_size = S_LEN * 2, ht_p_size = P_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_s_keys, *ht_s_values, *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t)); cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    int bs = 256;
    build_date_year_filter_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, 1997, 1998, ht_d_keys, ht_d_values, ht_d_size);
    build_supplier_nation_city_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN, S_NATION_US, ht_s_keys, ht_s_values, ht_s_size);
    build_part_category_brand_ht<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN, P_CATEGORY_MFGR14, ht_p_keys, ht_p_values, ht_p_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    timer.start();
    q43_kernel<<<(LO_LEN+TILE_SIZE-1)/TILE_SIZE, BLOCK_THREADS>>>(d_lo_orderdate, d_lo_suppkey, d_lo_partkey, d_lo_revenue, d_lo_supplycost, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_s_keys, ht_s_values, ht_s_size, ht_p_keys, ht_p_values, ht_p_size, d_agg_profit);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::vector<long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q4.3 Results ===" << std::endl;
    long long total = 0; int groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) if (h_agg[i] != 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Total profit: " << total << std::endl;
    timing.print("Q4.3");

    cudaFree(d_lo_orderdate); cudaFree(d_lo_suppkey); cudaFree(d_lo_partkey); cudaFree(d_lo_revenue); cudaFree(d_lo_supplycost);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_s_keys); cudaFree(ht_s_values); cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(d_agg_profit);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];
    std::cout << "=== SSB Q4.3 - Decompress First ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    QueryTiming timing;
    runQ43DecompressFirst(data, timing);
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ43DecompressFirst(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
