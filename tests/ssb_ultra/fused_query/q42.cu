/**
 * @file q42.cu
 * @brief SSB Q4.2 Implementation - Fused Query Strategy
 *
 * Query:
 *   SELECT d_year, s_nation, p_category, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM date, customer, supplier, part, lineorder
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *     AND c_region = 'AMERICA' AND s_region = 'AMERICA'
 *     AND (d_year = 1997 OR d_year = 1998)
 *     AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 *   GROUP BY d_year, s_nation, p_category ORDER BY d_year, s_nation, p_category;
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

constexpr int CHUNK_SIZE = 1024 * 1024;
constexpr uint32_t C_REGION_AMERICA = 1;
constexpr uint32_t S_REGION_AMERICA = 1;
constexpr uint32_t P_MFGR_1 = 1;
constexpr uint32_t P_MFGR_2 = 2;
constexpr int NUM_YEARS = 2;
constexpr int NUM_NATIONS = 25;
constexpr int NUM_CATEGORIES = 25;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS * NUM_CATEGORIES;

__global__ void q42_fused_kernel(
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

        // Probe customer (AMERICA)
        uint32_t ck = custkey[i];
        int c_slot = hash_murmur3(ck, ht_c_size);
        bool c_found = false;
        for (int p = 0; p < ht_c_size && !c_found; ++p) {
            int slot = (c_slot + p) % ht_c_size;
            if (ht_c_keys[slot] == ck) c_found = true;
            else if (ht_c_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!c_found) { selection_flags[i] = 0; continue; }

        // Probe supplier (AMERICA, get nation)
        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        uint32_t s_nation = 0;
        bool s_found = false;
        for (int p = 0; p < ht_s_size && !s_found; ++p) {
            int slot = (s_slot + p) % ht_s_size;
            if (ht_s_keys[slot] == sk) { s_nation = ht_s_values[slot]; s_found = true; }
            else if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) { selection_flags[i] = 0; continue; }

        // Probe part (MFGR#1 or #2, get category)
        uint32_t pk = partkey[i];
        int p_slot = hash_murmur3(pk, ht_p_size);
        uint32_t p_category = 0;
        bool p_found = false;
        for (int p = 0; p < ht_p_size && !p_found; ++p) {
            int slot = (p_slot + p) % ht_p_size;
            if (ht_p_keys[slot] == pk) { p_category = ht_p_values[slot]; p_found = true; }
            else if (ht_p_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!p_found) { selection_flags[i] = 0; continue; }

        int year_idx = d_year - 1997;
        if (year_idx >= 0 && year_idx < NUM_YEARS && s_nation < NUM_NATIONS && p_category < NUM_CATEGORIES) {
            int agg_idx = year_idx * NUM_NATIONS * NUM_CATEGORIES + s_nation * NUM_CATEGORIES + p_category;
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

__global__ void build_customer_region_only_ht(const uint32_t* c_custkey, const uint32_t* c_region, int n,
                                               uint32_t filter_region, uint32_t* ht_k, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (c_region[idx] != filter_region) return;
    uint32_t key = c_custkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

__global__ void build_supplier_region_nation_ht(const uint32_t* s_suppkey, const uint32_t* s_region, const uint32_t* s_nation,
                                                 int n, uint32_t filter_region, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx], val = s_nation[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_part_mfgr_category_ht(const uint32_t* p_partkey, const uint32_t* p_mfgr, const uint32_t* p_category,
                                             int n, uint32_t mfgr1, uint32_t mfgr2, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;
    uint32_t key = p_partkey[idx], val = p_category[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

void runQ42Fused(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    timer.start();
    int ht_d_size = D_LEN * 2, ht_c_size = C_LEN * 2, ht_s_size = S_LEN * 2, ht_p_size = P_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_c_keys, *ht_s_keys, *ht_s_values, *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t)); cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    int bs = 256;
    build_date_year_filter_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, 1997, 1998, ht_d_keys, ht_d_values, ht_d_size);
    build_customer_region_only_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_region, C_LEN, C_REGION_AMERICA, ht_c_keys, ht_c_size);
    build_supplier_region_nation_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_values, ht_s_size);
    build_part_mfgr_category_ht<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_mfgr, data.d_p_category, P_LEN, P_MFGR_1, P_MFGR_2, ht_p_keys, ht_p_values, ht_p_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    uint32_t *d_orderdate, *d_custkey, *d_suppkey, *d_partkey, *d_revenue, *d_supplycost;
    cudaMalloc(&d_orderdate, CHUNK_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_custkey, CHUNK_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_suppkey, CHUNK_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_partkey, CHUNK_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_revenue, CHUNK_SIZE * sizeof(uint32_t));
    cudaMalloc(&d_supplycost, CHUNK_SIZE * sizeof(uint32_t));

    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);
    CompressedColumnAccessorVertical<uint32_t> acc_supplycost(&data.lo_supplycost);

    timer.start();
    for (int offset = 0; offset < LO_LEN; offset += CHUNK_SIZE) {
        int chunk_size = min(CHUNK_SIZE, (int)(LO_LEN - offset));
        acc_orderdate.decompressRange(offset, offset + chunk_size, d_orderdate);
        acc_custkey.decompressRange(offset, offset + chunk_size, d_custkey);
        acc_suppkey.decompressRange(offset, offset + chunk_size, d_suppkey);
        acc_partkey.decompressRange(offset, offset + chunk_size, d_partkey);
        acc_revenue.decompressRange(offset, offset + chunk_size, d_revenue);
        acc_supplycost.decompressRange(offset, offset + chunk_size, d_supplycost);

        int grid_size = (chunk_size + TILE_SIZE - 1) / TILE_SIZE;
        q42_fused_kernel<<<grid_size, BLOCK_THREADS>>>(d_orderdate, d_custkey, d_suppkey, d_partkey, d_revenue, d_supplycost, chunk_size,
            ht_d_keys, ht_d_values, ht_d_size, ht_c_keys, ht_c_size, ht_s_keys, ht_s_values, ht_s_size, ht_p_keys, ht_p_values, ht_p_size, d_agg_profit);
    }
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.data_load_ms = 0;
    timing.total_ms = timing.hash_build_ms + timing.kernel_ms;

    std::vector<long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q4.2 Results (Fused) ===" << std::endl;
    long long total = 0; int groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) if (h_agg[i] != 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Total profit: " << total << std::endl;
    timing.print("Q4.2");

    cudaFree(d_orderdate); cudaFree(d_custkey); cudaFree(d_suppkey); cudaFree(d_partkey); cudaFree(d_revenue); cudaFree(d_supplycost);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_s_keys); cudaFree(ht_s_values); cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(d_agg_profit);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q4.2 - Fused Query Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ42Fused(data, timing);

    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ42Fused(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }

    data.free();
    return 0;
}
