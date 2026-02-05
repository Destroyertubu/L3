/**
 * @file q21.cu
 * @brief SSB Q2.1 Implementation - Predicate Pushdown Strategy
 *
 * Uses partition min/max on orderdate to skip irrelevant partitions.
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

constexpr uint32_t S_REGION_AMERICA = 1;
constexpr uint32_t P_CATEGORY_MFGR12 = 12;
constexpr int NUM_YEARS = 7;
constexpr int NUM_BRANDS = 1000;
constexpr int AGG_SIZE = NUM_YEARS * NUM_BRANDS;

__global__ void q21_predicate_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_partkey,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_revenue,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    const uint32_t* __restrict__ ht_s_keys, int ht_s_size,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_values, int ht_p_size,
    unsigned long long* agg_revenue)
{
    uint32_t orderdate[ITEMS_PER_THREAD], partkey[ITEMS_PER_THREAD];
    uint32_t suppkey[ITEMS_PER_THREAD], revenue[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);
    if (block_items <= 0) return;

    InitFlags(selection_flags, block_items);
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockLoad(lo_partkey + block_start, partkey, block_items);
    BlockLoad(lo_suppkey + block_start, suppkey, block_items);
    BlockLoad(lo_revenue + block_start, revenue, block_items);

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx >= block_items || !selection_flags[i]) continue;

        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        bool s_found = false;
        for (int p = 0; p < ht_s_size && !s_found; ++p) {
            int slot = (s_slot + p) % ht_s_size;
            if (ht_s_keys[slot] == sk) s_found = true;
            else if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) { selection_flags[i] = 0; continue; }

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

        int year_idx = d_year - 1992;
        if (year_idx >= 0 && year_idx < NUM_YEARS && p_brand < NUM_BRANDS) {
            int agg_idx = year_idx * NUM_BRANDS + p_brand;
            atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
        }
    }
}

__global__ void build_date_ht(const uint32_t* d_datekey, const uint32_t* d_year, int n, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t key = d_datekey[idx], val = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_supplier_region_ht(const uint32_t* s_suppkey, const uint32_t* s_region, int n, uint32_t filter_region, uint32_t* ht_k, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

__global__ void build_part_category_ht(const uint32_t* p_partkey, const uint32_t* p_category, const uint32_t* p_brand1,
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

void runQ21PredicatePushdown(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    // Build hash tables
    timer.start();
    int ht_d_size = D_LEN * 2, ht_s_size = S_LEN * 2, ht_p_size = P_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_s_keys, *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t)); cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    int bs = 256;
    build_date_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_supplier_region_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_size);
    build_part_category_ht<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN, P_CATEGORY_MFGR12, ht_p_keys, ht_p_values, ht_p_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // Use fast warp-optimized decompression for all data (no partition pruning for Q2.1)
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    uint32_t *d_orderdate, *d_partkey, *d_suppkey, *d_revenue;
    cudaMalloc(&d_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_partkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_revenue, LO_LEN * sizeof(uint32_t));

    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    // Fast full decompression using warp-optimized kernel
    acc_orderdate.decompressAll(d_orderdate);
    acc_partkey.decompressAll(d_partkey);
    acc_suppkey.decompressAll(d_suppkey);
    acc_revenue.decompressAll(d_revenue);

    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q21_predicate_kernel<<<grid_size, BLOCK_THREADS>>>(d_orderdate, d_partkey, d_suppkey, d_revenue, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_s_keys, ht_s_size, ht_p_keys, ht_p_values, ht_p_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.data_load_ms = 0;
    timing.total_ms = timing.hash_build_ms + timing.kernel_ms;

    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 Results (Predicate Pushdown) ===" << std::endl;
    unsigned long long total = 0; int groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) if (h_agg[i] > 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Total: " << total << std::endl;
    timing.print("Q2.1");

    cudaFree(d_orderdate); cudaFree(d_partkey); cudaFree(d_suppkey); cudaFree(d_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_s_keys); cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.1 - Predicate Pushdown Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ21PredicatePushdown(data, timing);
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ21PredicatePushdown(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
