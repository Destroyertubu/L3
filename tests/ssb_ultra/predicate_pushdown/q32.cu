/**
 * @file q32.cu
 * @brief SSB Q3.2 Implementation - Predicate Pushdown Strategy
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


constexpr uint32_t C_NATION_US = 24;
constexpr uint32_t S_NATION_US = 24;
constexpr int NUM_YEARS = 6;
constexpr int NUM_CITIES = 250;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_CITIES;

__global__ void q32_predicate_kernel(
    const uint32_t* __restrict__ lo_orderdate, const uint32_t* __restrict__ lo_custkey,
    const uint32_t* __restrict__ lo_suppkey, const uint32_t* __restrict__ lo_revenue,
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

        uint32_t ck = custkey[i];
        int c_slot = hash_murmur3(ck, ht_c_size);
        uint32_t c_city = 0;
        bool c_found = false;
        for (int p = 0; p < ht_c_size && !c_found; ++p) {
            int slot = (c_slot + p) % ht_c_size;
            if (ht_c_keys[slot] == ck) { c_city = ht_c_values[slot]; c_found = true; }
            else if (ht_c_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!c_found) { selection_flags[i] = 0; continue; }

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

        uint32_t od = orderdate[i];
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool d_found = false;
        for (int p = 0; p < ht_d_size && !d_found; ++p) {
            int slot = (d_slot + p) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; d_found = true; }
            else if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!d_found || d_year < 1992 || d_year > 1997) { selection_flags[i] = 0; continue; }

        int year_idx = d_year - 1992;
        if (year_idx >= 0 && year_idx < NUM_YEARS && c_city < NUM_CITIES && s_city < NUM_CITIES) {
            int agg_idx = year_idx * NUM_CITIES * NUM_CITIES + c_city * NUM_CITIES + s_city;
            atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
        }
    }
}

__global__ void build_date_ht(const uint32_t* d_datekey, const uint32_t* d_year, int n, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
    uint32_t key = d_datekey[idx], val = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_customer_city_ht(const uint32_t* c_custkey, const uint32_t* c_nation, const uint32_t* c_city,
                                        int n, uint32_t filter_nation, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
    if (c_nation[idx] != filter_nation) return;
    uint32_t key = c_custkey[idx], val = c_city[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_supplier_city_ht(const uint32_t* s_suppkey, const uint32_t* s_nation, const uint32_t* s_city,
                                        int n, uint32_t filter_nation, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx >= n) return;
    if (s_nation[idx] != filter_nation) return;
    uint32_t key = s_suppkey[idx], val = s_city[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

void runQ32PredicatePushdown(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
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
    build_customer_city_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_nation, data.d_c_city, C_LEN, C_NATION_US, ht_c_keys, ht_c_values, ht_c_size);
    build_supplier_city_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN, S_NATION_US, ht_s_keys, ht_s_values, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    uint32_t *d_orderdate, *d_custkey, *d_suppkey, *d_revenue;
    cudaMalloc(&d_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_custkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_revenue, LO_LEN * sizeof(uint32_t));

    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    // Fast full decompression using warp-optimized kernel
    acc_orderdate.decompressAll(d_orderdate);
    acc_custkey.decompressAll(d_custkey);
    acc_suppkey.decompressAll(d_suppkey);
    acc_revenue.decompressAll(d_revenue);

    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q32_predicate_kernel<<<grid_size, BLOCK_THREADS>>>(d_orderdate, d_custkey, d_suppkey, d_revenue, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_c_keys, ht_c_values, ht_c_size, ht_s_keys, ht_s_values, ht_s_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.data_load_ms = 0;
    timing.total_ms = timing.hash_build_ms + timing.kernel_ms;

    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q3.2 Results (Predicate Pushdown) ===" << std::endl;
    unsigned long long total = 0; int groups = 0;
    for (size_t i = 0; i < AGG_SIZE; ++i) if (h_agg[i] > 0) { total += h_agg[i]; groups++; }
    std::cout << "Groups: " << groups << ", Total: " << total << std::endl;
    timing.print("Q3.2");

    cudaFree(d_orderdate); cudaFree(d_custkey); cudaFree(d_suppkey); cudaFree(d_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.2 - Predicate Pushdown Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ32PredicatePushdown(data, timing);
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ32PredicatePushdown(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
