/**
 * @file q31.cu
 * @brief SSB Q3.1 Implementation - Decompress First Strategy
 *
 * Query:
 *   SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, date
 *   WHERE lo_custkey = c_custkey
 *     AND lo_suppkey = s_suppkey
 *     AND lo_orderdate = d_datekey
 *     AND c_region = 'ASIA'
 *     AND s_region = 'ASIA'
 *     AND d_year >= 1992 AND d_year <= 1997
 *   GROUP BY c_nation, s_nation, d_year
 *   ORDER BY d_year ASC, revenue DESC;
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <algorithm>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t C_REGION_ASIA = 2;
constexpr uint32_t S_REGION_ASIA = 2;
constexpr int NUM_YEARS = 6;  // 1992-1997
constexpr int NUM_NATIONS = 25;  // Maximum nations per region
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS * NUM_NATIONS;

__global__ void q31_kernel(
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
    uint32_t orderdate[ITEMS_PER_THREAD];
    uint32_t custkey[ITEMS_PER_THREAD];
    uint32_t suppkey[ITEMS_PER_THREAD];
    uint32_t revenue[ITEMS_PER_THREAD];
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

        // Probe date (1992-1997)
        uint32_t od = orderdate[i];
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool d_found = false;
        for (int probe = 0; probe < ht_d_size && !d_found; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; d_found = true; }
            else if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!d_found || d_year < 1992 || d_year > 1997) { selection_flags[i] = 0; continue; }

        // Probe customer (c_region = ASIA, get c_nation)
        uint32_t ck = custkey[i];
        int c_slot = hash_murmur3(ck, ht_c_size);
        uint32_t c_nation = 0;
        bool c_found = false;
        for (int probe = 0; probe < ht_c_size && !c_found; ++probe) {
            int slot = (c_slot + probe) % ht_c_size;
            if (ht_c_keys[slot] == ck) { c_nation = ht_c_values[slot]; c_found = true; }
            else if (ht_c_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!c_found) { selection_flags[i] = 0; continue; }

        // Probe supplier (s_region = ASIA, get s_nation)
        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        uint32_t s_nation = 0;
        bool s_found = false;
        for (int probe = 0; probe < ht_s_size && !s_found; ++probe) {
            int slot = (s_slot + probe) % ht_s_size;
            if (ht_s_keys[slot] == sk) { s_nation = ht_s_values[slot]; s_found = true; }
            else if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) { selection_flags[i] = 0; continue; }

        int year_idx = d_year - 1992;
        if (year_idx >= 0 && year_idx < NUM_YEARS && c_nation < NUM_NATIONS && s_nation < NUM_NATIONS) {
            int agg_idx = year_idx * NUM_NATIONS * NUM_NATIONS + c_nation * NUM_NATIONS + s_nation;
            atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
        }
    }
}

__global__ void build_date_ht_all(const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t key = d_datekey[idx], value = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_values[s] = value; return; }
    }
}

__global__ void build_customer_ht(const uint32_t* c_custkey, const uint32_t* c_region, const uint32_t* c_nation,
                                   int num_rows, uint32_t filter_region, uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (c_region[idx] != filter_region) return;
    uint32_t key = c_custkey[idx], value = c_nation[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_values[s] = value; return; }
    }
}

__global__ void build_supplier_ht_nation(const uint32_t* s_suppkey, const uint32_t* s_region, const uint32_t* s_nation,
                                          int num_rows, uint32_t filter_region, uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx], value = s_nation[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_values[s] = value; return; }
    }
}

void runQ31DecompressFirst(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    uint32_t *d_lo_orderdate, *d_lo_custkey, *d_lo_suppkey, *d_lo_revenue;
    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_custkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(uint32_t));

    timer.start();
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);
    acc_orderdate.decompressAll(d_lo_orderdate);
    acc_custkey.decompressAll(d_lo_custkey);
    acc_suppkey.decompressAll(d_lo_suppkey);
    acc_revenue.decompressAll(d_lo_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    timer.start();
    int ht_d_size = D_LEN * 2, ht_c_size = C_LEN * 2, ht_s_size = S_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_c_keys, *ht_c_values, *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    int bs = 256;
    build_date_ht_all<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_customer_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, C_REGION_ASIA, ht_c_keys, ht_c_values, ht_c_size);
    build_supplier_ht_nation<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, S_REGION_ASIA, ht_s_keys, ht_s_values, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q31_kernel<<<grid_size, BLOCK_THREADS>>>(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_c_keys, ht_c_values, ht_c_size, ht_s_keys, ht_s_values, ht_s_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::vector<unsigned long long> h_agg_revenue(AGG_SIZE);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q3.1 Results (Decompress-First) ===" << std::endl;
    unsigned long long total = 0;
    int num_groups = 0;
    for (int y = 0; y < NUM_YEARS; ++y) {
        for (int cn = 0; cn < NUM_NATIONS; ++cn) {
            for (int sn = 0; sn < NUM_NATIONS; ++sn) {
                int idx = y * NUM_NATIONS * NUM_NATIONS + cn * NUM_NATIONS + sn;
                if (h_agg_revenue[idx] > 0) {
                    total += h_agg_revenue[idx];
                    num_groups++;
                }
            }
        }
    }
    std::cout << "Total groups: " << num_groups << ", Total revenue: " << total << std::endl;
    timing.print("Q3.1");

    cudaFree(d_lo_orderdate); cudaFree(d_lo_custkey); cudaFree(d_lo_suppkey); cudaFree(d_lo_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.1 - Decompress First Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ31DecompressFirst(data, timing);

    std::cout << "\n=== Benchmark (3 runs) ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ31DecompressFirst(data, t);
        std::cout << "Run " << (i+1) << ": Total=" << t.total_ms << " ms" << std::endl;
    }

    data.free();
    return 0;
}
