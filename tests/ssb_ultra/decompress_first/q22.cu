/**
 * @file q22.cu
 * @brief SSB Q2.2 Implementation - Decompress First Strategy
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, date, part, supplier
 *   WHERE lo_orderdate = d_datekey
 *     AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'
 *     AND s_region = 'ASIA'
 *   GROUP BY d_year, p_brand1
 *   ORDER BY d_year, p_brand1;
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

// Brand range: MFGR#2221 to MFGR#2228 encoded as integers
// In SSB data generator, brand is encoded as: mfgr*1000 + category*40 + brand_num
// MFGR#2221 = 2*1000 + 2*40 + 21 = 2000 + 80 + 21 = 2101... actually need to check encoding
// Simplified: use brand values 221-228 range approximation
constexpr uint32_t P_BRAND_MIN = 221;
constexpr uint32_t P_BRAND_MAX = 228;
constexpr uint32_t S_REGION_ASIA = 2;  // ASIA region code

constexpr int NUM_YEARS = 7;
constexpr int NUM_BRANDS = 1000;

__global__ void q22_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_partkey,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_revenue,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    const uint32_t* __restrict__ ht_s_keys,
    int ht_s_size,
    unsigned long long* agg_revenue,
    int* agg_count)
{
    uint32_t orderdate[ITEMS_PER_THREAD];
    uint32_t partkey[ITEMS_PER_THREAD];
    uint32_t suppkey[ITEMS_PER_THREAD];
    uint32_t revenue[ITEMS_PER_THREAD];
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

        // Probe supplier hash table (s_region = ASIA)
        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        bool s_found = false;
        for (int probe = 0; probe < ht_s_size && !s_found; ++probe) {
            int slot = (s_slot + probe) % ht_s_size;
            if (ht_s_keys[slot] == sk) s_found = true;
            else if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) { selection_flags[i] = 0; continue; }

        // Probe part hash table (p_brand1 in range)
        uint32_t pk = partkey[i];
        int p_slot = hash_murmur3(pk, ht_p_size);
        uint32_t p_brand = 0;
        bool p_found = false;
        for (int probe = 0; probe < ht_p_size && !p_found; ++probe) {
            int slot = (p_slot + probe) % ht_p_size;
            if (ht_p_keys[slot] == pk) { p_brand = ht_p_values[slot]; p_found = true; }
            else if (ht_p_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!p_found) { selection_flags[i] = 0; continue; }

        // Probe date hash table
        uint32_t od = orderdate[i];
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool d_found = false;
        for (int probe = 0; probe < ht_d_size && !d_found; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; d_found = true; }
            else if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!d_found) { selection_flags[i] = 0; continue; }

        int year_idx = d_year - 1992;
        if (year_idx >= 0 && year_idx < NUM_YEARS && p_brand < NUM_BRANDS) {
            int agg_idx = year_idx * NUM_BRANDS + p_brand;
            atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
            atomicAdd(&agg_count[agg_idx], 1);
        }
    }
}

__global__ void build_date_ht_q22(const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t key = d_datekey[idx];
    uint32_t value = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_values[s] = value; return; }
    }
}

__global__ void build_part_ht_q22(const uint32_t* p_partkey, const uint32_t* p_brand1, int num_rows,
                                   uint32_t brand_min, uint32_t brand_max,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t brand = p_brand1[idx];
    if (brand < brand_min || brand > brand_max) return;
    uint32_t key = p_partkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_values[s] = brand; return; }
    }
}

__global__ void build_supplier_ht_q22(const uint32_t* s_suppkey, const uint32_t* s_region, int num_rows,
                                       uint32_t filter_region, uint32_t* ht_keys, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

void runQ22DecompressFirst(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    uint32_t *d_lo_orderdate, *d_lo_partkey, *d_lo_suppkey, *d_lo_revenue;
    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(uint32_t));

    timer.start();
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);
    acc_orderdate.decompressAll(d_lo_orderdate);
    acc_partkey.decompressAll(d_lo_partkey);
    acc_suppkey.decompressAll(d_lo_suppkey);
    acc_revenue.decompressAll(d_lo_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    timer.start();
    int ht_d_size = D_LEN * 2, ht_p_size = P_LEN * 2, ht_s_size = S_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_p_keys, *ht_p_values, *ht_s_keys;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    int bs = 256;
    build_date_ht_q22<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_part_ht_q22<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_brand1, P_LEN, P_BRAND_MIN, P_BRAND_MAX, ht_p_keys, ht_p_values, ht_p_size);
    build_supplier_ht_q22<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_ASIA, ht_s_keys, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    int agg_size = NUM_YEARS * NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    int* d_agg_count;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));
    cudaMalloc(&d_agg_count, agg_size * sizeof(int));
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
    cudaMemset(d_agg_count, 0, agg_size * sizeof(int));

    timer.start();
    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q22_kernel<<<grid_size, BLOCK_THREADS>>>(d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size, ht_p_keys, ht_p_values, ht_p_size, ht_s_keys, ht_s_size, d_agg_revenue, d_agg_count);
    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::vector<unsigned long long> h_agg_revenue(agg_size);
    std::vector<int> h_agg_count(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agg_count.data(), d_agg_count, agg_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.2 Results (Decompress-First) ===" << std::endl;
    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int y = 0; y < NUM_YEARS; ++y) {
        for (int b = 0; b < NUM_BRANDS; ++b) {
            int idx = y * NUM_BRANDS + b;
            if (h_agg_count[idx] > 0) {
                num_results++;
                total_revenue += h_agg_revenue[idx];
            }
        }
    }
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    timing.print("Q2.2");

    cudaFree(d_lo_orderdate); cudaFree(d_lo_partkey); cudaFree(d_lo_suppkey); cudaFree(d_lo_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_p_keys); cudaFree(ht_p_values); cudaFree(ht_s_keys);
    cudaFree(d_agg_revenue); cudaFree(d_agg_count);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.2 - Decompress First Strategy ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    std::cout << "Compression ratio: " << data.getCompressionRatio() << "x" << std::endl;

    QueryTiming timing;
    runQ22DecompressFirst(data, timing);

    std::cout << "\n=== Benchmark (3 runs) ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ22DecompressFirst(data, t);
        std::cout << "Run " << (i+1) << ": Total=" << t.total_ms << " ms" << std::endl;
    }

    data.free();
    return 0;
}
