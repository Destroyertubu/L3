/**
 * @file q32.cu
 * @brief SSB Q3.2 Implementation - Random Access Optimization Strategy
 *
 * Query:
 *   SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey AND lo_orderdate = d_datekey
 *     AND c_nation = 'UNITED STATES' AND s_nation = 'UNITED STATES'
 *     AND d_year >= 1992 AND d_year <= 1997
 *   GROUP BY c_city, s_city, d_year
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "ssb_filter_kernels.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t C_NATION_US = 24;
constexpr uint32_t S_NATION_US = 24;
constexpr int NUM_YEARS = 6;
constexpr int NUM_CITIES = 250;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_CITIES;

__global__ void probeJoinFilterQ32Kernel(
    const uint32_t* __restrict__ lo_custkey,
    const uint32_t* __restrict__ lo_suppkey,
    int num_rows,
    const uint32_t* __restrict__ ht_c_keys,
    const uint32_t* __restrict__ ht_c_values,
    int ht_c_size,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    int* __restrict__ passing_indices,
    uint32_t* __restrict__ passing_c_cities,
    uint32_t* __restrict__ passing_s_cities,
    int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    int passes = 0;
    uint32_t c_city = 0, s_city = 0;

    if (tid < num_rows) {
        uint32_t ck = lo_custkey[tid];
        uint32_t sk = lo_suppkey[tid];

        int c_slot = hash_murmur3(ck, ht_c_size);
        bool c_found = false;
        for (int probe = 0; probe < 32; ++probe) {
            int slot = (c_slot + probe) % ht_c_size;
            if (ht_c_keys[slot] == ck) { c_city = ht_c_values[slot]; c_found = true; break; }
            if (ht_c_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }

        if (c_found) {
            int s_slot = hash_murmur3(sk, ht_s_size);
            for (int probe = 0; probe < 32; ++probe) {
                int slot = (s_slot + probe) % ht_s_size;
                if (ht_s_keys[slot] == sk) { s_city = ht_s_values[slot]; passes = 1; break; }
                if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
            }
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) { s_warp_offsets[w] = total; total += s_warp_counts[w]; }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (passes && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
        passing_c_cities[global_pos] = c_city;
        passing_s_cities[global_pos] = s_city;
    }
}

__global__ void aggregateQ32Kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ passing_c_cities,
    const uint32_t* __restrict__ passing_s_cities,
    int num_rows,
    const uint32_t* __restrict__ ht_d_keys,
    const uint32_t* __restrict__ ht_d_values,
    int ht_d_size,
    unsigned long long* __restrict__ agg_revenue)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t od = lo_orderdate[i];
        uint32_t rev = lo_revenue[i];
        uint32_t c_city = passing_c_cities[i];
        uint32_t s_city = passing_s_cities[i];

        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool found = false;
        for (int probe = 0; probe < 32; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) { d_year = ht_d_values[slot]; found = true; break; }
            if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }

        if (found && d_year >= 1992 && d_year <= 1997) {
            int year_idx = d_year - 1992;
            if (c_city < NUM_CITIES && s_city < NUM_CITIES) {
                int agg_idx = year_idx * NUM_CITIES * NUM_CITIES + c_city * NUM_CITIES + s_city;
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(rev));
            }
        }
    }
}

__global__ void build_date_ht_q32(const uint32_t* d_datekey, const uint32_t* d_year, int n, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
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

__global__ void build_customer_city_ht_q32(const uint32_t* c_custkey, const uint32_t* c_nation, const uint32_t* c_city,
                                            int n, uint32_t filter_nation, uint32_t* ht_k, uint32_t* ht_v, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (c_nation[idx] != filter_nation) return;
    uint32_t key = c_custkey[idx], val = c_city[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int p = 0; p < ht_size; ++p) {
        int s = (slot + p) % ht_size;
        uint32_t old = atomicCAS(&ht_k[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) { ht_v[s] = val; return; }
    }
}

__global__ void build_supplier_city_ht_q32(const uint32_t* s_suppkey, const uint32_t* s_nation, const uint32_t* s_city,
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

void runQ32RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_custkey.getTotalElements();

    timer.start();
    int ht_d_size = D_LEN * 2, ht_c_size = C_LEN * 2, ht_s_size = S_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values, *ht_c_keys, *ht_c_values, *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t)); cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t)); cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t)); cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_date_ht_q32<<<(D_LEN+255)/256, 256>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);
    build_customer_city_ht_q32<<<(C_LEN+255)/256, 256>>>(data.d_c_custkey, data.d_c_nation, data.d_c_city, C_LEN, C_NATION_US, ht_c_keys, ht_c_values, ht_c_size);
    build_supplier_city_ht_q32<<<(S_LEN+255)/256, 256>>>(data.d_s_suppkey, data.d_s_nation, data.d_s_city, S_LEN, S_NATION_US, ht_s_keys, ht_s_values, ht_s_size);
    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    uint32_t *d_custkey, *d_suppkey;
    cudaMalloc(&d_custkey, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_suppkey, total_rows * sizeof(uint32_t));
    timer.start();
    acc_custkey.decompressAll(d_custkey, stream);
    acc_suppkey.decompressAll(d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_keys_ms = timer.elapsed_ms();

    int *d_passing_indices, *d_num_passing;
    uint32_t *d_passing_c_cities, *d_passing_s_cities;
    cudaMalloc(&d_passing_indices, total_rows * sizeof(int));
    cudaMalloc(&d_passing_c_cities, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_passing_s_cities, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_num_passing, sizeof(int));
    cudaMemset(d_num_passing, 0, sizeof(int));

    timer.start();
    probeJoinFilterQ32Kernel<<<(total_rows+255)/256, 256>>>(d_custkey, d_suppkey, total_rows,
        ht_c_keys, ht_c_values, ht_c_size, ht_s_keys, ht_s_values, ht_s_size,
        d_passing_indices, d_passing_c_cities, d_passing_s_cities, d_num_passing);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_ms = timer.elapsed_ms();

    int h_num_passing;
    cudaMemcpy(&h_num_passing, d_num_passing, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Selectivity: " << (100.0f * h_num_passing / total_rows) << "%" << std::endl;
    cudaFree(d_custkey); cudaFree(d_suppkey);

    if (h_num_passing == 0) {
        timing.data_load_ms = decompress_keys_ms; timing.kernel_ms = probe_ms;
        timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;
        std::cout << "\n=== Q3.2 Results (Random Access) ===\nTotal: 0\n";
        cudaFree(d_passing_indices); cudaFree(d_passing_c_cities); cudaFree(d_passing_s_cities); cudaFree(d_num_passing);
        cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
        return;
    }

    uint32_t *d_orderdate, *d_revenue;
    cudaMalloc(&d_orderdate, h_num_passing * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_passing * sizeof(uint32_t));
    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_orderdate, stream);
    acc_revenue.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_ms = timer.elapsed_ms();

    unsigned long long* d_agg;
    cudaMalloc(&d_agg, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg, 0, AGG_SIZE * sizeof(unsigned long long));
    timer.start();
    aggregateQ32Kernel<<<min((h_num_passing+255)/256, 256), 256>>>(d_orderdate, d_revenue, d_passing_c_cities, d_passing_s_cities,
        h_num_passing, ht_d_keys, ht_d_values, ht_d_size, d_agg);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    unsigned long long total = 0; int groups = 0;
    for (size_t i = 0; i < AGG_SIZE; ++i) if (h_agg[i] > 0) { total += h_agg[i]; groups++; }

    timing.data_load_ms = decompress_keys_ms + random_access_ms;
    timing.kernel_ms = probe_ms + aggregate_ms;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q3.2 Results (Random Access) ===" << std::endl;
    std::cout << "Groups: " << groups << ", Total: " << total << std::endl;
    timing.print("Q3.2");

    cudaFree(d_passing_indices); cudaFree(d_passing_c_cities); cudaFree(d_passing_s_cities); cudaFree(d_num_passing);
    cudaFree(d_orderdate); cudaFree(d_revenue); cudaFree(d_agg);
    cudaFree(ht_d_keys); cudaFree(ht_d_values); cudaFree(ht_c_keys); cudaFree(ht_c_values); cudaFree(ht_s_keys); cudaFree(ht_s_values);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];
    std::cout << "=== SSB Q3.2 - Random Access ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);
    QueryTiming timing;
    runQ32RandomAccess(data, timing);
    for (int i = 0; i < 3; ++i) { QueryTiming t; runQ32RandomAccess(data, t); std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n"; }
    data.free();
    return 0;
}
