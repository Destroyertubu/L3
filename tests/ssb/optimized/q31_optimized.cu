/**
 * @file q31_optimized.cu
 * @brief SSB Q3.1 - Optimized with Two-Level Fast Hash + Shared Memory Date Cache
 * Query: c_region='ASIA', s_region='ASIA', d_year 1992-1997
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

constexpr uint32_t C_REGION_ASIA = 2;
constexpr uint32_t S_REGION_ASIA = 2;
constexpr int NUM_YEARS = 6;
constexpr int NUM_NATIONS = 25;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS * NUM_NATIONS;

__global__ void probeSupplierQ31Opt(
    const uint32_t* __restrict__ suppkeys, int num_rows,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_values, int ht_s_size,
    int* __restrict__ passing_indices, uint32_t* __restrict__ passing_s_nations, int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8], s_warp_offsets[8], s_block_offset;
    bool found = false;
    uint32_t s_nation = 0;

    if (tid < num_rows) {
        uint32_t sk = suppkeys[tid];
        found = twoLevelProbeWithValue(sk, ht_s_keys, ht_s_values, ht_s_size, s_nation);
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);
    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) { s_warp_offsets[w] = total; total += s_warp_counts[w]; }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos = s_block_offset + s_warp_offsets[warp_id] + __popc(warp_ballot & mask);
        passing_indices[pos] = tid;
        passing_s_nations[pos] = s_nation;
    }
}

__global__ void probeCustomerQ31Opt(
    const uint32_t* __restrict__ custkeys, const int* __restrict__ input_indices,
    const uint32_t* __restrict__ input_s_nations, int num_rows,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_values, int ht_c_size,
    int* __restrict__ output_indices, uint32_t* __restrict__ output_c_nations,
    uint32_t* __restrict__ output_s_nations, int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8], s_warp_offsets[8], s_block_offset;
    bool found = false;
    uint32_t c_nation = 0, s_nation = 0;
    int orig_idx = -1;

    if (tid < num_rows) {
        uint32_t ck = custkeys[tid];
        orig_idx = input_indices[tid];
        s_nation = input_s_nations[tid];
        found = twoLevelProbeWithValue(ck, ht_c_keys, ht_c_values, ht_c_size, c_nation);
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);
    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) { s_warp_offsets[w] = total; total += s_warp_counts[w]; }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos = s_block_offset + s_warp_offsets[warp_id] + __popc(warp_ballot & mask);
        output_indices[pos] = orig_idx;
        output_c_nations[pos] = c_nation;
        output_s_nations[pos] = s_nation;
    }
}

__global__ void aggregateQ31Opt(
    const uint32_t* __restrict__ lo_orderdate, const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ c_nations, const uint32_t* __restrict__ s_nations, int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    unsigned long long* __restrict__ agg_revenue)
{
    __shared__ SharedDateCache s_date_cache;
    loadDateCacheCooperative(&s_date_cache, ht_d_keys, ht_d_values, ht_d_size);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t d_year = 0;
        if (probeSharedDateCache(lo_orderdate[i], &s_date_cache, ht_d_size, d_year)) {
            if (d_year >= 1992 && d_year <= 1997) {
                int year_idx = d_year - 1992;
                int agg_idx = year_idx * NUM_NATIONS * NUM_NATIONS + c_nations[i] * NUM_NATIONS + s_nations[i];
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(lo_revenue[i]));
            }
        }
    }
}

__global__ void build_date_ht_q31_opt(const uint32_t* d_datekey, const uint32_t* d_year, int n,
                                       uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    twoLevelInsert(d_datekey[idx], d_year[idx], ht_keys, ht_values, ht_size);
}

__global__ void build_customer_ht_q31_opt(const uint32_t* c_custkey, const uint32_t* c_region,
    const uint32_t* c_nation, int n, uint32_t filter_region, uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || c_region[idx] != filter_region) return;
    twoLevelInsert(c_custkey[idx], c_nation[idx], ht_keys, ht_values, ht_size);
}

__global__ void build_supplier_ht_q31_opt(const uint32_t* s_suppkey, const uint32_t* s_region,
    const uint32_t* s_nation, int n, uint32_t filter_region, uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n || s_region[idx] != filter_region) return;
    twoLevelInsert(s_suppkey[idx], s_nation[idx], ht_keys, ht_values, ht_size);
}

void runQ31Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_suppkey.getTotalElements();

    // Build hash tables
    timer.start();
    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    build_date_ht_q31_opt<<<(D_LEN+block_size-1)/block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_c_size = C_LEN * 2;
    uint32_t *ht_c_keys, *ht_c_values;
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    build_customer_ht_q31_opt<<<(C_LEN+block_size-1)/block_size, block_size>>>(
        data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, C_REGION_ASIA, ht_c_keys, ht_c_values, ht_c_size);

    int ht_s_size = S_LEN * 2;
    uint32_t *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    build_supplier_ht_q31_opt<<<(S_LEN+block_size-1)/block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, S_REGION_ASIA, ht_s_keys, ht_s_values, ht_s_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // Stage 1: Decompress suppkey + probe supplier
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, total_rows * sizeof(uint32_t));
    timer.start();
    acc_suppkey.decompressAll(d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decomp_supp_ms = timer.elapsed_ms();

    int *d_s1_idx, *d_num_s1;
    uint32_t* d_s1_snations;
    cudaMalloc(&d_s1_idx, total_rows * sizeof(int));
    cudaMalloc(&d_s1_snations, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_num_s1, sizeof(int));
    cudaMemset(d_num_s1, 0, sizeof(int));

    timer.start();
    probeSupplierQ31Opt<<<(total_rows+block_size-1)/block_size, block_size>>>(
        d_suppkey, total_rows, ht_s_keys, ht_s_values, ht_s_size, d_s1_idx, d_s1_snations, d_num_s1);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_s1_ms = timer.elapsed_ms();

    int h_num_s1;
    cudaMemcpy(&h_num_s1, d_num_s1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    std::cout << "Stage 1 (supplier): " << (100.0f * h_num_s1 / total_rows) << "%" << std::endl;

    if (h_num_s1 == 0) { timing.total_ms = decomp_supp_ms + probe_s1_ms; return; }

    // Stage 2: Random access custkey + probe customer
    uint32_t* d_custkey;
    cudaMalloc(&d_custkey, h_num_s1 * sizeof(uint32_t));
    timer.start();
    acc_custkey.randomAccessBatchIndices(d_s1_idx, h_num_s1, d_custkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_cust_ms = timer.elapsed_ms();

    int *d_s2_idx, *d_num_s2;
    uint32_t *d_s2_cnations, *d_s2_snations;
    cudaMalloc(&d_s2_idx, h_num_s1 * sizeof(int));
    cudaMalloc(&d_s2_cnations, h_num_s1 * sizeof(uint32_t));
    cudaMalloc(&d_s2_snations, h_num_s1 * sizeof(uint32_t));
    cudaMalloc(&d_num_s2, sizeof(int));
    cudaMemset(d_num_s2, 0, sizeof(int));

    timer.start();
    probeCustomerQ31Opt<<<(h_num_s1+block_size-1)/block_size, block_size>>>(
        d_custkey, d_s1_idx, d_s1_snations, h_num_s1, ht_c_keys, ht_c_values, ht_c_size,
        d_s2_idx, d_s2_cnations, d_s2_snations, d_num_s2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_s2_ms = timer.elapsed_ms();

    int h_num_s2;
    cudaMemcpy(&h_num_s2, d_num_s2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_custkey); cudaFree(d_s1_idx); cudaFree(d_s1_snations);
    std::cout << "Stage 2 (customer): " << (100.0f * h_num_s2 / total_rows) << "%" << std::endl;

    if (h_num_s2 == 0) { timing.total_ms = decomp_supp_ms + ra_cust_ms + probe_s1_ms + probe_s2_ms; return; }

    // Stage 3: Random access orderdate, revenue + aggregate
    uint32_t *d_orderdate, *d_revenue;
    cudaMalloc(&d_orderdate, h_num_s2 * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_s2 * sizeof(uint32_t));
    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_s2_idx, h_num_s2, d_orderdate, stream);
    acc_revenue.randomAccessBatchIndices(d_s2_idx, h_num_s2, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_final_ms = timer.elapsed_ms();

    unsigned long long* d_agg;
    cudaMalloc(&d_agg, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    aggregateQ31Opt<<<min((h_num_s2+block_size-1)/block_size, 256), block_size>>>(
        d_orderdate, d_revenue, d_s2_cnations, d_s2_snations, h_num_s2,
        ht_d_keys, ht_d_values, ht_d_size, d_agg);
    cudaDeviceSynchronize();
    timer.stop();
    float agg_ms = timer.elapsed_ms();

    std::vector<unsigned long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total = 0;
    int groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) { if (h_agg[i] > 0) { total += h_agg[i]; groups++; } }

    timing.data_load_ms = decomp_supp_ms + ra_cust_ms + ra_final_ms;
    timing.kernel_ms = probe_s1_ms + probe_s2_ms + agg_ms;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q3.1 Results (OPTIMIZED) ===" << std::endl;
    std::cout << "Groups: " << groups << ", Total revenue: " << total << std::endl;
    timing.print("Q3.1");

    cudaFree(d_s2_idx); cudaFree(d_s2_cnations); cudaFree(d_s2_snations);
    cudaFree(d_orderdate); cudaFree(d_revenue); cudaFree(d_agg);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_c_keys); cudaFree(ht_c_values);
    cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_num_s1); cudaFree(d_num_s2);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.1 - OPTIMIZED ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ31Optimized(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ31Optimized(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
