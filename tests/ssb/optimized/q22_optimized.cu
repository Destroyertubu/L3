/**
 * @file q22_optimized.cu
 * @brief SSB Q2.2 Implementation - Optimized with Two-Level Fast Hash
 * Query: s_region = 'ASIA', p_brand between 'MFGR#2221' and 'MFGR#2228'
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "ssb_filter_kernels.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t P_BRAND_MIN = 221;
constexpr uint32_t P_BRAND_MAX = 228;
constexpr uint32_t S_REGION_ASIA = 2;
constexpr int NUM_YEARS = 7;
constexpr int NUM_BRANDS = 1000;

__global__ void probeSingleFilterKernelQ22(
    const uint32_t* __restrict__ keys, int num_rows,
    const uint32_t* __restrict__ ht_keys, int ht_size,
    int* __restrict__ passing_indices, int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    bool found = false;
    if (tid < num_rows) {
        uint32_t key = keys[tid];
        int slot;
        found = twoLevelProbe(key, ht_keys, ht_size, slot);
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
    }
}

__global__ void probePartWithBrandRangeQ22(
    const uint32_t* __restrict__ partkeys, const int* __restrict__ input_indices, int num_rows,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_values, int ht_p_size,
    int* __restrict__ output_indices, uint32_t* __restrict__ output_brands, int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    bool found = false;
    uint32_t brand = 0;
    int original_idx = -1;

    if (tid < num_rows) {
        uint32_t pk = partkeys[tid];
        original_idx = input_indices[tid];
        if (twoLevelProbeWithValue(pk, ht_p_keys, ht_p_values, ht_p_size, brand)) {
            found = (brand >= P_BRAND_MIN && brand <= P_BRAND_MAX);
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(num_passing, total);
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        output_indices[global_pos] = original_idx;
        output_brands[global_pos] = brand;
    }
}

__global__ void aggregateQ22Kernel(
    const uint32_t* __restrict__ lo_orderdate, const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ passing_brands, int num_rows,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    unsigned long long* __restrict__ agg_revenue)
{
    __shared__ SharedDateCache s_date_cache;
    loadDateCacheCooperative(&s_date_cache, ht_d_keys, ht_d_values, ht_d_size);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < num_rows; i += blockDim.x * gridDim.x) {
        uint32_t od = lo_orderdate[i];
        uint32_t rev = lo_revenue[i];
        uint32_t brand = passing_brands[i];
        uint32_t d_year = 0;
        if (probeSharedDateCache(od, &s_date_cache, ht_d_size, d_year)) {
            int year_idx = d_year - 1992;
            if (year_idx >= 0 && year_idx < NUM_YEARS && brand < NUM_BRANDS) {
                int agg_idx = year_idx * NUM_BRANDS + brand;
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(rev));
            }
        }
    }
}

__global__ void build_date_ht_q22(const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    twoLevelInsert(d_datekey[idx], d_year[idx], ht_keys, ht_values, ht_size);
}

__global__ void build_part_ht_q22(const uint32_t* p_partkey, const uint32_t* p_brand1, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    twoLevelInsert(p_partkey[idx], p_brand1[idx], ht_keys, ht_values, ht_size);
}

__global__ void build_supplier_region_ht_q22(const uint32_t* s_suppkey, const uint32_t* s_region,
                                              int num_rows, uint32_t filter_region,
                                              uint32_t* ht_keys, int ht_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx];
    int slot = hash_fast(key, ht_size);
    uint32_t old = atomicCAS(&ht_keys[slot], static_cast<uint32_t>(HT_EMPTY), key);
    if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    slot = hash_murmur3(static_cast<int>(key), ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) return;
    }
}

void runQ22Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_partkey.getTotalElements();

    timer.start();
    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    build_date_ht_q22<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_p_size = P_LEN * 2;
    uint32_t *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));
    build_part_ht_q22<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_brand1, P_LEN, ht_p_keys, ht_p_values, ht_p_size);

    int ht_s_size = S_LEN * 2;
    uint32_t* ht_s_keys;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    build_supplier_region_ht_q22<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_ASIA, ht_s_keys, ht_s_size);

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
    float decompress_suppkey_ms = timer.elapsed_ms();

    int *d_stage1_indices, *d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probeSingleFilterKernelQ22<<<grid_stage1, block_size>>>(d_suppkey, total_rows, ht_s_keys, ht_s_size, d_stage1_indices, d_num_stage1);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage1_ms = timer.elapsed_ms();

    int h_num_stage1;
    cudaMemcpy(&h_num_stage1, d_num_stage1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    cudaFree(d_num_stage1);

    std::cout << "Stage 1 (supplier): " << (100.0f * h_num_stage1 / total_rows) << "% (" << h_num_stage1 << " rows)" << std::endl;

    if (h_num_stage1 == 0) {
        std::cout << "\n=== Q2.2 Results (OPTIMIZED) ===\nTotal revenue: 0\n";
        cudaFree(d_stage1_indices);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        cudaFree(ht_s_keys);
        return;
    }

    // Stage 2: Random access partkey, probe part with brand range
    uint32_t* d_partkey;
    cudaMalloc(&d_partkey, h_num_stage1 * sizeof(uint32_t));
    timer.start();
    acc_partkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_partkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_partkey_ms = timer.elapsed_ms();

    int *d_stage2_indices, *d_num_stage2;
    uint32_t* d_stage2_brands;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_stage2_brands, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probePartWithBrandRangeQ22<<<grid_stage2, block_size>>>(d_partkey, d_stage1_indices, h_num_stage1,
        ht_p_keys, ht_p_values, ht_p_size, d_stage2_indices, d_stage2_brands, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_partkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_num_stage2);

    std::cout << "Stage 2 (part brand): " << (100.0f * h_num_stage2 / total_rows) << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q2.2 Results (OPTIMIZED) ===\nTotal revenue: 0\n";
        cudaFree(d_stage2_indices); cudaFree(d_stage2_brands);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        cudaFree(ht_s_keys);
        return;
    }

    // Stage 3: Random access final columns, aggregate
    uint32_t *d_orderdate, *d_revenue;
    cudaMalloc(&d_orderdate, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_stage2 * sizeof(uint32_t));
    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_orderdate, stream);
    acc_revenue.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_final_ms = timer.elapsed_ms();

    int agg_size = NUM_YEARS * NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

    timer.start();
    int grid_agg = min((h_num_stage2 + block_size - 1) / block_size, 256);
    aggregateQ22Kernel<<<grid_agg, block_size>>>(d_orderdate, d_revenue, d_stage2_brands, h_num_stage2,
        ht_d_keys, ht_d_values, ht_d_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    std::vector<unsigned long long> h_agg_revenue(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total_revenue = 0;
    int num_results = 0;
    for (int i = 0; i < agg_size; ++i) {
        if (h_agg_revenue[i] > 0) { total_revenue += h_agg_revenue[i]; num_results++; }
    }

    timing.data_load_ms = decompress_suppkey_ms + ra_partkey_ms + ra_final_ms;
    timing.kernel_ms = probe_stage1_ms + probe_stage2_ms + aggregate_ms;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q2.2 Results (OPTIMIZED) ===" << std::endl;
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    timing.print("Q2.2");

    cudaFree(d_stage2_indices); cudaFree(d_stage2_brands);
    cudaFree(d_orderdate); cudaFree(d_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(ht_s_keys);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.2 - OPTIMIZED ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ22Optimized(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ22Optimized(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
