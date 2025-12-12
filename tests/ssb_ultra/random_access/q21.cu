/**
 * @file q21.cu
 * @brief SSB Q2.1 Implementation - Random Access Optimization Strategy
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, date, part, supplier
 *   WHERE lo_orderdate = d_datekey AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_category = 'MFGR#12' AND s_region = 'AMERICA'
 *   GROUP BY d_year, p_brand1
 *
 * Strategy:
 *   1. Build hash tables for dimension tables (pre-filtered)
 *   2. Full decompress JOIN key columns (lo_partkey, lo_suppkey)
 *   3. Probe hash tables → passing_indices
 *   4. Random access non-JOIN columns (lo_orderdate, lo_revenue)
 *   5. Final aggregation with date lookup
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

constexpr uint32_t P_CATEGORY_MFGR12 = 12;
constexpr uint32_t S_REGION_AMERICA = 1;
constexpr int NUM_YEARS = 7;
constexpr int NUM_BRANDS = 1000;

/**
 * @brief Stage 1: Single hash probe filter - probes supplier hash table only
 * Outputs passing row indices for stage 2
 */
__global__ void probeSingleFilterKernel(
    const uint32_t* __restrict__ keys,
    int num_rows,
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    int* __restrict__ passing_indices,
    int* __restrict__ num_passing)
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
        int slot = hash_murmur3(key, ht_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_size;
            if (ht_keys[s] == key) { found = true; break; }
            if (ht_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    if (found && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = tid;
    }
}

/**
 * @brief Stage 2: Probe part hash table with input indices
 * Takes partkeys already fetched via random access, outputs original indices and brands
 */
__global__ void probePartWithIndicesKernel(
    const uint32_t* __restrict__ partkeys,       // random-accessed partkeys
    const int* __restrict__ input_indices,        // original row indices from stage 1
    int num_rows,                                 // count from stage 1 (e.g., 24M)
    const uint32_t* __restrict__ ht_p_keys,
    const uint32_t* __restrict__ ht_p_values,
    int ht_p_size,
    int* __restrict__ output_indices,             // passing original indices
    uint32_t* __restrict__ output_brands,         // passing brands
    int* __restrict__ num_passing)
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

        int slot = hash_murmur3(pk, ht_p_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_p_size;
            if (ht_p_keys[s] == pk) {
                brand = ht_p_values[s];
                found = true;
                break;
            }
            if (ht_p_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, found);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
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

/**
 * @brief Final aggregation with date hash lookup
 */
__global__ void aggregateQ21Kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ passing_brands,
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
        uint32_t brand = passing_brands[i];

        // Probe date hash table
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool found = false;
        for (int probe = 0; probe < 32; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) {
                d_year = ht_d_values[slot];
                found = true;
                break;
            }
            if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }

        if (found) {
            int year_idx = d_year - 1992;
            if (year_idx >= 0 && year_idx < NUM_YEARS && brand < NUM_BRANDS) {
                int agg_idx = year_idx * NUM_BRANDS + brand;
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(rev));
            }
        }
    }
}

// Hash table build kernels (same as decompress_first)
__global__ void build_date_ht_kernel(
    const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
    uint32_t* ht_keys, uint32_t* ht_values, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    uint32_t key = d_datekey[idx];
    uint32_t value = d_year[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

__global__ void build_part_ht_kernel(
    const uint32_t* p_partkey, const uint32_t* p_category, const uint32_t* p_brand1,
    int num_rows, uint32_t filter_category,
    uint32_t* ht_keys, uint32_t* ht_values, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (p_category[idx] != filter_category) return;
    uint32_t key = p_partkey[idx];
    uint32_t value = p_brand1[idx];
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

__global__ void build_supplier_ht_kernel(
    const uint32_t* s_suppkey, const uint32_t* s_region, int num_rows,
    uint32_t filter_region, uint32_t* ht_keys, int ht_size)
{
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

void runQ21RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_partkey.getTotalElements();

    // Step 1: Build dimension hash tables (same as before)
    timer.start();

    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));

    build_date_ht_kernel<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_p_size = P_LEN * 2;
    uint32_t *ht_p_keys, *ht_p_values;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    build_part_ht_kernel<<<(P_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        P_CATEGORY_MFGR12, ht_p_keys, ht_p_values, ht_p_size);

    int ht_s_size = S_LEN * 2;
    uint32_t* ht_s_keys;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_supplier_ht_kernel<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== STAGED DECOMPOSITION ==========

    // === STAGE 1: Decompress suppkey + probe supplier (most selective first) ===
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, total_rows * sizeof(uint32_t));

    timer.start();
    acc_suppkey.decompressAll(d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_suppkey_ms = timer.elapsed_ms();

    int* d_stage1_indices;
    int* d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probeSingleFilterKernel<<<grid_stage1, block_size>>>(
        d_suppkey, total_rows,
        ht_s_keys, ht_s_size,
        d_stage1_indices, d_num_stage1);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage1_ms = timer.elapsed_ms();

    int h_num_stage1;
    cudaMemcpy(&h_num_stage1, d_num_stage1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_suppkey);
    cudaFree(d_num_stage1);

    float stage1_selectivity = 100.0f * h_num_stage1 / total_rows;
    std::cout << "Stage 1 (supplier filter): " << stage1_selectivity << "% (" << h_num_stage1 << " rows)" << std::endl;

    if (h_num_stage1 == 0) {
        std::cout << "\n=== Q2.1 Results (Staged Random Access) ===" << std::endl;
        std::cout << "Total revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_stage1_indices);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        cudaFree(ht_s_keys);
        return;
    }

    // === STAGE 2: Random access partkey for ONLY stage1 rows, probe part ===
    uint32_t* d_partkey;
    cudaMalloc(&d_partkey, h_num_stage1 * sizeof(uint32_t));

    timer.start();
    acc_partkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_partkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_partkey_ms = timer.elapsed_ms();

    int* d_stage2_indices;
    uint32_t* d_stage2_brands;
    int* d_num_stage2;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_stage2_brands, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probePartWithIndicesKernel<<<grid_stage2, block_size>>>(
        d_partkey, d_stage1_indices, h_num_stage1,
        ht_p_keys, ht_p_values, ht_p_size,
        d_stage2_indices, d_stage2_brands, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_partkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_num_stage2);

    float stage2_selectivity = 100.0f * h_num_stage2 / total_rows;
    std::cout << "Stage 2 (part filter):     " << stage2_selectivity << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q2.1 Results (Staged Random Access) ===" << std::endl;
        std::cout << "Total revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_stage2_indices); cudaFree(d_stage2_brands);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_p_keys); cudaFree(ht_p_values);
        cudaFree(ht_s_keys);
        return;
    }

    // === STAGE 3: Random access orderdate + revenue, aggregate ===
    uint32_t *d_orderdate, *d_revenue;
    cudaMalloc(&d_orderdate, h_num_stage2 * sizeof(uint32_t));
    cudaMalloc(&d_revenue, h_num_stage2 * sizeof(uint32_t));

    timer.start();
    acc_orderdate.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_orderdate, stream);
    acc_revenue.randomAccessBatchIndices(d_stage2_indices, h_num_stage2, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_final_ms = timer.elapsed_ms();

    // Aggregation with date lookup
    int agg_size = NUM_YEARS * NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));

    timer.start();
    int grid_agg = min((h_num_stage2 + block_size - 1) / block_size, 256);
    aggregateQ21Kernel<<<grid_agg, block_size>>>(
        d_orderdate, d_revenue, d_stage2_brands, h_num_stage2,
        ht_d_keys, ht_d_values, ht_d_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    // Collect results
    std::vector<unsigned long long> h_agg_revenue(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total_revenue = 0;
    int num_results = 0;
    for (int i = 0; i < agg_size; ++i) {
        if (h_agg_revenue[i] > 0) {
            total_revenue += h_agg_revenue[i];
            num_results++;
        }
    }

    // Calculate timing breakdown
    float data_load_total = decompress_suppkey_ms + ra_partkey_ms + ra_final_ms;
    float kernel_total = probe_stage1_ms + probe_stage2_ms + aggregate_ms;

    timing.data_load_ms = data_load_total;
    timing.kernel_ms = kernel_total;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q2.1 Results (Staged Random Access) ===" << std::endl;
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    std::cout << "\nTiming breakdown (STAGED):" << std::endl;
    std::cout << "  Hash build:           " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Stage 1 decomp supp:  " << decompress_suppkey_ms << " ms" << std::endl;
    std::cout << "  Stage 1 probe supp:   " << probe_stage1_ms << " ms (120M rows)" << std::endl;
    std::cout << "  Stage 2 RA partkey:   " << ra_partkey_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 2 probe part:   " << probe_stage2_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 3 RA final:     " << ra_final_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q2.1");

    // Cleanup
    cudaFree(d_stage2_indices); cudaFree(d_stage2_brands);
    cudaFree(d_orderdate); cudaFree(d_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_p_keys); cudaFree(ht_p_values);
    cudaFree(ht_s_keys);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.1 - Random Access Optimization ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ21RandomAccess(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ21RandomAccess(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
