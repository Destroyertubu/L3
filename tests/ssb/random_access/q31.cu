/**
 * @file q31.cu
 * @brief SSB Q3.1 Implementation - Random Access Optimization Strategy
 *
 * Query:
 *   SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, date
 *   WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *     AND lo_orderdate = d_datekey
 *     AND c_region = 'ASIA' AND s_region = 'ASIA'
 *     AND d_year >= 1992 AND d_year <= 1997
 *   GROUP BY c_nation, s_nation, d_year
 *
 * Strategy:
 *   1. Build hash tables for dimension tables (pre-filtered)
 *   2. Full decompress JOIN key columns (lo_custkey, lo_suppkey)
 *   3. Probe hash tables → passing_indices + passing_nations
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

constexpr uint32_t C_REGION_ASIA = 2;
constexpr uint32_t S_REGION_ASIA = 2;
constexpr int NUM_YEARS = 6;  // 1992-1997
constexpr int NUM_NATIONS = 25;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS * NUM_NATIONS;

/**
 * @brief Stage 1: Probe supplier hash table (s_region = 'ASIA')
 * Outputs passing row indices and s_nation values
 */
__global__ void probeSupplierFilterQ31Kernel(
    const uint32_t* __restrict__ suppkeys,
    int num_rows,
    const uint32_t* __restrict__ ht_s_keys,
    const uint32_t* __restrict__ ht_s_values,
    int ht_s_size,
    int* __restrict__ passing_indices,
    uint32_t* __restrict__ passing_s_nations,
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
    uint32_t s_nation = 0;

    if (tid < num_rows) {
        uint32_t sk = suppkeys[tid];
        int slot = hash_murmur3(sk, ht_s_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_s_size;
            if (ht_s_keys[s] == sk) {
                s_nation = ht_s_values[s];
                found = true;
                break;
            }
            if (ht_s_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
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
        passing_s_nations[global_pos] = s_nation;
    }
}

/**
 * @brief Stage 2: Probe customer hash table with input indices
 * Takes custkeys already fetched via random access
 */
__global__ void probeCustomerWithIndicesQ31Kernel(
    const uint32_t* __restrict__ custkeys,
    const int* __restrict__ input_indices,
    const uint32_t* __restrict__ input_s_nations,
    int num_rows,
    const uint32_t* __restrict__ ht_c_keys,
    const uint32_t* __restrict__ ht_c_values,
    int ht_c_size,
    int* __restrict__ output_indices,
    uint32_t* __restrict__ output_c_nations,
    uint32_t* __restrict__ output_s_nations,
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
    uint32_t c_nation = 0;
    int original_idx = -1;
    uint32_t s_nation = 0;

    if (tid < num_rows) {
        uint32_t ck = custkeys[tid];
        original_idx = input_indices[tid];
        s_nation = input_s_nations[tid];

        int slot = hash_murmur3(ck, ht_c_size);
        for (int probe = 0; probe < 32; ++probe) {
            int s = (slot + probe) % ht_c_size;
            if (ht_c_keys[s] == ck) {
                c_nation = ht_c_values[s];
                found = true;
                break;
            }
            if (ht_c_keys[s] == static_cast<uint32_t>(HT_EMPTY)) break;
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
        output_c_nations[global_pos] = c_nation;
        output_s_nations[global_pos] = s_nation;
    }
}

/**
 * @brief Final aggregation with date hash lookup for Q3.1
 */
__global__ void aggregateQ31Kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_revenue,
    const uint32_t* __restrict__ passing_c_nations,
    const uint32_t* __restrict__ passing_s_nations,
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
        uint32_t c_nation = passing_c_nations[i];
        uint32_t s_nation = passing_s_nations[i];

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

        if (found && d_year >= 1992 && d_year <= 1997) {
            int year_idx = d_year - 1992;
            if (c_nation < NUM_NATIONS && s_nation < NUM_NATIONS) {
                int agg_idx = year_idx * NUM_NATIONS * NUM_NATIONS + c_nation * NUM_NATIONS + s_nation;
                atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(rev));
            }
        }
    }
}

// Hash table build kernels
__global__ void build_date_ht_q31(const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
                                   uint32_t* ht_keys, uint32_t* ht_values, int ht_size) {
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

__global__ void build_customer_ht_q31(
    const uint32_t* c_custkey, const uint32_t* c_region, const uint32_t* c_nation,
    int num_rows, uint32_t filter_region,
    uint32_t* ht_keys, uint32_t* ht_values, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (c_region[idx] != filter_region) return;
    uint32_t key = c_custkey[idx];
    uint32_t value = c_nation[idx];
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

__global__ void build_supplier_ht_q31(
    const uint32_t* s_suppkey, const uint32_t* s_region, const uint32_t* s_nation,
    int num_rows, uint32_t filter_region,
    uint32_t* ht_keys, uint32_t* ht_values, int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    if (s_region[idx] != filter_region) return;
    uint32_t key = s_suppkey[idx];
    uint32_t value = s_nation[idx];
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

void runQ31RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int block_size = 256;

    CompressedColumnAccessorVertical<uint32_t> acc_custkey(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    int total_rows = acc_custkey.getTotalElements();

    // Step 1: Build dimension hash tables
    timer.start();

    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));

    build_date_ht_q31<<<(D_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_c_size = C_LEN * 2;
    uint32_t *ht_c_keys, *ht_c_values;
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));

    build_customer_ht_q31<<<(C_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, C_REGION_ASIA,
        ht_c_keys, ht_c_values, ht_c_size);

    int ht_s_size = S_LEN * 2;
    uint32_t *ht_s_keys, *ht_s_values;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMalloc(&ht_s_values, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    build_supplier_ht_q31<<<(S_LEN + block_size - 1) / block_size, block_size>>>(
        data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, S_REGION_ASIA,
        ht_s_keys, ht_s_values, ht_s_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== STAGED DECOMPOSITION ==========

    // === STAGE 1: Decompress suppkey + probe supplier ===
    uint32_t* d_suppkey;
    cudaMalloc(&d_suppkey, total_rows * sizeof(uint32_t));

    timer.start();
    acc_suppkey.decompressAll(d_suppkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_suppkey_ms = timer.elapsed_ms();

    int* d_stage1_indices;
    uint32_t* d_stage1_s_nations;
    int* d_num_stage1;
    cudaMalloc(&d_stage1_indices, total_rows * sizeof(int));
    cudaMalloc(&d_stage1_s_nations, total_rows * sizeof(uint32_t));
    cudaMalloc(&d_num_stage1, sizeof(int));
    cudaMemset(d_num_stage1, 0, sizeof(int));

    timer.start();
    int grid_stage1 = (total_rows + block_size - 1) / block_size;
    probeSupplierFilterQ31Kernel<<<grid_stage1, block_size>>>(
        d_suppkey, total_rows,
        ht_s_keys, ht_s_values, ht_s_size,
        d_stage1_indices, d_stage1_s_nations, d_num_stage1);
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
        std::cout << "\n=== Q3.1 Results (Staged Random Access) ===" << std::endl;
        std::cout << "Total revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_stage1_indices); cudaFree(d_stage1_s_nations);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_c_keys); cudaFree(ht_c_values);
        cudaFree(ht_s_keys); cudaFree(ht_s_values);
        return;
    }

    // === STAGE 2: Random access custkey + probe customer ===
    uint32_t* d_custkey;
    cudaMalloc(&d_custkey, h_num_stage1 * sizeof(uint32_t));

    timer.start();
    acc_custkey.randomAccessBatchIndices(d_stage1_indices, h_num_stage1, d_custkey, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float ra_custkey_ms = timer.elapsed_ms();

    int* d_stage2_indices;
    uint32_t* d_stage2_c_nations;
    uint32_t* d_stage2_s_nations;
    int* d_num_stage2;
    cudaMalloc(&d_stage2_indices, h_num_stage1 * sizeof(int));
    cudaMalloc(&d_stage2_c_nations, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_stage2_s_nations, h_num_stage1 * sizeof(uint32_t));
    cudaMalloc(&d_num_stage2, sizeof(int));
    cudaMemset(d_num_stage2, 0, sizeof(int));

    timer.start();
    int grid_stage2 = (h_num_stage1 + block_size - 1) / block_size;
    probeCustomerWithIndicesQ31Kernel<<<grid_stage2, block_size>>>(
        d_custkey, d_stage1_indices, d_stage1_s_nations, h_num_stage1,
        ht_c_keys, ht_c_values, ht_c_size,
        d_stage2_indices, d_stage2_c_nations, d_stage2_s_nations, d_num_stage2);
    cudaDeviceSynchronize();
    timer.stop();
    float probe_stage2_ms = timer.elapsed_ms();

    int h_num_stage2;
    cudaMemcpy(&h_num_stage2, d_num_stage2, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_custkey);
    cudaFree(d_stage1_indices);
    cudaFree(d_stage1_s_nations);
    cudaFree(d_num_stage2);

    float stage2_selectivity = 100.0f * h_num_stage2 / total_rows;
    std::cout << "Stage 2 (customer filter): " << stage2_selectivity << "% (" << h_num_stage2 << " rows)" << std::endl;

    if (h_num_stage2 == 0) {
        std::cout << "\n=== Q3.1 Results (Staged Random Access) ===" << std::endl;
        std::cout << "Total revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_stage2_indices); cudaFree(d_stage2_c_nations); cudaFree(d_stage2_s_nations);
        cudaFree(ht_d_keys); cudaFree(ht_d_values);
        cudaFree(ht_c_keys); cudaFree(ht_c_values);
        cudaFree(ht_s_keys); cudaFree(ht_s_values);
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

    unsigned long long* d_agg_revenue;
    cudaMalloc(&d_agg_revenue, AGG_SIZE * sizeof(unsigned long long));
    cudaMemset(d_agg_revenue, 0, AGG_SIZE * sizeof(unsigned long long));

    timer.start();
    int grid_agg = min((h_num_stage2 + block_size - 1) / block_size, 256);
    aggregateQ31Kernel<<<grid_agg, block_size>>>(
        d_orderdate, d_revenue, d_stage2_c_nations, d_stage2_s_nations, h_num_stage2,
        ht_d_keys, ht_d_values, ht_d_size, d_agg_revenue);
    cudaDeviceSynchronize();
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    // Collect results
    std::vector<unsigned long long> h_agg_revenue(AGG_SIZE);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, AGG_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    unsigned long long total_revenue = 0;
    int num_groups = 0;
    for (int i = 0; i < AGG_SIZE; ++i) {
        if (h_agg_revenue[i] > 0) {
            total_revenue += h_agg_revenue[i];
            num_groups++;
        }
    }

    float data_load_total = decompress_suppkey_ms + ra_custkey_ms + ra_final_ms;
    float kernel_total = probe_stage1_ms + probe_stage2_ms + aggregate_ms;

    timing.data_load_ms = data_load_total;
    timing.kernel_ms = kernel_total;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    std::cout << "\n=== Q3.1 Results (Staged Random Access) ===" << std::endl;
    std::cout << "Total groups: " << num_groups << ", Total revenue: " << total_revenue << std::endl;
    std::cout << "\nTiming breakdown (STAGED):" << std::endl;
    std::cout << "  Hash build:           " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Stage 1 decomp supp:  " << decompress_suppkey_ms << " ms" << std::endl;
    std::cout << "  Stage 1 probe supp:   " << probe_stage1_ms << " ms (120M rows)" << std::endl;
    std::cout << "  Stage 2 RA custkey:   " << ra_custkey_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 2 probe cust:   " << probe_stage2_ms << " ms (" << h_num_stage1 << " rows)" << std::endl;
    std::cout << "  Stage 3 RA final:     " << ra_final_ms << " ms (" << h_num_stage2 << " rows)" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q3.1");

    // Cleanup
    cudaFree(d_stage2_indices); cudaFree(d_stage2_c_nations); cudaFree(d_stage2_s_nations);
    cudaFree(d_orderdate); cudaFree(d_revenue);
    cudaFree(ht_d_keys); cudaFree(ht_d_values);
    cudaFree(ht_c_keys); cudaFree(ht_c_values);
    cudaFree(ht_s_keys); cudaFree(ht_s_values);
    cudaFree(d_agg_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.1 - Random Access Optimization ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ31RandomAccess(data, timing);

    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ31RandomAccess(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
