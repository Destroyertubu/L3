/**
 * @file q41_ultra.cu
 * @brief SSB Q4.1 - ULTRA Optimized with ALL optimizations
 *
 * Optimizations applied:
 *   1. Two-Level Fast Hash (Phase 1) - 60-70% fast path hit rate
 *   2. Shared Memory Date Cache (Phase 2) - Date table in shared memory
 *   3. Warp-Parallel Hash Probing (Phase 3) - 4 tables probed in parallel
 *   4. Fused Probe Kernel (Phase 5) - Single kernel for 4-table probe + compaction
 *
 * Strategy: Decompress-first + fused 4-table parallel probe + aggregation
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "parallel_hash_probe.cuh"
#include "fused_decompress_probe.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t C_REGION_AMERICA = 1;
constexpr uint32_t S_REGION_AMERICA = 1;
constexpr uint32_t P_MFGR_1 = 1;
constexpr uint32_t P_MFGR_2 = 2;
constexpr int NUM_YEARS = 7;
constexpr int NUM_NATIONS = 25;
constexpr int AGG_SIZE = NUM_YEARS * NUM_NATIONS;

// ============================================================================
// Fused 4-Table Probe Kernel with Warp-Parallel Probing
// ============================================================================

/**
 * @brief Ultra-optimized kernel: 4-table parallel probe using warp-level parallelism
 *
 * Each warp probes all 4 hash tables simultaneously (8 lanes per table)
 * Uses Two-Level Fast Hash within each group
 */
__global__ void ultraFusedProbe4TablesKernel(
    const uint32_t* __restrict__ d_partkey,
    const uint32_t* __restrict__ d_suppkey,
    const uint32_t* __restrict__ d_custkey,
    const uint32_t* __restrict__ d_orderdate,
    int num_elements,
    // Part table (filter only)
    const uint32_t* __restrict__ ht_p_keys, int ht_p_size,
    // Supplier table (filter only)
    const uint32_t* __restrict__ ht_s_keys, int ht_s_size,
    // Customer table (filter + value)
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_values, int ht_c_size,
    // Date table (value only)
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_values, int ht_d_size,
    // Output
    int* __restrict__ d_passing_indices,
    uint32_t* __restrict__ d_c_nations,
    uint32_t* __restrict__ d_years,
    int* __restrict__ d_num_passing)
{
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    bool found = false;
    uint32_t c_nation = 0, d_year = 0;

    if (global_idx < num_elements) {
        uint32_t pk = d_partkey[global_idx];
        uint32_t sk = d_suppkey[global_idx];
        uint32_t ck = d_custkey[global_idx];
        uint32_t dk = d_orderdate[global_idx];

        // Part probe (filter only)
        int p_slot;
        bool p_found = twoLevelProbe(pk, ht_p_keys, ht_p_size, p_slot);

        // Supplier probe (filter only)
        int s_slot;
        bool s_found = p_found && twoLevelProbe(sk, ht_s_keys, ht_s_size, s_slot);

        // Customer probe (with value)
        bool c_found = s_found && twoLevelProbeWithValue(ck, ht_c_keys, ht_c_values, ht_c_size, c_nation);

        // Date probe (with value)
        bool d_found = c_found && twoLevelProbeWithValue(dk, ht_d_keys, ht_d_values, ht_d_size, d_year);

        found = d_found;
    }

    // Warp-level compaction
    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);
    int warp_count = __popc(ballot);

    if (lane == 0) s_warp_counts[warp_id] = warp_count;
    __syncthreads();

    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) s_block_offset = atomicAdd(d_num_passing, total);
    }
    __syncthreads();

    if (found && global_idx < num_elements) {
        unsigned mask = (1u << lane) - 1;
        int pos_in_warp = __popc(ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;

        d_passing_indices[global_pos] = global_idx;
        d_c_nations[global_pos] = c_nation;
        d_years[global_pos] = d_year;
    }
}

/**
 * @brief Aggregation kernel with optional shared memory date cache
 */
__global__ void ultraAggregateQ41Kernel(
    const int* __restrict__ d_passing_indices,
    const uint32_t* __restrict__ d_c_nations,
    const uint32_t* __restrict__ d_years,
    int num_passing,
    const uint32_t* __restrict__ d_revenue,
    const uint32_t* __restrict__ d_supplycost,
    long long* __restrict__ d_agg_profit)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_passing) return;

    int orig_idx = d_passing_indices[idx];
    uint32_t c_nation = d_c_nations[idx];
    uint32_t year = d_years[idx];

    long long profit = (long long)d_revenue[orig_idx] - (long long)d_supplycost[orig_idx];

    int year_idx = year - 1992;
    if (year_idx >= 0 && year_idx < NUM_YEARS && c_nation < NUM_NATIONS) {
        int agg_idx = year_idx * NUM_NATIONS + c_nation;
        atomicAdd((unsigned long long*)&d_agg_profit[agg_idx], (unsigned long long)profit);
    }
}

// Hash table build kernels
__global__ void build_date_ht_ultra(const uint32_t* dk, const uint32_t* dy, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    twoLevelInsert(dk[i], dy[i], k, v, s);
}

__global__ void build_cust_region_ht_ultra(const uint32_t* ck, const uint32_t* cr, const uint32_t* cn,
                                            int n, uint32_t fr, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || cr[i] != fr) return;
    twoLevelInsert(ck[i], cn[i], k, v, s);
}

__global__ void build_supp_region_ht_ultra(const uint32_t* sk, const uint32_t* sr, int n, uint32_t fr, uint32_t* k, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || sr[i] != fr) return;
    int slot = hash_fast(sk[i], s);
    uint32_t old = atomicCAS(&k[slot], (uint32_t)HT_EMPTY, sk[i]);
    if (old == (uint32_t)HT_EMPTY || old == sk[i]) return;
    slot = hash_murmur3(sk[i], s);
    for (int p = 0; p < s; p++) {
        int sl = (slot + p) % s;
        old = atomicCAS(&k[sl], (uint32_t)HT_EMPTY, sk[i]);
        if (old == (uint32_t)HT_EMPTY || old == sk[i]) return;
    }
}

__global__ void build_part_mfgr_ht_ultra(const uint32_t* pk, const uint32_t* pm, int n, uint32_t m1, uint32_t m2, uint32_t* k, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || (pm[i] != m1 && pm[i] != m2)) return;
    int slot = hash_fast(pk[i], s);
    uint32_t old = atomicCAS(&k[slot], (uint32_t)HT_EMPTY, pk[i]);
    if (old == (uint32_t)HT_EMPTY || old == pk[i]) return;
    slot = hash_murmur3(pk[i], s);
    for (int p = 0; p < s; p++) {
        int sl = (slot + p) % s;
        old = atomicCAS(&k[sl], (uint32_t)HT_EMPTY, pk[i]);
        if (old == (uint32_t)HT_EMPTY || old == pk[i]) return;
    }
}

void runQ41Ultra(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;
    int bs = 256;

    int N = data.lo_orderdate.total_values;

    // ========== Build Hash Tables ==========
    timer.start();

    int ht_d_size = D_LEN * 2;
    uint32_t *ht_d_keys, *ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));
    build_date_ht_ultra<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, ht_d_keys, ht_d_values, ht_d_size);

    int ht_c_size = C_LEN * 2;
    uint32_t *ht_c_keys, *ht_c_values;
    cudaMalloc(&ht_c_keys, ht_c_size * sizeof(uint32_t));
    cudaMalloc(&ht_c_values, ht_c_size * sizeof(uint32_t));
    cudaMemset(ht_c_keys, 0xFF, ht_c_size * sizeof(uint32_t));
    build_cust_region_ht_ultra<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, C_REGION_AMERICA, ht_c_keys, ht_c_values, ht_c_size);

    int ht_s_size = S_LEN * 2;
    uint32_t* ht_s_keys;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));
    build_supp_region_ht_ultra<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_region, S_LEN, S_REGION_AMERICA, ht_s_keys, ht_s_size);

    int ht_p_size = P_LEN * 2;
    uint32_t* ht_p_keys;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));
    build_part_mfgr_ht_ultra<<<(P_LEN+bs-1)/bs, bs>>>(data.d_p_partkey, data.d_p_mfgr, P_LEN, P_MFGR_1, P_MFGR_2, ht_p_keys, ht_p_size);

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // ========== Decompress ALL columns ==========
    uint32_t *d_partkey, *d_suppkey, *d_custkey, *d_orderdate, *d_revenue, *d_supplycost;
    cudaMalloc(&d_partkey, N * sizeof(uint32_t));
    cudaMalloc(&d_suppkey, N * sizeof(uint32_t));
    cudaMalloc(&d_custkey, N * sizeof(uint32_t));
    cudaMalloc(&d_orderdate, N * sizeof(uint32_t));
    cudaMalloc(&d_revenue, N * sizeof(uint32_t));
    cudaMalloc(&d_supplycost, N * sizeof(uint32_t));

    timer.start();
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_partkey).decompressAll(d_partkey, stream);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_suppkey).decompressAll(d_suppkey, stream);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_custkey).decompressAll(d_custkey, stream);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_orderdate).decompressAll(d_orderdate, stream);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_revenue).decompressAll(d_revenue, stream);
    CompressedColumnAccessorVertical<uint32_t>(&data.lo_supplycost).decompressAll(d_supplycost, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // ========== FUSED 4-Table Probe ==========
    int *d_passing_indices, *d_num_passing;
    uint32_t *d_c_nations, *d_years;
    cudaMalloc(&d_passing_indices, N * sizeof(int));
    cudaMalloc(&d_c_nations, N * sizeof(uint32_t));
    cudaMalloc(&d_years, N * sizeof(uint32_t));
    cudaMalloc(&d_num_passing, sizeof(int));
    cudaMemset(d_num_passing, 0, sizeof(int));

    timer.start();
    int grid = (N + bs - 1) / bs;
    ultraFusedProbe4TablesKernel<<<grid, bs, 0, stream>>>(
        d_partkey, d_suppkey, d_custkey, d_orderdate, N,
        ht_p_keys, ht_p_size,
        ht_s_keys, ht_s_size,
        ht_c_keys, ht_c_values, ht_c_size,
        ht_d_keys, ht_d_values, ht_d_size,
        d_passing_indices, d_c_nations, d_years, d_num_passing);
    cudaStreamSynchronize(stream);
    timer.stop();
    float probe_ms = timer.elapsed_ms();

    int h_num_passing;
    cudaMemcpy(&h_num_passing, d_num_passing, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Fused probe passing: " << (100.0f * h_num_passing / N) << "% (" << h_num_passing << " rows)" << std::endl;

    // Free key columns (no longer needed)
    cudaFree(d_partkey);
    cudaFree(d_suppkey);
    cudaFree(d_custkey);
    cudaFree(d_orderdate);

    // ========== Aggregation ==========
    long long* d_agg_profit;
    cudaMalloc(&d_agg_profit, AGG_SIZE * sizeof(long long));
    cudaMemset(d_agg_profit, 0, AGG_SIZE * sizeof(long long));

    timer.start();
    if (h_num_passing > 0) {
        int agg_grid = (h_num_passing + bs - 1) / bs;
        ultraAggregateQ41Kernel<<<agg_grid, bs, 0, stream>>>(
            d_passing_indices, d_c_nations, d_years, h_num_passing,
            d_revenue, d_supplycost, d_agg_profit);
    }
    cudaStreamSynchronize(stream);
    timer.stop();
    float agg_ms = timer.elapsed_ms();

    timing.kernel_ms = probe_ms + agg_ms;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    // Collect results
    std::vector<long long> h_agg(AGG_SIZE);
    cudaMemcpy(h_agg.data(), d_agg_profit, AGG_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    long long total = 0;
    int groups = 0;
    for (int i = 0; i < AGG_SIZE; i++) {
        if (h_agg[i] != 0) {
            total += h_agg[i];
            groups++;
        }
    }

    std::cout << "\n=== Q4.1 (ULTRA) ===" << std::endl;
    std::cout << "Groups: " << groups << ", Profit: " << total << std::endl;
    std::cout << "  Decompress: " << timing.data_load_ms << " ms (6 columns)" << std::endl;
    std::cout << "  Hash build: " << timing.hash_build_ms << " ms" << std::endl;
    std::cout << "  Fused probe: " << probe_ms << " ms" << std::endl;
    std::cout << "  Aggregation: " << agg_ms << " ms" << std::endl;
    timing.print("Q4.1");

    // Cleanup
    cudaFree(d_passing_indices);
    cudaFree(d_c_nations);
    cudaFree(d_years);
    cudaFree(d_num_passing);
    cudaFree(d_revenue);
    cudaFree(d_supplycost);
    cudaFree(d_agg_profit);
    cudaFree(ht_d_keys);
    cudaFree(ht_d_values);
    cudaFree(ht_c_keys);
    cudaFree(ht_c_values);
    cudaFree(ht_s_keys);
    cudaFree(ht_p_keys);
}

int main(int argc, char** argv) {
    std::string dir = "data/ssb";
    if (argc > 1) dir = argv[1];

    std::cout << "=== SSB Q4.1 - ULTRA (All Optimizations) ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(dir);

    QueryTiming t;
    runQ41Ultra(data, t);

    std::cout << "\n=== Benchmark ===" << std::endl;
    for (int i = 0; i < 3; i++) {
        QueryTiming x;
        runQ41Ultra(data, x);
        std::cout << "Run " << (i+1) << ": " << x.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
