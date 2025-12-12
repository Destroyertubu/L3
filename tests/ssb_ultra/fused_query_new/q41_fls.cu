/**
 * @file q41_fls.cu
 * @brief SSB Q4.1 Implementation - Vertical-style FOR+BitPack with 4-way Hash Joins
 *
 * Query:
 *   SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
 *   FROM ddate, customer, supplier, part, lineorder
 *   WHERE lo_custkey = c_custkey
 *     AND lo_suppkey = s_suppkey
 *     AND lo_partkey = p_partkey
 *     AND lo_orderdate = d_datekey
 *     AND c_region = 'AMERICA'
 *     AND s_region = 'AMERICA'
 *     AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 *   GROUP BY d_year, c_nation
 *   ORDER BY d_year, c_nation;
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "common/fls_constants.cuh"
#include "common/fls_unpack.cuh"
#include "common/crystal_fls.cuh"
#include "common/ssb_data_fls.hpp"
#include "common/perfect_hash_table.cuh"

using namespace l3_fls;

namespace {
constexpr int C_REGION_AMERICA = 1;
constexpr int S_REGION_AMERICA = 1;
constexpr int P_MFGR_1 = 1;
constexpr int P_MFGR_2 = 2;

constexpr int C_HT_SIZE = 600037;
constexpr int S_HT_SIZE = 50021;
constexpr int P_HT_SIZE = 1000003;
constexpr int D_HT_SIZE = 2557;

// 5 nations (AMERICA) * 7 years
constexpr int NUM_NATIONS = 5;
constexpr int NUM_YEARS = 7;
constexpr int GROUP_SIZE = NUM_YEARS * NUM_NATIONS;
}

constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;
constexpr int TILE_SIZE = FLS_TILE_SIZE;

// Custom atomicAdd for signed long long (using unsigned long long underneath)
__device__ __forceinline__ void atomicAddLL(long long* address, long long val) {
    atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(val));
}

// Build part hash table for MFGR#1 OR MFGR#2
__global__ void build_part_pht_1_mfgr(
    const int* __restrict__ keys,
    const int* __restrict__ mfgr,
    int mfgr1, int mfgr2,
    int num_rows,
    int* ht, int ht_len, int keys_min)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        int m = mfgr[tid];
        if (m == mfgr1 || m == mfgr2) {
            int key = keys[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash], 0, key);
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q41_fls_kernel(
    const uint32_t* __restrict__ enc_lo_custkey,
    const uint32_t* __restrict__ enc_lo_suppkey,
    const uint32_t* __restrict__ enc_lo_partkey,
    const uint32_t* __restrict__ enc_lo_orderdate,
    const uint32_t* __restrict__ enc_lo_revenue,
    const uint32_t* __restrict__ enc_lo_supplycost,
    int32_t lo_custkey_min, uint8_t lo_custkey_bw,
    int32_t lo_suppkey_min, uint8_t lo_suppkey_bw,
    int32_t lo_partkey_min, uint8_t lo_partkey_bw,
    int32_t lo_orderdate_min, uint8_t lo_orderdate_bw,
    int32_t lo_revenue_min, uint8_t lo_revenue_bw,
    int32_t lo_supplycost_min, uint8_t lo_supplycost_bw,
    int64_t n_tup_lineorder,
    int* __restrict__ c_ht, int* __restrict__ s_ht, int* __restrict__ p_ht, int* __restrict__ d_ht,
    int c_ht_size, int s_ht_size, int p_ht_size, int d_ht_size,
    int c_min_key, int s_min_key, int p_min_key, int d_min_key,
    long long* __restrict__ group_result)  // profit can be negative
{
    int items[ITEMS_PER_THREAD];
    int revenue[ITEMS_PER_THREAD];
    int supplycost[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // Step 1: lo_custkey -> c_nation (AMERICA)
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_custkey_bw * 32;
        unpack_for(enc_lo_custkey + offset, items, lo_custkey_min, lo_custkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, c_nation, selection_flags, c_ht, c_ht_size, c_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 2: lo_suppkey (AMERICA, existence check)
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_suppkey_bw * 32;
        unpack_for(enc_lo_suppkey + offset, items, lo_suppkey_min, lo_suppkey_bw);
        BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, selection_flags, s_ht, s_ht_size, s_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 3: lo_partkey (MFGR#1 OR MFGR#2, existence check)
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_partkey_bw * 32;
        unpack_for(enc_lo_partkey + offset, items, lo_partkey_min, lo_partkey_bw);
        BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, selection_flags, p_ht, p_ht_size, p_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 4: lo_orderdate -> year
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_orderdate_bw * 32;
        unpack_for(enc_lo_orderdate + offset, items, lo_orderdate_min, lo_orderdate_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, year, selection_flags, d_ht, d_ht_size, d_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 5: lo_revenue
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_revenue_bw * 32;
        unpack_for(enc_lo_revenue + offset, revenue, lo_revenue_min, lo_revenue_bw);
    }

    // Step 6: lo_supplycost
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_supplycost_bw * 32;
        unpack_for(enc_lo_supplycost + offset, supplycost, lo_supplycost_min, lo_supplycost_bw);
    }

    // Aggregate: profit = revenue - supplycost
    // group_idx = (year - 1992) * NUM_NATIONS + c_nation
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            int y = year[ITEM] - 1992;
            int cn = c_nation[ITEM];
            if (y >= 0 && y < NUM_YEARS && cn >= 0 && cn < NUM_NATIONS) {
                int idx = y * NUM_NATIONS + cn;
                long long profit = static_cast<long long>(revenue[ITEM]) - static_cast<long long>(supplycost[ITEM]);
                atomicAddLL(&group_result[idx], profit);
            }
        }
    }
}

void buildHashTables(SSBDataFLS& data,
    int*& d_c_ht, int*& d_s_ht, int*& d_p_ht, int*& d_d_ht,
    int& c_min_key, int& s_min_key, int& p_min_key, int& d_min_key) {
    c_min_key = 1; s_min_key = 1; p_min_key = 1; d_min_key = 19920101;

    cudaMalloc(&d_c_ht, C_HT_SIZE * 2 * sizeof(int));  // PHT_2
    cudaMalloc(&d_s_ht, S_HT_SIZE * sizeof(int));      // PHT_1
    cudaMalloc(&d_p_ht, P_HT_SIZE * sizeof(int));      // PHT_1
    cudaMalloc(&d_d_ht, D_HT_SIZE * 2 * sizeof(int));  // PHT_2

    cudaMemset(d_c_ht, 0, C_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_s_ht, 0, S_HT_SIZE * sizeof(int));
    cudaMemset(d_p_ht, 0, P_HT_SIZE * sizeof(int));
    cudaMemset(d_d_ht, 0, D_HT_SIZE * 2 * sizeof(int));

    int bs = 128;

    // Customer: region=AMERICA -> nation
    build_pht_2_kernel<int,int><<<(SSB_C_LEN+bs-1)/bs, bs>>>(
        data.d_c_custkey, data.d_c_nation, data.d_c_region, C_REGION_AMERICA,
        SSB_C_LEN, d_c_ht, C_HT_SIZE, c_min_key);

    // Supplier: region=AMERICA (existence)
    build_pht_1_kernel<int><<<(SSB_S_LEN+bs-1)/bs, bs>>>(
        data.d_s_suppkey, data.d_s_region, S_REGION_AMERICA,
        SSB_S_LEN, d_s_ht, S_HT_SIZE, s_min_key);

    // Part: mfgr IN (1, 2) (existence)
    build_part_pht_1_mfgr<<<(SSB_P_LEN+bs-1)/bs, bs>>>(
        data.d_p_partkey, data.d_p_mfgr, P_MFGR_1, P_MFGR_2,
        SSB_P_LEN, d_p_ht, P_HT_SIZE, p_min_key);

    // Date: all -> year
    build_pht_2_all_kernel<int,int><<<(SSB_D_LEN+bs-1)/bs, bs>>>(
        data.d_d_datekey, data.d_d_year,
        SSB_D_LEN, d_d_ht, D_HT_SIZE, d_min_key);

    cudaDeviceSynchronize();
}

void runQ41FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int *d_c_ht, *d_s_ht, *d_p_ht, *d_d_ht;
    int c_min_key, s_min_key, p_min_key, d_min_key;
    buildHashTables(data, d_c_ht, d_s_ht, d_p_ht, d_d_ht, c_min_key, s_min_key, p_min_key, d_min_key);

    long long* d_result;
    cudaMalloc(&d_result, GROUP_SIZE * sizeof(long long));
    cudaMemset(d_result, 0, GROUP_SIZE * sizeof(long long));

    int num_blocks = static_cast<int>(data.n_tiles);
    std::cout << "Q4.1: " << num_blocks << " blocks, " << BLOCK_THREADS << " threads" << std::endl;

    // Warmup
    q41_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
        data.lo_partkey.d_encoded, data.lo_orderdate.d_encoded,
        data.lo_revenue.d_encoded, data.lo_supplycost.d_encoded,
        data.lo_custkey.min_value, data.lo_custkey.bitwidth,
        data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
        data.lo_partkey.min_value, data.lo_partkey.bitwidth,
        data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
        data.lo_revenue.min_value, data.lo_revenue.bitwidth,
        data.lo_supplycost.min_value, data.lo_supplycost.bitwidth,
        data.n_tup_lineorder,
        d_c_ht, d_s_ht, d_p_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
        c_min_key, s_min_key, p_min_key, d_min_key, d_result);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    for (int run = 0; run < 3; run++) {
        cudaMemset(d_result, 0, GROUP_SIZE * sizeof(long long));
        cudaEventRecord(start);
        q41_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
            data.lo_partkey.d_encoded, data.lo_orderdate.d_encoded,
            data.lo_revenue.d_encoded, data.lo_supplycost.d_encoded,
            data.lo_custkey.min_value, data.lo_custkey.bitwidth,
            data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
            data.lo_partkey.min_value, data.lo_partkey.bitwidth,
            data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
            data.lo_revenue.min_value, data.lo_revenue.bitwidth,
            data.lo_supplycost.min_value, data.lo_supplycost.bitwidth,
            data.n_tup_lineorder,
            d_c_ht, d_s_ht, d_p_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
            c_min_key, s_min_key, p_min_key, d_min_key, d_result);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run+1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / 3;

    std::vector<long long> h_result(GROUP_SIZE);
    cudaMemcpy(h_result.data(), d_result, GROUP_SIZE * sizeof(long long), cudaMemcpyDeviceToHost);

    const char* nations[] = {"ARGENTINA", "BRAZIL", "CANADA", "PERU", "UNITED STATES"};

    std::cout << "\n=== Q4.1 Results ===" << std::endl;
    std::cout << "Average: " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "\n(year, c_nation, profit):" << std::endl;
    for (int y = 0; y < NUM_YEARS; y++) {
        for (int cn = 0; cn < NUM_NATIONS; cn++) {
            int idx = y * NUM_NATIONS + cn;
            if (h_result[idx] != 0) {
                std::cout << "  " << (1992 + y) << ", " << nations[cn] << ", " << h_result[idx] << std::endl;
            }
        }
    }

    cudaFree(d_c_ht); cudaFree(d_s_ht); cudaFree(d_p_ht); cudaFree(d_d_ht); cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q4.1 - Vertical Style ===" << std::endl;
    SSBDataFLS data;
    data.loadAndEncode(data_dir);

    FLSQueryTiming timing;
    runQ41FLS(data, timing);

    data.free();
    return 0;
}
