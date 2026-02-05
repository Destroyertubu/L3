/**
 * @file q34_fls.cu
 * @brief SSB Q3.4 Implementation - Vertical-style FOR+BitPack
 *
 * Query:
 *   SELECT c_city, s_city, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, ddate
 *   WHERE lo_custkey = c_custkey
 *     AND lo_suppkey = s_suppkey
 *     AND lo_orderdate = d_datekey
 *     AND (c_city = 'UNITED KI1' OR c_city = 'UNITED KI5')
 *     AND (s_city = 'UNITED KI1' OR s_city = 'UNITED KI5')
 *     AND d_yearmonth = 'Dec1997'
 *   GROUP BY c_city, s_city, d_year;
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
constexpr int C_CITY_UK1 = 231;
constexpr int C_CITY_UK5 = 235;
// d_yearmonthnum for Dec1997 = 199712
constexpr int D_YEARMONTH_DEC1997 = 199712;

constexpr int C_HT_SIZE = 600037;
constexpr int S_HT_SIZE = 50021;
constexpr int D_HT_SIZE = 2557;

// 2 cities * 2 cities * 1 yearmonth = 4 groups
constexpr int NUM_CITIES = 2;
constexpr int GROUP_SIZE = NUM_CITIES * NUM_CITIES;
}

constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;
constexpr int TILE_SIZE = FLS_TILE_SIZE;

// Custom build kernel for city filter
__global__ void build_city_pht_2_q34(
    const int* __restrict__ keys,
    const int* __restrict__ cities,
    int city1, int city2,
    int num_rows,
    int* ht, int ht_len, int keys_min)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        int city = cities[tid];
        if (city == city1 || city == city2) {
            int key = keys[tid];
            int city_idx = (city == city1) ? 0 : 1;
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash << 1], 0, key);
            ht[(hash << 1) + 1] = city_idx;
        }
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q34_fls_kernel(
    const uint32_t* __restrict__ enc_lo_custkey,
    const uint32_t* __restrict__ enc_lo_suppkey,
    const uint32_t* __restrict__ enc_lo_orderdate,
    const uint32_t* __restrict__ enc_lo_revenue,
    int32_t lo_custkey_min, uint8_t lo_custkey_bw,
    int32_t lo_suppkey_min, uint8_t lo_suppkey_bw,
    int32_t lo_orderdate_min, uint8_t lo_orderdate_bw,
    int32_t lo_revenue_min, uint8_t lo_revenue_bw,
    int64_t n_tup_lineorder,
    int* __restrict__ c_ht, int* __restrict__ s_ht, int* __restrict__ d_ht,
    int c_ht_size, int s_ht_size, int d_ht_size,
    int c_min_key, int s_min_key, int d_min_key,
    unsigned long long* __restrict__ group_result)
{
    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int c_city_idx[ITEMS_PER_THREAD];
    int s_city_idx[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // Step 1: lo_custkey -> c_city_idx
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_custkey_bw * 32;
        unpack_for(enc_lo_custkey + offset, items, lo_custkey_min, lo_custkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, c_city_idx, selection_flags, c_ht, c_ht_size, c_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 2: lo_suppkey -> s_city_idx
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_suppkey_bw * 32;
        unpack_for(enc_lo_suppkey + offset, items, lo_suppkey_min, lo_suppkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, s_city_idx, selection_flags, s_ht, s_ht_size, s_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 3: lo_orderdate -> Dec1997 only (existence check)
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_orderdate_bw * 32;
        unpack_for(enc_lo_orderdate + offset, items, lo_orderdate_min, lo_orderdate_bw);
        // Use PHT_1 for existence check
        BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, selection_flags, d_ht, d_ht_size, d_min_key, num_tile_items);
    }
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 4: lo_revenue
    {
        int64_t offset = static_cast<int64_t>(blockIdx.x) * lo_revenue_bw * 32;
        unpack_for(enc_lo_revenue + offset, items2, lo_revenue_min, lo_revenue_bw);
    }

    // Aggregate: group_idx = c_city_idx * 2 + s_city_idx
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            int cc = c_city_idx[ITEM];
            int sc = s_city_idx[ITEM];
            if (cc >= 0 && cc < NUM_CITIES && sc >= 0 && sc < NUM_CITIES) {
                int idx = cc * NUM_CITIES + sc;
                atomicAdd(&group_result[idx], static_cast<unsigned long long>(items2[ITEM]));
            }
        }
    }
}

// Build date hash table for single yearmonth
__global__ void build_date_pht_1_yearmonth(
    const int* __restrict__ datekeys,
    const int* __restrict__ yearmonthnum,
    int target_yearmonth,
    int num_rows,
    int* ht, int ht_len, int keys_min)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        if (yearmonthnum[tid] == target_yearmonth) {
            int key = datekeys[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash], 0, key);
        }
    }
}

void buildHashTables(SSBDataFLS& data,
    int*& d_c_ht, int*& d_s_ht, int*& d_d_ht,
    int& c_min_key, int& s_min_key, int& d_min_key) {
    c_min_key = 1; s_min_key = 1; d_min_key = 19920101;

    cudaMalloc(&d_c_ht, C_HT_SIZE * 2 * sizeof(int));
    cudaMalloc(&d_s_ht, S_HT_SIZE * 2 * sizeof(int));
    cudaMalloc(&d_d_ht, D_HT_SIZE * sizeof(int));  // PHT_1
    cudaMemset(d_c_ht, 0, C_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_s_ht, 0, S_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_d_ht, 0, D_HT_SIZE * sizeof(int));

    int bs = 128;

    // Customer: city IN (UK1, UK5)
    build_city_pht_2_q34<<<(SSB_C_LEN+bs-1)/bs, bs>>>(
        data.d_c_custkey, data.d_c_city, C_CITY_UK1, C_CITY_UK5,
        SSB_C_LEN, d_c_ht, C_HT_SIZE, c_min_key);

    // Supplier: city IN (UK1, UK5)
    build_city_pht_2_q34<<<(SSB_S_LEN+bs-1)/bs, bs>>>(
        data.d_s_suppkey, data.d_s_city, C_CITY_UK1, C_CITY_UK5,
        SSB_S_LEN, d_s_ht, S_HT_SIZE, s_min_key);

    // Date: yearmonthnum = 199712 (Dec1997)
    build_date_pht_1_yearmonth<<<(SSB_D_LEN+bs-1)/bs, bs>>>(
        data.d_d_datekey, data.d_d_yearmonthnum, D_YEARMONTH_DEC1997,
        SSB_D_LEN, d_d_ht, D_HT_SIZE, d_min_key);

    cudaDeviceSynchronize();
}

void runQ34FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    int *d_c_ht, *d_s_ht, *d_d_ht;
    int c_min_key, s_min_key, d_min_key;
    buildHashTables(data, d_c_ht, d_s_ht, d_d_ht, c_min_key, s_min_key, d_min_key);

    unsigned long long* d_result;
    cudaMalloc(&d_result, GROUP_SIZE * sizeof(unsigned long long));
    cudaMemset(d_result, 0, GROUP_SIZE * sizeof(unsigned long long));

    int num_blocks = static_cast<int>(data.n_tiles);
    std::cout << "Q3.4: " << num_blocks << " blocks" << std::endl;

    // Warmup + timed
    q34_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
        data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
        data.lo_custkey.min_value, data.lo_custkey.bitwidth,
        data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
        data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
        data.lo_revenue.min_value, data.lo_revenue.bitwidth,
        data.n_tup_lineorder,
        d_c_ht, d_s_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, D_HT_SIZE,
        c_min_key, s_min_key, d_min_key, d_result);
    cudaDeviceSynchronize();

    float total_time = 0.0f;
    for (int run = 0; run < 3; run++) {
        cudaMemset(d_result, 0, GROUP_SIZE * sizeof(unsigned long long));
        cudaEventRecord(start);
        q34_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
            data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
            data.lo_custkey.min_value, data.lo_custkey.bitwidth,
            data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
            data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
            data.lo_revenue.min_value, data.lo_revenue.bitwidth,
            data.n_tup_lineorder,
            d_c_ht, d_s_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, D_HT_SIZE,
            c_min_key, s_min_key, d_min_key, d_result);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run+1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / 3;

    std::vector<unsigned long long> h_result(GROUP_SIZE);
    cudaMemcpy(h_result.data(), d_result, GROUP_SIZE * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q3.4 Results ===" << std::endl;
    std::cout << "Average: " << timing.kernel_ms << " ms" << std::endl;
    const char* cities[] = {"UNITED KI1", "UNITED KI5"};
    for (int cc = 0; cc < NUM_CITIES; cc++) {
        for (int sc = 0; sc < NUM_CITIES; sc++) {
            int idx = cc * NUM_CITIES + sc;
            if (h_result[idx] > 0) {
                std::cout << "  " << cities[cc] << ", " << cities[sc] << ", 1997, " << h_result[idx] << std::endl;
            }
        }
    }

    cudaFree(d_c_ht); cudaFree(d_s_ht); cudaFree(d_d_ht); cudaFree(d_result);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.4 - Vertical Style ===" << std::endl;
    SSBDataFLS data;
    data.loadAndEncode(data_dir);

    FLSQueryTiming timing;
    runQ34FLS(data, timing);

    data.free();
    return 0;
}
