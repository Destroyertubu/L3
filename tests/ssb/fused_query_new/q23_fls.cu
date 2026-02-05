/**
 * @file q23_fls.cu
 * @brief SSB Q2.3 Implementation - Vertical-style FOR+BitPack with Hash Joins
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, ddate, part, supplier
 *   WHERE lo_orderdate = d_datekey
 *     AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_brand1 = 'MFGR#2239'
 *     AND s_region = 'EUROPE'
 *   GROUP BY d_year, p_brand1
 *   ORDER BY d_year, p_brand1;
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

// L3 FLS headers
#include "common/fls_constants.cuh"
#include "common/fls_unpack.cuh"
#include "common/crystal_fls.cuh"
#include "common/ssb_data_fls.hpp"
#include "common/perfect_hash_table.cuh"

using namespace l3_fls;

// ============================================================================
// Q2.3 Constants
// ============================================================================

namespace {
constexpr int P_BRAND1_TARGET = 2239;  // 'MFGR#2239'
constexpr int S_REGION_EUROPE = 3;     // 'EUROPE' encoded as region 3

constexpr int S_HT_SIZE = 50021;
constexpr int P_HT_SIZE = 1000003;
constexpr int D_HT_SIZE = 2557;

// Single brand -> 7 years
constexpr int NUM_YEARS = 7;
}  // anonymous namespace

constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;
constexpr int TILE_SIZE = FLS_TILE_SIZE;

// ============================================================================
// Q2.3 Kernel (single brand - simpler aggregation)
// ============================================================================

__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q23_fls_kernel(
    const uint32_t* __restrict__ enc_lo_suppkey,
    const uint32_t* __restrict__ enc_lo_partkey,
    const uint32_t* __restrict__ enc_lo_orderdate,
    const uint32_t* __restrict__ enc_lo_revenue,
    int32_t lo_suppkey_min, uint8_t lo_suppkey_bw,
    int32_t lo_partkey_min, uint8_t lo_partkey_bw,
    int32_t lo_orderdate_min, uint8_t lo_orderdate_bw,
    int32_t lo_revenue_min, uint8_t lo_revenue_bw,
    int64_t n_tup_lineorder,
    int* __restrict__ s_ht, int* __restrict__ p_ht, int* __restrict__ d_ht,
    int s_ht_size, int p_ht_size, int d_ht_size,
    int s_min_key, int p_min_key, int d_min_key,
    unsigned long long* __restrict__ year_result)  // [year - 1992]
{
    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int brand1[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // Step 1: lo_suppkey - probe supplier hash table (EUROPE)
    {
        int64_t suppkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_suppkey_bw * 32;
        unpack_for(enc_lo_suppkey + suppkey_tile_offset, items, lo_suppkey_min, lo_suppkey_bw);
        BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, selection_flags, s_ht, s_ht_size, s_min_key, num_tile_items);
    }

    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 2: lo_partkey - probe part hash table (brand1 = 2239)
    {
        int64_t partkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_partkey_bw * 32;
        unpack_for(enc_lo_partkey + partkey_tile_offset, items, lo_partkey_min, lo_partkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, brand1, selection_flags, p_ht, p_ht_size, p_min_key, num_tile_items);
    }

    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 3: lo_orderdate - probe date hash table (year)
    {
        int64_t orderdate_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_orderdate_bw * 32;
        unpack_for(enc_lo_orderdate + orderdate_tile_offset, items, lo_orderdate_min, lo_orderdate_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, year, selection_flags, d_ht, d_ht_size, d_min_key, num_tile_items);
    }

    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 4: lo_revenue
    {
        int64_t revenue_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_revenue_bw * 32;
        unpack_for(enc_lo_revenue + revenue_tile_offset, items2, lo_revenue_min, lo_revenue_bw);
    }

    // Step 5: Aggregate by year only (single brand)
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            int y = year[ITEM] - 1992;
            if (y >= 0 && y < NUM_YEARS) {
                atomicAdd(&year_result[y], static_cast<unsigned long long>(items2[ITEM]));
            }
        }
    }
}

// ============================================================================
// Build Hash Tables (single brand filter)
// ============================================================================

void buildHashTables(
    SSBDataFLS& data,
    int*& d_s_ht, int*& d_p_ht, int*& d_d_ht,
    int& s_min_key, int& p_min_key, int& d_min_key)
{
    s_min_key = 1;
    p_min_key = 1;
    d_min_key = 19920101;

    cudaMalloc(&d_s_ht, S_HT_SIZE * sizeof(int));
    cudaMalloc(&d_p_ht, P_HT_SIZE * 2 * sizeof(int));
    cudaMalloc(&d_d_ht, D_HT_SIZE * 2 * sizeof(int));

    cudaMemset(d_s_ht, 0, S_HT_SIZE * sizeof(int));
    cudaMemset(d_p_ht, 0, P_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_d_ht, 0, D_HT_SIZE * 2 * sizeof(int));

    int block_size = 128;

    // Supplier: WHERE s_region = 'EUROPE'
    int s_num_blocks = (SSB_S_LEN + block_size - 1) / block_size;
    build_pht_1_kernel<int><<<s_num_blocks, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_REGION_EUROPE,
        SSB_S_LEN, d_s_ht, S_HT_SIZE, s_min_key);

    // Part: WHERE p_brand1 = 2239 (single brand)
    int p_num_blocks = (SSB_P_LEN + block_size - 1) / block_size;
    build_pht_2_kernel<int, int><<<p_num_blocks, block_size>>>(
        data.d_p_partkey, data.d_p_brand1, data.d_p_brand1, P_BRAND1_TARGET,
        SSB_P_LEN, d_p_ht, P_HT_SIZE, p_min_key);

    // Date: return year
    int d_num_blocks = (SSB_D_LEN + block_size - 1) / block_size;
    build_pht_2_all_kernel<int, int><<<d_num_blocks, block_size>>>(
        data.d_d_datekey, data.d_d_year,
        SSB_D_LEN, d_d_ht, D_HT_SIZE, d_min_key);

    cudaDeviceSynchronize();

    std::cout << "Hash tables built (Q2.3):" << std::endl;
    std::cout << "  Supplier: region=EUROPE" << std::endl;
    std::cout << "  Part: brand1=MFGR#2239" << std::endl;
}

void runQ23FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *d_s_ht, *d_p_ht, *d_d_ht;
    int s_min_key, p_min_key, d_min_key;
    buildHashTables(data, d_s_ht, d_p_ht, d_d_ht, s_min_key, p_min_key, d_min_key);

    unsigned long long* d_year_result;
    cudaMalloc(&d_year_result, NUM_YEARS * sizeof(unsigned long long));
    cudaMemset(d_year_result, 0, NUM_YEARS * sizeof(unsigned long long));

    int num_blocks = static_cast<int>(data.n_tiles);

    std::cout << "Q2.3 FLS Kernel Configuration:" << std::endl;
    std::cout << "  Blocks: " << num_blocks << ", Threads: " << BLOCK_THREADS << std::endl;

    // Warmup
    q23_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_suppkey.d_encoded, data.lo_partkey.d_encoded,
        data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
        data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
        data.lo_partkey.min_value, data.lo_partkey.bitwidth,
        data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
        data.lo_revenue.min_value, data.lo_revenue.bitwidth,
        data.n_tup_lineorder,
        d_s_ht, d_p_ht, d_d_ht, S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
        s_min_key, p_min_key, d_min_key, d_year_result);
    cudaDeviceSynchronize();

    // Timed runs
    float total_time = 0.0f;
    const int NUM_RUNS = 3;

    for (int run = 0; run < NUM_RUNS; run++) {
        cudaMemset(d_year_result, 0, NUM_YEARS * sizeof(unsigned long long));
        cudaEventRecord(start);

        q23_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_suppkey.d_encoded, data.lo_partkey.d_encoded,
            data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
            data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
            data.lo_partkey.min_value, data.lo_partkey.bitwidth,
            data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
            data.lo_revenue.min_value, data.lo_revenue.bitwidth,
            data.n_tup_lineorder,
            d_s_ht, d_p_ht, d_d_ht, S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
            s_min_key, p_min_key, d_min_key, d_year_result);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / NUM_RUNS;
    timing.total_ms = timing.kernel_ms;

    std::vector<unsigned long long> h_year_result(NUM_YEARS);
    cudaMemcpy(h_year_result.data(), d_year_result, NUM_YEARS * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.3 Results (FLS) ===" << std::endl;
    std::cout << "Average kernel time: " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "\nResults (year, brand1=MFGR#2239, revenue):" << std::endl;

    for (int y = 0; y < NUM_YEARS; y++) {
        if (h_year_result[y] > 0) {
            std::cout << "  " << (1992 + y) << ", MFGR#2239, " << h_year_result[y] << std::endl;
        }
    }

    cudaFree(d_s_ht);
    cudaFree(d_p_ht);
    cudaFree(d_d_ht);
    cudaFree(d_year_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.3 - Vertical Style FOR+BitPack ===" << std::endl;

    SSBDataFLS data;
    data.loadAndEncode(data_dir);
    data.printStats();

    std::cout << "\n--- Running Q2.3 ---" << std::endl;
    FLSQueryTiming timing;
    runQ23FLS(data, timing);

    data.free();
    return 0;
}
