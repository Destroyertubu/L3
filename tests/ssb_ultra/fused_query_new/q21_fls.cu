/**
 * @file q21_fls.cu
 * @brief SSB Q2.1 Implementation - Vertical-style FOR+BitPack with Hash Joins
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, ddate, part, supplier
 *   WHERE lo_orderdate = d_datekey
 *     AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_category = 'MFGR#12'
 *     AND s_region = 'AMERICA'
 *   GROUP BY d_year, p_brand1
 *   ORDER BY d_year, p_brand1;
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <algorithm>

// L3 FLS headers
#include "common/fls_constants.cuh"
#include "common/fls_unpack.cuh"
#include "common/crystal_fls.cuh"
#include "common/ssb_data_fls.hpp"
#include "common/perfect_hash_table.cuh"

using namespace l3_fls;

// ============================================================================
// Q2.1 Constants
// ============================================================================

namespace {
// Filter values (encoded as integers in SSB)
constexpr int P_CATEGORY_MFGR12 = 12;  // 'MFGR#12' encoded as category 12
constexpr int S_REGION_AMERICA = 1;     // 'AMERICA' encoded as region 1

// Hash table sizes (prime numbers for better distribution)
constexpr int S_HT_SIZE = 50021;        // Supplier hash table (40K suppliers)
constexpr int P_HT_SIZE = 1000003;      // Part hash table (1M parts, ~10% match)
constexpr int D_HT_SIZE = 2557;         // Date hash table

// Group result size: 7 years (1992-1998) * 1000 brands (MFGR#12 has brand1 ranges from 1201-1240)
constexpr int NUM_YEARS = 7;
constexpr int NUM_BRANDS = 40;  // brands 1201-1240
constexpr int GROUP_SIZE = NUM_YEARS * NUM_BRANDS;
}  // anonymous namespace

// Kernel configuration
constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;
constexpr int TILE_SIZE = FLS_TILE_SIZE;

// ============================================================================
// Q2.1 Kernel - Hash Join with Grouping
// ============================================================================

__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q21_fls_kernel(
    // Encoded LINEORDER columns
    const uint32_t* __restrict__ enc_lo_suppkey,
    const uint32_t* __restrict__ enc_lo_partkey,
    const uint32_t* __restrict__ enc_lo_orderdate,
    const uint32_t* __restrict__ enc_lo_revenue,
    // Encoding metadata
    int32_t lo_suppkey_min, uint8_t lo_suppkey_bw,
    int32_t lo_partkey_min, uint8_t lo_partkey_bw,
    int32_t lo_orderdate_min, uint8_t lo_orderdate_bw,
    int32_t lo_revenue_min, uint8_t lo_revenue_bw,
    int64_t n_tup_lineorder,
    // Hash tables
    int* __restrict__ s_ht,       // Supplier PHT_1 (existence)
    int* __restrict__ p_ht,       // Part PHT_2 (key,brand1)
    int* __restrict__ d_ht,       // Date PHT_2 (key,year)
    int s_ht_size, int p_ht_size, int d_ht_size,
    int s_min_key, int p_min_key, int d_min_key,
    // Result
    unsigned long long* __restrict__ group_result)  // [year * NUM_BRANDS + (brand - 1201)]
{
    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int brand1[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    __shared__ long long buffer[32];

    // Tile offset calculation
    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    // Initialize selection flags
    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // ========================================
    // Step 1: lo_suppkey - probe supplier hash table
    // ========================================
    {
        int64_t suppkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_suppkey_bw * 32;
        unpack_for(enc_lo_suppkey + suppkey_tile_offset, items, lo_suppkey_min, lo_suppkey_bw);

        BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, selection_flags, s_ht, s_ht_size, s_min_key, num_tile_items);
    }

    // Early termination
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Step 2: lo_partkey - probe part hash table (get brand1)
    // ========================================
    {
        int64_t partkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_partkey_bw * 32;
        unpack_for(enc_lo_partkey + partkey_tile_offset, items, lo_partkey_min, lo_partkey_bw);

        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, brand1, selection_flags, p_ht, p_ht_size, p_min_key, num_tile_items);
    }

    // Early termination
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Step 3: lo_orderdate - probe date hash table (get year)
    // ========================================
    {
        int64_t orderdate_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_orderdate_bw * 32;
        unpack_for(enc_lo_orderdate + orderdate_tile_offset, items, lo_orderdate_min, lo_orderdate_bw);

        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, year, selection_flags, d_ht, d_ht_size, d_min_key, num_tile_items);
    }

    // Early termination
    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) {
        return;
    }

    // ========================================
    // Step 4: lo_revenue - load revenue values
    // ========================================
    {
        int64_t revenue_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_revenue_bw * 32;
        unpack_for(enc_lo_revenue + revenue_tile_offset, items2, lo_revenue_min, lo_revenue_bw);
    }

    // ========================================
    // Step 5: Aggregate to group result
    // group_idx = (year - 1992) * NUM_BRANDS + (brand1 - 1201)
    // ========================================
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            int y = year[ITEM] - 1992;
            int b = brand1[ITEM] - 1201;
            if (y >= 0 && y < NUM_YEARS && b >= 0 && b < NUM_BRANDS) {
                int group_idx = y * NUM_BRANDS + b;
                atomicAdd(&group_result[group_idx], static_cast<unsigned long long>(items2[ITEM]));
            }
        }
    }
}

// ============================================================================
// Build Hash Tables
// ============================================================================

void buildHashTables(
    SSBDataFLS& data,
    int*& d_s_ht, int*& d_p_ht, int*& d_d_ht,
    int& s_min_key, int& p_min_key, int& d_min_key)
{
    // Get min keys
    s_min_key = 1;   // Supplier keys start from 1
    p_min_key = 1;   // Part keys start from 1
    d_min_key = 19920101;  // Date keys start from this

    // Allocate hash tables
    cudaMalloc(&d_s_ht, S_HT_SIZE * sizeof(int));
    cudaMalloc(&d_p_ht, P_HT_SIZE * 2 * sizeof(int));  // PHT_2 needs 2x space
    cudaMalloc(&d_d_ht, D_HT_SIZE * 2 * sizeof(int));  // PHT_2 needs 2x space

    cudaMemset(d_s_ht, 0, S_HT_SIZE * sizeof(int));
    cudaMemset(d_p_ht, 0, P_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_d_ht, 0, D_HT_SIZE * 2 * sizeof(int));

    int block_size = 128;

    // Build supplier hash table: WHERE s_region = 'AMERICA'
    int s_num_blocks = (SSB_S_LEN + block_size - 1) / block_size;
    build_pht_1_kernel<int><<<s_num_blocks, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_REGION_AMERICA,
        SSB_S_LEN, d_s_ht, S_HT_SIZE, s_min_key);

    // Build part hash table: WHERE p_category = 'MFGR#12', return p_brand1
    int p_num_blocks = (SSB_P_LEN + block_size - 1) / block_size;
    build_pht_2_kernel<int, int><<<p_num_blocks, block_size>>>(
        data.d_p_partkey, data.d_p_brand1, data.d_p_category, P_CATEGORY_MFGR12,
        SSB_P_LEN, d_p_ht, P_HT_SIZE, p_min_key);

    // Build date hash table: return d_year (no filter)
    int d_num_blocks = (SSB_D_LEN + block_size - 1) / block_size;
    build_pht_2_all_kernel<int, int><<<d_num_blocks, block_size>>>(
        data.d_d_datekey, data.d_d_year,
        SSB_D_LEN, d_d_ht, D_HT_SIZE, d_min_key);

    cudaDeviceSynchronize();

    std::cout << "Hash tables built:" << std::endl;
    std::cout << "  Supplier: " << S_HT_SIZE << " slots (PHT_1, region=AMERICA)" << std::endl;
    std::cout << "  Part: " << P_HT_SIZE << " slots (PHT_2, category=MFGR#12->brand1)" << std::endl;
    std::cout << "  Date: " << D_HT_SIZE << " slots (PHT_2, datekey->year)" << std::endl;
}

// ============================================================================
// Query Execution
// ============================================================================

void runQ21FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Build hash tables
    int *d_s_ht, *d_p_ht, *d_d_ht;
    int s_min_key, p_min_key, d_min_key;
    buildHashTables(data, d_s_ht, d_p_ht, d_d_ht, s_min_key, p_min_key, d_min_key);

    // Allocate group result
    unsigned long long* d_group_result;
    cudaMalloc(&d_group_result, GROUP_SIZE * sizeof(unsigned long long));
    cudaMemset(d_group_result, 0, GROUP_SIZE * sizeof(unsigned long long));

    int num_blocks = static_cast<int>(data.n_tiles);

    std::cout << "Q2.1 FLS Kernel Configuration:" << std::endl;
    std::cout << "  Blocks: " << num_blocks << std::endl;
    std::cout << "  Threads per block: " << BLOCK_THREADS << std::endl;
    std::cout << "  Tile size: " << TILE_SIZE << std::endl;

    // Warmup run
    q21_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_suppkey.d_encoded,
        data.lo_partkey.d_encoded,
        data.lo_orderdate.d_encoded,
        data.lo_revenue.d_encoded,
        data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
        data.lo_partkey.min_value, data.lo_partkey.bitwidth,
        data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
        data.lo_revenue.min_value, data.lo_revenue.bitwidth,
        data.n_tup_lineorder,
        d_s_ht, d_p_ht, d_d_ht,
        S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
        s_min_key, p_min_key, d_min_key,
        d_group_result
    );
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (warmup): " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Timed runs
    float total_time = 0.0f;
    const int NUM_RUNS = 3;

    for (int run = 0; run < NUM_RUNS; run++) {
        cudaMemset(d_group_result, 0, GROUP_SIZE * sizeof(unsigned long long));

        cudaEventRecord(start);

        q21_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_suppkey.d_encoded,
            data.lo_partkey.d_encoded,
            data.lo_orderdate.d_encoded,
            data.lo_revenue.d_encoded,
            data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
            data.lo_partkey.min_value, data.lo_partkey.bitwidth,
            data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
            data.lo_revenue.min_value, data.lo_revenue.bitwidth,
            data.n_tup_lineorder,
            d_s_ht, d_p_ht, d_d_ht,
            S_HT_SIZE, P_HT_SIZE, D_HT_SIZE,
            s_min_key, p_min_key, d_min_key,
            d_group_result
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / NUM_RUNS;
    timing.total_ms = timing.kernel_ms;

    // Copy and display results
    std::vector<unsigned long long> h_group_result(GROUP_SIZE);
    cudaMemcpy(h_group_result.data(), d_group_result, GROUP_SIZE * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 Results (FLS) ===" << std::endl;
    std::cout << "Average kernel time: " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "\nTop results (year, brand1, revenue):" << std::endl;

    // Print non-zero results
    int count = 0;
    for (int y = 0; y < NUM_YEARS && count < 20; y++) {
        for (int b = 0; b < NUM_BRANDS && count < 20; b++) {
            int idx = y * NUM_BRANDS + b;
            if (h_group_result[idx] > 0) {
                std::cout << "  " << (1992 + y) << ", MFGR#12" << (b + 1)
                          << ", " << h_group_result[idx] << std::endl;
                count++;
            }
        }
    }

    // Cleanup
    cudaFree(d_s_ht);
    cudaFree(d_p_ht);
    cudaFree(d_d_ht);
    cudaFree(d_group_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q2.1 - Vertical Style FOR+BitPack with Hash Joins ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;

    SSBDataFLS data;
    data.loadAndEncode(data_dir);
    data.printStats();

    std::cout << "\n--- Running Q2.1 ---" << std::endl;
    FLSQueryTiming timing;
    runQ21FLS(data, timing);

    data.free();

    return 0;
}
