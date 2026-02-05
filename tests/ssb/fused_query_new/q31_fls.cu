/**
 * @file q31_fls.cu
 * @brief SSB Q3.1 Implementation - Vertical-style FOR+BitPack with Multi-way Hash Joins
 *
 * Query:
 *   SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
 *   FROM customer, lineorder, supplier, ddate
 *   WHERE lo_custkey = c_custkey
 *     AND lo_suppkey = s_suppkey
 *     AND lo_orderdate = d_datekey
 *     AND c_region = 'ASIA'
 *     AND s_region = 'ASIA'
 *     AND d_year >= 1992 AND d_year <= 1997
 *   GROUP BY c_nation, s_nation, d_year
 *   ORDER BY d_year ASC, revenue DESC;
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
// Q3.1 Constants
// ============================================================================

namespace {
// Filter values
constexpr int C_REGION_ASIA = 2;     // 'ASIA'
constexpr int S_REGION_ASIA = 2;     // 'ASIA'
constexpr int D_YEAR_MIN = 1992;
constexpr int D_YEAR_MAX = 1997;

// Hash table sizes
constexpr int C_HT_SIZE = 600037;    // Customer (600K)
constexpr int S_HT_SIZE = 50021;     // Supplier (40K)
constexpr int D_HT_SIZE = 2557;      // Date

// Group dimensions: 5 nations (ASIA) * 5 nations * 6 years (1992-1997)
// ASIA nations: INDONESIA(0), VIETNAM(1), CHINA(2), INDIA(3), JAPAN(4)
constexpr int NUM_NATIONS = 5;
constexpr int NUM_YEARS = 6;  // 1992-1997
constexpr int GROUP_SIZE = NUM_NATIONS * NUM_NATIONS * NUM_YEARS;
}  // anonymous namespace

constexpr int BLOCK_THREADS = FLS_BLOCK_THREADS;
constexpr int ITEMS_PER_THREAD = FLS_ITEMS_PER_THREAD;
constexpr int TILE_SIZE = FLS_TILE_SIZE;

// ============================================================================
// Q3.1 Kernel
// ============================================================================

__global__ __launch_bounds__(BLOCK_THREADS, 8)
void q31_fls_kernel(
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
    unsigned long long* __restrict__ group_result)  // [c_nation * NUM_NATIONS * NUM_YEARS + s_nation * NUM_YEARS + (year-1992)]
{
    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int s_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset = blockIdx.x * TILE_SIZE;
    int64_t num_tiles = (n_tup_lineorder + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;
    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = static_cast<int>(n_tup_lineorder - tile_offset);
    }

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags, num_tile_items);

    // Step 1: lo_custkey - probe customer hash table (ASIA -> c_nation)
    {
        int64_t custkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_custkey_bw * 32;
        unpack_for(enc_lo_custkey + custkey_tile_offset, items, lo_custkey_min, lo_custkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, c_nation, selection_flags, c_ht, c_ht_size, c_min_key, num_tile_items);
    }

    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 2: lo_suppkey - probe supplier hash table (ASIA -> s_nation)
    {
        int64_t suppkey_tile_offset = static_cast<int64_t>(blockIdx.x) * lo_suppkey_bw * 32;
        unpack_for(enc_lo_suppkey + suppkey_tile_offset, items, lo_suppkey_min, lo_suppkey_bw);
        BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
            items, s_nation, selection_flags, s_ht, s_ht_size, s_min_key, num_tile_items);
    }

    if (IsTerm<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags)) return;

    // Step 3: lo_orderdate - probe date hash table (year 1992-1997)
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

    // Step 5: Aggregate by (c_nation, s_nation, year)
    // group_idx = c_nation * NUM_NATIONS * NUM_YEARS + s_nation * NUM_YEARS + (year - 1992)
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items) && selection_flags[ITEM]) {
            int cn = c_nation[ITEM];
            int sn = s_nation[ITEM];
            int y = year[ITEM] - D_YEAR_MIN;
            if (cn >= 0 && cn < NUM_NATIONS && sn >= 0 && sn < NUM_NATIONS && y >= 0 && y < NUM_YEARS) {
                int group_idx = cn * NUM_NATIONS * NUM_YEARS + sn * NUM_YEARS + y;
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
    int*& d_c_ht, int*& d_s_ht, int*& d_d_ht,
    int& c_min_key, int& s_min_key, int& d_min_key)
{
    c_min_key = 1;
    s_min_key = 1;
    d_min_key = 19920101;

    cudaMalloc(&d_c_ht, C_HT_SIZE * 2 * sizeof(int));  // PHT_2
    cudaMalloc(&d_s_ht, S_HT_SIZE * 2 * sizeof(int));  // PHT_2
    cudaMalloc(&d_d_ht, D_HT_SIZE * 2 * sizeof(int));  // PHT_2

    cudaMemset(d_c_ht, 0, C_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_s_ht, 0, S_HT_SIZE * 2 * sizeof(int));
    cudaMemset(d_d_ht, 0, D_HT_SIZE * 2 * sizeof(int));

    int block_size = 128;

    // Customer: WHERE c_region = 'ASIA', return c_nation
    int c_num_blocks = (SSB_C_LEN + block_size - 1) / block_size;
    build_pht_2_kernel<int, int><<<c_num_blocks, block_size>>>(
        data.d_c_custkey, data.d_c_nation, data.d_c_region, C_REGION_ASIA,
        SSB_C_LEN, d_c_ht, C_HT_SIZE, c_min_key);

    // Supplier: WHERE s_region = 'ASIA', return s_nation
    int s_num_blocks = (SSB_S_LEN + block_size - 1) / block_size;
    build_pht_2_kernel<int, int><<<s_num_blocks, block_size>>>(
        data.d_s_suppkey, data.d_s_nation, data.d_s_region, S_REGION_ASIA,
        SSB_S_LEN, d_s_ht, S_HT_SIZE, s_min_key);

    // Date: WHERE d_year BETWEEN 1992 AND 1997, return d_year
    int d_num_blocks = (SSB_D_LEN + block_size - 1) / block_size;
    build_pht_2_range_kernel<int, int><<<d_num_blocks, block_size>>>(
        data.d_d_datekey, data.d_d_year, data.d_d_year, D_YEAR_MIN, D_YEAR_MAX,
        SSB_D_LEN, d_d_ht, D_HT_SIZE, d_min_key);

    cudaDeviceSynchronize();

    std::cout << "Hash tables built (Q3.1):" << std::endl;
    std::cout << "  Customer: region=ASIA->nation" << std::endl;
    std::cout << "  Supplier: region=ASIA->nation" << std::endl;
    std::cout << "  Date: year in [1992, 1997]" << std::endl;
}

void runQ31FLS(SSBDataFLS& data, FLSQueryTiming& timing) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int *d_c_ht, *d_s_ht, *d_d_ht;
    int c_min_key, s_min_key, d_min_key;
    buildHashTables(data, d_c_ht, d_s_ht, d_d_ht, c_min_key, s_min_key, d_min_key);

    unsigned long long* d_group_result;
    cudaMalloc(&d_group_result, GROUP_SIZE * sizeof(unsigned long long));
    cudaMemset(d_group_result, 0, GROUP_SIZE * sizeof(unsigned long long));

    int num_blocks = static_cast<int>(data.n_tiles);

    std::cout << "Q3.1 FLS Kernel Configuration:" << std::endl;
    std::cout << "  Blocks: " << num_blocks << ", Threads: " << BLOCK_THREADS << std::endl;

    // Warmup
    q31_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
        data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
        data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
        data.lo_custkey.min_value, data.lo_custkey.bitwidth,
        data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
        data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
        data.lo_revenue.min_value, data.lo_revenue.bitwidth,
        data.n_tup_lineorder,
        d_c_ht, d_s_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, D_HT_SIZE,
        c_min_key, s_min_key, d_min_key, d_group_result);
    cudaDeviceSynchronize();

    // Timed runs
    float total_time = 0.0f;
    const int NUM_RUNS = 3;

    for (int run = 0; run < NUM_RUNS; run++) {
        cudaMemset(d_group_result, 0, GROUP_SIZE * sizeof(unsigned long long));
        cudaEventRecord(start);

        q31_fls_kernel<<<num_blocks, BLOCK_THREADS>>>(
            data.lo_custkey.d_encoded, data.lo_suppkey.d_encoded,
            data.lo_orderdate.d_encoded, data.lo_revenue.d_encoded,
            data.lo_custkey.min_value, data.lo_custkey.bitwidth,
            data.lo_suppkey.min_value, data.lo_suppkey.bitwidth,
            data.lo_orderdate.min_value, data.lo_orderdate.bitwidth,
            data.lo_revenue.min_value, data.lo_revenue.bitwidth,
            data.n_tup_lineorder,
            d_c_ht, d_s_ht, d_d_ht, C_HT_SIZE, S_HT_SIZE, D_HT_SIZE,
            c_min_key, s_min_key, d_min_key, d_group_result);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_time += ms;
        std::cout << "Run " << (run + 1) << ": " << ms << " ms" << std::endl;
    }

    timing.kernel_ms = total_time / NUM_RUNS;
    timing.total_ms = timing.kernel_ms;

    std::vector<unsigned long long> h_group_result(GROUP_SIZE);
    cudaMemcpy(h_group_result.data(), d_group_result, GROUP_SIZE * sizeof(unsigned long long),
               cudaMemcpyDeviceToHost);

    // Nation names for ASIA
    const char* nations[] = {"INDONESIA", "VIETNAM", "CHINA", "INDIA", "JAPAN"};

    std::cout << "\n=== Q3.1 Results (FLS) ===" << std::endl;
    std::cout << "Average kernel time: " << timing.kernel_ms << " ms" << std::endl;
    std::cout << "\nTop results (c_nation, s_nation, year, revenue):" << std::endl;

    int count = 0;
    for (int y = 0; y < NUM_YEARS && count < 30; y++) {
        for (int cn = 0; cn < NUM_NATIONS && count < 30; cn++) {
            for (int sn = 0; sn < NUM_NATIONS && count < 30; sn++) {
                int idx = cn * NUM_NATIONS * NUM_YEARS + sn * NUM_YEARS + y;
                if (h_group_result[idx] > 0) {
                    std::cout << "  " << nations[cn] << ", " << nations[sn] << ", "
                              << (D_YEAR_MIN + y) << ", " << h_group_result[idx] << std::endl;
                    count++;
                }
            }
        }
    }

    cudaFree(d_c_ht);
    cudaFree(d_s_ht);
    cudaFree(d_d_ht);
    cudaFree(d_group_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q3.1 - Vertical Style FOR+BitPack ===" << std::endl;

    SSBDataFLS data;
    data.loadAndEncode(data_dir);
    data.printStats();

    std::cout << "\n--- Running Q3.1 ---" << std::endl;
    FLSQueryTiming timing;
    runQ31FLS(data, timing);

    data.free();
    return 0;
}
