/**
 * @file q21.cu
 * @brief SSB Q2.1 Implementation - Decompress First Strategy
 *
 * Query:
 *   SELECT SUM(lo_revenue), d_year, p_brand1
 *   FROM lineorder, date, part, supplier
 *   WHERE lo_orderdate = d_datekey
 *     AND lo_partkey = p_partkey
 *     AND lo_suppkey = s_suppkey
 *     AND p_category = 'MFGR#12'
 *     AND s_region = 'AMERICA'
 *   GROUP BY d_year, p_brand1
 *   ORDER BY d_year, p_brand1;
 *
 * Strategy: Decompress all required columns first, build hash tables for dimension
 * tables, then run Crystal-style query kernel with hash probes
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

// SSB common headers
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"

// L3 headers
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

// ============================================================================
// Constants for Q2.1
// ============================================================================

// p_category encoding: 'MFGR#12' is category 12 in MFGR range
// In SSB generator: p_category = 'MFGR#' + (1 to 5) + (1 to 5) encoded as integers
// MFGR#12 = manufacturer 1, category 2 -> encoded as some integer
// Let's use the integer encoding from the data generator

constexpr uint32_t P_CATEGORY_MFGR12 = 12;  // Encoded category value
constexpr uint32_t S_REGION_AMERICA = 1;     // AMERICA region code

// Group by dimensions
constexpr int NUM_YEARS = 7;  // 1992-1998
constexpr int NUM_BRANDS = 1000;  // Maximum brand values
constexpr int HASH_TABLE_SIZE = NUM_YEARS * NUM_BRANDS * 2;  // With load factor

// ============================================================================
// Q2.1 Query Kernel
// ============================================================================

/**
 * @brief Q2.1 query kernel with hash table probes
 */
__global__ void q21_kernel(
    const uint32_t* __restrict__ lo_orderdate,
    const uint32_t* __restrict__ lo_partkey,
    const uint32_t* __restrict__ lo_suppkey,
    const uint32_t* __restrict__ lo_revenue,
    int num_rows,
    // Hash tables for dimension tables
    const uint32_t* __restrict__ ht_d_keys,    // d_datekey
    const uint32_t* __restrict__ ht_d_values,  // d_year
    int ht_d_size,
    const uint32_t* __restrict__ ht_p_keys,    // p_partkey
    const uint32_t* __restrict__ ht_p_values,  // p_brand1
    int ht_p_size,
    const uint32_t* __restrict__ ht_s_keys,    // s_suppkey (filtered by region)
    int ht_s_size,
    // Output: aggregation results
    unsigned long long* agg_revenue,  // [year-1992][brand] = revenue
    int* agg_count)
{
    // Items processed by this thread
    uint32_t orderdate[ITEMS_PER_THREAD];
    uint32_t partkey[ITEMS_PER_THREAD];
    uint32_t suppkey[ITEMS_PER_THREAD];
    uint32_t revenue[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    // Calculate block range
    int block_start = blockIdx.x * TILE_SIZE;
    int block_items = min(TILE_SIZE, num_rows - block_start);

    if (block_items <= 0) return;

    // Initialize selection flags
    InitFlags(selection_flags, block_items);

    // Load all columns
    BlockLoad(lo_orderdate + block_start, orderdate, block_items);
    BlockLoad(lo_partkey + block_start, partkey, block_items);
    BlockLoad(lo_suppkey + block_start, suppkey, block_items);
    BlockLoad(lo_revenue + block_start, revenue, block_items);

    // Process each item
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x + i * BLOCK_THREADS;
        if (idx >= block_items || !selection_flags[i]) continue;

        // Probe supplier hash table (s_region = AMERICA filter)
        uint32_t sk = suppkey[i];
        int s_slot = hash_murmur3(sk, ht_s_size);
        bool s_found = false;
        for (int probe = 0; probe < ht_s_size; ++probe) {
            int slot = (s_slot + probe) % ht_s_size;
            if (ht_s_keys[slot] == sk) {
                s_found = true;
                break;
            }
            if (ht_s_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!s_found) {
            selection_flags[i] = 0;
            continue;
        }

        // Probe part hash table (p_category = MFGR#12 filter, get p_brand1)
        uint32_t pk = partkey[i];
        int p_slot = hash_murmur3(pk, ht_p_size);
        uint32_t p_brand = 0;
        bool p_found = false;
        for (int probe = 0; probe < ht_p_size; ++probe) {
            int slot = (p_slot + probe) % ht_p_size;
            if (ht_p_keys[slot] == pk) {
                p_brand = ht_p_values[slot];
                p_found = true;
                break;
            }
            if (ht_p_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!p_found) {
            selection_flags[i] = 0;
            continue;
        }

        // Probe date hash table (get d_year)
        uint32_t od = orderdate[i];
        int d_slot = hash_murmur3(od, ht_d_size);
        uint32_t d_year = 0;
        bool d_found = false;
        for (int probe = 0; probe < ht_d_size; ++probe) {
            int slot = (d_slot + probe) % ht_d_size;
            if (ht_d_keys[slot] == od) {
                d_year = ht_d_values[slot];
                d_found = true;
                break;
            }
            if (ht_d_keys[slot] == static_cast<uint32_t>(HT_EMPTY)) break;
        }
        if (!d_found) {
            selection_flags[i] = 0;
            continue;
        }

        // Aggregate: add to (year, brand) bucket
        int year_idx = d_year - 1992;
        if (year_idx >= 0 && year_idx < NUM_YEARS && p_brand < NUM_BRANDS) {
            int agg_idx = year_idx * NUM_BRANDS + p_brand;
            atomicAdd(&agg_revenue[agg_idx], static_cast<unsigned long long>(revenue[i]));
            atomicAdd(&agg_count[agg_idx], 1);
        }
    }
}

// ============================================================================
// Hash Table Build Kernels
// ============================================================================

/**
 * @brief Build date hash table (all dates, maps datekey -> year)
 */
__global__ void build_date_ht_kernel(
    const uint32_t* d_datekey,
    const uint32_t* d_year,
    int num_rows,
    uint32_t* ht_keys,
    uint32_t* ht_values,
    int ht_size)
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

/**
 * @brief Build part hash table (filtered by p_category = MFGR#12, maps partkey -> brand1)
 */
__global__ void build_part_ht_kernel(
    const uint32_t* p_partkey,
    const uint32_t* p_category,
    const uint32_t* p_brand1,
    int num_rows,
    uint32_t filter_category,
    uint32_t* ht_keys,
    uint32_t* ht_values,
    int ht_size)
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

/**
 * @brief Build supplier hash table (filtered by s_region = AMERICA)
 */
__global__ void build_supplier_ht_kernel(
    const uint32_t* s_suppkey,
    const uint32_t* s_region,
    int num_rows,
    uint32_t filter_region,
    uint32_t* ht_keys,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_region[idx] != filter_region) return;

    uint32_t key = s_suppkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], static_cast<uint32_t>(HT_EMPTY), key);
        if (old == static_cast<uint32_t>(HT_EMPTY) || old == key) {
            return;
        }
    }
}

// ============================================================================
// Q2.1 Execution Function
// ============================================================================

void runQ21DecompressFirst(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;

    // -------------------------------------------------------------------------
    // Step 1: Allocate and decompress LINEORDER columns
    // -------------------------------------------------------------------------
    uint32_t* d_lo_orderdate;
    uint32_t* d_lo_partkey;
    uint32_t* d_lo_suppkey;
    uint32_t* d_lo_revenue;

    cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(uint32_t));
    cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(uint32_t));

    timer.start();

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_partkey(&data.lo_partkey);
    CompressedColumnAccessorVertical<uint32_t> acc_suppkey(&data.lo_suppkey);
    CompressedColumnAccessorVertical<uint32_t> acc_revenue(&data.lo_revenue);

    acc_orderdate.decompressAll(d_lo_orderdate);
    acc_partkey.decompressAll(d_lo_partkey);
    acc_suppkey.decompressAll(d_lo_suppkey);
    acc_revenue.decompressAll(d_lo_revenue);

    cudaDeviceSynchronize();
    timer.stop();
    timing.data_load_ms = timer.elapsed_ms();

    // -------------------------------------------------------------------------
    // Step 2: Build hash tables for dimension tables
    // -------------------------------------------------------------------------
    timer.start();

    // Date hash table
    int ht_d_size = D_LEN * 2;
    uint32_t* ht_d_keys;
    uint32_t* ht_d_values;
    cudaMalloc(&ht_d_keys, ht_d_size * sizeof(uint32_t));
    cudaMalloc(&ht_d_values, ht_d_size * sizeof(uint32_t));
    cudaMemset(ht_d_keys, 0xFF, ht_d_size * sizeof(uint32_t));  // HT_EMPTY = -1 = 0xFFFFFFFF

    int block_size = 256;
    int grid_size_d = (D_LEN + block_size - 1) / block_size;
    build_date_ht_kernel<<<grid_size_d, block_size>>>(
        data.d_d_datekey, data.d_d_year, D_LEN,
        ht_d_keys, ht_d_values, ht_d_size
    );

    // Part hash table (filtered)
    int ht_p_size = P_LEN * 2;
    uint32_t* ht_p_keys;
    uint32_t* ht_p_values;
    cudaMalloc(&ht_p_keys, ht_p_size * sizeof(uint32_t));
    cudaMalloc(&ht_p_values, ht_p_size * sizeof(uint32_t));
    cudaMemset(ht_p_keys, 0xFF, ht_p_size * sizeof(uint32_t));

    int grid_size_p = (P_LEN + block_size - 1) / block_size;
    build_part_ht_kernel<<<grid_size_p, block_size>>>(
        data.d_p_partkey, data.d_p_category, data.d_p_brand1, P_LEN,
        P_CATEGORY_MFGR12,
        ht_p_keys, ht_p_values, ht_p_size
    );

    // Supplier hash table (filtered)
    int ht_s_size = S_LEN * 2;
    uint32_t* ht_s_keys;
    cudaMalloc(&ht_s_keys, ht_s_size * sizeof(uint32_t));
    cudaMemset(ht_s_keys, 0xFF, ht_s_size * sizeof(uint32_t));

    int grid_size_s = (S_LEN + block_size - 1) / block_size;
    build_supplier_ht_kernel<<<grid_size_s, block_size>>>(
        data.d_s_suppkey, data.d_s_region, S_LEN,
        S_REGION_AMERICA,
        ht_s_keys, ht_s_size
    );

    cudaDeviceSynchronize();
    timer.stop();
    timing.hash_build_ms = timer.elapsed_ms();

    // -------------------------------------------------------------------------
    // Step 3: Allocate aggregation arrays
    // -------------------------------------------------------------------------
    int agg_size = NUM_YEARS * NUM_BRANDS;
    unsigned long long* d_agg_revenue;
    int* d_agg_count;
    cudaMalloc(&d_agg_revenue, agg_size * sizeof(unsigned long long));
    cudaMalloc(&d_agg_count, agg_size * sizeof(int));
    cudaMemset(d_agg_revenue, 0, agg_size * sizeof(unsigned long long));
    cudaMemset(d_agg_count, 0, agg_size * sizeof(int));

    // -------------------------------------------------------------------------
    // Step 4: Execute query kernel
    // -------------------------------------------------------------------------
    timer.start();

    int grid_size = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;
    q21_kernel<<<grid_size, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue,
        LO_LEN,
        ht_d_keys, ht_d_values, ht_d_size,
        ht_p_keys, ht_p_values, ht_p_size,
        ht_s_keys, ht_s_size,
        d_agg_revenue, d_agg_count
    );

    cudaDeviceSynchronize();
    timer.stop();
    timing.kernel_ms = timer.elapsed_ms();
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;

    // -------------------------------------------------------------------------
    // Step 5: Collect results
    // -------------------------------------------------------------------------
    std::vector<unsigned long long> h_agg_revenue(agg_size);
    std::vector<int> h_agg_count(agg_size);
    cudaMemcpy(h_agg_revenue.data(), d_agg_revenue, agg_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agg_count.data(), d_agg_count, agg_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n=== Q2.1 Results (Decompress-First) ===" << std::endl;
    std::cout << "Year\tBrand\tRevenue" << std::endl;

    int num_results = 0;
    unsigned long long total_revenue = 0;
    for (int y = 0; y < NUM_YEARS; ++y) {
        for (int b = 0; b < NUM_BRANDS; ++b) {
            int idx = y * NUM_BRANDS + b;
            if (h_agg_count[idx] > 0) {
                num_results++;
                total_revenue += h_agg_revenue[idx];
                if (num_results <= 10) {  // Show first 10 results
                    std::cout << (1992 + y) << "\t" << b << "\t" << h_agg_revenue[idx] << std::endl;
                }
            }
        }
    }
    if (num_results > 10) {
        std::cout << "... and " << (num_results - 10) << " more results" << std::endl;
    }
    std::cout << "Total groups: " << num_results << ", Total revenue: " << total_revenue << std::endl;
    timing.print("Q2.1");

    // Cleanup
    cudaFree(d_lo_orderdate);
    cudaFree(d_lo_partkey);
    cudaFree(d_lo_suppkey);
    cudaFree(d_lo_revenue);
    cudaFree(ht_d_keys);
    cudaFree(ht_d_values);
    cudaFree(ht_p_keys);
    cudaFree(ht_p_values);
    cudaFree(ht_s_keys);
    cudaFree(d_agg_revenue);
    cudaFree(d_agg_count);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";

    if (argc > 1) {
        data_dir = argv[1];
    }

    std::cout << "=== SSB Q2.1 - Decompress First Strategy ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;

    // Load and compress SSB data
    SSBDataCompressedVertical data;

    auto start = std::chrono::high_resolution_clock::now();
    data.loadAndCompress(data_dir);
    auto end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "\nData loading + compression time: " << load_time << " ms" << std::endl;
    std::cout << "Compression ratio: " << data.getCompressionRatio() << "x" << std::endl;

    // Run Q2.1
    QueryTiming timing;
    runQ21DecompressFirst(data, timing);

    // Warmup and benchmark
    std::cout << "\n=== Benchmark (3 runs) ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ21DecompressFirst(data, t);
        std::cout << "Run " << (i+1) << ": Total=" << t.total_ms << " ms"
                  << " (Decompress=" << t.data_load_ms << " ms, HashBuild=" << t.hash_build_ms
                  << " ms, Kernel=" << t.kernel_ms << " ms)" << std::endl;
    }

    // Cleanup
    data.free();

    return 0;
}
