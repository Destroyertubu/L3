/**
 * SSB Q3.1 Query with nvcomp Cascade Compression
 *
 * SQL:
 * SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
 * FROM lineorder, customer, supplier, date
 * WHERE lo_custkey = c_custkey
 *   AND lo_suppkey = s_suppkey
 *   AND lo_orderdate = d_datekey
 *   AND c_region = 'ASIA'
 *   AND s_region = 'ASIA'
 *   AND d_year >= 1992 AND d_year <= 1997
 * GROUP BY c_nation, s_nation, d_year
 */

#include "crystal/crystal.cuh"
#include "nvcomp_ssb.cuh"
#include "ssb_cascade_config.h"
#include "ssb_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// Build hash table for supplier dimension (filter by s_region = 'ASIA', store s_nation)
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int* filter_col, int* dim_key, int* dim_val,
                                  int num_tuples, int* hash_table, int num_slots) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // Filter: s_region = 'ASIA' (encoded as 2)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

// Build hash table for customer dimension (filter by c_region = 'ASIA', store c_nation)
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c(int* filter_col, int* dim_key, int* dim_val,
                                  int num_tuples, int* hash_table, int num_slots) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // Filter: c_region = 'ASIA' (encoded as 2)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 2, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

// Build hash table for date dimension (filter by d_year in [1992,1997], store d_year)
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d(int* dim_key, int* dim_val, int num_tuples,
                                  int* hash_table, int num_slots, int val_min) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int items2[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // Filter: d_year >= 1992 AND d_year <= 1997
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1992, selection_flags, num_tile_items);
    BlockPredAndLTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1997, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items2, num_tile_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items2, items, selection_flags, hash_table, num_slots, val_min, num_tile_items);
}

// Probe kernel: join and aggregate
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q31_Kernel(const int* d_lo_orderdate,
                           const int* d_lo_custkey,
                           const int* d_lo_suppkey,
                           const int* d_lo_revenue,
                           int        lo_len,
                           int*       ht_s,
                           int        s_len,
                           int*       ht_c,
                           int        c_len,
                           int*       ht_d,
                           int        d_len,
                           int        d_val_min,
                           int*       res) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int s_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int revenue[ITEMS_PER_THREAD];
    __shared__ int shared_any[32];
    __shared__ int block_has_valid;

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = lo_len - tile_offset;
    }

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

    // Probe customer hash table (get c_nation)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_custkey) + tile_offset, items, num_tile_items);
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, c_nation, selection_flags, ht_c, c_len, num_tile_items);
    int local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    int any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // Probe supplier hash table (get s_nation)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_suppkey) + tile_offset, items, num_tile_items);
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, s_nation, selection_flags, ht_s, s_len, num_tile_items);
    local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // Probe date hash table (get year)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_orderdate) + tile_offset, items, num_tile_items);
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, year, selection_flags, ht_d, d_len, d_val_min, num_tile_items);
    local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // Late materialization: load revenue only for valid rows
    BlockPredLoad<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_revenue) + tile_offset, revenue, num_tile_items, selection_flags);

    // Aggregate by (c_nation, s_nation, year)
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                int hash = (s_nation[ITEM] * 25 * 7 + c_nation[ITEM] * 7 + (year[ITEM] - 1992)) %
                           ((1998 - 1992 + 1) * 25 * 25);
                res[hash * 6]     = year[ITEM];
                res[hash * 6 + 1] = c_nation[ITEM];
                res[hash * 6 + 2] = s_nation[ITEM];
                atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]),
                          static_cast<unsigned long long>(revenue[ITEM]));
            }
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int* d_lo_orderdate, int* d_lo_custkey, int* d_lo_suppkey,
              int* d_lo_revenue, int lo_len,
              int* d_d_datekey, int* d_d_year, int d_len,
              int* d_s_suppkey, int* d_s_region, int* d_s_nation, int s_len,
              int* d_c_custkey, int* d_c_region, int* d_c_nation, int c_len,
              float time_h2d, float time_decompress) {
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate hash tables
    int  d_val_len = 19981230 - 19920101 + 1;
    int *ht_d, *ht_c, *ht_s;
    CUDA_CHECK(cudaMalloc(&ht_d, 2 * d_val_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_c, 2 * c_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_s, 2 * s_len * sizeof(int)));

    // Time hash table build
    cudaEvent_t start_ht, stop_ht;
    cudaEventCreate(&start_ht);
    cudaEventCreate(&stop_ht);

    cudaEventRecord(start_ht);
    CUDA_CHECK(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
    CUDA_CHECK(cudaMemset(ht_c, 0, 2 * c_len * sizeof(int)));
    CUDA_CHECK(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

    build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(s_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
            d_s_region, d_s_suppkey, d_s_nation, s_len, ht_s, s_len);

    build_hashtable_c<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(c_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
            d_c_region, d_c_custkey, d_c_nation, c_len, ht_c, c_len);

    int d_val_min = 19920101;
    build_hashtable_d<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(d_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
            d_d_datekey, d_d_year, d_len, ht_d, d_val_len, d_val_min);

    cudaEventRecord(stop_ht);
    cudaEventSynchronize(stop_ht);
    float time_ht_build;
    cudaEventElapsedTime(&time_ht_build, start_ht, stop_ht);
    cudaEventDestroy(start_ht);
    cudaEventDestroy(stop_ht);

    // Allocate result buffer
    int  res_size       = ((1998 - 1992 + 1) * 25 * 25);
    int  res_array_size = res_size * 6;
    int* d_res;
    CUDA_CHECK(cudaMalloc(&d_res, res_array_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

    int num_blocks = (lo_len + tile_items - 1) / tile_items;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    Q31_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, lo_len,
        ht_s, s_len, ht_c, c_len, ht_d, d_val_len, d_val_min, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

    const int NUM_RUNS = 3;
    for (int run = 0; run < NUM_RUNS; run++) {
        CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

        cudaEventRecord(start);
        Q31_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
            d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, lo_len,
            ht_s, s_len, ht_c, c_len, ht_d, d_val_len, d_val_min, d_res);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_kernel;
        cudaEventElapsedTime(&time_kernel, start, stop);

        // D2H transfer
        cudaEvent_t start_d2h, stop_d2h;
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_d2h);
        cudaEventRecord(start_d2h);
        int* h_res = new int[res_array_size];
        CUDA_CHECK(cudaMemcpy(h_res, d_res, res_array_size * sizeof(int),
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(stop_d2h);
        cudaEventSynchronize(stop_d2h);
        float time_d2h;
        cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);

        // Count results
        int                result_rows   = 0;
        unsigned long long total_revenue = 0;
        for (int i = 0; i < res_size; i++) {
            if (h_res[6 * i] != 0) {
                result_rows++;
                total_revenue += reinterpret_cast<unsigned long long*>(&h_res[6 * i + 4])[0];
            }
        }

        float time_total = time_h2d + time_decompress + time_ht_build + time_kernel + time_d2h;

        cout << "-- Q31 result_rows: " << result_rows << ", total_revenue: " << total_revenue << endl;
        cout << "{\"query\":31,\"run\":" << run
             << ",\"time_h2d\":" << time_h2d
             << ",\"time_decompress\":" << time_decompress
             << ",\"time_ht_build\":" << time_ht_build
             << ",\"time_kernel\":" << time_kernel
             << ",\"time_d2h\":" << time_d2h
             << ",\"time_total\":" << time_total << "}" << endl;

        delete[] h_res;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(ht_d));
    CUDA_CHECK(cudaFree(ht_c));
    CUDA_CHECK(cudaFree(ht_s));
}

int main() {
    cout << "=== SSB Q3.1 with nvcomp Cascade ===" << endl;

    // Load lineorder columns
    cout << "Loading lineorder data..." << endl;
    int* h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    int* h_lo_custkey   = loadColumn<int>("lo_custkey", LO_LEN);
    int* h_lo_suppkey   = loadColumn<int>("lo_suppkey", LO_LEN);
    int* h_lo_revenue   = loadColumn<int>("lo_revenue", LO_LEN);

    // Load dimension tables
    cout << "Loading dimension tables..." << endl;
    int* h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    int* h_d_year    = loadColumn<int>("d_year", D_LEN);

    int* h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
    int* h_s_region  = loadColumn<int>("s_region", S_LEN);
    int* h_s_nation  = loadColumn<int>("s_nation", S_LEN);

    int* h_c_custkey = loadColumn<int>("c_custkey", C_LEN);
    int* h_c_region  = loadColumn<int>("c_region", C_LEN);
    int* h_c_nation  = loadColumn<int>("c_nation", C_LEN);

    cudaFree(0);

    // Compress lineorder columns
    cout << "Compressing lineorder columns..." << endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CompressedColumn comp_orderdate = nvcomp_ssb::compressColumn(
        h_lo_orderdate, LO_LEN, ssb_cascade_config::orderdate_config, stream);
    CompressedColumn comp_custkey = nvcomp_ssb::compressColumn(
        h_lo_custkey, LO_LEN, ssb_cascade_config::custkey_config, stream);
    CompressedColumn comp_suppkey = nvcomp_ssb::compressColumn(
        h_lo_suppkey, LO_LEN, ssb_cascade_config::suppkey_config, stream);
    CompressedColumn comp_revenue = nvcomp_ssb::compressColumn(
        h_lo_revenue, LO_LEN, ssb_cascade_config::revenue_config, stream);

    cout << "Compression statistics:" << endl;
    nvcomp_ssb::printCompressionStats("orderdate", comp_orderdate);
    nvcomp_ssb::printCompressionStats("custkey", comp_custkey);
    nvcomp_ssb::printCompressionStats("suppkey", comp_suppkey);
    nvcomp_ssb::printCompressionStats("revenue", comp_revenue);

    float time_h2d = 0;

    // Allocate and transfer dimension tables to GPU
    int *d_d_datekey, *d_d_year;
    int *d_s_suppkey, *d_s_region, *d_s_nation;
    int *d_c_custkey, *d_c_region, *d_c_nation;

    CUDA_CHECK(cudaMalloc(&d_d_datekey, D_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_d_year, D_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_suppkey, S_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_region, S_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_nation, S_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c_custkey, C_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c_region, C_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_c_nation, C_LEN * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_d_datekey, h_d_datekey, D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year, h_d_year, D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey, h_s_suppkey, S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_region, h_s_region, S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_nation, h_s_nation, S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey, h_c_custkey, C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_region, h_c_region, C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_nation, h_c_nation, C_LEN * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate decompression buffers
    int *d_lo_orderdate, *d_lo_custkey, *d_lo_suppkey, *d_lo_revenue;
    CUDA_CHECK(cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_custkey, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(int)));

    // Time decompression
    cudaEvent_t start_decomp, stop_decomp;
    cudaEventCreate(&start_decomp);
    cudaEventCreate(&stop_decomp);

    cudaEventRecord(start_decomp);
    nvcomp_ssb::decompressColumn(comp_orderdate, d_lo_orderdate, stream);
    nvcomp_ssb::decompressColumn(comp_custkey, d_lo_custkey, stream);
    nvcomp_ssb::decompressColumn(comp_suppkey, d_lo_suppkey, stream);
    nvcomp_ssb::decompressColumn(comp_revenue, d_lo_revenue, stream);
    cudaEventRecord(stop_decomp);
    cudaEventSynchronize(stop_decomp);
    float time_decompress;
    cudaEventElapsedTime(&time_decompress, start_decomp, stop_decomp);
    cudaEventDestroy(start_decomp);
    cudaEventDestroy(stop_decomp);

    cout << "Decompression time: " << time_decompress << " ms" << endl;

    // Run query
    cout << "Running query..." << endl;
    runQuery<128, 8>(d_lo_orderdate, d_lo_custkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
                     d_d_datekey, d_d_year, D_LEN,
                     d_s_suppkey, d_s_region, d_s_nation, S_LEN,
                     d_c_custkey, d_c_region, d_c_nation, C_LEN,
                     time_h2d, time_decompress);

    // Cleanup
    comp_orderdate.free();
    comp_custkey.free();
    comp_suppkey.free();
    comp_revenue.free();

    CUDA_CHECK(cudaFree(d_lo_orderdate));
    CUDA_CHECK(cudaFree(d_lo_custkey));
    CUDA_CHECK(cudaFree(d_lo_suppkey));
    CUDA_CHECK(cudaFree(d_lo_revenue));
    CUDA_CHECK(cudaFree(d_d_datekey));
    CUDA_CHECK(cudaFree(d_d_year));
    CUDA_CHECK(cudaFree(d_s_suppkey));
    CUDA_CHECK(cudaFree(d_s_region));
    CUDA_CHECK(cudaFree(d_s_nation));
    CUDA_CHECK(cudaFree(d_c_custkey));
    CUDA_CHECK(cudaFree(d_c_region));
    CUDA_CHECK(cudaFree(d_c_nation));

    delete[] h_lo_orderdate;
    delete[] h_lo_custkey;
    delete[] h_lo_suppkey;
    delete[] h_lo_revenue;
    delete[] h_d_datekey;
    delete[] h_d_year;
    delete[] h_s_suppkey;
    delete[] h_s_region;
    delete[] h_s_nation;
    delete[] h_c_custkey;
    delete[] h_c_region;
    delete[] h_c_nation;

    cudaStreamDestroy(stream);

    return 0;
}
