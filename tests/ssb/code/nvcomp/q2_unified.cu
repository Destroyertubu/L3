/**
 * SSB Q2.1 Query with nvcomp Cascade Compression
 *
 * SQL:
 * SELECT SUM(lo_revenue), d_year, p_brand1
 * FROM lineorder, date, part, supplier
 * WHERE lo_orderdate = d_datekey
 *   AND lo_partkey = p_partkey
 *   AND lo_suppkey = s_suppkey
 *   AND p_category = 'MFGR#12'
 *   AND s_region = 'AMERICA'
 * GROUP BY d_year, p_brand1
 */

#include "crystal/crystal.cuh"
#include "nvcomp_ssb.cuh"
#include "ssb_cascade_config.h"
#include "ssb_utils.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

// Build hash table for supplier dimension (filter by s_region = 'AMERICA')
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s(int* filter_col, int* dim_key, int num_tuples,
                                  int* hash_table, int num_slots) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (num_tuples + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = num_tuples - tile_offset;
    }

    // Filter: s_region = 'AMERICA' (encoded as 1)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 1, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
    BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, selection_flags, hash_table, num_slots, num_tile_items);
}

// Build hash table for part dimension (filter by p_category = 'MFGR#12')
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p(int* filter_col, int* dim_key, int* dim_val,
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

    // Filter: p_category = 'MFGR#12' (encoded as 12)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(filter_col + tile_offset, items, num_tile_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, 12, selection_flags, num_tile_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, items2, selection_flags, hash_table, num_slots, num_tile_items);
}

// Build hash table for date dimension (no filter, store year)
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

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_key + tile_offset, items, num_tile_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(dim_val + tile_offset, items2, num_tile_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, items2, selection_flags, hash_table, num_slots, val_min, num_tile_items);
}

// Probe kernel: join and aggregate
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q21_Kernel(const int* d_lo_orderdate,
                           const int* d_lo_partkey,
                           const int* d_lo_suppkey,
                           const int* d_lo_revenue,
                           int        lo_len,
                           int*       ht_s,
                           int        s_len,
                           int*       ht_p,
                           int        p_len,
                           int*       ht_d,
                           int        d_len,
                           int        d_val_min,
                           int*       res) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];
    int brand[ITEMS_PER_THREAD];
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

    // Probe supplier hash table
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_suppkey) + tile_offset, items, num_tile_items);
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, selection_flags, ht_s, s_len, num_tile_items);
    int local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    int any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // Probe part hash table (get brand)
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_partkey) + tile_offset, items, num_tile_items);
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, brand, selection_flags, ht_p, p_len, num_tile_items);
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

    // Aggregate by (year, brand)
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) {
            if (selection_flags[ITEM]) {
                // Hash: (brand * 7 + (year - 1992)) % table_size
                int hash = (brand[ITEM] * 7 + (year[ITEM] - 1992)) % ((1998 - 1992 + 1) * (5 * 5 * 40));
                res[hash * 4]     = year[ITEM];
                res[hash * 4 + 1] = brand[ITEM];
                atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]),
                          static_cast<unsigned long long>(revenue[ITEM]));
            }
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int* d_lo_orderdate, int* d_lo_partkey, int* d_lo_suppkey,
              int* d_lo_revenue, int lo_len,
              int* d_p_partkey, int* d_p_brand1, int* d_p_category, int p_len,
              int* d_d_datekey, int* d_d_year, int d_len,
              int* d_s_suppkey, int* d_s_region, int s_len,
              float time_h2d, float time_decompress) {
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Allocate hash tables
    int  d_val_len = 19981230 - 19920101 + 1;
    int *ht_d, *ht_p, *ht_s;
    CUDA_CHECK(cudaMalloc(&ht_d, 2 * d_val_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_p, 2 * p_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ht_s, 2 * s_len * sizeof(int)));

    // Time hash table build
    cudaEvent_t start_ht, stop_ht;
    cudaEventCreate(&start_ht);
    cudaEventCreate(&stop_ht);

    cudaEventRecord(start_ht);
    CUDA_CHECK(cudaMemset(ht_d, 0, 2 * d_val_len * sizeof(int)));
    CUDA_CHECK(cudaMemset(ht_p, 0, 2 * p_len * sizeof(int)));
    CUDA_CHECK(cudaMemset(ht_s, 0, 2 * s_len * sizeof(int)));

    build_hashtable_s<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(s_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
            d_s_region, d_s_suppkey, s_len, ht_s, s_len);

    build_hashtable_p<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<(p_len + tile_items - 1) / tile_items, BLOCK_THREADS>>>(
            d_p_category, d_p_partkey, d_p_brand1, p_len, ht_p, p_len);

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
    int  res_size       = ((1998 - 1992 + 1) * (5 * 5 * 40));
    int  res_array_size = res_size * 4;
    int* d_res;
    CUDA_CHECK(cudaMalloc(&d_res, res_array_size * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

    int num_blocks = (lo_len + tile_items - 1) / tile_items;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    Q21_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, lo_len,
        ht_s, s_len, ht_p, p_len, ht_d, d_val_len, d_val_min, d_res);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

    const int NUM_RUNS = 3;
    for (int run = 0; run < NUM_RUNS; run++) {
        CUDA_CHECK(cudaMemset(d_res, 0, res_array_size * sizeof(int)));

        cudaEventRecord(start);
        Q21_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
            d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, lo_len,
            ht_s, s_len, ht_p, p_len, ht_d, d_val_len, d_val_min, d_res);
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
            if (h_res[4 * i] != 0) {
                result_rows++;
                total_revenue += reinterpret_cast<unsigned long long*>(&h_res[4 * i + 2])[0];
            }
        }

        float time_total = time_h2d + time_decompress + time_ht_build + time_kernel + time_d2h;

        cout << "-- Q21 result_rows: " << result_rows << ", total_revenue: " << total_revenue << endl;
        cout << "{\"query\":21,\"run\":" << run
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
    CUDA_CHECK(cudaFree(ht_p));
    CUDA_CHECK(cudaFree(ht_s));
}

int main() {
    cout << "=== SSB Q2.1 with nvcomp Cascade ===" << endl;

    // Load lineorder columns
    cout << "Loading lineorder data..." << endl;
    int* h_lo_orderdate = loadColumn<int>("lo_orderdate", LO_LEN);
    int* h_lo_partkey   = loadColumn<int>("lo_partkey", LO_LEN);
    int* h_lo_suppkey   = loadColumn<int>("lo_suppkey", LO_LEN);
    int* h_lo_revenue   = loadColumn<int>("lo_revenue", LO_LEN);

    // Load dimension tables
    cout << "Loading dimension tables..." << endl;
    int* h_p_partkey  = loadColumn<int>("p_partkey", P_LEN);
    int* h_p_brand1   = loadColumn<int>("p_brand1", P_LEN);
    int* h_p_category = loadColumn<int>("p_category", P_LEN);

    int* h_d_datekey = loadColumn<int>("d_datekey", D_LEN);
    int* h_d_year    = loadColumn<int>("d_year", D_LEN);

    int* h_s_suppkey = loadColumn<int>("s_suppkey", S_LEN);
    int* h_s_region  = loadColumn<int>("s_region", S_LEN);

    cudaFree(0);

    // Compress lineorder columns
    cout << "Compressing lineorder columns..." << endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CompressedColumn comp_orderdate = nvcomp_ssb::compressColumn(
        h_lo_orderdate, LO_LEN, ssb_cascade_config::orderdate_config, stream);
    CompressedColumn comp_partkey = nvcomp_ssb::compressColumn(
        h_lo_partkey, LO_LEN, ssb_cascade_config::partkey_config, stream);
    CompressedColumn comp_suppkey = nvcomp_ssb::compressColumn(
        h_lo_suppkey, LO_LEN, ssb_cascade_config::suppkey_config, stream);
    CompressedColumn comp_revenue = nvcomp_ssb::compressColumn(
        h_lo_revenue, LO_LEN, ssb_cascade_config::revenue_config, stream);

    cout << "Compression statistics:" << endl;
    nvcomp_ssb::printCompressionStats("orderdate", comp_orderdate);
    nvcomp_ssb::printCompressionStats("partkey", comp_partkey);
    nvcomp_ssb::printCompressionStats("suppkey", comp_suppkey);
    nvcomp_ssb::printCompressionStats("revenue", comp_revenue);

    float time_h2d = 0;  // Compressed data already on GPU from compression

    // Allocate and transfer dimension tables to GPU
    int *d_p_partkey, *d_p_brand1, *d_p_category;
    int *d_d_datekey, *d_d_year;
    int *d_s_suppkey, *d_s_region;

    CUDA_CHECK(cudaMalloc(&d_p_partkey, P_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p_brand1, P_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_p_category, P_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_d_datekey, D_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_d_year, D_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_suppkey, S_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_s_region, S_LEN * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_p_partkey, h_p_partkey, P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_brand1, h_p_brand1, P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_category, h_p_category, P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_datekey, h_d_datekey, D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year, h_d_year, D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey, h_s_suppkey, S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_region, h_s_region, S_LEN * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate decompression buffers
    int *d_lo_orderdate, *d_lo_partkey, *d_lo_suppkey, *d_lo_revenue;
    CUDA_CHECK(cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_partkey, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_suppkey, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_revenue, LO_LEN * sizeof(int)));

    // Time decompression
    cudaEvent_t start_decomp, stop_decomp;
    cudaEventCreate(&start_decomp);
    cudaEventCreate(&stop_decomp);

    cudaEventRecord(start_decomp);
    nvcomp_ssb::decompressColumn(comp_orderdate, d_lo_orderdate, stream);
    nvcomp_ssb::decompressColumn(comp_partkey, d_lo_partkey, stream);
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
    runQuery<128, 8>(d_lo_orderdate, d_lo_partkey, d_lo_suppkey, d_lo_revenue, LO_LEN,
                     d_p_partkey, d_p_brand1, d_p_category, P_LEN,
                     d_d_datekey, d_d_year, D_LEN,
                     d_s_suppkey, d_s_region, S_LEN,
                     time_h2d, time_decompress);

    // Cleanup
    comp_orderdate.free();
    comp_partkey.free();
    comp_suppkey.free();
    comp_revenue.free();

    CUDA_CHECK(cudaFree(d_lo_orderdate));
    CUDA_CHECK(cudaFree(d_lo_partkey));
    CUDA_CHECK(cudaFree(d_lo_suppkey));
    CUDA_CHECK(cudaFree(d_lo_revenue));
    CUDA_CHECK(cudaFree(d_p_partkey));
    CUDA_CHECK(cudaFree(d_p_brand1));
    CUDA_CHECK(cudaFree(d_p_category));
    CUDA_CHECK(cudaFree(d_d_datekey));
    CUDA_CHECK(cudaFree(d_d_year));
    CUDA_CHECK(cudaFree(d_s_suppkey));
    CUDA_CHECK(cudaFree(d_s_region));

    delete[] h_lo_orderdate;
    delete[] h_lo_partkey;
    delete[] h_lo_suppkey;
    delete[] h_lo_revenue;
    delete[] h_p_partkey;
    delete[] h_p_brand1;
    delete[] h_p_category;
    delete[] h_d_datekey;
    delete[] h_d_year;
    delete[] h_s_suppkey;
    delete[] h_s_region;

    cudaStreamDestroy(stream);

    return 0;
}
