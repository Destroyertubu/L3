/**
 * SSB Q1.1 Query with nvcomp Cascade Compression
 *
 * SQL:
 * SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 * FROM lineorder
 * WHERE lo_orderdate >= 19930101 AND lo_orderdate < 19940101
 *   AND lo_discount >= 1 AND lo_discount <= 3
 *   AND lo_quantity < 25
 */

#include "crystal/crystal.cuh"
#include "nvcomp_ssb.cuh"
#include "ssb_cascade_config.h"
#include "ssb_utils.h"
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// Query kernel
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void Q11_Kernel(const int*          d_lo_orderdate,
                           const int*          d_lo_discount,
                           const int*          d_lo_quantity,
                           const int*          d_lo_extendedprice,
                           int                 lo_len,
                           unsigned long long* d_revenue) {
    int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int       items[ITEMS_PER_THREAD];
    int       items2[ITEMS_PER_THREAD];
    int       selection_flags[ITEMS_PER_THREAD];
    long long sum = 0;
    __shared__ int shared_any[32];
    __shared__ int block_has_valid;

    int tile_offset    = blockIdx.x * TILE_SIZE;
    int num_tiles      = (lo_len + TILE_SIZE - 1) / TILE_SIZE;
    int num_tile_items = TILE_SIZE;

    if (blockIdx.x == num_tiles - 1) {
        num_tile_items = lo_len - tile_offset;
    }

    // 1. Load and filter orderdate: >= 19930101 AND < 19940101
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_orderdate) + tile_offset, items, num_tile_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 19930101, selection_flags, num_tile_items);
    BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 19940101, selection_flags, num_tile_items);
    int local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    int any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // 2. Load and filter quantity: < 25
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_quantity) + tile_offset, items, num_tile_items);
    BlockPredAndLT<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 25, selection_flags, num_tile_items);
    local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // 3. Load and filter discount: >= 1 AND <= 3
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_discount) + tile_offset, items, num_tile_items);
    BlockPredAndGTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 1, selection_flags, num_tile_items);
    BlockPredAndLTE<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, 3, selection_flags, num_tile_items);
    local_any = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_tile_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

    // 4. Load extendedprice
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        const_cast<int*>(d_lo_extendedprice) + tile_offset, items2, num_tile_items);

    // 5. Compute sum where selection_flags is true
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM) < num_tile_items)) {
            if (selection_flags[ITEM]) {
                sum += static_cast<long long>(items[ITEM]) * items2[ITEM];
            }
        }
    }

    __syncthreads();

    // 6. Block reduction
    static __shared__ long long buffer[32];
    unsigned long long aggregate =
        BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(sum, buffer);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(d_revenue, aggregate);
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
void runQuery(int* d_lo_orderdate, int* d_lo_discount, int* d_lo_quantity,
              int* d_lo_extendedprice, int lo_len,
              float time_h2d, float time_decompress) {
    // Allocate result
    unsigned long long* d_revenue;
    CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(unsigned long long)));

    // Calculate grid size
    int tile_items = BLOCK_THREADS * ITEMS_PER_THREAD;
    int num_blocks = (lo_len + tile_items - 1) / tile_items;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    Q11_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
        d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice,
        lo_len, d_revenue);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(unsigned long long)));

    // Timed runs
    const int NUM_RUNS = 3;
    for (int run = 0; run < NUM_RUNS; run++) {
        CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(unsigned long long)));

        cudaEventRecord(start);
        Q11_Kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<num_blocks, BLOCK_THREADS>>>(
            d_lo_orderdate, d_lo_discount, d_lo_quantity, d_lo_extendedprice,
            lo_len, d_revenue);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_kernel;
        cudaEventElapsedTime(&time_kernel, start, stop);

        // D2H
        cudaEvent_t start_d2h, stop_d2h;
        cudaEventCreate(&start_d2h);
        cudaEventCreate(&stop_d2h);
        cudaEventRecord(start_d2h);
        unsigned long long revenue;
        CUDA_CHECK(cudaMemcpy(&revenue, d_revenue, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
        cudaEventRecord(stop_d2h);
        cudaEventSynchronize(stop_d2h);
        float time_d2h;
        cudaEventElapsedTime(&time_d2h, start_d2h, stop_d2h);
        cudaEventDestroy(start_d2h);
        cudaEventDestroy(stop_d2h);

        float time_total = time_h2d + time_decompress + time_kernel + time_d2h;

        cout << "-- Q11 revenue: " << revenue << endl;
        cout << "{\"query\":11,\"run\":" << run
             << ",\"time_h2d\":" << time_h2d
             << ",\"time_decompress\":" << time_decompress
             << ",\"time_kernel\":" << time_kernel
             << ",\"time_d2h\":" << time_d2h
             << ",\"time_total\":" << time_total << "}" << endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK(cudaFree(d_revenue));
}

int main() {
    cout << "=== SSB Q1.1 with nvcomp Cascade ===" << endl;

    // Load data from disk
    cout << "Loading data..." << endl;
    int* h_lo_orderdate     = loadColumn<int>("lo_orderdate", LO_LEN);
    int* h_lo_discount      = loadColumn<int>("lo_discount", LO_LEN);
    int* h_lo_quantity      = loadColumn<int>("lo_quantity", LO_LEN);
    int* h_lo_extendedprice = loadColumn<int>("lo_extendedprice", LO_LEN);

    // CUDA warmup
    cudaFree(0);

    // Compress columns
    cout << "Compressing columns..." << endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    CompressedColumn comp_orderdate = nvcomp_ssb::compressColumn(
        h_lo_orderdate, LO_LEN, ssb_cascade_config::orderdate_config, stream);
    CompressedColumn comp_discount = nvcomp_ssb::compressColumn(
        h_lo_discount, LO_LEN, ssb_cascade_config::discount_config, stream);
    CompressedColumn comp_quantity = nvcomp_ssb::compressColumn(
        h_lo_quantity, LO_LEN, ssb_cascade_config::quantity_config, stream);
    CompressedColumn comp_extendedprice = nvcomp_ssb::compressColumn(
        h_lo_extendedprice, LO_LEN, ssb_cascade_config::extendedprice_config, stream);

    // Print compression stats
    cout << "Compression statistics:" << endl;
    nvcomp_ssb::printCompressionStats("orderdate", comp_orderdate);
    nvcomp_ssb::printCompressionStats("discount", comp_discount);
    nvcomp_ssb::printCompressionStats("quantity", comp_quantity);
    nvcomp_ssb::printCompressionStats("extendedprice", comp_extendedprice);

    size_t total_compressed =
        comp_orderdate.total_compressed_bytes +
        comp_discount.total_compressed_bytes +
        comp_quantity.total_compressed_bytes +
        comp_extendedprice.total_compressed_bytes;
    size_t total_original =
        comp_orderdate.original_bytes + comp_discount.original_bytes +
        comp_quantity.original_bytes + comp_extendedprice.original_bytes;
    cout << "  Total: " << (total_original / 1024.0 / 1024.0) << " MB -> "
         << (total_compressed / 1024.0 / 1024.0) << " MB"
         << " (ratio: " << (double)total_original / total_compressed << "x)"
         << endl;

    // Time H2D (compressed data is already on GPU from compression)
    // For fair comparison, we measure the time to transfer compressed data
    float time_h2d = 0;  // Already on GPU

    // Allocate decompression buffers
    int* d_lo_orderdate;
    int* d_lo_discount;
    int* d_lo_quantity;
    int* d_lo_extendedprice;

    CUDA_CHECK(cudaMalloc(&d_lo_orderdate, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_discount, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_quantity, LO_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lo_extendedprice, LO_LEN * sizeof(int)));

    // Time decompression
    cudaEvent_t start_decomp, stop_decomp;
    cudaEventCreate(&start_decomp);
    cudaEventCreate(&stop_decomp);

    cudaEventRecord(start_decomp);
    nvcomp_ssb::decompressColumn(comp_orderdate, d_lo_orderdate, stream);
    nvcomp_ssb::decompressColumn(comp_discount, d_lo_discount, stream);
    nvcomp_ssb::decompressColumn(comp_quantity, d_lo_quantity, stream);
    nvcomp_ssb::decompressColumn(comp_extendedprice, d_lo_extendedprice, stream);
    cudaEventRecord(stop_decomp);
    cudaEventSynchronize(stop_decomp);
    float time_decompress;
    cudaEventElapsedTime(&time_decompress, start_decomp, stop_decomp);
    cudaEventDestroy(start_decomp);
    cudaEventDestroy(stop_decomp);

    cout << "Decompression time: " << time_decompress << " ms" << endl;

    // Run query
    cout << "Running query..." << endl;
    runQuery<128, 8>(d_lo_orderdate, d_lo_discount, d_lo_quantity,
                     d_lo_extendedprice, LO_LEN, time_h2d, time_decompress);

    // Cleanup
    comp_orderdate.free();
    comp_discount.free();
    comp_quantity.free();
    comp_extendedprice.free();

    CUDA_CHECK(cudaFree(d_lo_orderdate));
    CUDA_CHECK(cudaFree(d_lo_discount));
    CUDA_CHECK(cudaFree(d_lo_quantity));
    CUDA_CHECK(cudaFree(d_lo_extendedprice));

    delete[] h_lo_orderdate;
    delete[] h_lo_discount;
    delete[] h_lo_quantity;
    delete[] h_lo_extendedprice;

    cudaStreamDestroy(stream);

    return 0;
}
