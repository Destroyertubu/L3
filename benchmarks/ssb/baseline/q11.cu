// SSB Query 1.1 with Optimized L3 Compression
// SELECT SUM(lo_extendedprice * lo_discount) AS revenue
// FROM lineorder
// WHERE lo_orderdate >= 19930000 AND lo_orderdate < 19940000
//       AND lo_discount >= 1 AND lo_discount <= 3
//       AND lo_quantity < 25

#include "l3_codec.hpp"

#include "ssb_utils.h"
#include <iostream>
#include <chrono>

using namespace std;

// GPU kernel for SSB Q11
template<typename T>
__global__ void ssb_q11_kernel(
    const T* __restrict__ lo_orderdate, const T* __restrict__ lo_discount, const T* __restrict__ lo_quantity, const T* __restrict__ lo_extendedprice,
    int num_entries,
    unsigned long long* revenue)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    T orderdate = lo_orderdate[idx];
    T discount = lo_discount[idx];
    T quantity = lo_quantity[idx];
    T extendedprice = lo_extendedprice[idx];

    // Apply filters
    if (orderdate >= 19930000 && orderdate < 19940000 &&
        discount >= 1 && discount <= 3 &&
        quantity < 25) {
        unsigned long long local_revenue = (unsigned long long)extendedprice * discount;
        atomicAdd(revenue, local_revenue);
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 1.1 with Optimized L3 Compression" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT SUM(lo_extendedprice * lo_discount) AS revenue" << endl;
    cout << "       FROM lineorder" << endl;
    cout << "       WHERE lo_orderdate >= 19930000 AND lo_orderdate < 19940000" << endl;
    cout << "             AND lo_discount >= 1 AND lo_discount <= 3" << endl;
    cout << "             AND lo_quantity < 25" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    // Load columns
    cout << "Loading LINEORDER columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_discount = loadColumn<uint32_t>("lo_discount", LO_LEN);
    vector<uint32_t> lo_quantity = loadColumn<uint32_t>("lo_quantity", LO_LEN);
    vector<uint32_t> lo_extendedprice = loadColumn<uint32_t>("lo_extendedprice", LO_LEN);

    if (lo_orderdate.empty() || lo_discount.empty() || lo_quantity.empty() || lo_extendedprice.empty()) {
        cerr << "Error: Failed to load one or more columns" << endl;
        return 1;
    }

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded 4 columns (" << LO_LEN << " rows each) in " << load_time << " seconds" << endl;
    cout << endl;

    // Compress columns with optimized L3
    cout << "Compressing columns with L3..." << endl;
    auto compress_start = chrono::high_resolution_clock::now();

    CompressionStats comp_stats;
    CompressedDataL3<uint32_t>* c_orderdate = compressData(lo_orderdate, 4096, &comp_stats);
    CompressedDataL3<uint32_t>* c_discount = compressData(lo_discount, 4096);
    CompressedDataL3<uint32_t>* c_quantity = compressData(lo_quantity, 4096);
    CompressedDataL3<uint32_t>* c_extendedprice = compressData(lo_extendedprice, 4096);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << "  Partitions: " << c_orderdate->num_partitions << endl;
    cout << "  Compression ratio: " << comp_stats.compression_ratio << "x" << endl;
    cout << endl;

    // Allocate device memory for decompressed data
    uint32_t *d_orderdate, *d_discount, *d_quantity, *d_extendedprice;
    CUDA_CHECK(cudaMalloc(&d_orderdate, LO_LEN * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_discount, LO_LEN * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_quantity, LO_LEN * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_extendedprice, LO_LEN * sizeof(uint32_t)));

    unsigned long long* d_revenue;
    CUDA_CHECK(cudaMalloc(&d_revenue, sizeof(unsigned long long)));

    cout << "** Running Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_revenue, 0, sizeof(unsigned long long)));

        auto query_start = chrono::high_resolution_clock::now();

        // Decompress columns using optimized decoder
        DecompressionStats decomp_stats_tmp;
        DecompressionStats decomp_stats;
        decompressData(c_orderdate, d_orderdate, LO_LEN, &decomp_stats);
        decompressData(c_discount, d_discount, LO_LEN);
        decompressData(c_quantity, d_quantity, LO_LEN);
        decompressData(c_extendedprice, d_extendedprice, LO_LEN);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Run query kernel
        int block_size = 256;
        int grid_size = (LO_LEN + block_size - 1) / block_size;
        ssb_q11_kernel<<<grid_size, block_size>>>(
            d_orderdate, d_discount, d_quantity, d_extendedprice,
            LO_LEN, d_revenue
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        unsigned long long h_revenue;
        CUDA_CHECK(cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        cout << "{\"query\":11,\"time_ms\":" << query_time
             << ",\"revenue\":" << h_revenue << "}" << endl;

        if (t == 0) {
            cout << "Revenue: " << h_revenue << endl;
        }
    }

    cout << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_orderdate));
    CUDA_CHECK(cudaFree(d_discount));
    CUDA_CHECK(cudaFree(d_quantity));
    CUDA_CHECK(cudaFree(d_extendedprice));
    CUDA_CHECK(cudaFree(d_revenue));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_discount);
    freeCompressedData(c_quantity);
    freeCompressedData(c_extendedprice);

    cout << "========================================================================" << endl;

    return 0;
}
