// SSB Query Q1.1 - Using L32 Variance-Based Compression
// This version uses the variance-based partitioning algorithm from l32.cu

#include "l32_adapter.hpp"
#include "ssb_utils.h"
#include <iostream>
#include <chrono>
#include <vector>

using namespace std;

// Stage 1: Filter using predicate pushdown (simplified version)
template<typename T>
__global__ void stage1_filter_simple(
    const CompressedDataL3<T>* c_data,
    int total_len,
    T min_val,
    T max_val,
    int* d_candidate_indices,
    int* d_num_candidates)
{
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_idx >= total_len) return;

    T value = randomAccessDecompress(c_data, global_idx);

    if (value >= min_val && value <= max_val) {
        int pos = atomicAdd(d_num_candidates, 1);
        d_candidate_indices[pos] = global_idx;
    }
}

// Stage 2: Random access other columns
template<typename T>
__global__ void stage2_q11(
    const int* d_candidate_indices,
    int num_candidates,
    const CompressedDataL3<T>* c_lo_discount,
    const CompressedDataL3<T>* c_lo_quantity,
    const CompressedDataL3<T>* c_lo_extendedprice,
    unsigned long long* result)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int global_idx = d_candidate_indices[idx];

    T discount = randomAccessDecompress(c_lo_discount, global_idx);
    T quantity = randomAccessDecompress(c_lo_quantity, global_idx);
    T extendedprice = randomAccessDecompress(c_lo_extendedprice, global_idx);

    if (!(discount >= 1 && discount <= 3)) return;
    if (!(quantity < 25)) return;

    unsigned long long local_result = (unsigned long long)(extendedprice * discount);
    atomicAdd(result, local_result);
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query Q1.1 - L32 Variance-Based Compression" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    // Load columns
    cout << "Loading columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_discount = loadColumn<uint32_t>("lo_discount", LO_LEN);
    vector<uint32_t> lo_quantity = loadColumn<uint32_t>("lo_quantity", LO_LEN);
    vector<uint32_t> lo_extendedprice = loadColumn<uint32_t>("lo_extendedprice", LO_LEN);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded columns in " << load_time << " seconds" << endl;
    cout << endl;

    // Compress columns with L32 variance-based algorithm
    cout << "Compressing columns with L32 variance-based algorithm..." << endl;
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataL3<uint32_t>* c_lo_orderdate = compressData(lo_orderdate, 2048);
    CompressedDataL3<uint32_t>* c_lo_discount = compressData(lo_discount, 2048);
    CompressedDataL3<uint32_t>* c_lo_quantity = compressData(lo_quantity, 2048);
    CompressedDataL3<uint32_t>* c_lo_extendedprice = compressData(lo_extendedprice, 2048);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << "  Partitions created:" << endl;
    cout << "    lo_orderdate: " << c_lo_orderdate->num_partitions << endl;
    cout << "    lo_discount: " << c_lo_discount->num_partitions << endl;
    cout << "    lo_quantity: " << c_lo_quantity->num_partitions << endl;
    cout << "    lo_extendedprice: " << c_lo_extendedprice->num_partitions << endl;
    cout << endl;

    // Update device-side copy of compressed structures
    CUDA_CHECK(cudaMemcpy(c_lo_orderdate->d_self, c_lo_orderdate,
                          sizeof(CompressedDataL3<uint32_t>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_lo_discount->d_self, c_lo_discount,
                          sizeof(CompressedDataL3<uint32_t>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_lo_quantity->d_self, c_lo_quantity,
                          sizeof(CompressedDataL3<uint32_t>), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c_lo_extendedprice->d_self, c_lo_extendedprice,
                          sizeof(CompressedDataL3<uint32_t>), cudaMemcpyHostToDevice));

    // Allocate device memory
    unsigned long long* d_result;
    int* d_candidate_indices;
    int* d_num_candidates;

    CUDA_CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_indices, sizeof(int) * LO_LEN));
    CUDA_CHECK(cudaMalloc(&d_num_candidates, sizeof(int)));

    cout << "** Running Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_num_candidates, 0, sizeof(int)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Filter on orderdate
        int block_size = 256;
        int grid_size = (LO_LEN + block_size - 1) / block_size;

        stage1_filter_simple<uint32_t><<<grid_size, block_size>>>(
            c_lo_orderdate->d_self,
            LO_LEN,
            19930000,
            19940000,
            d_candidate_indices,
            d_num_candidates
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int num_candidates = 0;
        CUDA_CHECK(cudaMemcpy(&num_candidates, d_num_candidates, sizeof(int), cudaMemcpyDeviceToHost));

        // Stage 2: Random access
        if (num_candidates > 0) {
            grid_size = (num_candidates + block_size - 1) / block_size;

            stage2_q11<uint32_t><<<grid_size, block_size>>>(
                d_candidate_indices,
                num_candidates,
                c_lo_discount->d_self,
                c_lo_quantity->d_self,
                c_lo_extendedprice->d_self,
                d_result
            );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        unsigned long long h_result;
        CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        cout << "{" << R"("query":"Q1.1","method":"l32_variance","time_ms":)" << query_time
             << R"(,"num_candidates":)" << num_candidates
             << R"(,"result":)" << h_result << "}" << endl;

        if (t == 0) {
            cout << "  First trial details:" << endl;
            cout << "    Candidates: " << num_candidates << endl;
            cout << "    Result: " << h_result << endl;
        }
    }

    cout << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_candidate_indices));
    CUDA_CHECK(cudaFree(d_num_candidates));

    freeCompressedData(c_lo_orderdate);
    freeCompressedData(c_lo_discount);
    freeCompressedData(c_lo_quantity);
    freeCompressedData(c_lo_extendedprice);

    cout << "========================================================================" << endl;

    return 0;
}
