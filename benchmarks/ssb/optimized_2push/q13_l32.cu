// SSB Query Q1.3 - Two-Stage + Predicate Pushdown
// Automatically generated

#include "l32_adapter.hpp"
#include "l3_ra_utils.cuh"
#include "l3_predicate_pushdown.cuh"
#include "ssb_utils.h"
#include <iostream>
#include <chrono>

using namespace std;

// Forward declaration
template<typename T>
void launchComputePartitionBounds(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions,
    cudaStream_t stream = 0);

#define SHMEM_BUFFER_SIZE 512

// Stage 2: Random access other columns
template<typename T>
__global__ void stage2_q13(
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

    if (!(discount >= 5 && discount <= 7)) return;
    if (!(quantity >= 26 && quantity <= 35)) return;

    unsigned long long local_result = (unsigned long long)(extendedprice * discount);
    atomicAdd(result, local_result);
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query Q1.3 - Two-Stage + Predicate Pushdown" << endl;
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

    // Compress columns
    cout << "Compressing columns with L3..." << endl;
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataL3<uint32_t>* c_lo_orderdate = compressData(lo_orderdate, 1024);
    CompressedDataL3<uint32_t>* c_lo_discount = compressData(lo_discount, 1024);
    CompressedDataL3<uint32_t>* c_lo_quantity = compressData(lo_quantity, 1024);
    CompressedDataL3<uint32_t>* c_lo_extendedprice = compressData(lo_extendedprice, 1024);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << endl;

    // Compute partition bounds for predicate pushdown
    cout << "Computing partition bounds for predicate pushdown..." << endl;
    uint32_t* d_lo_orderdate_min;
    uint32_t* d_lo_orderdate_max;
    uint32_t* d_lo_orderdate_values;

    CUDA_CHECK(cudaMalloc(&d_lo_orderdate_min, c_lo_orderdate->num_partitions * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lo_orderdate_max, c_lo_orderdate->num_partitions * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_lo_orderdate_values, LO_LEN * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_lo_orderdate_values, lo_orderdate.data(),
                          LO_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice));

    launchComputePartitionBounds<uint32_t>(
        d_lo_orderdate_values,
        c_lo_orderdate->d_start_indices,
        c_lo_orderdate->d_end_indices,
        d_lo_orderdate_min,
        d_lo_orderdate_max,
        c_lo_orderdate->num_partitions
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Update compressed data structure with bounds
    c_lo_orderdate->d_partition_min_values = d_lo_orderdate_min;
    c_lo_orderdate->d_partition_max_values = d_lo_orderdate_max;

    // Upload updated structure to device
    CUDA_CHECK(cudaMemcpy(c_lo_orderdate->d_self, c_lo_orderdate,
                          sizeof(CompressedDataL3<uint32_t>), cudaMemcpyHostToDevice));

    // Can free d_lo_orderdate_values now
    CUDA_CHECK(cudaFree(d_lo_orderdate_values));

    cout << "✓ Partition bounds computed" << endl;
    cout << endl;

    // Allocate device memory
    unsigned long long* d_result;
    int* d_candidate_indices;
    int* d_num_candidates;
    unsigned long long* d_partitions_pruned;

    CUDA_CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_candidate_indices, sizeof(int) * LO_LEN));
    CUDA_CHECK(cudaMalloc(&d_num_candidates, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partitions_pruned, sizeof(unsigned long long)));

    cout << "** Running Two-Stage + Predicate Pushdown Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_result, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_num_candidates, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_partitions_pruned, 0, sizeof(unsigned long long)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Filter with predicate pushdown
        stage1_filter_with_predicate_pushdown<uint32_t><<<c_lo_orderdate->num_partitions, 256>>>(
            c_lo_orderdate->d_self,
            LO_LEN,
            19940100,
            19940200,
            d_candidate_indices,
            d_num_candidates,
            d_partitions_pruned
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        int num_candidates = 0;
        CUDA_CHECK(cudaMemcpy(&num_candidates, d_num_candidates, sizeof(int), cudaMemcpyDeviceToHost));

        unsigned long long partitions_pruned = 0;
        CUDA_CHECK(cudaMemcpy(&partitions_pruned, d_partitions_pruned, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // Stage 2: Random access
        if (num_candidates > 0) {
            int block_size = 256;
            int grid_size = (num_candidates + block_size - 1) / block_size;

            stage2_q13<uint32_t><<<grid_size, block_size>>>(
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

        cout << "{" << R"("query":"Q1.3","optimization":"two_stage_predicate_pushdown","time_ms":)" << query_time
             << R"(,"num_candidates":)" << num_candidates
             << R"(,"partitions_pruned":)" << partitions_pruned
             << R"(,"result":)" << h_result << "}" << endl;

        if (t == 0) {
            cout << "  Candidates: " << num_candidates << endl;
            cout << "  Partitions pruned: " << partitions_pruned << " / " << c_lo_orderdate->num_partitions << endl;
            cout << "  Result: " << h_result << endl;
        }
    }

    cout << endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_result));
    CUDA_CHECK(cudaFree(d_candidate_indices));
    CUDA_CHECK(cudaFree(d_num_candidates));
    CUDA_CHECK(cudaFree(d_partitions_pruned));

    freeCompressedData(c_lo_orderdate);
    freeCompressedData(c_lo_discount);
    freeCompressedData(c_lo_quantity);
    freeCompressedData(c_lo_extendedprice);

    cout << "========================================================================" << endl;

    return 0;
}
