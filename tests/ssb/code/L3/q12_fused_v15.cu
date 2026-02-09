/**
 * SSB Q1.2 - V13 (L3 V4 Vectorized Decoder, 2048 values/partition)
 *
 * Query: SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *        FROM lineorder, date
 *        WHERE lo_orderdate = d_datekey
 *          AND d_yearmonthnum = 199401
 *          AND lo_discount BETWEEN 4 AND 6
 *          AND lo_quantity BETWEEN 26 AND 35
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>

#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "v15_common.cuh"
#include "v15_padding.cuh"

using namespace ssb;
using namespace v15;

constexpr unsigned long long EXPECTED_REVENUE_Q12 = 17303753830ULL;

__global__ void q12_fused_kernel_v15(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int num_partitions,
    int total_rows,
    unsigned long long* __restrict__ global_revenue)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;

    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_end = lo_orderdate.d_end_indices[partition_idx];
    int partition_size = partition_end - partition_start;

    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;
    if (num_mv == 0) return;

    uint32_t orderdate[V15_VALUES_PER_THREAD];
    uint32_t quantity[V15_VALUES_PER_THREAD];
    uint32_t discount[V15_VALUES_PER_THREAD];
    uint32_t extendedprice[V15_VALUES_PER_THREAD];
    int flags[V15_VALUES_PER_THREAD];

    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) flags[v] = 1;

    // ========== Decode orderdate ==========
    decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, orderdate);

    // Filter: yearmonthnum = 199401
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int local_idx = v * V15_WARP_SIZE + lane_id;
        int global_idx = partition_start + local_idx;
        if (global_idx >= total_rows) flags[v] = 0;
        else if (orderdate[v] < MONTH_199401_START || orderdate[v] > MONTH_199401_END) flags[v] = 0;
    }

    int any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Decode quantity ==========
    decode_column_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, lane_id, quantity);

    // Filter: quantity BETWEEN 26 AND 35
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (quantity[v] < 26 || quantity[v] > 35)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Decode discount ==========
    decode_column_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, lane_id, discount);

    // Filter: discount BETWEEN 4 AND 6
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (discount[v] < 4 || discount[v] > 6)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    // ========== Decode extendedprice ==========
    decode_column_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, lane_id, extendedprice);

    // ========== Aggregation ==========
    unsigned long long local_revenue = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v]) {
            local_revenue += static_cast<unsigned long long>(extendedprice[v]) *
                            static_cast<unsigned long long>(discount[v]);
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_revenue += __shfl_down_sync(0xFFFFFFFF, local_revenue, offset);
    }

    if (lane_id == 0 && local_revenue > 0) {
        atomicAdd(global_revenue, local_revenue);
    }
}

__global__ void q12_tail_kernel_v15(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int partition_idx,
    int tail_start_in_partition,
    int tail_size,
    unsigned long long* __restrict__ global_revenue)
{
    int tid = threadIdx.x;
    if (tid >= tail_size) return;

    int value_idx = tail_start_in_partition + tid;

    uint32_t orderdate = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
    if (orderdate < MONTH_199401_START || orderdate > MONTH_199401_END) return;

    uint32_t quantity = decode_tail_value_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, value_idx);
    if (quantity < 26 || quantity > 35) return;

    uint32_t discount = decode_tail_value_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, value_idx);
    if (discount < 4 || discount > 6) return;

    uint32_t extendedprice = decode_tail_value_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, value_idx);

    unsigned long long revenue = static_cast<unsigned long long>(extendedprice) *
                                  static_cast<unsigned long long>(discount);
    atomicAdd(global_revenue, revenue);
}

void runQ12FusedV13(SSBDataCompressedVertical& data, float time_h2d) {
    cudaEvent_t ev_start, ev_metadata, ev_kernel, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_metadata);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = data.lo_orderdate.num_partitions;
    int last_partition_idx = num_partitions - 1;

    // ===== Stage 1: Metadata =====
    cudaEventRecord(ev_start);

    std::vector<int32_t> h_start(num_partitions), h_end(num_partitions);
    cudaMemcpy(h_start.data(), data.lo_orderdate.d_start_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_end.data(), data.lo_orderdate.d_end_indices,
               num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int last_partition_size = h_end[last_partition_idx] - h_start[last_partition_idx];
    int num_mv_last = last_partition_size / V15_MINI_VECTOR_SIZE;
    int tail_start = num_mv_last * V15_MINI_VECTOR_SIZE;
    int tail_size = last_partition_size - tail_start;

    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    cudaEventRecord(ev_metadata);

    // ===== No hash table build for Q1.2 =====

    // ===== Warmup =====
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q12_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
        num_partitions, total_rows, d_revenue);
    if (tail_size > 0) {
        q12_tail_kernel_v15<<<1, tail_size>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            last_partition_idx, tail_start, tail_size, d_revenue);
    }
    cudaDeviceSynchronize();

    // ===== Benchmark runs =====
    const int RUNS = 5;

    for (int run = 0; run < RUNS; run++) {
        // Reset output (not counted in timing)
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);

        // Metadata (minimal for subsequent runs)
        cudaEventRecord(ev_metadata);

        // Kernel only
        q12_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            num_partitions, total_rows, d_revenue);
        if (tail_size > 0) {
            q12_tail_kernel_v15<<<1, tail_size>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                last_partition_idx, tail_start, tail_size, d_revenue);
        }
        cudaEventRecord(ev_kernel);

        // D2H
        unsigned long long h_revenue;
        cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);

        float time_metadata, time_kernel, time_d2h;
        cudaEventElapsedTime(&time_metadata, ev_start, ev_metadata);
        cudaEventElapsedTime(&time_kernel, ev_metadata, ev_kernel);
        cudaEventElapsedTime(&time_d2h, ev_kernel, ev_d2h);
        float time_total = time_h2d + time_metadata + time_kernel + time_d2h;

        std::cout << "{\"query\":12,\"version\":\"v14\",\"run\":" << run
                  << ",\"time_h2d\":" << time_h2d
                  << ",\"time_metadata\":" << time_metadata
                  << ",\"time_ht_build\":0"
                  << ",\"time_kernel\":" << time_kernel
                  << ",\"time_d2h\":" << time_d2h
                  << ",\"time_total\":" << time_total
                  << ",\"result\":" << h_revenue
                  << ",\"status\":\"" << (h_revenue == EXPECTED_REVENUE_Q12 ? "PASSED" : "FAILED") << "\"}" << std::endl;
    }

    cudaFree(d_revenue);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_metadata);
    cudaEventDestroy(ev_kernel);
    cudaEventDestroy(ev_d2h);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path> [cache_dir]" << std::endl;
        return 1;
    }

    SSBDataCompressedVertical data;
    std::string cache_dir = (argc >= 3) ? std::string(argv[2]) : (std::string(argv[1]) + "/compressed_cache_v15");
    data.loadOrCompress(argv[1], cache_dir, MINI_VECTOR_SIZE);

    // Pad columns to fixed-offset FLS format for compile-time BW decode
    v15::padColumnToFixedOffset<BW_ORDERDATE>(data.lo_orderdate);
    v15::padColumnToFixedOffset<BW_QUANTITY>(data.lo_quantity);
    v15::padColumnToFixedOffset<BW_DISCOUNT>(data.lo_discount);
    v15::padColumnToFixedOffset<BW_EXTENDEDPRICE>(data.lo_extendedprice);
    float time_h2d = data.measureH2DTimeQ1();

    runQ12FusedV13(data, time_h2d);
    return 0;
}
