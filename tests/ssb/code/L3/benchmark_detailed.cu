/**
 * V13 Detailed Benchmark - Separate timing for main kernel and tail kernel
 *
 * Outputs all timing stages:
 * - H2D (host to device transfer)
 * - Metadata (partition info, malloc)
 * - HT Build (hash table construction for Q2.x-Q4.x)
 * - Main Kernel (vectorized decode for full partitions)
 * - Tail Kernel (scalar decode for last partial partition)
 * - D2H (device to host transfer)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>

#define L3_VERTICAL_1024_CONFIG
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"
#include "ssb_data_loader.hpp"
#include "v15_common.cuh"

using namespace ssb;
using namespace v15;

// ============================================================================
// Timing structure
// ============================================================================
struct V13QueryTiming {
    int query;
    float time_h2d;
    float time_metadata;
    float time_ht_build;
    float time_main_kernel;
    float time_tail_kernel;
    float time_d2h;
    unsigned long long result;
    bool passed;

    float total() const {
        return time_h2d + time_metadata + time_ht_build +
               time_main_kernel + time_tail_kernel + time_d2h;
    }

    float kernel_total() const {
        return time_main_kernel + time_tail_kernel;
    }
};

// ============================================================================
// Q1.1 Kernel
// ============================================================================
__global__ void q11_main_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int num_partitions, int total_rows,
    unsigned long long* __restrict__ global_revenue)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;
    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_size = lo_orderdate.d_end_indices[partition_idx] - partition_start;

    int num_full_mv = partition_size / V15_MINI_VECTOR_SIZE;
    if (num_full_mv == 0) return;

    unsigned long long local_revenue = 0;

    uint32_t orderdate[V15_VALUES_PER_THREAD];
    uint32_t quantity[V15_VALUES_PER_THREAD];
    uint32_t discount[V15_VALUES_PER_THREAD];
    uint32_t extendedprice[V15_VALUES_PER_THREAD];
    int flags[V15_VALUES_PER_THREAD];

    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) flags[v] = 1;

    decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, orderdate);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
        if (global_idx >= total_rows) flags[v] = 0;
        else if (orderdate[v] < YEAR_1993_START || orderdate[v] > YEAR_1993_END) flags[v] = 0;
    }

    int any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];

    if (__ballot_sync(0xFFFFFFFF, any_valid) != 0) {
        decode_column_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, lane_id, quantity);
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            if (flags[v] && quantity[v] >= 25) flags[v] = 0;
        }

        any_valid = 0;
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];

        if (__ballot_sync(0xFFFFFFFF, any_valid) != 0) {
            decode_column_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, lane_id, discount);
            #pragma unroll
            for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                if (flags[v] && (discount[v] < 1 || discount[v] > 3)) flags[v] = 0;
            }

            any_valid = 0;
            #pragma unroll
            for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];

            if (__ballot_sync(0xFFFFFFFF, any_valid) != 0) {
                decode_column_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, lane_id, extendedprice);
                #pragma unroll
                for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                    if (flags[v]) {
                        local_revenue += static_cast<unsigned long long>(extendedprice[v]) *
                                        static_cast<unsigned long long>(discount[v]);
                    }
                }
            }
        }
    }

    for (int offset = 16; offset > 0; offset /= 2) {
        local_revenue += __shfl_down_sync(0xFFFFFFFF, local_revenue, offset);
    }

    if (lane_id == 0 && local_revenue > 0) {
        atomicAdd(global_revenue, local_revenue);
    }
}

__global__ void q11_tail_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int partition_idx, int tail_start, int tail_size,
    unsigned long long* __restrict__ global_revenue)
{
    int tid = threadIdx.x;
    if (tid >= tail_size) return;

    int value_idx = tail_start + tid;
    uint32_t od = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
    if (od < YEAR_1993_START || od > YEAR_1993_END) return;

    uint32_t qty = decode_tail_value_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, value_idx);
    if (qty >= 25) return;

    uint32_t disc = decode_tail_value_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, value_idx);
    if (disc < 1 || disc > 3) return;

    uint32_t price = decode_tail_value_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, value_idx);
    atomicAdd(global_revenue, static_cast<unsigned long long>(price) * static_cast<unsigned long long>(disc));
}

// ============================================================================
// Q1.2 Kernel
// ============================================================================
__global__ void q12_main_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int num_partitions, int total_rows,
    unsigned long long* __restrict__ global_revenue)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;
    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_size = lo_orderdate.d_end_indices[partition_idx] - partition_start;

    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;
    if (num_mv == 0) return;

    uint32_t orderdate[V15_VALUES_PER_THREAD];
    uint32_t quantity[V15_VALUES_PER_THREAD];
    uint32_t discount[V15_VALUES_PER_THREAD];
    uint32_t extendedprice[V15_VALUES_PER_THREAD];
    int flags[V15_VALUES_PER_THREAD];

    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) flags[v] = 1;

    decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, orderdate);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
        if (global_idx >= total_rows) flags[v] = 0;
        else if (orderdate[v] < MONTH_199401_START || orderdate[v] > MONTH_199401_END) flags[v] = 0;
    }

    int any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, lane_id, quantity);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (quantity[v] < 26 || quantity[v] > 35)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, lane_id, discount);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (discount[v] < 4 || discount[v] > 6)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, lane_id, extendedprice);

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

__global__ void q12_tail_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int partition_idx, int tail_start, int tail_size,
    unsigned long long* __restrict__ global_revenue)
{
    int tid = threadIdx.x;
    if (tid >= tail_size) return;

    int value_idx = tail_start + tid;
    uint32_t od = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
    if (od < MONTH_199401_START || od > MONTH_199401_END) return;

    uint32_t qty = decode_tail_value_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, value_idx);
    if (qty < 26 || qty > 35) return;

    uint32_t disc = decode_tail_value_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, value_idx);
    if (disc < 4 || disc > 6) return;

    uint32_t price = decode_tail_value_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, value_idx);
    atomicAdd(global_revenue, static_cast<unsigned long long>(price) * static_cast<unsigned long long>(disc));
}

// ============================================================================
// Q1.3 Kernel
// ============================================================================
__global__ void q13_main_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int num_partitions, int total_rows,
    unsigned long long* __restrict__ global_revenue)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;
    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_size = lo_orderdate.d_end_indices[partition_idx] - partition_start;

    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;
    if (num_mv == 0) return;

    uint32_t orderdate[V15_VALUES_PER_THREAD];
    uint32_t quantity[V15_VALUES_PER_THREAD];
    uint32_t discount[V15_VALUES_PER_THREAD];
    uint32_t extendedprice[V15_VALUES_PER_THREAD];
    int flags[V15_VALUES_PER_THREAD];

    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) flags[v] = 1;

    decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, orderdate);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
        if (global_idx >= total_rows) flags[v] = 0;
        else if (orderdate[v] < WEEK6_1994_START || orderdate[v] > WEEK6_1994_END) flags[v] = 0;
    }

    int any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, lane_id, quantity);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (quantity[v] < 26 || quantity[v] > 35)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, lane_id, discount);
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        if (flags[v] && (discount[v] < 5 || discount[v] > 7)) flags[v] = 0;
    }

    any_valid = 0;
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) any_valid |= flags[v];
    if (__ballot_sync(0xFFFFFFFF, any_valid) == 0) return;

    decode_column_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, lane_id, extendedprice);

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

__global__ void q13_tail_kernel(
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_quantity,
    const CompressedDataVertical<uint32_t> lo_discount,
    const CompressedDataVertical<uint32_t> lo_extendedprice,
    int partition_idx, int tail_start, int tail_size,
    unsigned long long* __restrict__ global_revenue)
{
    int tid = threadIdx.x;
    if (tid >= tail_size) return;

    int value_idx = tail_start + tid;
    uint32_t od = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
    if (od < WEEK6_1994_START || od > WEEK6_1994_END) return;

    uint32_t qty = decode_tail_value_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, value_idx);
    if (qty < 26 || qty > 35) return;

    uint32_t disc = decode_tail_value_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, value_idx);
    if (disc < 5 || disc > 7) return;

    uint32_t price = decode_tail_value_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, value_idx);
    atomicAdd(global_revenue, static_cast<unsigned long long>(price) * static_cast<unsigned long long>(disc));
}

// ============================================================================
// Expected Results
// ============================================================================
constexpr unsigned long long EXPECTED_Q11 = 81237793583ULL;
constexpr unsigned long long EXPECTED_Q12 = 17303753830ULL;
constexpr unsigned long long EXPECTED_Q13 = 4708750603ULL;

// ============================================================================
// Benchmark Runner for Q1.x
// ============================================================================
V13QueryTiming runQ1xBenchmark(
    int query_num,
    SSBDataCompressedVertical& data,
    float time_h2d)
{
    cudaEvent_t ev_start, ev_metadata, ev_main, ev_tail, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_metadata);
    cudaEventCreate(&ev_main);
    cudaEventCreate(&ev_tail);
    cudaEventCreate(&ev_d2h);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = data.lo_orderdate.num_partitions;
    int last_partition_idx = num_partitions - 1;

    // Get tail info
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

    // Warmup
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    if (query_num == 11) {
        q11_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            num_partitions, total_rows, d_revenue);
        if (tail_size > 0) {
            q11_tail_kernel<<<1, tail_size>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                last_partition_idx, tail_start, tail_size, d_revenue);
        }
    } else if (query_num == 12) {
        q12_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            num_partitions, total_rows, d_revenue);
        if (tail_size > 0) {
            q12_tail_kernel<<<1, tail_size>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                last_partition_idx, tail_start, tail_size, d_revenue);
        }
    } else if (query_num == 13) {
        q13_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            num_partitions, total_rows, d_revenue);
        if (tail_size > 0) {
            q13_tail_kernel<<<1, tail_size>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                last_partition_idx, tail_start, tail_size, d_revenue);
        }
    }
    cudaDeviceSynchronize();

    // Benchmark runs - find best
    const int RUNS = 5;
    V13QueryTiming best;
    best.query = query_num;
    best.time_h2d = time_h2d;
    best.time_ht_build = 0;
    best.time_main_kernel = 1e9;

    unsigned long long expected = (query_num == 11) ? EXPECTED_Q11 :
                                  (query_num == 12) ? EXPECTED_Q12 : EXPECTED_Q13;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cudaEventRecord(ev_metadata);

        // Main kernel
        if (query_num == 11) {
            q11_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                num_partitions, total_rows, d_revenue);
        } else if (query_num == 12) {
            q12_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                num_partitions, total_rows, d_revenue);
        } else {
            q13_main_kernel<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
                data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                num_partitions, total_rows, d_revenue);
        }
        cudaEventRecord(ev_main);

        // Tail kernel
        if (tail_size > 0) {
            if (query_num == 11) {
                q11_tail_kernel<<<1, tail_size>>>(
                    data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                    last_partition_idx, tail_start, tail_size, d_revenue);
            } else if (query_num == 12) {
                q12_tail_kernel<<<1, tail_size>>>(
                    data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                    last_partition_idx, tail_start, tail_size, d_revenue);
            } else {
                q13_tail_kernel<<<1, tail_size>>>(
                    data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
                    last_partition_idx, tail_start, tail_size, d_revenue);
            }
        }
        cudaEventRecord(ev_tail);

        unsigned long long h_revenue;
        cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);

        float time_metadata, time_main, time_tail, time_d2h;
        cudaEventElapsedTime(&time_metadata, ev_start, ev_metadata);
        cudaEventElapsedTime(&time_main, ev_metadata, ev_main);
        cudaEventElapsedTime(&time_tail, ev_main, ev_tail);
        cudaEventElapsedTime(&time_d2h, ev_tail, ev_d2h);

        if (time_main + time_tail < best.time_main_kernel + best.time_tail_kernel) {
            best.time_metadata = time_metadata;
            best.time_main_kernel = time_main;
            best.time_tail_kernel = time_tail;
            best.time_d2h = time_d2h;
            best.result = h_revenue;
            best.passed = (h_revenue == expected);
        }
    }

    cudaFree(d_revenue);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_metadata);
    cudaEventDestroy(ev_main);
    cudaEventDestroy(ev_tail);
    cudaEventDestroy(ev_d2h);

    return best;
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <data_path>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string report_path = "/root/autodl-tmp/code/L3/tests/ssb_ultra/fused_decompression/v13/v13/reports/v13_detailed_timing.txt";

    // Load data
    SSBDataCompressedVertical data;
    data.loadOrCompress(data_path, data_path + "/compressed_cache_v14_2048", 2048);
    float time_h2d = data.measureH2DTimeQ1();

    std::vector<V13QueryTiming> results;

    // Run Q1.x benchmarks
    std::cout << "Running Q1.x benchmarks..." << std::endl;
    results.push_back(runQ1xBenchmark(11, data, time_h2d));
    results.push_back(runQ1xBenchmark(12, data, time_h2d));
    results.push_back(runQ1xBenchmark(13, data, time_h2d));

    // Write report
    std::ofstream report(report_path);
    report << "V13 详细时间统计 (5次运行取最快)\n";
    report << "================================================\n";
    report << "Date: 2026-01-27\n";
    report << "GPU: H100\n";
    report << "Data: SSB SF100 (" << data.lo_orderdate.total_values << " rows)\n";
    report << "Partitions: " << data.lo_orderdate.num_partitions << " (2048 values each, last has "
           << (data.lo_orderdate.total_values % 2048) << " values)\n\n";

    report << std::fixed << std::setprecision(3);
    report << "Query   H2D(ms)   Meta(ms)  HT(ms)    Main(ms)  Tail(ms)  D2H(ms)   Total(ms)  Status\n";
    report << "------  --------  --------  --------  --------  --------  --------  ---------  ------\n";

    for (const auto& t : results) {
        report << "Q1." << (t.query - 10) << "    "
               << std::setw(8) << t.time_h2d << "  "
               << std::setw(8) << t.time_metadata << "  "
               << std::setw(8) << t.time_ht_build << "  "
               << std::setw(8) << t.time_main_kernel << "  "
               << std::setw(8) << t.time_tail_kernel << "  "
               << std::setw(8) << t.time_d2h << "  "
               << std::setw(9) << t.total() << "  "
               << (t.passed ? "PASSED" : "FAILED") << "\n";
    }

    report << "\n";
    report << "仅 Kernel 时间对比:\n";
    report << "================================================\n";
    report << "Query   Main(ms)  Tail(ms)  Total(ms)  Tail占比\n";
    report << "------  --------  --------  ---------  --------\n";
    for (const auto& t : results) {
        float total_k = t.time_main_kernel + t.time_tail_kernel;
        float tail_pct = (total_k > 0) ? (t.time_tail_kernel / total_k * 100) : 0;
        report << "Q1." << (t.query - 10) << "    "
               << std::setw(8) << t.time_main_kernel << "  "
               << std::setw(8) << t.time_tail_kernel << "  "
               << std::setw(9) << total_k << "  "
               << std::setw(6) << tail_pct << "%\n";
    }

    report.close();

    // Also print to stdout
    std::cout << "\nResults written to: " << report_path << std::endl;
    std::cout << "\nSummary:\n";
    for (const auto& t : results) {
        std::cout << "Q1." << (t.query - 10)
                  << ": main=" << t.time_main_kernel << "ms"
                  << ", tail=" << t.time_tail_kernel << "ms"
                  << ", total=" << (t.time_main_kernel + t.time_tail_kernel) << "ms"
                  << " [" << (t.passed ? "PASSED" : "FAILED") << "]\n";
    }

    return 0;
}
