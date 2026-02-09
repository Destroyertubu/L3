/**
 * SSB Q1.1 - V13 (L3 V4 Vectorized Decoder, 2048 values per mini-vector)
 *
 * Query: SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *        FROM lineorder, date
 *        WHERE lo_orderdate = d_datekey
 *          AND d_year = 1993
 *          AND lo_discount BETWEEN 1 AND 3
 *          AND lo_quantity < 25
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

__global__ void q11_fused_kernel_v15(
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
    int tail_start = num_full_mv * V15_MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    unsigned long long local_revenue = 0;

    // ===== Process full mini-vectors =====
    if (num_full_mv > 0) {
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
    }

    // ===== Process tail (scalar, loop for tail_size > 32) =====
    for (int t = lane_id; t < tail_size; t += V15_WARP_SIZE) {
        int value_idx = tail_start + t;

        uint32_t od = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
        if (od >= YEAR_1993_START && od <= YEAR_1993_END) {
            uint32_t qty = decode_tail_value_v15_opt<BW_QUANTITY>(lo_quantity, partition_idx, value_idx);
            if (qty < 25) {
                uint32_t disc = decode_tail_value_v15_opt<BW_DISCOUNT>(lo_discount, partition_idx, value_idx);
                if (disc >= 1 && disc <= 3) {
                    uint32_t price = decode_tail_value_v15_opt<BW_EXTENDEDPRICE>(lo_extendedprice, partition_idx, value_idx);
                    local_revenue += static_cast<unsigned long long>(price) * static_cast<unsigned long long>(disc);
                }
            }
        }
    }

    // ===== Warp reduction =====
    for (int offset = 16; offset > 0; offset /= 2) {
        local_revenue += __shfl_down_sync(0xFFFFFFFF, local_revenue, offset);
    }

    if (lane_id == 0 && local_revenue > 0) {
        atomicAdd(global_revenue, local_revenue);
    }
}

void runQ11FusedV13(SSBDataCompressedVertical& data, float time_h2d) {
    cudaEvent_t ev_start, ev_metadata, ev_kernel, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_metadata);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = data.lo_orderdate.num_partitions;

    // ===== Stage 1: Metadata =====
    cudaEventRecord(ev_start);

    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));

    cudaEventRecord(ev_metadata);

    // ===== Warmup =====
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));
    q11_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
        num_partitions, total_rows, d_revenue);
    cudaDeviceSynchronize();

    // ===== Benchmark runs =====
    const int RUNS = 5;
    constexpr unsigned long long EXPECTED_SF20 = 81237793583ULL;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_revenue, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cudaEventRecord(ev_metadata);

        q11_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_orderdate, data.lo_quantity, data.lo_discount, data.lo_extendedprice,
            num_partitions, total_rows, d_revenue);
        cudaEventRecord(ev_kernel);

        unsigned long long h_revenue;
        cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);

        float time_metadata, time_kernel, time_d2h;
        cudaEventElapsedTime(&time_metadata, ev_start, ev_metadata);
        cudaEventElapsedTime(&time_kernel, ev_metadata, ev_kernel);
        cudaEventElapsedTime(&time_d2h, ev_kernel, ev_d2h);
        float time_total = time_h2d + time_metadata + time_kernel + time_d2h;

        std::cout << "{\"query\":11,\"version\":\"v14\",\"run\":" << run
                  << ",\"time_h2d\":" << time_h2d
                  << ",\"time_metadata\":" << time_metadata
                  << ",\"time_ht_build\":0"
                  << ",\"time_kernel\":" << time_kernel
                  << ",\"time_d2h\":" << time_d2h
                  << ",\"time_total\":" << time_total
                  << ",\"result\":" << h_revenue
                  << ",\"status\":\"" << (h_revenue == EXPECTED_SF20 ? "PASSED" : "FAILED") << "\"}" << std::endl;
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

    runQ11FusedV13(data, time_h2d);
    return 0;
}
