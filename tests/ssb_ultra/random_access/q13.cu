/**
 * @file q13.cu
 * @brief SSB Q1.3 Implementation - Random Access Optimization Strategy
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey AND d_weeknuminyear = 6 AND d_year = 1994
 *     AND lo_discount BETWEEN 5 AND 7 AND lo_quantity BETWEEN 26 AND 35;
 *
 * Strategy: Filter column full decompress + non-filter column random access
 * Note: Week 6 of 1994 = Feb 7-13, 1994 (dates 19940207-19940213)
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <chrono>

#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "ssb_filter_kernels.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

// Q1.3 Parameters: Week 6 of 1994 (Feb 7-13)
constexpr uint32_t Q13_DATE_MIN = 19940207;
constexpr uint32_t Q13_DATE_MAX = 19940213;
constexpr uint32_t DISCOUNT_MIN = 5;
constexpr uint32_t DISCOUNT_MAX = 7;
constexpr uint32_t QUANTITY_MIN = 26;
constexpr uint32_t QUANTITY_MAX = 35;

/**
 * @brief Secondary filter kernel for Q1.3 (quantity range)
 */
__global__ void filterSecondaryQ13Kernel(
    const uint32_t* __restrict__ lo_discount,
    const uint32_t* __restrict__ lo_quantity,
    const int* __restrict__ input_indices,
    int num_rows,
    uint32_t discount_min, uint32_t discount_max,
    uint32_t quantity_min, uint32_t quantity_max,
    int* __restrict__ passing_indices,
    int* __restrict__ num_passing)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    __shared__ int s_warp_counts[8];
    __shared__ int s_warp_offsets[8];
    __shared__ int s_block_offset;

    int passes = 0;
    int original_idx = -1;
    if (tid < num_rows) {
        uint32_t discount = lo_discount[tid];
        uint32_t quantity = lo_quantity[tid];
        original_idx = input_indices[tid];

        passes = (discount >= discount_min && discount <= discount_max &&
                  quantity >= quantity_min && quantity <= quantity_max) ? 1 : 0;
    }

    unsigned int warp_ballot = __ballot_sync(0xffffffff, passes);
    int warp_count = __popc(warp_ballot);

    if (lane == 0) {
        s_warp_counts[warp_id] = warp_count;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) {
            s_warp_offsets[w] = total;
            total += s_warp_counts[w];
        }
        if (total > 0) {
            s_block_offset = atomicAdd(num_passing, total);
        }
    }
    __syncthreads();

    if (passes && tid < num_rows) {
        unsigned int mask = (1u << lane) - 1;
        int pos_in_warp = __popc(warp_ballot & mask);
        int global_pos = s_block_offset + s_warp_offsets[warp_id] + pos_in_warp;
        passing_indices[global_pos] = original_idx;
    }
}

/**
 * @brief Run Q1.3 with random access optimization
 */
void runQ13RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    int total_rows = acc_orderdate.getTotalElements();

    // Step 1: Full decompress filter column
    uint32_t* d_orderdate;
    cudaMalloc(&d_orderdate, total_rows * sizeof(uint32_t));

    timer.start();
    acc_orderdate.decompressAll(d_orderdate, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_orderdate_ms = timer.elapsed_ms();

    // Step 2: Date filter -> passing indices
    int* d_passing_indices;
    int* d_num_passing;
    cudaMalloc(&d_passing_indices, total_rows * sizeof(int));
    cudaMalloc(&d_num_passing, sizeof(int));
    cudaMemset(d_num_passing, 0, sizeof(int));

    timer.start();
    launchDateRangeFilter(d_orderdate, total_rows, Q13_DATE_MIN, Q13_DATE_MAX,
                          d_passing_indices, d_num_passing, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float filter_ms = timer.elapsed_ms();

    int h_num_passing;
    cudaMemcpy(&h_num_passing, d_num_passing, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_orderdate);

    float date_selectivity = 100.0f * h_num_passing / total_rows;
    std::cout << "Date filter selectivity: " << date_selectivity << "%" << std::endl;

    if (h_num_passing == 0) {
        std::cout << "\n=== Q1.3 Results (Random Access) ===" << std::endl;
        std::cout << "Revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_passing_indices); cudaFree(d_num_passing);
        return;
    }

    // Step 3: Random access discount, quantity
    uint32_t *d_discount, *d_quantity;
    cudaMalloc(&d_discount, h_num_passing * sizeof(uint32_t));
    cudaMalloc(&d_quantity, h_num_passing * sizeof(uint32_t));

    timer.start();
    acc_discount.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_discount, stream);
    acc_quantity.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_quantity, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_dq_ms = timer.elapsed_ms();

    // Step 4: Secondary filter (discount + quantity range)
    int* d_final_indices;
    int* d_num_final;
    cudaMalloc(&d_final_indices, h_num_passing * sizeof(int));
    cudaMalloc(&d_num_final, sizeof(int));
    cudaMemset(d_num_final, 0, sizeof(int));

    timer.start();
    int grid_size = (h_num_passing + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
    filterSecondaryQ13Kernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_discount, d_quantity, d_passing_indices,
        h_num_passing,
        DISCOUNT_MIN, DISCOUNT_MAX, QUANTITY_MIN, QUANTITY_MAX,
        d_final_indices, d_num_final);
    cudaStreamSynchronize(stream);
    timer.stop();
    float filter_secondary_ms = timer.elapsed_ms();

    int h_num_final;
    cudaMemcpy(&h_num_final, d_num_final, sizeof(int), cudaMemcpyDeviceToHost);

    float final_selectivity = 100.0f * h_num_final / total_rows;
    std::cout << "Final selectivity: " << final_selectivity << "%" << std::endl;

    cudaFree(d_passing_indices); cudaFree(d_num_passing);

    if (h_num_final == 0) {
        std::cout << "\n=== Q1.3 Results (Random Access) ===" << std::endl;
        std::cout << "Revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_discount); cudaFree(d_quantity);
        cudaFree(d_final_indices); cudaFree(d_num_final);
        return;
    }

    // Step 5: Random access extendedprice + discount for final rows
    uint32_t *d_extendedprice, *d_discount_final;
    cudaMalloc(&d_extendedprice, h_num_final * sizeof(uint32_t));
    cudaMalloc(&d_discount_final, h_num_final * sizeof(uint32_t));

    timer.start();
    acc_extendedprice.randomAccessBatchIndices(d_final_indices, h_num_final, d_extendedprice, stream);
    acc_discount.randomAccessBatchIndices(d_final_indices, h_num_final, d_discount_final, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_ep_ms = timer.elapsed_ms();

    // Step 6: Aggregation
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    timer.start();
    launchRevenueAggregation(d_extendedprice, d_discount_final, h_num_final, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    timing.data_load_ms = decompress_orderdate_ms + random_access_dq_ms + random_access_ep_ms;
    timing.hash_build_ms = 0;
    timing.kernel_ms = filter_ms + filter_secondary_ms + aggregate_ms;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    std::cout << "\n=== Q1.3 Results (Random Access) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Decompress orderdate: " << decompress_orderdate_ms << " ms" << std::endl;
    std::cout << "  Date filter:          " << filter_ms << " ms" << std::endl;
    std::cout << "  Random access D+Q:    " << random_access_dq_ms << " ms" << std::endl;
    std::cout << "  Secondary filter:     " << filter_secondary_ms << " ms" << std::endl;
    std::cout << "  Random access EP:     " << random_access_ep_ms << " ms" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q1.3");

    // Cleanup
    cudaFree(d_discount); cudaFree(d_quantity);
    cudaFree(d_extendedprice); cudaFree(d_discount_final);
    cudaFree(d_final_indices); cudaFree(d_num_final);
    cudaFree(d_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.3 - Random Access Optimization ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    QueryTiming timing;
    runQ13RandomAccess(data, timing);

    // Benchmark runs
    std::cout << "\n=== Benchmark Runs ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ13RandomAccess(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
