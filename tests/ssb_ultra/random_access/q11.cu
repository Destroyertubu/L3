/**
 * @file q11_random_access.cu
 * @brief SSB Q1.1 Implementation - Random Access Optimization Strategy
 *
 * Query:
 *   SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 *   FROM lineorder, date
 *   WHERE lo_orderdate = d_datekey AND d_year = 1993
 *     AND lo_discount BETWEEN 1 AND 3 AND lo_quantity < 25;
 *
 * Strategy:
 *   1. Full decompress filter column (lo_orderdate)
 *   2. GPU stream compaction to generate passing_indices
 *   3. Random access non-filter columns using passing_indices
 *   4. Final aggregation on filtered data
 *
 * This reduces data decompression when selectivity is low (<25%).
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

constexpr uint32_t DATE_1993_MIN = 19930101;
constexpr uint32_t DATE_1993_MAX = 19931231;
constexpr uint32_t DISCOUNT_MIN = 1;
constexpr uint32_t DISCOUNT_MAX = 3;
constexpr uint32_t QUANTITY_MAX = 25;

/**
 * @brief Run Q1.1 with random access optimization
 *
 * Execution flow:
 * 1. Full decompress lo_orderdate (filter column)
 * 2. GPU filter: generate date_passing_indices
 * 3. Random access lo_discount, lo_quantity using date_passing_indices
 * 4. GPU filter: generate final_passing_indices (date + discount + quantity)
 * 5. Random access lo_extendedprice using final_passing_indices
 * 6. Aggregation on final filtered data
 */
void runQ11RandomAccess(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;

    // Create accessors
    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    int total_rows = acc_orderdate.getTotalElements();
    std::cout << "Total rows: " << total_rows << std::endl;

    // =========================================================================
    // Step 1: Full decompress filter column (lo_orderdate)
    // =========================================================================
    uint32_t* d_orderdate;
    cudaMalloc(&d_orderdate, total_rows * sizeof(uint32_t));

    timer.start();
    acc_orderdate.decompressAll(d_orderdate, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_orderdate_ms = timer.elapsed_ms();

    // =========================================================================
    // Step 2: GPU stream compaction - filter by date range
    // =========================================================================
    int* d_date_passing_indices;
    int* d_num_date_passing;
    cudaMalloc(&d_date_passing_indices, total_rows * sizeof(int));  // Worst case
    cudaMalloc(&d_num_date_passing, sizeof(int));
    cudaMemset(d_num_date_passing, 0, sizeof(int));

    timer.start();
    launchDateRangeFilter(
        d_orderdate, total_rows,
        DATE_1993_MIN, DATE_1993_MAX,
        d_date_passing_indices, d_num_date_passing, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float filter_date_ms = timer.elapsed_ms();

    int h_num_date_passing;
    cudaMemcpy(&h_num_date_passing, d_num_date_passing, sizeof(int), cudaMemcpyDeviceToHost);

    float date_selectivity = 100.0f * h_num_date_passing / total_rows;
    std::cout << "Date filter: " << h_num_date_passing << "/" << total_rows
              << " (" << date_selectivity << "% selectivity)" << std::endl;

    // Free orderdate - no longer needed
    cudaFree(d_orderdate);

    if (h_num_date_passing == 0) {
        timing.data_load_ms = decompress_orderdate_ms + filter_date_ms;
        timing.kernel_ms = 0;
        timing.hash_build_ms = 0;
        timing.total_ms = timing.data_load_ms;
        std::cout << "\n=== Q1.1 Results (Random Access) ===" << std::endl;
        std::cout << "Revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_date_passing_indices);
        cudaFree(d_num_date_passing);
        return;
    }

    // =========================================================================
    // Step 3: Random access lo_discount, lo_quantity using date_passing_indices
    // =========================================================================
    uint32_t *d_discount, *d_quantity;
    cudaMalloc(&d_discount, h_num_date_passing * sizeof(uint32_t));
    cudaMalloc(&d_quantity, h_num_date_passing * sizeof(uint32_t));

    timer.start();
    acc_discount.randomAccessBatchIndices(d_date_passing_indices, h_num_date_passing, d_discount, stream);
    acc_quantity.randomAccessBatchIndices(d_date_passing_indices, h_num_date_passing, d_quantity, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_dq_ms = timer.elapsed_ms();

    // =========================================================================
    // Step 4: Secondary filter - apply discount/quantity predicates
    // =========================================================================
    int* d_final_passing_indices;
    int* d_num_final_passing;
    cudaMalloc(&d_final_passing_indices, h_num_date_passing * sizeof(int));
    cudaMalloc(&d_num_final_passing, sizeof(int));
    cudaMemset(d_num_final_passing, 0, sizeof(int));

    timer.start();
    int grid_size = (h_num_date_passing + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE;
    filterSecondaryCompactKernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_discount, d_quantity, d_date_passing_indices,
        h_num_date_passing,
        DISCOUNT_MIN, DISCOUNT_MAX, QUANTITY_MAX,
        d_final_passing_indices, d_num_final_passing);
    cudaStreamSynchronize(stream);
    timer.stop();
    float filter_secondary_ms = timer.elapsed_ms();

    int h_num_final_passing;
    cudaMemcpy(&h_num_final_passing, d_num_final_passing, sizeof(int), cudaMemcpyDeviceToHost);

    float final_selectivity = 100.0f * h_num_final_passing / total_rows;
    std::cout << "Final filter: " << h_num_final_passing << "/" << total_rows
              << " (" << final_selectivity << "% selectivity)" << std::endl;

    // Free intermediate buffers
    cudaFree(d_date_passing_indices);
    cudaFree(d_num_date_passing);

    if (h_num_final_passing == 0) {
        timing.data_load_ms = decompress_orderdate_ms + filter_date_ms + random_access_dq_ms + filter_secondary_ms;
        timing.kernel_ms = 0;
        timing.hash_build_ms = 0;
        timing.total_ms = timing.data_load_ms;
        std::cout << "\n=== Q1.1 Results (Random Access) ===" << std::endl;
        std::cout << "Revenue: 0 (no matching rows)" << std::endl;
        cudaFree(d_discount); cudaFree(d_quantity);
        cudaFree(d_final_passing_indices); cudaFree(d_num_final_passing);
        return;
    }

    // =========================================================================
    // Step 5: Random access lo_extendedprice using final_passing_indices
    // =========================================================================
    uint32_t* d_extendedprice;
    cudaMalloc(&d_extendedprice, h_num_final_passing * sizeof(uint32_t));

    timer.start();
    acc_extendedprice.randomAccessBatchIndices(d_final_passing_indices, h_num_final_passing, d_extendedprice, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_ep_ms = timer.elapsed_ms();

    // Also need discount for aggregation - random access again for final rows
    uint32_t* d_discount_final;
    cudaMalloc(&d_discount_final, h_num_final_passing * sizeof(uint32_t));
    acc_discount.randomAccessBatchIndices(d_final_passing_indices, h_num_final_passing, d_discount_final, stream);
    cudaStreamSynchronize(stream);

    // =========================================================================
    // Step 6: Final aggregation (all inputs are pre-filtered)
    // =========================================================================
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    timer.start();
    launchRevenueAggregation(d_extendedprice, d_discount_final, h_num_final_passing, d_revenue, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // =========================================================================
    // Timing summary
    // =========================================================================
    timing.data_load_ms = decompress_orderdate_ms + random_access_dq_ms + random_access_ep_ms;
    timing.hash_build_ms = 0;
    timing.kernel_ms = filter_date_ms + filter_secondary_ms + aggregate_ms;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    std::cout << "\n=== Q1.1 Results (Random Access) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming breakdown:" << std::endl;
    std::cout << "  Decompress orderdate: " << decompress_orderdate_ms << " ms" << std::endl;
    std::cout << "  Date filter:          " << filter_date_ms << " ms" << std::endl;
    std::cout << "  Random access D+Q:    " << random_access_dq_ms << " ms" << std::endl;
    std::cout << "  Secondary filter:     " << filter_secondary_ms << " ms" << std::endl;
    std::cout << "  Random access EP:     " << random_access_ep_ms << " ms" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q1.1");

    // Cleanup
    cudaFree(d_discount); cudaFree(d_quantity);
    cudaFree(d_extendedprice); cudaFree(d_discount_final);
    cudaFree(d_final_passing_indices); cudaFree(d_num_final_passing);
    cudaFree(d_revenue);
}

/**
 * @brief Simpler version: Full decompress orderdate, random access everything else
 *
 * This is a simpler 1-stage approach for comparison.
 */
void runQ11RandomAccessSimple(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer;
    cudaStream_t stream = 0;

    CompressedColumnAccessorVertical<uint32_t> acc_orderdate(&data.lo_orderdate);
    CompressedColumnAccessorVertical<uint32_t> acc_discount(&data.lo_discount);
    CompressedColumnAccessorVertical<uint32_t> acc_quantity(&data.lo_quantity);
    CompressedColumnAccessorVertical<uint32_t> acc_extendedprice(&data.lo_extendedprice);

    int total_rows = acc_orderdate.getTotalElements();

    // Step 1: Full decompress orderdate
    uint32_t* d_orderdate;
    cudaMalloc(&d_orderdate, total_rows * sizeof(uint32_t));

    timer.start();
    acc_orderdate.decompressAll(d_orderdate, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float decompress_ms = timer.elapsed_ms();

    // Step 2: Date filter -> passing indices
    int* d_passing_indices;
    int* d_num_passing;
    cudaMalloc(&d_passing_indices, total_rows * sizeof(int));
    cudaMalloc(&d_num_passing, sizeof(int));
    cudaMemset(d_num_passing, 0, sizeof(int));

    timer.start();
    launchDateRangeFilter(d_orderdate, total_rows, DATE_1993_MIN, DATE_1993_MAX,
                          d_passing_indices, d_num_passing, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float filter_ms = timer.elapsed_ms();

    int h_num_passing;
    cudaMemcpy(&h_num_passing, d_num_passing, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_orderdate);

    std::cout << "Date filter selectivity: " << (100.0f * h_num_passing / total_rows) << "%" << std::endl;

    if (h_num_passing == 0) {
        std::cout << "Revenue: 0" << std::endl;
        cudaFree(d_passing_indices); cudaFree(d_num_passing);
        return;
    }

    // Step 3: Random access discount, quantity, extendedprice
    uint32_t *d_discount, *d_quantity, *d_extendedprice;
    cudaMalloc(&d_discount, h_num_passing * sizeof(uint32_t));
    cudaMalloc(&d_quantity, h_num_passing * sizeof(uint32_t));
    cudaMalloc(&d_extendedprice, h_num_passing * sizeof(uint32_t));

    timer.start();
    acc_discount.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_discount, stream);
    acc_quantity.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_quantity, stream);
    acc_extendedprice.randomAccessBatchIndices(d_passing_indices, h_num_passing, d_extendedprice, stream);
    cudaStreamSynchronize(stream);
    timer.stop();
    float random_access_ms = timer.elapsed_ms();

    // Step 4: Aggregation with secondary filter
    unsigned long long* d_revenue;
    cudaMalloc(&d_revenue, sizeof(unsigned long long));
    cudaMemset(d_revenue, 0, sizeof(unsigned long long));

    timer.start();
    int grid_size = min((h_num_passing + FILTER_BLOCK_SIZE - 1) / FILTER_BLOCK_SIZE, 256);
    aggregateRevenueWithFilterKernel<<<grid_size, FILTER_BLOCK_SIZE, 0, stream>>>(
        d_extendedprice, d_discount, d_quantity, h_num_passing,
        DISCOUNT_MIN, DISCOUNT_MAX, QUANTITY_MAX, d_revenue);
    cudaStreamSynchronize(stream);
    timer.stop();
    float aggregate_ms = timer.elapsed_ms();

    unsigned long long h_revenue;
    cudaMemcpy(&h_revenue, d_revenue, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    timing.data_load_ms = decompress_ms + random_access_ms;
    timing.kernel_ms = filter_ms + aggregate_ms;
    timing.hash_build_ms = 0;
    timing.total_ms = timing.data_load_ms + timing.kernel_ms;

    std::cout << "\n=== Q1.1 Results (Random Access Simple) ===" << std::endl;
    std::cout << "Revenue: " << h_revenue << std::endl;
    std::cout << "\nTiming:" << std::endl;
    std::cout << "  Decompress orderdate: " << decompress_ms << " ms" << std::endl;
    std::cout << "  Date filter:          " << filter_ms << " ms" << std::endl;
    std::cout << "  Random access 3 cols: " << random_access_ms << " ms" << std::endl;
    std::cout << "  Aggregation:          " << aggregate_ms << " ms" << std::endl;
    timing.print("Q1.1");

    cudaFree(d_passing_indices); cudaFree(d_num_passing);
    cudaFree(d_discount); cudaFree(d_quantity); cudaFree(d_extendedprice);
    cudaFree(d_revenue);
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    if (argc > 1) data_dir = argv[1];

    std::cout << "=== SSB Q1.1 - Random Access Optimization ===" << std::endl;
    SSBDataCompressedVertical data;
    data.loadAndCompress(data_dir);

    std::cout << "\n--- Two-Stage Random Access ---" << std::endl;
    QueryTiming timing1;
    runQ11RandomAccess(data, timing1);

    std::cout << "\n--- Simple Random Access ---" << std::endl;
    QueryTiming timing2;
    runQ11RandomAccessSimple(data, timing2);

    // Warmup and benchmark runs
    std::cout << "\n=== Benchmark Runs (Simple Version) ===" << std::endl;
    for (int i = 0; i < 3; ++i) {
        QueryTiming t;
        runQ11RandomAccessSimple(data, t);
        std::cout << "Run " << (i+1) << ": " << t.total_ms << " ms\n";
    }

    data.free();
    return 0;
}
