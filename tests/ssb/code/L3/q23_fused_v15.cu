/**
 * SSB Q2.3 - V13 (L3 V4 Vectorized Decoder, 2048 values/partition)
 *
 * Query: SELECT SUM(lo_revenue), d_year, p_brand1
 *        FROM lineorder, date, part, supplier
 *        WHERE lo_orderdate = d_datekey AND lo_partkey = p_partkey
 *          AND lo_suppkey = s_suppkey AND p_brand1 = 'MFGR#2239'
 *          AND s_region = 'EUROPE'
 *        GROUP BY d_year, p_brand1
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
#include "crystal_hash_v7.cuh"
#include "crystal_hash_build_v7.cuh"

using namespace l3_crystal;
using namespace ssb;
using namespace v15;

constexpr int AGG_NUM_YEARS = 7;
constexpr int AGG_NUM_BRANDS = 1000;
constexpr unsigned long long EXPECTED_TOTAL_Q23 = 791955433ULL;

__global__ void q23_fused_kernel_v15(
    const CompressedDataVertical<uint32_t> lo_suppkey,
    const CompressedDataVertical<uint32_t> lo_partkey,
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_revenue,
    int num_partitions, int total_rows,
    const uint32_t* __restrict__ ht_s, int ht_s_len,
    const uint32_t* __restrict__ ht_p, int ht_p_len,
    const uint32_t* __restrict__ ht_d, int ht_d_len,
    unsigned long long* __restrict__ agg_revenue)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_size = lo_orderdate.d_end_indices[partition_idx] - partition_start;

    int num_full_mv = partition_size / V15_MINI_VECTOR_SIZE;
    int tail_start = num_full_mv * V15_MINI_VECTOR_SIZE;
    int tail_size = partition_size - tail_start;

    const uint64_t* ht_p64 = reinterpret_cast<const uint64_t*>(ht_p);
    const uint64_t* ht_d64 = reinterpret_cast<const uint64_t*>(ht_d);

    // ===== Process full mini-vectors =====
    if (num_full_mv > 0) {
        uint32_t vals[V15_VALUES_PER_THREAD];
        uint16_t brand[V15_VALUES_PER_THREAD];
        uint16_t agg_idx[V15_VALUES_PER_THREAD];
        uint64_t valid_mask = 0;

        // 1) Supplier filter
        decode_column_v15_opt<BW_SUPPKEY>(lo_suppkey, partition_idx, lane_id, vals);
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
            if (global_idx < total_rows) {
                if (__ldg(&ht_s[vals[v] % ht_s_len]) != 0) {
                    valid_mask |= (1ull << v);
                }
            }
        }

        if (__any_sync(FULL_MASK, valid_mask != 0)) {
            // 2) Part join + filter, payload: brand
            decode_column_v15_opt<BW_PARTKEY>(lo_partkey, partition_idx, lane_id, vals);
            #pragma unroll
            for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                uint64_t bit = (1ull << v);
                if (valid_mask & bit) {
                    uint64_t kv = __ldg(&ht_p64[vals[v] % ht_p_len]);
                    if (kv == 0) {
                        valid_mask &= ~bit;
                    } else {
                        uint32_t b = static_cast<uint32_t>(kv >> 32);
                        if (b < AGG_NUM_BRANDS) {
                            brand[v] = static_cast<uint16_t>(b);
                        } else {
                            valid_mask &= ~bit;
                        }
                    }
                }
            }

            if (__any_sync(FULL_MASK, valid_mask != 0)) {
                // 3) Date join, payload: year
                decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, vals);
                #pragma unroll
                for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                    uint64_t bit = (1ull << v);
                    if (valid_mask & bit) {
                        int slot = CRYSTAL_HASH(vals[v], ht_d_len, DATE_KEY_MIN);
                        uint64_t kv = __ldg(&ht_d64[slot]);
                        if (kv == 0) {
                            valid_mask &= ~bit;
                        } else {
                            uint32_t yr = static_cast<uint32_t>(kv >> 32);
                            int year_idx = static_cast<int>(yr) - 1992;
                            if ((unsigned)year_idx < (unsigned)AGG_NUM_YEARS) {
                                agg_idx[v] = static_cast<uint16_t>(year_idx * AGG_NUM_BRANDS + brand[v]);
                            } else {
                                valid_mask &= ~bit;
                            }
                        }
                    }
                }

                // 4) Late materialization: decode revenue only for valid rows
                if (__any_sync(FULL_MASK, valid_mask != 0)) {
                    uint32_t base_rev = 0;
                    if (lane_id == 0) {
                        base_rev = static_cast<uint32_t>(lo_revenue.d_model_params[partition_idx * 4]);
                    }
                    base_rev = __shfl_sync(FULL_MASK, base_rev, 0);

                    uint64_t m = valid_mask;
                    while (m) {
                        int v = __ffsll((long long)m) - 1;
                        m &= (m - 1);

                        int value_idx_in_partition = v * V15_WARP_SIZE + lane_id;
                        uint32_t rev = decode_fls_single_transposed_v15<BW_REVENUE>(
                            lo_revenue.d_interleaved_deltas, partition_idx, value_idx_in_partition, base_rev);

                        atomicAdd(&agg_revenue[agg_idx[v]], static_cast<unsigned long long>(rev));
                    }
                }
            }
        }
    }

    // ===== Process tail =====
    for (int t = lane_id; t < tail_size; t += V15_WARP_SIZE) {
        int value_idx = tail_start + t;

        uint32_t suppkey = decode_tail_value_v15_opt<BW_SUPPKEY>(lo_suppkey, partition_idx, value_idx);
        if (__ldg(&ht_s[suppkey % ht_s_len]) == 0) continue;

        uint32_t partkey = decode_tail_value_v15_opt<BW_PARTKEY>(lo_partkey, partition_idx, value_idx);
        uint64_t kv_p = *reinterpret_cast<const uint64_t*>(&ht_p[(partkey % ht_p_len) << 1]);
        if (kv_p == 0) continue;
        uint32_t b = static_cast<uint32_t>(kv_p >> 32);

        uint32_t orderdate = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
        int slot = CRYSTAL_HASH(orderdate, ht_d_len, DATE_KEY_MIN);
        uint64_t kv_d = *reinterpret_cast<const uint64_t*>(&ht_d[slot << 1]);
        if (kv_d == 0) continue;
        uint32_t yr = static_cast<uint32_t>(kv_d >> 32);

        int year_idx = yr - 1992;
        if (year_idx >= 0 && year_idx < AGG_NUM_YEARS && b < AGG_NUM_BRANDS) {
            uint32_t rev = decode_tail_value_v15_opt<BW_REVENUE>(lo_revenue, partition_idx, value_idx);
            atomicAdd(&agg_revenue[year_idx * AGG_NUM_BRANDS + b], static_cast<unsigned long long>(rev));
        }
    }
}

void runQ23FusedV13(SSBDataCompressedVertical& data, float time_h2d) {
    cudaEvent_t ev_start, ev_metadata, ev_ht_build, ev_kernel, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_metadata);
    cudaEventCreate(&ev_ht_build);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = data.lo_orderdate.num_partitions;

    cudaEventRecord(ev_start);
    cudaEventRecord(ev_metadata);

    uint32_t *ht_s, *ht_p, *ht_d;
    int ht_s_len, ht_p_len, ht_d_len;

    buildSupplierHashTable_Region(data.d_s_suppkey, data.d_s_region, S_LEN, &ht_s, &ht_s_len, REGION_EUROPE);
    buildPartHashTable_Brand(data.d_p_partkey, data.d_p_brand1, P_LEN, &ht_p, &ht_p_len, BRAND_MFGR2239);
    buildDateHashTable_All(data.d_d_datekey, data.d_d_year, D_LEN, &ht_d, &ht_d_len);

    cudaEventRecord(ev_ht_build);

    unsigned long long* d_agg;
    cudaMalloc(&d_agg, AGG_NUM_YEARS * AGG_NUM_BRANDS * sizeof(unsigned long long));

    // Warmup
    cudaMemset(d_agg, 0, AGG_NUM_YEARS * AGG_NUM_BRANDS * sizeof(unsigned long long));
    q23_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_suppkey, data.lo_partkey, data.lo_orderdate, data.lo_revenue,
        num_partitions, total_rows, ht_s, ht_s_len, ht_p, ht_p_len, ht_d, ht_d_len, d_agg);
    cudaDeviceSynchronize();

    float time_ht_build;
    cudaEventElapsedTime(&time_ht_build, ev_metadata, ev_ht_build);

    const int RUNS = 5;
    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg, 0, AGG_NUM_YEARS * AGG_NUM_BRANDS * sizeof(unsigned long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cudaEventRecord(ev_metadata);

        q23_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_suppkey, data.lo_partkey, data.lo_orderdate, data.lo_revenue,
            num_partitions, total_rows, ht_s, ht_s_len, ht_p, ht_p_len, ht_d, ht_d_len, d_agg);
        cudaEventRecord(ev_kernel);

        std::vector<unsigned long long> h_agg(AGG_NUM_YEARS * AGG_NUM_BRANDS);
        cudaMemcpy(h_agg.data(), d_agg, h_agg.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);

        unsigned long long total = 0;
        for (auto v : h_agg) total += v;

        float time_metadata, time_kernel, time_d2h;
        cudaEventElapsedTime(&time_metadata, ev_start, ev_metadata);
        cudaEventElapsedTime(&time_kernel, ev_metadata, ev_kernel);
        cudaEventElapsedTime(&time_d2h, ev_kernel, ev_d2h);
        float time_total = time_h2d + time_metadata + time_ht_build + time_kernel + time_d2h;

        std::cout << "{\"query\":23,\"version\":\"v14\",\"run\":" << run
                  << ",\"time_h2d\":" << time_h2d
                  << ",\"time_metadata\":" << time_metadata
                  << ",\"time_ht_build\":" << time_ht_build
                  << ",\"time_kernel\":" << time_kernel
                  << ",\"time_d2h\":" << time_d2h
                  << ",\"time_total\":" << time_total
                  << ",\"result\":" << total
                  << ",\"status\":\"" << (total == EXPECTED_TOTAL_Q23 ? "PASSED" : "FAILED") << "\"}" << std::endl;
    }

    cudaFree(d_agg);
    cudaFree(ht_s);
    cudaFree(ht_p);
    cudaFree(ht_d);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_metadata);
    cudaEventDestroy(ev_ht_build);
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
    v15::padColumnToFixedOffset<BW_SUPPKEY>(data.lo_suppkey);
    v15::padColumnToFixedOffset<BW_PARTKEY>(data.lo_partkey);
    v15::padColumnToFixedOffset<BW_ORDERDATE>(data.lo_orderdate);
    v15::padColumnToFixedOffset<BW_REVENUE>(data.lo_revenue);
    float time_h2d = data.measureH2DTimeQ2();

    runQ23FusedV13(data, time_h2d);
    return 0;
}
