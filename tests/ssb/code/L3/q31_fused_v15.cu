/**
 * SSB Q3.1 - V13 (L3 V4 Vectorized Decoder)
 *
 * Query: SELECT c_nation, s_nation, d_year, SUM(lo_revenue)
 *        FROM lineorder, customer, supplier, date
 *        WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey AND lo_orderdate = d_datekey
 *          AND c_region = 'ASIA' AND s_region = 'ASIA' AND d_year >= 1992 AND d_year <= 1997
 *        GROUP BY c_nation, s_nation, d_year
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

constexpr int AGG_NUM_YEARS = 6;
constexpr int AGG_NUM_NATIONS = 25;
constexpr unsigned long long EXPECTED_TOTAL_Q31 = 134114571007ULL;

__global__ void q31_fused_kernel_v15(
    const CompressedDataVertical<uint32_t> lo_custkey,
    const CompressedDataVertical<uint32_t> lo_suppkey,
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_revenue,
    int num_partitions, int total_rows,
    const uint32_t* __restrict__ ht_c, int ht_c_len,
    const uint32_t* __restrict__ ht_s, int ht_s_len,
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

    const uint64_t* ht_c64 = reinterpret_cast<const uint64_t*>(ht_c);
    const uint64_t* ht_s64 = reinterpret_cast<const uint64_t*>(ht_s);
    const uint64_t* ht_d64 = reinterpret_cast<const uint64_t*>(ht_d);

    // ===== Process full mini-vectors =====
    if (num_full_mv > 0) {
        uint32_t vals[V15_VALUES_PER_THREAD];
        uint16_t c_nation[V15_VALUES_PER_THREAD];
        uint16_t agg_idx[V15_VALUES_PER_THREAD];
        uint64_t valid_mask = 0;

        // 1) Customer join + filter, payload: c_nation
        decode_column_v15_opt<BW_CUSTKEY>(lo_custkey, partition_idx, lane_id, vals);
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
            if (global_idx < total_rows) {
                uint64_t kv = __ldg(&ht_c64[vals[v] % ht_c_len]);
                if (kv != 0) {
                    uint32_t cn = static_cast<uint32_t>(kv >> 32);
                    if (cn < AGG_NUM_NATIONS) {
                        c_nation[v] = static_cast<uint16_t>(cn);
                        valid_mask |= (1ull << v);
                    }
                }
            }
        }

        if (__any_sync(FULL_MASK, valid_mask != 0)) {
            // 2) Supplier join + filter, payload: s_nation
            decode_column_v15_opt<BW_SUPPKEY>(lo_suppkey, partition_idx, lane_id, vals);
            #pragma unroll
            for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                uint64_t bit = (1ull << v);
                if (valid_mask & bit) {
                    uint64_t kv = __ldg(&ht_s64[vals[v] % ht_s_len]);
                    if (kv == 0) {
                        valid_mask &= ~bit;
                    } else {
                        uint32_t sn = static_cast<uint32_t>(kv >> 32);
                        if (sn < AGG_NUM_NATIONS) {
                            // Store partial index: c_nation * 25 + s_nation
                            agg_idx[v] = static_cast<uint16_t>(c_nation[v] * AGG_NUM_NATIONS + sn);
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
                                agg_idx[v] = static_cast<uint16_t>(year_idx * AGG_NUM_NATIONS * AGG_NUM_NATIONS + agg_idx[v]);
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

        uint32_t custkey = decode_tail_value_v15_opt<BW_CUSTKEY>(lo_custkey, partition_idx, value_idx);
        uint64_t kv_c = *reinterpret_cast<const uint64_t*>(&ht_c[(custkey % ht_c_len) << 1]);
        if (kv_c == 0) continue;
        uint32_t cn = static_cast<uint32_t>(kv_c >> 32);

        uint32_t suppkey = decode_tail_value_v15_opt<BW_SUPPKEY>(lo_suppkey, partition_idx, value_idx);
        uint64_t kv_s = *reinterpret_cast<const uint64_t*>(&ht_s[(suppkey % ht_s_len) << 1]);
        if (kv_s == 0) continue;
        uint32_t sn = static_cast<uint32_t>(kv_s >> 32);

        uint32_t orderdate = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
        int slot = CRYSTAL_HASH(orderdate, ht_d_len, DATE_KEY_MIN);
        uint64_t kv_d = *reinterpret_cast<const uint64_t*>(&ht_d[slot << 1]);
        if (kv_d == 0) continue;
        uint32_t yr = static_cast<uint32_t>(kv_d >> 32);

        int year_idx = yr - 1992;
        if (year_idx >= 0 && year_idx < AGG_NUM_YEARS &&
            cn < AGG_NUM_NATIONS && sn < AGG_NUM_NATIONS) {
            uint32_t rev = decode_tail_value_v15_opt<BW_REVENUE>(lo_revenue, partition_idx, value_idx);
            int idx = year_idx * AGG_NUM_NATIONS * AGG_NUM_NATIONS + cn * AGG_NUM_NATIONS + sn;
            atomicAdd(&agg_revenue[idx], static_cast<unsigned long long>(rev));
        }
    }
}

void runQ31FusedV13(SSBDataCompressedVertical& data, float time_h2d) {
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

    uint32_t *ht_c, *ht_s, *ht_d;
    int ht_c_len, ht_s_len, ht_d_len;

    buildCustomerHashTable_Region(data.d_c_custkey, data.d_c_region, data.d_c_nation, C_LEN, &ht_c, &ht_c_len, REGION_ASIA);
    buildSupplierHashTable_RegionNation(data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN, &ht_s, &ht_s_len, REGION_ASIA);
    buildDateHashTable_YearRange(data.d_d_datekey, data.d_d_year, D_LEN, &ht_d, &ht_d_len, 1992, 1997);

    cudaEventRecord(ev_ht_build);

    int agg_size = AGG_NUM_YEARS * AGG_NUM_NATIONS * AGG_NUM_NATIONS;
    unsigned long long* d_agg;
    cudaMalloc(&d_agg, agg_size * sizeof(unsigned long long));

    // Warmup
    cudaMemset(d_agg, 0, agg_size * sizeof(unsigned long long));
    q31_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_custkey, data.lo_suppkey, data.lo_orderdate, data.lo_revenue,
        num_partitions, total_rows, ht_c, ht_c_len, ht_s, ht_s_len, ht_d, ht_d_len, d_agg);
    cudaDeviceSynchronize();

    float time_ht_build;
    cudaEventElapsedTime(&time_ht_build, ev_metadata, ev_ht_build);

    const int RUNS = 5;
    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg, 0, agg_size * sizeof(unsigned long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cudaEventRecord(ev_metadata);

        q31_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_custkey, data.lo_suppkey, data.lo_orderdate, data.lo_revenue,
            num_partitions, total_rows, ht_c, ht_c_len, ht_s, ht_s_len, ht_d, ht_d_len, d_agg);
        cudaEventRecord(ev_kernel);

        std::vector<unsigned long long> h_agg(agg_size);
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

        std::cout << "{\"query\":31,\"version\":\"v14\",\"run\":" << run
                  << ",\"time_h2d\":" << time_h2d
                  << ",\"time_metadata\":" << time_metadata
                  << ",\"time_ht_build\":" << time_ht_build
                  << ",\"time_kernel\":" << time_kernel
                  << ",\"time_d2h\":" << time_d2h
                  << ",\"time_total\":" << time_total
                  << ",\"result\":" << total
                  << ",\"status\":\"" << (total == EXPECTED_TOTAL_Q31 ? "PASSED" : "FAILED") << "\"}" << std::endl;
    }

    cudaFree(d_agg);
    cudaFree(ht_c);
    cudaFree(ht_s);
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
    v15::padColumnToFixedOffset<BW_CUSTKEY>(data.lo_custkey);
    v15::padColumnToFixedOffset<BW_SUPPKEY>(data.lo_suppkey);
    v15::padColumnToFixedOffset<BW_ORDERDATE>(data.lo_orderdate);
    v15::padColumnToFixedOffset<BW_REVENUE>(data.lo_revenue);
    float time_h2d = data.measureH2DTimeQ3();

    runQ31FusedV13(data, time_h2d);
    return 0;
}
