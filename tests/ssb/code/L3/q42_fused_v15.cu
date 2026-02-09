/**
 * SSB Q4.2 - V13 (L3 V4 Vectorized Decoder)
 *
 * SELECT d_year, s_nation, p_category, SUM(lo_revenue - lo_supplycost) as profit
 * FROM lineorder, customer, supplier, part, date
 * WHERE lo_custkey = c_custkey AND lo_suppkey = s_suppkey
 *   AND lo_partkey = p_partkey AND lo_orderdate = d_datekey
 *   AND c_region = 'AMERICA' AND s_region = 'AMERICA'
 *   AND (d_year = 1997 OR d_year = 1998)
 *   AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 * GROUP BY d_year, s_nation, p_category
 *
 * Expected result: 17011866171
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

constexpr int AGG_NUM_YEARS = 2;  // Compress (1997,1998) into 2-year space via modulo
constexpr int AGG_NUM_NATIONS = 25;
constexpr int AGG_NUM_CATEGORIES = 25;
constexpr long long EXPECTED_TOTAL_Q42 = 17011866171LL;

__global__ void q42_fused_kernel_v15(
    const CompressedDataVertical<uint32_t> lo_custkey,
    const CompressedDataVertical<uint32_t> lo_suppkey,
    const CompressedDataVertical<uint32_t> lo_partkey,
    const CompressedDataVertical<uint32_t> lo_orderdate,
    const CompressedDataVertical<uint32_t> lo_revenue,
    const CompressedDataVertical<uint32_t> lo_supplycost,
    int num_partitions, int total_rows,
    const uint32_t* __restrict__ ht_c, int ht_c_len,
    const uint32_t* __restrict__ ht_s, int ht_s_len,
    const uint32_t* __restrict__ ht_p, int ht_p_len,
    const uint32_t* __restrict__ ht_d, int ht_d_len,
    long long* __restrict__ agg_profit)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int lane_id = threadIdx.x;
    constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

    int partition_start = lo_orderdate.d_start_indices[partition_idx];
    int partition_size  = lo_orderdate.d_end_indices[partition_idx] - partition_start;

    int num_full_mv = partition_size / V15_MINI_VECTOR_SIZE;
    int tail_start  = num_full_mv * V15_MINI_VECTOR_SIZE;
    int tail_size   = partition_size - tail_start;

    // 64-bit view for packed (key,payload) tables
    const uint64_t* ht_s64 = reinterpret_cast<const uint64_t*>(ht_s);
    const uint64_t* ht_p64 = reinterpret_cast<const uint64_t*>(ht_p);
    const uint64_t* ht_d64 = reinterpret_cast<const uint64_t*>(ht_d);

    // ===== Process full mini-vectors =====
    if (num_full_mv > 0) {
        uint32_t vals[V15_VALUES_PER_THREAD];
        uint16_t idx[V15_VALUES_PER_THREAD];
        uint64_t valid_mask = 0;

        // 1) Customer filter (c_region = AMERICA)
        decode_column_v15_opt<BW_CUSTKEY>(lo_custkey, partition_idx, lane_id, vals);

        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            int global_idx = partition_start + v * V15_WARP_SIZE + lane_id;
            if (global_idx < total_rows) {
                uint32_t ck = vals[v];
                if (__ldg(&ht_c[ck % ht_c_len]) != 0) {
                    valid_mask |= (1ull << v);
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
                    uint32_t sk = vals[v];
                    uint64_t kv = __ldg(&ht_s64[sk % ht_s_len]);
                    if (kv == 0) {
                        valid_mask &= ~bit;
                    } else {
                        uint32_t sn = static_cast<uint32_t>(kv >> 32);
                        if (sn < AGG_NUM_NATIONS) {
                            idx[v] = static_cast<uint16_t>(sn * AGG_NUM_CATEGORIES);
                        } else {
                            valid_mask &= ~bit;
                        }
                    }
                }
            }

            if (__any_sync(FULL_MASK, valid_mask != 0)) {
                // 3) Part join + filter, payload: p_category
                decode_column_v15_opt<BW_PARTKEY>(lo_partkey, partition_idx, lane_id, vals);

                #pragma unroll
                for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                    uint64_t bit = (1ull << v);
                    if (valid_mask & bit) {
                        uint32_t pk = vals[v];
                        uint64_t kv = __ldg(&ht_p64[pk % ht_p_len]);
                        if (kv == 0) {
                            valid_mask &= ~bit;
                        } else {
                            uint32_t pc = static_cast<uint32_t>(kv >> 32);
                            if (pc < AGG_NUM_CATEGORIES) {
                                idx[v] = static_cast<uint16_t>(idx[v] + pc);
                            } else {
                                valid_mask &= ~bit;
                            }
                        }
                    }
                }

                if (__any_sync(FULL_MASK, valid_mask != 0)) {
                    // 4) Date join + filter, payload: year
                    decode_column_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, lane_id, vals);

                    #pragma unroll
                    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
                        uint64_t bit = (1ull << v);
                        if (valid_mask & bit) {
                            uint32_t od = vals[v];
                            int slot = CRYSTAL_HASH(od, ht_d_len, DATE_KEY_MIN);
                            uint64_t kv = __ldg(&ht_d64[slot]);
                            if (kv == 0) {
                                valid_mask &= ~bit;
                            } else {
                                uint32_t yr = static_cast<uint32_t>(kv >> 32);
                                int year_mod = (static_cast<int>(yr) - 1992) & 1;  // 1998->0, 1997->1
                                idx[v] = static_cast<uint16_t>(
                                    year_mod * (AGG_NUM_NATIONS * AGG_NUM_CATEGORIES) + idx[v]
                                );
                            }
                        }
                    }

                    // 5) Late materialization: decode revenue/supplycost only for valid rows
                    if (__any_sync(FULL_MASK, valid_mask != 0)) {
                        // Broadcast base values
                        uint32_t base_rev = 0, base_sc = 0;
                        if (lane_id == 0) {
                            base_rev = static_cast<uint32_t>(lo_revenue.d_model_params[partition_idx * 4]);
                            base_sc  = static_cast<uint32_t>(lo_supplycost.d_model_params[partition_idx * 4]);
                        }
                        base_rev = __shfl_sync(FULL_MASK, base_rev, 0);
                        base_sc  = __shfl_sync(FULL_MASK, base_sc, 0);

                        uint64_t m = valid_mask;
                        while (m) {
                            int v = __ffsll((long long)m) - 1;
                            m &= (m - 1);

                            int value_idx_in_partition = v * V15_WARP_SIZE + lane_id;

                            // Use TRANSPOSED decode for values within full mini-vector
                            uint32_t rev = decode_fls_single_transposed_v15<BW_REVENUE>(
                                lo_revenue.d_interleaved_deltas, partition_idx, value_idx_in_partition, base_rev);

                            uint32_t sc = decode_fls_single_transposed_v15<BW_SUPPLYCOST>(
                                lo_supplycost.d_interleaved_deltas, partition_idx, value_idx_in_partition, base_sc);

                            long long profit = static_cast<long long>(rev) - static_cast<long long>(sc);
                            int agg_i = static_cast<int>(idx[v]);

                            atomicAdd(reinterpret_cast<unsigned long long*>(&agg_profit[agg_i]),
                                      static_cast<unsigned long long>(profit));
                        }
                    }
                }
            }
        }
    }

    // ===== Process tail (scalar, uses sequential decode) =====
    for (int t = lane_id; t < tail_size; t += V15_WARP_SIZE) {
        int value_idx = tail_start + t;

        uint32_t ck = decode_tail_value_v15_opt<BW_CUSTKEY>(lo_custkey, partition_idx, value_idx);
        if (__ldg(&ht_c[ck % ht_c_len]) == 0) continue;

        uint32_t sk = decode_tail_value_v15_opt<BW_SUPPKEY>(lo_suppkey, partition_idx, value_idx);
        uint64_t kv_s = *reinterpret_cast<const uint64_t*>(&ht_s[(sk % ht_s_len) << 1]);
        if (kv_s == 0) continue;
        uint32_t sn = static_cast<uint32_t>(kv_s >> 32);

        uint32_t pk = decode_tail_value_v15_opt<BW_PARTKEY>(lo_partkey, partition_idx, value_idx);
        uint64_t kv_p = *reinterpret_cast<const uint64_t*>(&ht_p[(pk % ht_p_len) << 1]);
        if (kv_p == 0) continue;
        uint32_t pc = static_cast<uint32_t>(kv_p >> 32);

        uint32_t od = decode_tail_value_v15_opt<BW_ORDERDATE>(lo_orderdate, partition_idx, value_idx);
        int slot = CRYSTAL_HASH(od, ht_d_len, DATE_KEY_MIN);
        uint64_t kv_d = *reinterpret_cast<const uint64_t*>(&ht_d[slot << 1]);
        if (kv_d == 0) continue;
        uint32_t yr = static_cast<uint32_t>(kv_d >> 32);

        uint32_t rev = decode_tail_value_v15_opt<BW_REVENUE>(lo_revenue, partition_idx, value_idx);
        uint32_t sc  = decode_tail_value_v15_opt<BW_SUPPLYCOST>(lo_supplycost, partition_idx, value_idx);

        if (sn < AGG_NUM_NATIONS && pc < AGG_NUM_CATEGORIES) {
            int year_mod = (static_cast<int>(yr) - 1992) & 1;  // 1998->0, 1997->1
            long long profit = static_cast<long long>(rev) - static_cast<long long>(sc);
            int agg_idx = year_mod * AGG_NUM_NATIONS * AGG_NUM_CATEGORIES +
                          sn * AGG_NUM_CATEGORIES + pc;
            atomicAdd(reinterpret_cast<unsigned long long*>(&agg_profit[agg_idx]),
                      static_cast<unsigned long long>(profit));
        }
    }
}

void runQ42FusedV13(SSBDataCompressedVertical& data, float time_h2d) {
    cudaEvent_t ev_start, ev_metadata, ev_ht_build, ev_kernel, ev_d2h;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_metadata);
    cudaEventCreate(&ev_ht_build);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);

    int total_rows = data.lo_orderdate.total_values;
    int num_partitions = data.lo_orderdate.num_partitions;

    // ===== Stage 1: Metadata =====
    cudaEventRecord(ev_start);
    cudaEventRecord(ev_metadata);

    // ===== Stage 2: Hash table build =====
    uint32_t *ht_c, *ht_s, *ht_p, *ht_d;
    int ht_c_len, ht_s_len, ht_p_len, ht_d_len;

    buildCustomerHashTable_RegionOnly(data.d_c_custkey, data.d_c_region, C_LEN,
                                      &ht_c, &ht_c_len, REGION_AMERICA);
    buildSupplierHashTable_RegionNation(data.d_s_suppkey, data.d_s_region, data.d_s_nation, S_LEN,
                                        &ht_s, &ht_s_len, REGION_AMERICA);
    buildPartHashTable_MfgrCategory(data.d_p_partkey, data.d_p_mfgr, data.d_p_category, P_LEN,
                                    &ht_p, &ht_p_len, 1, 2);
    buildDateHashTable_YearRange(data.d_d_datekey, data.d_d_year, D_LEN, &ht_d, &ht_d_len, 1997, 1998);

    cudaEventRecord(ev_ht_build);

    int agg_size = AGG_NUM_YEARS * AGG_NUM_NATIONS * AGG_NUM_CATEGORIES;
    long long* d_agg;
    cudaMalloc(&d_agg, agg_size * sizeof(long long));

    // ===== Warmup =====
    cudaMemset(d_agg, 0, agg_size * sizeof(long long));
    q42_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
        data.lo_custkey, data.lo_suppkey, data.lo_partkey, data.lo_orderdate,
        data.lo_revenue, data.lo_supplycost,
        num_partitions, total_rows, ht_c, ht_c_len, ht_s, ht_s_len, ht_p, ht_p_len, ht_d, ht_d_len, d_agg);
    cudaDeviceSynchronize();

    // Get hash table build time
    float time_ht_build;
    cudaEventElapsedTime(&time_ht_build, ev_metadata, ev_ht_build);

    // ===== Benchmark runs =====
    const int RUNS = 5;

    for (int run = 0; run < RUNS; run++) {
        cudaMemset(d_agg, 0, agg_size * sizeof(long long));
        cudaDeviceSynchronize();

        cudaEventRecord(ev_start);
        cudaEventRecord(ev_metadata);

        q42_fused_kernel_v15<<<num_partitions, V15_THREADS_PER_BLOCK>>>(
            data.lo_custkey, data.lo_suppkey, data.lo_partkey, data.lo_orderdate,
            data.lo_revenue, data.lo_supplycost,
            num_partitions, total_rows, ht_c, ht_c_len, ht_s, ht_s_len, ht_p, ht_p_len, ht_d, ht_d_len, d_agg);
        cudaEventRecord(ev_kernel);

        std::vector<long long> h_agg(agg_size);
        cudaMemcpy(h_agg.data(), d_agg, h_agg.size() * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);

        long long total_profit = 0;
        for (auto v : h_agg) total_profit += v;

        float time_metadata, time_kernel, time_d2h;
        cudaEventElapsedTime(&time_metadata, ev_start, ev_metadata);
        cudaEventElapsedTime(&time_kernel, ev_metadata, ev_kernel);
        cudaEventElapsedTime(&time_d2h, ev_kernel, ev_d2h);
        float time_total = time_h2d + time_metadata + time_ht_build + time_kernel + time_d2h;

        std::cout << "{\"query\":42,\"version\":\"v14\",\"run\":" << run
                  << ",\"time_h2d\":" << time_h2d
                  << ",\"time_metadata\":" << time_metadata
                  << ",\"time_ht_build\":" << time_ht_build
                  << ",\"time_kernel\":" << time_kernel
                  << ",\"time_d2h\":" << time_d2h
                  << ",\"time_total\":" << time_total
                  << ",\"result\":" << total_profit
                  << ",\"status\":\"" << (total_profit == EXPECTED_TOTAL_Q42 ? "PASSED" : "FAILED") << "\"}" << std::endl;
    }

    cudaFree(d_agg);
    cudaFree(ht_c);
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
    v15::padColumnToFixedOffset<BW_CUSTKEY>(data.lo_custkey);
    v15::padColumnToFixedOffset<BW_SUPPKEY>(data.lo_suppkey);
    v15::padColumnToFixedOffset<BW_PARTKEY>(data.lo_partkey);
    v15::padColumnToFixedOffset<BW_ORDERDATE>(data.lo_orderdate);
    v15::padColumnToFixedOffset<BW_REVENUE>(data.lo_revenue);
    v15::padColumnToFixedOffset<BW_SUPPLYCOST>(data.lo_supplycost);
    float time_h2d = data.measureH2DTimeQ4();

    runQ42FusedV13(data, time_h2d);
    return 0;
}
