#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "common/cuda_utils.hpp"
#include "common/device_buffer.hpp"
#include "common/timing.hpp"
#include "planner/decompress.cuh"
#include "planner/encoded_column.hpp"
#include "planner/fused_decode.cuh"
#include "ssb/crystal_like.cuh"
#include "ssb/queries.hpp"
#include "ssb/ssb_dataset.hpp"

namespace ssb {
namespace {

constexpr int kDateMin = 19920101;
constexpr int kDateMax = 19981231;

constexpr int kYearMin = 1992;
constexpr int kYearMaxQ31 = 1997;
constexpr int kYearMaxQ32 = 1997;

constexpr int kNationUS = 24;

constexpr int kNumNations = 25;
constexpr int kNumCities = 250;

constexpr int kCityUK1 = 249;
constexpr int kCityUK5 = 244;

const int* prepare_column_for_fused(
    const planner::EncodedColumn& enc,
    planner::DeviceBuffer<int>& decode_buf,
    planner::DeltaWorkspace& delta_ws,
    planner::RleWorkspace& rle_ws,
    cudaStream_t stream = 0)
{
    switch (enc.scheme) {
        case planner::Scheme::Uncompressed:
            return enc.d_ints_ptr();
        case planner::Scheme::NS:
        case planner::Scheme::FOR_NS:
            return nullptr;
        case planner::Scheme::DELTA_NS:
            decode_buf.resize(enc.n);
            planner::delta_ns_decode_into(enc, decode_buf.data(), delta_ws, stream);
            return decode_buf.data();
        case planner::Scheme::DELTA_FOR_NS:
            decode_buf.resize(enc.n);
            planner::delta_for_ns_decode_into(enc, decode_buf.data(), delta_ws, stream);
            return decode_buf.data();
        case planner::Scheme::RLE:
            decode_buf.resize(enc.n);
            planner::rle_decode_into(enc, decode_buf.data(), rle_ws, stream);
            return decode_buf.data();
        default:
            return nullptr;
    }
}

planner::fused::FusedColumnAccessor make_accessor(
    const planner::EncodedColumn& enc,
    const int* pre_decoded)
{
    planner::fused::FusedColumnAccessor acc;
    acc.scheme = enc.scheme;
    acc.d_bytes = enc.d_bytes_ptr();
    acc.d_ints = enc.d_ints_ptr();
    acc.d_decoded = pre_decoded;
    acc.byte_width = enc.byte_width;
    acc.base = enc.base;
    acc.first = enc.first;
    acc.delta_base = enc.delta_base;
    acc.n = enc.n;
    return acc;
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_region_to_nation(const int* s_region,
                                                   const int* s_suppkey,
                                                   const int* s_nation,
                                                   int s_len,
                                                   int region_filter,
                                                   int* ht_s,
                                                   int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_region + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, items, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_nation + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, vals, selection_flags, ht_s, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_region_to_nation(const int* c_region,
                                                   const int* c_custkey,
                                                   const int* c_nation,
                                                   int c_len,
                                                   int region_filter,
                                                   int* ht_c,
                                                   int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_region + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, items, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_nation + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, vals, selection_flags, ht_c, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_nation_to_city(const int* s_nation,
                                                 const int* s_suppkey,
                                                 const int* s_city,
                                                 int s_len,
                                                 int nation_filter,
                                                 int* ht_s,
                                                 int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_nation + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, nation_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, items, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_city + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, vals, selection_flags, ht_s, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_nation_to_city(const int* c_nation,
                                                 const int* c_custkey,
                                                 const int* c_city,
                                                 int c_len,
                                                 int nation_filter,
                                                 int* ht_c,
                                                 int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_nation + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, nation_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, items, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_city + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, vals, selection_flags, ht_c, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_city_in_set(const int* s_city, const int* s_suppkey, int s_len, int* ht_s, int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int city[ITEMS_PER_THREAD];
    int key[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_city + tile_offset, city, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(city, kCityUK1, selection_flags, num_items);
    BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(city, kCityUK5, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, key, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, city, selection_flags, ht_s, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_city_in_set(const int* c_city, const int* c_custkey, int c_len, int* ht_c, int ht_len) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int city[ITEMS_PER_THREAD];
    int key[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_city + tile_offset, city, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(city, kCityUK1, selection_flags, num_items);
    BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(city, kCityUK5, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, key, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, city, selection_flags, ht_c, ht_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d_year_range(const int* d_datekey,
                                             const int* d_year,
                                             int d_len,
                                             int year_min,
                                             int year_max,
                                             int* ht_d,
                                             int ht_len,
                                             int keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int year[ITEMS_PER_THREAD];
    int key[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, year, num_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(year, year_min, selection_flags, num_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(year, year_max, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, key, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, year, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d_datekey_range(const int* d_datekey,
                                                const int* d_year,
                                                int d_len,
                                                int datekey_min,
                                                int datekey_max,
                                                int* ht_d,
                                                int ht_len,
                                                int keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int key[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, key, num_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, datekey_min, selection_flags, num_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, datekey_max, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, year, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(key, year, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q31(const int* lo_orderdate,
                          const int* lo_custkey,
                          const int* lo_suppkey,
                          const int* lo_revenue,
                          int lo_len,
                          const int* ht_s,
                          int s_ht_len,
                          const int* ht_c,
                          int c_ht_len,
                          const int* ht_d,
                          int d_ht_len,
                          int d_min,
                          int* res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int items[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int s_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_custkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_nation, selection_flags, ht_c, c_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_suppkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_nation, selection_flags, ht_s, s_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_orderdate[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue[row];
            const int hash =
                (s_nation[ITEM] * kNumNations * 7 + c_nation[ITEM] * 7 + (year[ITEM] - 1992)) % (7 * kNumNations * kNumNations);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = c_nation[ITEM];
            res[hash * 6 + 2] = s_nation[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q32(const int* lo_orderdate,
                          const int* lo_custkey,
                          const int* lo_suppkey,
                          const int* lo_revenue,
                          int lo_len,
                          const int* ht_s,
                          int s_ht_len,
                          const int* ht_c,
                          int c_ht_len,
                          const int* ht_d,
                          int d_ht_len,
                          int d_min,
                          int* res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int items[ITEMS_PER_THREAD];
    int c_city[ITEMS_PER_THREAD];
    int s_city[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_suppkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_city, selection_flags, ht_s, s_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_custkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_city, selection_flags, ht_c, c_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_orderdate[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue[row];
            const int hash = (s_city[ITEM] * kNumCities * 6 + c_city[ITEM] * 6 + (year[ITEM] - 1992)) % (6 * kNumCities * kNumCities);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = c_city[ITEM];
            res[hash * 6 + 2] = s_city[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q31_fused(planner::fused::FusedColumnAccessor lo_orderdate,
                                planner::fused::FusedColumnAccessor lo_custkey,
                                planner::fused::FusedColumnAccessor lo_suppkey,
                                planner::fused::FusedColumnAccessor lo_revenue,
                                int lo_len,
                                const int* ht_s,
                                int s_ht_len,
                                const int* ht_c,
                                int c_ht_len,
                                const int* ht_d,
                                int d_ht_len,
                                int d_min,
                                int* res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int items[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int s_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    __shared__ int shared_any[32];
    __shared__ int block_has_valid;

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);
    if (num_items <= 0) return;

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_custkey.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_nation, selection_flags, ht_c, c_ht_len, num_items);

    int local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    int any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_suppkey.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_nation, selection_flags, ht_s, s_ht_len, num_items);

    local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_orderdate.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

    local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue.decode(row);
            const int hash =
                (s_nation[ITEM] * kNumNations * 7 + c_nation[ITEM] * 7 + (year[ITEM] - 1992)) % (7 * kNumNations * kNumNations);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = c_nation[ITEM];
            res[hash * 6 + 2] = s_nation[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q32_fused(planner::fused::FusedColumnAccessor lo_orderdate,
                                planner::fused::FusedColumnAccessor lo_custkey,
                                planner::fused::FusedColumnAccessor lo_suppkey,
                                planner::fused::FusedColumnAccessor lo_revenue,
                                int lo_len,
                                const int* ht_s,
                                int s_ht_len,
                                const int* ht_c,
                                int c_ht_len,
                                const int* ht_d,
                                int d_ht_len,
                                int d_min,
                                int* res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int items[ITEMS_PER_THREAD];
    int c_city[ITEMS_PER_THREAD];
    int s_city[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];
    __shared__ int shared_any[32];
    __shared__ int block_has_valid;

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);
    if (num_items <= 0) return;

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_suppkey.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, s_city, selection_flags, ht_s, s_ht_len, num_items);

    int local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    int any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_custkey.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, c_city, selection_flags, ht_c, c_ht_len, num_items);

    local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        items[ITEM] = (row < lo_len) ? lo_orderdate.decode(row) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

    local_any = 0;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) local_any |= selection_flags[ITEM];
    }
    any = BlockSum<int, BLOCK_THREADS, ITEMS_PER_THREAD>(local_any, shared_any);
    if (threadIdx.x == 0) block_has_valid = (any != 0);
    __syncthreads();
    if (!block_has_valid) return;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue.decode(row);
            const int hash = (s_city[ITEM] * kNumCities * 6 + c_city[ITEM] * 6 + (year[ITEM] - 1992)) % (6 * kNumCities * kNumCities);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = c_city[ITEM];
            res[hash * 6 + 2] = s_city[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

struct Q3EncodedLO {
    planner::EncodedColumn orderdate;
    planner::EncodedColumn custkey;
    planner::EncodedColumn suppkey;
    planner::EncodedColumn revenue;
};

Q3EncodedLO load_and_encode_lo_q3(const RunOptions& opt) {
    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_custkey = load_column<int>(opt.data_dir, "lo_custkey", LO_LEN);
    auto h_lo_suppkey = load_column<int>(opt.data_dir, "lo_suppkey", LO_LEN);
    auto h_lo_revenue = load_column<int>(opt.data_dir, "lo_revenue", LO_LEN);

    Q3EncodedLO cols;
    cols.orderdate = planner::encode_column_planner(h_lo_orderdate.get(), LO_LEN);
    cols.custkey = planner::encode_column_planner(h_lo_custkey.get(), LO_LEN);
    cols.suppkey = planner::encode_column_planner(h_lo_suppkey.get(), LO_LEN);
    cols.revenue = planner::encode_column_planner(h_lo_revenue.get(), LO_LEN);

    cols.orderdate.upload(h_lo_orderdate.get());
    cols.custkey.upload(h_lo_custkey.get());
    cols.suppkey.upload(h_lo_suppkey.get());
    cols.revenue.upload(h_lo_revenue.get());
    return cols;
}

size_t get_compressed_size_q3(const planner::EncodedColumn& enc) {
    if (enc.scheme == planner::Scheme::Uncompressed) return enc.n * sizeof(int);
    if (enc.scheme == planner::Scheme::RLE) return enc.rle_values.size() * sizeof(int) * 2;
    return enc.bytes.size();
}

} // namespace

void run_q31(const RunOptions& opt) {
    std::cout << "=== Q31 (Planner - Fused Decompression) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_region = load_column<int>(opt.data_dir, "s_region", S_LEN);
    auto h_s_nation = load_column<int>(opt.data_dir, "s_nation", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_region = load_column<int>(opt.data_dir, "c_region", C_LEN);
    auto h_c_nation = load_column<int>(opt.data_dir, "c_nation", C_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_region(S_LEN);
    planner::DeviceBuffer<int> d_s_nation(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_region(C_LEN);
    planner::DeviceBuffer<int> d_c_nation(C_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_region.data(), h_s_region.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_region.data(), h_c_region.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

    build_hashtable_s_region_to_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_region.data(), d_s_suppkey.data(), d_s_nation.data(), static_cast<int>(S_LEN), 2, ht_s.data(), static_cast<int>(S_LEN));

    build_hashtable_c_region_to_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_region.data(), d_c_custkey.data(), d_c_nation.data(), static_cast<int>(C_LEN), 2, ht_c.data(), static_cast<int>(C_LEN));

    build_hashtable_d_year_range<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), kYearMin, kYearMaxQ31, ht_d.data(), date_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 7 * kNumNations * kNumNations;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate_dec;
    planner::DeviceBuffer<int> d_lo_custkey_dec;
    planner::DeviceBuffer<int> d_lo_suppkey_dec;
    planner::DeviceBuffer<int> d_lo_revenue_dec;
    planner::DeltaWorkspace ws_orderdate, ws_custkey, ws_suppkey, ws_revenue;
    planner::RleWorkspace ws_orderdate_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle;

    // Warmup (fused decode + probe, not reported).
    {
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q31_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate,
                acc_custkey,
                acc_suppkey,
                acc_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 压缩率统计
    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = get_compressed_size_q3(cols.orderdate) + get_compressed_size_q3(cols.custkey) +
                             get_compressed_size_q3(cols.suppkey) + get_compressed_size_q3(cols.revenue);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x (" << (original_size/1e6) << " MB -> " << (compressed_size/1e6) << " MB)" << std::endl;

    planner::CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        probe_q31_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate,
                acc_custkey,
                acc_suppkey,
                acc_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0];
            }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/Planner/q3_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) csv << "query,run,rows,revenue,time_decomp_query_ms,throughput_gbps,compression_ratio\n";
        csv << "31," << run << "," << rows << "," << total_revenue << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

void run_q32(const RunOptions& opt) {
    std::cout << "=== Q32 (Planner - Fused Decompression) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_nation = load_column<int>(opt.data_dir, "s_nation", S_LEN);
    auto h_s_city = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_nation = load_column<int>(opt.data_dir, "c_nation", C_LEN);
    auto h_c_city = load_column<int>(opt.data_dir, "c_city", C_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_nation(S_LEN);
    planner::DeviceBuffer<int> d_s_city(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_nation(C_LEN);
    planner::DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

    build_hashtable_s_nation_to_city<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_nation.data(), d_s_suppkey.data(), d_s_city.data(), static_cast<int>(S_LEN), kNationUS, ht_s.data(), static_cast<int>(S_LEN));

    build_hashtable_c_nation_to_city<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_nation.data(), d_c_custkey.data(), d_c_city.data(), static_cast<int>(C_LEN), kNationUS, ht_c.data(), static_cast<int>(C_LEN));

    build_hashtable_d_year_range<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), kYearMin, kYearMaxQ32, ht_d.data(), date_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 6 * kNumCities * kNumCities;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate_dec;
    planner::DeviceBuffer<int> d_lo_custkey_dec;
    planner::DeviceBuffer<int> d_lo_suppkey_dec;
    planner::DeviceBuffer<int> d_lo_revenue_dec;
    planner::DeltaWorkspace ws_orderdate, ws_custkey, ws_suppkey, ws_revenue;
    planner::RleWorkspace ws_orderdate_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle;

    // Warmup.
    {
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate,
                acc_custkey,
                acc_suppkey,
                acc_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = get_compressed_size_q3(cols.orderdate) + get_compressed_size_q3(cols.custkey) +
                             get_compressed_size_q3(cols.suppkey) + get_compressed_size_q3(cols.revenue);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    planner::CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate, acc_custkey, acc_suppkey, acc_revenue, static_cast<int>(LO_LEN),
                ht_s.data(), static_cast<int>(S_LEN), ht_c.data(), static_cast<int>(C_LEN),
                ht_d.data(), date_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) { ++rows; total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/Planner/q3_results.csv";
        std::ofstream csv(csv_path, std::ios::app);
        csv << "32," << run << "," << rows << "," << total_revenue << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

void run_q33(const RunOptions& opt) {
    std::cout << "=== Q33 (Planner - Fused Decompression) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_city = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_city = load_column<int>(opt.data_dir, "c_city", C_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_city(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

    build_hashtable_s_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_city.data(), d_s_suppkey.data(), static_cast<int>(S_LEN), ht_s.data(), static_cast<int>(S_LEN));

    build_hashtable_c_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_city.data(), d_c_custkey.data(), static_cast<int>(C_LEN), ht_c.data(), static_cast<int>(C_LEN));

    build_hashtable_d_year_range<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), kYearMin, kYearMaxQ32, ht_d.data(), date_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 6 * kNumCities * kNumCities;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate_dec;
    planner::DeviceBuffer<int> d_lo_custkey_dec;
    planner::DeviceBuffer<int> d_lo_suppkey_dec;
    planner::DeviceBuffer<int> d_lo_revenue_dec;
    planner::DeltaWorkspace ws_orderdate, ws_custkey, ws_suppkey, ws_revenue;
    planner::RleWorkspace ws_orderdate_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle;

    // Warmup.
    {
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate,
                acc_custkey,
                acc_suppkey,
                acc_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = get_compressed_size_q3(cols.orderdate) + get_compressed_size_q3(cols.custkey) +
                             get_compressed_size_q3(cols.suppkey) + get_compressed_size_q3(cols.revenue);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    planner::CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate, acc_custkey, acc_suppkey, acc_revenue, static_cast<int>(LO_LEN),
                ht_s.data(), static_cast<int>(S_LEN), ht_c.data(), static_cast<int>(C_LEN),
                ht_d.data(), date_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) { ++rows; total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::ofstream csv("/root/autodl-tmp/code/L3/H20/ssb/Planner/q3_results.csv", std::ios::app);
        csv << "33," << run << "," << rows << "," << total_revenue << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

void run_q34(const RunOptions& opt) {
    std::cout << "=== Q34 (Planner - Fused Decompression) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_city = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_city = load_column<int>(opt.data_dir, "c_city", C_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_city(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

    build_hashtable_s_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_city.data(), d_s_suppkey.data(), static_cast<int>(S_LEN), ht_s.data(), static_cast<int>(S_LEN));

    build_hashtable_c_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_city.data(), d_c_custkey.data(), static_cast<int>(C_LEN), ht_c.data(), static_cast<int>(C_LEN));

    build_hashtable_d_datekey_range<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), 19971201, 19971231, ht_d.data(), date_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 6 * kNumCities * kNumCities;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate_dec;
    planner::DeviceBuffer<int> d_lo_custkey_dec;
    planner::DeviceBuffer<int> d_lo_suppkey_dec;
    planner::DeviceBuffer<int> d_lo_revenue_dec;
    planner::DeltaWorkspace ws_orderdate, ws_custkey, ws_suppkey, ws_revenue;
    planner::RleWorkspace ws_orderdate_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle;

    // Warmup.
    {
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate,
                acc_custkey,
                acc_suppkey,
                acc_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = get_compressed_size_q3(cols.orderdate) + get_compressed_size_q3(cols.custkey) +
                             get_compressed_size_q3(cols.suppkey) + get_compressed_size_q3(cols.revenue);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    planner::CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        CUDA_CHECK(cudaDeviceSynchronize());

        timer.start();
        const int* p_orderdate = prepare_column_for_fused(cols.orderdate, d_lo_orderdate_dec, ws_orderdate, ws_orderdate_rle);
        const int* p_custkey = prepare_column_for_fused(cols.custkey, d_lo_custkey_dec, ws_custkey, ws_custkey_rle);
        const int* p_suppkey = prepare_column_for_fused(cols.suppkey, d_lo_suppkey_dec, ws_suppkey, ws_suppkey_rle);
        const int* p_revenue = prepare_column_for_fused(cols.revenue, d_lo_revenue_dec, ws_revenue, ws_revenue_rle);
        CUDA_CHECK(cudaDeviceSynchronize());

        auto acc_orderdate = make_accessor(cols.orderdate, p_orderdate);
        auto acc_custkey = make_accessor(cols.custkey, p_custkey);
        auto acc_suppkey = make_accessor(cols.suppkey, p_suppkey);
        auto acc_revenue = make_accessor(cols.revenue, p_revenue);

        probe_q32_fused<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                acc_orderdate, acc_custkey, acc_suppkey, acc_revenue, static_cast<int>(LO_LEN),
                ht_s.data(), static_cast<int>(S_LEN), ht_c.data(), static_cast<int>(C_LEN),
                ht_d.data(), date_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) { ++rows; total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::ofstream csv("/root/autodl-tmp/code/L3/H20/ssb/Planner/q3_results.csv", std::ios::app);
        csv << "34," << run << "," << rows << "," << total_revenue << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

} // namespace ssb

