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
#include "ssb/crystal_like.cuh"
#include "ssb/queries.hpp"
#include "ssb/ssb_dataset.hpp"

namespace ssb {
namespace {

constexpr int kDateMin = 19920101;
constexpr int kDateMax = 19981231;

constexpr int kRegionAmerica = 1;
constexpr int kNationUS = 24;

constexpr int kNumNations = 25;
constexpr int kNumCities = 250;
constexpr int kNumBrands = 1000;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_region_membership(const int* s_region,
                                                    const int* s_suppkey,
                                                    int s_len,
                                                    int region_filter,
                                                    int* ht_s) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_region + tile_offset, keys, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, keys, num_items);
    BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht_s, s_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_region_to_nation(const int* s_region,
                                                   const int* s_suppkey,
                                                   const int* s_nation,
                                                   int s_len,
                                                   int region_filter,
                                                   int* ht_s) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_region + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_nation + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_s, s_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_nation_to_city(const int* s_nation,
                                                 const int* s_suppkey,
                                                 const int* s_city,
                                                 int s_len,
                                                 int nation_filter,
                                                 int* ht_s) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_nation + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, nation_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_city + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_s, s_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_mfgr_membership(const int* p_mfgr, const int* p_partkey, int p_len, int* ht_p) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int mfgr[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, p_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_mfgr + tile_offset, mfgr, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(mfgr, 1, selection_flags, num_items);
    BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(mfgr, 2, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_partkey + tile_offset, keys, num_items);
    BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht_p, p_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_mfgr_to_category(const int* p_mfgr,
                                                   const int* p_partkey,
                                                   const int* p_category,
                                                   int p_len,
                                                   int* ht_p) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, p_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_mfgr + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, 1, selection_flags, num_items);
    BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, 2, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_partkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_category + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_p, p_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_category_to_brand(const int* p_category,
                                                    const int* p_partkey,
                                                    const int* p_brand1,
                                                    int p_len,
                                                    int category_filter,
                                                    int* ht_p) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, p_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_category + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, category_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_partkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_brand1 + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_p, p_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_region_to_nation(const int* c_region,
                                                   const int* c_custkey,
                                                   const int* c_nation,
                                                   int c_len,
                                                   int region_filter,
                                                   int* ht_c) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_region + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_nation + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_c, c_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_region_membership(const int* c_region,
                                                    const int* c_custkey,
                                                    int c_len,
                                                    int region_filter,
                                                    int* ht_c) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_region + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, region_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, keys, num_items);
    BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht_c, c_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_c_nation_to_city(const int* c_nation,
                                                 const int* c_custkey,
                                                 const int* c_city,
                                                 int c_len,
                                                 int nation_filter,
                                                 int* ht_c) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, c_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_nation + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, nation_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_custkey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(c_city + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_c, c_len, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d_all(const int* d_datekey, const int* d_year, int d_len, int* ht_d, int ht_len, int keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, keys, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void
build_hashtable_d_year_set(const int* d_datekey, const int* d_year, int d_len, int year_a, int year_b, int* ht_d, int ht_len, int keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, vals, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, year_a, selection_flags, num_items);
    BlockPredOrEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, year_b, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, keys, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q41(const int* lo_orderdate,
                          const int* lo_partkey,
                          const int* lo_custkey,
                          const int* lo_suppkey,
                          const int* lo_revenue,
                          const int* lo_supplycost,
                          int lo_len,
                          const int* ht_p,
                          int p_ht_len,
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
    int partkey[ITEMS_PER_THREAD];
    int custkey[ITEMS_PER_THREAD];
    int suppkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
    int c_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        custkey[ITEM] = (row < lo_len) ? lo_custkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(custkey, c_nation, selection_flags, ht_c, c_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        suppkey[ITEM] = (row < lo_len) ? lo_suppkey[row] : 0;
    }
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, selection_flags, ht_s, s_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        partkey[ITEM] = (row < lo_len) ? lo_partkey[row] : 0;
    }
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(partkey, selection_flags, ht_p, p_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        orderdate[ITEM] = (row < lo_len) ? lo_orderdate[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(orderdate, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue[row];
            const int supplycost = lo_supplycost[row];
            const unsigned long long profit = static_cast<unsigned long long>(revenue - supplycost);
            const int hash = (c_nation[ITEM] * 7 + (year[ITEM] - 1992)) % (7 * kNumNations);
            res[hash * 4] = year[ITEM];
            res[hash * 4 + 1] = c_nation[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), profit);
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q42(const int* lo_orderdate,
                          const int* lo_partkey,
                          const int* lo_custkey,
                          const int* lo_suppkey,
                          const int* lo_revenue,
                          const int* lo_supplycost,
                          int lo_len,
                          const int* ht_p,
                          int p_ht_len,
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
    int partkey[ITEMS_PER_THREAD];
    int custkey[ITEMS_PER_THREAD];
    int suppkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
    int p_category[ITEMS_PER_THREAD];
    int s_nation[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        custkey[ITEM] = (row < lo_len) ? lo_custkey[row] : 0;
    }
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(custkey, selection_flags, ht_c, c_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        suppkey[ITEM] = (row < lo_len) ? lo_suppkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, s_nation, selection_flags, ht_s, s_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        partkey[ITEM] = (row < lo_len) ? lo_partkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(partkey, p_category, selection_flags, ht_p, p_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        orderdate[ITEM] = (row < lo_len) ? lo_orderdate[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(orderdate, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue[row];
            const int supplycost = lo_supplycost[row];
            const unsigned long long profit = static_cast<unsigned long long>(revenue - supplycost);
            const int hash = ((year[ITEM] - 1992) * kNumNations * kNumNations + s_nation[ITEM] * kNumNations + p_category[ITEM]) %
                             (2 * kNumNations * kNumNations);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = s_nation[ITEM];
            res[hash * 6 + 2] = p_category[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), profit);
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q43(const int* lo_orderdate,
                          const int* lo_partkey,
                          const int* lo_custkey,
                          const int* lo_suppkey,
                          const int* lo_revenue,
                          const int* lo_supplycost,
                          int lo_len,
                          const int* ht_p,
                          int p_ht_len,
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
    int partkey[ITEMS_PER_THREAD];
    int custkey[ITEMS_PER_THREAD];
    int suppkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
    int p_brand[ITEMS_PER_THREAD];
    int s_city[ITEMS_PER_THREAD];
    int c_city[ITEMS_PER_THREAD];
    int year[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, lo_len - tile_offset);

    InitFlags<BLOCK_THREADS, ITEMS_PER_THREAD>(selection_flags);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        custkey[ITEM] = (row < lo_len) ? lo_custkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(custkey, c_city, selection_flags, ht_c, c_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        suppkey[ITEM] = (row < lo_len) ? lo_suppkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, s_city, selection_flags, ht_s, s_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        partkey[ITEM] = (row < lo_len) ? lo_partkey[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(partkey, p_brand, selection_flags, ht_p, p_ht_len, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        orderdate[ITEM] = (row < lo_len) ? lo_orderdate[row] : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(orderdate, year, selection_flags, ht_d, d_ht_len, d_min, num_items);

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < lo_len && selection_flags[ITEM]) {
            const int revenue = lo_revenue[row];
            const int supplycost = lo_supplycost[row];
            const unsigned long long profit = static_cast<unsigned long long>(revenue - supplycost);
            const int hash = ((year[ITEM] - 1992) * kNumCities * kNumBrands + s_city[ITEM] * kNumBrands + p_brand[ITEM]) %
                             (2 * kNumCities * kNumBrands);
            res[hash * 6] = year[ITEM];
            res[hash * 6 + 1] = s_city[ITEM];
            res[hash * 6 + 2] = c_city[ITEM];
            res[hash * 6 + 3] = p_brand[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), profit);
        }
    }
}

struct Q4EncodedLO {
    planner::EncodedColumn orderdate;
    planner::EncodedColumn partkey;
    planner::EncodedColumn custkey;
    planner::EncodedColumn suppkey;
    planner::EncodedColumn revenue;
    planner::EncodedColumn supplycost;
};

Q4EncodedLO load_and_encode_lo_q4(const RunOptions& opt) {
    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_partkey = load_column<int>(opt.data_dir, "lo_partkey", LO_LEN);
    auto h_lo_custkey = load_column<int>(opt.data_dir, "lo_custkey", LO_LEN);
    auto h_lo_suppkey = load_column<int>(opt.data_dir, "lo_suppkey", LO_LEN);
    auto h_lo_revenue = load_column<int>(opt.data_dir, "lo_revenue", LO_LEN);
    auto h_lo_supplycost = load_column<int>(opt.data_dir, "lo_supplycost", LO_LEN);

    Q4EncodedLO cols;
    cols.orderdate = planner::encode_column_planner(h_lo_orderdate.get(), LO_LEN);
    cols.partkey = planner::encode_column_planner(h_lo_partkey.get(), LO_LEN);
    cols.custkey = planner::encode_column_planner(h_lo_custkey.get(), LO_LEN);
    cols.suppkey = planner::encode_column_planner(h_lo_suppkey.get(), LO_LEN);
    cols.revenue = planner::encode_column_planner(h_lo_revenue.get(), LO_LEN);
    cols.supplycost = planner::encode_column_planner(h_lo_supplycost.get(), LO_LEN);

    cols.orderdate.upload(h_lo_orderdate.get());
    cols.partkey.upload(h_lo_partkey.get());
    cols.custkey.upload(h_lo_custkey.get());
    cols.suppkey.upload(h_lo_suppkey.get());
    cols.revenue.upload(h_lo_revenue.get());
    cols.supplycost.upload(h_lo_supplycost.get());
    return cols;
}

struct DecompressedQ4Ptrs {
    const int* orderdate = nullptr;
    const int* partkey = nullptr;
    const int* custkey = nullptr;
    const int* suppkey = nullptr;
    const int* revenue = nullptr;
    const int* supplycost = nullptr;
};

DecompressedQ4Ptrs decompress_q4_cols(const Q4EncodedLO& cols,
                                      planner::DeviceBuffer<int>& d_orderdate,
                                      planner::DeviceBuffer<int>& d_partkey,
                                      planner::DeviceBuffer<int>& d_custkey,
                                      planner::DeviceBuffer<int>& d_suppkey,
                                      planner::DeviceBuffer<int>& d_revenue,
                                      planner::DeviceBuffer<int>& d_supplycost,
                                      planner::DeltaWorkspace& ws_orderdate,
                                      planner::DeltaWorkspace& ws_partkey,
                                      planner::DeltaWorkspace& ws_custkey,
                                      planner::DeltaWorkspace& ws_suppkey,
                                      planner::DeltaWorkspace& ws_revenue,
                                      planner::DeltaWorkspace& ws_supplycost,
                                      planner::RleWorkspace& ws_orderdate_rle,
                                      planner::RleWorkspace& ws_partkey_rle,
                                      planner::RleWorkspace& ws_custkey_rle,
                                      planner::RleWorkspace& ws_suppkey_rle,
                                      planner::RleWorkspace& ws_revenue_rle,
                                      planner::RleWorkspace& ws_supplycost_rle) {
    DecompressedQ4Ptrs out;

    auto decode_one = [&](const planner::EncodedColumn& enc,
                          planner::DeviceBuffer<int>& buf,
                          planner::DeltaWorkspace& dws,
                          planner::RleWorkspace& rws) -> const int* {
        if (enc.scheme == planner::Scheme::Uncompressed) {
            return enc.d_ints_ptr();
        }
        buf.resize(enc.n);
        switch (enc.scheme) {
            case planner::Scheme::NS: planner::ns_decode_into(enc, buf.data()); break;
            case planner::Scheme::FOR_NS: planner::for_ns_decode_into(enc, buf.data()); break;
            case planner::Scheme::DELTA_NS: planner::delta_ns_decode_into(enc, buf.data(), dws); break;
            case planner::Scheme::DELTA_FOR_NS: planner::delta_for_ns_decode_into(enc, buf.data(), dws); break;
            case planner::Scheme::RLE: planner::rle_decode_into(enc, buf.data(), rws); break;
            case planner::Scheme::Uncompressed: break;
        }
        return buf.data();
    };

    out.orderdate = decode_one(cols.orderdate, d_orderdate, ws_orderdate, ws_orderdate_rle);
    out.partkey = decode_one(cols.partkey, d_partkey, ws_partkey, ws_partkey_rle);
    out.custkey = decode_one(cols.custkey, d_custkey, ws_custkey, ws_custkey_rle);
    out.suppkey = decode_one(cols.suppkey, d_suppkey, ws_suppkey, ws_suppkey_rle);
    out.revenue = decode_one(cols.revenue, d_revenue, ws_revenue, ws_revenue_rle);
    out.supplycost = decode_one(cols.supplycost, d_supplycost, ws_supplycost, ws_supplycost_rle);
    return out;
}

size_t get_compressed_size_q4(const planner::EncodedColumn& enc) {
    if (enc.scheme == planner::Scheme::Uncompressed) return enc.n * sizeof(int);
    if (enc.scheme == planner::Scheme::RLE) return enc.rle_values.size() * sizeof(int) * 2;
    return enc.bytes.size();
}

} // namespace

void run_q41(const RunOptions& opt) {
    std::cout << "=== Q41 (Planner - Cascading Decompression) ===" << std::endl;

    const int d_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q4(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_region = load_column<int>(opt.data_dir, "s_region", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_region = load_column<int>(opt.data_dir, "c_region", C_LEN);
    auto h_c_nation = load_column<int>(opt.data_dir, "c_nation", C_LEN);

    auto h_p_partkey = load_column<int>(opt.data_dir, "p_partkey", P_LEN);
    auto h_p_mfgr = load_column<int>(opt.data_dir, "p_mfgr", P_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_region(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_region(C_LEN);
    planner::DeviceBuffer<int> d_c_nation(C_LEN);
    planner::DeviceBuffer<int> d_p_partkey(P_LEN);
    planner::DeviceBuffer<int> d_p_mfgr(P_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_region.data(), h_s_region.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_region.data(), h_c_region.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_partkey.data(), h_p_partkey.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_mfgr.data(), h_p_mfgr.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * d_ht_len);
    planner::DeviceBuffer<int> ht_s(S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);
    planner::DeviceBuffer<int> ht_p(P_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * d_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));
    CUDA_CHECK(cudaMemset(ht_p.data(), 0, sizeof(int) * P_LEN));

    build_hashtable_s_region_membership<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_region.data(), d_s_suppkey.data(), static_cast<int>(S_LEN), kRegionAmerica, ht_s.data());

    build_hashtable_c_region_to_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_region.data(), d_c_custkey.data(), d_c_nation.data(), static_cast<int>(C_LEN), kRegionAmerica, ht_c.data());

    build_hashtable_p_mfgr_membership<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((P_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_p_mfgr.data(), d_p_partkey.data(), static_cast<int>(P_LEN), ht_p.data());

    build_hashtable_d_all<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), ht_d.data(), d_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 7 * kNumNations;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 4);

    planner::DeviceBuffer<int> d_lo_orderdate;
    planner::DeviceBuffer<int> d_lo_partkey;
    planner::DeviceBuffer<int> d_lo_custkey;
    planner::DeviceBuffer<int> d_lo_suppkey;
    planner::DeviceBuffer<int> d_lo_revenue;
    planner::DeviceBuffer<int> d_lo_supplycost;
    planner::DeltaWorkspace ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost;
    planner::RleWorkspace ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle;

    // Warmup (decode + probe, not reported).
    {
        planner::CudaEventTimer decomp_timer;
        decomp_timer.start();
        auto ptrs = decompress_q4_cols(cols,
                                       d_lo_orderdate,
                                       d_lo_partkey,
                                       d_lo_custkey,
                                       d_lo_suppkey,
                                       d_lo_revenue,
                                       d_lo_supplycost,
                                       ws_orderdate,
                                       ws_partkey,
                                       ws_custkey,
                                       ws_suppkey,
                                       ws_revenue,
                                       ws_supplycost,
                                       ws_orderdate_rle,
                                       ws_partkey_rle,
                                       ws_custkey_rle,
                                       ws_suppkey_rle,
                                       ws_revenue_rle,
                                       ws_supplycost_rle);
        (void)decomp_timer.stop();

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 4));
        probe_q41<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate,
                ptrs.partkey,
                ptrs.custkey,
                ptrs.suppkey,
                ptrs.revenue,
                ptrs.supplycost,
                static_cast<int>(LO_LEN),
                ht_p.data(),
                static_cast<int>(P_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                d_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 6;
    size_t compressed_size = get_compressed_size_q4(cols.orderdate) + get_compressed_size_q4(cols.partkey) +
                             get_compressed_size_q4(cols.custkey) + get_compressed_size_q4(cols.suppkey) +
                             get_compressed_size_q4(cols.revenue) + get_compressed_size_q4(cols.supplycost);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x (" << (original_size/1e6) << " MB -> " << (compressed_size/1e6) << " MB)" << std::endl;

    planner::CudaEventTimer timer;

    for (int run = 0; run < opt.runs; ++run) {
        if (d_lo_orderdate.size() > 0) cudaMemset(d_lo_orderdate.data(), 0, d_lo_orderdate.size() * sizeof(int));
        if (d_lo_partkey.size() > 0) cudaMemset(d_lo_partkey.data(), 0, d_lo_partkey.size() * sizeof(int));
        if (d_lo_custkey.size() > 0) cudaMemset(d_lo_custkey.data(), 0, d_lo_custkey.size() * sizeof(int));
        if (d_lo_suppkey.size() > 0) cudaMemset(d_lo_suppkey.data(), 0, d_lo_suppkey.size() * sizeof(int));
        if (d_lo_revenue.size() > 0) cudaMemset(d_lo_revenue.data(), 0, d_lo_revenue.size() * sizeof(int));
        if (d_lo_supplycost.size() > 0) cudaMemset(d_lo_supplycost.data(), 0, d_lo_supplycost.size() * sizeof(int));
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 4));
        cudaDeviceSynchronize();

        timer.start();
        auto ptrs = decompress_q4_cols(cols, d_lo_orderdate, d_lo_partkey, d_lo_custkey, d_lo_suppkey, d_lo_revenue, d_lo_supplycost,
                                       ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost,
                                       ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle);

        probe_q41<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate, ptrs.partkey, ptrs.custkey, ptrs.suppkey, ptrs.revenue, ptrs.supplycost,
                static_cast<int>(LO_LEN), ht_p.data(), static_cast<int>(P_LEN), ht_s.data(), static_cast<int>(S_LEN),
                ht_c.data(), static_cast<int>(C_LEN), ht_d.data(), d_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 4);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 4, cudaMemcpyDeviceToHost));

        unsigned long long total_profit = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 4] != 0) { ++rows; total_profit += reinterpret_cast<const unsigned long long*>(&h_res[i * 4 + 2])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", profit=" << total_profit
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/Planner/q4_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) csv << "query,run,rows,profit,time_decomp_query_ms,throughput_gbps,compression_ratio\n";
        csv << "41," << run << "," << rows << "," << total_profit << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

void run_q42(const RunOptions& opt) {
    std::cout << "=== Q42 (Planner - Cascading Decompression) ===" << std::endl;

    const int d_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q4(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_region = load_column<int>(opt.data_dir, "s_region", S_LEN);
    auto h_s_nation = load_column<int>(opt.data_dir, "s_nation", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_region = load_column<int>(opt.data_dir, "c_region", C_LEN);

    auto h_p_partkey = load_column<int>(opt.data_dir, "p_partkey", P_LEN);
    auto h_p_mfgr = load_column<int>(opt.data_dir, "p_mfgr", P_LEN);
    auto h_p_category = load_column<int>(opt.data_dir, "p_category", P_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_region(S_LEN);
    planner::DeviceBuffer<int> d_s_nation(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_region(C_LEN);
    planner::DeviceBuffer<int> d_p_partkey(P_LEN);
    planner::DeviceBuffer<int> d_p_mfgr(P_LEN);
    planner::DeviceBuffer<int> d_p_category(P_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_region.data(), h_s_region.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_region.data(), h_c_region.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_partkey.data(), h_p_partkey.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_mfgr.data(), h_p_mfgr.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_category.data(), h_p_category.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * d_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(C_LEN);
    planner::DeviceBuffer<int> ht_p(static_cast<std::size_t>(2) * P_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * d_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * C_LEN));
    CUDA_CHECK(cudaMemset(ht_p.data(), 0, sizeof(int) * 2 * P_LEN));

    build_hashtable_c_region_membership<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_region.data(), d_c_custkey.data(), static_cast<int>(C_LEN), kRegionAmerica, ht_c.data());

    build_hashtable_s_region_to_nation<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_region.data(), d_s_suppkey.data(), d_s_nation.data(), static_cast<int>(S_LEN), kRegionAmerica, ht_s.data());

    build_hashtable_p_mfgr_to_category<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((P_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_p_mfgr.data(), d_p_partkey.data(), d_p_category.data(), static_cast<int>(P_LEN), ht_p.data());

    build_hashtable_d_year_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), 1997, 1998, ht_d.data(), d_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 2 * kNumNations * kNumNations;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate;
    planner::DeviceBuffer<int> d_lo_partkey;
    planner::DeviceBuffer<int> d_lo_custkey;
    planner::DeviceBuffer<int> d_lo_suppkey;
    planner::DeviceBuffer<int> d_lo_revenue;
    planner::DeviceBuffer<int> d_lo_supplycost;
    planner::DeltaWorkspace ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost;
    planner::RleWorkspace ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle;

    // Warmup (decode + probe, not reported).
    {
        planner::CudaEventTimer decomp_timer;
        decomp_timer.start();
        auto ptrs = decompress_q4_cols(cols,
                                       d_lo_orderdate,
                                       d_lo_partkey,
                                       d_lo_custkey,
                                       d_lo_suppkey,
                                       d_lo_revenue,
                                       d_lo_supplycost,
                                       ws_orderdate,
                                       ws_partkey,
                                       ws_custkey,
                                       ws_suppkey,
                                       ws_revenue,
                                       ws_supplycost,
                                       ws_orderdate_rle,
                                       ws_partkey_rle,
                                       ws_custkey_rle,
                                       ws_suppkey_rle,
                                       ws_revenue_rle,
                                       ws_supplycost_rle);
        (void)decomp_timer.stop();

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q42<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate,
                ptrs.partkey,
                ptrs.custkey,
                ptrs.suppkey,
                ptrs.revenue,
                ptrs.supplycost,
                static_cast<int>(LO_LEN),
                ht_p.data(),
                static_cast<int>(P_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                d_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 6;
    size_t compressed_size = get_compressed_size_q4(cols.orderdate) + get_compressed_size_q4(cols.partkey) +
                             get_compressed_size_q4(cols.custkey) + get_compressed_size_q4(cols.suppkey) +
                             get_compressed_size_q4(cols.revenue) + get_compressed_size_q4(cols.supplycost);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    planner::CudaEventTimer timer;

    for (int run = 0; run < opt.runs; ++run) {
        if (d_lo_orderdate.size() > 0) cudaMemset(d_lo_orderdate.data(), 0, d_lo_orderdate.size() * sizeof(int));
        if (d_lo_partkey.size() > 0) cudaMemset(d_lo_partkey.data(), 0, d_lo_partkey.size() * sizeof(int));
        if (d_lo_custkey.size() > 0) cudaMemset(d_lo_custkey.data(), 0, d_lo_custkey.size() * sizeof(int));
        if (d_lo_suppkey.size() > 0) cudaMemset(d_lo_suppkey.data(), 0, d_lo_suppkey.size() * sizeof(int));
        if (d_lo_revenue.size() > 0) cudaMemset(d_lo_revenue.data(), 0, d_lo_revenue.size() * sizeof(int));
        if (d_lo_supplycost.size() > 0) cudaMemset(d_lo_supplycost.data(), 0, d_lo_supplycost.size() * sizeof(int));
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        cudaDeviceSynchronize();

        timer.start();
        auto ptrs = decompress_q4_cols(cols, d_lo_orderdate, d_lo_partkey, d_lo_custkey, d_lo_suppkey, d_lo_revenue, d_lo_supplycost,
                                       ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost,
                                       ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle);

        probe_q42<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate, ptrs.partkey, ptrs.custkey, ptrs.suppkey, ptrs.revenue, ptrs.supplycost,
                static_cast<int>(LO_LEN), ht_p.data(), static_cast<int>(P_LEN), ht_s.data(), static_cast<int>(S_LEN),
                ht_c.data(), static_cast<int>(C_LEN), ht_d.data(), d_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_profit = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) { ++rows; total_profit += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", profit=" << total_profit
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::ofstream csv("/root/autodl-tmp/code/L3/H20/ssb/Planner/q4_results.csv", std::ios::app);
        csv << "42," << run << "," << rows << "," << total_profit << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

void run_q43(const RunOptions& opt) {
    std::cout << "=== Q43 (Planner - Cascading Decompression) ===" << std::endl;

    const int d_ht_len = (kDateMax - kDateMin + 1);

    planner::CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q4(opt);

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_nation = load_column<int>(opt.data_dir, "s_nation", S_LEN);
    auto h_s_city = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_nation = load_column<int>(opt.data_dir, "c_nation", C_LEN);
    auto h_c_city = load_column<int>(opt.data_dir, "c_city", C_LEN);

    auto h_p_partkey = load_column<int>(opt.data_dir, "p_partkey", P_LEN);
    auto h_p_category = load_column<int>(opt.data_dir, "p_category", P_LEN);
    auto h_p_brand1 = load_column<int>(opt.data_dir, "p_brand1", P_LEN);

    planner::DeviceBuffer<int> d_d_datekey(D_LEN);
    planner::DeviceBuffer<int> d_d_year(D_LEN);
    planner::DeviceBuffer<int> d_s_suppkey(S_LEN);
    planner::DeviceBuffer<int> d_s_nation(S_LEN);
    planner::DeviceBuffer<int> d_s_city(S_LEN);
    planner::DeviceBuffer<int> d_c_custkey(C_LEN);
    planner::DeviceBuffer<int> d_c_nation(C_LEN);
    planner::DeviceBuffer<int> d_c_city(C_LEN);
    planner::DeviceBuffer<int> d_p_partkey(P_LEN);
    planner::DeviceBuffer<int> d_p_category(P_LEN);
    planner::DeviceBuffer<int> d_p_brand1(P_LEN);

    CUDA_CHECK(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_partkey.data(), h_p_partkey.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_category.data(), h_p_category.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_p_brand1.data(), h_p_brand1.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    planner::DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * d_ht_len);
    planner::DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    planner::DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);
    planner::DeviceBuffer<int> ht_p(static_cast<std::size_t>(2) * P_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    planner::CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * d_ht_len));
    CUDA_CHECK(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));
    CUDA_CHECK(cudaMemset(ht_p.data(), 0, sizeof(int) * 2 * P_LEN));

    build_hashtable_s_nation_to_city<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_nation.data(), d_s_suppkey.data(), d_s_city.data(), static_cast<int>(S_LEN), kNationUS, ht_s.data());

    build_hashtable_c_nation_to_city<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_nation.data(), d_c_custkey.data(), d_c_city.data(), static_cast<int>(C_LEN), kNationUS, ht_c.data());

    build_hashtable_p_category_to_brand<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((P_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_p_category.data(), d_p_partkey.data(), d_p_brand1.data(), static_cast<int>(P_LEN), 13, ht_p.data());

    build_hashtable_d_year_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), 1997, 1998, ht_d.data(), d_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 2 * kNumCities * kNumBrands;
    planner::DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    planner::DeviceBuffer<int> d_lo_orderdate;
    planner::DeviceBuffer<int> d_lo_partkey;
    planner::DeviceBuffer<int> d_lo_custkey;
    planner::DeviceBuffer<int> d_lo_suppkey;
    planner::DeviceBuffer<int> d_lo_revenue;
    planner::DeviceBuffer<int> d_lo_supplycost;
    planner::DeltaWorkspace ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost;
    planner::RleWorkspace ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle;

    // Warmup (decode + probe, not reported).
    {
        planner::CudaEventTimer decomp_timer;
        decomp_timer.start();
        auto ptrs = decompress_q4_cols(cols,
                                       d_lo_orderdate,
                                       d_lo_partkey,
                                       d_lo_custkey,
                                       d_lo_suppkey,
                                       d_lo_revenue,
                                       d_lo_supplycost,
                                       ws_orderdate,
                                       ws_partkey,
                                       ws_custkey,
                                       ws_suppkey,
                                       ws_revenue,
                                       ws_supplycost,
                                       ws_orderdate_rle,
                                       ws_partkey_rle,
                                       ws_custkey_rle,
                                       ws_suppkey_rle,
                                       ws_revenue_rle,
                                       ws_supplycost_rle);
        (void)decomp_timer.stop();

        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        probe_q43<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate,
                ptrs.partkey,
                ptrs.custkey,
                ptrs.suppkey,
                ptrs.revenue,
                ptrs.supplycost,
                static_cast<int>(LO_LEN),
                ht_p.data(),
                static_cast<int>(P_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                d_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    size_t original_size = LO_LEN * sizeof(int) * 6;
    size_t compressed_size = get_compressed_size_q4(cols.orderdate) + get_compressed_size_q4(cols.partkey) +
                             get_compressed_size_q4(cols.custkey) + get_compressed_size_q4(cols.suppkey) +
                             get_compressed_size_q4(cols.revenue) + get_compressed_size_q4(cols.supplycost);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;
    std::cout << "  Compression ratio: " << compression_ratio << "x" << std::endl;

    planner::CudaEventTimer timer;

    for (int run = 0; run < opt.runs; ++run) {
        if (d_lo_orderdate.size() > 0) cudaMemset(d_lo_orderdate.data(), 0, d_lo_orderdate.size() * sizeof(int));
        if (d_lo_partkey.size() > 0) cudaMemset(d_lo_partkey.data(), 0, d_lo_partkey.size() * sizeof(int));
        if (d_lo_custkey.size() > 0) cudaMemset(d_lo_custkey.data(), 0, d_lo_custkey.size() * sizeof(int));
        if (d_lo_suppkey.size() > 0) cudaMemset(d_lo_suppkey.data(), 0, d_lo_suppkey.size() * sizeof(int));
        if (d_lo_revenue.size() > 0) cudaMemset(d_lo_revenue.data(), 0, d_lo_revenue.size() * sizeof(int));
        if (d_lo_supplycost.size() > 0) cudaMemset(d_lo_supplycost.data(), 0, d_lo_supplycost.size() * sizeof(int));
        CUDA_CHECK(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
        cudaDeviceSynchronize();

        timer.start();
        auto ptrs = decompress_q4_cols(cols, d_lo_orderdate, d_lo_partkey, d_lo_custkey, d_lo_suppkey, d_lo_revenue, d_lo_supplycost,
                                       ws_orderdate, ws_partkey, ws_custkey, ws_suppkey, ws_revenue, ws_supplycost,
                                       ws_orderdate_rle, ws_partkey_rle, ws_custkey_rle, ws_suppkey_rle, ws_revenue_rle, ws_supplycost_rle);

        probe_q43<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                ptrs.orderdate, ptrs.partkey, ptrs.custkey, ptrs.suppkey, ptrs.revenue, ptrs.supplycost,
                static_cast<int>(LO_LEN), ht_p.data(), static_cast<int>(P_LEN), ht_s.data(), static_cast<int>(S_LEN),
                ht_c.data(), static_cast<int>(C_LEN), ht_d.data(), d_ht_len, kDateMin, d_res.data());
        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_profit = 0; int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) { ++rows; total_profit += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0]; }
        }

        double throughput_gbps = (original_size / 1e9) / (time_decomp_query / 1000.0);
        std::cout << "  Run " << run << ": rows=" << rows << ", profit=" << total_profit
                  << ", decomp+query=" << time_decomp_query << " ms, throughput=" << throughput_gbps << " GB/s" << std::endl;

        std::ofstream csv("/root/autodl-tmp/code/L3/H20/ssb/Planner/q4_results.csv", std::ios::app);
        csv << "43," << run << "," << rows << "," << total_profit << "," << time_decomp_query << "," << throughput_gbps << "," << compression_ratio << "\n";
    }
}

} // namespace ssb

