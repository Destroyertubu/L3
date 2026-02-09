#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "ssb/crystal_like.cuh"
#include "ssb/device_buffer.hpp"
#include "ssb/gpu_bp_column.hpp"
#include "ssb/gpu_bp_decode.cuh"
#include "ssb/queries.hpp"
#include "ssb/ssb_dataset.hpp"
#include "ssb/timing.hpp"

namespace ssb {
namespace {

constexpr int kDateMin = 19920101;
constexpr int kDateMax = 19981231;

constexpr int kYearMin = 1992;
constexpr int kYearMaxQ31 = 1997;
constexpr int kYearMaxQ32 = 1997;

constexpr int kNationUS = 24;

constexpr int kNumNations = 25;
constexpr int kNumCities  = 250;

constexpr int kCityUK1 = 249;
constexpr int kCityUK5 = 244;

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_region_to_nation(const int* s_region,
                                                   const int* s_suppkey,
                                                   const int* s_nation,
                                                   int        s_len,
                                                   int        region_filter,
                                                   int*       ht_s,
                                                   int        ht_len) {
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
                                                   int        c_len,
                                                   int        region_filter,
                                                   int*       ht_c,
                                                   int        ht_len) {
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
                                                 int        s_len,
                                                 int        nation_filter,
                                                 int*       ht_s,
                                                 int        ht_len) {
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
                                                 int        c_len,
                                                 int        nation_filter,
                                                 int*       ht_c,
                                                 int        ht_len) {
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
                                             int        d_len,
                                             int        year_min,
                                             int        year_max,
                                             int*       ht_d,
                                             int        ht_len,
                                             int        keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, vals, num_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, year_min, selection_flags, num_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(vals, year_max, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, keys, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_d_datekey_range(const int* d_datekey,
                                                const int* d_year,
                                                int        d_len,
                                                int        date_min,
                                                int        date_max,
                                                int*       ht_d,
                                                int        ht_len,
                                                int        keys_min) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int keys[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, d_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_datekey + tile_offset, keys, num_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, date_min, selection_flags, num_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, date_max, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(d_year + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht_d, ht_len, keys_min, num_items);
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q31(bp::DeviceColumn lo_orderdate,
                          bp::DeviceColumn lo_custkey,
                          bp::DeviceColumn lo_suppkey,
                          bp::DeviceColumn lo_revenue,
                          int              lo_len,
                          const int*       ht_s,
                          int              s_ht_len,
                          const int*       ht_c,
                          int              c_ht_len,
                          const int*       ht_d,
                          int              d_ht_len,
                          int              d_keys_min,
                          int*             res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int custkey[ITEMS_PER_THREAD];
    int suppkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
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
        custkey[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_custkey, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(custkey, c_nation, selection_flags, ht_c, c_ht_len, num_items);

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
        suppkey[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_suppkey, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, s_nation, selection_flags, ht_s, s_ht_len, num_items);

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
        orderdate[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_orderdate, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(orderdate, year, selection_flags, ht_d, d_ht_len, d_keys_min, num_items);

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
            const int revenue = static_cast<int>(bp::decode_u32(lo_revenue, static_cast<uint32_t>(row)));
            const int hash = (s_nation[ITEM] * kNumNations * 7 + c_nation[ITEM] * 7 + (year[ITEM] - 1992)) % (7 * kNumNations * kNumNations);
            res[hash * 6]     = year[ITEM];
            res[hash * 6 + 1] = c_nation[ITEM];
            res[hash * 6 + 2] = s_nation[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void probe_q32(bp::DeviceColumn lo_orderdate,
                          bp::DeviceColumn lo_custkey,
                          bp::DeviceColumn lo_suppkey,
                          bp::DeviceColumn lo_revenue,
                          int              lo_len,
                          const int*       ht_s,
                          int              s_ht_len,
                          const int*       ht_c,
                          int              c_ht_len,
                          const int*       ht_d,
                          int              d_ht_len,
                          int              d_keys_min,
                          int*             res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int selection_flags[ITEMS_PER_THREAD];
    int custkey[ITEMS_PER_THREAD];
    int suppkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
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
        suppkey[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_suppkey, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, s_city, selection_flags, ht_s, s_ht_len, num_items);

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
        custkey[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_custkey, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(custkey, c_city, selection_flags, ht_c, c_ht_len, num_items);

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
        orderdate[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_orderdate, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(orderdate, year, selection_flags, ht_d, d_ht_len, d_keys_min, num_items);

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
            const int revenue = static_cast<int>(bp::decode_u32(lo_revenue, static_cast<uint32_t>(row)));
            const int hash = (s_city[ITEM] * kNumCities * 6 + c_city[ITEM] * 6 + (year[ITEM] - 1992)) % (6 * kNumCities * kNumCities);
            res[hash * 6]     = year[ITEM];
            res[hash * 6 + 1] = c_city[ITEM];
            res[hash * 6 + 2] = s_city[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 6 + 4]), static_cast<unsigned long long>(revenue));
        }
    }
}

struct Q3DeviceColumns {
    bp::EncodedColumn orderdate;
    bp::EncodedColumn custkey;
    bp::EncodedColumn suppkey;
    bp::EncodedColumn revenue;

    // 压缩统计
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;

    bp::DeviceColumn d_orderdate() const { return orderdate.device_view(); }
    bp::DeviceColumn d_custkey() const { return custkey.device_view(); }
    bp::DeviceColumn d_suppkey() const { return suppkey.device_view(); }
    bp::DeviceColumn d_revenue() const { return revenue.device_view(); }
};

Q3DeviceColumns load_and_encode_lo_q3(const RunOptions& opt) {
    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_custkey   = load_column<int>(opt.data_dir, "lo_custkey", LO_LEN);
    auto h_lo_suppkey   = load_column<int>(opt.data_dir, "lo_suppkey", LO_LEN);
    auto h_lo_revenue   = load_column<int>(opt.data_dir, "lo_revenue", LO_LEN);

    Q3DeviceColumns cols;
    cols.orderdate = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_orderdate.get()), LO_LEN);
    cols.custkey   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_custkey.get()), LO_LEN);
    cols.suppkey   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_suppkey.get()), LO_LEN);
    cols.revenue   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_revenue.get()), LO_LEN);

    // 压缩率统计
    cols.original_size = LO_LEN * sizeof(int) * 4;
    cols.compressed_size = cols.orderdate.bytes.size() + cols.custkey.bytes.size() +
                           cols.suppkey.bytes.size() + cols.revenue.bytes.size();
    cols.compression_ratio = static_cast<double>(cols.original_size) / cols.compressed_size;

    cols.orderdate.upload();
    cols.custkey.upload();
    cols.suppkey.upload();
    cols.revenue.upload();
    return cols;
}

} // namespace

void run_q31(const RunOptions& opt) {
    std::cout << "=== Q31 (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << cols.orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_custkey:       " << cols.custkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_suppkey:       " << cols.suppkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_revenue:       " << cols.revenue.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (cols.original_size/1e6) << " MB -> " << (cols.compressed_size/1e6)
              << " MB (ratio: " << cols.compression_ratio << "x)" << std::endl;

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year    = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_region  = load_column<int>(opt.data_dir, "s_region", S_LEN);
    auto h_s_nation  = load_column<int>(opt.data_dir, "s_nation", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_region  = load_column<int>(opt.data_dir, "c_region", C_LEN);
    auto h_c_nation  = load_column<int>(opt.data_dir, "c_nation", C_LEN);

    DeviceBuffer<int> d_d_datekey(D_LEN);
    DeviceBuffer<int> d_d_year(D_LEN);
    DeviceBuffer<int> d_s_suppkey(S_LEN);
    DeviceBuffer<int> d_s_region(S_LEN);
    DeviceBuffer<int> d_s_nation(S_LEN);
    DeviceBuffer<int> d_c_custkey(C_LEN);
    DeviceBuffer<int> d_c_region(C_LEN);
    DeviceBuffer<int> d_c_nation(C_LEN);

    CUDA_CHECK_ERROR(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_region.data(), h_s_region.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_region.data(), h_c_region.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    // HT sizes follow the same "perfect hash" strategy as ssb-fslgpu.
    DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Q31: filter s_region=ASIA(2), c_region=ASIA(2), d_year in [1992,1997]
    CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK_ERROR(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK_ERROR(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK_ERROR(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

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
    DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    // Warmup (not timed / not reported).
    CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
    probe_q31<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            cols.d_orderdate(),
            cols.d_custkey(),
            cols.d_suppkey(),
            cols.d_revenue(),
            static_cast<int>(LO_LEN),
            ht_s.data(),
            static_cast<int>(S_LEN),
            ht_c.data(),
            static_cast<int>(C_LEN),
            ht_d.data(),
            date_ht_len,
            kDateMin,
            d_res.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));

        timer.start();
        probe_q31<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                cols.d_orderdate(),
                cols.d_custkey(),
                cols.d_suppkey(),
                cols.d_revenue(),
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK_ERROR(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0];
            }
        }

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (cols.original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q3_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,rows,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << 31 << "," << run << "," << rows << "," << total_revenue << ","
            << time_kernel << "," << throughput_gbps << ","
            << cols.compression_ratio << "," << (cols.original_size/1e6) << "," << (cols.compressed_size/1e6) << "\n";
        csv.close();
    }

    cols.orderdate.reset_device();
    cols.custkey.reset_device();
    cols.suppkey.reset_device();
    cols.revenue.reset_device();
}

void run_q32(const RunOptions& opt) {
    std::cout << "=== Q32 (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);

    CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << cols.orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_custkey:       " << cols.custkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_suppkey:       " << cols.suppkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_revenue:       " << cols.revenue.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (cols.original_size/1e6) << " MB -> " << (cols.compressed_size/1e6)
              << " MB (ratio: " << cols.compression_ratio << "x)" << std::endl;

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year    = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_nation  = load_column<int>(opt.data_dir, "s_nation", S_LEN);
    auto h_s_city    = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_nation  = load_column<int>(opt.data_dir, "c_nation", C_LEN);
    auto h_c_city    = load_column<int>(opt.data_dir, "c_city", C_LEN);

    DeviceBuffer<int> d_d_datekey(D_LEN);
    DeviceBuffer<int> d_d_year(D_LEN);
    DeviceBuffer<int> d_s_suppkey(S_LEN);
    DeviceBuffer<int> d_s_nation(S_LEN);
    DeviceBuffer<int> d_s_city(S_LEN);
    DeviceBuffer<int> d_c_custkey(C_LEN);
    DeviceBuffer<int> d_c_nation(C_LEN);
    DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK_ERROR(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_nation.data(), h_s_nation.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_nation.data(), h_c_nation.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK_ERROR(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK_ERROR(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK_ERROR(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

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
    DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    // Warmup (not timed / not reported).
    CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
    probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            cols.d_orderdate(),
            cols.d_custkey(),
            cols.d_suppkey(),
            cols.d_revenue(),
            static_cast<int>(LO_LEN),
            ht_s.data(),
            static_cast<int>(S_LEN),
            ht_c.data(),
            static_cast<int>(C_LEN),
            ht_d.data(),
            date_ht_len,
            kDateMin,
            d_res.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));

        timer.start();
        probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                cols.d_orderdate(),
                cols.d_custkey(),
                cols.d_suppkey(),
                cols.d_revenue(),
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK_ERROR(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0];
            }
        }

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (cols.original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q3_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,rows,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << 32 << "," << run << "," << rows << "," << total_revenue << ","
            << time_kernel << "," << throughput_gbps << ","
            << cols.compression_ratio << "," << (cols.original_size/1e6) << "," << (cols.compressed_size/1e6) << "\n";
        csv.close();
    }

    cols.orderdate.reset_device();
    cols.custkey.reset_device();
    cols.suppkey.reset_device();
    cols.revenue.reset_device();
}

void run_q33(const RunOptions& opt) {
    std::cout << "=== Q33 (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);
    CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << cols.orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_custkey:       " << cols.custkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_suppkey:       " << cols.suppkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_revenue:       " << cols.revenue.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (cols.original_size/1e6) << " MB -> " << (cols.compressed_size/1e6)
              << " MB (ratio: " << cols.compression_ratio << "x)" << std::endl;

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year    = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_city    = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_city    = load_column<int>(opt.data_dir, "c_city", C_LEN);

    DeviceBuffer<int> d_d_datekey(D_LEN);
    DeviceBuffer<int> d_d_year(D_LEN);
    DeviceBuffer<int> d_s_suppkey(S_LEN);
    DeviceBuffer<int> d_s_city(S_LEN);
    DeviceBuffer<int> d_c_custkey(C_LEN);
    DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK_ERROR(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK_ERROR(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK_ERROR(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK_ERROR(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

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
    DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    // Warmup (not timed / not reported).
    CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
    probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            cols.d_orderdate(),
            cols.d_custkey(),
            cols.d_suppkey(),
            cols.d_revenue(),
            static_cast<int>(LO_LEN),
            ht_s.data(),
            static_cast<int>(S_LEN),
            ht_c.data(),
            static_cast<int>(C_LEN),
            ht_d.data(),
            date_ht_len,
            kDateMin,
            d_res.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));

        timer.start();
        probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                cols.d_orderdate(),
                cols.d_custkey(),
                cols.d_suppkey(),
                cols.d_revenue(),
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK_ERROR(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0];
            }
        }

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (cols.original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q3_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,rows,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << 33 << "," << run << "," << rows << "," << total_revenue << ","
            << time_kernel << "," << throughput_gbps << ","
            << cols.compression_ratio << "," << (cols.original_size/1e6) << "," << (cols.compressed_size/1e6) << "\n";
        csv.close();
    }

    cols.orderdate.reset_device();
    cols.custkey.reset_device();
    cols.suppkey.reset_device();
    cols.revenue.reset_device();
}

void run_q34(const RunOptions& opt) {
    std::cout << "=== Q34 (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    const int date_ht_len = (kDateMax - kDateMin + 1);
    CudaEventTimer h2d_timer;
    h2d_timer.start();
    auto cols = load_and_encode_lo_q3(opt);

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << cols.orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_custkey:       " << cols.custkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_suppkey:       " << cols.suppkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_revenue:       " << cols.revenue.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (cols.original_size/1e6) << " MB -> " << (cols.compressed_size/1e6)
              << " MB (ratio: " << cols.compression_ratio << "x)" << std::endl;

    auto h_d_datekey = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year    = load_column<int>(opt.data_dir, "d_year", D_LEN);

    auto h_s_suppkey = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_city    = load_column<int>(opt.data_dir, "s_city", S_LEN);

    auto h_c_custkey = load_column<int>(opt.data_dir, "c_custkey", C_LEN);
    auto h_c_city    = load_column<int>(opt.data_dir, "c_city", C_LEN);

    DeviceBuffer<int> d_d_datekey(D_LEN);
    DeviceBuffer<int> d_d_year(D_LEN);
    DeviceBuffer<int> d_s_suppkey(S_LEN);
    DeviceBuffer<int> d_s_city(S_LEN);
    DeviceBuffer<int> d_c_custkey(C_LEN);
    DeviceBuffer<int> d_c_city(C_LEN);

    CUDA_CHECK_ERROR(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_city.data(), h_s_city.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_custkey.data(), h_c_custkey.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_c_city.data(), h_c_city.get(), C_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * date_ht_len);
    DeviceBuffer<int> ht_s(static_cast<std::size_t>(2) * S_LEN);
    DeviceBuffer<int> ht_c(static_cast<std::size_t>(2) * C_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK_ERROR(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * date_ht_len));
    CUDA_CHECK_ERROR(cudaMemset(ht_s.data(), 0, sizeof(int) * 2 * S_LEN));
    CUDA_CHECK_ERROR(cudaMemset(ht_c.data(), 0, sizeof(int) * 2 * C_LEN));

    build_hashtable_s_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_city.data(), d_s_suppkey.data(), static_cast<int>(S_LEN), ht_s.data(), static_cast<int>(S_LEN));

    build_hashtable_c_city_in_set<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((C_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_c_city.data(), d_c_custkey.data(), static_cast<int>(C_LEN), ht_c.data(), static_cast<int>(C_LEN));

    // Q34: d_yearmonth = Dec1997 => d_datekey in [19971201, 19971231]
    build_hashtable_d_datekey_range<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), 19971201, 19971231, ht_d.data(), date_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    const int res_size = 6 * kNumCities * kNumCities;
    DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 6);

    // Warmup (not timed / not reported).
    CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));
    probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            cols.d_orderdate(),
            cols.d_custkey(),
            cols.d_suppkey(),
            cols.d_revenue(),
            static_cast<int>(LO_LEN),
            ht_s.data(),
            static_cast<int>(S_LEN),
            ht_c.data(),
            static_cast<int>(C_LEN),
            ht_d.data(),
            date_ht_len,
            kDateMin,
            d_res.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 6));

        timer.start();
        probe_q32<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                cols.d_orderdate(),
                cols.d_custkey(),
                cols.d_suppkey(),
                cols.d_revenue(),
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_c.data(),
                static_cast<int>(C_LEN),
                ht_d.data(),
                date_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 6);
        CUDA_CHECK_ERROR(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 6, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 6] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 6 + 4])[0];
            }
        }

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (cols.original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q3_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,rows,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << 34 << "," << run << "," << rows << "," << total_revenue << ","
            << time_kernel << "," << throughput_gbps << ","
            << cols.compression_ratio << "," << (cols.original_size/1e6) << "," << (cols.compressed_size/1e6) << "\n";
        csv.close();
    }

    cols.orderdate.reset_device();
    cols.custkey.reset_device();
    cols.suppkey.reset_device();
    cols.revenue.reset_device();
}

} // namespace ssb
