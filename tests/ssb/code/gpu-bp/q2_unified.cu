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

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_s_region(const int* s_region, const int* s_suppkey, int s_len, int region_filter, int* ht_s) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, s_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_region + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, region_filter, selection_flags, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(s_suppkey + tile_offset, items, num_items);
    BlockBuildSelectivePHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht_s, s_len, num_items);
}

// Q2.1: filter p_category=12, return p_brand1
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_cat_eq(const int* p_category,
                                        const int* p_partkey,
                                        const int* p_brand1,
                                        int        p_len,
                                        int        category_filter,
                                        int*       ht_p) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int items[ITEMS_PER_THREAD];
    int vals[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, p_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_category + tile_offset, items, num_items);
    BlockPredEQ<int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, category_filter, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_partkey + tile_offset, items, num_items);
    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_brand1 + tile_offset, vals, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(items, vals, selection_flags, ht_p, p_len, num_items);
}

// Q2.2/Q2.3: filter p_brand1 range/eq, return p_brand1
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void build_hashtable_p_brand_range(const int* p_brand1,
                                              const int* p_partkey,
                                              int        p_len,
                                              int        brand_min,
                                              int        brand_max,
                                              int*       ht_p) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    int brand[ITEMS_PER_THREAD];
    int partkey[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];

    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;
    const int num_items = min(TILE_SIZE, p_len - tile_offset);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_brand1 + tile_offset, brand, num_items);
    BlockPredGTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(brand, brand_min, selection_flags, num_items);
    BlockPredAndLTE<int, BLOCK_THREADS, ITEMS_PER_THREAD>(brand, brand_max, selection_flags, num_items);

    BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD>(p_partkey + tile_offset, partkey, num_items);
    BlockBuildSelectivePHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(partkey, brand, selection_flags, ht_p, p_len, num_items);
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
__global__ void probe_q2(bp::DeviceColumn lo_orderdate,
                         bp::DeviceColumn lo_partkey,
                         bp::DeviceColumn lo_suppkey,
                         bp::DeviceColumn lo_revenue,
                         int              lo_len,
                         const int*       ht_s,
                         int              s_len,
                         const int*       ht_p,
                         int              p_len,
                         const int*       ht_d,
                         int              d_ht_len,
                         int              d_keys_min,
                         int*             res) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    int suppkey[ITEMS_PER_THREAD];
    int partkey[ITEMS_PER_THREAD];
    int orderdate[ITEMS_PER_THREAD];
    int selection_flags[ITEMS_PER_THREAD];
    int brand[ITEMS_PER_THREAD];
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
    BlockProbeAndPHT_1<int, BLOCK_THREADS, ITEMS_PER_THREAD>(suppkey, selection_flags, ht_s, s_len, num_items);

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
        partkey[ITEM] = (row < lo_len) ? static_cast<int>(bp::decode_u32(lo_partkey, static_cast<uint32_t>(row))) : 0;
    }
    BlockProbeAndPHT_2<int, int, BLOCK_THREADS, ITEMS_PER_THREAD>(partkey, brand, selection_flags, ht_p, p_len, num_items);

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
            const int hash = (brand[ITEM] * 7 + (year[ITEM] - 1992)) % (7 * 1000);
            res[hash * 4]     = year[ITEM];
            res[hash * 4 + 1] = brand[ITEM];
            atomicAdd(reinterpret_cast<unsigned long long*>(&res[hash * 4 + 2]), static_cast<unsigned long long>(revenue));
        }
    }
}


void run_q2_impl(int query_id, const RunOptions& opt, int s_region_filter, int p_mode, int p_a, int p_b) {
    std::cout << "=== Q" << query_id << " (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_partkey   = load_column<int>(opt.data_dir, "lo_partkey", LO_LEN);
    auto h_lo_suppkey   = load_column<int>(opt.data_dir, "lo_suppkey", LO_LEN);
    auto h_lo_revenue   = load_column<int>(opt.data_dir, "lo_revenue", LO_LEN);

    auto enc_orderdate = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_orderdate.get()), LO_LEN);
    auto enc_partkey   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_partkey.get()), LO_LEN);
    auto enc_suppkey   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_suppkey.get()), LO_LEN);
    auto enc_revenue   = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_revenue.get()), LO_LEN);

    // 压缩率统计
    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = enc_orderdate.bytes.size() + enc_partkey.bytes.size() +
                             enc_suppkey.bytes.size() + enc_revenue.bytes.size();
    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << enc_orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_partkey:       " << enc_partkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_suppkey:       " << enc_suppkey.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_revenue:       " << enc_revenue.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (original_size/1e6) << " MB -> " << (compressed_size/1e6)
              << " MB (ratio: " << compression_ratio << "x)" << std::endl;

    CudaEventTimer h2d_timer;
    h2d_timer.start();
    enc_orderdate.upload();
    enc_partkey.upload();
    enc_suppkey.upload();
    enc_revenue.upload();

    const bp::DeviceColumn d_orderdate = enc_orderdate.device_view();
    const bp::DeviceColumn d_partkey   = enc_partkey.device_view();
    const bp::DeviceColumn d_suppkey   = enc_suppkey.device_view();
    const bp::DeviceColumn d_revenue   = enc_revenue.device_view();

    // Dimension tables (uncompressed)
    auto h_p_partkey  = load_column<int>(opt.data_dir, "p_partkey", P_LEN);
    auto h_p_category = load_column<int>(opt.data_dir, "p_category", P_LEN);
    auto h_p_brand1   = load_column<int>(opt.data_dir, "p_brand1", P_LEN);
    auto h_s_suppkey  = load_column<int>(opt.data_dir, "s_suppkey", S_LEN);
    auto h_s_region   = load_column<int>(opt.data_dir, "s_region", S_LEN);
    auto h_d_datekey  = load_column<int>(opt.data_dir, "d_datekey", D_LEN);
    auto h_d_year     = load_column<int>(opt.data_dir, "d_year", D_LEN);

    DeviceBuffer<int> d_p_partkey(P_LEN);
    DeviceBuffer<int> d_p_category(P_LEN);
    DeviceBuffer<int> d_p_brand1(P_LEN);
    DeviceBuffer<int> d_s_suppkey(S_LEN);
    DeviceBuffer<int> d_s_region(S_LEN);
    DeviceBuffer<int> d_d_datekey(D_LEN);
    DeviceBuffer<int> d_d_year(D_LEN);

    CUDA_CHECK_ERROR(cudaMemcpy(d_p_partkey.data(), h_p_partkey.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_p_category.data(), h_p_category.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_p_brand1.data(), h_p_brand1.get(), P_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_suppkey.data(), h_s_suppkey.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_s_region.data(), h_s_region.get(), S_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_datekey.data(), h_d_datekey.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_d_year.data(), h_d_year.get(), D_LEN * sizeof(int), cudaMemcpyHostToDevice));
    const float time_h2d = h2d_timer.stop();

    // Hash tables
    const int d_ht_len = (kDateMax - kDateMin + 1);
    DeviceBuffer<int> ht_d(static_cast<std::size_t>(2) * d_ht_len);
    DeviceBuffer<int> ht_p(static_cast<std::size_t>(2) * P_LEN);
    DeviceBuffer<int> ht_s(S_LEN);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 8;
    constexpr int TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Build HTs once, reuse for runs.
    CudaEventTimer ht_timer;
    ht_timer.start();
    CUDA_CHECK_ERROR(cudaMemset(ht_d.data(), 0, sizeof(int) * 2 * d_ht_len));
    CUDA_CHECK_ERROR(cudaMemset(ht_p.data(), 0, sizeof(int) * 2 * P_LEN));
    CUDA_CHECK_ERROR(cudaMemset(ht_s.data(), 0, sizeof(int) * S_LEN));

    build_hashtable_s_region<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((S_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_s_region.data(), d_s_suppkey.data(), static_cast<int>(S_LEN), s_region_filter, ht_s.data());

    if (p_mode == 1) {
        build_hashtable_p_cat_eq<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((P_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                d_p_category.data(), d_p_partkey.data(), d_p_brand1.data(), static_cast<int>(P_LEN), p_a, ht_p.data());
    } else if (p_mode == 2) {
        build_hashtable_p_brand_range<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((P_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                d_p_brand1.data(), d_p_partkey.data(), static_cast<int>(P_LEN), p_a, p_b, ht_p.data());
    } else {
        throw std::runtime_error("run_q2_impl: unknown p_mode");
    }

    build_hashtable_d_all<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((D_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_d_datekey.data(), d_d_year.data(), static_cast<int>(D_LEN), ht_d.data(), d_ht_len, kDateMin);
    const float time_ht_build = ht_timer.stop();

    // Result: 7 years * 1000 brands, 4 ints per bucket (year, brand, long long revenue).
    const int res_size = 7 * 1000;
    DeviceBuffer<int> d_res(static_cast<std::size_t>(res_size) * 4);

    // Warmup (not timed / not reported).
    CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 4));
    probe_q2<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
            d_orderdate,
            d_partkey,
            d_suppkey,
            d_revenue,
            static_cast<int>(LO_LEN),
            ht_s.data(),
            static_cast<int>(S_LEN),
            ht_p.data(),
            static_cast<int>(P_LEN),
            ht_d.data(),
            d_ht_len,
            kDateMin,
            d_res.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_res.data(), 0, sizeof(int) * res_size * 4));

        timer.start();
        probe_q2<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<static_cast<int>((LO_LEN + TILE_ITEMS - 1) / TILE_ITEMS), BLOCK_THREADS>>>(
                d_orderdate,
                d_partkey,
                d_suppkey,
                d_revenue,
                static_cast<int>(LO_LEN),
                ht_s.data(),
                static_cast<int>(S_LEN),
                ht_p.data(),
                static_cast<int>(P_LEN),
                ht_d.data(),
                d_ht_len,
                kDateMin,
                d_res.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        std::vector<int> h_res(static_cast<std::size_t>(res_size) * 4);
        CUDA_CHECK_ERROR(cudaMemcpy(h_res.data(), d_res.data(), sizeof(int) * res_size * 4, cudaMemcpyDeviceToHost));

        unsigned long long total_revenue = 0;
        int rows = 0;
        for (int i = 0; i < res_size; ++i) {
            if (h_res[i * 4] != 0) {
                ++rows;
                total_revenue += reinterpret_cast<const unsigned long long*>(&h_res[i * 4 + 2])[0];
            }
        }

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": rows=" << rows << ", revenue=" << total_revenue
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q2_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,rows,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << query_id << "," << run << "," << rows << "," << total_revenue << ","
            << time_kernel << "," << throughput_gbps << ","
            << compression_ratio << "," << (original_size/1e6) << "," << (compressed_size/1e6) << "\n";
        csv.close();
    }

    enc_orderdate.reset_device();
    enc_partkey.reset_device();
    enc_suppkey.reset_device();
    enc_revenue.reset_device();
}

} // namespace

void run_q21(const RunOptions& opt) {
    // Q2.1: s_region=AMERICA(1), p_category=MFGR#12(12)
    run_q2_impl(21, opt, /*s_region*/ 1, /*p_mode*/ 1, /*p_category*/ 12, /*unused*/ 0);
}

void run_q22(const RunOptions& opt) {
    // Q2.2: s_region=ASIA(2), p_brand1 BETWEEN 260 AND 267
    run_q2_impl(22, opt, /*s_region*/ 2, /*p_mode*/ 2, /*brand_min*/ 260, /*brand_max*/ 267);
}

void run_q23(const RunOptions& opt) {
    // Q2.3: s_region=EUROPE(3), p_brand1 = 278
    run_q2_impl(23, opt, /*s_region*/ 3, /*p_mode*/ 2, /*brand_min*/ 278, /*brand_max*/ 278);
}

} // namespace ssb
