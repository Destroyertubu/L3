#include <cstdint>
#include <iostream>
#include <fstream>
#include <stdexcept>

#include <chrono>
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

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q1_sum_revenue_kernel(const int* lo_orderdate,
                                      const int* lo_discount,
                                      const int* lo_quantity,
                                      const int* lo_extendedprice,
                                      int        n,
                                      int        date_min,
                                      int        date_max,
                                      int        discount_min,
                                      int        discount_max,
                                      int        quantity_min,
                                      int        quantity_max,
                                      unsigned long long* out_sum) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    long long local_sum = 0;
    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < n) {
            const int orderdate = lo_orderdate[row];
            const int discount = lo_discount[row];
            const int quantity = lo_quantity[row];
            const int extendedprice = lo_extendedprice[row];

            if (orderdate >= date_min && orderdate <= date_max && discount >= discount_min && discount <= discount_max &&
                quantity >= quantity_min && quantity <= quantity_max) {
                local_sum += static_cast<long long>(extendedprice) * static_cast<long long>(discount);
            }
        }
    }

    __shared__ long long shared[32];
    const unsigned long long block_sum =
        static_cast<unsigned long long>(BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(local_sum, shared));
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum);
    }
}

struct DecompressedPtrs {
    const int* orderdate = nullptr;
    const int* discount = nullptr;
    const int* quantity = nullptr;
    const int* extendedprice = nullptr;
};

DecompressedPtrs decompress_q1_cols(const planner::EncodedColumn& enc_orderdate,
                                   const planner::EncodedColumn& enc_discount,
                                   const planner::EncodedColumn& enc_quantity,
                                   const planner::EncodedColumn& enc_extprice,
                                   planner::DeviceBuffer<int>&   d_orderdate_out,
                                   planner::DeviceBuffer<int>&   d_discount_out,
                                   planner::DeviceBuffer<int>&   d_quantity_out,
                                   planner::DeviceBuffer<int>&   d_extprice_out,
                                   planner::DeltaWorkspace&      ws_orderdate,
                                   planner::DeltaWorkspace&      ws_discount,
                                   planner::DeltaWorkspace&      ws_quantity,
                                   planner::DeltaWorkspace&      ws_extprice,
                                   planner::RleWorkspace&        ws_orderdate_rle,
                                   planner::RleWorkspace&        ws_discount_rle,
                                   planner::RleWorkspace&        ws_quantity_rle,
                                   planner::RleWorkspace&        ws_extprice_rle) {
    DecompressedPtrs out;

    auto decode_one = [&](const planner::EncodedColumn& enc,
                          planner::DeviceBuffer<int>&   buf,
                          planner::DeltaWorkspace&      dws,
                          planner::RleWorkspace&        rws) -> const int* {
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

    out.orderdate = decode_one(enc_orderdate, d_orderdate_out, ws_orderdate, ws_orderdate_rle);
    out.discount = decode_one(enc_discount, d_discount_out, ws_discount, ws_discount_rle);
    out.quantity = decode_one(enc_quantity, d_quantity_out, ws_quantity, ws_quantity_rle);
    out.extendedprice = decode_one(enc_extprice, d_extprice_out, ws_extprice, ws_extprice_rle);
    return out;
}

void run_q1_impl(int query_id,
                 const RunOptions& opt,
                 int date_min,
                 int date_max,
                 int discount_min,
                 int discount_max,
                 int quantity_min,
                 int quantity_max) {
    std::cout << "=== Q" << query_id << " (Planner - Cascading Decompression) ===" << std::endl;

    // 加载数据
    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_discount = load_column<int>(opt.data_dir, "lo_discount", LO_LEN);
    auto h_lo_quantity = load_column<int>(opt.data_dir, "lo_quantity", LO_LEN);
    auto h_lo_extendedprice = load_column<int>(opt.data_dir, "lo_extendedprice", LO_LEN);

    // CPU压缩
    auto enc_orderdate = planner::encode_column_planner(h_lo_orderdate.get(), LO_LEN);
    auto enc_discount = planner::encode_column_planner(h_lo_discount.get(), LO_LEN);
    auto enc_quantity = planner::encode_column_planner(h_lo_quantity.get(), LO_LEN);
    auto enc_extprice = planner::encode_column_planner(h_lo_extendedprice.get(), LO_LEN);

    // 计算压缩率
    size_t original_size = LO_LEN * sizeof(int) * 4;  // 4列原始大小
    size_t compressed_size = enc_orderdate.bytes.size() + enc_discount.bytes.size() +
                             enc_quantity.bytes.size() + enc_extprice.bytes.size();
    // 如果是Uncompressed，bytes为空，使用原始大小
    if (enc_orderdate.scheme == planner::Scheme::Uncompressed) compressed_size += LO_LEN * sizeof(int);
    else compressed_size += enc_orderdate.bytes.size() > 0 ? 0 : LO_LEN * enc_orderdate.byte_width;
    if (enc_discount.scheme == planner::Scheme::Uncompressed) compressed_size += LO_LEN * sizeof(int);
    if (enc_quantity.scheme == planner::Scheme::Uncompressed) compressed_size += LO_LEN * sizeof(int);
    if (enc_extprice.scheme == planner::Scheme::Uncompressed) compressed_size += LO_LEN * sizeof(int);

    // 重新计算压缩大小
    compressed_size = 0;
    compressed_size += (enc_orderdate.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_orderdate.bytes.size();
    compressed_size += (enc_discount.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_discount.bytes.size();
    compressed_size += (enc_quantity.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_quantity.bytes.size();
    compressed_size += (enc_extprice.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_extprice.bytes.size();

    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    std::cout << "  Compression:" << std::endl;
    std::cout << "    lo_orderdate:     " << planner::scheme_name(enc_orderdate.scheme)
              << " (bw=" << enc_orderdate.byte_width << ", "
              << (enc_orderdate.bytes.size() / 1e6) << " MB)" << std::endl;
    std::cout << "    lo_discount:      " << planner::scheme_name(enc_discount.scheme)
              << " (bw=" << enc_discount.byte_width << ", "
              << (enc_discount.bytes.size() / 1e6) << " MB)" << std::endl;
    std::cout << "    lo_quantity:      " << planner::scheme_name(enc_quantity.scheme)
              << " (bw=" << enc_quantity.byte_width << ", "
              << (enc_quantity.bytes.size() / 1e6) << " MB)" << std::endl;
    std::cout << "    lo_extendedprice: " << planner::scheme_name(enc_extprice.scheme)
              << " (bw=" << enc_extprice.byte_width << ", "
              << (enc_extprice.bytes.size() / 1e6) << " MB)" << std::endl;
    std::cout << "    Total: " << (original_size/1e6) << " MB -> " << (compressed_size/1e6)
              << " MB (ratio: " << compression_ratio << "x)" << std::endl;

    // 上传到GPU
    enc_orderdate.upload(h_lo_orderdate.get());
    enc_discount.upload(h_lo_discount.get());
    enc_quantity.upload(h_lo_quantity.get());
    enc_extprice.upload(h_lo_extendedprice.get());

    planner::DeviceBuffer<int> d_orderdate;
    planner::DeviceBuffer<int> d_discount;
    planner::DeviceBuffer<int> d_quantity;
    planner::DeviceBuffer<int> d_extprice;
    planner::DeltaWorkspace ws_orderdate, ws_discount, ws_quantity, ws_extprice;
    planner::RleWorkspace ws_orderdate_rle, ws_discount_rle, ws_quantity_rle, ws_extprice_rle;

    planner::DeviceBuffer<unsigned long long> d_sum(1);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 32;
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    const int num_blocks = static_cast<int>((LO_LEN + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    {
        auto ptrs = decompress_q1_cols(enc_orderdate, enc_discount, enc_quantity, enc_extprice,
                                       d_orderdate, d_discount, d_quantity, d_extprice,
                                       ws_orderdate, ws_discount, ws_quantity, ws_extprice,
                                       ws_orderdate_rle, ws_discount_rle, ws_quantity_rle, ws_extprice_rle);
        cudaDeviceSynchronize();
        CUDA_CHECK(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));
        q1_sum_revenue_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<num_blocks, BLOCK_THREADS>>>(ptrs.orderdate, ptrs.discount, ptrs.quantity, ptrs.extendedprice,
                                            static_cast<int>(LO_LEN), date_min, date_max,
                                            discount_min, discount_max, quantity_min, quantity_max, d_sum.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    planner::CudaEventTimer timer;

    for (int run = 0; run < opt.runs; ++run) {
        // 清零buffer强制重新解压
        if (d_orderdate.size() > 0) cudaMemset(d_orderdate.data(), 0, d_orderdate.size() * sizeof(int));
        if (d_discount.size() > 0) cudaMemset(d_discount.data(), 0, d_discount.size() * sizeof(int));
        if (d_quantity.size() > 0) cudaMemset(d_quantity.data(), 0, d_quantity.size() * sizeof(int));
        if (d_extprice.size() > 0) cudaMemset(d_extprice.data(), 0, d_extprice.size() * sizeof(int));
        CUDA_CHECK(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));
        cudaDeviceSynchronize();

        // ========== Decompress + Query 一起计时 ==========
        timer.start();

        // Pass 1-N: Cascading Decompression (多次global memory读写)
        auto ptrs = decompress_q1_cols(enc_orderdate, enc_discount, enc_quantity, enc_extprice,
                                       d_orderdate, d_discount, d_quantity, d_extprice,
                                       ws_orderdate, ws_discount, ws_quantity, ws_extprice,
                                       ws_orderdate_rle, ws_discount_rle, ws_quantity_rle, ws_extprice_rle);

        // Final Pass: Query Kernel (再次读取global memory)
        q1_sum_revenue_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<num_blocks, BLOCK_THREADS>>>(ptrs.orderdate, ptrs.discount, ptrs.quantity, ptrs.extendedprice,
                                            static_cast<int>(LO_LEN), date_min, date_max,
                                            discount_min, discount_max, quantity_min, quantity_max, d_sum.data());

        cudaDeviceSynchronize();
        float time_decomp_query = timer.stop();

        unsigned long long h_sum = 0;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // 计算吞吐量
        double data_gb = original_size / 1e9;
        double throughput_gbps = data_gb / (time_decomp_query / 1000.0);

        std::cout << "  Run " << run << ": revenue=" << h_sum
                  << ", decomp+query=" << time_decomp_query << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/Planner/q1_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,revenue,time_decomp_query_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << query_id << "," << run << "," << h_sum << ","
            << time_decomp_query << "," << throughput_gbps << ","
            << compression_ratio << "," << (original_size/1e6) << "," << (compressed_size/1e6) << "\n";
        csv.close();
    }
}

} // namespace

void run_q11(const RunOptions& opt) {
    // Q1.1: orderdate (19930000,19940000), discount [1,3], quantity < 25
    run_q1_impl(11, opt, 19930000, 19940000, 1, 3, 0, 24);
}

void run_q12(const RunOptions& opt) {
    // Q1.2: orderdate [19940101,19940131], discount [4,6], quantity [26,35]
    run_q1_impl(12, opt, 19940101, 19940131, 4, 6, 26, 35);
}

void run_q13(const RunOptions& opt) {
    // Q1.3: orderdate [19940206,19940212], discount [5,7], quantity [26,35]
    run_q1_impl(13, opt, 19940206, 19940212, 5, 7, 26, 35);
}

} // namespace ssb

