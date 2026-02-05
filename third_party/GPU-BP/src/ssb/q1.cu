#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

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

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q1_sum_revenue_kernel(bp::DeviceColumn lo_orderdate,
                                      bp::DeviceColumn lo_discount,
                                      bp::DeviceColumn lo_quantity,
                                      bp::DeviceColumn lo_extendedprice,
                                      int              n,
                                      int              date_min,
                                      int              date_max,
                                      int              discount_min,
                                      int              discount_max,
                                      int              quantity_min,
                                      int              quantity_max,
                                      unsigned long long* out_sum) {
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

    long long local_sum = 0;
    const int tile_offset = static_cast<int>(blockIdx.x) * TILE_SIZE;

#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ++ITEM) {
        const int row = tile_offset + static_cast<int>(threadIdx.x) + (BLOCK_THREADS * ITEM);
        if (row < n) {
            const int orderdate     = static_cast<int>(bp::decode_u32(lo_orderdate, static_cast<uint32_t>(row)));
            const int discount      = static_cast<int>(bp::decode_u32(lo_discount, static_cast<uint32_t>(row)));
            const int quantity      = static_cast<int>(bp::decode_u32(lo_quantity, static_cast<uint32_t>(row)));
            const int extendedprice = static_cast<int>(bp::decode_u32(lo_extendedprice, static_cast<uint32_t>(row)));

            if (orderdate >= date_min && orderdate <= date_max && discount >= discount_min && discount <= discount_max &&
                quantity >= quantity_min && quantity <= quantity_max) {
                local_sum += static_cast<long long>(extendedprice) * static_cast<long long>(discount);
            }
        }
    }

    // Block reduce (BLOCK_THREADS is always 32 in our usage).
    __shared__ long long shared[32];
    const unsigned long long block_sum = static_cast<unsigned long long>(
        BlockSum<long long, BLOCK_THREADS, ITEMS_PER_THREAD>(local_sum, shared));
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, block_sum);
    }
}

void run_q1_impl(int query_id,
                 const RunOptions& opt,
                 int date_min,
                 int date_max,
                 int discount_min,
                 int discount_max,
                 int quantity_min,
                 int quantity_max) {
    std::cout << "=== Q" << query_id << " (GPU-BP - Single Layer Bit-Packing) ===" << std::endl;

    auto h_lo_orderdate     = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_discount      = load_column<int>(opt.data_dir, "lo_discount", LO_LEN);
    auto h_lo_quantity      = load_column<int>(opt.data_dir, "lo_quantity", LO_LEN);
    auto h_lo_extendedprice = load_column<int>(opt.data_dir, "lo_extendedprice", LO_LEN);

    auto enc_orderdate = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_orderdate.get()), LO_LEN);
    auto enc_discount  = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_discount.get()), LO_LEN);
    auto enc_quantity  = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_quantity.get()), LO_LEN);
    auto enc_extprice  = bp::encode_u32(reinterpret_cast<const uint32_t*>(h_lo_extendedprice.get()), LO_LEN);

    // 压缩率统计
    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = enc_orderdate.bytes.size() + enc_discount.bytes.size() +
                             enc_quantity.bytes.size() + enc_extprice.bytes.size();
    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    std::cout << "  Compression (GPU-BP single-layer bit-packing, no FOR/Delta/RLE):" << std::endl;
    std::cout << "    lo_orderdate:     " << enc_orderdate.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_discount:      " << enc_discount.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_quantity:      " << enc_quantity.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    lo_extendedprice: " << enc_extprice.bytes.size()/1e6 << " MB" << std::endl;
    std::cout << "    Total: " << (original_size/1e6) << " MB -> " << (compressed_size/1e6)
              << " MB (ratio: " << compression_ratio << "x)" << std::endl;

    CudaEventTimer h2d_timer;
    h2d_timer.start();
    enc_orderdate.upload();
    enc_discount.upload();
    enc_quantity.upload();
    enc_extprice.upload();
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    const float time_h2d = h2d_timer.stop();

    const bp::DeviceColumn d_orderdate = enc_orderdate.device_view();
    const bp::DeviceColumn d_discount  = enc_discount.device_view();
    const bp::DeviceColumn d_quantity  = enc_quantity.device_view();
    const bp::DeviceColumn d_extprice  = enc_extprice.device_view();

    DeviceBuffer<unsigned long long> d_sum(1);

    constexpr int BLOCK_THREADS = 32;
    constexpr int ITEMS_PER_THREAD = 32;
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    const int num_blocks = static_cast<int>((LO_LEN + TILE_SIZE - 1) / TILE_SIZE);

    // Warmup
    CUDA_CHECK_ERROR(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));
    q1_sum_revenue_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
        <<<num_blocks, BLOCK_THREADS>>>(d_orderdate,
                                        d_discount,
                                        d_quantity,
                                        d_extprice,
                                        static_cast<int>(LO_LEN),
                                        date_min,
                                        date_max,
                                        discount_min,
                                        discount_max,
                                        quantity_min,
                                        quantity_max,
                                        d_sum.data());
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    CudaEventTimer timer;
    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK_ERROR(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));

        // GPU-BP: 解压和查询是融合的（on-the-fly decompression）
        timer.start();
        q1_sum_revenue_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<num_blocks, BLOCK_THREADS>>>(d_orderdate,
                                            d_discount,
                                            d_quantity,
                                            d_extprice,
                                            static_cast<int>(LO_LEN),
                                            date_min,
                                            date_max,
                                            discount_min,
                                            discount_max,
                                            quantity_min,
                                            quantity_max,
                                            d_sum.data());
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        const float time_kernel = timer.stop();

        unsigned long long h_sum = 0;
        CUDA_CHECK_ERROR(cudaMemcpy(&h_sum, d_sum.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        // 以原始数据大小计算吞吐量（因为解压是融合的）
        double throughput_gbps = (original_size / 1e9) / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": revenue=" << h_sum
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV输出
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/GPU-BP/gpu_bp_q1_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,revenue,time_kernel_ms,throughput_gbps,compression_ratio,original_mb,compressed_mb\n";
        }
        csv << query_id << "," << run << "," << h_sum << ","
            << time_kernel << "," << throughput_gbps << ","
            << compression_ratio << "," << (original_size/1e6) << "," << (compressed_size/1e6) << "\n";
        csv.close();
    }

    enc_orderdate.reset_device();
    enc_discount.reset_device();
    enc_quantity.reset_device();
    enc_extprice.reset_device();
}

} // namespace

void run_q11(const RunOptions& opt) {
    // Q1.1:
    // where lo_orderdate >= 19930000 and lo_orderdate <= 19940000
    // and lo_discount between 1 and 3
    // and lo_quantity < 25
    run_q1_impl(11, opt, 19930000, 19940000, 1, 3, 0, 24);
}

void run_q12(const RunOptions& opt) {
    // Q1.2:
    // where lo_orderdate between 19940101 and 19940131
    // and lo_discount between 4 and 6
    // and lo_quantity between 26 and 35
    run_q1_impl(12, opt, 19940101, 19940131, 4, 6, 26, 35);
}

void run_q13(const RunOptions& opt) {
    // Q1.3:
    // where lo_orderdate between 19940206 and 19940212
    // and lo_discount between 5 and 7
    // and lo_quantity between 26 and 35
    run_q1_impl(13, opt, 19940206, 19940212, 5, 7, 26, 35);
}

} // namespace ssb
