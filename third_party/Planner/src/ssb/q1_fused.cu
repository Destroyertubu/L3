/**
 * Planner SSB Q1 Fused Implementation
 *
 * Uses fused decompression + query kernel with:
 * - Early Exit: skip columns when no valid tuples in warp
 * - Late Materialization: only decode aggregation column for valid rows
 *
 * For DELTA/RLE compressed columns, pre-decode to buffer then use in fused kernel.
 * For NS/FOR_NS columns, decode inline in the kernel.
 */

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
#include "planner/fused_decode.cuh"
#include "ssb/crystal_like.cuh"
#include "ssb/queries.hpp"
#include "ssb/ssb_dataset.hpp"

namespace ssb {
namespace {

// Pre-decode columns that require prefix sum (DELTA/RLE)
// Returns pointer to decoded data (either original d_ints or newly decoded buffer)
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
            // These can be decoded inline in fused kernel
            return nullptr;  // Signal to use inline decode

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

void run_q1_fused_impl(int query_id,
                       const RunOptions& opt,
                       int date_min, int date_max,
                       int discount_min, int discount_max,
                       int quantity_min, int quantity_max)
{
    std::cout << "=== Q" << query_id << " (Planner - Fused Decompression) ===" << std::endl;

    // Load data
    auto h_lo_orderdate = load_column<int>(opt.data_dir, "lo_orderdate", LO_LEN);
    auto h_lo_discount = load_column<int>(opt.data_dir, "lo_discount", LO_LEN);
    auto h_lo_quantity = load_column<int>(opt.data_dir, "lo_quantity", LO_LEN);
    auto h_lo_extendedprice = load_column<int>(opt.data_dir, "lo_extendedprice", LO_LEN);

    // Compress
    auto enc_orderdate = planner::encode_column_planner(h_lo_orderdate.get(), LO_LEN);
    auto enc_discount = planner::encode_column_planner(h_lo_discount.get(), LO_LEN);
    auto enc_quantity = planner::encode_column_planner(h_lo_quantity.get(), LO_LEN);
    auto enc_extprice = planner::encode_column_planner(h_lo_extendedprice.get(), LO_LEN);

    // Calculate compression ratio
    size_t original_size = LO_LEN * sizeof(int) * 4;
    size_t compressed_size = 0;
    compressed_size += (enc_orderdate.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_orderdate.bytes.size();
    compressed_size += (enc_discount.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_discount.bytes.size();
    compressed_size += (enc_quantity.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_quantity.bytes.size();
    compressed_size += (enc_extprice.scheme == planner::Scheme::Uncompressed) ? LO_LEN * 4 : enc_extprice.bytes.size();
    if (enc_orderdate.scheme == planner::Scheme::RLE) compressed_size += enc_orderdate.rle_values.size() * 8;
    if (enc_discount.scheme == planner::Scheme::RLE) compressed_size += enc_discount.rle_values.size() * 8;
    if (enc_quantity.scheme == planner::Scheme::RLE) compressed_size += enc_quantity.rle_values.size() * 8;
    if (enc_extprice.scheme == planner::Scheme::RLE) compressed_size += enc_extprice.rle_values.size() * 8;

    double compression_ratio = static_cast<double>(original_size) / std::max(compressed_size, size_t(1));

    std::cout << "  Compression schemes:" << std::endl;
    std::cout << "    lo_orderdate:     " << planner::scheme_name(enc_orderdate.scheme) << std::endl;
    std::cout << "    lo_discount:      " << planner::scheme_name(enc_discount.scheme) << std::endl;
    std::cout << "    lo_quantity:      " << planner::scheme_name(enc_quantity.scheme) << std::endl;
    std::cout << "    lo_extendedprice: " << planner::scheme_name(enc_extprice.scheme) << std::endl;

    // Upload to GPU
    enc_orderdate.upload(h_lo_orderdate.get());
    enc_discount.upload(h_lo_discount.get());
    enc_quantity.upload(h_lo_quantity.get());
    enc_extprice.upload(h_lo_extendedprice.get());

    // Buffers for pre-decoding DELTA/RLE columns
    planner::DeviceBuffer<int> d_orderdate_dec, d_discount_dec, d_quantity_dec, d_extprice_dec;
    planner::DeltaWorkspace ws_od, ws_disc, ws_qty, ws_ext;
    planner::RleWorkspace rws_od, rws_disc, rws_qty, rws_ext;

    planner::DeviceBuffer<unsigned long long> d_sum(1);

    constexpr int BLOCK_THREADS = 128;
    constexpr int ITEMS_PER_THREAD = 4;
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    const int num_blocks = (LO_LEN + TILE_SIZE - 1) / TILE_SIZE;

    // Warmup
    {
        const int* p_od = prepare_column_for_fused(enc_orderdate, d_orderdate_dec, ws_od, rws_od);
        const int* p_disc = prepare_column_for_fused(enc_discount, d_discount_dec, ws_disc, rws_disc);
        const int* p_qty = prepare_column_for_fused(enc_quantity, d_quantity_dec, ws_qty, rws_qty);
        const int* p_ext = prepare_column_for_fused(enc_extprice, d_extprice_dec, ws_ext, rws_ext);
        cudaDeviceSynchronize();

        auto acc_od = make_accessor(enc_orderdate, p_od);
        auto acc_disc = make_accessor(enc_discount, p_disc);
        auto acc_qty = make_accessor(enc_quantity, p_qty);
        auto acc_ext = make_accessor(enc_extprice, p_ext);

        CUDA_CHECK(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));
        planner::fused::q1_fused_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<num_blocks, BLOCK_THREADS>>>(
                acc_od, acc_disc, acc_qty, acc_ext,
                LO_LEN, date_min, date_max, discount_min, discount_max,
                quantity_min, quantity_max, d_sum.data());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    planner::CudaEventTimer timer;

    for (int run = 0; run < opt.runs; ++run) {
        CUDA_CHECK(cudaMemset(d_sum.data(), 0, sizeof(unsigned long long)));

        // ========== Single fused kernel timing ==========
        timer.start();

        // Pre-decode DELTA/RLE columns (if any) - this is part of "fused" time
        // because NS/FOR_NS columns don't need this step
        const int* p_od = prepare_column_for_fused(enc_orderdate, d_orderdate_dec, ws_od, rws_od);
        const int* p_disc = prepare_column_for_fused(enc_discount, d_discount_dec, ws_disc, rws_disc);
        const int* p_qty = prepare_column_for_fused(enc_quantity, d_quantity_dec, ws_qty, rws_qty);
        const int* p_ext = prepare_column_for_fused(enc_extprice, d_extprice_dec, ws_ext, rws_ext);

        auto acc_od = make_accessor(enc_orderdate, p_od);
        auto acc_disc = make_accessor(enc_discount, p_disc);
        auto acc_qty = make_accessor(enc_quantity, p_qty);
        auto acc_ext = make_accessor(enc_extprice, p_ext);

        // Fused query kernel with Early Exit + Late Materialization
        planner::fused::q1_fused_kernel<BLOCK_THREADS, ITEMS_PER_THREAD>
            <<<num_blocks, BLOCK_THREADS>>>(
                acc_od, acc_disc, acc_qty, acc_ext,
                LO_LEN, date_min, date_max, discount_min, discount_max,
                quantity_min, quantity_max, d_sum.data());

        cudaDeviceSynchronize();
        float time_kernel = timer.stop();

        unsigned long long h_sum = 0;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum.data(), sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        double data_gb = original_size / 1e9;
        double throughput_gbps = data_gb / (time_kernel / 1000.0);

        std::cout << "  Run " << run << ": revenue=" << h_sum
                  << ", kernel=" << time_kernel << " ms"
                  << ", throughput=" << throughput_gbps << " GB/s" << std::endl;

        // CSV output
        std::string csv_path = "/root/autodl-tmp/code/L3/H20/ssb/Planner/q1_fused_results.csv";
        bool file_exists = std::ifstream(csv_path).good();
        std::ofstream csv(csv_path, std::ios::app);
        if (!file_exists) {
            csv << "query,run,revenue,time_kernel_ms,throughput_gbps,compression_ratio\n";
        }
        csv << query_id << "," << run << "," << h_sum << ","
            << time_kernel << "," << throughput_gbps << "," << compression_ratio << "\n";
        csv.close();
    }
}

} // namespace

void run_q11_fused(const RunOptions& opt) {
    run_q1_fused_impl(11, opt, 19930000, 19940000, 1, 3, 0, 24);
}

void run_q12_fused(const RunOptions& opt) {
    run_q1_fused_impl(12, opt, 19940101, 19940131, 4, 6, 26, 35);
}

void run_q13_fused(const RunOptions& opt) {
    run_q1_fused_impl(13, opt, 19940206, 19940212, 5, 7, 26, 35);
}

} // namespace ssb
