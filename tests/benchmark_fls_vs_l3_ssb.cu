/**
 * @file benchmark_fls_vs_l3_ssb.cu
 * @brief Benchmark Vertical GPU vs L3 compression/decompression on SSB columns
 *
 * Tests:
 * 1. Vertical GPU: Pure bitpack (like crystal-opt uses)
 * 2. L3: FOR + interleaved delta encoding
 *
 * Columns tested:
 * - lo_orderdate (narrow range)
 * - lo_quantity (very narrow: 1-50)
 * - lo_discount (very narrow: 0-10)
 * - lo_extendedprice (wide range)
 * - lo_revenue (wide range)
 * - lo_supplycost (medium range)
 * - lo_custkey (foreign key)
 * - lo_partkey (foreign key)
 * - lo_suppkey (foreign key)
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <numeric>

// L3 headers (not needed for this standalone benchmark)
// #include "L3_Vertical_format.hpp"

// ============================================================================
// Vertical GPU Kernels (from pack.cpp translated to CUDA)
// ============================================================================

// Simplified Vertical-style unpack kernel
__global__ void fls_unpack_kernel(
    const uint32_t* __restrict__ packed_data,
    const uint32_t* __restrict__ base_values,
    const uint8_t* __restrict__ bit_widths,
    uint32_t* __restrict__ output,
    int num_blocks,
    int block_size)
{
    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    int tid = threadIdx.x;
    uint32_t base = base_values[block_idx];
    uint8_t bw = bit_widths[block_idx];

    // Calculate packed data offset (simplified)
    int packed_words_per_block = (block_size * bw + 31) / 32;
    const uint32_t* block_packed = packed_data + block_idx * packed_words_per_block;

    // Unpack each value
    for (int i = tid; i < block_size; i += blockDim.x) {
        int bit_offset = i * bw;
        int word_idx = bit_offset / 32;
        int bit_pos = bit_offset % 32;

        uint32_t val = (block_packed[word_idx] >> bit_pos);
        if (bit_pos + bw > 32 && word_idx + 1 < packed_words_per_block) {
            val |= (block_packed[word_idx + 1] << (32 - bit_pos));
        }
        val &= ((1u << bw) - 1);

        output[block_idx * block_size + i] = base + val;
    }
}

// L3 interleaved unpack kernel (matches V5 format)
__global__ void l3_interleaved_unpack_kernel(
    const uint32_t* __restrict__ interleaved_data,
    const uint32_t* __restrict__ base_values,
    const uint8_t* __restrict__ bit_widths,
    uint32_t* __restrict__ output,
    int num_partitions,
    int partition_size)
{
    int part_idx = blockIdx.x;
    if (part_idx >= num_partitions) return;

    int lane = threadIdx.x;
    uint32_t base = base_values[part_idx];
    uint8_t bw = bit_widths[part_idx];

    // Each partition is interleaved: 4 mini-vectors of 256 values = 1024 values
    // Mini-vector = 32 threads × 8 values
    constexpr int MINI_VEC = 256;
    constexpr int MVS_PER_PART = 4;
    constexpr int WARP_SZ = 32;

    int64_t part_bit_base = static_cast<int64_t>(part_idx) * partition_size * bw;

    for (int mv = 0; mv < MVS_PER_PART; mv++) {
        int64_t mv_bit_base = part_bit_base + static_cast<int64_t>(mv) * MINI_VEC * bw;

        // Extract 8 values per thread
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            int idx_in_mv = v * WARP_SZ + lane;
            int64_t bit_offset = mv_bit_base + static_cast<int64_t>(idx_in_mv) * bw;
            int word_idx = bit_offset / 32;
            int bit_pos = bit_offset % 32;

            uint32_t val = (interleaved_data[word_idx] >> bit_pos);
            if (bit_pos + bw > 32) {
                val |= (interleaved_data[word_idx + 1] << (32 - bit_pos));
            }
            val &= ((1u << bw) - 1);

            int global_idx = part_idx * partition_size + mv * MINI_VEC + idx_in_mv;
            output[global_idx] = base + val;
        }
    }
}

// ============================================================================
// Simple FOR Compression (CPU - for benchmark prep)
// ============================================================================

struct CompressedColumn {
    std::vector<uint32_t> packed_data;
    std::vector<uint32_t> base_values;
    std::vector<uint8_t> bit_widths;
    int num_blocks;
    int block_size;
    size_t original_elements;
};

CompressedColumn compressFOR(const std::vector<uint32_t>& data, int block_size) {
    CompressedColumn result;
    result.block_size = block_size;
    result.original_elements = data.size();
    result.num_blocks = (data.size() + block_size - 1) / block_size;

    result.base_values.resize(result.num_blocks);
    result.bit_widths.resize(result.num_blocks);

    size_t total_packed_words = 0;

    // First pass: calculate metadata
    for (int b = 0; b < result.num_blocks; b++) {
        int start = b * block_size;
        int end = std::min(start + block_size, (int)data.size());

        uint32_t min_val = data[start];
        uint32_t max_val = data[start];
        for (int i = start + 1; i < end; i++) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }

        result.base_values[b] = min_val;
        uint32_t range = max_val - min_val;
        uint8_t bw = (range == 0) ? 1 : (32 - __builtin_clz(range));
        if (bw == 0) bw = 1;
        result.bit_widths[b] = bw;

        int packed_words = (block_size * bw + 31) / 32;
        total_packed_words += packed_words;
    }

    // Second pass: pack data
    result.packed_data.resize(total_packed_words, 0);

    size_t packed_offset = 0;
    for (int b = 0; b < result.num_blocks; b++) {
        int start = b * block_size;
        int end = std::min(start + block_size, (int)data.size());
        uint32_t base = result.base_values[b];
        uint8_t bw = result.bit_widths[b];

        int packed_words = (block_size * bw + 31) / 32;

        for (int i = start; i < end; i++) {
            uint32_t delta = data[i] - base;
            int idx_in_block = i - start;
            int bit_offset = idx_in_block * bw;
            int word_idx = bit_offset / 32;
            int bit_pos = bit_offset % 32;

            result.packed_data[packed_offset + word_idx] |= (delta << bit_pos);
            if (bit_pos + bw > 32) {
                result.packed_data[packed_offset + word_idx + 1] |= (delta >> (32 - bit_pos));
            }
        }

        packed_offset += packed_words;
    }

    return result;
}

// ============================================================================
// L3 Interleaved Compression (matches L3 format)
// ============================================================================

struct L3CompressedColumn {
    std::vector<uint32_t> interleaved_data;
    std::vector<uint32_t> base_values;
    std::vector<uint8_t> bit_widths;
    int num_partitions;
    int partition_size;
    size_t original_elements;
};

L3CompressedColumn compressL3Interleaved(const std::vector<uint32_t>& data, int partition_size = 1024) {
    L3CompressedColumn result;
    result.partition_size = partition_size;
    result.original_elements = data.size();
    result.num_partitions = (data.size() + partition_size - 1) / partition_size;

    result.base_values.resize(result.num_partitions);
    result.bit_widths.resize(result.num_partitions);

    // Calculate total bits needed
    size_t total_bits = 0;
    for (int p = 0; p < result.num_partitions; p++) {
        int start = p * partition_size;
        int end = std::min(start + partition_size, (int)data.size());

        uint32_t min_val = data[start];
        uint32_t max_val = data[start];
        for (int i = start + 1; i < end; i++) {
            min_val = std::min(min_val, data[i]);
            max_val = std::max(max_val, data[i]);
        }

        result.base_values[p] = min_val;
        uint32_t range = max_val - min_val;
        uint8_t bw = (range == 0) ? 1 : (32 - __builtin_clz(range));
        if (bw == 0) bw = 1;
        result.bit_widths[p] = bw;

        total_bits += static_cast<size_t>(partition_size) * bw;
    }

    size_t total_words = (total_bits + 31) / 32;
    result.interleaved_data.resize(total_words, 0);

    // Pack data
    for (int p = 0; p < result.num_partitions; p++) {
        int start = p * partition_size;
        int end = std::min(start + partition_size, (int)data.size());
        uint32_t base = result.base_values[p];
        uint8_t bw = result.bit_widths[p];

        int64_t part_bit_base = static_cast<int64_t>(p) * partition_size * bw;

        for (int i = start; i < end; i++) {
            uint32_t delta = data[i] - base;
            int idx_in_part = i - start;
            int64_t bit_offset = part_bit_base + static_cast<int64_t>(idx_in_part) * bw;
            int word_idx = bit_offset / 32;
            int bit_pos = bit_offset % 32;

            result.interleaved_data[word_idx] |= (delta << bit_pos);
            if (bit_pos + bw > 32 && word_idx + 1 < (int)result.interleaved_data.size()) {
                result.interleaved_data[word_idx + 1] |= (delta >> (32 - bit_pos));
            }
        }
    }

    return result;
}

// ============================================================================
// Benchmark Results
// ============================================================================

struct ColumnBenchmark {
    std::string column_name;
    size_t num_elements;
    double original_mb;

    // Data characteristics
    uint32_t min_val;
    uint32_t max_val;
    double avg_bit_width;

    // Vertical style (block_size=128)
    double fls_compressed_mb;
    double fls_ratio;
    double fls_decompress_ms;
    double fls_throughput_gbps;

    // L3 style (partition_size=1024)
    double l3_compressed_mb;
    double l3_ratio;
    double l3_decompress_ms;
    double l3_throughput_gbps;

    // Comparison
    double l3_vs_fls_ratio;  // >1 means L3 is slower
};

// ============================================================================
// Load SSB Column
// ============================================================================

std::vector<uint32_t> loadSSBColumn(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening: " << path << std::endl;
        return {};
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(uint32_t);
    std::vector<uint32_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);

    return data;
}

// ============================================================================
// Benchmark Function
// ============================================================================

ColumnBenchmark benchmarkColumn(const std::string& name, const std::vector<uint32_t>& data) {
    ColumnBenchmark result;
    result.column_name = name;
    result.num_elements = data.size();
    result.original_mb = data.size() * sizeof(uint32_t) / (1024.0 * 1024.0);

    // Data characteristics
    result.min_val = *std::min_element(data.begin(), data.end());
    result.max_val = *std::max_element(data.begin(), data.end());

    // ========== Vertical style compression (block_size=128) ==========
    auto fls_compressed = compressFOR(data, 128);

    double avg_bw_fls = 0;
    for (auto bw : fls_compressed.bit_widths) avg_bw_fls += bw;
    avg_bw_fls /= fls_compressed.bit_widths.size();
    result.avg_bit_width = avg_bw_fls;

    result.fls_compressed_mb = (fls_compressed.packed_data.size() * sizeof(uint32_t) +
                                fls_compressed.base_values.size() * sizeof(uint32_t) +
                                fls_compressed.bit_widths.size()) / (1024.0 * 1024.0);
    result.fls_ratio = result.original_mb / result.fls_compressed_mb;

    // GPU decompression benchmark
    uint32_t *d_packed, *d_bases, *d_output;
    uint8_t *d_bws;

    cudaMalloc(&d_packed, fls_compressed.packed_data.size() * sizeof(uint32_t));
    cudaMalloc(&d_bases, fls_compressed.base_values.size() * sizeof(uint32_t));
    cudaMalloc(&d_bws, fls_compressed.bit_widths.size());
    cudaMalloc(&d_output, data.size() * sizeof(uint32_t));

    cudaMemcpy(d_packed, fls_compressed.packed_data.data(),
               fls_compressed.packed_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bases, fls_compressed.base_values.data(),
               fls_compressed.base_values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bws, fls_compressed.bit_widths.data(),
               fls_compressed.bit_widths.size(), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 5; i++) {
        fls_unpack_kernel<<<fls_compressed.num_blocks, 128>>>(
            d_packed, d_bases, d_bws, d_output, fls_compressed.num_blocks, 128);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int RUNS = 20;
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++) {
        fls_unpack_kernel<<<fls_compressed.num_blocks, 128>>>(
            d_packed, d_bases, d_bws, d_output, fls_compressed.num_blocks, 128);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float fls_ms;
    cudaEventElapsedTime(&fls_ms, start, stop);
    result.fls_decompress_ms = fls_ms / RUNS;
    result.fls_throughput_gbps = (result.original_mb / 1024.0) / (result.fls_decompress_ms / 1000.0);

    cudaFree(d_packed);
    cudaFree(d_bases);
    cudaFree(d_bws);
    cudaFree(d_output);

    // ========== L3 style compression (partition_size=1024) ==========
    auto l3_compressed = compressL3Interleaved(data, 1024);

    result.l3_compressed_mb = (l3_compressed.interleaved_data.size() * sizeof(uint32_t) +
                               l3_compressed.base_values.size() * sizeof(uint32_t) +
                               l3_compressed.bit_widths.size()) / (1024.0 * 1024.0);
    result.l3_ratio = result.original_mb / result.l3_compressed_mb;

    // GPU decompression
    uint32_t *d_interleaved, *d_l3_bases, *d_l3_output;
    uint8_t *d_l3_bws;

    cudaMalloc(&d_interleaved, l3_compressed.interleaved_data.size() * sizeof(uint32_t));
    cudaMalloc(&d_l3_bases, l3_compressed.base_values.size() * sizeof(uint32_t));
    cudaMalloc(&d_l3_bws, l3_compressed.bit_widths.size());
    cudaMalloc(&d_l3_output, data.size() * sizeof(uint32_t));

    cudaMemcpy(d_interleaved, l3_compressed.interleaved_data.data(),
               l3_compressed.interleaved_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l3_bases, l3_compressed.base_values.data(),
               l3_compressed.base_values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_l3_bws, l3_compressed.bit_widths.data(),
               l3_compressed.bit_widths.size(), cudaMemcpyHostToDevice);

    // Warmup
    for (int i = 0; i < 5; i++) {
        l3_interleaved_unpack_kernel<<<l3_compressed.num_partitions, 32>>>(
            d_interleaved, d_l3_bases, d_l3_bws, d_l3_output,
            l3_compressed.num_partitions, 1024);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEventRecord(start);
    for (int i = 0; i < RUNS; i++) {
        l3_interleaved_unpack_kernel<<<l3_compressed.num_partitions, 32>>>(
            d_interleaved, d_l3_bases, d_l3_bws, d_l3_output,
            l3_compressed.num_partitions, 1024);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float l3_ms;
    cudaEventElapsedTime(&l3_ms, start, stop);
    result.l3_decompress_ms = l3_ms / RUNS;
    result.l3_throughput_gbps = (result.original_mb / 1024.0) / (result.l3_decompress_ms / 1000.0);

    cudaFree(d_interleaved);
    cudaFree(d_l3_bases);
    cudaFree(d_l3_bws);
    cudaFree(d_l3_output);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.l3_vs_fls_ratio = result.l3_decompress_ms / result.fls_decompress_ms;

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "================================================================" << std::endl;
    std::cout << "Vertical GPU vs L3 Decompression Benchmark on SSB Columns" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::endl;

    // Column mappings (LINEORDER columns)
    std::vector<std::pair<std::string, int>> columns = {
        {"lo_orderdate", 0},    // LINEORDER0.bin - orderdate
        {"lo_quantity", 1},     // LINEORDER1.bin - quantity
        {"lo_extendedprice", 10}, // LINEORDER10.bin - extendedprice
        {"lo_discount", 11},    // LINEORDER11.bin - discount
        {"lo_revenue", 12},     // LINEORDER12.bin - revenue
        {"lo_supplycost", 13},  // LINEORDER13.bin - supplycost
        {"lo_custkey", 2},      // LINEORDER2.bin - custkey
        {"lo_partkey", 3},      // LINEORDER3.bin - partkey
        {"lo_suppkey", 4},      // LINEORDER4.bin - suppkey
    };

    std::vector<ColumnBenchmark> results;

    for (const auto& col : columns) {
        std::string path = data_dir + "/LINEORDER" + std::to_string(col.second) + ".bin";
        std::cout << "Loading " << col.first << " from " << path << "..." << std::endl;

        auto data = loadSSBColumn(path);
        if (data.empty()) {
            std::cerr << "Failed to load " << col.first << std::endl;
            continue;
        }

        std::cout << "  Elements: " << data.size() << std::endl;
        auto result = benchmarkColumn(col.first, data);
        results.push_back(result);

        std::cout << "  Range: [" << result.min_val << ", " << result.max_val << "]" << std::endl;
        std::cout << "  Avg bit width: " << std::fixed << std::setprecision(1)
                  << result.avg_bit_width << std::endl;
        std::cout << "  FLS decompress: " << std::fixed << std::setprecision(3)
                  << result.fls_decompress_ms << " ms (" << std::setprecision(1)
                  << result.fls_throughput_gbps << " GB/s)" << std::endl;
        std::cout << "  L3  decompress: " << std::fixed << std::setprecision(3)
                  << result.l3_decompress_ms << " ms (" << std::setprecision(1)
                  << result.l3_throughput_gbps << " GB/s)" << std::endl;
        std::cout << "  L3/FLS ratio: " << std::fixed << std::setprecision(2)
                  << result.l3_vs_fls_ratio << "x" << std::endl;
        std::cout << std::endl;
    }

    // Summary table
    std::cout << "================================================================" << std::endl;
    std::cout << "Summary: Decompression Throughput (GB/s)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << std::left << std::setw(18) << "Column"
              << std::right << std::setw(8) << "AvgBW"
              << std::setw(12) << "FLS(GB/s)"
              << std::setw(12) << "L3(GB/s)"
              << std::setw(10) << "L3/FLS"
              << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    double total_fls = 0, total_l3 = 0;
    for (const auto& r : results) {
        std::cout << std::left << std::setw(18) << r.column_name
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(8) << r.avg_bit_width
                  << std::setw(12) << r.fls_throughput_gbps
                  << std::setw(12) << r.l3_throughput_gbps
                  << std::setw(10) << std::setprecision(2) << r.l3_vs_fls_ratio << "x"
                  << std::endl;
        total_fls += r.fls_throughput_gbps;
        total_l3 += r.l3_throughput_gbps;
    }

    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::left << std::setw(18) << "Average"
              << std::right << std::setw(8) << ""
              << std::fixed << std::setprecision(1)
              << std::setw(12) << (total_fls / results.size())
              << std::setw(12) << (total_l3 / results.size())
              << std::setw(10) << std::setprecision(2)
              << ((total_l3 / results.size()) / (total_fls / results.size())) << "x"
              << std::endl;

    std::cout << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "  - FLS: block_size=128, 1 block per CUDA block" << std::endl;
    std::cout << "  - L3:  partition_size=1024, 1 warp per partition (interleaved)" << std::endl;
    std::cout << "  - L3/FLS < 1.0 means L3 is faster" << std::endl;

    return 0;
}
