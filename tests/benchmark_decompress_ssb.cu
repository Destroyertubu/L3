/**
 * @file benchmark_decompress_ssb.cu
 * @brief Direct comparison of L3 vs Vertical decompression for SSB queries
 *
 * This benchmark measures:
 * 1. Pure decompression throughput (no filtering)
 * 2. L3's actual V5 fused format
 * 3. Vertical-style block format
 */

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

// ============================================================================
// L3 V5 Style: Interleaved Format (1 warp per tile, 4 mini-vectors)
// ============================================================================

constexpr int TILE_SIZE = 1024;
constexpr int MINI_VEC_SIZE = 256;
constexpr int WARP_SZ = 32;

__global__ void l3_decompress_kernel(
    const uint32_t* __restrict__ interleaved_data,
    const uint32_t* __restrict__ base_values,
    const uint8_t* __restrict__ bit_widths,
    const int64_t* __restrict__ bit_offsets,  // Pre-computed bit offsets per partition
    uint32_t* __restrict__ output,
    int num_partitions)
{
    int part_idx = blockIdx.x;
    if (part_idx >= num_partitions) return;

    int lane = threadIdx.x;
    uint32_t base = base_values[part_idx];
    uint8_t bw = bit_widths[part_idx];
    int64_t part_bit_base = bit_offsets[part_idx];

    // Process 4 mini-vectors per partition
    for (int mv = 0; mv < 4; mv++) {
        int64_t mv_bit_base = part_bit_base + static_cast<int64_t>(mv) * MINI_VEC_SIZE * bw;

        // Each thread extracts 8 values
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

            int global_idx = part_idx * TILE_SIZE + mv * MINI_VEC_SIZE + idx_in_mv;
            output[global_idx] = base + val;
        }
    }
}

// ============================================================================
// Vertical Style: Block Format (128 threads per block)
// ============================================================================

__global__ void fls_decompress_kernel(
    const uint32_t* __restrict__ packed_data,
    const uint32_t* __restrict__ base_values,
    const uint8_t* __restrict__ bit_widths,
    const int64_t* __restrict__ packed_offsets,  // Pre-computed word offsets per block
    uint32_t* __restrict__ output,
    int num_blocks,
    int block_size = 128)
{
    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    int tid = threadIdx.x;
    uint32_t base = base_values[block_idx];
    uint8_t bw = bit_widths[block_idx];
    int64_t word_offset = packed_offsets[block_idx];

    const uint32_t* block_packed = packed_data + word_offset;

    // Each thread unpacks 4 values (block_size/num_threads items)
    for (int i = tid; i < block_size; i += blockDim.x) {
        int64_t bit_offset = static_cast<int64_t>(i) * bw;
        int word_idx = bit_offset / 32;
        int bit_pos = bit_offset % 32;

        uint32_t val = (block_packed[word_idx] >> bit_pos);
        if (bit_pos + bw > 32) {
            val |= (block_packed[word_idx + 1] << (32 - bit_pos));
        }
        val &= ((1u << bw) - 1);

        output[block_idx * block_size + i] = base + val;
    }
}

// ============================================================================
// Data Compression (CPU)
// ============================================================================

struct L3CompressedData {
    std::vector<uint32_t> interleaved_data;
    std::vector<uint32_t> base_values;
    std::vector<uint8_t> bit_widths;
    std::vector<int64_t> bit_offsets;
    int num_partitions;
};

struct FLSCompressedData {
    std::vector<uint32_t> packed_data;
    std::vector<uint32_t> base_values;
    std::vector<uint8_t> bit_widths;
    std::vector<int64_t> packed_offsets;
    int num_blocks;
};

L3CompressedData compressL3(const std::vector<uint32_t>& data) {
    L3CompressedData result;
    result.num_partitions = (data.size() + TILE_SIZE - 1) / TILE_SIZE;

    result.base_values.resize(result.num_partitions);
    result.bit_widths.resize(result.num_partitions);
    result.bit_offsets.resize(result.num_partitions);

    // First pass: compute metadata
    int64_t total_bits = 0;
    for (int p = 0; p < result.num_partitions; p++) {
        int start = p * TILE_SIZE;
        int end = std::min(start + TILE_SIZE, (int)data.size());

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
        result.bit_offsets[p] = total_bits;
        total_bits += static_cast<int64_t>(TILE_SIZE) * bw;
    }

    // Second pass: pack data
    size_t total_words = (total_bits + 31) / 32;
    result.interleaved_data.resize(total_words, 0);

    for (int p = 0; p < result.num_partitions; p++) {
        int start = p * TILE_SIZE;
        int end = std::min(start + TILE_SIZE, (int)data.size());
        uint32_t base = result.base_values[p];
        uint8_t bw = result.bit_widths[p];
        int64_t part_bit_base = result.bit_offsets[p];

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

FLSCompressedData compressFLS(const std::vector<uint32_t>& data, int block_size = 128) {
    FLSCompressedData result;
    result.num_blocks = (data.size() + block_size - 1) / block_size;

    result.base_values.resize(result.num_blocks);
    result.bit_widths.resize(result.num_blocks);
    result.packed_offsets.resize(result.num_blocks);

    // First pass
    int64_t total_words = 0;
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
        result.packed_offsets[b] = total_words;
        total_words += (static_cast<int64_t>(block_size) * bw + 31) / 32;
    }

    // Second pass
    result.packed_data.resize(total_words, 0);

    for (int b = 0; b < result.num_blocks; b++) {
        int start = b * block_size;
        int end = std::min(start + block_size, (int)data.size());
        uint32_t base = result.base_values[b];
        uint8_t bw = result.bit_widths[b];
        int64_t word_offset = result.packed_offsets[b];

        for (int i = start; i < end; i++) {
            uint32_t delta = data[i] - base;
            int idx_in_block = i - start;
            int64_t bit_offset = static_cast<int64_t>(idx_in_block) * bw;
            int local_word = bit_offset / 32;
            int bit_pos = bit_offset % 32;

            result.packed_data[word_offset + local_word] |= (delta << bit_pos);
            if (bit_pos + bw > 32) {
                result.packed_data[word_offset + local_word + 1] |= (delta >> (32 - bit_pos));
            }
        }
    }

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string data_dir = "data/ssb";
    if (argc > 1) data_dir = argv[1];

    std::cout << "================================================================" << std::endl;
    std::cout << "L3 vs Vertical Pure Decompression Benchmark (SSB LINEORDER)" << std::endl;
    std::cout << "================================================================" << std::endl;
    std::cout << "L3: partition_size=1024, 1 warp per partition" << std::endl;
    std::cout << "FLS: block_size=128, 128 threads per block" << std::endl;
    std::cout << std::endl;

    // Column mappings
    std::vector<std::pair<std::string, int>> columns = {
        {"lo_orderdate", 0},
        {"lo_quantity", 1},
        {"lo_extendedprice", 10},
        {"lo_discount", 11},
        {"lo_revenue", 12},
        {"lo_supplycost", 13},
        {"lo_custkey", 2},
        {"lo_partkey", 3},
        {"lo_suppkey", 4},
    };

    std::cout << std::left << std::setw(18) << "Column"
              << std::right << std::setw(10) << "AvgBW"
              << std::setw(12) << "L3(GB/s)"
              << std::setw(12) << "FLS(GB/s)"
              << std::setw(10) << "Ratio"
              << std::endl;
    std::cout << std::string(62, '-') << std::endl;

    double total_l3 = 0, total_fls = 0;
    int count = 0;

    for (const auto& col : columns) {
        std::string path = data_dir + "/LINEORDER" + std::to_string(col.second) + ".bin";

        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) continue;

        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint32_t> data(file_size / sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(data.data()), file_size);
        file.close();

        double original_mb = data.size() * sizeof(uint32_t) / (1024.0 * 1024.0);

        // Compress
        auto l3_data = compressL3(data);
        auto fls_data = compressFLS(data);

        // Calculate average bit width
        double avg_bw = 0;
        for (auto bw : l3_data.bit_widths) avg_bw += bw;
        avg_bw /= l3_data.bit_widths.size();

        // L3 GPU benchmark
        uint32_t *d_l3_packed, *d_l3_bases, *d_l3_output;
        uint8_t *d_l3_bws;
        int64_t *d_l3_offsets;

        cudaMalloc(&d_l3_packed, l3_data.interleaved_data.size() * sizeof(uint32_t));
        cudaMalloc(&d_l3_bases, l3_data.base_values.size() * sizeof(uint32_t));
        cudaMalloc(&d_l3_bws, l3_data.bit_widths.size());
        cudaMalloc(&d_l3_offsets, l3_data.bit_offsets.size() * sizeof(int64_t));
        cudaMalloc(&d_l3_output, data.size() * sizeof(uint32_t));

        cudaMemcpy(d_l3_packed, l3_data.interleaved_data.data(),
                   l3_data.interleaved_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l3_bases, l3_data.base_values.data(),
                   l3_data.base_values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l3_bws, l3_data.bit_widths.data(),
                   l3_data.bit_widths.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_l3_offsets, l3_data.bit_offsets.data(),
                   l3_data.bit_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < 5; i++) {
            l3_decompress_kernel<<<l3_data.num_partitions, WARP_SZ>>>(
                d_l3_packed, d_l3_bases, d_l3_bws, d_l3_offsets,
                d_l3_output, l3_data.num_partitions);
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int RUNS = 20;
        cudaEventRecord(start);
        for (int i = 0; i < RUNS; i++) {
            l3_decompress_kernel<<<l3_data.num_partitions, WARP_SZ>>>(
                d_l3_packed, d_l3_bases, d_l3_bws, d_l3_offsets,
                d_l3_output, l3_data.num_partitions);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float l3_ms;
        cudaEventElapsedTime(&l3_ms, start, stop);
        double l3_throughput = (original_mb / 1024.0) / ((l3_ms / RUNS) / 1000.0);

        cudaFree(d_l3_packed);
        cudaFree(d_l3_bases);
        cudaFree(d_l3_bws);
        cudaFree(d_l3_offsets);
        cudaFree(d_l3_output);

        // FLS GPU benchmark
        uint32_t *d_fls_packed, *d_fls_bases, *d_fls_output;
        uint8_t *d_fls_bws;
        int64_t *d_fls_offsets;

        cudaMalloc(&d_fls_packed, fls_data.packed_data.size() * sizeof(uint32_t));
        cudaMalloc(&d_fls_bases, fls_data.base_values.size() * sizeof(uint32_t));
        cudaMalloc(&d_fls_bws, fls_data.bit_widths.size());
        cudaMalloc(&d_fls_offsets, fls_data.packed_offsets.size() * sizeof(int64_t));
        cudaMalloc(&d_fls_output, data.size() * sizeof(uint32_t));

        cudaMemcpy(d_fls_packed, fls_data.packed_data.data(),
                   fls_data.packed_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fls_bases, fls_data.base_values.data(),
                   fls_data.base_values.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fls_bws, fls_data.bit_widths.data(),
                   fls_data.bit_widths.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_fls_offsets, fls_data.packed_offsets.data(),
                   fls_data.packed_offsets.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

        // Warmup
        for (int i = 0; i < 5; i++) {
            fls_decompress_kernel<<<fls_data.num_blocks, 128>>>(
                d_fls_packed, d_fls_bases, d_fls_bws, d_fls_offsets,
                d_fls_output, fls_data.num_blocks);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int i = 0; i < RUNS; i++) {
            fls_decompress_kernel<<<fls_data.num_blocks, 128>>>(
                d_fls_packed, d_fls_bases, d_fls_bws, d_fls_offsets,
                d_fls_output, fls_data.num_blocks);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float fls_ms;
        cudaEventElapsedTime(&fls_ms, start, stop);
        double fls_throughput = (original_mb / 1024.0) / ((fls_ms / RUNS) / 1000.0);

        cudaFree(d_fls_packed);
        cudaFree(d_fls_bases);
        cudaFree(d_fls_bws);
        cudaFree(d_fls_offsets);
        cudaFree(d_fls_output);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << std::left << std::setw(18) << col.first
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(10) << avg_bw
                  << std::setw(12) << l3_throughput
                  << std::setw(12) << fls_throughput
                  << std::setw(10) << std::setprecision(2) << (l3_throughput / fls_throughput) << "x"
                  << std::endl;

        total_l3 += l3_throughput;
        total_fls += fls_throughput;
        count++;
    }

    std::cout << std::string(62, '-') << std::endl;
    std::cout << std::left << std::setw(18) << "Average"
              << std::right << std::setw(10) << ""
              << std::fixed << std::setprecision(1)
              << std::setw(12) << (total_l3 / count)
              << std::setw(12) << (total_fls / count)
              << std::setw(10) << std::setprecision(2) << ((total_l3 / count) / (total_fls / count)) << "x"
              << std::endl;

    std::cout << std::endl;
    std::cout << "Ratio > 1.0 means L3 is faster" << std::endl;

    return 0;
}
