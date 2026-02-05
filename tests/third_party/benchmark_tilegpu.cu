/**
 * tile-gpu-compression Benchmark
 *
 * Tests GPU-FOR (BitPack), GPU-DFOR (Delta+BitPack), and GPU-RFOR (RLE+BitPack) compression
 * Measures: compression ratio, compression throughput, decompression throughput
 *
 * Usage: benchmark_tilegpu [options]
 *   -d, --data_dir <path>    Data directory (default: /root/autodl-tmp/test/data/sosd/)
 *   -o, --output <file>      Output CSV file (default: reports/tilegpu_results.csv)
 *   -a, --algorithm <alg>    Algorithm: all|gpufor|gpudfor|gpurfor (default: all)
 *   -n, --trials <num>       Number of trials (default: 10)
 *   -w, --warmup <num>       Warmup iterations (default: 3)
 *   -g, --gpu <id>           GPU device ID (default: 0)
 */

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <system_error>

// ============================================================================
// Data Structures
// ============================================================================

struct BenchmarkResult {
    std::string framework;
    std::string algorithm;
    std::string dataset;
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double compress_throughput_gbps;
    double decompress_throughput_gbps;
    bool verified;
};

// ============================================================================
// GPU-FOR (BinPack) Encoding - CPU
// ============================================================================

void binPackEncode(uint32_t* input, uint32_t* output, uint32_t* block_offsets, int num_entries) {
    uint32_t offset = 4; // Skip header
    const int block_size = 128;
    const int miniblock_count = 4;
    const int miniblock_size = 32;

    // Header
    output[0] = block_size;
    output[1] = miniblock_count;
    output[2] = num_entries;
    output[3] = input[0];

    int num_blocks = (num_entries + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        block_offsets[b] = offset;
        int block_start = b * block_size;
        // Find min value
        uint32_t min_val = input[block_start];
        for (int i = 1; i < block_size; i++) {
            if (input[block_start + i] < min_val)
                min_val = input[block_start + i];
        }

        // Store reference
        output[offset++] = min_val;

        // Compute bitwidths for each miniblock
        uint32_t bitwidths[4] = {0, 0, 0, 0};
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = mb_start + miniblock_size;

            for (int i = mb_start; i < mb_end; i++) {
                uint32_t delta = input[i] - min_val;
                if (delta > 0) {
                    uint32_t bw = 32 - __builtin_clz(delta);
                    if (bw > bitwidths[m]) bitwidths[m] = bw;
                }
            }
        }

        // Use max bitwidth for all miniblocks (Simple BinPack)
        uint32_t max_bw = bitwidths[0];
        for (int i = 1; i < 4; i++) {
            max_bw = std::max(max_bw, bitwidths[i]);
        }
        for (int i = 0; i < 4; i++) {
            bitwidths[i] = max_bw;
        }

        // Store bitwidths
        uint32_t packed_bw = bitwidths[0] | (bitwidths[1] << 8) | (bitwidths[2] << 16) | (bitwidths[3] << 24);
        output[offset++] = packed_bw;

        // Pack each miniblock
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = mb_start + miniblock_size;
            uint32_t bitwidth = bitwidths[m];

            uint32_t shift = 0;
            output[offset] = 0;

            for (int i = mb_start; i < mb_end; i++) {
                uint32_t delta = input[i] - min_val;

                if (shift + bitwidth > 32) {
                    if (shift != 32) {
                        output[offset] += (delta << shift);
                    }
                    offset++;
                    output[offset] = 0;
                    shift = (shift + bitwidth) & 31;
                    output[offset] = delta >> (bitwidth - shift);
                } else {
                    output[offset] += (delta << shift);
                    shift += bitwidth;
                }
            }
            offset++;
            output[offset] = 0;
        }
    }

    block_offsets[num_blocks] = offset;
}

// ============================================================================
// GPU-DFOR (Delta+BinPack) Encoding - CPU
// ============================================================================

void deltaBinPackEncode(uint32_t* input, uint32_t* output, uint32_t* block_offsets, int num_entries) {
    const int block_size = 128;
    const int miniblock_count = 4;
    const int miniblock_size = 32;
    const int tile_size = block_size * 4; // 512

    uint32_t offset = 4; // Skip header

    // Header
    output[0] = block_size;
    output[1] = miniblock_count;
    output[2] = num_entries;
    output[3] = input[0]; // First value

    int num_tiles = (num_entries + tile_size - 1) / tile_size;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_size;
        int tile_end = std::min(tile_start + tile_size, num_entries);
        int tile_len = tile_end - tile_start;

        // Temporary delta values (signed to match original DFOR)
        int32_t* tile_deltas = new int32_t[tile_size];
        memset(tile_deltas, 0, tile_size * sizeof(int32_t));

        // Compute delta encoding (signed)
        tile_deltas[0] = 0;
        for (int i = 1; i < tile_len; i++) {
            tile_deltas[i] = static_cast<int32_t>(input[tile_start + i]) -
                             static_cast<int32_t>(input[tile_start + i - 1]);
        }

        // Store first value for this tile
        output[offset] = input[tile_start];
        offset++;

        // Process 4 blocks in this tile
        for (int block_idx = 0; block_idx < 4; block_idx++) {
            int blk_start = block_idx * block_size;

            if (blk_start >= tile_len) break;

            int global_block_idx = tile_idx * 4 + block_idx;
            block_offsets[global_block_idx] = offset;

            // Find min value (reference) - signed deltas
            int32_t min_val = tile_deltas[blk_start];
            for (int i = blk_start + 1; i < blk_start + block_size; i++) {
                if (tile_deltas[i] < min_val)
                    min_val = tile_deltas[i];
            }

            // Store reference
            output[offset++] = static_cast<uint32_t>(min_val);

            // Compute bitwidths
            uint32_t bitwidths[4] = {0, 0, 0, 0};
            for (int m = 0; m < miniblock_count; m++) {
                int mb_start = blk_start + m * miniblock_size;
                int mb_end = mb_start + miniblock_size;

                for (int i = mb_start; i < mb_end; i++) {
                    uint32_t delta = static_cast<uint32_t>(
                        static_cast<int64_t>(tile_deltas[i]) - static_cast<int64_t>(min_val));
                    if (delta > 0) {
                        uint32_t bw = 32 - __builtin_clz(delta);
                        if (bw > bitwidths[m]) bitwidths[m] = bw;
                    }
                }
            }

            // Use max bitwidth
            uint32_t max_bw = bitwidths[0];
            for (int i = 1; i < 4; i++) {
                max_bw = std::max(max_bw, bitwidths[i]);
            }
            for (int i = 0; i < 4; i++) {
                bitwidths[i] = max_bw;
            }

            // Store bitwidths
            uint32_t packed_bw = bitwidths[0] | (bitwidths[1] << 8) |
                                 (bitwidths[2] << 16) | (bitwidths[3] << 24);
            output[offset++] = packed_bw;

            // Pack each miniblock
            for (int m = 0; m < miniblock_count; m++) {
                int mb_start = blk_start + m * miniblock_size;
                int mb_end = mb_start + miniblock_size;
                uint32_t bitwidth = bitwidths[m];

                uint32_t shift = 0;
                output[offset] = 0;

                for (int i = mb_start; i < mb_end; i++) {
                    uint32_t delta = static_cast<uint32_t>(
                        static_cast<int64_t>(tile_deltas[i]) - static_cast<int64_t>(min_val));

                    if (shift + bitwidth > 32) {
                        if (shift != 32) {
                            output[offset] += (delta << shift);
                        }
                        offset++;
                        output[offset] = 0;
                        shift = (shift + bitwidth) & 31;
                        output[offset] = delta >> (bitwidth - shift);
                    } else {
                        output[offset] += (delta << shift);
                        shift += bitwidth;
                    }
                }
                offset++;
                output[offset] = 0;
            }
        }

        delete[] tile_deltas;
    }

    int num_blocks = (num_entries + block_size - 1) / block_size;
    block_offsets[num_blocks] = offset;
}

// ============================================================================
// GPU-RFOR (RLE+BinPack) Encoding - CPU
// ============================================================================

std::pair<uint32_t, uint32_t> rleBinPackEncode(uint32_t* input, uint32_t* value_out, uint32_t* runlen_out,
                                                uint32_t* val_offsets, uint32_t* rl_offsets, int num_entries) {
    const int tile_size = 512;

    uint32_t val_offset = 4; // Skip header
    uint32_t rl_offset = 4;

    // Header
    value_out[0] = tile_size;
    value_out[1] = 4;
    value_out[2] = num_entries;
    value_out[3] = input[0];

    runlen_out[0] = tile_size;
    runlen_out[1] = 4;
    runlen_out[2] = num_entries;
    runlen_out[3] = input[0];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_size;
        int tile_end = std::min(tile_start + tile_size, num_entries);

        val_offsets[tile_idx] = val_offset;
        rl_offsets[tile_idx] = rl_offset;

        // RLE encoding
        std::vector<uint32_t> values;
        std::vector<uint32_t> run_lengths;

        uint32_t current_val = input[tile_start];
        uint32_t run_len = 1;

        for (int i = tile_start + 1; i < tile_end; i++) {
            if (input[i] == current_val) {
                run_len++;
            } else {
                values.push_back(current_val);
                run_lengths.push_back(run_len);
                current_val = input[i];
                run_len = 1;
            }
        }
        values.push_back(current_val);
        run_lengths.push_back(run_len);

        int rle_count = values.size();

        // Find min values (reference)
        uint32_t val_min = values[0];
        uint32_t rl_min = run_lengths[0];
        for (int i = 1; i < rle_count; i++) {
            if (values[i] < val_min) val_min = values[i];
            if (run_lengths[i] < rl_min) rl_min = run_lengths[i];
        }

        // Compute bitwidths
        uint32_t val_bitwidth = 0;
        uint32_t rl_bitwidth = 0;
        for (int i = 0; i < rle_count; i++) {
            uint32_t val_delta = values[i] - val_min;
            uint32_t rl_delta = run_lengths[i] - rl_min;
            if (val_delta > 0) {
                uint32_t bw = 32 - __builtin_clz(val_delta);
                if (bw > val_bitwidth) val_bitwidth = bw;
            }
            if (rl_delta > 0) {
                uint32_t bw = 32 - __builtin_clz(rl_delta);
                if (bw > rl_bitwidth) rl_bitwidth = bw;
            }
        }

        // Store value data: reference, bitwidth, count
        value_out[val_offset++] = val_min;
        value_out[val_offset++] = val_bitwidth | (val_bitwidth << 8) | (val_bitwidth << 16) | (val_bitwidth << 24);
        value_out[val_offset++] = rle_count;

        // Pack values
        uint32_t shift = 0;
        value_out[val_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint32_t delta = values[i] - val_min;

            if (shift + val_bitwidth > 32) {
                if (shift != 32) {
                    value_out[val_offset] += (delta << shift);
                }
                val_offset++;
                value_out[val_offset] = 0;
                shift = (shift + val_bitwidth) & 31;
                value_out[val_offset] = delta >> (val_bitwidth - shift);
            } else {
                value_out[val_offset] += (delta << shift);
                shift += val_bitwidth;
            }
        }
        val_offset++;

        // Store run_length data: reference, bitwidth, count
        runlen_out[rl_offset++] = rl_min;
        runlen_out[rl_offset++] = rl_bitwidth | (rl_bitwidth << 8) | (rl_bitwidth << 16) | (rl_bitwidth << 24);
        runlen_out[rl_offset++] = rle_count;

        // Pack run_lengths
        shift = 0;
        runlen_out[rl_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint32_t delta = run_lengths[i] - rl_min;

            if (shift + rl_bitwidth > 32) {
                if (shift != 32) {
                    runlen_out[rl_offset] += (delta << shift);
                }
                rl_offset++;
                runlen_out[rl_offset] = 0;
                shift = (shift + rl_bitwidth) & 31;
                runlen_out[rl_offset] = delta >> (rl_bitwidth - shift);
            } else {
                runlen_out[rl_offset] += (delta << shift);
                shift += rl_bitwidth;
            }
        }
        rl_offset++;
    }

    val_offsets[num_tiles] = val_offset;
    rl_offsets[num_tiles] = rl_offset;

    return std::make_pair(val_offset, rl_offset);
}

// ============================================================================
// 64-bit GPU-FOR (BinPack) Encoding - CPU
// ============================================================================

void binPackEncode64(uint64_t* input, uint64_t* output, uint32_t* block_offsets, int num_entries) {
    uint32_t offset = 4; // Skip header (4 x 64-bit words)
    const int block_size = 128;
    const int miniblock_count = 4;
    const int miniblock_size = 32;

    // Header (64-bit words)
    output[0] = block_size;
    output[1] = miniblock_count;
    output[2] = num_entries;
    output[3] = input[0];

    int num_blocks = (num_entries + block_size - 1) / block_size;

    for (int b = 0; b < num_blocks; b++) {
        block_offsets[b] = offset;
        int block_start = b * block_size;
        // Find min value
        uint64_t min_val = input[block_start];
        for (int i = 1; i < block_size; i++) {
            if (input[block_start + i] < min_val)
                min_val = input[block_start + i];
        }

        // Store reference (64-bit)
        output[offset++] = min_val;

        // Compute bitwidths for each miniblock
        uint32_t bitwidths[4] = {0, 0, 0, 0};
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = mb_start + miniblock_size;

            for (int i = mb_start; i < mb_end; i++) {
                uint64_t delta = input[i] - min_val;
                if (delta > 0) {
                    uint32_t bw = 64 - __builtin_clzll(delta);
                    if (bw > bitwidths[m]) bitwidths[m] = bw;
                }
            }
        }

        // Use max bitwidth for all miniblocks
        uint32_t max_bw = bitwidths[0];
        for (int i = 1; i < 4; i++) {
            max_bw = std::max(max_bw, bitwidths[i]);
        }
        for (int i = 0; i < 4; i++) {
            bitwidths[i] = max_bw;
        }

        // Store bitwidths (packed into 64-bit word, 16 bits each for future extension)
        uint64_t packed_bw = bitwidths[0] | ((uint64_t)bitwidths[1] << 16) |
                            ((uint64_t)bitwidths[2] << 32) | ((uint64_t)bitwidths[3] << 48);
        output[offset++] = packed_bw;

        // Pack each miniblock
        for (int m = 0; m < miniblock_count; m++) {
            int mb_start = block_start + m * miniblock_size;
            int mb_end = mb_start + miniblock_size;
            uint32_t bitwidth = bitwidths[m];

            uint32_t shift = 0;
            output[offset] = 0;

            for (int i = mb_start; i < mb_end; i++) {
                uint64_t delta = input[i] - min_val;

                if (shift + bitwidth > 64) {
                    if (shift != 64) {
                        output[offset] += (delta << shift);
                    }
                    offset++;
                    output[offset] = 0;
                    shift = (shift + bitwidth) & 63;
                    output[offset] = delta >> (bitwidth - shift);
                } else {
                    output[offset] += (delta << shift);
                    shift += bitwidth;
                }
            }
            offset++;
            output[offset] = 0;
        }
    }

    block_offsets[num_blocks] = offset;
}

// ============================================================================
// 64-bit GPU-DFOR (Delta+BinPack) Encoding - CPU
// ============================================================================

void deltaBinPackEncode64(uint64_t* input, uint64_t* output, uint32_t* block_offsets, int num_entries) {
    const int block_size = 128;
    const int miniblock_count = 4;
    const int miniblock_size = 32;
    const int tile_size = block_size * 4; // 512

    uint32_t offset = 4; // Skip header

    // Header
    output[0] = block_size;
    output[1] = miniblock_count;
    output[2] = num_entries;
    output[3] = input[0];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_size;
        int tile_end = std::min(tile_start + tile_size, num_entries);
        int tile_len = tile_end - tile_start;

        // Temporary delta values (signed to match original DFOR semantics)
        int64_t* tile_deltas = new int64_t[tile_size];
        memset(tile_deltas, 0, tile_size * sizeof(int64_t));

        // Compute delta encoding (signed)
        tile_deltas[0] = 0;
        for (int i = 1; i < tile_len; i++) {
            tile_deltas[i] = static_cast<int64_t>(input[tile_start + i]) -
                             static_cast<int64_t>(input[tile_start + i - 1]);
        }

        // Store first value for this tile
        output[offset] = input[tile_start];
        offset++;

        // Process 4 blocks in this tile
        for (int block_idx = 0; block_idx < 4; block_idx++) {
            int blk_start = block_idx * block_size;

            if (blk_start >= tile_len) break;

            int global_block_idx = tile_idx * 4 + block_idx;
            block_offsets[global_block_idx] = offset;

            // Find min value (reference) - signed deltas
            int64_t min_val = tile_deltas[blk_start];
            for (int i = blk_start + 1; i < blk_start + block_size; i++) {
                if (tile_deltas[i] < min_val)
                    min_val = tile_deltas[i];
            }

            // Store reference
            output[offset++] = static_cast<uint64_t>(min_val);

            // Compute bitwidths
            uint32_t bitwidths[4] = {0, 0, 0, 0};
            for (int m = 0; m < miniblock_count; m++) {
                int mb_start = blk_start + m * miniblock_size;
                int mb_end = mb_start + miniblock_size;

                for (int i = mb_start; i < mb_end; i++) {
                    uint64_t delta = static_cast<uint64_t>(
                        static_cast<int64_t>(tile_deltas[i]) - static_cast<int64_t>(min_val));
                    if (delta > 0) {
                        uint32_t bw = 64 - __builtin_clzll(delta);
                        if (bw > bitwidths[m]) bitwidths[m] = bw;
                    }
                }
            }

            // Use max bitwidth
            uint32_t max_bw = bitwidths[0];
            for (int i = 1; i < 4; i++) {
                max_bw = std::max(max_bw, bitwidths[i]);
            }
            for (int i = 0; i < 4; i++) {
                bitwidths[i] = max_bw;
            }

            // Store bitwidths
            uint64_t packed_bw = bitwidths[0] | ((uint64_t)bitwidths[1] << 16) |
                                ((uint64_t)bitwidths[2] << 32) | ((uint64_t)bitwidths[3] << 48);
            output[offset++] = packed_bw;

            // Pack each miniblock
            for (int m = 0; m < miniblock_count; m++) {
                int mb_start = blk_start + m * miniblock_size;
                int mb_end = mb_start + miniblock_size;
                uint32_t bitwidth = bitwidths[m];

                uint32_t shift = 0;
                output[offset] = 0;

                for (int i = mb_start; i < mb_end; i++) {
                    uint64_t delta = static_cast<uint64_t>(
                        static_cast<int64_t>(tile_deltas[i]) - static_cast<int64_t>(min_val));

                    if (shift + bitwidth > 64) {
                        if (shift != 64) {
                            output[offset] += (delta << shift);
                        }
                        offset++;
                        output[offset] = 0;
                        shift = (shift + bitwidth) & 63;
                        output[offset] = delta >> (bitwidth - shift);
                    } else {
                        output[offset] += (delta << shift);
                        shift += bitwidth;
                    }
                }
                offset++;
                output[offset] = 0;
            }
        }

        delete[] tile_deltas;
    }

    int num_blocks = (num_entries + block_size - 1) / block_size;
    block_offsets[num_blocks] = offset;
}

// ============================================================================
// 64-bit GPU-RFOR (RLE+BinPack) Encoding - CPU
// ============================================================================

std::pair<uint32_t, uint32_t> rleBinPackEncode64(uint64_t* input, uint64_t* value_out, uint64_t* runlen_out,
                                                  uint32_t* val_offsets, uint32_t* rl_offsets, int num_entries) {
    const int tile_size = 512;

    uint32_t val_offset = 4; // Skip header
    uint32_t rl_offset = 4;

    // Header
    value_out[0] = tile_size;
    value_out[1] = 4;
    value_out[2] = num_entries;
    value_out[3] = input[0];

    runlen_out[0] = tile_size;
    runlen_out[1] = 4;
    runlen_out[2] = num_entries;
    runlen_out[3] = input[0];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;

    // Debug stats
    long total_rle_pairs = 0;
    int min_rle_count = tile_size;
    int max_rle_count = 0;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        int tile_start = tile_idx * tile_size;
        int tile_end = std::min(tile_start + tile_size, num_entries);

        val_offsets[tile_idx] = val_offset;
        rl_offsets[tile_idx] = rl_offset;

        // RLE encoding
        std::vector<uint64_t> values;
        std::vector<uint64_t> run_lengths;

        uint64_t current_val = input[tile_start];
        uint64_t run_len = 1;

        for (int i = tile_start + 1; i < tile_end; i++) {
            if (input[i] == current_val) {
                run_len++;
            } else {
                values.push_back(current_val);
                run_lengths.push_back(run_len);
                current_val = input[i];
                run_len = 1;
            }
        }
        values.push_back(current_val);
        run_lengths.push_back(run_len);

        int rle_count = values.size();

        // Debug stats update
        total_rle_pairs += rle_count;
        if (rle_count < min_rle_count) min_rle_count = rle_count;
        if (rle_count > max_rle_count) max_rle_count = rle_count;

        // Find min values (reference)
        uint64_t val_min = values[0];
        uint64_t rl_min = run_lengths[0];
        for (int i = 1; i < rle_count; i++) {
            if (values[i] < val_min) val_min = values[i];
            if (run_lengths[i] < rl_min) rl_min = run_lengths[i];
        }

        // Compute bitwidths
        uint32_t val_bitwidth = 0;
        uint32_t rl_bitwidth = 0;
        for (int i = 0; i < rle_count; i++) {
            uint64_t val_delta = values[i] - val_min;
            uint64_t rl_delta = run_lengths[i] - rl_min;
            if (val_delta > 0) {
                uint32_t bw = 64 - __builtin_clzll(val_delta);
                if (bw > val_bitwidth) val_bitwidth = bw;
            }
            if (rl_delta > 0) {
                uint32_t bw = 64 - __builtin_clzll(rl_delta);
                if (bw > rl_bitwidth) rl_bitwidth = bw;
            }
        }

        // Store value data: reference, bitwidth, count
        value_out[val_offset++] = val_min;
        value_out[val_offset++] = static_cast<uint64_t>(val_bitwidth) |
                                  (static_cast<uint64_t>(val_bitwidth) << 8) |
                                  (static_cast<uint64_t>(val_bitwidth) << 16) |
                                  (static_cast<uint64_t>(val_bitwidth) << 24);
        value_out[val_offset++] = rle_count;

        // Pack values
        uint32_t shift = 0;
        value_out[val_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint64_t delta = values[i] - val_min;

            if (shift + val_bitwidth > 64) {
                if (shift != 64) {
                    value_out[val_offset] += (delta << shift);
                }
                val_offset++;
                value_out[val_offset] = 0;
                shift = (shift + val_bitwidth) & 63;
                value_out[val_offset] = delta >> (val_bitwidth - shift);
            } else {
                value_out[val_offset] += (delta << shift);
                shift += val_bitwidth;
            }
        }
        val_offset++;

        // Store run_length data: reference, bitwidth, count
        runlen_out[rl_offset++] = rl_min;
        runlen_out[rl_offset++] = static_cast<uint64_t>(rl_bitwidth) |
                                  (static_cast<uint64_t>(rl_bitwidth) << 8) |
                                  (static_cast<uint64_t>(rl_bitwidth) << 16) |
                                  (static_cast<uint64_t>(rl_bitwidth) << 24);
        runlen_out[rl_offset++] = rle_count;

        // Pack run_lengths
        shift = 0;
        runlen_out[rl_offset] = 0;
        for (int i = 0; i < rle_count; i++) {
            uint64_t delta = run_lengths[i] - rl_min;

            if (shift + rl_bitwidth > 64) {
                if (shift != 64) {
                    runlen_out[rl_offset] += (delta << shift);
                }
                rl_offset++;
                runlen_out[rl_offset] = 0;
                shift = (shift + rl_bitwidth) & 63;
                runlen_out[rl_offset] = delta >> (rl_bitwidth - shift);
            } else {
                runlen_out[rl_offset] += (delta << shift);
                shift += rl_bitwidth;
            }
        }
        rl_offset++;
    }

    val_offsets[num_tiles] = val_offset;
    rl_offsets[num_tiles] = rl_offset;

    // Print debug stats
    double avg_rle_pairs = (double)total_rle_pairs / num_tiles;
    std::cerr << "[RFOR64-DEBUG] tiles=" << num_tiles
              << " total_rle_pairs=" << total_rle_pairs
              << " avg_rle_pairs=" << std::fixed << std::setprecision(2) << avg_rle_pairs
              << " min=" << min_rle_count
              << " max=" << max_rle_count << std::endl;

    return std::make_pair(val_offset, rl_offset);
}

// ============================================================================
// GPU Decompression Kernels - GPU-FOR
// ============================================================================

__forceinline__ __device__ int decodeElement(int i, uint miniblock_index, uint index_into_miniblock,
                                              uint* data_block, uint* bitwidths, uint* offsets) {
    int reference = reinterpret_cast<int*>(data_block)[0];
    uint miniblock_offset = offsets[miniblock_index];
    uint bitwidth = bitwidths[miniblock_index];

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 2 + (start_bitindex >> 5);
    start_bitindex = start_bitindex & 31;

    unsigned long long element_block = (((unsigned long long)data_block[miniblock_offset + start_intindex + 1]) << 32) |
                                        data_block[miniblock_offset + start_intindex];
    uint element = (element_block >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);

    return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack(uint* block_start, uint* data,
    uint* shared_buffer, int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    int tile_idx = blockIdx.x;
    int threadId = threadIdx.x;

    uint* block_starts = &shared_buffer[0];
    if (threadId < ITEMS_PER_THREAD + 1) {
        block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
    }
    __syncthreads();

    uint* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

    uint start_offset = block_starts[0];
    uint end_offset = block_starts[ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = start_offset + threadIdx.x + (i << 7);
        if (index < end_offset)
            data_block[threadIdx.x + (i << 7)] = data[index];
    }
    __syncthreads();

    uint* bitwidths = &shared_buffer[ITEMS_PER_THREAD + 1];
    uint* offsets = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)];

    if (threadId < (ITEMS_PER_THREAD << 2)) {
        int i = threadId >> 2;
        int miniblock_index = threadId & 3;

        uint miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
        uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
        uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
        uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

        offsets[threadId] = miniblock_offset;
        bitwidths[threadId] = bitwidth;
    }
    __syncthreads();

    uint miniblock_index = threadIdx.x >> 5;
    uint index_into_miniblock = threadIdx.x & 31;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = decodeElement(threadIdx.x, miniblock_index, index_into_miniblock,
                                 data_block + block_starts[i] - block_starts[0],
                                 bitwidths + (i << 2), offsets + (i << 2));
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runBinKernel(int* col, uint* col_block_start, uint* col_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int col_block[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint shared_buffer[];
    LoadBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = col_block[i];
        }
    }
}

// ============================================================================
// GPU Decompression Kernels - GPU-DFOR
// ============================================================================

__forceinline__ __device__ int decodeElementDBin(int i, uint* data_block) {
    int reference = reinterpret_cast<int*>(data_block)[0];
    uint miniblock_index = i / 32;
    uint miniblock_bitwidths = data_block[1];

    uint miniblock_offset = 0;
    for (int j = 0; j < (int)miniblock_index; j++) {
        miniblock_offset += (miniblock_bitwidths & 255);
        miniblock_bitwidths >>= 8;
    }

    uint bitwidth = miniblock_bitwidths & 255;
    uint index_into_miniblock = i & 31;

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 2 + start_bitindex / 32;

    unsigned long long element_block = (((unsigned long long)data_block[miniblock_offset + start_intindex + 1]) << 32) |
                                        data_block[miniblock_offset + start_intindex];
    start_bitindex = start_bitindex & 31;

    uint element = (element_block & (((1ULL << bitwidth) - 1ULL) << start_bitindex)) >> start_bitindex;

    return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadDBinPack(uint* block_start, uint* data,
    uint* shared_buffer, int (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    int tile_idx = blockIdx.x;

    uint* block_starts = &shared_buffer[0];
    if (threadIdx.x < 5) {
        block_starts[threadIdx.x] = block_start[tile_idx * 4 + threadIdx.x];
    }
    __syncthreads();

    uint* data_block = &shared_buffer[5];

    uint start_offset = block_starts[0] - 1;
    uint end_offset = block_starts[4];
    uint data_size = end_offset - start_offset;
    // Load all tile data into shared memory (may need more than 512 words for large bitwidths)
    for (uint i = 0; i < data_size; i += BLOCK_THREADS) {
        uint idx = i + threadIdx.x;
        if (idx < data_size) {
            data_block[idx] = data[start_offset + idx];
        }
    }
    __syncthreads();

    int first_value = data_block[0];
    data_block = data_block + 1;

    for (int i = 0; i < 4; i++) {
        if (is_last_tile) {
            if ((int)(threadIdx.x + i * 128) < num_tile_items) {
                items[i] = decodeElementDBin(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
            }
        } else {
            items[i] = decodeElementDBin(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
        }
    }

    if (threadIdx.x == 0) {
        items[0] = first_value;
    }

    __syncthreads();

    typename BlockScan::TempStorage* temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage* temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items, items);
    __syncthreads();

    BlockExchange(*temp_storage_exchange).BlockedToStriped(items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel(int* col, uint* col_block_start, uint* col_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int col_block[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint shared_buffer[];
    LoadDBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        col_block_start, col_data, shared_buffer, col_block, is_last_tile, num_tile_items);

    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = col_block[i];
        }
    }
}

// ============================================================================
// GPU Decompression Kernels - GPU-RFOR
// ============================================================================

__forceinline__ __device__ int decodeElementRBin(int i, uint* data_block, uint reference, uint bitwidth) {
    uint start_bitindex = (bitwidth * i);
    uint start_intindex = (start_bitindex >> 5);
    start_bitindex = start_bitindex & 31;

    unsigned long long element_block = (((unsigned long long)data_block[start_intindex + 1]) << 32) |
                                        data_block[start_intindex];
    uint element = (element_block >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);

    return reference + element;
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack(uint* val_block_start, uint* rl_block_start,
    uint* value, uint* run_length, uint* shared_buffer, int (&items_value)[ITEMS_PER_THREAD],
    int (&items_run_length)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int, BLOCK_THREADS> BlockScan;

    uint num_decode;
    int tile_idx = blockIdx.x;
    const int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Extended buffer size for worst-case compressed data (metadata + packed data)
    // Max packed data size: ceil(tile_size * 32 / 32) + 3 metadata = tile_size + 3
    const int MAX_COMPRESSED_SIZE = tile_size + 64;  // Add margin for safety

    // Shared memory layout:
    // [0..1]: val_block_starts (2 elements)
    // [2..3]: rl_block_starts (2 elements)
    // [4..4+MAX_COMPRESSED_SIZE-1]: val_data_block
    // [4+MAX_COMPRESSED_SIZE..]: rl_data_block

    uint* val_block_starts = &shared_buffer[0];
    uint* rl_block_starts = &shared_buffer[2];
    if (threadIdx.x < 2) {
        val_block_starts[threadIdx.x] = val_block_start[tile_idx + threadIdx.x];
        rl_block_starts[threadIdx.x] = rl_block_start[tile_idx + threadIdx.x];
    }
    __syncthreads();

    uint* val_data_block = &shared_buffer[4];
    uint* rl_data_block = &shared_buffer[4 + MAX_COMPRESSED_SIZE];

    uint start_offset_val = val_block_starts[0];
    uint end_offset_val = val_block_starts[1];
    uint start_offset_rl = rl_block_starts[0];
    uint end_offset_rl = rl_block_starts[1];

    uint val_data_size = end_offset_val - start_offset_val;
    uint rl_data_size = end_offset_rl - start_offset_rl;

    __syncthreads();

    // Load compressed data with extended range
    // Use multiple iterations to load potentially larger compressed data
    const int LOAD_ITERATIONS = (MAX_COMPRESSED_SIZE + BLOCK_THREADS - 1) / BLOCK_THREADS;
    for (int iter = 0; iter < LOAD_ITERATIONS; iter++) {
        uint local_idx = iter * BLOCK_THREADS + threadIdx.x;
        if (local_idx < val_data_size && local_idx < MAX_COMPRESSED_SIZE) {
            val_data_block[local_idx] = value[start_offset_val + local_idx];
        }
        if (local_idx < rl_data_size && local_idx < MAX_COMPRESSED_SIZE) {
            rl_data_block[local_idx] = run_length[start_offset_rl + local_idx];
        }
    }
    __syncthreads();

    uint count = val_data_block[2];
    uint offset = 0;
    num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint* val_ptr = val_data_block + 3;
        uint* rl_ptr = rl_data_block + 3;

        uint reference, bitwidth;
        if (threadIdx.x < num_decode && threadIdx.x + offset < count) {
            reference = val_data_block[0];
            bitwidth = val_data_block[1] & 255;
            items_value[i] = decodeElementRBin(threadIdx.x + offset, val_ptr, reference, bitwidth);
            reference = rl_data_block[0];
            bitwidth = rl_data_block[1] & 255;
            items_run_length[i] = decodeElementRBin(threadIdx.x + offset, rl_ptr, reference, bitwidth);
        } else {
            items_value[i] = 0;
            items_run_length[i] = 0;
        }
        offset += num_decode;
    }
    __syncthreads();

    typename BlockScan::TempStorage* temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(rl_data_block);
    typename BlockExchange::TempStorage* temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(rl_data_block);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        val_data_block[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
    }
    __syncthreads();

    const int tile_size_32 = BLOCK_THREADS * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        // Boundary check: items_run_length[i] can equal tile_size for the last RLE element
        if (items_run_length[i] < tile_size_32) {
            val_data_block[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_run_length[i] = val_data_block[threadIdx.x * ITEMS_PER_THREAD + i];
    }
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    BlockExchange(*temp_storage_exchange).BlockedToStriped(items_run_length);
    __syncthreads();

    offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode) val_data_block[threadIdx.x + offset] = items_value[i];
        offset += num_decode;
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_value[i] = val_data_block[items_run_length[i]];
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRBinKernel(int* col, uint* val_block_start, uint* val_data,
                               uint* rl_block_start, uint* rl_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int items_value[ITEMS_PER_THREAD];
    int items_run_length[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint shared_buffer[];
    LoadRBinPack<BLOCK_THREADS, ITEMS_PER_THREAD>(
        val_block_start, rl_block_start, val_data, rl_data, shared_buffer,
        items_value, items_run_length, is_last_tile, num_tile_items);

    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = items_value[i];
        }
    }
}

// ============================================================================
// 64-bit GPU Decompression Kernels - GPU-FOR
// ============================================================================

__forceinline__ __device__ int64_t decodeElement64(int i, uint miniblock_index, uint index_into_miniblock,
                                                    uint64_t* data_block, uint* bitwidths, uint* offsets) {
    int64_t reference = reinterpret_cast<int64_t*>(data_block)[0];
    uint miniblock_offset = offsets[miniblock_index];
    uint bitwidth = bitwidths[miniblock_index];

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 2 + (start_bitindex >> 6);  // 64-bit word index
    start_bitindex = start_bitindex & 63;

    // Read two 64-bit words for cross-boundary handling
    uint64_t lo = data_block[miniblock_offset + start_intindex];
    uint64_t hi = data_block[miniblock_offset + start_intindex + 1];

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        element = (lo >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);
    } else {
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack64(uint* block_start, uint64_t* data,
    uint64_t* shared_buffer, int64_t (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    int tile_idx = blockIdx.x;
    int threadId = threadIdx.x;

    uint* block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    if (threadId < ITEMS_PER_THREAD + 1) {
        block_starts[threadIdx.x] = block_start[tile_idx * ITEMS_PER_THREAD + threadIdx.x];
    }
    __syncthreads();

    uint64_t* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

    uint start_offset = block_starts[0];
    uint end_offset = block_starts[ITEMS_PER_THREAD];
    uint data_size = end_offset - start_offset;
    for (uint i = 0; i < data_size; i += BLOCK_THREADS) {
        uint idx = i + threadIdx.x;
        if (idx < data_size) {
            data_block[idx] = data[start_offset + idx];
        }
    }
    __syncthreads();

    uint* bitwidths = reinterpret_cast<uint*>(&shared_buffer[ITEMS_PER_THREAD + 1]);
    uint* offsets = reinterpret_cast<uint*>(&shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)]);

    if (threadId < (ITEMS_PER_THREAD << 2)) {
        int i = threadId >> 2;
        int miniblock_index = threadId & 3;

        uint64_t miniblock_bitwidths = *(data_block + block_starts[i] - block_starts[0] + 1);
        uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 4)) & 0xFFFF;

        // For 64-bit: each miniblock of 32 elements occupies ceil(32*bw/64) 64-bit words
        // Must accumulate offset from all previous miniblocks
        uint miniblock_offset = 0;
        for (int m = 0; m < miniblock_index; m++) {
            uint prev_bw = (miniblock_bitwidths >> (m << 4)) & 0xFFFF;
            miniblock_offset += (32 * prev_bw + 63) / 64;
        }

        offsets[threadId] = miniblock_offset;
        bitwidths[threadId] = bitwidth;
    }
    __syncthreads();

    uint miniblock_index = threadIdx.x >> 5;
    uint index_into_miniblock = threadIdx.x & 31;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = decodeElement64(threadIdx.x, miniblock_index, index_into_miniblock,
                                   data_block + block_starts[i] - block_starts[0],
                                   bitwidths + (i << 2), offsets + (i << 2));
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runBinKernel64(int64_t* col, uint* col_block_start, uint64_t* col_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int64_t col_block[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint64_t shared_buffer64[];
    LoadBinPack64<BLOCK_THREADS, ITEMS_PER_THREAD>(
        col_block_start, col_data, shared_buffer64, col_block, is_last_tile, num_tile_items);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = col_block[i];
        }
    }
}

// ============================================================================
// 64-bit GPU Decompression Kernels - GPU-DFOR
// ============================================================================

__forceinline__ __device__ int64_t decodeElementDBin64(int i, uint64_t* data_block) {
    int64_t reference = reinterpret_cast<int64_t*>(data_block)[0];
    uint miniblock_index = i / 32;
    uint64_t miniblock_bitwidths = data_block[1];

    // For 64-bit: each miniblock of 32 elements occupies ceil(32*bw/64) 64-bit words
    uint miniblock_offset = 0;
    for (int j = 0; j < (int)miniblock_index; j++) {
        uint bw = (miniblock_bitwidths >> (j << 4)) & 0xFFFF;
        miniblock_offset += (32 * bw + 63) / 64;
    }

    uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 4)) & 0xFFFF;
    uint index_into_miniblock = i & 31;

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 2 + start_bitindex / 64;

    uint64_t lo = data_block[miniblock_offset + start_intindex];
    uint64_t hi = data_block[miniblock_offset + start_intindex + 1];
    start_bitindex = start_bitindex & 63;

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        element = (lo >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);
    } else {
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadDBinPack64(uint* block_start, uint64_t* data,
    uint64_t* shared_buffer, int64_t (&items)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int64_t, BLOCK_THREADS> BlockScan;

    int tile_idx = blockIdx.x;

    uint* block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    if (threadIdx.x < 5) {
        block_starts[threadIdx.x] = block_start[tile_idx * 4 + threadIdx.x];
    }
    __syncthreads();

    uint64_t* data_block = &shared_buffer[5];

    uint start_offset = block_starts[0] - 1;
    uint end_offset = block_starts[4];
    uint data_size = end_offset - start_offset;
    for (uint i = 0; i < data_size; i += BLOCK_THREADS) {
        uint idx = i + threadIdx.x;
        if (idx < data_size) {
            data_block[idx] = data[start_offset + idx];
        }
    }
    __syncthreads();

    int64_t first_value = data_block[0];
    data_block = data_block + 1;

    for (int i = 0; i < 4; i++) {
        if (is_last_tile) {
            if ((int)(threadIdx.x + i * 128) < num_tile_items) {
                items[i] = decodeElementDBin64(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
            }
        } else {
            items[i] = decodeElementDBin64(threadIdx.x, data_block + block_starts[i] - block_starts[0]);
        }
    }

    if (threadIdx.x == 0) {
        items[0] = first_value;
    }

    __syncthreads();

    typename BlockScan::TempStorage* temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(shared_buffer);
    typename BlockExchange::TempStorage* temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(shared_buffer);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items, items);
    __syncthreads();

    BlockExchange(*temp_storage_exchange).BlockedToStriped(items);
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runDBinKernel64(int64_t* col, uint* col_block_start, uint64_t* col_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int64_t col_block[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint64_t shared_buffer64[];
    LoadDBinPack64<BLOCK_THREADS, ITEMS_PER_THREAD>(
        col_block_start, col_data, shared_buffer64, col_block, is_last_tile, num_tile_items);

    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = col_block[i];
        }
    }
}

// ============================================================================
// 64-bit GPU Decompression Kernels - GPU-RFOR
// ============================================================================

__forceinline__ __device__ int64_t decodeElementRBin64(int i, uint64_t* data_block, uint64_t reference, uint bitwidth) {
    // Handle bitwidth=0 case: all values equal to reference
    if (bitwidth == 0) {
        return static_cast<int64_t>(reference);
    }

    uint start_bitindex = (bitwidth * i);
    uint start_intindex = (start_bitindex >> 6);
    start_bitindex = start_bitindex & 63;

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        // Value fits entirely within one word - only read lo
        uint64_t lo = data_block[start_intindex];
        element = (lo >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);
    } else {
        // Value crosses word boundary - need both lo and hi
        uint64_t lo = data_block[start_intindex];
        uint64_t hi = data_block[start_intindex + 1];
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

// 64-bit RFOR decoder - mirrors 32-bit LoadRBinPack exactly
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack64(uint* val_block_start, uint* rl_block_start,
    uint64_t* value, uint64_t* run_length, uint64_t* shared_buffer, int64_t (&items_value)[ITEMS_PER_THREAD],
    int64_t (&items_run_length)[ITEMS_PER_THREAD], bool is_last_tile, int num_tile_items) {

    typedef cub::BlockExchange<int64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int64_t, BLOCK_THREADS> BlockScan;

    uint num_decode;
    int tile_idx = blockIdx.x;
    const int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Extended buffer size for worst-case compressed data (metadata + packed data)
    // Max packed data size: ceil(tile_size * 64 / 64) + 3 metadata = tile_size + 3
    const int MAX_COMPRESSED_SIZE = tile_size + 64;  // Add margin for safety

    // Shared memory layout (in uint64_t units):
    // [0]: val_block_starts (2 uint32_t packed)
    // [1]: rl_block_starts (2 uint32_t packed)
    // [2..2+MAX_COMPRESSED_SIZE-1]: val_data_block
    // [2+MAX_COMPRESSED_SIZE..]: rl_data_block

    uint* val_block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    uint* rl_block_starts = reinterpret_cast<uint*>(&shared_buffer[1]);
    if (threadIdx.x < 2) {
        val_block_starts[threadIdx.x] = val_block_start[tile_idx + threadIdx.x];
        rl_block_starts[threadIdx.x] = rl_block_start[tile_idx + threadIdx.x];
    }
    __syncthreads();

    uint64_t* val_data_block = &shared_buffer[2];
    uint64_t* rl_data_block = &shared_buffer[2 + MAX_COMPRESSED_SIZE];

    uint start_offset_val = val_block_starts[0];
    uint end_offset_val = val_block_starts[1];
    uint start_offset_rl = rl_block_starts[0];
    uint end_offset_rl = rl_block_starts[1];

    uint val_data_size = end_offset_val - start_offset_val;
    uint rl_data_size = end_offset_rl - start_offset_rl;

    __syncthreads();

    // Load compressed data with extended range
    const int LOAD_ITERATIONS = (MAX_COMPRESSED_SIZE + BLOCK_THREADS - 1) / BLOCK_THREADS;
    for (int iter = 0; iter < LOAD_ITERATIONS; iter++) {
        uint local_idx = iter * BLOCK_THREADS + threadIdx.x;
        if (local_idx < val_data_size && local_idx < MAX_COMPRESSED_SIZE) {
            val_data_block[local_idx] = value[start_offset_val + local_idx];
        }
        if (local_idx < rl_data_size && local_idx < MAX_COMPRESSED_SIZE) {
            rl_data_block[local_idx] = run_length[start_offset_rl + local_idx];
        }
    }
    __syncthreads();

    uint count = val_data_block[2];
    uint offset = 0;
    num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint64_t* val_ptr = val_data_block + 3;
        uint64_t* rl_ptr = rl_data_block + 3;

        uint64_t reference;
        uint bitwidth;
        if (threadIdx.x < num_decode && threadIdx.x + offset < count) {
            reference = val_data_block[0];
            bitwidth = val_data_block[1] & 0xFF;
            items_value[i] = decodeElementRBin64(threadIdx.x + offset, val_ptr, reference, bitwidth);
            reference = rl_data_block[0];
            bitwidth = rl_data_block[1] & 0xFF;
            items_run_length[i] = decodeElementRBin64(threadIdx.x + offset, rl_ptr, reference, bitwidth);
        } else {
            items_value[i] = 0;
            items_run_length[i] = 0;
        }
        offset += num_decode;
    }
    __syncthreads();

    typename BlockScan::TempStorage* temp_storage_scan = reinterpret_cast<typename BlockScan::TempStorage*>(rl_data_block);
    typename BlockExchange::TempStorage* temp_storage_exchange = reinterpret_cast<typename BlockExchange::TempStorage*>(rl_data_block);

    BlockExchange(*temp_storage_exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        val_data_block[threadIdx.x * ITEMS_PER_THREAD + i] = 0;
    }
    __syncthreads();

    const int tile_size_64 = BLOCK_THREADS * ITEMS_PER_THREAD;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        // Boundary check: items_run_length[i] can equal tile_size for the last RLE element
        if (items_run_length[i] < tile_size_64) {
            val_data_block[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_run_length[i] = val_data_block[threadIdx.x * ITEMS_PER_THREAD + i];
    }
    __syncthreads();

    BlockScan(*temp_storage_scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    BlockExchange(*temp_storage_exchange).BlockedToStriped(items_run_length);
    __syncthreads();

    offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (threadIdx.x < num_decode) val_data_block[threadIdx.x + offset] = items_value[i];
        offset += num_decode;
    }
    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_value[i] = val_data_block[items_run_length[i]];
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void runRBinKernel64(int64_t* col, uint* val_block_start, uint64_t* val_data,
                                 uint* rl_block_start, uint64_t* rl_data, int num_entries) {
    int tile_size = BLOCK_THREADS * ITEMS_PER_THREAD;
    int tile_idx = blockIdx.x;
    int tile_offset = tile_idx * tile_size;

    int64_t items_value[ITEMS_PER_THREAD];
    int64_t items_run_length[ITEMS_PER_THREAD];

    int num_tiles = (num_entries + tile_size - 1) / tile_size;
    int num_tile_items = tile_size;
    bool is_last_tile = false;
    if (tile_idx == num_tiles - 1) {
        num_tile_items = num_entries - tile_offset;
        is_last_tile = true;
    }

    extern __shared__ uint64_t shared_buffer64[];
    LoadRBinPack64<BLOCK_THREADS, ITEMS_PER_THREAD>(
        val_block_start, rl_block_start, val_data, rl_data, shared_buffer64,
        items_value, items_run_length, is_last_tile, num_tile_items);

    __syncthreads();

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = tile_size * tile_idx + i * BLOCK_THREADS + threadIdx.x;
        if (idx < num_entries) {
            col[idx] = items_value[i];
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
T* load_binary_file(const char* filename, size_t& count) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Check if file has uint64_t count header (SOSD format)
    // If file size matches count_header + count*sizeof(T), use header
    uint64_t count_header;
    file.read(reinterpret_cast<char*>(&count_header), sizeof(uint64_t));

    size_t expected_size = sizeof(uint64_t) + count_header * sizeof(T);
    if (expected_size == file_size && count_header > 0 && count_header < (1ULL << 40)) {
        // File has valid count header
        count = count_header;
        T* data = new T[count];
        file.read(reinterpret_cast<char*>(data), count * sizeof(T));
        file.close();
        return data;
    }

    // No valid header, treat entire file as data
    file.seekg(0, std::ios::beg);
    count = file_size / sizeof(T);
    T* data = new T[count];
    file.read(reinterpret_cast<char*>(data), file_size);
    file.close();

    return data;
}

std::string extract_dataset_name(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    size_t last_dot = filename.find_last_of(".");
    return (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
}

void write_csv_header(std::ofstream& csv) {
    csv << "framework,algorithm,dataset,data_size_bytes,compressed_size_bytes,"
        << "compression_ratio,compress_time_ms,decompress_time_ms,"
        << "compress_throughput_gbps,decompress_throughput_gbps,verified\n";
}

void write_csv_result(std::ofstream& csv, const BenchmarkResult& result) {
    csv << std::fixed << std::setprecision(4)
        << result.framework << "," << result.algorithm << "," << result.dataset << ","
        << result.original_size << "," << result.compressed_size << ","
        << result.compression_ratio << "," << result.compress_time_ms << ","
        << result.decompress_time_ms << ","
        << result.compress_throughput_gbps << "," << result.decompress_throughput_gbps << ","
        << (result.verified ? "true" : "false") << "\n";
    csv.flush();
}

// ============================================================================
// Benchmark Functions
// ============================================================================

BenchmarkResult benchmark_gpufor(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-FOR";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t tile_size = 512;
    size_t aligned_n_tup = ((n_tup + tile_size - 1) / tile_size) * tile_size;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t block_size = 128;
    size_t num_blocks = aligned_n_tup / block_size;
    uint32_t* encoded_data = new uint32_t[aligned_n_tup * 2]();
    uint32_t* ofs_arr = new uint32_t[num_blocks + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    binPackEncode(aligned_data, encoded_data, ofs_arr, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t encoded_data_bsz = ofs_arr[num_blocks];

    // Upload to GPU
    uint* d_col_block_start;
    uint* d_col_data;
    int* d_col;

    cudaMalloc(&d_col_block_start, (num_blocks + 1) * sizeof(uint));
    cudaMalloc(&d_col_data, encoded_data_bsz * sizeof(uint));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int));

    cudaMemcpy(d_col_block_start, ofs_arr, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_data, encoded_data, encoded_data_bsz * sizeof(uint), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    size_t Ns = 8192;  // Increased for large bitwidth tiles

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int* temp = new int[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint32_t)temp[i]) {
            std::cerr << "GPU-FOR ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = encoded_data_bsz * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] ofs_arr;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_col_block_start);
    cudaFree(d_col_data);

    return result;
}

BenchmarkResult benchmark_gpudfor(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-DFOR";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t tile_size = 512;
    size_t aligned_n_tup = ((n_tup + tile_size - 1) / tile_size) * tile_size;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t block_size = 128;
    size_t num_blocks = aligned_n_tup / block_size;
    uint32_t* encoded_data = new uint32_t[aligned_n_tup * 2]();
    uint32_t* ofs_arr = new uint32_t[num_blocks + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    deltaBinPackEncode(aligned_data, encoded_data, ofs_arr, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t encoded_data_bsz = ofs_arr[num_blocks];

    // Upload to GPU
    uint* d_col_block_start;
    uint* d_col_data;
    int* d_col;

    cudaMalloc(&d_col_block_start, (num_blocks + 1) * sizeof(uint));
    cudaMalloc(&d_col_data, encoded_data_bsz * sizeof(uint));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int));

    cudaMemcpy(d_col_block_start, ofs_arr, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_data, encoded_data, encoded_data_bsz * sizeof(uint), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    size_t Ns = 8192;  // Increased for large bitwidth tiles

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runDBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runDBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int* temp = new int[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint32_t)temp[i]) {
            std::cerr << "GPU-DFOR ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = encoded_data_bsz * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] ofs_arr;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_col_block_start);
    cudaFree(d_col_data);

    return result;
}

BenchmarkResult benchmark_gpurfor(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-RFOR";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t block_size = 512;
    size_t aligned_n_tup = ((n_tup + block_size - 1) / block_size) * block_size;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t num_tiles = aligned_n_tup / block_size;
    // Allocate extra space for RLE encoding (worst case: each element is unique)
    size_t rle_buffer_size = aligned_n_tup * 2;
    uint32_t* value = new uint32_t[rle_buffer_size]();
    uint32_t* run_length = new uint32_t[rle_buffer_size]();
    uint32_t* val_offsets = new uint32_t[num_tiles + 1]();
    uint32_t* rl_offsets = new uint32_t[num_tiles + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    std::pair<uint32_t, uint32_t> sizes = rleBinPackEncode(aligned_data, value, run_length, val_offsets, rl_offsets, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t val_size = sizes.first;
    uint32_t rl_size = sizes.second;

    // Upload to GPU
    uint* d_val_block_start;
    uint* d_val_data;
    uint* d_rl_block_start;
    uint* d_rl_data;
    int* d_col;

    cudaMalloc(&d_val_block_start, (num_tiles + 1) * sizeof(uint));
    cudaMalloc(&d_val_data, val_size * sizeof(uint));
    cudaMalloc(&d_rl_block_start, (num_tiles + 1) * sizeof(uint));
    cudaMalloc(&d_rl_data, rl_size * sizeof(uint));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int));

    cudaMemcpy(d_val_block_start, val_offsets, (num_tiles + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_data, value, val_size * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl_block_start, rl_offsets, (num_tiles + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl_data, run_length, rl_size * sizeof(uint), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    int tile_size = block_size;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    // Increased shared memory for extended compressed data buffers
    // Layout: 4 + MAX_COMPRESSED_SIZE + MAX_COMPRESSED_SIZE where MAX_COMPRESSED_SIZE = tile_size + 64
    size_t Ns = (4 + 2 * (tile_size + 64)) * sizeof(uint) + 4096;  // Extra for CUB temp storage

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runRBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(
            d_col, d_val_block_start, d_val_data, d_rl_block_start, d_rl_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runRBinKernel<num_threads, items_per_thread><<<Dg, Db, Ns>>>(
            d_col, d_val_block_start, d_val_data, d_rl_block_start, d_rl_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int* temp = new int[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint32_t)temp[i]) {
            std::cerr << "GPU-RFOR ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = (val_size + rl_size) * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] value;
    delete[] run_length;
    delete[] val_offsets;
    delete[] rl_offsets;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_val_block_start);
    cudaFree(d_val_data);
    cudaFree(d_rl_block_start);
    cudaFree(d_rl_data);

    return result;
}

// ============================================================================
// 64-bit Benchmark Functions
// ============================================================================

BenchmarkResult benchmark_gpufor64(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-FOR-64";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t tile_size = 512;
    size_t aligned_n_tup = ((n_tup + tile_size - 1) / tile_size) * tile_size;

    uint64_t* aligned_data = new uint64_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint64_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t block_size = 128;
    size_t num_blocks = aligned_n_tup / block_size;
    uint64_t* encoded_data = new uint64_t[aligned_n_tup * 2]();
    uint32_t* ofs_arr = new uint32_t[num_blocks + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    binPackEncode64(aligned_data, encoded_data, ofs_arr, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t encoded_data_bsz = ofs_arr[num_blocks];

    // Upload to GPU
    uint* d_col_block_start;
    uint64_t* d_col_data;
    int64_t* d_col;

    cudaMalloc(&d_col_block_start, (num_blocks + 1) * sizeof(uint));
    cudaMalloc(&d_col_data, encoded_data_bsz * sizeof(uint64_t));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int64_t));

    cudaMemcpy(d_col_block_start, ofs_arr, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_data, encoded_data, encoded_data_bsz * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    size_t Ns = 16384;  // Larger for 64-bit

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int64_t* temp = new int64_t[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int64_t), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint64_t)temp[i]) {
            std::cerr << "GPU-FOR-64 ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint64_t);
    result.compressed_size = encoded_data_bsz * sizeof(uint64_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] ofs_arr;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_col_block_start);
    cudaFree(d_col_data);

    return result;
}

BenchmarkResult benchmark_gpudfor64(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-DFOR-64";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t tile_size = 512;
    size_t aligned_n_tup = ((n_tup + tile_size - 1) / tile_size) * tile_size;

    uint64_t* aligned_data = new uint64_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint64_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t block_size = 128;
    size_t num_blocks = aligned_n_tup / block_size;
    uint64_t* encoded_data = new uint64_t[aligned_n_tup * 2]();
    uint32_t* ofs_arr = new uint32_t[num_blocks + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    deltaBinPackEncode64(aligned_data, encoded_data, ofs_arr, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t encoded_data_bsz = ofs_arr[num_blocks];

    // Upload to GPU
    uint* d_col_block_start;
    uint64_t* d_col_data;
    int64_t* d_col;

    cudaMalloc(&d_col_block_start, (num_blocks + 1) * sizeof(uint));
    cudaMalloc(&d_col_data, encoded_data_bsz * sizeof(uint64_t));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int64_t));

    cudaMemcpy(d_col_block_start, ofs_arr, (num_blocks + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_data, encoded_data, encoded_data_bsz * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    size_t Ns = 16384;  // Larger for 64-bit

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runDBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runDBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(d_col, d_col_block_start, d_col_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int64_t* temp = new int64_t[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int64_t), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint64_t)temp[i]) {
            std::cerr << "GPU-DFOR-64 ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint64_t);
    result.compressed_size = encoded_data_bsz * sizeof(uint64_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] ofs_arr;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_col_block_start);
    cudaFree(d_col_data);

    return result;
}

BenchmarkResult benchmark_gpurfor64(const std::string& data_file, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "tile-gpu-compression";
    result.algorithm = "GPU-RFOR-64";
    result.dataset = extract_dataset_name(data_file);

    // Load data
    size_t n_tup = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to 512 (tile size)
    size_t block_size = 512;
    size_t aligned_n_tup = ((n_tup + block_size - 1) / block_size) * block_size;

    uint64_t* aligned_data = new uint64_t[aligned_n_tup];
    memcpy(aligned_data, original_data, n_tup * sizeof(uint64_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t num_tiles = aligned_n_tup / block_size;
    // Allocate extra space for RLE encoding (worst case: each element is unique)
    size_t rle_buffer_size = aligned_n_tup * 2;
    uint64_t* value = new uint64_t[rle_buffer_size]();
    uint64_t* run_length = new uint64_t[rle_buffer_size]();
    uint32_t* val_offsets = new uint32_t[num_tiles + 1]();
    uint32_t* rl_offsets = new uint32_t[num_tiles + 1]();

    // Measure compression time
    auto encode_start = std::chrono::high_resolution_clock::now();
    std::pair<uint32_t, uint32_t> sizes = rleBinPackEncode64(aligned_data, value, run_length, val_offsets, rl_offsets, aligned_n_tup);
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    uint32_t val_size = sizes.first;
    uint32_t rl_size = sizes.second;

    // Upload to GPU
    uint* d_val_block_start;
    uint64_t* d_val_data;
    uint* d_rl_block_start;
    uint64_t* d_rl_data;
    int64_t* d_col;

    cudaMalloc(&d_val_block_start, (num_tiles + 1) * sizeof(uint));
    cudaMalloc(&d_val_data, val_size * sizeof(uint64_t));
    cudaMalloc(&d_rl_block_start, (num_tiles + 1) * sizeof(uint));
    cudaMalloc(&d_rl_data, rl_size * sizeof(uint64_t));
    cudaMalloc(&d_col, aligned_n_tup * sizeof(int64_t));

    cudaMemcpy(d_val_block_start, val_offsets, (num_tiles + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val_data, value, val_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl_block_start, rl_offsets, (num_tiles + 1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rl_data, run_length, rl_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    const int num_threads = 128;
    const int items_per_thread = 4;
    int tile_size = block_size;
    size_t Dg = (aligned_n_tup + tile_size - 1) / tile_size;
    size_t Db = num_threads;
    // Increased shared memory for extended compressed data buffers (64-bit)
    // Layout: 2 + MAX_COMPRESSED_SIZE + MAX_COMPRESSED_SIZE where MAX_COMPRESSED_SIZE = tile_size * 2
    // Shared memory for 64-bit RFOR: 2 + 2*(tile_size+64) uint64_t + CUB temp storage
    size_t Ns = (2 + 2 * (tile_size + 64)) * sizeof(uint64_t) + 8192;  // Extra for CUB temp storage

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        runRBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(
            d_col, d_val_block_start, d_val_data, d_rl_block_start, d_rl_data, aligned_n_tup);
        cudaDeviceSynchronize();
    }

    // Measure decompression time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start);
        runRBinKernel64<num_threads, items_per_thread><<<Dg, Db, Ns>>>(
            d_col, d_val_block_start, d_val_data, d_rl_block_start, d_rl_data, aligned_n_tup);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float trial_time = 0.0f;
        cudaEventElapsedTime(&trial_time, start, stop);
        total_time += trial_time;
    }

    float avg_decode_time_ms = total_time / num_trials;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Verify
    int64_t* temp = new int64_t[aligned_n_tup];
    cudaMemcpy(temp, d_col, aligned_n_tup * sizeof(int64_t), cudaMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < n_tup && correct; i++) {
        if (original_data[i] != (uint64_t)temp[i]) {
            std::cerr << "GPU-RFOR-64 ERROR at " << i << ": " << original_data[i] << " vs " << temp[i] << std::endl;
            correct = false;
        }
    }

    // Calculate metrics
    result.original_size = n_tup * sizeof(uint64_t);
    result.compressed_size = (val_size + rl_size) * sizeof(uint64_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    result.verified = correct;

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] value;
    delete[] run_length;
    delete[] val_offsets;
    delete[] rl_offsets;
    delete[] temp;
    cudaFree(d_col);
    cudaFree(d_val_block_start);
    cudaFree(d_val_data);
    cudaFree(d_rl_block_start);
    cudaFree(d_rl_data);

    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -d, --data_dir <path>    Data directory (default: /root/autodl-tmp/test/data/sosd/)\n"
              << "  -o, --output <file>      Output CSV file (default: reports/tilegpu_results.csv)\n"
              << "  -a, --algorithm <alg>    Algorithm: all|all64|gpufor|gpudfor|gpurfor|gpufor64|gpudfor64|gpurfor64 (default: all)\n"
              << "  -f, --file <filename>    Specific dataset file (can use multiple times)\n"
              << "  -n, --trials <num>       Number of trials (default: 10)\n"
              << "  -w, --warmup <num>       Warmup iterations (default: 3)\n"
              << "  -g, --gpu <id>           GPU device ID (default: 0)\n"
              << "  -h, --help               Show this help message\n";
}

int main(int argc, char** argv) {
    // Default parameters
    std::string data_dir = "/root/autodl-tmp/test/data/sosd/";
    std::string output_file = "reports/tilegpu_results.csv";
    std::string algorithm = "all";
    std::vector<std::string> specific_files;
    int num_trials = 10;
    int warmup_count = 3;
    int gpu_id = 0;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-d" || arg == "--data_dir") && i + 1 < argc) {
            data_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            algorithm = argv[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            specific_files.push_back(argv[++i]);
        } else if ((arg == "-n" || arg == "--trials") && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup_count = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        }
    }

    cudaSetDevice(gpu_id);

    {
        std::filesystem::path out_path(output_file);
        if (out_path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(out_path.parent_path(), ec);
            if (ec) {
                std::cerr << "Failed to create output directory: " << out_path.parent_path()
                          << " (" << ec.message() << ")\n";
                return 1;
            }
        }
    }

    // Get list of datasets
    std::vector<std::string> datasets;
    if (!specific_files.empty()) {
        // Combine data_dir with specific file names
        for (const auto& file : specific_files) {
            // If file already has a path, use it as-is
            if (file.find('/') != std::string::npos) {
                datasets.push_back(file);
            } else {
                datasets.push_back(data_dir + file);
            }
        }
    } else {
        // Default 32-bit datasets
        std::vector<std::string> default_datasets = {
            "1-linear_200M_uint32.bin",
            "2-normal_200M_uint32.bin",
            "5-books_200M_uint32.bin",
            "9-movieid_uint32.bin",
            "14-cosmos_int32.bin",
            "18-site_250k_uint32.bin",
            "19-weight_25k_uint32.bin",
            "20-adult_30k_uint32.bin"
        };
        for (const auto& ds : default_datasets) {
            datasets.push_back(data_dir + ds);
        }
    }

    // Open output file
    std::ofstream csv_file;
    bool file_exists = std::ifstream(output_file).good();
    csv_file.open(output_file, std::ios::app);

    if (!file_exists) {
        write_csv_header(csv_file);
    }

    std::cout << "tile-gpu-compression Benchmark\n";
    std::cout << "===============================\n";
    std::cout << "Algorithms: " << algorithm << "\n";
    std::cout << "Trials: " << num_trials << "\n";
    std::cout << "Warmup: " << warmup_count << "\n";
    std::cout << "GPU: " << gpu_id << "\n";
    std::cout << "Datasets: " << datasets.size() << "\n\n";

    // Run benchmarks
    for (const auto& data_file : datasets) {
        std::cout << "Processing: " << data_file << std::endl;

        // Check if file exists
        std::ifstream test_file(data_file);
        if (!test_file.good()) {
            std::cerr << "  File not found, skipping.\n";
            continue;
        }
        test_file.close();

        if (algorithm == "all" || algorithm == "gpufor") {
            std::cout << "  Running GPU-FOR..." << std::flush;
            BenchmarkResult result = benchmark_gpufor(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "gpudfor") {
            std::cout << "  Running GPU-DFOR..." << std::flush;
            BenchmarkResult result = benchmark_gpudfor(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "gpurfor") {
            std::cout << "  Running GPU-RFOR..." << std::flush;
            BenchmarkResult result = benchmark_gpurfor(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        // 64-bit algorithms
        if (algorithm == "all64" || algorithm == "gpufor64") {
            std::cout << "  Running GPU-FOR-64..." << std::flush;
            BenchmarkResult result = benchmark_gpufor64(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all64" || algorithm == "gpudfor64") {
            std::cout << "  Running GPU-DFOR-64..." << std::flush;
            BenchmarkResult result = benchmark_gpudfor64(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all64" || algorithm == "gpurfor64") {
            std::cout << "  Running GPU-RFOR-64..." << std::flush;
            BenchmarkResult result = benchmark_gpurfor64(data_file, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nResults written to: " << output_file << std::endl;

    return 0;
}
