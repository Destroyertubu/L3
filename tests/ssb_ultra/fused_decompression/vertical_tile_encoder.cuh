/**
 * @file Vertical_tile_encoder.cuh
 * @brief Vertical-compatible tile encoder for FOR-bitpack columns
 *
 * Produces packed data in Vertical GPU format:
 * - 1024 values per tile (32 threads × 32 items per thread)
 * - Tile offset = tile_idx * bitwidth * 32 (words)
 * - Thread i reads from: in[i], in[32+i], in[64+i], ...
 *
 * This encoder is designed for SSB columns which are 100% FOR model.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>

namespace Vertical_tile {

constexpr int TILE_SIZE = 1024;
constexpr int BLOCK_THREADS = 32;
constexpr int ITEMS_PER_THREAD = 32;

/**
 * @brief Column metadata for Vertical tile format
 */
struct ColumnMeta {
    uint32_t base_value;     // FOR base (min value)
    uint8_t bitwidth;        // Bits per delta
    int32_t num_tiles;       // Number of complete 1024-element tiles
    int32_t num_elements;    // Total elements
    int64_t packed_words;    // Total words in packed data
};

/**
 * @brief Compute required bitwidth for a range of values
 */
inline int computeBitwidth(uint32_t max_delta) {
    if (max_delta == 0) return 0;
    return 32 - __builtin_clz(max_delta);
}

// ============================================================================
// Host-side packing functions (adapted from Vertical)
// ============================================================================

/**
 * @brief Pack 1024 values with given bitwidth
 *
 * Input: 1024 uint32_t values (deltas from base)
 * Output: bitwidth * 32 uint32_t words
 *
 * Layout matches Vertical GPU unpack expectations:
 * - Thread i reads words at positions: i, 32+i, 64+i, ...
 * - Each word contains multiple values packed together
 */
inline void packTile(const uint32_t* in, uint32_t* out, int bitwidth) {
    if (bitwidth == 0) {
        // No data to write for 0-bit
        return;
    }

    // Clear output
    int out_words = bitwidth * 32;
    std::fill(out, out + out_words, 0);

    // Pack using Vertical layout
    // For bitwidth bw, thread i's word j is at out[(j * 32) + i]
    // and contains values at input positions that map to thread i's j-th group

    for (int i = 0; i < 32; i++) {
        // Thread i processes input values at positions: i, 32+i, 64+i, ...
        int bit_pos = 0;
        int word_idx = 0;

        for (int v = 0; v < ITEMS_PER_THREAD; v++) {
            int input_idx = v * 32 + i;  // Input index for thread i's v-th value
            uint32_t val = in[input_idx] & ((1ULL << bitwidth) - 1);

            // Pack into output
            int out_pos = word_idx * 32 + i;
            out[out_pos] |= (val << bit_pos);

            bit_pos += bitwidth;
            if (bit_pos >= 32) {
                // Overflow to next word
                int overflow_bits = bit_pos - 32;
                word_idx++;
                if (overflow_bits > 0 && word_idx < bitwidth) {
                    out_pos = word_idx * 32 + i;
                    out[out_pos] |= (val >> (bitwidth - overflow_bits));
                }
                bit_pos = overflow_bits;
            }
        }
    }
}

/**
 * @brief Encode a column in Vertical tile format
 *
 * @param raw_data Original column data
 * @param num_elements Number of elements
 * @param packed_data Output: packed data (must be pre-allocated)
 * @param meta Output: column metadata
 */
inline void encodeColumn(
    const uint32_t* raw_data,
    int num_elements,
    uint32_t* packed_data,
    ColumnMeta& meta)
{
    // Step 1: Find min/max to determine base and bitwidth
    uint32_t min_val = raw_data[0];
    uint32_t max_val = raw_data[0];

    for (int i = 1; i < num_elements; i++) {
        min_val = std::min(min_val, raw_data[i]);
        max_val = std::max(max_val, raw_data[i]);
    }

    uint32_t max_delta = max_val - min_val;
    int bitwidth = computeBitwidth(max_delta);

    // Store metadata
    meta.base_value = min_val;
    meta.bitwidth = bitwidth;
    meta.num_elements = num_elements;
    meta.num_tiles = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
    meta.packed_words = static_cast<int64_t>(meta.num_tiles) * bitwidth * 32;

    // Step 2: Pack each tile
    std::vector<uint32_t> tile_deltas(TILE_SIZE);

    int64_t out_offset = 0;
    for (int tile = 0; tile < meta.num_tiles; tile++) {
        int tile_start = tile * TILE_SIZE;
        int tile_end = std::min(tile_start + TILE_SIZE, num_elements);
        int tile_size = tile_end - tile_start;

        // Compute deltas for this tile
        for (int i = 0; i < tile_size; i++) {
            tile_deltas[i] = raw_data[tile_start + i] - min_val;
        }
        // Pad with zeros if needed
        for (int i = tile_size; i < TILE_SIZE; i++) {
            tile_deltas[i] = 0;
        }

        // Pack the tile
        packTile(tile_deltas.data(), packed_data + out_offset, bitwidth);
        out_offset += bitwidth * 32;
    }
}

/**
 * @brief Calculate packed size for a column
 */
inline int64_t calcPackedSize(int num_elements, int bitwidth) {
    int num_tiles = (num_elements + TILE_SIZE - 1) / TILE_SIZE;
    return static_cast<int64_t>(num_tiles) * bitwidth * 32;
}

// ============================================================================
// GPU data structure for Q1.1
// ============================================================================

/**
 * @brief Query metadata for Q1.1 Vertical-style execution
 */
struct Q11VerticalMeta {
    // Orderdate
    uint8_t od_bw;
    uint32_t od_base;

    // Discount
    uint8_t disc_bw;
    uint32_t disc_base;

    // Quantity
    uint8_t qty_bw;
    uint32_t qty_base;

    // Extendedprice
    uint8_t price_bw;
    uint32_t price_base;

    // Common
    int32_t num_tiles;
    int32_t num_elements;
};

/**
 * @brief GPU-side packed column data
 */
struct SSBColumnPacked {
    uint32_t* d_data;      // Device pointer to packed data
    ColumnMeta meta;       // Column metadata

    void allocate(int64_t words) {
        cudaMalloc(&d_data, words * sizeof(uint32_t));
    }

    void upload(const uint32_t* host_data) {
        cudaMemcpy(d_data, host_data,
                   meta.packed_words * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);
    }

    void free() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }
};

} // namespace Vertical_tile
