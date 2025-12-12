/**
 * @file fls_constants.cuh
 * @brief Vertical-style configuration constants for L3 fused queries
 *
 * This file defines the configuration to match Vertical-GPU BitPack performance.
 * Key: Use 32 threads x 32 items = 1024 elements per tile (same as Vertical).
 */

#pragma once

#include <cstdint>

namespace l3_fls {

// ============================================================================
// Tile Configuration (Match Vertical exactly)
// ============================================================================

// Vertical uses 32 threads, each processing 32 items = 1024 elements per tile
constexpr int FLS_BLOCK_THREADS = 32;
constexpr int FLS_ITEMS_PER_THREAD = 32;
constexpr int FLS_TILE_SIZE = FLS_BLOCK_THREADS * FLS_ITEMS_PER_THREAD;  // 1024

// Memory layout: tile N starts at offset N * bitwidth * 32
// Each tile contains bitwidth * 32 uint32_t words

// ============================================================================
// Alternative Configuration (128 threads x 4 items = 512)
// Use this if 32x32 causes register pressure issues
// ============================================================================

constexpr int ALT_BLOCK_THREADS = 128;
constexpr int ALT_ITEMS_PER_THREAD = 4;
constexpr int ALT_TILE_SIZE = ALT_BLOCK_THREADS * ALT_ITEMS_PER_THREAD;  // 512

// ============================================================================
// SSB Query Constants
// ============================================================================

// Q1.1 predicate bounds
constexpr int Q11_ORDERDATE_LO = 19930101;  // lo_orderdate >= 19930101
constexpr int Q11_ORDERDATE_HI = 19931231;  // lo_orderdate <= 19931231
constexpr int Q11_DISCOUNT_LO = 1;          // lo_discount >= 1
constexpr int Q11_DISCOUNT_HI = 3;          // lo_discount <= 3
constexpr int Q11_QUANTITY_HI = 25;         // lo_quantity < 25

// Q1.2 predicate bounds
constexpr int Q12_ORDERDATE_LO = 199401;    // lo_orderdate >= 199401 (YYYYMM)
constexpr int Q12_ORDERDATE_HI = 199412;    // lo_orderdate <= 199412
constexpr int Q12_DISCOUNT_LO = 4;          // lo_discount >= 4
constexpr int Q12_DISCOUNT_HI = 6;          // lo_discount <= 6
constexpr int Q12_QUANTITY_LO = 26;         // lo_quantity >= 26
constexpr int Q12_QUANTITY_HI = 35;         // lo_quantity <= 35

// Q1.3 predicate bounds
constexpr int Q13_ORDERDATE_LO = 19940101;  // lo_orderdate >= 19940101
constexpr int Q13_ORDERDATE_HI = 19940131;  // specific week
constexpr int Q13_DISCOUNT_LO = 5;          // lo_discount >= 5
constexpr int Q13_DISCOUNT_HI = 7;          // lo_discount <= 7
constexpr int Q13_QUANTITY_LO = 26;         // lo_quantity >= 26
constexpr int Q13_QUANTITY_HI = 35;         // lo_quantity <= 35

// ============================================================================
// Compression Metadata Structure
// ============================================================================

// FOR+BitPack column metadata (matches Vertical SSB structure)
struct ColumnMetadataFLS {
    int32_t* d_encoded_data;     // Pointer to bit-packed data on GPU
    int32_t min_value;           // Base value for FOR encoding
    int32_t max_value;           // Max value (for statistics)
    uint8_t bitwidth;            // Bits per delta value
    int64_t num_values;          // Total number of values
    int64_t num_tiles;           // Number of 1024-element tiles
};

// Query metadata structure
struct SSBQueryMetadata {
    int64_t n_tup_line_order;    // Total tuples in lineorder table
    int64_t n_tiles;             // Number of tiles

    // Column metadata
    ColumnMetadataFLS lo_orderdate;
    ColumnMetadataFLS lo_discount;
    ColumnMetadataFLS lo_quantity;
    ColumnMetadataFLS lo_extendedprice;
    ColumnMetadataFLS lo_revenue;
    ColumnMetadataFLS lo_supplycost;

    // Foreign keys
    ColumnMetadataFLS lo_partkey;
    ColumnMetadataFLS lo_suppkey;
    ColumnMetadataFLS lo_custkey;
    ColumnMetadataFLS lo_orderkey;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Calculate tile offset in compressed array
__host__ __device__ __forceinline__
int64_t calcTileOffset(int tile_idx, int bitwidth) {
    return (int64_t)tile_idx * bitwidth * 32;
}

// Calculate number of tiles for a given data size
__host__ __forceinline__
int64_t calcNumTiles(int64_t num_elements) {
    return (num_elements + FLS_TILE_SIZE - 1) / FLS_TILE_SIZE;
}

// Calculate compressed array size in uint32_t words
__host__ __forceinline__
int64_t calcCompressedSize(int64_t num_tiles, int bitwidth) {
    return num_tiles * bitwidth * 32;
}

}  // namespace l3_fls
