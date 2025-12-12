/**
 * @file tile_metadata.cuh
 * @brief Tile Metadata for L3 Fused Decompression + Query
 *
 * Pre-computes per-tile metadata to avoid per-element partition lookups.
 * Each 1024-element tile gets metadata for all columns involved in the query.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "L3_Vertical_format.hpp"

namespace l3_fused {

// Tile size matches Vertical: 32 threads × 32 items
constexpr int TILE_SIZE = 1024;
constexpr int WARP_SIZE = 32;
constexpr int ITEMS_PER_THREAD = 32;

// Model type constants (match L3_format.hpp)
constexpr int MODEL_CONSTANT = 0;
constexpr int MODEL_LINEAR = 1;
constexpr int MODEL_POLYNOMIAL2 = 2;
constexpr int MODEL_POLYNOMIAL3 = 3;
constexpr int MODEL_FOR_BITPACK = 4;
constexpr int MODEL_DIRECT_COPY = 5;

/**
 * @brief Per-column metadata for a single tile
 *
 * Contains all information needed to decompress values in this tile
 * without looking up partition boundaries.
 *
 * INTERLEAVED FORMAT (256-value mini-vectors):
 *   For each mini-vector: Lane 0's 8 values, Lane 1's 8 values, ..., Lane 31's 8 values
 *   Lane L has values at indices: L, L+32, L+64, L+96, L+128, L+160, L+192, L+224
 *
 *   Bit offset for value at local_idx within partition:
 *     mv_idx = local_idx / 256
 *     local_in_mv = local_idx % 256
 *     lane = local_in_mv % 32
 *     val_idx = local_in_mv / 32
 *     bit_offset = partition_bit_base + mv_idx * 256 * delta_bits + lane * 8 * delta_bits + val_idx * delta_bits
 */
struct ColumnTileMeta {
    int32_t partition_id;      // Which L3 partition this tile falls in
    int32_t model_type;        // MODEL_FOR_BITPACK, MODEL_LINEAR, etc.
    int32_t delta_bits;        // Bit width for deltas
    int64_t partition_bit_base;// Bit offset to start of this partition's interleaved data
    double params[4];          // Polynomial coefficients (a0, a1, a2, a3)
    int32_t partition_start;   // Global index where partition starts
    int32_t local_offset;      // tile_start - partition_start (for polynomial index)
    bool is_FOR;               // Fast path: MODEL_FOR_BITPACK with base only
    bool spans_partition;      // True if tile crosses partition boundary
};

/**
 * @brief Unified tile metadata for Q1.1 (4 columns)
 *
 * Pre-computed for each 1024-element tile.
 */
struct Q11TileMetadata {
    ColumnTileMeta orderdate;
    ColumnTileMeta discount;
    ColumnTileMeta quantity;
    ColumnTileMeta extendedprice;

    // Combined flags for fast path selection
    bool all_FOR;              // All 4 columns use FOR in this tile
};

/**
 * @brief Q2.x Tile Metadata (4 columns: suppkey, partkey, orderdate, revenue)
 */
struct Q2xTileMetadata {
    ColumnTileMeta suppkey;
    ColumnTileMeta partkey;
    ColumnTileMeta orderdate;
    ColumnTileMeta revenue;
    bool all_FOR;
};

/**
 * @brief Q3.x Tile Metadata (4 columns: custkey, suppkey, orderdate, revenue)
 */
struct Q3xTileMetadata {
    ColumnTileMeta custkey;
    ColumnTileMeta suppkey;
    ColumnTileMeta orderdate;
    ColumnTileMeta revenue;
    bool all_FOR;
};

/**
 * @brief Q4.x Tile Metadata (6 columns: custkey, suppkey, partkey, orderdate, revenue, supplycost)
 */
struct Q4xTileMetadata {
    ColumnTileMeta custkey;
    ColumnTileMeta suppkey;
    ColumnTileMeta partkey;
    ColumnTileMeta orderdate;
    ColumnTileMeta revenue;
    ColumnTileMeta supplycost;
    bool all_FOR;
};

/**
 * @brief Host-side tile metadata builder
 *
 * Analyzes L3 partition structure and pre-computes tile metadata.
 */
template<typename T>
class TileMetadataBuilder {
public:
    /**
     * @brief Build tile metadata for a single column
     *
     * @param compressed L3 compressed column data
     * @param total_values Number of values in the column
     * @return Vector of ColumnTileMeta, one per tile
     */
    static std::vector<ColumnTileMeta> buildColumnMetadata(
        const CompressedDataVertical<T>& compressed,
        int total_values)
    {
        int num_tiles = (total_values + TILE_SIZE - 1) / TILE_SIZE;
        std::vector<ColumnTileMeta> result(num_tiles);

        // Download partition boundaries to host
        std::vector<int32_t> h_start_indices(compressed.num_partitions);
        std::vector<int32_t> h_end_indices(compressed.num_partitions);
        std::vector<int32_t> h_model_types(compressed.num_partitions);
        std::vector<int32_t> h_delta_bits(compressed.num_partitions);
        std::vector<int64_t> h_interleaved_offsets(compressed.num_partitions);
        std::vector<double> h_model_params(compressed.num_partitions * 4);

        cudaMemcpy(h_start_indices.data(), compressed.d_start_indices,
                   compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_end_indices.data(), compressed.d_end_indices,
                   compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_model_types.data(), compressed.d_model_types,
                   compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
                   compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_interleaved_offsets.data(), compressed.d_interleaved_offsets,
                   compressed.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_model_params.data(), compressed.d_model_params,
                   compressed.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);

        // Build metadata for each tile
        for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
            int tile_start = tile_idx * TILE_SIZE;
            int tile_end = std::min(tile_start + TILE_SIZE, total_values);

            // Binary search for partition containing tile_start
            int pid = findPartition(h_start_indices, h_end_indices, tile_start);

            ColumnTileMeta& meta = result[tile_idx];
            meta.partition_id = pid;
            meta.model_type = h_model_types[pid];
            meta.delta_bits = h_delta_bits[pid];
            meta.partition_start = h_start_indices[pid];
            meta.local_offset = tile_start - meta.partition_start;

            // Copy polynomial parameters
            for (int i = 0; i < 4; i++) {
                meta.params[i] = h_model_params[pid * 4 + i];
            }

            // Store partition bit base for interleaved offset calculation
            // Interleaved offset is in words (32-bit), convert to bits
            meta.partition_bit_base = h_interleaved_offsets[pid] * 32;

            // Check if tile spans partition boundary
            meta.spans_partition = (tile_end > h_end_indices[pid]);

            // Fast path detection
            meta.is_FOR = (meta.model_type == MODEL_FOR_BITPACK) && !meta.spans_partition;
        }

        return result;
    }

    /**
     * @brief Build Q1.1 tile metadata for all 4 columns
     */
    static std::vector<Q11TileMetadata> buildQ11Metadata(
        const CompressedDataVertical<T>& orderdate,
        const CompressedDataVertical<T>& discount,
        const CompressedDataVertical<T>& quantity,
        const CompressedDataVertical<T>& extendedprice,
        int total_values)
    {
        auto od_meta = buildColumnMetadata(orderdate, total_values);
        auto disc_meta = buildColumnMetadata(discount, total_values);
        auto qty_meta = buildColumnMetadata(quantity, total_values);
        auto price_meta = buildColumnMetadata(extendedprice, total_values);

        int num_tiles = od_meta.size();
        std::vector<Q11TileMetadata> result(num_tiles);

        for (int i = 0; i < num_tiles; i++) {
            result[i].orderdate = od_meta[i];
            result[i].discount = disc_meta[i];
            result[i].quantity = qty_meta[i];
            result[i].extendedprice = price_meta[i];

            // Check if all columns use FOR for fast path
            result[i].all_FOR = od_meta[i].is_FOR &&
                                disc_meta[i].is_FOR &&
                                qty_meta[i].is_FOR &&
                                price_meta[i].is_FOR;
        }

        return result;
    }

    /**
     * @brief Build Q2.x tile metadata (suppkey, partkey, orderdate, revenue)
     */
    static std::vector<Q2xTileMetadata> buildQ2xMetadata(
        const CompressedDataVertical<T>& suppkey,
        const CompressedDataVertical<T>& partkey,
        const CompressedDataVertical<T>& orderdate,
        const CompressedDataVertical<T>& revenue,
        int total_values)
    {
        auto supp_meta = buildColumnMetadata(suppkey, total_values);
        auto part_meta = buildColumnMetadata(partkey, total_values);
        auto od_meta = buildColumnMetadata(orderdate, total_values);
        auto rev_meta = buildColumnMetadata(revenue, total_values);

        int num_tiles = supp_meta.size();
        std::vector<Q2xTileMetadata> result(num_tiles);

        for (int i = 0; i < num_tiles; i++) {
            result[i].suppkey = supp_meta[i];
            result[i].partkey = part_meta[i];
            result[i].orderdate = od_meta[i];
            result[i].revenue = rev_meta[i];

            result[i].all_FOR = supp_meta[i].is_FOR &&
                                part_meta[i].is_FOR &&
                                od_meta[i].is_FOR &&
                                rev_meta[i].is_FOR;
        }

        return result;
    }

    /**
     * @brief Build Q3.x tile metadata (custkey, suppkey, orderdate, revenue)
     */
    static std::vector<Q3xTileMetadata> buildQ3xMetadata(
        const CompressedDataVertical<T>& custkey,
        const CompressedDataVertical<T>& suppkey,
        const CompressedDataVertical<T>& orderdate,
        const CompressedDataVertical<T>& revenue,
        int total_values)
    {
        auto cust_meta = buildColumnMetadata(custkey, total_values);
        auto supp_meta = buildColumnMetadata(suppkey, total_values);
        auto od_meta = buildColumnMetadata(orderdate, total_values);
        auto rev_meta = buildColumnMetadata(revenue, total_values);

        int num_tiles = cust_meta.size();
        std::vector<Q3xTileMetadata> result(num_tiles);

        for (int i = 0; i < num_tiles; i++) {
            result[i].custkey = cust_meta[i];
            result[i].suppkey = supp_meta[i];
            result[i].orderdate = od_meta[i];
            result[i].revenue = rev_meta[i];

            result[i].all_FOR = cust_meta[i].is_FOR &&
                                supp_meta[i].is_FOR &&
                                od_meta[i].is_FOR &&
                                rev_meta[i].is_FOR;
        }

        return result;
    }

    /**
     * @brief Build Q4.x tile metadata (custkey, suppkey, partkey, orderdate, revenue, supplycost)
     */
    static std::vector<Q4xTileMetadata> buildQ4xMetadata(
        const CompressedDataVertical<T>& custkey,
        const CompressedDataVertical<T>& suppkey,
        const CompressedDataVertical<T>& partkey,
        const CompressedDataVertical<T>& orderdate,
        const CompressedDataVertical<T>& revenue,
        const CompressedDataVertical<T>& supplycost,
        int total_values)
    {
        auto cust_meta = buildColumnMetadata(custkey, total_values);
        auto supp_meta = buildColumnMetadata(suppkey, total_values);
        auto part_meta = buildColumnMetadata(partkey, total_values);
        auto od_meta = buildColumnMetadata(orderdate, total_values);
        auto rev_meta = buildColumnMetadata(revenue, total_values);
        auto cost_meta = buildColumnMetadata(supplycost, total_values);

        int num_tiles = cust_meta.size();
        std::vector<Q4xTileMetadata> result(num_tiles);

        for (int i = 0; i < num_tiles; i++) {
            result[i].custkey = cust_meta[i];
            result[i].suppkey = supp_meta[i];
            result[i].partkey = part_meta[i];
            result[i].orderdate = od_meta[i];
            result[i].revenue = rev_meta[i];
            result[i].supplycost = cost_meta[i];

            result[i].all_FOR = cust_meta[i].is_FOR &&
                                supp_meta[i].is_FOR &&
                                part_meta[i].is_FOR &&
                                od_meta[i].is_FOR &&
                                rev_meta[i].is_FOR &&
                                cost_meta[i].is_FOR;
        }

        return result;
    }

private:
    /**
     * @brief Binary search for partition containing a global index
     */
    static int findPartition(const std::vector<int32_t>& starts,
                              const std::vector<int32_t>& ends,
                              int global_idx)
    {
        int left = 0;
        int right = starts.size() - 1;

        while (left < right) {
            int mid = (left + right) / 2;
            if (ends[mid] <= global_idx) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        return left;
    }
};

/**
 * @brief GPU-side tile metadata (device arrays)
 */
struct Q11TileMetadataGPU {
    Q11TileMetadata* d_tiles;
    int num_tiles;

    void allocate(int n) {
        num_tiles = n;
        cudaMalloc(&d_tiles, n * sizeof(Q11TileMetadata));
    }

    void upload(const std::vector<Q11TileMetadata>& host_data) {
        cudaMemcpy(d_tiles, host_data.data(),
                   host_data.size() * sizeof(Q11TileMetadata),
                   cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_tiles);
        d_tiles = nullptr;
        num_tiles = 0;
    }
};

// ============================================================================
// Q2.x GPU Tile Metadata
// ============================================================================

struct Q2xTileMetadataGPU {
    Q2xTileMetadata* d_tiles;
    int num_tiles;

    void allocate(int n) {
        num_tiles = n;
        cudaMalloc(&d_tiles, n * sizeof(Q2xTileMetadata));
    }

    void upload(const std::vector<Q2xTileMetadata>& host_data) {
        cudaMemcpy(d_tiles, host_data.data(),
                   host_data.size() * sizeof(Q2xTileMetadata),
                   cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_tiles);
        d_tiles = nullptr;
        num_tiles = 0;
    }
};

// ============================================================================
// Q3.x GPU Tile Metadata
// ============================================================================

struct Q3xTileMetadataGPU {
    Q3xTileMetadata* d_tiles;
    int num_tiles;

    void allocate(int n) {
        num_tiles = n;
        cudaMalloc(&d_tiles, n * sizeof(Q3xTileMetadata));
    }

    void upload(const std::vector<Q3xTileMetadata>& host_data) {
        cudaMemcpy(d_tiles, host_data.data(),
                   host_data.size() * sizeof(Q3xTileMetadata),
                   cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_tiles);
        d_tiles = nullptr;
        num_tiles = 0;
    }
};

// ============================================================================
// Q4.x GPU Tile Metadata
// ============================================================================

struct Q4xTileMetadataGPU {
    Q4xTileMetadata* d_tiles;
    int num_tiles;

    void allocate(int n) {
        num_tiles = n;
        cudaMalloc(&d_tiles, n * sizeof(Q4xTileMetadata));
    }

    void upload(const std::vector<Q4xTileMetadata>& host_data) {
        cudaMemcpy(d_tiles, host_data.data(),
                   host_data.size() * sizeof(Q4xTileMetadata),
                   cudaMemcpyHostToDevice);
    }

    void free() {
        cudaFree(d_tiles);
        d_tiles = nullptr;
        num_tiles = 0;
    }
};

// ============================================================================
// Device-side Utility Functions
// ============================================================================

// Mini-vector constants for interleaved format
constexpr int MV_SIZE = 256;      // Values per mini-vector
constexpr int MV_VALUES_PER_LANE = 8;  // Values per lane in mini-vector

/**
 * @brief Compute bit offset for value at local_idx within partition (INTERLEAVED format)
 *
 * For interleaved format:
 *   mv_idx = local_idx / 256
 *   local_in_mv = local_idx % 256
 *   lane = local_in_mv % 32
 *   val_idx = local_in_mv / 32
 *   bit_offset = partition_bit_base + mv_idx * 256 * delta_bits + lane * 8 * delta_bits + val_idx * delta_bits
 */
__device__ __forceinline__
int64_t computeInterleavedBitOffset(int64_t partition_bit_base, int local_idx, int delta_bits) {
    int mv_idx = local_idx / MV_SIZE;
    int local_in_mv = local_idx % MV_SIZE;
    int lane = local_in_mv % WARP_SIZE;
    int val_idx = local_in_mv / WARP_SIZE;

    return partition_bit_base +
           static_cast<int64_t>(mv_idx) * MV_SIZE * delta_bits +
           static_cast<int64_t>(lane) * MV_VALUES_PER_LANE * delta_bits +
           static_cast<int64_t>(val_idx) * delta_bits;
}

/**
 * @brief Mask for extracting delta_bits from a 64-bit value
 */
__device__ __forceinline__
uint64_t mask64(int bits) {
    return (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1);
}

/**
 * @brief Sign extend a value from delta_bits to 64 bits
 */
__device__ __forceinline__
int64_t signExtend64(uint64_t val, int bits) {
    if (bits == 0 || bits >= 64) return val;
    uint64_t sign_bit = 1ULL << (bits - 1);
    return (val ^ sign_bit) - sign_bit;
}

/**
 * @brief Branchless 64-bit extraction from delta array
 *
 * Loads two 64-bit words and extracts bits crossing word boundaries.
 */
__device__ __forceinline__
uint64_t extractBits64(const uint32_t* __restrict__ delta_array,
                       int64_t start_bit, int bits)
{
    if (bits == 0) return 0;

    int64_t word64_idx = start_bit >> 6;
    int bit_offset = start_bit & 63;

    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(delta_array);
    uint64_t lo = __ldg(&p64[word64_idx]);
    uint64_t hi = __ldg(&p64[word64_idx + 1]);

    uint64_t shifted_lo = lo >> bit_offset;
    uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));

    return (shifted_lo | shifted_hi) & mask64(bits);
}

/**
 * @brief Compute polynomial prediction
 */
__device__ __forceinline__
int64_t computePoly(int model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double pred;

    switch (model_type) {
        case MODEL_CONSTANT:
            pred = params[0];
            break;
        case MODEL_LINEAR:
            pred = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:
            pred = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:
            pred = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        case MODEL_FOR_BITPACK:
        case MODEL_DIRECT_COPY:
        default:
            // FOR: params[0] is base value stored as double-encoded int64
            return __double_as_longlong(params[0]);
    }

    return __double2ll_rn(pred);
}

} // namespace l3_fused
