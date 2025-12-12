/**
 * @file fls_block_load.cuh
 * @brief Vertical-style Block Load with L3 Metadata Integration
 *
 * This file bridges L3's partition-based metadata with Vertical-style
 * compile-time specialized unpack functions.
 *
 * Key Features:
 * - Crystal-opt style BlockLoad interface
 * - Integration with L3 partition metadata (start/end indices, bit offsets)
 * - FOR + BitPack decompression (no polynomial models)
 * - 32 threads x 32 items = 1024 elements per tile
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "fls_constants.cuh"
#include "fls_unpack.cuh"

namespace l3_fls {

// ============================================================================
// Column Metadata for FOR+BitPack Encoding
// ============================================================================

/**
 * @brief Metadata for a single column's compressed data
 *
 * This structure is designed to be loaded into shared memory once per block.
 * Each partition can have a different base value and bitwidth.
 */
struct FLSColumnMeta {
    int32_t base;           // FOR base value (min value in partition)
    uint8_t bitwidth;       // Bits per delta value
    int64_t bit_offset;     // Bit offset in delta array for this partition
    int32_t partition_size; // Number of elements in this partition
};

// ============================================================================
// Shared Memory Structure for Block-level Caching
// ============================================================================

#define FLS_MAX_COLUMNS 8

struct FLSSharedMem {
    int32_t partition_size;
    int32_t start_idx;
    FLSColumnMeta columns[FLS_MAX_COLUMNS];
    long long warp_sums[32];  // For warp reduction
};

// ============================================================================
// Metadata Loading Functions
// ============================================================================

/**
 * @brief Load column metadata from global memory to shared memory
 *
 * Should be called by thread 0 only, followed by __syncthreads()
 */
__device__ __forceinline__
void loadFLSColumnMeta(
    FLSColumnMeta& meta,
    int partition_id,
    const int32_t* __restrict__ base_values,      // FOR base values per partition
    const int32_t* __restrict__ delta_bits,       // Bitwidth per partition
    const int64_t* __restrict__ bit_offsets)      // Bit offset per partition
{
    meta.base = base_values[partition_id];
    meta.bitwidth = static_cast<uint8_t>(delta_bits[partition_id]);
    meta.bit_offset = bit_offsets[partition_id];
}

/**
 * @brief Load all column metadata for a partition
 *
 * Template parameter NUM_COLUMNS specifies how many columns to load.
 */
template<int NUM_COLUMNS>
__device__ __forceinline__ void loadAllFLSMeta(
    FLSSharedMem& smem,
    int partition_id,
    const int32_t* const* base_values,      // Array of pointers to base values
    const int32_t* const* delta_bits,       // Array of pointers to bitwidths
    const int64_t* const* bit_offsets,      // Array of pointers to bit offsets
    const int32_t* start_indices,
    const int32_t* end_indices)
{
    if (threadIdx.x == 0) {
        smem.start_idx = start_indices[partition_id];
        smem.partition_size = end_indices[partition_id] - smem.start_idx;

        #pragma unroll
        for (int c = 0; c < NUM_COLUMNS; c++) {
            loadFLSColumnMeta(smem.columns[c], partition_id,
                base_values[c], delta_bits[c], bit_offsets[c]);
        }
    }
    __syncthreads();
}

// ============================================================================
// Tile-based Data Access
// ============================================================================

/**
 * @brief Calculate tile pointer in compressed data
 *
 * Memory layout: tile N at offset N * bitwidth * 32 (in uint32_t words)
 *
 * @param delta_array Base pointer to compressed delta array
 * @param tile_idx Tile index within partition
 * @param bitwidth Bits per value
 * @return Pointer to start of tile data
 */
__device__ __forceinline__
const uint32_t* getTilePtr(
    const uint32_t* __restrict__ delta_array,
    int64_t bit_offset,
    int tile_idx,
    uint8_t bitwidth)
{
    // Calculate byte offset from bit offset
    int64_t byte_offset = bit_offset / 8;

    // Tile offset within partition
    int64_t tile_offset = static_cast<int64_t>(tile_idx) * bitwidth * 32;

    // Return pointer (assuming word-aligned)
    return delta_array + (byte_offset / 4) + tile_offset;
}

/**
 * @brief Alternative: Direct tile addressing for fixed-tile layouts
 *
 * When data is laid out with fixed 1024-element tiles, use this simpler method.
 */
__device__ __forceinline__
const uint32_t* getTilePtrFixed(
    const uint32_t* __restrict__ base_ptr,
    int global_tile_idx,
    uint8_t bitwidth)
{
    return base_ptr + static_cast<int64_t>(global_tile_idx) * bitwidth * 32;
}

// ============================================================================
// Block Load Functions (Crystal-style Interface)
// ============================================================================

/**
 * @brief Load and decompress a tile using Vertical unpack
 *
 * This function uses the runtime dispatch unpack function.
 * For maximum performance with known bitwidths, use the specialized versions.
 *
 * @param tile_data Pointer to compressed tile data
 * @param items Output array (32 items per thread, strided access)
 * @param base FOR base value
 * @param bw Bitwidth
 */
__device__ __forceinline__ void BlockLoadFLSRuntime(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base,
    uint8_t bw)
{
    // Each thread unpacks 32 values to registers
    int32_t local_items[32];
    unpack_for(tile_data, local_items, base, bw);

    // Store to output with strided pattern (same as Vertical)
    // Thread i stores at positions [i, i+32, i+64, ...]
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = local_items[j];
    }
}

/**
 * @brief Load tile with bounds checking
 *
 * For partial tiles at the end of partitions
 */
__device__ __forceinline__ void BlockLoadFLSRuntimeBounded(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base,
    uint8_t bw,
    int num_valid_items)
{
    int32_t local_items[32];
    unpack_for(tile_data, local_items, base, bw);

    int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int global_idx = j * 32 + tid;
        if (global_idx < num_valid_items) {
            items[j] = local_items[j];
        }
    }
}

// ============================================================================
// Compile-time Specialized Block Load (for known bitwidths)
// ============================================================================

template<int BW>
__device__ __forceinline__ void BlockLoadFLS(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base);

// Specialization for 0-bit (constant values)
template<>
__device__ __forceinline__ void BlockLoadFLS<0>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base;
    }
}

// Specialization for 1-bit
template<>
__device__ __forceinline__ void BlockLoadFLS<1>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_1bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// Specialization for 2-bit
template<>
__device__ __forceinline__ void BlockLoadFLS<2>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_2bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// Specialization for 4-bit
template<>
__device__ __forceinline__ void BlockLoadFLS<4>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_4bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// Specialization for 8-bit
template<>
__device__ __forceinline__ void BlockLoadFLS<8>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_8bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// Specialization for 16-bit
template<>
__device__ __forceinline__ void BlockLoadFLS<16>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_16bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// Specialization for 32-bit (no compression)
template<>
__device__ __forceinline__ void BlockLoadFLS<32>(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base)
{
    uint32_t deltas[32];
    unpack_32bw(tile_data, deltas);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// ============================================================================
// Predicate-Gated Block Load (Crystal-style)
// ============================================================================

/**
 * @brief Load tile data only for items where selection flag is set
 *
 * This enables predicate pushdown - only decompress values that passed
 * previous predicates.
 */
__device__ __forceinline__ void BlockPredLoadFLS(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base,
    uint8_t bw,
    const int* __restrict__ selection_flags)
{
    // Check if any thread in warp needs data
    unsigned int warp_mask = __ballot_sync(0xFFFFFFFF,
        selection_flags[0] | selection_flags[1] |
        selection_flags[2] | selection_flags[3]);

    if (warp_mask == 0) return;  // No thread needs this data

    // Decompress (all threads participate for memory coalescing)
    int32_t local_items[32];
    unpack_for(tile_data, local_items, base, bw);

    // Store only selected items
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        if (selection_flags[j]) {
            items[j] = local_items[j];
        }
    }
}

// ============================================================================
// Store Functions (Crystal-style Interface)
// ============================================================================

/**
 * @brief Store decompressed values to global memory with strided pattern
 *
 * Thread i stores at positions [base + i, base + i + 32, ...]
 */
template<int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreFLS(
    const int32_t* __restrict__ items,
    int32_t* __restrict__ output,
    int tile_offset)
{
    int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        output[tile_offset + j * 32 + tid] = items[j];
    }
}

/**
 * @brief Store with bounds checking
 */
template<int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockStoreFLSBounded(
    const int32_t* __restrict__ items,
    int32_t* __restrict__ output,
    int tile_offset,
    int num_valid_items)
{
    int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; j++) {
        int idx = j * 32 + tid;
        if (idx < num_valid_items) {
            output[tile_offset + idx] = items[j];
        }
    }
}

// ============================================================================
// Warp Reduction for Aggregation
// ============================================================================

/**
 * @brief Warp-level sum reduction using shuffle
 */
__device__ __forceinline__ long long warpReduceSum(long long val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Block-level sum reduction
 *
 * @param val Thread's local value
 * @param warp_sums Shared memory for warp partial sums (needs 32 entries)
 * @return Final sum (valid only in thread 0)
 */
__device__ __forceinline__ long long blockReduceSumFLS(
    long long val,
    long long* __restrict__ warp_sums)
{
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp-level reduction
    val = warpReduceSum(val);

    // Write warp sum to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();

    // Only warp 0 does final reduction (for 32-thread blocks, this is just thread 0)
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? warp_sums[threadIdx.x] : 0;
        val = warpReduceSum(val);
    }

    return val;
}

// ============================================================================
// Selection Flags Initialization
// ============================================================================

/**
 * @brief Initialize selection flags for tile processing
 */
template<int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlagsFLS(
    int (&flags)[ITEMS_PER_THREAD],
    int tile_size)
{
    int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = i * 32 + tid;
        flags[i] = (idx < tile_size) ? 1 : 0;
    }
}

// ============================================================================
// Early Termination Check
// ============================================================================

/**
 * @brief Check if all items in block have been filtered out
 *
 * Use this for early termination when all predicates have failed.
 */
template<int ITEMS_PER_THREAD>
__device__ __forceinline__ bool IsTermFLS(
    const int (&flags)[ITEMS_PER_THREAD])
{
    int has_active = 0;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        has_active |= flags[i];
    }

    // Use ballot to check across all threads
    unsigned int warp_active = __ballot_sync(0xFFFFFFFF, has_active);
    return (warp_active == 0);
}

}  // namespace l3_fls
