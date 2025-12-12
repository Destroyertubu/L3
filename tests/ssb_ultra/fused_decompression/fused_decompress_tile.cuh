/**
 * @file fused_decompress_tile.cuh
 * @brief Tile-Based Coarsened Decompression for Fused Kernels
 *
 * Provides optimized decompression with:
 * - Thread coarsening (4 items per thread)
 * - Register buffering for delta words
 * - Strided access for coalescing
 * - Predicate-conditional decompression
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "fused_kernel_common.cuh"
#include "l3_decompress_device.cuh"

namespace l3_fused {

// ============================================================================
// Register Buffering Constants
// ============================================================================

// Maximum words to prefetch per item (supports up to 64-bit deltas with misalignment)
constexpr int WORDS_PER_ITEM = 3;

// ============================================================================
// Register Buffer Extraction
// ============================================================================

/**
 * @brief Extract delta from prefetched register buffer
 *
 * The register buffer contains 3 consecutive 32-bit words loaded around
 * the bit offset. This function extracts the delta value without additional
 * global memory accesses.
 *
 * @param reg_buffer  3 consecutive words prefetched around the bit offset
 * @param bit_in_first_word  Bit offset within the first word (0-31)
 * @param delta_bits  Number of bits to extract
 * @return Extracted unsigned delta value
 */
__device__ __forceinline__
uint64_t extractFromRegisterBuffer(
    const uint32_t* reg_buffer,
    int bit_in_first_word,
    int delta_bits)
{
    if (delta_bits <= 0) return 0ULL;

    // Combine first two words into 64-bit value
    uint64_t combined = ((uint64_t)reg_buffer[1] << 32) | reg_buffer[0];
    uint64_t extracted = combined >> bit_in_first_word;

    // For wide deltas, may need third word
    if (delta_bits > 32 && bit_in_first_word > 0) {
        int bits_from_first_two = 64 - bit_in_first_word;
        if (bits_from_first_two < delta_bits) {
            extracted |= ((uint64_t)reg_buffer[2] << bits_from_first_two);
        }
    }

    return extracted & mask64_rt(delta_bits);
}

// ============================================================================
// Tile-Based Decompression with Register Buffering
// ============================================================================

/**
 * @brief Decompress a tile of values with thread coarsening and register buffering
 *
 * Each thread decompresses ITEMS_PER_THREAD values using:
 * 1. Prefetched delta words in registers (reduces global memory accesses)
 * 2. Strided access pattern for coalescing
 * 3. Selection flags for conditional decompression
 *
 * @tparam T              Output value type (uint32_t or uint64_t)
 * @tparam IPT            Items per thread (default 4)
 * @param delta_array     Bit-packed delta array
 * @param meta            Cached partition metadata
 * @param tile_start      Starting index within partition for this tile
 * @param values          Output: decompressed values
 * @param selection_flags Input/Output: only decompress where flag=1
 * @param partition_size  Total size of partition
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void decompressTileCoarsened(
    const uint32_t* __restrict__ delta_array,
    const PartitionMetadataCache& meta,
    int tile_start,
    T (&values)[IPT],
    int (&selection_flags)[IPT],
    int partition_size)
{
    int tid = threadIdx.x;
    int delta_bits = meta.delta_bits;

    // Prefetch delta words for all items into registers
    uint32_t reg_buffers[IPT][WORDS_PER_ITEM];

    if (delta_bits > 0) {
        #pragma unroll
        for (int i = 0; i < IPT; ++i) {
            int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

            if (local_idx < partition_size && selection_flags[i]) {
                // Calculate bit offset and word index
                int64_t bit_offset = meta.bit_offset_base +
                                     static_cast<int64_t>(local_idx) * delta_bits;
                int64_t word_idx = bit_offset >> 5;

                // Prefetch 3 consecutive words using cached loads
                reg_buffers[i][0] = __ldg(&delta_array[word_idx]);
                reg_buffers[i][1] = __ldg(&delta_array[word_idx + 1]);
                reg_buffers[i][2] = __ldg(&delta_array[word_idx + 2]);
            }
        }
    }

    // Decompress each item
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

        // Skip if out of bounds or already filtered
        if (local_idx >= partition_size) {
            selection_flags[i] = 0;
            continue;
        }

        if (!selection_flags[i]) continue;

        // Compute polynomial prediction
        double x = static_cast<double>(local_idx);
        double predicted;

        switch (meta.model_type) {
            case MODEL_CONSTANT:
                predicted = meta.model_params[0];
                break;
            case MODEL_LINEAR:
                predicted = meta.model_params[0] + meta.model_params[1] * x;
                break;
            case MODEL_POLYNOMIAL2:
                predicted = meta.model_params[0] + x * (meta.model_params[1] + x * meta.model_params[2]);
                break;
            case MODEL_POLYNOMIAL3:
                predicted = meta.model_params[0] + x * (meta.model_params[1] + x * (meta.model_params[2] + x * meta.model_params[3]));
                break;
            case MODEL_FOR_BITPACK:
            case MODEL_DIRECT_COPY:
                // FOR/Direct: base is model_params[0]
                if (delta_bits == 0) {
                    values[i] = static_cast<T>(__double2ll_rn(meta.model_params[0]));
                    continue;
                }
                // Fall through to extract delta
                predicted = meta.model_params[0];
                break;
            default:
                predicted = meta.model_params[0] + meta.model_params[1] * x;
                break;
        }

        // Zero-bit case (perfect prediction)
        if (delta_bits == 0) {
            values[i] = static_cast<T>(__double2ll_rn(predicted));
            continue;
        }

        // Extract delta from prefetched buffer
        int64_t bit_offset = meta.bit_offset_base +
                             static_cast<int64_t>(local_idx) * delta_bits;
        int bit_in_word = bit_offset & 31;

        uint64_t delta_unsigned = extractFromRegisterBuffer(
            reg_buffers[i], bit_in_word, delta_bits);
        int64_t delta = sign_extend_64(delta_unsigned, delta_bits);

        // Combine prediction + delta
        if (meta.model_type == MODEL_FOR_BITPACK || meta.model_type == MODEL_DIRECT_COPY) {
            values[i] = static_cast<T>(static_cast<int64_t>(__double2ll_rn(predicted)) + delta);
        } else {
            values[i] = static_cast<T>(__double2ll_rn(predicted) + delta);
        }
    }
}

/**
 * @brief Conditional decompression - only decompress if selection_flags[i] == 1
 *
 * Optimized for subsequent columns where many rows may have been filtered.
 * Saves decompression work for filtered rows.
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void decompressTileConditional(
    const uint32_t* __restrict__ delta_array,
    const PartitionMetadataCache& meta,
    int tile_start,
    T (&values)[IPT],
    const int (&selection_flags)[IPT],
    int partition_size)
{
    int tid = threadIdx.x;
    int delta_bits = meta.delta_bits;

    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

        // Skip if out of bounds or filtered out
        if (local_idx >= partition_size || !selection_flags[i]) {
            continue;
        }

        // Compute polynomial prediction
        double x = static_cast<double>(local_idx);
        double predicted;

        switch (meta.model_type) {
            case MODEL_CONSTANT:
                predicted = meta.model_params[0];
                break;
            case MODEL_LINEAR:
                predicted = meta.model_params[0] + meta.model_params[1] * x;
                break;
            case MODEL_POLYNOMIAL2:
                predicted = meta.model_params[0] + x * (meta.model_params[1] + x * meta.model_params[2]);
                break;
            case MODEL_POLYNOMIAL3:
                predicted = meta.model_params[0] + x * (meta.model_params[1] + x * (meta.model_params[2] + x * meta.model_params[3]));
                break;
            case MODEL_FOR_BITPACK:
            case MODEL_DIRECT_COPY:
                if (delta_bits == 0) {
                    values[i] = static_cast<T>(__double2ll_rn(meta.model_params[0]));
                    continue;
                }
                predicted = meta.model_params[0];
                break;
            default:
                predicted = meta.model_params[0] + meta.model_params[1] * x;
                break;
        }

        if (delta_bits == 0) {
            values[i] = static_cast<T>(__double2ll_rn(predicted));
            continue;
        }

        // Direct extraction (no prefetch for conditional path - fewer items)
        int64_t bit_offset = meta.bit_offset_base +
                             static_cast<int64_t>(local_idx) * delta_bits;
        uint64_t delta_unsigned = extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
        int64_t delta = sign_extend_64(delta_unsigned, delta_bits);

        if (meta.model_type == MODEL_FOR_BITPACK || meta.model_type == MODEL_DIRECT_COPY) {
            values[i] = static_cast<T>(static_cast<int64_t>(__double2ll_rn(predicted)) + delta);
        } else {
            values[i] = static_cast<T>(__double2ll_rn(predicted) + delta);
        }
    }
}

// ============================================================================
// Block-Level Data Loading (for uncompressed dimension tables)
// ============================================================================

/**
 * @brief Load a tile from uncompressed data with coalesced access
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockLoadTile(
    const T* __restrict__ data,
    int tile_start,
    T (&values)[IPT],
    int num_items)
{
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int idx = tile_start + tid + i * FUSED_BLOCK_SIZE;
        if (idx < num_items) {
            values[i] = data[idx];
        }
    }
}

/**
 * @brief Conditional load - only load if selection_flags[i] == 1
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void BlockLoadTileConditional(
    const T* __restrict__ data,
    int tile_start,
    T (&values)[IPT],
    const int (&selection_flags)[IPT],
    int num_items)
{
    int tid = threadIdx.x;
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        if (selection_flags[i]) {
            int idx = tile_start + tid + i * FUSED_BLOCK_SIZE;
            if (idx < num_items) {
                values[i] = data[idx];
            }
        }
    }
}

} // namespace l3_fused
