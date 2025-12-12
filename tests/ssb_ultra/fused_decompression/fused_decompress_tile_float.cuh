/**
 * @file fused_decompress_tile_float.cuh
 * @brief Float-Optimized Tile-Based Decompression for Fused Kernels
 *
 * OPTIMIZATION: Uses float instead of double for polynomial SLOPE computation.
 * The base value (theta0) is kept as int64 to preserve precision for large values.
 *
 * Key insight: For values like 19920101, float loses precision. But the SLOPE
 * computation (theta1*x, theta2*x^2, etc.) involves much smaller values and
 * can safely use float.
 *
 * Formula: value = (int64_t)theta0 + (int32_t)(float_slope_result) + delta
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "fused_kernel_common.cuh"  // For FUSED_BLOCK_SIZE, FUSED_ITEMS_PER_THREAD, PartitionMetadataCache
#include "l3_decompress_device.cuh"

namespace l3_fused_float {

// Use constants from l3_fused namespace
using l3_fused::MODEL_CONSTANT;
using l3_fused::MODEL_LINEAR;
using l3_fused::MODEL_POLYNOMIAL2;
using l3_fused::MODEL_POLYNOMIAL3;
using l3_fused::MODEL_FOR_BITPACK;
using l3_fused::MODEL_DIRECT_COPY;
using l3_fused::mask64_rt;
using l3_fused::sign_extend_64;
using l3_fused::extract_branchless_64_rt;
using l3_fused::PartitionMetadataCache;

// Use fused kernel constants
using l3_fused::FUSED_BLOCK_SIZE;
using l3_fused::FUSED_ITEMS_PER_THREAD;

// ============================================================================
// Register Buffering Constants
// ============================================================================

constexpr int WORDS_PER_ITEM = 3;

// ============================================================================
// Register Buffer Extraction (same as original)
// ============================================================================

__device__ __forceinline__
uint64_t extractFromRegisterBuffer(
    const uint32_t* reg_buffer,
    int bit_in_first_word,
    int delta_bits)
{
    if (delta_bits <= 0) return 0ULL;

    uint64_t combined = ((uint64_t)reg_buffer[1] << 32) | reg_buffer[0];
    uint64_t extracted = combined >> bit_in_first_word;

    if (delta_bits > 32 && bit_in_first_word > 0) {
        int bits_from_first_two = 64 - bit_in_first_word;
        if (bits_from_first_two < delta_bits) {
            extracted |= ((uint64_t)reg_buffer[2] << bits_from_first_two);
        }
    }

    return extracted & mask64_rt(delta_bits);
}

// ============================================================================
// Hybrid Float-Int Tile Decompression
// ============================================================================

/**
 * @brief Hybrid float-int tile decompression
 *
 * Uses int64 for base (theta0) to preserve precision for large values,
 * and float for slope computation (theta1*x + theta2*x^2 + ...).
 *
 * @tparam T              Output value type (uint32_t or uint64_t)
 * @tparam IPT            Items per thread (default 4)
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void decompressTileCoarsenedFloat(
    const uint32_t* __restrict__ delta_array,
    const PartitionMetadataCache& meta,
    int tile_start,
    T (&values)[IPT],
    int (&selection_flags)[IPT],
    int partition_size)
{
    int tid = threadIdx.x;
    int delta_bits = meta.delta_bits;

    // Keep base as int64 for precision, convert slopes to float
    int64_t base_int = __double2ll_rn(meta.model_params[0]);
    float p1 = static_cast<float>(meta.model_params[1]);
    float p2 = static_cast<float>(meta.model_params[2]);
    float p3 = static_cast<float>(meta.model_params[3]);
    int model_type = meta.model_type;
    int64_t bit_offset_base = meta.bit_offset_base;

    // Prefetch delta words for all items into registers
    uint32_t reg_buffers[IPT][WORDS_PER_ITEM];

    if (delta_bits > 0) {
        #pragma unroll
        for (int i = 0; i < IPT; ++i) {
            int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

            if (local_idx < partition_size && selection_flags[i]) {
                int64_t bit_offset = bit_offset_base +
                                     static_cast<int64_t>(local_idx) * delta_bits;
                int64_t word_idx = bit_offset >> 5;

                reg_buffers[i][0] = __ldg(&delta_array[word_idx]);
                reg_buffers[i][1] = __ldg(&delta_array[word_idx + 1]);
                reg_buffers[i][2] = __ldg(&delta_array[word_idx + 2]);
            }
        }
    }

    // Decompress each item using hybrid int64+float arithmetic
    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

        if (local_idx >= partition_size) {
            selection_flags[i] = 0;
            continue;
        }

        if (!selection_flags[i]) continue;

        // Compute prediction: base_int + float_slope_correction
        float x = static_cast<float>(local_idx);
        int32_t slope_correction = 0;

        switch (model_type) {
            case MODEL_CONSTANT:
                // No slope correction
                break;
            case MODEL_LINEAR:
                slope_correction = __float2int_rn(p1 * x);
                break;
            case MODEL_POLYNOMIAL2:
                slope_correction = __float2int_rn(x * (p1 + x * p2));
                break;
            case MODEL_POLYNOMIAL3:
                slope_correction = __float2int_rn(x * (p1 + x * (p2 + x * p3)));
                break;
            case MODEL_FOR_BITPACK:
            case MODEL_DIRECT_COPY:
                if (delta_bits == 0) {
                    values[i] = static_cast<T>(base_int);
                    continue;
                }
                break;
            default:
                slope_correction = __float2int_rn(p1 * x);
                break;
        }

        int64_t predicted = base_int + slope_correction;

        if (delta_bits == 0) {
            values[i] = static_cast<T>(predicted);
            continue;
        }

        // Extract delta from prefetched buffer
        int64_t bit_offset = bit_offset_base +
                             static_cast<int64_t>(local_idx) * delta_bits;
        int bit_in_word = bit_offset & 31;

        uint64_t delta_unsigned = extractFromRegisterBuffer(
            reg_buffers[i], bit_in_word, delta_bits);
        int64_t delta = sign_extend_64(delta_unsigned, delta_bits);

        // Combine: base + slope_correction + delta
        values[i] = static_cast<T>(predicted + delta);
    }
}

/**
 * @brief Conditional hybrid float-int decompression
 */
template<typename T, int IPT = FUSED_ITEMS_PER_THREAD>
__device__ __forceinline__
void decompressTileConditionalFloat(
    const uint32_t* __restrict__ delta_array,
    const PartitionMetadataCache& meta,
    int tile_start,
    T (&values)[IPT],
    const int (&selection_flags)[IPT],
    int partition_size)
{
    int tid = threadIdx.x;
    int delta_bits = meta.delta_bits;

    // Keep base as int64 for precision, convert slopes to float
    int64_t base_int = __double2ll_rn(meta.model_params[0]);
    float p1 = static_cast<float>(meta.model_params[1]);
    float p2 = static_cast<float>(meta.model_params[2]);
    float p3 = static_cast<float>(meta.model_params[3]);
    int model_type = meta.model_type;
    int64_t bit_offset_base = meta.bit_offset_base;

    #pragma unroll
    for (int i = 0; i < IPT; ++i) {
        int local_idx = tile_start + tid + i * FUSED_BLOCK_SIZE;

        if (local_idx >= partition_size || !selection_flags[i]) {
            continue;
        }

        // Compute prediction: base_int + float_slope_correction
        float x = static_cast<float>(local_idx);
        int32_t slope_correction = 0;

        switch (model_type) {
            case MODEL_CONSTANT:
                break;
            case MODEL_LINEAR:
                slope_correction = __float2int_rn(p1 * x);
                break;
            case MODEL_POLYNOMIAL2:
                slope_correction = __float2int_rn(x * (p1 + x * p2));
                break;
            case MODEL_POLYNOMIAL3:
                slope_correction = __float2int_rn(x * (p1 + x * (p2 + x * p3)));
                break;
            case MODEL_FOR_BITPACK:
            case MODEL_DIRECT_COPY:
                if (delta_bits == 0) {
                    values[i] = static_cast<T>(base_int);
                    continue;
                }
                break;
            default:
                slope_correction = __float2int_rn(p1 * x);
                break;
        }

        int64_t predicted = base_int + slope_correction;

        if (delta_bits == 0) {
            values[i] = static_cast<T>(predicted);
            continue;
        }

        // Direct extraction
        int64_t bit_offset = bit_offset_base +
                             static_cast<int64_t>(local_idx) * delta_bits;
        uint64_t delta_unsigned = extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
        int64_t delta = sign_extend_64(delta_unsigned, delta_bits);

        values[i] = static_cast<T>(predicted + delta);
    }
}

} // namespace l3_fused_float
