/**
 * @file l3_block_load.cuh
 * @brief L3 Decompression as Block Load (Crystal-opt style)
 *
 * Replaces FSL-GPU/Crystal-opt's BlockLoad with L3 decompression.
 * Uses same strided access pattern: thread i loads items at i, i+128, i+256, i+384
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// ============================================================================
// L3 Model Types
// ============================================================================

enum L3ModelType : int32_t {
    L3_MODEL_CONSTANT = 0,      // f(x) = θ₀
    L3_MODEL_LINEAR = 1,        // f(x) = θ₀ + θ₁·x
    L3_MODEL_POLYNOMIAL2 = 2,   // f(x) = θ₀ + θ₁·x + θ₂·x²
    L3_MODEL_POLYNOMIAL3 = 3,   // f(x) = θ₀ + θ₁·x + θ₂·x² + θ₃·x³
    L3_MODEL_FOR_BITPACK = 4,   // FOR + BitPacking
    L3_MODEL_DIRECT_COPY = 5    // Direct copy
};

// ============================================================================
// L3 Partition Metadata (Shared Memory Cache)
// ============================================================================

struct L3PartitionMeta {
    int32_t model_type;
    double params[4];    // theta0, theta1, theta2, theta3
    int32_t delta_bits;
    int64_t bit_offset;
};

// ============================================================================
// Helper Functions
// ============================================================================

__device__ __forceinline__
uint64_t l3_mask64(int k) {
    if (k <= 0) return 0ULL;
    if (k >= 64) return ~0ULL;
    return (1ULL << k) - 1ULL;
}

__device__ __forceinline__
int64_t l3_sign_extend(uint64_t value, int bit_width) {
    if (bit_width >= 64) return static_cast<int64_t>(value);
    if (bit_width <= 0) return 0;

    const uint64_t sign_bit = value >> (bit_width - 1);
    const uint64_t sign_mask = -sign_bit;
    const uint64_t extend_mask = ~l3_mask64(bit_width);

    return static_cast<int64_t>(value | (sign_mask & extend_mask));
}

// Branchless bit extraction (eliminates warp divergence)
__device__ __forceinline__
uint64_t l3_extract_bits(const uint32_t* __restrict__ words, int64_t start_bit, int bits) {
    if (bits <= 0) return 0ULL;
    if (bits > 64) bits = 64;

    const uint64_t word64_idx = start_bit >> 6;
    const int bit_offset = start_bit & 63;

    const uint64_t* __restrict__ p64 = reinterpret_cast<const uint64_t*>(words);
    const uint64_t lo = __ldg(&p64[word64_idx]);
    const uint64_t hi = __ldg(&p64[word64_idx + 1]);

    const uint64_t shifted_lo = lo >> bit_offset;
    const uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));

    return (shifted_lo | shifted_hi) & l3_mask64(bits);
}

// ============================================================================
// Polynomial Prediction (Horner's Method)
// ============================================================================

__device__ __forceinline__
int l3_predict(int32_t model_type, const double* params, int idx) {
    double x = static_cast<double>(idx);
    double predicted;

    switch (model_type) {
        case L3_MODEL_CONSTANT:
            predicted = params[0];
            break;
        case L3_MODEL_LINEAR:
            predicted = params[0] + params[1] * x;
            break;
        case L3_MODEL_POLYNOMIAL2:
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case L3_MODEL_POLYNOMIAL3:
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        case L3_MODEL_FOR_BITPACK:
        case L3_MODEL_DIRECT_COPY:
            return static_cast<int>(__double2ll_rn(params[0]));
        default:
            predicted = params[0] + params[1] * x;
            break;
    }

    return static_cast<int>(__double2ll_rn(predicted));
}

// ============================================================================
// Core L3 Decompression
// ============================================================================

__device__ __forceinline__
int l3_decompress_value(
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int local_idx)
{
    // FOR/Direct copy special handling
    if (meta.model_type == L3_MODEL_FOR_BITPACK || meta.model_type == L3_MODEL_DIRECT_COPY) {
        int base = static_cast<int>(__double2ll_rn(meta.params[0]));
        bool is_direct = (meta.params[0] == 0.0 && meta.params[1] == 0.0);

        if (meta.delta_bits == 0) {
            return base;
        }

        int64_t bit_offset = meta.bit_offset + static_cast<int64_t>(local_idx) * meta.delta_bits;
        uint64_t extracted = l3_extract_bits(delta_array, bit_offset, meta.delta_bits);

        if (is_direct) {
            return static_cast<int>(l3_sign_extend(extracted, meta.delta_bits));
        } else {
            return base + static_cast<int>(extracted);
        }
    }

    // Polynomial model
    int predicted = l3_predict(meta.model_type, meta.params, local_idx);

    if (meta.delta_bits == 0) {
        return predicted;
    }

    int64_t bit_offset = meta.bit_offset + static_cast<int64_t>(local_idx) * meta.delta_bits;
    uint64_t extracted = l3_extract_bits(delta_array, bit_offset, meta.delta_bits);
    int64_t delta = l3_sign_extend(extracted, meta.delta_bits);

    return predicted + static_cast<int>(delta);
}

// ============================================================================
// Load L3 Partition Metadata
// ============================================================================

__device__ __forceinline__
void loadL3Meta(
    L3PartitionMeta& meta,
    int partition_id,
    const int32_t* __restrict__ model_types,
    const double* __restrict__ model_params,
    const int32_t* __restrict__ delta_bits,
    const int64_t* __restrict__ bit_offsets)
{
    meta.model_type = model_types[partition_id];
    meta.params[0] = model_params[partition_id * 4];
    meta.params[1] = model_params[partition_id * 4 + 1];
    meta.params[2] = model_params[partition_id * 4 + 2];
    meta.params[3] = model_params[partition_id * 4 + 3];
    meta.delta_bits = delta_bits[partition_id];
    meta.bit_offset = bit_offsets[partition_id];
}

// ============================================================================
// Block Load L3 (Direct - Crystal-opt style strided access)
// ============================================================================

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadL3Direct(
    const unsigned int tid,
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD])
{
    // Crystal-opt strided access: thread i loads [i, i+128, i+256, i+384]
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int local_idx = tile_offset + tid + ITEM * BLOCK_THREADS;
        items[ITEM] = l3_decompress_value(delta_array, meta, local_idx);
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadL3Direct(
    const unsigned int tid,
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD],
    int num_tile_items)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx_in_tile = tid + ITEM * BLOCK_THREADS;
        if (idx_in_tile < num_tile_items) {
            int local_idx = tile_offset + idx_in_tile;
            items[ITEM] = l3_decompress_value(delta_array, meta, local_idx);
        }
    }
}

// Main BlockLoadL3 function (matches crystal-opt BlockLoad interface)
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadL3(
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD],
    int num_tile_items)
{
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_tile_items) {
        BlockLoadL3Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, delta_array, meta, tile_offset, items);
    } else {
        BlockLoadL3Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, delta_array, meta, tile_offset, items, num_tile_items);
    }
}

// ============================================================================
// Block Predicated Load L3 (only decompress selected elements)
// ============================================================================

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadL3Direct(
    const unsigned int tid,
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD])
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int local_idx = tile_offset + tid + ITEM * BLOCK_THREADS;
            items[ITEM] = l3_decompress_value(delta_array, meta, local_idx);
        }
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadL3Direct(
    const unsigned int tid,
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD],
    int num_tile_items,
    int (&selection_flags)[ITEMS_PER_THREAD])
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int idx_in_tile = tid + ITEM * BLOCK_THREADS;
            if (idx_in_tile < num_tile_items) {
                int local_idx = tile_offset + idx_in_tile;
                items[ITEM] = l3_decompress_value(delta_array, meta, local_idx);
            }
        }
    }
}

// Main BlockPredLoadL3 function (matches crystal-opt BlockPredLoad interface)
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadL3(
    const uint32_t* __restrict__ delta_array,
    const L3PartitionMeta& meta,
    int tile_offset,
    int (&items)[ITEMS_PER_THREAD],
    int num_tile_items,
    int (&selection_flags)[ITEMS_PER_THREAD])
{
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_tile_items) {
        BlockPredLoadL3Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, delta_array, meta, tile_offset, items, selection_flags);
    } else {
        BlockPredLoadL3Direct<BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, delta_array, meta, tile_offset, items, num_tile_items, selection_flags);
    }
}

// ============================================================================
// Shared Memory Structure for Multiple Columns
// ============================================================================

#define L3_MAX_COLUMNS 8

struct L3SharedMem {
    int partition_size;
    int start_idx;
    L3PartitionMeta columns[L3_MAX_COLUMNS];
    long long buffer[32];  // For warp reduction
};

// Load all column metadata for a partition
template<int NUM_COLUMNS>
__device__ __forceinline__ void loadAllL3Meta(
    L3SharedMem& smem,
    int partition_id,
    const int32_t* const* model_types,     // Array of pointers
    const double* const* model_params,
    const int32_t* const* delta_bits,
    const int64_t* const* bit_offsets,
    const int32_t* start_indices,
    const int32_t* end_indices)
{
    if (threadIdx.x == 0) {
        smem.start_idx = start_indices[partition_id];
        smem.partition_size = end_indices[partition_id] - smem.start_idx;

        #pragma unroll
        for (int c = 0; c < NUM_COLUMNS; c++) {
            loadL3Meta(smem.columns[c], partition_id,
                model_types[c], model_params[c], delta_bits[c], bit_offsets[c]);
        }
    }
    __syncthreads();
}
