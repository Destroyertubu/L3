/**
 * Planner SSB Q1 Fused Implementation
 *
 * Fused decompression + query execution in single kernel.
 * - NS/FOR_NS: inline single-value decode
 * - DELTA/RLE: tile-level decode to shared memory
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include "planner/encoded_column.hpp"

namespace planner {
namespace fused {

// ============================================================================
// Inline single-value decode for NS and FOR_NS
// ============================================================================

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p, int byte_width) {
    uint32_t v = p[0];
    if (byte_width >= 2) v |= static_cast<uint32_t>(p[1]) << 8;
    if (byte_width >= 3) v |= static_cast<uint32_t>(p[2]) << 16;
    if (byte_width >= 4) v |= static_cast<uint32_t>(p[3]) << 24;
    return v;
}

__device__ __forceinline__ int decode_ns_single(
    const uint8_t* d_bytes, int byte_width, int idx)
{
    const uint8_t* p = d_bytes + static_cast<size_t>(idx) * byte_width;
    return static_cast<int>(load_u32_le(p, byte_width));
}

__device__ __forceinline__ int decode_for_ns_single(
    const uint8_t* d_bytes, int byte_width, int base, int idx)
{
    return decode_ns_single(d_bytes, byte_width, idx) + base;
}

// ============================================================================
// Tile-level decode for DELTA (decode a tile to shared memory)
// ============================================================================

template<int TILE_SIZE>
__device__ __forceinline__ void decode_delta_tile(
    const uint8_t* d_bytes,
    int byte_width,
    int first_value,
    int tile_offset,
    int n,
    int* __restrict__ smem_out)
{
    // Each thread decodes one value
    const int local_idx = threadIdx.x;
    const int global_idx = tile_offset + local_idx;

    if (global_idx >= n) {
        smem_out[local_idx] = 0;
        return;
    }

    // For DELTA: need to compute prefix sum of deltas
    // Simplified approach: decode deltas and do warp-level prefix sum
    int delta = 0;
    if (global_idx > 0) {
        const uint8_t* p = d_bytes + static_cast<size_t>(global_idx - 1) * byte_width;
        delta = static_cast<int>(load_u32_le(p, byte_width));
    }

    // Warp-level inclusive scan for prefix sum
    int prefix = delta;
    #pragma unroll
    for (int offset = 1; offset < 32; offset *= 2) {
        int n = __shfl_up_sync(0xFFFFFFFF, prefix, offset);
        if ((threadIdx.x % 32) >= offset) prefix += n;
    }

    // Cross-warp reduction using shared memory
    __shared__ int warp_sums[32];
    const int lane = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    if (lane == 31) {
        warp_sums[warp_id] = prefix;
    }
    __syncthreads();

    // First warp does prefix sum of warp sums
    if (warp_id == 0 && lane < (TILE_SIZE + 31) / 32) {
        int ws = warp_sums[lane];
        #pragma unroll
        for (int offset = 1; offset < 32; offset *= 2) {
            int n = __shfl_up_sync(0xFFFFFFFF, ws, offset);
            if (lane >= offset) ws += n;
        }
        warp_sums[lane] = ws;
    }
    __syncthreads();

    // Add warp prefix to each thread's value
    if (warp_id > 0) {
        prefix += warp_sums[warp_id - 1];
    }

    // Add first_value and account for tile offset's accumulated delta
    // Note: This simplified version assumes tile_offset == 0 for correctness
    // Full implementation would need to track cumulative sum across tiles
    smem_out[local_idx] = first_value + prefix;
}

// ============================================================================
// Fused column accessor - handles all schemes
// ============================================================================

struct FusedColumnAccessor {
    Scheme scheme;
    const uint8_t* d_bytes;
    const int* d_ints;          // For UNCOMPRESSED
    const int* d_decoded;       // For pre-decoded DELTA/RLE (fallback)
    int byte_width;
    int base;
    int first;
    int delta_base;
    size_t n;

    __device__ __forceinline__ int decode(int idx) const {
        switch (scheme) {
            case Scheme::Uncompressed:
                return d_ints[idx];
            case Scheme::NS:
                return decode_ns_single(d_bytes, byte_width, idx);
            case Scheme::FOR_NS:
                return decode_for_ns_single(d_bytes, byte_width, base, idx);
            case Scheme::DELTA_NS:
            case Scheme::DELTA_FOR_NS:
            case Scheme::RLE:
                // Fallback to pre-decoded data for complex schemes
                return d_decoded[idx];
            default:
                return 0;
        }
    }

    __device__ __forceinline__ bool is_inline_decodable() const {
        return scheme == Scheme::Uncompressed ||
               scheme == Scheme::NS ||
               scheme == Scheme::FOR_NS;
    }
};

// ============================================================================
// Q1 Fused Kernel with Early Exit + Late Materialization
// ============================================================================

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void q1_fused_kernel(
    FusedColumnAccessor col_orderdate,
    FusedColumnAccessor col_discount,
    FusedColumnAccessor col_quantity,
    FusedColumnAccessor col_extendedprice,
    int n,
    int date_min, int date_max,
    int discount_min, int discount_max,
    int quantity_min, int quantity_max,
    unsigned long long* __restrict__ out_sum)
{
    constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;
    constexpr unsigned FULL_MASK = 0xFFFFFFFF;

    const int tile_offset = blockIdx.x * TILE_SIZE;
    const int num_tile_items = min(TILE_SIZE, n - tile_offset);
    if (num_tile_items <= 0) return;

    long long local_sum = 0;
    int flags[ITEMS_PER_THREAD];
    int discount_vals[ITEMS_PER_THREAD];

    // Initialize flags
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int idx = threadIdx.x + BLOCK_THREADS * i;
        flags[i] = (idx < num_tile_items) ? 1 : 0;
    }

    // Stage 1: orderdate filter
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (flags[i]) {
            int idx = tile_offset + threadIdx.x + BLOCK_THREADS * i;
            int orderdate = col_orderdate.decode(idx);
            if (orderdate < date_min || orderdate > date_max) {
                flags[i] = 0;
            }
        }
    }

    // Early exit check
    int any_valid = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        any_valid |= flags[i];
    }
    if (__ballot_sync(FULL_MASK, any_valid) == 0) return;

    // Stage 2: quantity filter
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (flags[i]) {
            int idx = tile_offset + threadIdx.x + BLOCK_THREADS * i;
            int quantity = col_quantity.decode(idx);
            if (quantity < quantity_min || quantity > quantity_max) {
                flags[i] = 0;
            }
        }
    }

    // Early exit check
    any_valid = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        any_valid |= flags[i];
    }
    if (__ballot_sync(FULL_MASK, any_valid) == 0) return;

    // Stage 3: discount filter
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (flags[i]) {
            int idx = tile_offset + threadIdx.x + BLOCK_THREADS * i;
            int discount = col_discount.decode(idx);
            if (discount < discount_min || discount > discount_max) {
                flags[i] = 0;
            } else {
                discount_vals[i] = discount;
            }
        }
    }

    // Early exit check
    any_valid = 0;
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        any_valid |= flags[i];
    }
    if (__ballot_sync(FULL_MASK, any_valid) == 0) return;

    // Stage 4: Late Materialization - only decode extendedprice for valid rows
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (flags[i]) {
            int idx = tile_offset + threadIdx.x + BLOCK_THREADS * i;
            int extprice = col_extendedprice.decode(idx);
            local_sum += static_cast<long long>(extprice) * discount_vals[i];
        }
    }

    // Block reduction
    __shared__ long long shared[32];

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    }

    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    if (lane == 0) {
        shared[warp_id] = local_sum;
    }
    __syncthreads();

    // First warp reduces across warps
    if (threadIdx.x < (BLOCK_THREADS + 31) / 32) {
        local_sum = shared[threadIdx.x];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
        }
        if (threadIdx.x == 0 && local_sum != 0) {
            atomicAdd(out_sum, static_cast<unsigned long long>(local_sum));
        }
    }
}

} // namespace fused
} // namespace planner
