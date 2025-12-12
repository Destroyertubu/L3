/**
 * @file unpack_specialized.cuh
 * @brief Specialized bit-unpacking for L3 interleaved format
 *
 * Pre-generated unpack functions for common bit widths to avoid runtime
 * division/modulo overhead. Similar to Vertical approach.
 *
 * Interleaved format for 1024-element tile (4 mini-vectors of 256):
 *   For each MV: Lane 0's 8 deltas, Lane 1's 8 deltas, ..., Lane 31's 8 deltas
 *   Thread L reads: MV0[L,0..7], MV1[L,0..7], MV2[L,0..7], MV3[L,0..7]
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_fused {

// ============================================================================
// Common Constants
// ============================================================================

constexpr int WARP_SIZE_UNPACK = 32;
constexpr int VALUES_PER_LANE = 8;   // 8 values per lane in mini-vector
constexpr int MV_BITS_STRIDE = 256;  // Bits per mini-vector = 256 * delta_bits

// ============================================================================
// Bit Extraction Helper
// ============================================================================

/**
 * Extract delta_bits starting at bit_offset from 64-bit aligned array
 */
__device__ __forceinline__
uint32_t extractDelta(const uint32_t* __restrict__ data, int64_t bit_offset, int bits)
{
    if (bits == 0) return 0;

    int64_t word64_idx = bit_offset >> 6;
    int shift = bit_offset & 63;

    const uint64_t* p64 = reinterpret_cast<const uint64_t*>(data);
    uint64_t lo = __ldg(&p64[word64_idx]);

    // Fast path: no crossing 64-bit boundary
    if (shift + bits <= 64) {
        return (lo >> shift) & ((1ULL << bits) - 1);
    }

    // Slow path: crosses boundary
    uint64_t hi = __ldg(&p64[word64_idx + 1]);
    return ((lo >> shift) | (hi << (64 - shift))) & ((1ULL << bits) - 1);
}

// ============================================================================
// Template Specialized Unpack for Common Bit Widths
// ============================================================================

/**
 * Unpack 32 values for a single thread from interleaved L3 format.
 *
 * @tparam BITS Delta bit width (compile-time constant)
 * @param data Compressed delta array
 * @param partition_bit_base Bit offset to start of partition's deltas
 * @param local_offset Offset within partition (tile_start - partition_start)
 * @param lane_id Thread index (0-31)
 * @param base FOR base value
 * @param output Output array (32 values)
 */
template<int BITS>
__device__ __forceinline__
void unpackFOR(const uint32_t* __restrict__ data,
               int64_t partition_bit_base,
               int local_offset,
               int lane_id,
               uint32_t base,
               uint32_t (&output)[32])
{
    // Compute starting position for this thread
    int local_idx_base = local_offset + lane_id;

    // For interleaved format:
    //   Thread L's i-th value is at local_idx = local_idx_base + i*32
    //   mv_idx = local_idx / 256
    //   local_in_mv = local_idx % 256
    //   lane_in_mv = local_in_mv % 32 (constant for all i!)
    //   v_in_lane = local_in_mv / 32

    // Since we add 32 each iteration, lane_in_mv stays constant
    int lane_in_mv = (local_idx_base % 256) % WARP_SIZE_UNPACK;

    // Bit offset for lane within mini-vector (8 consecutive deltas)
    int64_t lane_bit_offset = static_cast<int64_t>(lane_in_mv) * VALUES_PER_LANE * BITS;

    // Process all 32 values
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        int local_idx = local_idx_base + i * WARP_SIZE_UNPACK;
        int mv_idx = local_idx / 256;
        int v_in_lane = (local_idx % 256) / WARP_SIZE_UNPACK;  // 0-7

        // Bit offset = partition_base + mv_idx * 256 * BITS + lane_bit_offset + v_in_lane * BITS
        int64_t bit_offset = partition_bit_base +
            static_cast<int64_t>(mv_idx) * MV_BITS_STRIDE * BITS +
            lane_bit_offset +
            static_cast<int64_t>(v_in_lane) * BITS;

        if constexpr (BITS == 0) {
            output[i] = base;
        } else {
            uint32_t delta = extractDelta(data, bit_offset, BITS);
            output[i] = base + delta;
        }
    }
}

/**
 * Runtime dispatcher for FOR unpack (for variable bit widths)
 */
__device__ __forceinline__
void unpackFORRuntime(const uint32_t* __restrict__ data,
                       int64_t partition_bit_base,
                       int local_offset,
                       int lane_id,
                       int delta_bits,
                       uint32_t base,
                       uint32_t (&output)[32])
{
    switch (delta_bits) {
        case 0:  unpackFOR<0>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 1:  unpackFOR<1>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 2:  unpackFOR<2>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 3:  unpackFOR<3>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 4:  unpackFOR<4>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 5:  unpackFOR<5>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 6:  unpackFOR<6>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 7:  unpackFOR<7>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 8:  unpackFOR<8>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 9:  unpackFOR<9>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 10: unpackFOR<10>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 11: unpackFOR<11>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 12: unpackFOR<12>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 13: unpackFOR<13>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 14: unpackFOR<14>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 15: unpackFOR<15>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 16: unpackFOR<16>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 17: unpackFOR<17>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 18: unpackFOR<18>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 19: unpackFOR<19>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        case 20: unpackFOR<20>(data, partition_bit_base, local_offset, lane_id, base, output); break;
        default: {
            // Fallback for rare cases
            int local_idx_base = local_offset + lane_id;
            int lane_in_mv = (local_idx_base % 256) % WARP_SIZE_UNPACK;
            int64_t lane_bit_offset = static_cast<int64_t>(lane_in_mv) * VALUES_PER_LANE * delta_bits;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                int local_idx = local_idx_base + i * WARP_SIZE_UNPACK;
                int mv_idx = local_idx / 256;
                int v_in_lane = (local_idx % 256) / WARP_SIZE_UNPACK;
                int64_t bit_offset = partition_bit_base +
                    static_cast<int64_t>(mv_idx) * MV_BITS_STRIDE * delta_bits +
                    lane_bit_offset +
                    static_cast<int64_t>(v_in_lane) * delta_bits;
                uint32_t delta = extractDelta(data, bit_offset, delta_bits);
                output[i] = base + delta;
            }
            break;
        }
    }
}

/**
 * Conditional unpack - only extract values where flags are set
 */
template<int BITS>
__device__ __forceinline__
void unpackFORConditional(const uint32_t* __restrict__ data,
                          int64_t partition_bit_base,
                          int local_offset,
                          int lane_id,
                          uint32_t base,
                          const int (&flags)[32],
                          uint32_t (&output)[32])
{
    int local_idx_base = local_offset + lane_id;
    int lane_in_mv = (local_idx_base % 256) % WARP_SIZE_UNPACK;
    int64_t lane_bit_offset = static_cast<int64_t>(lane_in_mv) * VALUES_PER_LANE * BITS;

    #pragma unroll
    for (int i = 0; i < 32; i++) {
        if (flags[i]) {
            if constexpr (BITS == 0) {
                output[i] = base;
            } else {
                int local_idx = local_idx_base + i * WARP_SIZE_UNPACK;
                int mv_idx = local_idx / 256;
                int v_in_lane = (local_idx % 256) / WARP_SIZE_UNPACK;

                int64_t bit_offset = partition_bit_base +
                    static_cast<int64_t>(mv_idx) * MV_BITS_STRIDE * BITS +
                    lane_bit_offset +
                    static_cast<int64_t>(v_in_lane) * BITS;

                uint32_t delta = extractDelta(data, bit_offset, BITS);
                output[i] = base + delta;
            }
        }
    }
}

__device__ __forceinline__
void unpackFORConditionalRuntime(const uint32_t* __restrict__ data,
                                  int64_t partition_bit_base,
                                  int local_offset,
                                  int lane_id,
                                  int delta_bits,
                                  uint32_t base,
                                  const int (&flags)[32],
                                  uint32_t (&output)[32])
{
    switch (delta_bits) {
        case 0:  unpackFORConditional<0>(data, partition_bit_base, local_offset, lane_id, base, flags, output); break;
        case 4:  unpackFORConditional<4>(data, partition_bit_base, local_offset, lane_id, base, flags, output); break;
        case 6:  unpackFORConditional<6>(data, partition_bit_base, local_offset, lane_id, base, flags, output); break;
        case 16: unpackFORConditional<16>(data, partition_bit_base, local_offset, lane_id, base, flags, output); break;
        default: {
            int local_idx_base = local_offset + lane_id;
            int lane_in_mv = (local_idx_base % 256) % WARP_SIZE_UNPACK;
            int64_t lane_bit_offset = static_cast<int64_t>(lane_in_mv) * VALUES_PER_LANE * delta_bits;
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                if (flags[i]) {
                    int local_idx = local_idx_base + i * WARP_SIZE_UNPACK;
                    int mv_idx = local_idx / 256;
                    int v_in_lane = (local_idx % 256) / WARP_SIZE_UNPACK;
                    int64_t bit_offset = partition_bit_base +
                        static_cast<int64_t>(mv_idx) * MV_BITS_STRIDE * delta_bits +
                        lane_bit_offset +
                        static_cast<int64_t>(v_in_lane) * delta_bits;
                    uint32_t delta = extractDelta(data, bit_offset, delta_bits);
                    output[i] = base + delta;
                }
            }
            break;
        }
    }
}

} // namespace l3_fused
