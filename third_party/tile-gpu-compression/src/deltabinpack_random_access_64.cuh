/**
 * 64-bit Delta Bin Packing Random Access for tile-gpu-compression
 *
 * Random access for delta-encoded 64-bit data.
 * Challenge: Delta encoding requires prefix sum reconstruction.
 * For random access, we need to decode all deltas up to the target index.
 */

#pragma once
#include <cub/cub.cuh>
#include <cstdint>
using namespace cub;

// Helper: decode single delta element from a 64-bit block
__forceinline__ __device__ int64_t decodeDeltaElement64(
    uint index_in_block,
    uint64_t* data_block)
{
    constexpr int MINIBLOCK_SIZE = 32;

    // Find which miniblock contains this element
    uint miniblock_idx = index_in_block / MINIBLOCK_SIZE;
    uint index_in_miniblock = index_in_block % MINIBLOCK_SIZE;

    // Read miniblock bitwidths (first 32 bits of first 64-bit word after header)
    uint miniblock_bitwidths = reinterpret_cast<uint*>(data_block)[0];

    // Calculate offset to this miniblock in 64-bit words
    uint miniblock_offset = 1;  // Skip bitwidths word
    for (uint i = 0; i < miniblock_idx; i++) {
        uint bw = (miniblock_bitwidths >> (i * 8)) & 0xFF;
        // Each miniblock: 32 elements * bitwidth bits, rounded up to 64-bit words
        miniblock_offset += (32 * bw + 63) / 64;
    }

    // Get bitwidth for this miniblock
    uint bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

    // Handle special cases
    if (bitwidth == 0) {
        return 0;
    }

    // Calculate bit position
    uint start_bitindex = bitwidth * index_in_miniblock;
    uint start_wordindex = start_bitindex >> 6;
    start_bitindex = start_bitindex & 63;

    // Read element with cross-word boundary handling
    uint64_t* element_ptr = data_block + miniblock_offset + start_wordindex;

    if (bitwidth == 64) {
        return static_cast<int64_t>(element_ptr[0]);
    }

    uint64_t lo = element_ptr[0];
    uint64_t hi = element_ptr[1];
    uint64_t mask = (1ULL << bitwidth) - 1ULL;

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        element = (lo >> start_bitindex) & mask;
    } else {
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) |
                 ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return static_cast<int64_t>(element);  // Return delta value
}

// Decode element with prefix sum (requires decoding all elements up to target)
__forceinline__ __device__ int64_t decodeElementAtIndexDelta64(
    uint global_index,
    uint* block_start,
    uint64_t* compressed_data)
{
    constexpr int BLOCK_SIZE = 128;

    // Find which block contains this element
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    // Get pointer to this block's compressed data
    uint block_offset = block_start[block_idx];
    uint64_t* data_block = compressed_data + block_offset;

    // Read first value (stored as first 64-bit word)
    int64_t first_value = reinterpret_cast<int64_t*>(data_block)[0];
    data_block += 1;  // Move past first value

    // For element 0, return first value directly
    if (index_in_block == 0) {
        return first_value;
    }

    // Decode and accumulate deltas up to target index
    int64_t accumulated = first_value;
    for (uint i = 0; i < index_in_block; i++) {
        int64_t delta = decodeDeltaElement64(i, data_block);
        accumulated += delta;
    }

    return accumulated;
}

// Batch random access kernel for 64-bit DFOR
template<int BLOCK_THREADS = 128>
__global__ void randomAccessDeltaKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint* indices,
    int64_t* output,
    int num_queries)
{
    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndexDelta64(index, block_start, compressed_data);
    }
}

// Optimized version: if queries are sorted and within same block, reuse computation
template<int BLOCK_THREADS = 128>
__global__ void randomAccessDeltaOptimizedKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint* sorted_indices,
    int64_t* output,
    int num_queries)
{
    constexpr int BLOCK_SIZE = 128;

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid >= num_queries) return;

    uint index = sorted_indices[tid];
    uint block_idx = index / BLOCK_SIZE;
    uint index_in_block = index % BLOCK_SIZE;

    // Check if previous query was in same block
    bool can_reuse = false;
    int64_t prev_value = 0;
    uint prev_index_in_block = 0;

    if (tid > 0) {
        uint prev_index = sorted_indices[tid - 1];
        uint prev_block_idx = prev_index / BLOCK_SIZE;
        if (prev_block_idx == block_idx && prev_index < index) {
            can_reuse = true;
            prev_value = output[tid - 1];
            prev_index_in_block = prev_index % BLOCK_SIZE;
        }
    }

    if (can_reuse) {
        // Continue accumulating from previous result
        uint block_offset = block_start[block_idx];
        uint64_t* data_block = compressed_data + block_offset + 1;  // skip first value

        int64_t accumulated = prev_value;
        for (uint i = prev_index_in_block; i < index_in_block; i++) {
            int64_t delta = decodeDeltaElement64(i, data_block);
            accumulated += delta;
        }
        output[tid] = accumulated;
    } else {
        // Decode from scratch
        output[tid] = decodeElementAtIndexDelta64(index, block_start, compressed_data);
    }
}

// Single element access
__global__ void randomAccessDeltaSingleKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint index,
    int64_t* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndexDelta64(index, block_start, compressed_data);
    }
}

// Cooperative random access: multiple threads work together to decode a single element
// More efficient for high-latency random access patterns
template<int BLOCK_THREADS = 128>
__global__ void randomAccessDeltaCooperativeKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint* indices,
    int64_t* output,
    int num_queries)
{
    constexpr int BLOCK_SIZE = 128;
    constexpr int THREADS_PER_QUERY = 32;  // One warp per query

    __shared__ int64_t partial_sums[BLOCK_THREADS / THREADS_PER_QUERY];

    int warp_id = threadIdx.x / THREADS_PER_QUERY;
    int lane_id = threadIdx.x % THREADS_PER_QUERY;
    int query_id = blockIdx.x * (BLOCK_THREADS / THREADS_PER_QUERY) + warp_id;

    if (query_id >= num_queries) return;

    uint global_index = indices[query_id];
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    uint block_offset = block_start[block_idx];
    uint64_t* data_block = compressed_data + block_offset;
    int64_t first_value = reinterpret_cast<int64_t*>(data_block)[0];
    data_block += 1;

    if (index_in_block == 0) {
        if (lane_id == 0) {
            output[query_id] = first_value;
        }
        return;
    }

    // Distribute delta decoding across warp lanes
    int64_t local_sum = 0;
    for (uint i = lane_id; i < index_in_block; i += THREADS_PER_QUERY) {
        local_sum += decodeDeltaElement64(i, data_block);
    }

    // Warp-level reduction
    for (int offset = THREADS_PER_QUERY / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane_id == 0) {
        output[query_id] = first_value + local_sum;
    }
}

