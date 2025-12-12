/**
 * 64-bit Random Access Bin Packing for tile-gpu-compression
 *
 * Allows accessing individual 64-bit elements without full decompression.
 * Uses 128-bit arithmetic for cross-word boundary extraction.
 */

#pragma once
#include <cstdint>

// Decode single 64-bit element at given global index
__forceinline__ __device__ int64_t decodeElementAtIndex64(
    uint global_index,
    uint* block_start,
    uint64_t* compressed_data)
{
    // Each block contains 128 elements, with 4 miniblocks of 32 elements each
    constexpr int BLOCK_SIZE = 128;
    constexpr int MINIBLOCK_SIZE = 32;

    // Find which block contains this element
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    // Find which miniblock (0-3) contains this element
    uint miniblock_idx = index_in_block / MINIBLOCK_SIZE;
    uint index_in_miniblock = index_in_block % MINIBLOCK_SIZE;

    // Get pointer to this block's compressed data
    uint block_offset = block_start[block_idx];
    uint64_t* data_block = compressed_data + block_offset;

    // Read reference value (first 64-bit word)
    int64_t reference = reinterpret_cast<int64_t*>(data_block)[0];

    // Read miniblock bitwidths (stored in bits 0-31 of second 64-bit word)
    uint miniblock_bitwidths = reinterpret_cast<uint*>(data_block)[2];

    // Calculate offset to this miniblock's data in 64-bit words
    // Skip: 1 word reference + 1 word bitwidths (aligned) = 2 words header
    uint miniblock_offset = 2;
    uint bitwidth = 0;

    for (uint i = 0; i < miniblock_idx; i++) {
        uint bw = (miniblock_bitwidths >> (i * 8)) & 0xFF;
        // Each miniblock: 32 elements * bitwidth bits, rounded up to 64-bit words
        miniblock_offset += (32 * bw + 63) / 64;
    }

    // Get this miniblock's bitwidth
    bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

    // Handle special cases
    if (bitwidth == 0) {
        return reference;
    }

    if (bitwidth == 64) {
        return static_cast<int64_t>(data_block[miniblock_offset + index_in_miniblock]);
    }

    // Calculate bit position within miniblock
    uint start_bitindex = bitwidth * index_in_miniblock;
    uint start_wordindex = start_bitindex >> 6;  // / 64
    start_bitindex = start_bitindex & 63;        // % 64

    // Get pointer to the 64-bit words containing our element
    uint64_t* element_ptr = data_block + miniblock_offset + start_wordindex;
    uint64_t lo = element_ptr[0];
    uint64_t hi = element_ptr[1];

    // Extract element handling cross-word boundary
    uint64_t element;
    uint64_t mask = (1ULL << bitwidth) - 1ULL;

    if (start_bitindex + bitwidth <= 64) {
        // Fits in single word
        element = (lo >> start_bitindex) & mask;
    } else {
        // Crosses word boundary
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) |
                 ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

// Batch random access kernel for 64-bit data
template<int BLOCK_THREADS = 128>
__global__ void randomAccessKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint* indices,
    int64_t* output,
    int num_queries)
{
    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndex64(index, block_start, compressed_data);
    }
}

// Single element random access kernel (useful for testing)
__global__ void randomAccessSingleKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint index,
    int64_t* output)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndex64(index, block_start, compressed_data);
    }
}

// Gather operation: retrieve multiple elements at given indices
// More efficient than individual random access for multiple elements
template<int BLOCK_THREADS = 128, int ITEMS_PER_THREAD = 4>
__global__ void gatherKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    const uint* __restrict__ indices,
    int64_t* __restrict__ output,
    int num_queries)
{
    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    int base_idx = tid * ITEMS_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int query_idx = base_idx + i;
        if (query_idx < num_queries) {
            uint data_index = indices[query_idx];
            output[query_idx] = decodeElementAtIndex64(data_index, block_start, compressed_data);
        }
    }
}

// Range query: decode elements from start_idx to end_idx
// More efficient than individual access for contiguous ranges
template<int BLOCK_THREADS = 128>
__global__ void rangeQueryKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint start_idx,
    uint count,
    int64_t* output)
{
    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < count) {
        uint global_idx = start_idx + tid;
        output[tid] = decodeElementAtIndex64(global_idx, block_start, compressed_data);
    }
}

// Host-side helper: compute shared memory size needed
inline size_t getRandomAccessSharedMemSize64() {
    return 0;  // No shared memory needed for simple random access
}

// Optimized random access with caching block headers
// Uses shared memory to cache block metadata for better performance
template<int BLOCK_THREADS = 128>
__global__ void randomAccessCachedKernel64(
    uint* block_start,
    uint64_t* compressed_data,
    uint* indices,
    int64_t* output,
    int num_queries,
    int total_blocks)
{
    // Cache block start positions in shared memory if queries are localized
    __shared__ struct {
        uint start_offset;
        int64_t reference;
        uint miniblock_bitwidths;
        uint miniblock_offsets[4];
    } block_cache;

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    if (tid >= num_queries) return;

    uint global_index = indices[tid];
    constexpr int BLOCK_SIZE = 128;

    // Find which block contains this element
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    // Thread 0 loads block header into shared memory
    // This assumes most queries within a thread block target same data block
    // In practice, might need more sophisticated caching
    if (threadIdx.x == 0) {
        block_cache.start_offset = block_start[block_idx];
        uint64_t* data_block = compressed_data + block_cache.start_offset;
        block_cache.reference = reinterpret_cast<int64_t*>(data_block)[0];
        block_cache.miniblock_bitwidths = reinterpret_cast<uint*>(data_block)[2];

        // Precompute miniblock offsets
        uint offset = 2;  // Header size in 64-bit words
        uint bw_temp = block_cache.miniblock_bitwidths;
        for (int i = 0; i < 4; i++) {
            block_cache.miniblock_offsets[i] = offset;
            uint bw = bw_temp & 0xFF;
            offset += (32 * bw + 63) / 64;
            bw_temp >>= 8;
        }
    }
    __syncthreads();

    // Each thread can now use cached block info
    // Note: This optimization only helps if queries are localized to same block
    // For scattered queries, use randomAccessKernel64 instead

    uint miniblock_idx = index_in_block / 32;
    uint index_in_miniblock = index_in_block % 32;

    uint bitwidth = (block_cache.miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;
    int64_t reference = block_cache.reference;

    if (bitwidth == 0) {
        output[tid] = reference;
        return;
    }

    uint64_t* data_block = compressed_data + block_cache.start_offset;
    uint miniblock_offset = block_cache.miniblock_offsets[miniblock_idx];

    if (bitwidth == 64) {
        output[tid] = static_cast<int64_t>(data_block[miniblock_offset + index_in_miniblock]);
        return;
    }

    uint start_bitindex = bitwidth * index_in_miniblock;
    uint start_wordindex = start_bitindex >> 6;
    start_bitindex &= 63;

    uint64_t* element_ptr = data_block + miniblock_offset + start_wordindex;
    uint64_t lo = element_ptr[0];
    uint64_t hi = element_ptr[1];

    uint64_t element;
    uint64_t mask = (1ULL << bitwidth) - 1ULL;

    if (start_bitindex + bitwidth <= 64) {
        element = (lo >> start_bitindex) & mask;
    } else {
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) |
                 ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    output[tid] = reference + static_cast<int64_t>(element);
}

