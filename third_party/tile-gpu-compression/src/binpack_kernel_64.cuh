/**
 * 64-bit Bin Packing Kernel for tile-gpu-compression
 *
 * Extension to support 64-bit integers with bit widths up to 64.
 * Uses 128-bit arithmetic for cross-word boundary extraction.
 */

#pragma once
#include <cstdint>

// Helper: decode single 64-bit element from a block
__forceinline__ __device__ int64_t decodeElement64(
    int i,
    uint miniblock_index,
    uint index_into_miniblock,
    uint64_t* data_block,
    uint* bitwidths,
    uint* offsets)
{
    // Reference for the frame (64-bit)
    int64_t reference = reinterpret_cast<int64_t*>(data_block)[0];

    uint miniblock_offset = offsets[miniblock_index];
    uint bitwidth = bitwidths[miniblock_index];

    uint start_bitindex = (bitwidth * index_into_miniblock);
    uint start_intindex = 1 + (start_bitindex >> 6);  // 64-bit word index, skip reference

    start_bitindex = start_bitindex & 63;  // bit offset within word

    // Read element using 128-bit to handle cross-word boundaries
    uint64_t lo = data_block[miniblock_offset + start_intindex];
    uint64_t hi = data_block[miniblock_offset + start_intindex + 1];

    // Extract value handling cross-boundary
    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        // Fits in single word
        element = (lo >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);
    } else {
        // Crosses word boundary
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) | ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

// Simplified decoder for known bit width (for SIMD-like access)
template<int BITWIDTH>
__forceinline__ __device__ int64_t decodeElement64_fixed(
    uint index_into_miniblock,
    uint64_t* data_ptr,
    int64_t reference)
{
    static_assert(BITWIDTH >= 0 && BITWIDTH <= 64, "BITWIDTH must be 0-64");

    if constexpr (BITWIDTH == 0) {
        return reference;
    }
    else if constexpr (BITWIDTH == 64) {
        return static_cast<int64_t>(data_ptr[index_into_miniblock]);
    }
    else {
        constexpr uint64_t MASK = (1ULL << BITWIDTH) - 1ULL;

        // Calculate bit position
        uint start_bit = BITWIDTH * index_into_miniblock;
        uint word_idx = start_bit >> 6;
        uint bit_offset = start_bit & 63;

        uint64_t lo = data_ptr[word_idx];
        uint64_t element;

        if constexpr (BITWIDTH <= 64) {
            if (bit_offset + BITWIDTH <= 64) {
                // Single word
                element = (lo >> bit_offset) & MASK;
            } else {
                // Cross boundary
                uint64_t hi = data_ptr[word_idx + 1];
                uint bits_from_lo = 64 - bit_offset;
                element = (lo >> bit_offset) | ((hi << bits_from_lo) & MASK);
            }
        }

        return reference + static_cast<int64_t>(element);
    }
}

// Load 64-bit binpacked data
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadBinPack64(
    uint* block_start,
    uint64_t* data,
    uint64_t* shared_buffer,
    int64_t (&items)[ITEMS_PER_THREAD],
    bool is_last_tile,
    int num_tile_items)
{
    int tile_idx = blockIdx.x;
    int threadId = threadIdx.x;

    // Block start indices (in 64-bit word offsets)
    uint* block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    if (threadId < ITEMS_PER_THREAD + 1) {
        block_starts[threadId] = block_start[tile_idx * ITEMS_PER_THREAD + threadId];
    }
    __syncthreads();

    // Shared memory for encoded data blocks
    uint64_t* data_block = &shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 3)];

    // Load blocks from encoded column
    uint start_offset = block_starts[0];
    uint end_offset = block_starts[ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = start_offset + threadIdx.x + (i << 7);
        if (index < end_offset)
            data_block[threadIdx.x + (i << 7)] = data[index];
    }
    __syncthreads();

    // Read bitwidths and offsets
    uint* bitwidths = reinterpret_cast<uint*>(&shared_buffer[ITEMS_PER_THREAD + 1]);
    uint* offsets = reinterpret_cast<uint*>(&shared_buffer[ITEMS_PER_THREAD + 1 + (ITEMS_PER_THREAD << 2)]);

    if (threadId < (ITEMS_PER_THREAD << 2)) {
        int i = threadId >> 2;
        int miniblock_index = threadId & 3;

        // Miniblock bitwidths (stored as 4 bytes per block)
        uint* bw_ptr = reinterpret_cast<uint*>(data_block + block_starts[i] - block_starts[0] + 1);
        uint miniblock_bitwidths = *bw_ptr;

        // Calculate offset for this miniblock
        uint miniblock_offsets = (miniblock_bitwidths << 8) + (miniblock_bitwidths << 16) + (miniblock_bitwidths << 24);
        uint miniblock_offset = (miniblock_offsets >> (miniblock_index << 3)) & 255;
        uint bitwidth = (miniblock_bitwidths >> (miniblock_index << 3)) & 255;

        offsets[threadId] = miniblock_offset;
        bitwidths[threadId] = bitwidth;
    }
    __syncthreads();

    // Decode elements
    uint miniblock_index = threadIdx.x >> 5;
    uint index_into_miniblock = threadIdx.x & 31;

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items[i] = decodeElement64(
            threadIdx.x,
            miniblock_index,
            index_into_miniblock,
            data_block + block_starts[i] - block_starts[0],
            bitwidths + (i << 2),
            offsets + (i << 2));
    }
}

// Batch decode 64-bit elements from packed array
// Suitable for decompression kernels
template<int BLOCK_SIZE = 128>
__global__ void decodeBinPack64Kernel(
    const uint64_t* __restrict__ packed_data,
    int64_t* __restrict__ output,
    const uint* __restrict__ block_starts,
    int num_blocks,
    int elements_per_block = 128)
{
    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    int tid = threadIdx.x;
    int global_offset = block_idx * elements_per_block;

    // Get this block's data
    uint block_offset = block_starts[block_idx];
    const uint64_t* data_block = packed_data + block_offset;

    // Read reference (first 64-bit word)
    int64_t reference = reinterpret_cast<const int64_t*>(data_block)[0];

    // Read miniblock bitwidths (next 32 bits, stored in second word)
    uint miniblock_bitwidths = reinterpret_cast<const uint*>(data_block)[2];

    // For simplicity, assume all miniblocks have same bitwidth
    // (Can be extended to handle different bitwidths per miniblock)
    uint bitwidth = miniblock_bitwidths & 0xFF;

    // Skip header: 1 64-bit reference + 1 32-bit bitwidths = 12 bytes = 1.5 words
    // Aligned to 64-bit: 2 words
    const uint64_t* value_data = data_block + 2;

    // Each thread decodes multiple elements
    constexpr int VALS_PER_THREAD = (elements_per_block + BLOCK_SIZE - 1) / BLOCK_SIZE;

    #pragma unroll
    for (int v = 0; v < VALS_PER_THREAD; v++) {
        int idx = tid + v * BLOCK_SIZE;
        if (idx < elements_per_block) {
            // Calculate bit position
            uint start_bit = bitwidth * idx;
            uint word_idx = start_bit >> 6;
            uint bit_offset = start_bit & 63;

            uint64_t element;
            if (bitwidth == 0) {
                element = 0;
            } else if (bitwidth == 64) {
                element = value_data[idx];
            } else {
                uint64_t lo = value_data[word_idx];
                uint64_t mask = (1ULL << bitwidth) - 1ULL;

                if (bit_offset + bitwidth <= 64) {
                    element = (lo >> bit_offset) & mask;
                } else {
                    uint64_t hi = value_data[word_idx + 1];
                    uint bits_from_lo = 64 - bit_offset;
                    element = (lo >> bit_offset) | ((hi << bits_from_lo) & mask);
                }
            }

            output[global_offset + idx] = reference + static_cast<int64_t>(element);
        }
    }
}
