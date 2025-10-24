#pragma once
#include <cub/cub.cuh>
using namespace cub;

// Random access interface for GPU-DFOR (Delta Frame of Reference)
// Challenge: Delta encoding requires prefix sum reconstruction
// For random access, we need to decode all deltas up to the target index

// Helper: decode delta element from a block
__forceinline__ __device__ int decodeDeltaElement(
    uint index_in_block,
    uint* data_block) {

    const int MINIBLOCK_SIZE = 32;

    // Find which miniblock contains this element
    uint miniblock_idx = index_in_block / MINIBLOCK_SIZE;
    uint index_in_miniblock = index_in_block % MINIBLOCK_SIZE;

    // Read reference value (first 4 bytes, but this is skipped in delta encoding)
    // The actual first value is stored separately

    // Read miniblock bitwidths
    uint miniblock_bitwidths = data_block[0];

    // Calculate offset to this miniblock
    uint miniblock_offset = 1; // skip bitwidths
    for (uint i = 0; i < miniblock_idx; i++) {
        uint bw = (miniblock_bitwidths >> (i * 8)) & 0xFF;
        miniblock_offset += (32 * bw + 31) / 32;
    }

    // Get bitwidth for this miniblock
    uint bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

    // Calculate bit position
    uint start_bitindex = bitwidth * index_in_miniblock;
    uint start_intindex = start_bitindex / 32;
    start_bitindex = start_bitindex % 32;

    // Read element
    uint* element_ptr = data_block + miniblock_offset + start_intindex;
    unsigned long long element_block =
        (((unsigned long long)element_ptr[1]) << 32) | element_ptr[0];

    uint element = (element_block >> start_bitindex) & ((1LL << bitwidth) - 1LL);
    return element; // Return delta value (not yet accumulated)
}

// Decode element with prefix sum (requires decoding all elements up to target)
__forceinline__ __device__ int decodeElementAtIndexDelta(
    uint global_index,
    uint* block_start,
    uint* compressed_data) {

    const int BLOCK_SIZE = 128;

    // Find which block contains this element
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    // Get pointer to this block's compressed data
    uint block_offset = block_start[block_idx];
    uint* data_block = compressed_data + block_offset;

    // Read first value (stored before the block)
    int first_value = reinterpret_cast<int*>(data_block)[0];
    data_block += 1; // Move past first value

    // For element 0, return first value directly
    if (index_in_block == 0) {
        return first_value;
    }

    // Decode and accumulate deltas up to target index
    int accumulated = first_value;
    for (uint i = 0; i < index_in_block; i++) {
        int delta = decodeDeltaElement(i, data_block);
        accumulated += delta;
    }

    return accumulated;
}

// Batch random access kernel for DFOR
// Note: This is less efficient than FOR because we need to reconstruct prefix sums
template<int BLOCK_THREADS>
__global__ void randomAccessDeltaKernel(
    uint* block_start,
    uint* compressed_data,
    uint* indices,
    int* output,
    int num_queries) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndexDelta(index, block_start, compressed_data);
    }
}

// Optimized version: if queries are sorted and within same block, we can reuse computation
// This kernel assumes indices are sorted
template<int BLOCK_THREADS>
__global__ void randomAccessDeltaOptimizedKernel(
    uint* block_start,
    uint* compressed_data,
    uint* sorted_indices,
    int* output,
    int num_queries) {

    const int BLOCK_SIZE = 128;

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = sorted_indices[tid];
        uint block_idx = index / BLOCK_SIZE;
        uint index_in_block = index % BLOCK_SIZE;

        // Check if previous query was in same block
        bool can_reuse = false;
        int prev_value = 0;
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
            uint* data_block = compressed_data + block_offset + 1; // skip first value

            int accumulated = prev_value;
            for (uint i = prev_index_in_block; i < index_in_block; i++) {
                int delta = decodeDeltaElement(i, data_block);
                accumulated += delta;
            }
            output[tid] = accumulated;
        } else {
            // Decode from scratch
            output[tid] = decodeElementAtIndexDelta(index, block_start, compressed_data);
        }
    }
}

// Single element access
__global__ void randomAccessDeltaSingleKernel(
    uint* block_start,
    uint* compressed_data,
    uint index,
    int* output) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndexDelta(index, block_start, compressed_data);
    }
}
