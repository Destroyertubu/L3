#pragma once
#include <cub/cub.cuh>
using namespace cub;

// Random access interface for GPU-RFOR (Run-Length Encoded Frame of Reference)
// Challenge: RLE encoding maps compressed runs to expanded positions
// Need to find which run contains the target index and expand it

// Helper: decode a run-length or value element at compressed position
__forceinline__ __device__ int decodeRLEElement(
    uint compressed_index,
    uint* data_block,
    uint reference,
    uint bitwidth) {

    uint start_bitindex = bitwidth * compressed_index;
    uint start_intindex = start_bitindex / 32;
    start_bitindex = start_bitindex % 32;

    unsigned long long element_block =
        (((unsigned long long)data_block[start_intindex + 1]) << 32) | data_block[start_intindex];

    uint element = (element_block >> start_bitindex) & ((1LL << bitwidth) - 1LL);
    return reference + element;
}

// Main function: decode element at global index with RLE
__forceinline__ __device__ int decodeElementAtIndexRLE(
    uint global_index,
    uint* val_block_start,
    uint* rl_block_start,
    uint* value_data,
    uint* run_length_data) {

    const int TILE_SIZE = 512; // Each RLE tile encodes 512 elements

    // Find which tile contains this element
    uint tile_idx = global_index / TILE_SIZE;

    // Get pointers to value and run-length blocks for this tile
    uint val_offset = val_block_start[tile_idx];
    uint rl_offset = rl_block_start[tile_idx];

    uint* val_block = value_data + val_offset;
    uint* rl_block = run_length_data + rl_offset;

    // Read metadata from both blocks
    uint val_reference = val_block[0];
    uint val_bitwidth = val_block[1] & 0xFF;
    uint count = val_block[2]; // Number of compressed runs

    uint rl_reference = rl_block[0];
    uint rl_bitwidth = rl_block[1] & 0xFF;

    // Data starts after 3-word header
    uint* val_data = val_block + 3;
    uint* rl_data = rl_block + 3;

    // Find which run contains our target index
    // We need to accumulate run lengths until we reach the target position
    uint accumulated_length = 0;
    uint target_position = global_index % TILE_SIZE;

    for (uint i = 0; i < count; i++) {
        int run_length = decodeRLEElement(i, rl_data, rl_reference, rl_bitwidth);

        if (accumulated_length + run_length > target_position) {
            // This run contains our target element
            int value = decodeRLEElement(i, val_data, val_reference, val_bitwidth);
            return value;
        }

        accumulated_length += run_length;
    }

    // If we reach here, index is out of bounds - return 0
    return 0;
}

// Batch random access kernel for RFOR
template<int BLOCK_THREADS>
__global__ void randomAccessRLEKernel(
    uint* val_block_start,
    uint* rl_block_start,
    uint* value_data,
    uint* run_length_data,
    uint* indices,
    int* output,
    int num_queries) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndexRLE(
            index, val_block_start, rl_block_start, value_data, run_length_data);
    }
}

// Optimized version with shared memory caching
// Useful when multiple queries target the same tile
template<int BLOCK_THREADS>
__global__ void randomAccessRLEOptimizedKernel(
    uint* val_block_start,
    uint* rl_block_start,
    uint* value_data,
    uint* run_length_data,
    uint* sorted_indices,
    int* output,
    int num_queries) {

    const int TILE_SIZE = 512;
    const int MAX_RUNS = 128; // Maximum runs we can cache

    __shared__ int cached_values[MAX_RUNS];
    __shared__ int cached_run_lengths[MAX_RUNS];
    __shared__ int cached_run_positions[MAX_RUNS]; // Prefix sum of run lengths
    __shared__ uint cached_tile_idx;
    __shared__ uint cached_count;

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = sorted_indices[tid];
        uint tile_idx = index / TILE_SIZE;
        uint target_position = index % TILE_SIZE;

        // Check if we need to load a new tile into cache
        bool need_reload = (threadIdx.x == 0) || (tid == 0);
        if (tid > 0) {
            uint prev_tile = sorted_indices[tid - 1] / TILE_SIZE;
            need_reload = (prev_tile != tile_idx);
        }

        // Load tile data into shared memory (cooperative loading)
        if (need_reload || threadIdx.x == 0) {
            if (threadIdx.x == 0) {
                cached_tile_idx = tile_idx;

                uint val_offset = val_block_start[tile_idx];
                uint rl_offset = rl_block_start[tile_idx];

                uint* val_block = value_data + val_offset;
                uint* rl_block = run_length_data + rl_offset;

                uint val_reference = val_block[0];
                uint val_bitwidth = val_block[1] & 0xFF;
                uint count = val_block[2];
                cached_count = count;

                uint rl_reference = rl_block[0];
                uint rl_bitwidth = rl_block[1] & 0xFF;

                uint* val_data = val_block + 3;
                uint* rl_data = rl_block + 3;

                // Decode all runs and compute prefix sums
                int accumulated = 0;
                for (uint i = 0; i < count && i < MAX_RUNS; i++) {
                    cached_values[i] = decodeRLEElement(i, val_data, val_reference, val_bitwidth);
                    cached_run_lengths[i] = decodeRLEElement(i, rl_data, rl_reference, rl_bitwidth);
                    cached_run_positions[i] = accumulated;
                    accumulated += cached_run_lengths[i];
                }
            }
        }

        __syncthreads();

        // Binary search to find the run containing target_position
        // For simplicity, using linear search here
        int result = 0;
        for (uint i = 0; i < cached_count && i < MAX_RUNS; i++) {
            if (target_position >= cached_run_positions[i] &&
                target_position < cached_run_positions[i] + cached_run_lengths[i]) {
                result = cached_values[i];
                break;
            }
        }

        output[tid] = result;
        __syncthreads();
    }
}

// Single element access
__global__ void randomAccessRLESingleKernel(
    uint* val_block_start,
    uint* rl_block_start,
    uint* value_data,
    uint* run_length_data,
    uint index,
    int* output) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndexRLE(
            index, val_block_start, rl_block_start, value_data, run_length_data);
    }
}
