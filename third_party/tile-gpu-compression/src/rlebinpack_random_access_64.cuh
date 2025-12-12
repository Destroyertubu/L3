/**
 * 64-bit RLE Random Access for tile-gpu-compression
 *
 * Random access interface for RLE-encoded 64-bit data.
 * Allows accessing individual elements without full decompression.
 */

#pragma once
#include <cub/cub.cuh>
#include <cstdint>
using namespace cub;

// ============================================================================
// Helper: decode a run-length or value element at compressed position
// ============================================================================
__forceinline__ __device__ int64_t decodeRLEElement64(
    uint compressed_index,
    uint64_t* data_block,
    int64_t reference,
    uint bitwidth) {

    // Handle special cases
    if (bitwidth == 0) {
        return reference;
    }

    if (bitwidth == 64) {
        return static_cast<int64_t>(data_block[compressed_index]);
    }

    uint start_bitindex = bitwidth * compressed_index;
    uint start_wordindex = start_bitindex >> 6;   // / 64
    start_bitindex = start_bitindex & 63;          // % 64

    // Read element with cross-word boundary handling
    uint64_t lo = data_block[start_wordindex];
    uint64_t hi = data_block[start_wordindex + 1];
    uint64_t mask = (1ULL << bitwidth) - 1ULL;

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        // Fits within single 64-bit word
        element = (lo >> start_bitindex) & mask;
    } else {
        // Crosses word boundary
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) |
                 ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

// ============================================================================
// Main function: decode element at global index with RLE
// ============================================================================
__forceinline__ __device__ int64_t decodeElementAtIndexRLE64(
    uint global_index,
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data) {

    const int TILE_SIZE = 512;  // Each RLE tile encodes 512 elements

    // Find which tile contains this element
    uint tile_idx = global_index / TILE_SIZE;

    // Get pointers to value and run-length blocks for this tile
    uint val_offset = val_block_start[tile_idx];
    uint rl_offset = rl_block_start[tile_idx];

    uint64_t* val_block = value_data + val_offset;
    uint64_t* rl_block = run_length_data + rl_offset;

    // Read metadata from both blocks
    // Header layout: [reference (8 bytes)][bitwidth (1 byte) + padding][count (4 bytes)]
    int64_t val_reference = reinterpret_cast<int64_t*>(val_block)[0];
    uint val_bitwidth = reinterpret_cast<uint*>(val_block)[2] & 0xFF;
    uint count = reinterpret_cast<uint*>(val_block)[4];  // Number of compressed runs

    int64_t rl_reference = reinterpret_cast<int64_t*>(rl_block)[0];
    uint rl_bitwidth = reinterpret_cast<uint*>(rl_block)[2] & 0xFF;

    // Data starts after header (2 64-bit words)
    uint64_t* val_data = val_block + 2;
    uint64_t* rl_data = rl_block + 2;

    // Find which run contains our target index
    // Accumulate run lengths until we reach the target position
    int64_t accumulated_length = 0;
    uint target_position = global_index % TILE_SIZE;

    for (uint i = 0; i < count; i++) {
        int64_t run_length = decodeRLEElement64(i, rl_data, rl_reference, rl_bitwidth);

        if (accumulated_length + run_length > target_position) {
            // This run contains our target element
            int64_t value = decodeRLEElement64(i, val_data, val_reference, val_bitwidth);
            return value;
        }

        accumulated_length += run_length;
    }

    // If we reach here, index is out of bounds - return 0
    return 0;
}

// ============================================================================
// Batch random access kernel for RLE 64-bit
// ============================================================================
template<int BLOCK_THREADS = 128>
__global__ void randomAccessRLEKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint* indices,
    int64_t* output,
    int num_queries) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndexRLE64(
            index, val_block_start, rl_block_start, value_data, run_length_data);
    }
}

// ============================================================================
// Optimized version with shared memory caching
// Useful when multiple queries target the same tile
// ============================================================================
template<int BLOCK_THREADS = 128>
__global__ void randomAccessRLEOptimizedKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint* sorted_indices,
    int64_t* output,
    int num_queries) {

    const int TILE_SIZE = 512;
    const int MAX_RUNS = 128;  // Maximum runs we can cache

    __shared__ int64_t cached_values[MAX_RUNS];
    __shared__ int64_t cached_run_lengths[MAX_RUNS];
    __shared__ int64_t cached_run_positions[MAX_RUNS];  // Prefix sum of run lengths
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

                uint64_t* val_block = value_data + val_offset;
                uint64_t* rl_block = run_length_data + rl_offset;

                int64_t val_reference = reinterpret_cast<int64_t*>(val_block)[0];
                uint val_bitwidth = reinterpret_cast<uint*>(val_block)[2] & 0xFF;
                uint count = reinterpret_cast<uint*>(val_block)[4];
                cached_count = count;

                int64_t rl_reference = reinterpret_cast<int64_t*>(rl_block)[0];
                uint rl_bitwidth = reinterpret_cast<uint*>(rl_block)[2] & 0xFF;

                uint64_t* val_data = val_block + 2;
                uint64_t* rl_data = rl_block + 2;

                // Decode all runs and compute prefix sums
                int64_t accumulated = 0;
                for (uint i = 0; i < count && i < MAX_RUNS; i++) {
                    cached_values[i] = decodeRLEElement64(i, val_data, val_reference, val_bitwidth);
                    cached_run_lengths[i] = decodeRLEElement64(i, rl_data, rl_reference, rl_bitwidth);
                    cached_run_positions[i] = accumulated;
                    accumulated += cached_run_lengths[i];
                }
            }
        }

        __syncthreads();

        // Binary search to find the run containing target_position
        // Using binary search for better performance with 64-bit data
        int64_t result = 0;
        int lo = 0, hi = min(cached_count, (uint)MAX_RUNS);

        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (cached_run_positions[mid - 1] + cached_run_lengths[mid - 1] <= target_position) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }

        // Verify and get result
        if (lo < cached_count && lo < MAX_RUNS) {
            if (target_position >= cached_run_positions[lo] &&
                target_position < cached_run_positions[lo] + cached_run_lengths[lo]) {
                result = cached_values[lo];
            }
        }

        output[tid] = result;
        __syncthreads();
    }
}

// ============================================================================
// Single element access
// ============================================================================
__global__ void randomAccessRLESingleKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint index,
    int64_t* output) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndexRLE64(
            index, val_block_start, rl_block_start, value_data, run_length_data);
    }
}

// ============================================================================
// Gather operation for multiple elements
// ============================================================================
template<int BLOCK_THREADS = 128, int ITEMS_PER_THREAD = 4>
__global__ void gatherRLEKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    const uint* __restrict__ indices,
    int64_t* __restrict__ output,
    int num_queries) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;
    int base_idx = tid * ITEMS_PER_THREAD;

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int query_idx = base_idx + i;
        if (query_idx < num_queries) {
            uint data_index = indices[query_idx];
            output[query_idx] = decodeElementAtIndexRLE64(
                data_index, val_block_start, rl_block_start, value_data, run_length_data);
        }
    }
}

// ============================================================================
// Range query: decode elements from start_idx for count elements
// ============================================================================
template<int BLOCK_THREADS = 128>
__global__ void rangeQueryRLEKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint start_idx,
    uint count,
    int64_t* output) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < count) {
        uint global_idx = start_idx + tid;
        output[tid] = decodeElementAtIndexRLE64(
            global_idx, val_block_start, rl_block_start, value_data, run_length_data);
    }
}

// ============================================================================
// Host-side helper: compute grid dimensions
// ============================================================================
inline dim3 getRLERandomAccessGridDim64(int num_queries, int block_threads = 128) {
    return dim3((num_queries + block_threads - 1) / block_threads);
}

