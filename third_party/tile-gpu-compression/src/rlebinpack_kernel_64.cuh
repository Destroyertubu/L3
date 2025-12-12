/**
 * 64-bit RLE Bin Packing Kernel for tile-gpu-compression
 *
 * Run-Length Encoded data with 64-bit values.
 * Uses bit-packing for both values and run lengths.
 */

#pragma once
#include <cub/cub.cuh>
#include <cstdint>
using namespace cub;

// Decode single element from RLE-binpacked 64-bit block
__forceinline__ __device__ int64_t decodeElementRBin64(
    int i,
    uint64_t* data_block,
    int64_t reference,
    uint bitwidth)
{
    // Handle special cases
    if (bitwidth == 0) {
        return reference;
    }

    uint start_bitindex = bitwidth * i;
    uint start_wordindex = start_bitindex >> 6;  // / 64
    start_bitindex = start_bitindex & 63;        // % 64

    if (bitwidth == 64) {
        return static_cast<int64_t>(data_block[start_wordindex]);
    }

    // Read element with cross-word boundary handling
    uint64_t lo = data_block[start_wordindex];
    uint64_t hi = data_block[start_wordindex + 1];
    uint64_t mask = (1ULL << bitwidth) - 1ULL;

    uint64_t element;
    if (start_bitindex + bitwidth <= 64) {
        element = (lo >> start_bitindex) & mask;
    } else {
        uint bits_from_lo = 64 - start_bitindex;
        element = (lo >> start_bitindex) |
                 ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
    }

    return reference + static_cast<int64_t>(element);
}

// Load RLE-binpacked 64-bit data
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadRBinPack64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint64_t* shared_buffer,
    int64_t (&items_value)[ITEMS_PER_THREAD],
    int64_t (&items_run_length)[ITEMS_PER_THREAD],
    bool is_last_tile,
    int num_tile_items)
{
    typedef cub::BlockExchange<int64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;
    typedef cub::BlockScan<int64_t, BLOCK_THREADS> BlockScan;

    uint num_decode;
    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Block start indices for values and run lengths
    uint* val_block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    uint* rl_block_starts = val_block_starts + 2;

    if (tid < 2) {
        val_block_starts[tid] = val_block_start[tile_idx + tid];
        rl_block_starts[tid] = rl_block_start[tile_idx + tid];
    }
    __syncthreads();

    // Shared memory for encoded data
    uint64_t* val_data_block = &shared_buffer[4];  // Skip block starts
    uint64_t* rl_data_block = val_data_block + BLOCK_THREADS * ITEMS_PER_THREAD;

    // Load value blocks
    uint start_offset_val = val_block_starts[0];
    uint end_offset_val = val_block_starts[1];
    uint start_offset_rl = rl_block_starts[0];
    uint end_offset_rl = rl_block_starts[1];

    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = start_offset_val + tid + (i * BLOCK_THREADS);
        if (index < end_offset_val) {
            val_data_block[tid + (i * BLOCK_THREADS)] = value_data[index];
        }
        index = start_offset_rl + tid + (i * BLOCK_THREADS);
        if (index < end_offset_rl) {
            rl_data_block[tid + (i * BLOCK_THREADS)] = run_length_data[index];
        }
    }
    __syncthreads();

    // Extract count (stored in header)
    // Header layout: [reference (8 bytes)][bitwidth + count (8 bytes)]
    uint count = reinterpret_cast<uint*>(val_data_block)[4];  // Count at word offset 4
    uint offset = 0;
    num_decode = ((count + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD);

    // Decode values and run lengths
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        // Skip header: 2 64-bit words (reference + metadata)
        uint64_t* val_ptr = val_data_block + 2;
        uint64_t* rl_ptr = rl_data_block + 2;

        int64_t val_reference, rl_reference;
        uint val_bitwidth, rl_bitwidth;

        if (tid < num_decode && tid + offset < count) {
            // Read value header
            val_reference = reinterpret_cast<int64_t*>(val_data_block)[0];
            val_bitwidth = reinterpret_cast<uint*>(val_data_block)[2] & 0xFF;
            items_value[i] = decodeElementRBin64(tid + offset, val_ptr, val_reference, val_bitwidth);

            // Read run length header
            rl_reference = reinterpret_cast<int64_t*>(rl_data_block)[0];
            rl_bitwidth = reinterpret_cast<uint*>(rl_data_block)[2] & 0xFF;
            items_run_length[i] = decodeElementRBin64(tid + offset, rl_ptr, rl_reference, rl_bitwidth);
        } else {
            items_value[i] = 0;
            items_run_length[i] = 0;
        }

        offset += num_decode;
    }
    __syncthreads();

    // Use union for shared memory reuse
    union SharedStorage {
        typename BlockScan::TempStorage scan;
        typename BlockExchange::TempStorage exchange;
    };
    SharedStorage* storage = reinterpret_cast<SharedStorage*>(rl_data_block);

    // Convert run lengths from striped to blocked arrangement
    BlockExchange(storage->exchange).StripedToBlocked(items_run_length);
    __syncthreads();

    // Compute prefix sum of run lengths
    BlockScan(storage->scan).InclusiveSum(items_run_length, items_run_length);

    // Build scatter table in shared memory
    int64_t* scatter_table = reinterpret_cast<int64_t*>(val_data_block);
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        scatter_table[tid * ITEMS_PER_THREAD + i] = 0;
    }
    __syncthreads();

    // Mark positions where values change
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (items_run_length[i] < BLOCK_THREADS * ITEMS_PER_THREAD) {
            scatter_table[items_run_length[i]] = 1;
        }
    }
    __syncthreads();

    // Convert markers to indices
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_run_length[i] = scatter_table[tid * ITEMS_PER_THREAD + i];
    }
    __syncthreads();

    // Prefix sum to get value indices
    BlockScan(storage->scan).InclusiveSum(items_run_length, items_run_length);
    __syncthreads();

    BlockExchange(storage->exchange).BlockedToStriped(items_run_length);
    __syncthreads();

    // Store values in order
    offset = 0;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (tid < num_decode) {
            scatter_table[tid + offset] = items_value[i];
        }
        offset += num_decode;
    }
    __syncthreads();

    // Gather final values based on run length expansion
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        items_value[i] = scatter_table[items_run_length[i]];
    }
}

// Random access for RLE-binpacked 64-bit data
__forceinline__ __device__ int64_t decodeRLEElementAtIndex64(
    uint global_index,
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data)
{
    // Find which RLE block contains this element
    uint block_idx = val_block_start[0];  // Simplified: assumes single block

    uint64_t* val_block = value_data + val_block_start[block_idx];
    uint64_t* rl_block = run_length_data + rl_block_start[block_idx];

    // Read metadata
    int64_t val_reference = reinterpret_cast<int64_t*>(val_block)[0];
    uint val_bitwidth = reinterpret_cast<uint*>(val_block)[2] & 0xFF;
    int64_t rl_reference = reinterpret_cast<int64_t*>(rl_block)[0];
    uint rl_bitwidth = reinterpret_cast<uint*>(rl_block)[2] & 0xFF;
    uint count = reinterpret_cast<uint*>(val_block)[4];

    uint64_t* val_ptr = val_block + 2;
    uint64_t* rl_ptr = rl_block + 2;

    // Linear search through run lengths to find the value
    // (Could be optimized with binary search for large blocks)
    int64_t cumulative_length = 0;
    for (uint i = 0; i < count; i++) {
        int64_t run_len = decodeElementRBin64(i, rl_ptr, rl_reference, rl_bitwidth);
        cumulative_length += run_len;
        if (global_index < cumulative_length) {
            return decodeElementRBin64(i, val_ptr, val_reference, val_bitwidth);
        }
    }

    // Index out of bounds
    return 0;
}

// Batch random access kernel for RLE 64-bit data
template<int BLOCK_THREADS = 128>
__global__ void randomAccessRLEKernel64(
    uint* val_block_start,
    uint* rl_block_start,
    uint64_t* value_data,
    uint64_t* run_length_data,
    uint* indices,
    int64_t* output,
    int num_queries)
{
    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeRLEElementAtIndex64(
            index, val_block_start, rl_block_start, value_data, run_length_data);
    }
}

