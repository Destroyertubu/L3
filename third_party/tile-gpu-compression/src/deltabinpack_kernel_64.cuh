/**
 * 64-bit Delta Bin Packing Kernel for tile-gpu-compression
 *
 * Extension to support 64-bit integers with delta encoding + bit-packing.
 * Uses CUB BlockScan for prefix sum and 128-bit arithmetic for cross-word extraction.
 */

#pragma once
#include <cub/cub.cuh>
#include <cstdint>
using namespace cub;

// Decode single element from delta-binpacked 64-bit block
__forceinline__ __device__ int64_t decodeElementDBin64(int i, uint64_t* data_block) {
    // Reference for the frame (first 64-bit word)
    int64_t reference = reinterpret_cast<int64_t*>(data_block)[0];

    // Index of miniblock containing i (32 elements per miniblock)
    uint miniblock_index = i / 32;

    // Miniblock bitwidths (stored in second 64-bit word as 4 bytes)
    // Only first 32 bits used for 4 miniblocks
    uint miniblock_bitwidths = reinterpret_cast<uint*>(data_block)[2];

    // Calculate miniblock offset in 64-bit words
    uint miniblock_offset = 0;
    uint temp_bw = miniblock_bitwidths;
    for (uint j = 0; j < miniblock_index; j++) {
        // Each miniblock: 32 elements * bitwidth bits
        // Rounded up to 64-bit word boundary
        uint bw = temp_bw & 255;
        miniblock_offset += (32 * bw + 63) / 64;
        temp_bw >>= 8;
    }

    // This miniblock's bitwidth
    uint bitwidth = temp_bw & 255;

    // Entry index within the miniblock
    uint index_into_miniblock = i & 31;

    // Calculate bit position
    uint start_bitindex = bitwidth * index_into_miniblock;
    // Skip first 2 64-bit words (reference + bitwidths)
    uint start_wordindex = 2 + (start_bitindex >> 6);
    start_bitindex = start_bitindex & 63;

    // Handle 64-bit extraction with potential cross-word boundary
    uint64_t element;
    if (bitwidth == 0) {
        element = 0;
    } else if (bitwidth == 64) {
        element = data_block[miniblock_offset + start_wordindex];
    } else {
        uint64_t lo = data_block[miniblock_offset + start_wordindex];
        uint64_t hi = data_block[miniblock_offset + start_wordindex + 1];

        if (start_bitindex + bitwidth <= 64) {
            // Fits in single word
            element = (lo >> start_bitindex) & ((1ULL << bitwidth) - 1ULL);
        } else {
            // Crosses word boundary
            uint bits_from_lo = 64 - start_bitindex;
            element = (lo >> start_bitindex) |
                     ((hi & ((1ULL << (bitwidth - bits_from_lo)) - 1ULL)) << bits_from_lo);
        }
    }

    return reference + static_cast<int64_t>(element);
}

// Load delta-binpacked 64-bit data with prefix sum
template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__forceinline__ __device__ void LoadDBinPack64(
    uint* block_start,       // Block start offsets (in 64-bit word units)
    uint64_t* data,          // Compressed data
    uint64_t* shared_buffer, // Shared memory buffer
    int64_t (&items)[ITEMS_PER_THREAD],
    bool is_last_tile,
    int num_tile_items)
{
    // CUB block exchange for data reordering
    typedef cub::BlockExchange<int64_t, BLOCK_THREADS, ITEMS_PER_THREAD> BlockExchange;

    // CUB BlockScan for inclusive prefix sum
    typedef cub::BlockScan<int64_t, BLOCK_THREADS> BlockScan;

    int tile_idx = blockIdx.x;
    int tid = threadIdx.x;

    // Load block start indices into shared memory
    uint* block_starts = reinterpret_cast<uint*>(&shared_buffer[0]);
    if (tid < ITEMS_PER_THREAD + 1) {
        block_starts[tid] = block_start[tile_idx * ITEMS_PER_THREAD + tid];
    }
    __syncthreads();

    // Shared memory for encoded data blocks
    uint64_t* data_block = &shared_buffer[ITEMS_PER_THREAD + 1];

    // Load blocks from encoded column
    uint start_offset = block_starts[0] - 1;  // First value stored separately
    uint end_offset = block_starts[ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        uint index = start_offset + tid + i * BLOCK_THREADS;
        if (index < end_offset) {
            data_block[tid + i * BLOCK_THREADS] = data[index];
        }
    }
    __syncthreads();

    // Extract first value (stored before the reference)
    int64_t first_value = reinterpret_cast<int64_t*>(data_block)[0];
    data_block = data_block + 1;  // Skip first value

    // Decode delta values from each block
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        if (is_last_tile) {
            if (tid + i * BLOCK_THREADS < num_tile_items) {
                items[i] = decodeElementDBin64(tid, data_block + block_starts[i] - block_starts[0]);
            } else {
                items[i] = 0;
            }
        } else {
            items[i] = decodeElementDBin64(tid, data_block + block_starts[i] - block_starts[0]);
        }
    }

    // First element gets the initial value
    if (tid == 0) {
        items[0] = first_value;
    }

    __syncthreads();

    // Use union for shared memory reuse
    union SharedStorage {
        typename BlockScan::TempStorage scan;
        typename BlockExchange::TempStorage exchange;
    };
    SharedStorage* storage = reinterpret_cast<SharedStorage*>(shared_buffer);

    // Convert from striped to blocked arrangement
    BlockExchange(storage->exchange).StripedToBlocked(items);
    __syncthreads();

    // Compute prefix sum (converts deltas to absolute values)
    BlockScan(storage->scan).InclusiveSum(items, items);
    __syncthreads();

    // Convert back to striped arrangement
    BlockExchange(storage->exchange).BlockedToStriped(items);
}

// Batch decode delta-binpacked 64-bit data
template<int BLOCK_SIZE = 128>
__global__ void decodeDeltaBinPack64Kernel(
    const uint64_t* __restrict__ packed_data,
    int64_t* __restrict__ output,
    const uint* __restrict__ block_starts,
    int64_t first_value,     // Initial value for prefix sum
    int num_blocks,
    int elements_per_block = 128)
{
    // Shared memory for scan and exchange
    __shared__ union {
        typename cub::BlockScan<int64_t, BLOCK_SIZE>::TempStorage scan;
    } shared;

    int block_idx = blockIdx.x;
    if (block_idx >= num_blocks) return;

    int tid = threadIdx.x;
    int global_offset = block_idx * elements_per_block;

    // Get this block's compressed data
    uint block_offset = block_starts[block_idx];
    const uint64_t* data_block = packed_data + block_offset;

    // Read reference (first 64-bit word)
    int64_t reference = reinterpret_cast<const int64_t*>(data_block)[0];

    // Read miniblock bitwidths
    uint miniblock_bitwidths = reinterpret_cast<const uint*>(data_block)[2];

    // For simplicity, assume uniform bitwidth (can be extended)
    uint bitwidth = miniblock_bitwidths & 0xFF;

    // Skip header: reference (8 bytes) + bitwidths (4 bytes aligned to 8)
    const uint64_t* value_data = data_block + 2;

    // Decode delta values
    int64_t delta;
    constexpr int VALS_PER_THREAD = (128 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int64_t local_vals[VALS_PER_THREAD];

    #pragma unroll
    for (int v = 0; v < VALS_PER_THREAD; v++) {
        int idx = tid + v * BLOCK_SIZE;
        if (idx < elements_per_block) {
            if (bitwidth == 0) {
                delta = 0;
            } else if (bitwidth == 64) {
                delta = static_cast<int64_t>(value_data[idx]);
            } else {
                uint start_bit = bitwidth * idx;
                uint word_idx = start_bit >> 6;
                uint bit_offset = start_bit & 63;

                uint64_t lo = value_data[word_idx];
                uint64_t mask = (1ULL << bitwidth) - 1ULL;

                uint64_t element;
                if (bit_offset + bitwidth <= 64) {
                    element = (lo >> bit_offset) & mask;
                } else {
                    uint64_t hi = value_data[word_idx + 1];
                    uint bits_from_lo = 64 - bit_offset;
                    element = (lo >> bit_offset) | ((hi << bits_from_lo) & mask);
                }
                delta = reference + static_cast<int64_t>(element);
            }
            local_vals[v] = delta;
        } else {
            local_vals[v] = 0;
        }
    }

    // Perform prefix sum to convert deltas to values
    int64_t thread_sum = 0;
    #pragma unroll
    for (int v = 0; v < VALS_PER_THREAD; v++) {
        thread_sum += local_vals[v];
        local_vals[v] = thread_sum;
    }

    // Block-level scan
    int64_t block_aggregate;
    cub::BlockScan<int64_t, BLOCK_SIZE>(shared.scan).ExclusiveSum(thread_sum, thread_sum, block_aggregate);

    // Add block prefix and initial value
    int64_t block_prefix = (block_idx == 0) ? first_value : 0;  // TODO: Handle multi-block prefix

    // Write output
    #pragma unroll
    for (int v = 0; v < VALS_PER_THREAD; v++) {
        int idx = tid + v * BLOCK_SIZE;
        if (idx < elements_per_block) {
            output[global_offset + idx] = local_vals[v] + thread_sum + block_prefix;
        }
    }
}

