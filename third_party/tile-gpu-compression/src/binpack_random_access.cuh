#pragma once

// Random access interface for GPU-FOR (Frame of Reference with Bit-packing)
// Allows accessing individual elements without full decompression

// Helper function: decode single element from a block
__forceinline__ __device__ int decodeElementAtIndex(
    uint global_index,
    uint* block_start,
    uint* compressed_data) {

    // Each block contains 128 elements, with 4 miniblocks of 32 elements each
    const int BLOCK_SIZE = 128;
    const int MINIBLOCK_SIZE = 32;

    // Find which block contains this element
    uint block_idx = global_index / BLOCK_SIZE;
    uint index_in_block = global_index % BLOCK_SIZE;

    // Find which miniblock (0-3) contains this element
    uint miniblock_idx = index_in_block / MINIBLOCK_SIZE;
    uint index_in_miniblock = index_in_block % MINIBLOCK_SIZE;

    // Get pointer to this block's compressed data
    uint block_offset = block_start[block_idx];
    uint* data_block = compressed_data + block_offset;

    // Read reference value (first 4 bytes)
    int reference = reinterpret_cast<int*>(data_block)[0];

    // Read miniblock bitwidths (second 4 bytes)
    uint miniblock_bitwidths = data_block[1];

    // Calculate offset to this miniblock's data
    uint miniblock_offset = 2; // skip reference and bitwidths
    uint bitwidth = 0;
    for (uint i = 0; i < miniblock_idx; i++) {
        uint bw = (miniblock_bitwidths >> (i * 8)) & 0xFF;
        // Each miniblock: 32 elements * bitwidth bits, rounded up to 32-bit words
        miniblock_offset += (32 * bw + 31) / 32;
    }

    // Get this miniblock's bitwidth
    bitwidth = (miniblock_bitwidths >> (miniblock_idx * 8)) & 0xFF;

    // Calculate bit position within miniblock
    uint start_bitindex = bitwidth * index_in_miniblock;
    uint start_intindex = start_bitindex / 32;
    start_bitindex = start_bitindex % 32;

    // Read element using 64-bit to handle cross-word boundaries
    uint* element_ptr = data_block + miniblock_offset + start_intindex;
    unsigned long long element_block =
        (((unsigned long long)element_ptr[1]) << 32) | element_ptr[0];

    // Extract and return the element
    uint element = (element_block >> start_bitindex) & ((1LL << bitwidth) - 1LL);
    return reference + element;
}

// Kernel for batch random access
// indices: array of element indices to retrieve
// output: array to store retrieved elements
// num_queries: number of elements to retrieve
template<int BLOCK_THREADS>
__global__ void randomAccessKernel(
    uint* block_start,
    uint* compressed_data,
    uint* indices,
    int* output,
    int num_queries) {

    int tid = blockIdx.x * BLOCK_THREADS + threadIdx.x;

    if (tid < num_queries) {
        uint index = indices[tid];
        output[tid] = decodeElementAtIndex(index, block_start, compressed_data);
    }
}

// Kernel for single element random access (useful for testing)
__global__ void randomAccessSingleKernel(
    uint* block_start,
    uint* compressed_data,
    uint index,
    int* output) {

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *output = decodeElementAtIndex(index, block_start, compressed_data);
    }
}
