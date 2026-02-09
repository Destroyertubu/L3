/**
 * V15 Padding - Convert variable-offset compressed data to FLS-style fixed-offset format
 *
 * FLS-GPU uses fixed partition sizes (2048 values), allowing direct offset calculation:
 *   partition_bit_base = partition_idx * MINI_VECTOR_SIZE * BIT_WIDTH
 *
 * V15's original format uses variable offsets stored in d_interleaved_offsets.
 * This module repacks the data to fixed offsets for FLS-style decoding.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include "L3_Vertical_format.hpp"

namespace v15 {

// Kernel to repack compressed data from variable to fixed offsets
template<int BIT_WIDTH>
__global__ void repack_to_fixed_offset_kernel(
    const uint32_t* __restrict__ src_data,
    const int64_t* __restrict__ src_offsets,  // word offsets for each partition
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    uint32_t* __restrict__ dst_data,
    int num_partitions)
{
    int partition_idx = blockIdx.x;
    if (partition_idx >= num_partitions) return;

    int tid = threadIdx.x;

    // Source: variable offset
    int64_t src_word_offset = src_offsets[partition_idx];
    int partition_size = end_indices[partition_idx] - start_indices[partition_idx];

    // Destination: fixed offset (use compile-time MINI_VECTOR_SIZE from L3_Vertical_format.hpp)
    constexpr int WORDS_PER_PARTITION = (MINI_VECTOR_SIZE * BIT_WIDTH + 31) / 32;
    int64_t dst_word_offset = static_cast<int64_t>(partition_idx) * WORDS_PER_PARTITION;

    // Calculate how many words to copy (based on actual partition size)
    int actual_bits = partition_size * BIT_WIDTH;
    int actual_words = (actual_bits + 31) / 32;

    // Copy data (one thread per word)
    for (int w = tid; w < WORDS_PER_PARTITION; w += blockDim.x) {
        uint32_t val = 0;
        if (w < actual_words && src_word_offset >= 0) {
            val = src_data[src_word_offset + w];
        }
        dst_data[dst_word_offset + w] = val;
    }
}

// Pad a compressed column to fixed-offset format
template<int BIT_WIDTH>
inline void padColumnToFixedOffset(CompressedDataVertical<uint32_t>& col) {
    // Use compile-time MINI_VECTOR_SIZE from L3_Vertical_format.hpp
    constexpr int WORDS_PER_PARTITION = (MINI_VECTOR_SIZE * BIT_WIDTH + 31) / 32;

    int num_partitions = col.num_partitions;
    size_t new_size = static_cast<size_t>(num_partitions) * WORDS_PER_PARTITION;

    // Allocate new buffer
    uint32_t* d_new_data;
    cudaMalloc(&d_new_data, new_size * sizeof(uint32_t));
    cudaMemset(d_new_data, 0, new_size * sizeof(uint32_t));

    // Repack data
    int threads = 256;
    repack_to_fixed_offset_kernel<BIT_WIDTH><<<num_partitions, threads>>>(
        col.d_interleaved_deltas,
        col.d_interleaved_offsets,
        col.d_start_indices,
        col.d_end_indices,
        d_new_data,
        num_partitions);
    cudaDeviceSynchronize();

    // Replace old data
    cudaFree(col.d_interleaved_deltas);
    col.d_interleaved_deltas = d_new_data;
    col.interleaved_delta_words = new_size;

    // Update offsets to fixed values (no longer needed, but keep for compatibility)
    std::vector<int64_t> h_offsets(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        h_offsets[i] = static_cast<int64_t>(i) * WORDS_PER_PARTITION;
    }
    cudaMemcpy(col.d_interleaved_offsets, h_offsets.data(),
               num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);
}

} // namespace v15
