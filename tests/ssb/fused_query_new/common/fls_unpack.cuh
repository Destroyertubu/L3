/**
 * @file fls_unpack.cuh
 * @brief Vertical-style compile-time specialized unpack functions for L3
 *
 * This file provides compile-time specialized bit unpacking that matches
 * Vertical-GPU's performance characteristics:
 * - 32 threads, each unpacking 32 values = 1024 elements per tile
 * - Compile-time specialized for each bitwidth (0-32)
 * - Strided memory access pattern for coalesced reads
 *
 * Memory Layout:
 *   Tile N starts at offset: N * bitwidth * 32 (in uint32_t words)
 *   Each tile contains: bitwidth * 32 uint32_t words
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_fls {

// ============================================================================
// Compile-time specialized unpack functions
// Pattern: 32 threads, each thread produces 32 output values
// Input: bitwidth * 32 uint32_t words per tile
// Output: 1024 uint32_t values per tile
// ============================================================================

__device__ __forceinline__ void unpack_0bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = 0;
    }
}

__device__ __forceinline__ void unpack_1bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg = __ldg(&in[i]);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = (reg >> j) & 0x1;
    }
}

__device__ __forceinline__ void unpack_2bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);

    out[0]  = (reg0) & 0x3;
    out[1]  = (reg0 >> 2) & 0x3;
    out[2]  = (reg0 >> 4) & 0x3;
    out[3]  = (reg0 >> 6) & 0x3;
    out[4]  = (reg0 >> 8) & 0x3;
    out[5]  = (reg0 >> 10) & 0x3;
    out[6]  = (reg0 >> 12) & 0x3;
    out[7]  = (reg0 >> 14) & 0x3;
    out[8]  = (reg0 >> 16) & 0x3;
    out[9]  = (reg0 >> 18) & 0x3;
    out[10] = (reg0 >> 20) & 0x3;
    out[11] = (reg0 >> 22) & 0x3;
    out[12] = (reg0 >> 24) & 0x3;
    out[13] = (reg0 >> 26) & 0x3;
    out[14] = (reg0 >> 28) & 0x3;
    out[15] = (reg0 >> 30) & 0x3;
    out[16] = (reg1) & 0x3;
    out[17] = (reg1 >> 2) & 0x3;
    out[18] = (reg1 >> 4) & 0x3;
    out[19] = (reg1 >> 6) & 0x3;
    out[20] = (reg1 >> 8) & 0x3;
    out[21] = (reg1 >> 10) & 0x3;
    out[22] = (reg1 >> 12) & 0x3;
    out[23] = (reg1 >> 14) & 0x3;
    out[24] = (reg1 >> 16) & 0x3;
    out[25] = (reg1 >> 18) & 0x3;
    out[26] = (reg1 >> 20) & 0x3;
    out[27] = (reg1 >> 22) & 0x3;
    out[28] = (reg1 >> 24) & 0x3;
    out[29] = (reg1 >> 26) & 0x3;
    out[30] = (reg1 >> 28) & 0x3;
    out[31] = (reg1 >> 30) & 0x3;
}

__device__ __forceinline__ void unpack_3bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);

    out[0]  = (reg0) & 0x7;
    out[1]  = (reg0 >> 3) & 0x7;
    out[2]  = (reg0 >> 6) & 0x7;
    out[3]  = (reg0 >> 9) & 0x7;
    out[4]  = (reg0 >> 12) & 0x7;
    out[5]  = (reg0 >> 15) & 0x7;
    out[6]  = (reg0 >> 18) & 0x7;
    out[7]  = (reg0 >> 21) & 0x7;
    out[8]  = (reg0 >> 24) & 0x7;
    out[9]  = (reg0 >> 27) & 0x7;
    out[10] = ((reg0 >> 30) | (reg1 << 2)) & 0x7;
    out[11] = (reg1 >> 1) & 0x7;
    out[12] = (reg1 >> 4) & 0x7;
    out[13] = (reg1 >> 7) & 0x7;
    out[14] = (reg1 >> 10) & 0x7;
    out[15] = (reg1 >> 13) & 0x7;
    out[16] = (reg1 >> 16) & 0x7;
    out[17] = (reg1 >> 19) & 0x7;
    out[18] = (reg1 >> 22) & 0x7;
    out[19] = (reg1 >> 25) & 0x7;
    out[20] = (reg1 >> 28) & 0x7;
    out[21] = ((reg1 >> 31) | (reg2 << 1)) & 0x7;
    out[22] = (reg2 >> 2) & 0x7;
    out[23] = (reg2 >> 5) & 0x7;
    out[24] = (reg2 >> 8) & 0x7;
    out[25] = (reg2 >> 11) & 0x7;
    out[26] = (reg2 >> 14) & 0x7;
    out[27] = (reg2 >> 17) & 0x7;
    out[28] = (reg2 >> 20) & 0x7;
    out[29] = (reg2 >> 23) & 0x7;
    out[30] = (reg2 >> 26) & 0x7;
    out[31] = (reg2 >> 29) & 0x7;
}

__device__ __forceinline__ void unpack_4bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);
    uint32_t reg3 = __ldg(&in[i + 96]);

    out[0]  = (reg0) & 0xF;
    out[1]  = (reg0 >> 4) & 0xF;
    out[2]  = (reg0 >> 8) & 0xF;
    out[3]  = (reg0 >> 12) & 0xF;
    out[4]  = (reg0 >> 16) & 0xF;
    out[5]  = (reg0 >> 20) & 0xF;
    out[6]  = (reg0 >> 24) & 0xF;
    out[7]  = (reg0 >> 28) & 0xF;
    out[8]  = (reg1) & 0xF;
    out[9]  = (reg1 >> 4) & 0xF;
    out[10] = (reg1 >> 8) & 0xF;
    out[11] = (reg1 >> 12) & 0xF;
    out[12] = (reg1 >> 16) & 0xF;
    out[13] = (reg1 >> 20) & 0xF;
    out[14] = (reg1 >> 24) & 0xF;
    out[15] = (reg1 >> 28) & 0xF;
    out[16] = (reg2) & 0xF;
    out[17] = (reg2 >> 4) & 0xF;
    out[18] = (reg2 >> 8) & 0xF;
    out[19] = (reg2 >> 12) & 0xF;
    out[20] = (reg2 >> 16) & 0xF;
    out[21] = (reg2 >> 20) & 0xF;
    out[22] = (reg2 >> 24) & 0xF;
    out[23] = (reg2 >> 28) & 0xF;
    out[24] = (reg3) & 0xF;
    out[25] = (reg3 >> 4) & 0xF;
    out[26] = (reg3 >> 8) & 0xF;
    out[27] = (reg3 >> 12) & 0xF;
    out[28] = (reg3 >> 16) & 0xF;
    out[29] = (reg3 >> 20) & 0xF;
    out[30] = (reg3 >> 24) & 0xF;
    out[31] = (reg3 >> 28) & 0xF;
}

__device__ __forceinline__ void unpack_5bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);
    uint32_t reg3 = __ldg(&in[i + 96]);
    uint32_t reg4 = __ldg(&in[i + 128]);

    out[0]  = (reg0) & 0x1F;
    out[1]  = (reg0 >> 5) & 0x1F;
    out[2]  = (reg0 >> 10) & 0x1F;
    out[3]  = (reg0 >> 15) & 0x1F;
    out[4]  = (reg0 >> 20) & 0x1F;
    out[5]  = (reg0 >> 25) & 0x1F;
    out[6]  = ((reg0 >> 30) | (reg1 << 2)) & 0x1F;
    out[7]  = (reg1 >> 3) & 0x1F;
    out[8]  = (reg1 >> 8) & 0x1F;
    out[9]  = (reg1 >> 13) & 0x1F;
    out[10] = (reg1 >> 18) & 0x1F;
    out[11] = (reg1 >> 23) & 0x1F;
    out[12] = ((reg1 >> 28) | (reg2 << 4)) & 0x1F;
    out[13] = (reg2 >> 1) & 0x1F;
    out[14] = (reg2 >> 6) & 0x1F;
    out[15] = (reg2 >> 11) & 0x1F;
    out[16] = (reg2 >> 16) & 0x1F;
    out[17] = (reg2 >> 21) & 0x1F;
    out[18] = (reg2 >> 26) & 0x1F;
    out[19] = ((reg2 >> 31) | (reg3 << 1)) & 0x1F;
    out[20] = (reg3 >> 4) & 0x1F;
    out[21] = (reg3 >> 9) & 0x1F;
    out[22] = (reg3 >> 14) & 0x1F;
    out[23] = (reg3 >> 19) & 0x1F;
    out[24] = (reg3 >> 24) & 0x1F;
    out[25] = ((reg3 >> 29) | (reg4 << 3)) & 0x1F;
    out[26] = (reg4 >> 2) & 0x1F;
    out[27] = (reg4 >> 7) & 0x1F;
    out[28] = (reg4 >> 12) & 0x1F;
    out[29] = (reg4 >> 17) & 0x1F;
    out[30] = (reg4 >> 22) & 0x1F;
    out[31] = (reg4 >> 27) & 0x1F;
}

__device__ __forceinline__ void unpack_6bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);
    uint32_t reg3 = __ldg(&in[i + 96]);
    uint32_t reg4 = __ldg(&in[i + 128]);
    uint32_t reg5 = __ldg(&in[i + 160]);

    out[0]  = (reg0) & 0x3F;
    out[1]  = (reg0 >> 6) & 0x3F;
    out[2]  = (reg0 >> 12) & 0x3F;
    out[3]  = (reg0 >> 18) & 0x3F;
    out[4]  = (reg0 >> 24) & 0x3F;
    out[5]  = ((reg0 >> 30) | (reg1 << 2)) & 0x3F;
    out[6]  = (reg1 >> 4) & 0x3F;
    out[7]  = (reg1 >> 10) & 0x3F;
    out[8]  = (reg1 >> 16) & 0x3F;
    out[9]  = (reg1 >> 22) & 0x3F;
    out[10] = ((reg1 >> 28) | (reg2 << 4)) & 0x3F;
    out[11] = (reg2 >> 2) & 0x3F;
    out[12] = (reg2 >> 8) & 0x3F;
    out[13] = (reg2 >> 14) & 0x3F;
    out[14] = (reg2 >> 20) & 0x3F;
    out[15] = (reg2 >> 26) & 0x3F;
    out[16] = (reg3) & 0x3F;
    out[17] = (reg3 >> 6) & 0x3F;
    out[18] = (reg3 >> 12) & 0x3F;
    out[19] = (reg3 >> 18) & 0x3F;
    out[20] = (reg3 >> 24) & 0x3F;
    out[21] = ((reg3 >> 30) | (reg4 << 2)) & 0x3F;
    out[22] = (reg4 >> 4) & 0x3F;
    out[23] = (reg4 >> 10) & 0x3F;
    out[24] = (reg4 >> 16) & 0x3F;
    out[25] = (reg4 >> 22) & 0x3F;
    out[26] = ((reg4 >> 28) | (reg5 << 4)) & 0x3F;
    out[27] = (reg5 >> 2) & 0x3F;
    out[28] = (reg5 >> 8) & 0x3F;
    out[29] = (reg5 >> 14) & 0x3F;
    out[30] = (reg5 >> 20) & 0x3F;
    out[31] = (reg5 >> 26) & 0x3F;
}

__device__ __forceinline__ void unpack_7bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);
    uint32_t reg3 = __ldg(&in[i + 96]);
    uint32_t reg4 = __ldg(&in[i + 128]);
    uint32_t reg5 = __ldg(&in[i + 160]);
    uint32_t reg6 = __ldg(&in[i + 192]);

    out[0]  = (reg0) & 0x7F;
    out[1]  = (reg0 >> 7) & 0x7F;
    out[2]  = (reg0 >> 14) & 0x7F;
    out[3]  = (reg0 >> 21) & 0x7F;
    out[4]  = ((reg0 >> 28) | (reg1 << 4)) & 0x7F;
    out[5]  = (reg1 >> 3) & 0x7F;
    out[6]  = (reg1 >> 10) & 0x7F;
    out[7]  = (reg1 >> 17) & 0x7F;
    out[8]  = (reg1 >> 24) & 0x7F;
    out[9]  = ((reg1 >> 31) | (reg2 << 1)) & 0x7F;
    out[10] = (reg2 >> 6) & 0x7F;
    out[11] = (reg2 >> 13) & 0x7F;
    out[12] = (reg2 >> 20) & 0x7F;
    out[13] = ((reg2 >> 27) | (reg3 << 5)) & 0x7F;
    out[14] = (reg3 >> 2) & 0x7F;
    out[15] = (reg3 >> 9) & 0x7F;
    out[16] = (reg3 >> 16) & 0x7F;
    out[17] = (reg3 >> 23) & 0x7F;
    out[18] = ((reg3 >> 30) | (reg4 << 2)) & 0x7F;
    out[19] = (reg4 >> 5) & 0x7F;
    out[20] = (reg4 >> 12) & 0x7F;
    out[21] = (reg4 >> 19) & 0x7F;
    out[22] = ((reg4 >> 26) | (reg5 << 6)) & 0x7F;
    out[23] = (reg5 >> 1) & 0x7F;
    out[24] = (reg5 >> 8) & 0x7F;
    out[25] = (reg5 >> 15) & 0x7F;
    out[26] = (reg5 >> 22) & 0x7F;
    out[27] = ((reg5 >> 29) | (reg6 << 3)) & 0x7F;
    out[28] = (reg6 >> 4) & 0x7F;
    out[29] = (reg6 >> 11) & 0x7F;
    out[30] = (reg6 >> 18) & 0x7F;
    out[31] = (reg6 >> 25) & 0x7F;
}

__device__ __forceinline__ void unpack_8bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg0 = __ldg(&in[i]);
    uint32_t reg1 = __ldg(&in[i + 32]);
    uint32_t reg2 = __ldg(&in[i + 64]);
    uint32_t reg3 = __ldg(&in[i + 96]);
    uint32_t reg4 = __ldg(&in[i + 128]);
    uint32_t reg5 = __ldg(&in[i + 160]);
    uint32_t reg6 = __ldg(&in[i + 192]);
    uint32_t reg7 = __ldg(&in[i + 224]);

    out[0]  = (reg0) & 0xFF;
    out[1]  = (reg0 >> 8) & 0xFF;
    out[2]  = (reg0 >> 16) & 0xFF;
    out[3]  = (reg0 >> 24) & 0xFF;
    out[4]  = (reg1) & 0xFF;
    out[5]  = (reg1 >> 8) & 0xFF;
    out[6]  = (reg1 >> 16) & 0xFF;
    out[7]  = (reg1 >> 24) & 0xFF;
    out[8]  = (reg2) & 0xFF;
    out[9]  = (reg2 >> 8) & 0xFF;
    out[10] = (reg2 >> 16) & 0xFF;
    out[11] = (reg2 >> 24) & 0xFF;
    out[12] = (reg3) & 0xFF;
    out[13] = (reg3 >> 8) & 0xFF;
    out[14] = (reg3 >> 16) & 0xFF;
    out[15] = (reg3 >> 24) & 0xFF;
    out[16] = (reg4) & 0xFF;
    out[17] = (reg4 >> 8) & 0xFF;
    out[18] = (reg4 >> 16) & 0xFF;
    out[19] = (reg4 >> 24) & 0xFF;
    out[20] = (reg5) & 0xFF;
    out[21] = (reg5 >> 8) & 0xFF;
    out[22] = (reg5 >> 16) & 0xFF;
    out[23] = (reg5 >> 24) & 0xFF;
    out[24] = (reg6) & 0xFF;
    out[25] = (reg6 >> 8) & 0xFF;
    out[26] = (reg6 >> 16) & 0xFF;
    out[27] = (reg6 >> 24) & 0xFF;
    out[28] = (reg7) & 0xFF;
    out[29] = (reg7 >> 8) & 0xFF;
    out[30] = (reg7 >> 16) & 0xFF;
    out[31] = (reg7 >> 24) & 0xFF;
}

__device__ __forceinline__ void unpack_16bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint32_t reg = __ldg(&in[i + j * 32]);
        out[j * 2] = reg & 0xFFFF;
        out[j * 2 + 1] = reg >> 16;
    }
}

__device__ __forceinline__ void unpack_32bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = __ldg(&in[i + j * 32]);
    }
}

// Additional common bitwidths for SSB queries
__device__ __forceinline__ void unpack_10bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg[10];

    #pragma unroll
    for (int j = 0; j < 10; j++) {
        reg[j] = __ldg(&in[i + j * 32]);
    }

    out[0]  = (reg[0]) & 0x3FF;
    out[1]  = (reg[0] >> 10) & 0x3FF;
    out[2]  = (reg[0] >> 20) & 0x3FF;
    out[3]  = ((reg[0] >> 30) | (reg[1] << 2)) & 0x3FF;
    out[4]  = (reg[1] >> 8) & 0x3FF;
    out[5]  = (reg[1] >> 18) & 0x3FF;
    out[6]  = ((reg[1] >> 28) | (reg[2] << 4)) & 0x3FF;
    out[7]  = (reg[2] >> 6) & 0x3FF;
    out[8]  = (reg[2] >> 16) & 0x3FF;
    out[9]  = ((reg[2] >> 26) | (reg[3] << 6)) & 0x3FF;
    out[10] = (reg[3] >> 4) & 0x3FF;
    out[11] = (reg[3] >> 14) & 0x3FF;
    out[12] = ((reg[3] >> 24) | (reg[4] << 8)) & 0x3FF;
    out[13] = (reg[4] >> 2) & 0x3FF;
    out[14] = (reg[4] >> 12) & 0x3FF;
    out[15] = (reg[4] >> 22) & 0x3FF;
    out[16] = (reg[5]) & 0x3FF;
    out[17] = (reg[5] >> 10) & 0x3FF;
    out[18] = (reg[5] >> 20) & 0x3FF;
    out[19] = ((reg[5] >> 30) | (reg[6] << 2)) & 0x3FF;
    out[20] = (reg[6] >> 8) & 0x3FF;
    out[21] = (reg[6] >> 18) & 0x3FF;
    out[22] = ((reg[6] >> 28) | (reg[7] << 4)) & 0x3FF;
    out[23] = (reg[7] >> 6) & 0x3FF;
    out[24] = (reg[7] >> 16) & 0x3FF;
    out[25] = ((reg[7] >> 26) | (reg[8] << 6)) & 0x3FF;
    out[26] = (reg[8] >> 4) & 0x3FF;
    out[27] = (reg[8] >> 14) & 0x3FF;
    out[28] = ((reg[8] >> 24) | (reg[9] << 8)) & 0x3FF;
    out[29] = (reg[9] >> 2) & 0x3FF;
    out[30] = (reg[9] >> 12) & 0x3FF;
    out[31] = (reg[9] >> 22) & 0x3FF;
}

__device__ __forceinline__ void unpack_12bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg[12];

    #pragma unroll
    for (int j = 0; j < 12; j++) {
        reg[j] = __ldg(&in[i + j * 32]);
    }

    out[0]  = (reg[0]) & 0xFFF;
    out[1]  = (reg[0] >> 12) & 0xFFF;
    out[2]  = ((reg[0] >> 24) | (reg[1] << 8)) & 0xFFF;
    out[3]  = (reg[1] >> 4) & 0xFFF;
    out[4]  = (reg[1] >> 16) & 0xFFF;
    out[5]  = ((reg[1] >> 28) | (reg[2] << 4)) & 0xFFF;
    out[6]  = (reg[2] >> 8) & 0xFFF;
    out[7]  = (reg[2] >> 20) & 0xFFF;
    out[8]  = (reg[3]) & 0xFFF;
    out[9]  = (reg[3] >> 12) & 0xFFF;
    out[10] = ((reg[3] >> 24) | (reg[4] << 8)) & 0xFFF;
    out[11] = (reg[4] >> 4) & 0xFFF;
    out[12] = (reg[4] >> 16) & 0xFFF;
    out[13] = ((reg[4] >> 28) | (reg[5] << 4)) & 0xFFF;
    out[14] = (reg[5] >> 8) & 0xFFF;
    out[15] = (reg[5] >> 20) & 0xFFF;
    out[16] = (reg[6]) & 0xFFF;
    out[17] = (reg[6] >> 12) & 0xFFF;
    out[18] = ((reg[6] >> 24) | (reg[7] << 8)) & 0xFFF;
    out[19] = (reg[7] >> 4) & 0xFFF;
    out[20] = (reg[7] >> 16) & 0xFFF;
    out[21] = ((reg[7] >> 28) | (reg[8] << 4)) & 0xFFF;
    out[22] = (reg[8] >> 8) & 0xFFF;
    out[23] = (reg[8] >> 20) & 0xFFF;
    out[24] = (reg[9]) & 0xFFF;
    out[25] = (reg[9] >> 12) & 0xFFF;
    out[26] = ((reg[9] >> 24) | (reg[10] << 8)) & 0xFFF;
    out[27] = (reg[10] >> 4) & 0xFFF;
    out[28] = (reg[10] >> 16) & 0xFFF;
    out[29] = ((reg[10] >> 28) | (reg[11] << 4)) & 0xFFF;
    out[30] = (reg[11] >> 8) & 0xFFF;
    out[31] = (reg[11] >> 20) & 0xFFF;
}

__device__ __forceinline__ void unpack_20bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg[20];

    #pragma unroll
    for (int j = 0; j < 20; j++) {
        reg[j] = __ldg(&in[i + j * 32]);
    }

    out[0]  = (reg[0]) & 0xFFFFF;
    out[1]  = ((reg[0] >> 20) | (reg[1] << 12)) & 0xFFFFF;
    out[2]  = (reg[1] >> 8) & 0xFFFFF;
    out[3]  = ((reg[1] >> 28) | (reg[2] << 4)) & 0xFFFFF;
    out[4]  = ((reg[2] >> 16) | (reg[3] << 16)) & 0xFFFFF;
    out[5]  = (reg[3] >> 4) & 0xFFFFF;
    out[6]  = ((reg[3] >> 24) | (reg[4] << 8)) & 0xFFFFF;
    out[7]  = (reg[4] >> 12) & 0xFFFFF;
    out[8]  = (reg[5]) & 0xFFFFF;
    out[9]  = ((reg[5] >> 20) | (reg[6] << 12)) & 0xFFFFF;
    out[10] = (reg[6] >> 8) & 0xFFFFF;
    out[11] = ((reg[6] >> 28) | (reg[7] << 4)) & 0xFFFFF;
    out[12] = ((reg[7] >> 16) | (reg[8] << 16)) & 0xFFFFF;
    out[13] = (reg[8] >> 4) & 0xFFFFF;
    out[14] = ((reg[8] >> 24) | (reg[9] << 8)) & 0xFFFFF;
    out[15] = (reg[9] >> 12) & 0xFFFFF;
    out[16] = (reg[10]) & 0xFFFFF;
    out[17] = ((reg[10] >> 20) | (reg[11] << 12)) & 0xFFFFF;
    out[18] = (reg[11] >> 8) & 0xFFFFF;
    out[19] = ((reg[11] >> 28) | (reg[12] << 4)) & 0xFFFFF;
    out[20] = ((reg[12] >> 16) | (reg[13] << 16)) & 0xFFFFF;
    out[21] = (reg[13] >> 4) & 0xFFFFF;
    out[22] = ((reg[13] >> 24) | (reg[14] << 8)) & 0xFFFFF;
    out[23] = (reg[14] >> 12) & 0xFFFFF;
    out[24] = (reg[15]) & 0xFFFFF;
    out[25] = ((reg[15] >> 20) | (reg[16] << 12)) & 0xFFFFF;
    out[26] = (reg[16] >> 8) & 0xFFFFF;
    out[27] = ((reg[16] >> 28) | (reg[17] << 4)) & 0xFFFFF;
    out[28] = ((reg[17] >> 16) | (reg[18] << 16)) & 0xFFFFF;
    out[29] = (reg[18] >> 4) & 0xFFFFF;
    out[30] = ((reg[18] >> 24) | (reg[19] << 8)) & 0xFFFFF;
    out[31] = (reg[19] >> 12) & 0xFFFFF;
}

__device__ __forceinline__ void unpack_24bw(const uint32_t* __restrict__ in, uint32_t* __restrict__ out) {
    int i = threadIdx.x;
    uint32_t reg[24];

    #pragma unroll
    for (int j = 0; j < 24; j++) {
        reg[j] = __ldg(&in[i + j * 32]);
    }

    out[0]  = (reg[0]) & 0xFFFFFF;
    out[1]  = ((reg[0] >> 24) | (reg[1] << 8)) & 0xFFFFFF;
    out[2]  = ((reg[1] >> 16) | (reg[2] << 16)) & 0xFFFFFF;
    out[3]  = (reg[2] >> 8) & 0xFFFFFF;
    out[4]  = (reg[3]) & 0xFFFFFF;
    out[5]  = ((reg[3] >> 24) | (reg[4] << 8)) & 0xFFFFFF;
    out[6]  = ((reg[4] >> 16) | (reg[5] << 16)) & 0xFFFFFF;
    out[7]  = (reg[5] >> 8) & 0xFFFFFF;
    out[8]  = (reg[6]) & 0xFFFFFF;
    out[9]  = ((reg[6] >> 24) | (reg[7] << 8)) & 0xFFFFFF;
    out[10] = ((reg[7] >> 16) | (reg[8] << 16)) & 0xFFFFFF;
    out[11] = (reg[8] >> 8) & 0xFFFFFF;
    out[12] = (reg[9]) & 0xFFFFFF;
    out[13] = ((reg[9] >> 24) | (reg[10] << 8)) & 0xFFFFFF;
    out[14] = ((reg[10] >> 16) | (reg[11] << 16)) & 0xFFFFFF;
    out[15] = (reg[11] >> 8) & 0xFFFFFF;
    out[16] = (reg[12]) & 0xFFFFFF;
    out[17] = ((reg[12] >> 24) | (reg[13] << 8)) & 0xFFFFFF;
    out[18] = ((reg[13] >> 16) | (reg[14] << 16)) & 0xFFFFFF;
    out[19] = (reg[14] >> 8) & 0xFFFFFF;
    out[20] = (reg[15]) & 0xFFFFFF;
    out[21] = ((reg[15] >> 24) | (reg[16] << 8)) & 0xFFFFFF;
    out[22] = ((reg[16] >> 16) | (reg[17] << 16)) & 0xFFFFFF;
    out[23] = (reg[17] >> 8) & 0xFFFFFF;
    out[24] = (reg[18]) & 0xFFFFFF;
    out[25] = ((reg[18] >> 24) | (reg[19] << 8)) & 0xFFFFFF;
    out[26] = ((reg[19] >> 16) | (reg[20] << 16)) & 0xFFFFFF;
    out[27] = (reg[20] >> 8) & 0xFFFFFF;
    out[28] = (reg[21]) & 0xFFFFFF;
    out[29] = ((reg[21] >> 24) | (reg[22] << 8)) & 0xFFFFFF;
    out[30] = ((reg[22] >> 16) | (reg[23] << 16)) & 0xFFFFFF;
    out[31] = (reg[23] >> 8) & 0xFFFFFF;
}

// ============================================================================
// Runtime dispatch function
// ============================================================================

__device__ __forceinline__ void unpack(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint8_t bw) {
    switch (bw) {
        case 0:  unpack_0bw(in, out); break;
        case 1:  unpack_1bw(in, out); break;
        case 2:  unpack_2bw(in, out); break;
        case 3:  unpack_3bw(in, out); break;
        case 4:  unpack_4bw(in, out); break;
        case 5:  unpack_5bw(in, out); break;
        case 6:  unpack_6bw(in, out); break;
        case 7:  unpack_7bw(in, out); break;
        case 8:  unpack_8bw(in, out); break;
        case 10: unpack_10bw(in, out); break;
        case 12: unpack_12bw(in, out); break;
        case 16: unpack_16bw(in, out); break;
        case 20: unpack_20bw(in, out); break;
        case 24: unpack_24bw(in, out); break;
        case 32: unpack_32bw(in, out); break;
        default: unpack_32bw(in, out); break;  // Fallback to 32-bit
    }
}

// ============================================================================
// FOR (Frame of Reference) decoding with unpack
// value = base + delta
// ============================================================================

__device__ __forceinline__ void unpack_for(
    const uint32_t* __restrict__ in,
    int32_t* __restrict__ out,
    int32_t base,
    uint8_t bw)
{
    uint32_t deltas[32];
    unpack(in, deltas, bw);

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j] = base + static_cast<int32_t>(deltas[j]);
    }
}

// ============================================================================
// Block Load with FOR decoding (Crystal-style interface)
// Each thread loads 32 values, stores at strided offsets
// ============================================================================

template<int BLOCK_THREADS>
__device__ __forceinline__ void BlockLoadFOR(
    const uint32_t* __restrict__ tile_data,
    int32_t* __restrict__ items,
    int32_t base,
    uint8_t bw)
{
    static_assert(BLOCK_THREADS == 32, "FLS requires 32 threads per block");

    int32_t local_items[32];
    unpack_for(tile_data, local_items, base, bw);

    // Store to output (thread i stores at positions [i, i+32, i+64, ...])
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        items[j] = local_items[j];
    }
}

// ============================================================================
// Store registers to global memory with strided pattern
// Thread i stores at positions [i, i+32, i+64, ..., i+31*32]
// ============================================================================

template<int BLOCK_THREADS>
__device__ __forceinline__ void StoreStrided(
    const int32_t* __restrict__ registers,
    int32_t* __restrict__ out)
{
    int tid = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[j * BLOCK_THREADS + tid] = registers[j];
    }
}

// ============================================================================
// Combined unpack + store for full tile decompression
// ============================================================================

__device__ __forceinline__ void UnpackTile(
    const uint32_t* __restrict__ in,
    int32_t* __restrict__ out,
    int32_t base,
    uint8_t bw)
{
    int32_t registers[32];
    unpack_for(in, registers, base, bw);
    StoreStrided<32>(registers, out);
}

}  // namespace l3_fls
