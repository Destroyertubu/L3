/**
 * FastLanes 64-bit Unpack Functions
 *
 * Generated 64-bit extension for FastLanesGPU.
 * Each function unpacks X-bit packed data into 64-bit output values.
 *
 * Structure:
 * - 32 threads per block
 * - Each thread extracts 32 values
 * - Total: 1024 64-bit values per block
 *
 * Naming: unpack_Xbw_64ow_32crw_1uf
 *   - Xbw: X bits per value
 *   - 64ow: 64-bit output width
 *   - 32crw: 32 consecutive reads per warp
 *   - 1uf: 1 unroll factor
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace generated { namespace unpack::cuda { namespace normal_64 {

// ============================================================================
// Bit width 0: All zeros
// ============================================================================
inline __device__ void unpack_0bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    [[maybe_unused]] auto out = reinterpret_cast<uint64_t*>(a_out_p);
    [[maybe_unused]] uint64_t base_0 = 0ULL;

    int i = threadIdx.x; // THREAD INDEX

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[(i * 1) + (0 * 32) + j * 32] = base_0;
    }
}

// ============================================================================
// Generic unpack function for bit widths 1-63
// Handles non-word-aligned thread starts correctly
// ============================================================================
template<int BIT_WIDTH>
inline __device__ void unpack_generic_64(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    static_assert(BIT_WIDTH >= 1 && BIT_WIDTH <= 63, "BIT_WIDTH must be 1-63");

    [[maybe_unused]] auto out = reinterpret_cast<uint64_t*>(a_out_p);
    [[maybe_unused]] auto in = reinterpret_cast<const uint64_t*>(a_in_p);

    constexpr uint64_t MASK = (1ULL << BIT_WIDTH) - 1;
    constexpr int BITS_PER_WORD = 64;
    constexpr int VALUES_PER_THREAD = 32;

    // Total bits needed per thread
    constexpr int TOTAL_BITS = BIT_WIDTH * VALUES_PER_THREAD;
    // Number of 64-bit words needed per thread (rounded up, +1 for cross-boundary)
    constexpr int WORDS_NEEDED = (TOTAL_BITS + BITS_PER_WORD - 1) / BITS_PER_WORD;

    int i = threadIdx.x; // THREAD INDEX

    // Calculate starting word for this thread
    int thread_start_bit = i * BIT_WIDTH * VALUES_PER_THREAD;
    int thread_start_word = thread_start_bit / BITS_PER_WORD;

    // Load input words for this thread
    uint64_t regs[WORDS_NEEDED + 1]; // +1 for cross-boundary access
    #pragma unroll
    for (int w = 0; w <= WORDS_NEEDED; w++) {
        regs[w] = in[thread_start_word + w];
    }

    // Extract each value
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
        // Calculate global bit position and convert to local indices
        int global_bit_pos = thread_start_bit + j * BIT_WIDTH;
        int global_word = global_bit_pos / BITS_PER_WORD;
        int word_idx = global_word - thread_start_word;
        int bit_offset = global_bit_pos % BITS_PER_WORD;

        uint64_t val;
        if (bit_offset + BIT_WIDTH <= BITS_PER_WORD) {
            // No cross-boundary
            val = (regs[word_idx] >> bit_offset) & MASK;
        } else {
            // Cross-boundary: combine bits from two words
            int bits_from_first = BITS_PER_WORD - bit_offset;
            val = (regs[word_idx] >> bit_offset) | ((regs[word_idx + 1] << bits_from_first) & MASK);
            val &= MASK;
        }
        out[i + j * 32] = val;
    }
}

// ============================================================================
// Specialized unpack functions for common bit widths (optimized)
// ============================================================================

// 1-bit: 32 values fit in 32 bits, so each thread uses part of one 64-bit word
inline __device__ void unpack_1bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint64_t reg = in[i / 2];  // Two threads share one 64-bit word
    int shift_base = (i & 1) * 32;  // 0 or 32

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[i + j * 32] = (reg >> (shift_base + j)) & 1ULL;
    }
}

// 2-bit: 64 bits per thread = 1 word
inline __device__ void unpack_2bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint64_t reg = in[i];

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[i + j * 32] = (reg >> (j * 2)) & 3ULL;
    }
}

// 4-bit: 128 bits per thread = 2 words
inline __device__ void unpack_4bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint64_t reg0 = in[i * 2];
    uint64_t reg1 = in[i * 2 + 1];

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        out[i + j * 32] = (reg0 >> (j * 4)) & 0xFULL;
    }
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        out[i + j * 32] = (reg1 >> ((j - 16) * 4)) & 0xFULL;
    }
}

// 8-bit: 256 bits per thread = 4 words
inline __device__ void unpack_8bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 4; w++) {
        uint64_t reg = in[i * 4 + w];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            out[i + (w * 8 + j) * 32] = (reg >> (j * 8)) & 0xFFULL;
        }
    }
}

// 16-bit: 512 bits per thread = 8 words
inline __device__ void unpack_16bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint64_t reg = in[i * 8 + w];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            out[i + (w * 4 + j) * 32] = (reg >> (j * 16)) & 0xFFFFULL;
        }
    }
}

// 32-bit: 1024 bits per thread = 16 words
inline __device__ void unpack_32bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 16; w++) {
        uint64_t reg = in[i * 16 + w];
        out[i + (w * 2) * 32] = reg & 0xFFFFFFFFULL;
        out[i + (w * 2 + 1) * 32] = reg >> 32;
    }
}

// 64-bit: Direct copy, 32 words per thread
inline __device__ void unpack_64bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[i + j * 32] = in[i * 32 + j];
    }
}

// ============================================================================
// Dispatcher function - selects appropriate unpack function based on bit width
// ============================================================================
inline __device__ void unpack(const uint64_t* a_in_p, uint64_t* a_out_p, uint8_t bw) {
    switch (bw) {
    case 0:  unpack_0bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 1:  unpack_1bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 2:  unpack_2bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 3:  unpack_generic_64<3>(a_in_p, a_out_p); break;
    case 4:  unpack_4bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 5:  unpack_generic_64<5>(a_in_p, a_out_p); break;
    case 6:  unpack_generic_64<6>(a_in_p, a_out_p); break;
    case 7:  unpack_generic_64<7>(a_in_p, a_out_p); break;
    case 8:  unpack_8bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 9:  unpack_generic_64<9>(a_in_p, a_out_p); break;
    case 10: unpack_generic_64<10>(a_in_p, a_out_p); break;
    case 11: unpack_generic_64<11>(a_in_p, a_out_p); break;
    case 12: unpack_generic_64<12>(a_in_p, a_out_p); break;
    case 13: unpack_generic_64<13>(a_in_p, a_out_p); break;
    case 14: unpack_generic_64<14>(a_in_p, a_out_p); break;
    case 15: unpack_generic_64<15>(a_in_p, a_out_p); break;
    case 16: unpack_16bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 17: unpack_generic_64<17>(a_in_p, a_out_p); break;
    case 18: unpack_generic_64<18>(a_in_p, a_out_p); break;
    case 19: unpack_generic_64<19>(a_in_p, a_out_p); break;
    case 20: unpack_generic_64<20>(a_in_p, a_out_p); break;
    case 21: unpack_generic_64<21>(a_in_p, a_out_p); break;
    case 22: unpack_generic_64<22>(a_in_p, a_out_p); break;
    case 23: unpack_generic_64<23>(a_in_p, a_out_p); break;
    case 24: unpack_generic_64<24>(a_in_p, a_out_p); break;
    case 25: unpack_generic_64<25>(a_in_p, a_out_p); break;
    case 26: unpack_generic_64<26>(a_in_p, a_out_p); break;
    case 27: unpack_generic_64<27>(a_in_p, a_out_p); break;
    case 28: unpack_generic_64<28>(a_in_p, a_out_p); break;
    case 29: unpack_generic_64<29>(a_in_p, a_out_p); break;
    case 30: unpack_generic_64<30>(a_in_p, a_out_p); break;
    case 31: unpack_generic_64<31>(a_in_p, a_out_p); break;
    case 32: unpack_32bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 33: unpack_generic_64<33>(a_in_p, a_out_p); break;
    case 34: unpack_generic_64<34>(a_in_p, a_out_p); break;
    case 35: unpack_generic_64<35>(a_in_p, a_out_p); break;
    case 36: unpack_generic_64<36>(a_in_p, a_out_p); break;
    case 37: unpack_generic_64<37>(a_in_p, a_out_p); break;
    case 38: unpack_generic_64<38>(a_in_p, a_out_p); break;
    case 39: unpack_generic_64<39>(a_in_p, a_out_p); break;
    case 40: unpack_generic_64<40>(a_in_p, a_out_p); break;
    case 41: unpack_generic_64<41>(a_in_p, a_out_p); break;
    case 42: unpack_generic_64<42>(a_in_p, a_out_p); break;
    case 43: unpack_generic_64<43>(a_in_p, a_out_p); break;
    case 44: unpack_generic_64<44>(a_in_p, a_out_p); break;
    case 45: unpack_generic_64<45>(a_in_p, a_out_p); break;
    case 46: unpack_generic_64<46>(a_in_p, a_out_p); break;
    case 47: unpack_generic_64<47>(a_in_p, a_out_p); break;
    case 48: unpack_generic_64<48>(a_in_p, a_out_p); break;
    case 49: unpack_generic_64<49>(a_in_p, a_out_p); break;
    case 50: unpack_generic_64<50>(a_in_p, a_out_p); break;
    case 51: unpack_generic_64<51>(a_in_p, a_out_p); break;
    case 52: unpack_generic_64<52>(a_in_p, a_out_p); break;
    case 53: unpack_generic_64<53>(a_in_p, a_out_p); break;
    case 54: unpack_generic_64<54>(a_in_p, a_out_p); break;
    case 55: unpack_generic_64<55>(a_in_p, a_out_p); break;
    case 56: unpack_generic_64<56>(a_in_p, a_out_p); break;
    case 57: unpack_generic_64<57>(a_in_p, a_out_p); break;
    case 58: unpack_generic_64<58>(a_in_p, a_out_p); break;
    case 59: unpack_generic_64<59>(a_in_p, a_out_p); break;
    case 60: unpack_generic_64<60>(a_in_p, a_out_p); break;
    case 61: unpack_generic_64<61>(a_in_p, a_out_p); break;
    case 62: unpack_generic_64<62>(a_in_p, a_out_p); break;
    case 63: unpack_generic_64<63>(a_in_p, a_out_p); break;
    case 64: unpack_64bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    }
}

}}}; // namespace generated::unpack::cuda::normal_64

// ============================================================================
// Global kernel wrapper
// ============================================================================
__global__ void unpack_global_64(const uint64_t* __restrict in, uint64_t* __restrict out, uint8_t bw) {
    int blc_idx = blockIdx.x;
    // Input offset: each block processes 1024 values packed at bw bits each
    // Total input bits = 1024 * bw, in 64-bit words = 1024 * bw / 64 = 16 * bw
    in = in + (blc_idx * bw * 16);
    // Output: 1024 64-bit values per block
    out = out + (blc_idx * 1024);

    generated::unpack::cuda::normal_64::unpack(in, out, bw);
}

// ============================================================================
// Device-callable wrapper
// ============================================================================
__device__ __forceinline__ void unpack_device_64(const uint64_t* __restrict in, uint64_t* __restrict out, uint8_t bw) {
    generated::unpack::cuda::normal_64::unpack(in, out, bw);
}

__device__ __forceinline__ void unpack_device_64(const int64_t* __restrict in, int64_t* __restrict out, uint8_t bw) {
    generated::unpack::cuda::normal_64::unpack(
        reinterpret_cast<const uint64_t*>(in), reinterpret_cast<uint64_t*>(out), bw);
}
