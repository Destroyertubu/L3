/**
 * FastLanes 64-bit Pack Functions
 *
 * GPU-based packing (encoding) functions for 64-bit values.
 * Inverse of unpack_64.cuh - compresses data on GPU.
 *
 * Structure:
 * - 32 threads per block
 * - Each thread packs 32 values
 * - Total: 1024 64-bit values per block
 *
 * Naming: pack_Xbw_64ow_32crw_1uf
 *   - Xbw: X bits per value (packed)
 *   - 64ow: 64-bit output width (input is 64-bit)
 *   - 32crw: 32 consecutive reads/writes per warp
 *   - 1uf: 1 unroll factor
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace generated { namespace pack { namespace cuda { namespace normal_64 {

// ============================================================================
// Bit width 0: No data stored (all values are the same as reference)
// ============================================================================
inline __device__ void pack_0bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    // Nothing to pack - all deltas are 0
}

// ============================================================================
// Generic pack function for bit widths 1-63
// Handles overlapping output regions with atomic operations
// ============================================================================
template<int BIT_WIDTH>
inline __device__ void pack_generic_64(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    static_assert(BIT_WIDTH >= 1 && BIT_WIDTH <= 63, "BIT_WIDTH must be 1-63");

    [[maybe_unused]] auto out = reinterpret_cast<uint64_t*>(a_out_p);
    [[maybe_unused]] auto in = reinterpret_cast<const uint64_t*>(a_in_p);

    constexpr uint64_t MASK = (1ULL << BIT_WIDTH) - 1;
    constexpr int BITS_PER_WORD = 64;
    constexpr int VALUES_PER_THREAD = 32;

    int i = threadIdx.x;  // THREAD INDEX

    // Calculate this thread's starting bit position in the output
    int thread_start_bit = i * BIT_WIDTH * VALUES_PER_THREAD;

    // Pack each value and write directly with atomic OR
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
        // Read value from striped layout
        uint64_t val = in[i + j * 32] & MASK;

        // Calculate global bit position for this value
        int global_bit_pos = thread_start_bit + j * BIT_WIDTH;
        int word_idx = global_bit_pos / BITS_PER_WORD;
        int bit_offset = global_bit_pos % BITS_PER_WORD;

        if (bit_offset + BIT_WIDTH <= BITS_PER_WORD) {
            // Fits in single word
            atomicOr((unsigned long long*)&out[word_idx], (unsigned long long)(val << bit_offset));
        } else {
            // Crosses word boundary
            atomicOr((unsigned long long*)&out[word_idx], (unsigned long long)(val << bit_offset));
            atomicOr((unsigned long long*)&out[word_idx + 1], (unsigned long long)(val >> (BITS_PER_WORD - bit_offset)));
        }
    }
}

// ============================================================================
// Specialized pack functions for common bit widths (optimized)
// ============================================================================

// 1-bit: 32 values fit in 32 bits, so 2 threads share one 64-bit output word
inline __device__ void pack_1bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint32_t packed32 = 0;

    // Each thread packs 32 1-bit values into 32 bits
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        uint32_t val = in[i + j * 32] & 1ULL;
        packed32 |= val << j;
    }

    // Use warp shuffle to combine pairs of threads' results
    // Thread 0 gets from thread 1, thread 2 gets from thread 3, etc.
    uint32_t partner_packed = __shfl_xor_sync(0xFFFFFFFF, packed32, 1);

    // Even threads write the combined 64-bit word
    if ((i & 1) == 0) {
        uint64_t combined = (uint64_t)packed32 | ((uint64_t)partner_packed << 32);
        out[i / 2] = combined;
    }
}

// 2-bit: 64 bits per thread = 1 word
inline __device__ void pack_2bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint64_t packed = 0ULL;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        uint64_t val = in[i + j * 32] & 3ULL;
        packed |= val << (j * 2);
    }

    out[i] = packed;
}

// 4-bit: 128 bits per thread = 2 words
inline __device__ void pack_4bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;
    uint64_t packed0 = 0ULL, packed1 = 0ULL;

    #pragma unroll
    for (int j = 0; j < 16; j++) {
        uint64_t val = in[i + j * 32] & 0xFULL;
        packed0 |= val << (j * 4);
    }
    #pragma unroll
    for (int j = 16; j < 32; j++) {
        uint64_t val = in[i + j * 32] & 0xFULL;
        packed1 |= val << ((j - 16) * 4);
    }

    out[i * 2] = packed0;
    out[i * 2 + 1] = packed1;
}

// 8-bit: 256 bits per thread = 4 words
inline __device__ void pack_8bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 4; w++) {
        uint64_t packed = 0ULL;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            uint64_t val = in[i + (w * 8 + j) * 32] & 0xFFULL;
            packed |= val << (j * 8);
        }
        out[i * 4 + w] = packed;
    }
}

// 16-bit: 512 bits per thread = 8 words
inline __device__ void pack_16bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 8; w++) {
        uint64_t packed = 0ULL;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t val = in[i + (w * 4 + j) * 32] & 0xFFFFULL;
            packed |= val << (j * 16);
        }
        out[i * 8 + w] = packed;
    }
}

// 32-bit: 1024 bits per thread = 16 words
inline __device__ void pack_32bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int w = 0; w < 16; w++) {
        uint64_t lo = in[i + (w * 2) * 32] & 0xFFFFFFFFULL;
        uint64_t hi = in[i + (w * 2 + 1) * 32] & 0xFFFFFFFFULL;
        out[i * 16 + w] = lo | (hi << 32);
    }
}

// 64-bit: Direct copy, 32 words per thread
inline __device__ void pack_64bw_64ow_32crw_1uf(const uint64_t* __restrict a_in_p, uint64_t* __restrict a_out_p) {
    auto out = a_out_p;
    auto in = a_in_p;

    int i = threadIdx.x;

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        out[i * 32 + j] = in[i + j * 32];
    }
}

// ============================================================================
// Dispatcher function - selects appropriate pack function based on bit width
// ============================================================================
inline __device__ void pack(const uint64_t* a_in_p, uint64_t* a_out_p, uint8_t bw) {
    switch (bw) {
    case 0:  pack_0bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 1:  pack_1bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 2:  pack_2bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 3:  pack_generic_64<3>(a_in_p, a_out_p); break;
    case 4:  pack_4bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 5:  pack_generic_64<5>(a_in_p, a_out_p); break;
    case 6:  pack_generic_64<6>(a_in_p, a_out_p); break;
    case 7:  pack_generic_64<7>(a_in_p, a_out_p); break;
    case 8:  pack_8bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 9:  pack_generic_64<9>(a_in_p, a_out_p); break;
    case 10: pack_generic_64<10>(a_in_p, a_out_p); break;
    case 11: pack_generic_64<11>(a_in_p, a_out_p); break;
    case 12: pack_generic_64<12>(a_in_p, a_out_p); break;
    case 13: pack_generic_64<13>(a_in_p, a_out_p); break;
    case 14: pack_generic_64<14>(a_in_p, a_out_p); break;
    case 15: pack_generic_64<15>(a_in_p, a_out_p); break;
    case 16: pack_16bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 17: pack_generic_64<17>(a_in_p, a_out_p); break;
    case 18: pack_generic_64<18>(a_in_p, a_out_p); break;
    case 19: pack_generic_64<19>(a_in_p, a_out_p); break;
    case 20: pack_generic_64<20>(a_in_p, a_out_p); break;
    case 21: pack_generic_64<21>(a_in_p, a_out_p); break;
    case 22: pack_generic_64<22>(a_in_p, a_out_p); break;
    case 23: pack_generic_64<23>(a_in_p, a_out_p); break;
    case 24: pack_generic_64<24>(a_in_p, a_out_p); break;
    case 25: pack_generic_64<25>(a_in_p, a_out_p); break;
    case 26: pack_generic_64<26>(a_in_p, a_out_p); break;
    case 27: pack_generic_64<27>(a_in_p, a_out_p); break;
    case 28: pack_generic_64<28>(a_in_p, a_out_p); break;
    case 29: pack_generic_64<29>(a_in_p, a_out_p); break;
    case 30: pack_generic_64<30>(a_in_p, a_out_p); break;
    case 31: pack_generic_64<31>(a_in_p, a_out_p); break;
    case 32: pack_32bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    case 33: pack_generic_64<33>(a_in_p, a_out_p); break;
    case 34: pack_generic_64<34>(a_in_p, a_out_p); break;
    case 35: pack_generic_64<35>(a_in_p, a_out_p); break;
    case 36: pack_generic_64<36>(a_in_p, a_out_p); break;
    case 37: pack_generic_64<37>(a_in_p, a_out_p); break;
    case 38: pack_generic_64<38>(a_in_p, a_out_p); break;
    case 39: pack_generic_64<39>(a_in_p, a_out_p); break;
    case 40: pack_generic_64<40>(a_in_p, a_out_p); break;
    case 41: pack_generic_64<41>(a_in_p, a_out_p); break;
    case 42: pack_generic_64<42>(a_in_p, a_out_p); break;
    case 43: pack_generic_64<43>(a_in_p, a_out_p); break;
    case 44: pack_generic_64<44>(a_in_p, a_out_p); break;
    case 45: pack_generic_64<45>(a_in_p, a_out_p); break;
    case 46: pack_generic_64<46>(a_in_p, a_out_p); break;
    case 47: pack_generic_64<47>(a_in_p, a_out_p); break;
    case 48: pack_generic_64<48>(a_in_p, a_out_p); break;
    case 49: pack_generic_64<49>(a_in_p, a_out_p); break;
    case 50: pack_generic_64<50>(a_in_p, a_out_p); break;
    case 51: pack_generic_64<51>(a_in_p, a_out_p); break;
    case 52: pack_generic_64<52>(a_in_p, a_out_p); break;
    case 53: pack_generic_64<53>(a_in_p, a_out_p); break;
    case 54: pack_generic_64<54>(a_in_p, a_out_p); break;
    case 55: pack_generic_64<55>(a_in_p, a_out_p); break;
    case 56: pack_generic_64<56>(a_in_p, a_out_p); break;
    case 57: pack_generic_64<57>(a_in_p, a_out_p); break;
    case 58: pack_generic_64<58>(a_in_p, a_out_p); break;
    case 59: pack_generic_64<59>(a_in_p, a_out_p); break;
    case 60: pack_generic_64<60>(a_in_p, a_out_p); break;
    case 61: pack_generic_64<61>(a_in_p, a_out_p); break;
    case 62: pack_generic_64<62>(a_in_p, a_out_p); break;
    case 63: pack_generic_64<63>(a_in_p, a_out_p); break;
    case 64: pack_64bw_64ow_32crw_1uf(a_in_p, a_out_p); break;
    }
}

}}}}; // namespace generated::pack::cuda::normal_64

// ============================================================================
// Global kernel wrapper
// ============================================================================
__global__ void pack_global_64(const uint64_t* __restrict in, uint64_t* __restrict out, uint8_t bw) {
    int blc_idx = blockIdx.x;
    // Input: 1024 64-bit values per block
    in = in + (blc_idx * 1024);
    // Output offset: each block writes 1024 * bw bits = 16 * bw 64-bit words
    out = out + (blc_idx * bw * 16);

    generated::pack::cuda::normal_64::pack(in, out, bw);
}

// ============================================================================
// Device-callable wrapper
// ============================================================================
__device__ __forceinline__ void pack_device_64(const uint64_t* __restrict in, uint64_t* __restrict out, uint8_t bw) {
    generated::pack::cuda::normal_64::pack(in, out, bw);
}

__device__ __forceinline__ void pack_device_64(const int64_t* __restrict in, int64_t* __restrict out, uint8_t bw) {
    generated::pack::cuda::normal_64::pack(
        reinterpret_cast<const uint64_t*>(in), reinterpret_cast<uint64_t*>(out), bw);
}

// ============================================================================
// Compute required bits for a value
// ============================================================================
__device__ __forceinline__ uint8_t required_bits_64(uint64_t value) {
    if (value == 0) return 0;
    return 64 - __clzll(value);
}

// ============================================================================
// Compute max required bits across a block (reduction)
// ============================================================================
template<int BLOCK_THREADS>
__device__ __forceinline__ uint8_t compute_block_bitwidth_64(uint64_t value) {
    __shared__ uint8_t shared_max[BLOCK_THREADS];

    int tid = threadIdx.x;
    shared_max[tid] = required_bits_64(value);
    __syncthreads();

    // Parallel reduction to find max
    for (int stride = BLOCK_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
            }
        }
        __syncthreads();
    }

    return shared_max[0];
}

// ============================================================================
// Utility: compute output size in bytes
// ============================================================================
__host__ __device__ __forceinline__ size_t packed_size_64(int num_values, uint8_t bw) {
    // Total bits needed
    size_t total_bits = (size_t)num_values * bw;
    // Round up to 64-bit words
    size_t num_words = (total_bits + 63) / 64;
    return num_words * sizeof(uint64_t);
}

