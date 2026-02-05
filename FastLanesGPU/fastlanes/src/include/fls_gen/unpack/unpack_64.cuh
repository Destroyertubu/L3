// unpack_64.cuh - GPU bit-unpacking for uint64
// Generated for FastLanes uint64 support
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

namespace generated { namespace unpack::cuda { namespace normal_64 {

// For uint64 FastLanes:
// - 1024 elements per block
// - 16 lanes, 64 elements per lane
// - Elements are packed consecutively in bit stream
// - in[word_idx * 16 + lane] contains bits for lane
// - Element j in lane i: starts at bit position j*BW in the lane's bit stream

// Generic unpack template for any bitwidth
// Matches FastLanes uint64 pack order (consecutive element packing)
template<int BW>
inline __device__ void unpack_generic(const uint64_t* __restrict in, uint64_t* __restrict out) {
    int tid = threadIdx.x;
    constexpr uint64_t mask = (BW >= 64) ? ~0ULL : ((1ULL << BW) - 1);

    // 32 threads handle 1024 elements: each thread handles 32 elements
    // Thread tid handles elements: tid, tid+32, tid+64, ..., tid+992

    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int out_idx = tid + k * 32;  // Output position in the 1024-element block

        // FastLanes layout: 16 lanes, 64 elements per lane
        // out_idx = lane + 16 * elem_in_lane
        int lane = out_idx % 16;           // Which lane (0-15)
        int elem_in_lane = out_idx / 16;   // Position within lane (0-63)

        uint64_t val = 0;

        if constexpr (BW == 0) {
            val = 0;
        } else if constexpr (BW == 64) {
            // Each element takes exactly one uint64
            val = in[elem_in_lane * 16 + lane];
        } else {
            // Calculate bit position in the lane's bit stream
            int bit_offset = elem_in_lane * BW;
            int word_idx = bit_offset / 64;
            int bit_in_word = bit_offset % 64;

            // Read from the appropriate word
            uint64_t w0 = in[word_idx * 16 + lane];
            val = (w0 >> bit_in_word) & mask;

            // Check if element spans two words
            if (bit_in_word + BW > 64) {
                // Need to read from next word
                uint64_t w1 = in[(word_idx + 1) * 16 + lane];
                int bits_from_w0 = 64 - bit_in_word;
                int bits_from_w1 = BW - bits_from_w0;
                val = (w0 >> bit_in_word) | ((w1 & ((1ULL << bits_from_w1) - 1)) << bits_from_w0);
            }
        }
        out[out_idx] = val;
    }
}

// Dispatch function
inline __device__ void unpack(const uint64_t* __restrict in, uint64_t* __restrict out, uint8_t bw) {
    switch (bw) {
        case 0: unpack_generic<0>(in, out); break;
        case 1: unpack_generic<1>(in, out); break;
        case 2: unpack_generic<2>(in, out); break;
        case 3: unpack_generic<3>(in, out); break;
        case 4: unpack_generic<4>(in, out); break;
        case 5: unpack_generic<5>(in, out); break;
        case 6: unpack_generic<6>(in, out); break;
        case 7: unpack_generic<7>(in, out); break;
        case 8: unpack_generic<8>(in, out); break;
        case 9: unpack_generic<9>(in, out); break;
        case 10: unpack_generic<10>(in, out); break;
        case 11: unpack_generic<11>(in, out); break;
        case 12: unpack_generic<12>(in, out); break;
        case 13: unpack_generic<13>(in, out); break;
        case 14: unpack_generic<14>(in, out); break;
        case 15: unpack_generic<15>(in, out); break;
        case 16: unpack_generic<16>(in, out); break;
        case 17: unpack_generic<17>(in, out); break;
        case 18: unpack_generic<18>(in, out); break;
        case 19: unpack_generic<19>(in, out); break;
        case 20: unpack_generic<20>(in, out); break;
        case 21: unpack_generic<21>(in, out); break;
        case 22: unpack_generic<22>(in, out); break;
        case 23: unpack_generic<23>(in, out); break;
        case 24: unpack_generic<24>(in, out); break;
        case 25: unpack_generic<25>(in, out); break;
        case 26: unpack_generic<26>(in, out); break;
        case 27: unpack_generic<27>(in, out); break;
        case 28: unpack_generic<28>(in, out); break;
        case 29: unpack_generic<29>(in, out); break;
        case 30: unpack_generic<30>(in, out); break;
        case 31: unpack_generic<31>(in, out); break;
        case 32: unpack_generic<32>(in, out); break;
        case 33: unpack_generic<33>(in, out); break;
        case 34: unpack_generic<34>(in, out); break;
        case 35: unpack_generic<35>(in, out); break;
        case 36: unpack_generic<36>(in, out); break;
        case 37: unpack_generic<37>(in, out); break;
        case 38: unpack_generic<38>(in, out); break;
        case 39: unpack_generic<39>(in, out); break;
        case 40: unpack_generic<40>(in, out); break;
        case 41: unpack_generic<41>(in, out); break;
        case 42: unpack_generic<42>(in, out); break;
        case 43: unpack_generic<43>(in, out); break;
        case 44: unpack_generic<44>(in, out); break;
        case 45: unpack_generic<45>(in, out); break;
        case 46: unpack_generic<46>(in, out); break;
        case 47: unpack_generic<47>(in, out); break;
        case 48: unpack_generic<48>(in, out); break;
        case 49: unpack_generic<49>(in, out); break;
        case 50: unpack_generic<50>(in, out); break;
        case 51: unpack_generic<51>(in, out); break;
        case 52: unpack_generic<52>(in, out); break;
        case 53: unpack_generic<53>(in, out); break;
        case 54: unpack_generic<54>(in, out); break;
        case 55: unpack_generic<55>(in, out); break;
        case 56: unpack_generic<56>(in, out); break;
        case 57: unpack_generic<57>(in, out); break;
        case 58: unpack_generic<58>(in, out); break;
        case 59: unpack_generic<59>(in, out); break;
        case 60: unpack_generic<60>(in, out); break;
        case 61: unpack_generic<61>(in, out); break;
        case 62: unpack_generic<62>(in, out); break;
        case 63: unpack_generic<63>(in, out); break;
        case 64: unpack_generic<64>(in, out); break;
        default: break;
    }
}

}}} // namespace generated::unpack::cuda::normal_64

// Device function for inline use
__device__ __forceinline__ void unpack_device_64(const uint64_t* __restrict in,
                                                   uint64_t* __restrict out,
                                                   uint8_t bw) {
    generated::unpack::cuda::normal_64::unpack(in, out, bw);
}

// Global kernel for standalone use
// Input: packed data, bw * 16 uint64 values per block
// Output: 1024 uint64 values per block
__global__ void unpack_global_64(const uint64_t* __restrict in,
                                  uint64_t* __restrict out,
                                  uint8_t bw) {
    int blc_idx = blockIdx.x;
    // Input offset: each block consumes bw * 16 uint64 values
    in  = in + (blc_idx * bw * 16);
    // Output offset: each block produces 1024 uint64 values
    out = out + (blc_idx * 1024);
    generated::unpack::cuda::normal_64::unpack(in, out, bw);
}
