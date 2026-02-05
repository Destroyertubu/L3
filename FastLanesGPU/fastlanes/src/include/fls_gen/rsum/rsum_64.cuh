// rsum_64.cuh - GPU prefix sum for uint64
// Generated for FastLanes uint64 support
// This is the inverse of unrsum - converts delta values back to original
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// uint64 rsum: inverse of unrsum pattern
// unrsum does: delta = in[next] - in[curr], and sets out[first_pos] = 0
// rsum does: out[first_pos] = base, out[next_pos] = prev + in[next_pos]
//
// For uint64, there are 16 chains (i=0..15), each with 64 elements
// Chain i visits positions in this order (8 groups of 8):
//   Group 0: i, i+128, i+256, i+384, i+512, i+640, i+768, i+896
//   Group 1: i+64, i+64+128, ..., i+64+896
//   Group 2: i+32, i+32+128, ..., i+32+896
//   Group 3: i+96, i+96+128, ..., i+96+896
//   Group 4: i+16, i+16+128, ..., i+16+896
//   Group 5: i+80, i+80+128, ..., i+80+896
//   Group 6: i+48, i+48+128, ..., i+48+896
//   Group 7: i+112, i+112+128, ..., i+112+896
//
// Only threads 0-15 do work (16 chains total)
// Base array has 16 elements (one per chain)

__device__ __forceinline__ void d_rsum_64(const uint64_t* in, uint64_t* out, const uint64_t* base) {
    uint32_t trd_idx = threadIdx.x;
    trd_idx = trd_idx % 32;

    // Only first 16 threads do work
    if (trd_idx >= 16) return;

    uint64_t acc;

    // Thread trd_idx handles chain trd_idx (64 elements total in 8 groups)
    // Base value is the first element of the chain
    acc = base[trd_idx];

    // Group 0: positions trd_idx + k*128 for k=0..7
    // First position gets the base value (delta at this pos is 0)
    out[trd_idx + 0] = acc;
    acc = acc + in[trd_idx + 128];
    out[trd_idx + 128] = acc;
    acc = acc + in[trd_idx + 256];
    out[trd_idx + 256] = acc;
    acc = acc + in[trd_idx + 384];
    out[trd_idx + 384] = acc;
    acc = acc + in[trd_idx + 512];
    out[trd_idx + 512] = acc;
    acc = acc + in[trd_idx + 640];
    out[trd_idx + 640] = acc;
    acc = acc + in[trd_idx + 768];
    out[trd_idx + 768] = acc;
    acc = acc + in[trd_idx + 896];
    out[trd_idx + 896] = acc;

    // Group 1: positions trd_idx + 64 + k*128 for k=0..7
    acc = acc + in[trd_idx + 64];
    out[trd_idx + 64] = acc;
    acc = acc + in[trd_idx + 192];
    out[trd_idx + 192] = acc;
    acc = acc + in[trd_idx + 320];
    out[trd_idx + 320] = acc;
    acc = acc + in[trd_idx + 448];
    out[trd_idx + 448] = acc;
    acc = acc + in[trd_idx + 576];
    out[trd_idx + 576] = acc;
    acc = acc + in[trd_idx + 704];
    out[trd_idx + 704] = acc;
    acc = acc + in[trd_idx + 832];
    out[trd_idx + 832] = acc;
    acc = acc + in[trd_idx + 960];
    out[trd_idx + 960] = acc;

    // Group 2: positions trd_idx + 32 + k*128 for k=0..7
    acc = acc + in[trd_idx + 32];
    out[trd_idx + 32] = acc;
    acc = acc + in[trd_idx + 160];
    out[trd_idx + 160] = acc;
    acc = acc + in[trd_idx + 288];
    out[trd_idx + 288] = acc;
    acc = acc + in[trd_idx + 416];
    out[trd_idx + 416] = acc;
    acc = acc + in[trd_idx + 544];
    out[trd_idx + 544] = acc;
    acc = acc + in[trd_idx + 672];
    out[trd_idx + 672] = acc;
    acc = acc + in[trd_idx + 800];
    out[trd_idx + 800] = acc;
    acc = acc + in[trd_idx + 928];
    out[trd_idx + 928] = acc;

    // Group 3: positions trd_idx + 96 + k*128 for k=0..7
    acc = acc + in[trd_idx + 96];
    out[trd_idx + 96] = acc;
    acc = acc + in[trd_idx + 224];
    out[trd_idx + 224] = acc;
    acc = acc + in[trd_idx + 352];
    out[trd_idx + 352] = acc;
    acc = acc + in[trd_idx + 480];
    out[trd_idx + 480] = acc;
    acc = acc + in[trd_idx + 608];
    out[trd_idx + 608] = acc;
    acc = acc + in[trd_idx + 736];
    out[trd_idx + 736] = acc;
    acc = acc + in[trd_idx + 864];
    out[trd_idx + 864] = acc;
    acc = acc + in[trd_idx + 992];
    out[trd_idx + 992] = acc;

    // Group 4: positions trd_idx + 16 + k*128 for k=0..7
    acc = acc + in[trd_idx + 16];
    out[trd_idx + 16] = acc;
    acc = acc + in[trd_idx + 144];
    out[trd_idx + 144] = acc;
    acc = acc + in[trd_idx + 272];
    out[trd_idx + 272] = acc;
    acc = acc + in[trd_idx + 400];
    out[trd_idx + 400] = acc;
    acc = acc + in[trd_idx + 528];
    out[trd_idx + 528] = acc;
    acc = acc + in[trd_idx + 656];
    out[trd_idx + 656] = acc;
    acc = acc + in[trd_idx + 784];
    out[trd_idx + 784] = acc;
    acc = acc + in[trd_idx + 912];
    out[trd_idx + 912] = acc;

    // Group 5: positions trd_idx + 80 + k*128 for k=0..7
    acc = acc + in[trd_idx + 80];
    out[trd_idx + 80] = acc;
    acc = acc + in[trd_idx + 208];
    out[trd_idx + 208] = acc;
    acc = acc + in[trd_idx + 336];
    out[trd_idx + 336] = acc;
    acc = acc + in[trd_idx + 464];
    out[trd_idx + 464] = acc;
    acc = acc + in[trd_idx + 592];
    out[trd_idx + 592] = acc;
    acc = acc + in[trd_idx + 720];
    out[trd_idx + 720] = acc;
    acc = acc + in[trd_idx + 848];
    out[trd_idx + 848] = acc;
    acc = acc + in[trd_idx + 976];
    out[trd_idx + 976] = acc;

    // Group 6: positions trd_idx + 48 + k*128 for k=0..7
    acc = acc + in[trd_idx + 48];
    out[trd_idx + 48] = acc;
    acc = acc + in[trd_idx + 176];
    out[trd_idx + 176] = acc;
    acc = acc + in[trd_idx + 304];
    out[trd_idx + 304] = acc;
    acc = acc + in[trd_idx + 432];
    out[trd_idx + 432] = acc;
    acc = acc + in[trd_idx + 560];
    out[trd_idx + 560] = acc;
    acc = acc + in[trd_idx + 688];
    out[trd_idx + 688] = acc;
    acc = acc + in[trd_idx + 816];
    out[trd_idx + 816] = acc;
    acc = acc + in[trd_idx + 944];
    out[trd_idx + 944] = acc;

    // Group 7: positions trd_idx + 112 + k*128 for k=0..7
    acc = acc + in[trd_idx + 112];
    out[trd_idx + 112] = acc;
    acc = acc + in[trd_idx + 240];
    out[trd_idx + 240] = acc;
    acc = acc + in[trd_idx + 368];
    out[trd_idx + 368] = acc;
    acc = acc + in[trd_idx + 496];
    out[trd_idx + 496] = acc;
    acc = acc + in[trd_idx + 624];
    out[trd_idx + 624] = acc;
    acc = acc + in[trd_idx + 752];
    out[trd_idx + 752] = acc;
    acc = acc + in[trd_idx + 880];
    out[trd_idx + 880] = acc;
    acc = acc + in[trd_idx + 1008];
    out[trd_idx + 1008] = acc;
}
