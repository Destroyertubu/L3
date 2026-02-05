/**
 * @file q33_optimized.cu
 * @brief SSB Q3.3 - Optimized. c_city in ('UNITED KI1','UNITED KI5'), s_city in same, d_year 1992-1997
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include "ssb_data_loader.hpp"
#include "ssb_common.cuh"
#include "ssb_compressed_utils.cuh"
#include "L3_format.hpp"
#include "L3_codec.hpp"

using namespace ssb;

constexpr uint32_t CITY_UK1 = 231;
constexpr uint32_t CITY_UK5 = 235;
constexpr int NUM_YEARS = 6;
constexpr int NUM_CITIES = 250;
constexpr int AGG_SIZE = NUM_YEARS * NUM_CITIES * NUM_CITIES;

__device__ inline bool isCityMatch(uint32_t city) { return city == CITY_UK1 || city == CITY_UK5; }

__global__ void probeSupplierQ33Opt(const uint32_t* suppkeys, int n, const uint32_t* ht_keys,
    const uint32_t* ht_vals, int ht_size, int* out_idx, uint32_t* out_city, int* num_passing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp_id = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    __shared__ int wc[8], wo[8], bo;
    bool found = false; uint32_t city = 0;
    if (tid < n) { if (twoLevelProbeWithValue(suppkeys[tid], ht_keys, ht_vals, ht_size, city)) found = isCityMatch(city); }
    unsigned ballot = __ballot_sync(0xffffffff, found);
    if (lane == 0) wc[warp_id] = __popc(ballot);
    __syncthreads();
    if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < nwarps; w++) { wo[w] = t; t += wc[w]; } if (t > 0) bo = atomicAdd(num_passing, t); }
    __syncthreads();
    if (found && tid < n) { int pos = bo + wo[warp_id] + __popc(ballot & ((1u << lane) - 1)); out_idx[pos] = tid; out_city[pos] = city; }
}

__global__ void probeCustomerQ33Opt(const uint32_t* custkeys, const int* in_idx, const uint32_t* in_scity,
    int n, const uint32_t* ht_keys, const uint32_t* ht_vals, int ht_size,
    int* out_idx, uint32_t* out_ccity, uint32_t* out_scity, int* num_passing) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31, warp_id = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
    __shared__ int wc[8], wo[8], bo;
    bool found = false; uint32_t ccity = 0, scity = 0; int orig = -1;
    if (tid < n) { orig = in_idx[tid]; scity = in_scity[tid]; if (twoLevelProbeWithValue(custkeys[tid], ht_keys, ht_vals, ht_size, ccity)) found = isCityMatch(ccity); }
    unsigned ballot = __ballot_sync(0xffffffff, found);
    if (lane == 0) wc[warp_id] = __popc(ballot);
    __syncthreads();
    if (threadIdx.x == 0) { int t = 0; for (int w = 0; w < nwarps; w++) { wo[w] = t; t += wc[w]; } if (t > 0) bo = atomicAdd(num_passing, t); }
    __syncthreads();
    if (found && tid < n) { int pos = bo + wo[warp_id] + __popc(ballot & ((1u << lane) - 1)); out_idx[pos] = orig; out_ccity[pos] = ccity; out_scity[pos] = scity; }
}

__global__ void aggregateQ33Opt(const uint32_t* orderdate, const uint32_t* revenue,
    const uint32_t* ccity, const uint32_t* scity, int n,
    const uint32_t* ht_d_keys, const uint32_t* ht_d_vals, int ht_d_size, unsigned long long* agg) {
    __shared__ SharedDateCache dc;
    loadDateCacheCooperative(&dc, ht_d_keys, ht_d_vals, ht_d_size);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        uint32_t yr = 0;
        if (probeSharedDateCache(orderdate[i], &dc, ht_d_size, yr) && yr >= 1992 && yr <= 1997) {
            int idx = (yr - 1992) * NUM_CITIES * NUM_CITIES + ccity[i] * NUM_CITIES + scity[i];
            atomicAdd(&agg[idx], (unsigned long long)revenue[i]);
        }
    }
}

__global__ void build_date_ht(const uint32_t* dk, const uint32_t* dy, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; twoLevelInsert(dk[i], dy[i], k, v, s);
}
__global__ void build_cust_ht(const uint32_t* ck, const uint32_t* cc, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; twoLevelInsert(ck[i], cc[i], k, v, s);
}
__global__ void build_supp_ht(const uint32_t* sk, const uint32_t* sc, int n, uint32_t* k, uint32_t* v, int s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return; twoLevelInsert(sk[i], sc[i], k, v, s);
}

void runQ33Optimized(SSBDataCompressedVertical& data, QueryTiming& timing) {
    CudaTimer timer; cudaStream_t stream = 0; int bs = 256;
    CompressedColumnAccessorVertical<uint32_t> a_sk(&data.lo_suppkey), a_ck(&data.lo_custkey);
    CompressedColumnAccessorVertical<uint32_t> a_od(&data.lo_orderdate), a_rv(&data.lo_revenue);
    int N = a_sk.getTotalElements();

    timer.start();
    int hd = D_LEN * 2; uint32_t *hdk, *hdv; cudaMalloc(&hdk, hd*4); cudaMalloc(&hdv, hd*4); cudaMemset(hdk, 0xFF, hd*4);
    build_date_ht<<<(D_LEN+bs-1)/bs, bs>>>(data.d_d_datekey, data.d_d_year, D_LEN, hdk, hdv, hd);
    int hc = C_LEN * 2; uint32_t *hck, *hcv; cudaMalloc(&hck, hc*4); cudaMalloc(&hcv, hc*4); cudaMemset(hck, 0xFF, hc*4);
    build_cust_ht<<<(C_LEN+bs-1)/bs, bs>>>(data.d_c_custkey, data.d_c_city, C_LEN, hck, hcv, hc);
    int hs = S_LEN * 2; uint32_t *hsk, *hsv; cudaMalloc(&hsk, hs*4); cudaMalloc(&hsv, hs*4); cudaMemset(hsk, 0xFF, hs*4);
    build_supp_ht<<<(S_LEN+bs-1)/bs, bs>>>(data.d_s_suppkey, data.d_s_city, S_LEN, hsk, hsv, hs);
    cudaDeviceSynchronize(); timer.stop(); timing.hash_build_ms = timer.elapsed_ms();

    uint32_t* d_sk; cudaMalloc(&d_sk, N*4);
    timer.start(); a_sk.decompressAll(d_sk, stream); cudaStreamSynchronize(stream); timer.stop();
    float dm = timer.elapsed_ms();

    int *s1i, *n1; uint32_t *s1c; cudaMalloc(&s1i, N*4); cudaMalloc(&s1c, N*4); cudaMalloc(&n1, 4); cudaMemset(n1, 0, 4);
    timer.start(); probeSupplierQ33Opt<<<(N+bs-1)/bs, bs>>>(d_sk, N, hsk, hsv, hs, s1i, s1c, n1); cudaDeviceSynchronize(); timer.stop();
    float p1 = timer.elapsed_ms();
    int h1; cudaMemcpy(&h1, n1, 4, cudaMemcpyDeviceToHost); cudaFree(d_sk);
    std::cout << "Stage 1: " << (100.0f*h1/N) << "%" << std::endl;
    if (h1 == 0) { timing.total_ms = dm + p1; return; }

    uint32_t* d_ck; cudaMalloc(&d_ck, h1*4);
    timer.start(); a_ck.randomAccessBatchIndices(s1i, h1, d_ck, stream); cudaStreamSynchronize(stream); timer.stop();
    float r1 = timer.elapsed_ms();

    int *s2i, *n2; uint32_t *s2cc, *s2sc; cudaMalloc(&s2i, h1*4); cudaMalloc(&s2cc, h1*4); cudaMalloc(&s2sc, h1*4); cudaMalloc(&n2, 4); cudaMemset(n2, 0, 4);
    timer.start(); probeCustomerQ33Opt<<<(h1+bs-1)/bs, bs>>>(d_ck, s1i, s1c, h1, hck, hcv, hc, s2i, s2cc, s2sc, n2); cudaDeviceSynchronize(); timer.stop();
    float p2 = timer.elapsed_ms();
    int h2; cudaMemcpy(&h2, n2, 4, cudaMemcpyDeviceToHost); cudaFree(d_ck); cudaFree(s1i); cudaFree(s1c);
    std::cout << "Stage 2: " << (100.0f*h2/N) << "%" << std::endl;
    if (h2 == 0) { timing.total_ms = dm + r1 + p1 + p2; return; }

    uint32_t *d_od, *d_rv; cudaMalloc(&d_od, h2*4); cudaMalloc(&d_rv, h2*4);
    timer.start(); a_od.randomAccessBatchIndices(s2i, h2, d_od, stream); a_rv.randomAccessBatchIndices(s2i, h2, d_rv, stream); cudaStreamSynchronize(stream); timer.stop();
    float r2 = timer.elapsed_ms();

    unsigned long long* agg; cudaMalloc(&agg, AGG_SIZE*8); cudaMemset(agg, 0, AGG_SIZE*8);
    timer.start(); aggregateQ33Opt<<<min((h2+bs-1)/bs, 256), bs>>>(d_od, d_rv, s2cc, s2sc, h2, hdk, hdv, hd, agg); cudaDeviceSynchronize(); timer.stop();
    float am = timer.elapsed_ms();

    std::vector<unsigned long long> hagg(AGG_SIZE); cudaMemcpy(hagg.data(), agg, AGG_SIZE*8, cudaMemcpyDeviceToHost);
    unsigned long long tot = 0; int grp = 0; for (int i = 0; i < AGG_SIZE; i++) if (hagg[i] > 0) { tot += hagg[i]; grp++; }

    timing.data_load_ms = dm + r1 + r2; timing.kernel_ms = p1 + p2 + am;
    timing.total_ms = timing.data_load_ms + timing.hash_build_ms + timing.kernel_ms;
    std::cout << "\n=== Q3.3 (OPTIMIZED) === Groups: " << grp << ", Total: " << tot << std::endl;
    timing.print("Q3.3");

    cudaFree(s2i); cudaFree(s2cc); cudaFree(s2sc); cudaFree(d_od); cudaFree(d_rv); cudaFree(agg);
    cudaFree(hdk); cudaFree(hdv); cudaFree(hck); cudaFree(hcv); cudaFree(hsk); cudaFree(hsv);
    cudaFree(n1); cudaFree(n2);
}

int main(int argc, char** argv) {
    std::string dir = "data/ssb"; if (argc > 1) dir = argv[1];
    std::cout << "=== SSB Q3.3 - OPTIMIZED ===" << std::endl;
    SSBDataCompressedVertical data; data.loadAndCompress(dir);
    QueryTiming t; runQ33Optimized(data, t);
    std::cout << "\n=== Benchmark ===" << std::endl;
    for (int i = 0; i < 3; i++) { QueryTiming x; runQ33Optimized(data, x); std::cout << "Run " << i+1 << ": " << x.total_ms << " ms\n"; }
    data.free(); return 0;
}
