/**
 * End-to-end test for FOR model with large integers
 */
#include <iostream>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "../src/kernels/compression/adaptive_selector.cuh"
#include "../src/kernels/utils/bitpack_utils_Vertical.cuh"

// Include encoder
#include "../src/kernels/compression/encoder_Vertical_opt.cu"

#undef WARP_SIZE

// Include decoder
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
} while(0)

int main() {
    // Create test data with values > 2^53
    const int N = 1024;
    std::vector<uint64_t> data(N);

    uint64_t base_val = 6987120121016418971ULL;  // > 2^53
    for (int i = 0; i < N; i++) {
        data[i] = base_val + i * 1000;  // Small deltas
    }

    uint64_t min_val = data[0];
    uint64_t max_val = data[N-1];
    uint64_t range = max_val - min_val;
    int delta_bits = 64 - __builtin_clzll(range);

    printf("Test data:\n");
    printf("  N = %d\n", N);
    printf("  min_val = %llu\n", (unsigned long long)min_val);
    printf("  max_val = %llu\n", (unsigned long long)max_val);
    printf("  range = %llu\n", (unsigned long long)range);
    printf("  delta_bits = %d\n", delta_bits);
    printf("  min_val > 2^53: %s\n", (min_val > (1ULL << 53)) ? "YES" : "NO");

    // Encode min_val as double using bit pattern
    double encoded_min;
    memcpy(&encoded_min, &min_val, sizeof(double));
    printf("  encoded as double: %g\n", encoded_min);

    // Verify encoding
    uint64_t decoded_min;
    memcpy(&decoded_min, &encoded_min, sizeof(uint64_t));
    printf("  decoded back: %llu\n", (unsigned long long)decoded_min);
    printf("  encoding correct: %s\n", (min_val == decoded_min) ? "YES" : "NO");

    // Allocate GPU memory
    uint64_t* d_data;
    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output, N * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(d_data, data.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Setup partition info
    int32_t h_start = 0;
    int32_t h_end = N;
    int32_t h_model_type = MODEL_FOR_BITPACK;
    int32_t h_delta_bits = delta_bits;
    int64_t h_bit_offset = 0;
    double h_model_params[4] = {encoded_min, 0.0, 0.0, 0.0};

    int32_t* d_start;
    int32_t* d_end;
    int32_t* d_model_type;
    int32_t* d_delta_bits;
    int64_t* d_bit_offset;
    double* d_model_params;
    uint32_t* d_delta_array;

    CUDA_CHECK(cudaMalloc(&d_start, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_end, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_model_type, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_delta_bits, sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_bit_offset, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_model_params, 4 * sizeof(double)));

    int64_t delta_words = (static_cast<int64_t>(N) * delta_bits + 31) / 32 + 4;
    CUDA_CHECK(cudaMalloc(&d_delta_array, delta_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_delta_array, 0, delta_words * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_start, &h_start, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_end, &h_end, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_model_type, &h_model_type, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_delta_bits, &h_delta_bits, sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bit_offset, &h_bit_offset, sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_model_params, h_model_params, 4 * sizeof(double), cudaMemcpyHostToDevice));

    // Encode
    printf("\nEncoding...\n");
    Vertical_encoder::packDeltasSequentialBranchless<uint64_t><<<(N+255)/256, 256>>>(
        d_data, d_start, d_end, d_model_type, d_model_params,
        d_delta_bits, d_bit_offset, 1, d_delta_array);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Decode
    printf("Decoding...\n");
    Vertical_decoder::decompressSequentialBranchless<uint64_t><<<(N+255)/256, 256>>>(
        d_delta_array, d_start, d_end, d_model_type, d_model_params,
        d_delta_bits, d_bit_offset, 1, d_output, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    std::vector<uint64_t> output(N);
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        if (data[i] != output[i]) {
            if (errors < 5) {
                printf("Mismatch at %d: expected %llu, got %llu (diff=%lld)\n",
                       i, (unsigned long long)data[i], (unsigned long long)output[i],
                       (long long)(output[i] - data[i]));
            }
            errors++;
        }
    }

    printf("\nResult: %d errors out of %d values\n", errors, N);
    printf("Test: %s\n", (errors == 0) ? "PASS" : "FAIL");

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_output);
    cudaFree(d_start);
    cudaFree(d_end);
    cudaFree(d_model_type);
    cudaFree(d_delta_bits);
    cudaFree(d_bit_offset);
    cudaFree(d_model_params);
    cudaFree(d_delta_array);

    return (errors == 0) ? 0 : 1;
}
