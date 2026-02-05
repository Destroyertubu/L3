#include "fastlanes.cuh"
#include "debug.hpp"
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "fls_gen/unpack/unpack_64.cuh"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

// ==================== FOR解码内核 ====================

// FOR解码内核 - uint32: unpack + add base
__global__ void unpack_for_global(const uint32_t* __restrict in, uint32_t* __restrict out,
                                   const uint32_t* __restrict base, uint8_t bw) {
    uint32_t blc_idx = blockIdx.x;
    uint32_t trd_idx = threadIdx.x;
    in  = in + ((blc_idx * bw) << 5);  // bw * 32 uint32s per block
    out = out + (blc_idx << 10);        // 1024 elements per block
    uint32_t base_val = base[blc_idx];

    __shared__ uint32_t sm_arr[1024];

    // Step 1: Unpack to shared memory
    unpack_device(in, sm_arr, bw);
    __syncthreads();

    // Step 2: Add base and write to global memory
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint32_t idx = trd_idx + i * 32;
        out[idx] = sm_arr[idx] + base_val;
    }
}

// FOR解码内核 - uint64: unpack + add base
__global__ void unpack_for_global_64(const uint64_t* __restrict in, uint64_t* __restrict out,
                                      const uint64_t* __restrict base, uint8_t bw) {
    uint32_t blc_idx = blockIdx.x;
    uint32_t trd_idx = threadIdx.x;
    in  = in + (blc_idx * bw * 16);  // bw * 16 uint64s per block
    out = out + (blc_idx * 1024);
    uint64_t base_val = base[blc_idx];

    __shared__ uint64_t sm_arr[1024];

    // Step 1: Unpack to shared memory
    unpack_device_64(in, sm_arr, bw);
    __syncthreads();

    // Step 2: Add base and write to global memory
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uint32_t idx = trd_idx + i * 32;
        out[idx] = sm_arr[idx] + base_val;
    }
}

// ==================== 辅助函数 ====================

// 计算需要的位宽 - uint32
uint8_t compute_bitwidth_32(uint32_t* arr, uint64_t n) {
    uint32_t max_val = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    if (max_val == 0) return 1;
    return static_cast<uint8_t>(std::ceil(std::log2(max_val + 1)));
}

// 计算需要的位宽 - uint64
uint8_t compute_bitwidth_64(uint64_t* arr, uint64_t n) {
    uint64_t max_val = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    if (max_val == 0) return 1;
    return static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(max_val) + 1)));
}

// 计算每个block的FOR参数 (min value和位宽) - uint32
void compute_for_params_32(uint32_t* arr, uint64_t n_vec, uint64_t vec_sz,
                            uint32_t* base_arr, uint8_t* bitwidths) {
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        uint32_t* block = arr + vec_idx * vec_sz;

        // Find min and max in this block
        uint32_t min_val = block[0];
        uint32_t max_val = block[0];
        for (uint64_t i = 1; i < vec_sz; i++) {
            if (block[i] < min_val) min_val = block[i];
            if (block[i] > max_val) max_val = block[i];
        }

        base_arr[vec_idx] = min_val;
        uint32_t range = max_val - min_val;
        bitwidths[vec_idx] = (range == 0) ? 1 : static_cast<uint8_t>(std::ceil(std::log2(range + 1)));
    }
}

// 计算每个block的FOR参数 - uint64
void compute_for_params_64(uint64_t* arr, uint64_t n_vec, uint64_t vec_sz,
                            uint64_t* base_arr, uint8_t* bitwidths) {
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        uint64_t* block = arr + vec_idx * vec_sz;

        uint64_t min_val = block[0];
        uint64_t max_val = block[0];
        for (uint64_t i = 1; i < vec_sz; i++) {
            if (block[i] < min_val) min_val = block[i];
            if (block[i] > max_val) max_val = block[i];
        }

        base_arr[vec_idx] = min_val;
        uint64_t range = max_val - min_val;
        bitwidths[vec_idx] = (range == 0) ? 1 : static_cast<uint8_t>(std::ceil(std::log2(static_cast<double>(range) + 1)));
    }
}

// ==================== uint32 基准测试 ====================

int run_benchmark_uint32(const char* data_file) {
    std::cout << "------------------------------------ \n";
    std::cout << "-- Init (uint32):  \n";
    cudaDeviceSynchronize();

    std::ifstream file(data_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << data_file << std::endl;
        return -1;
    }
    uint64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const uint64_t warp_sz         = 32;
    const uint64_t vec_sz          = 1024;
    uint64_t       total_elements  = file_size / sizeof(uint32_t);
    const uint64_t n_vec           = (total_elements / vec_sz);
    const uint64_t n_tup           = vec_sz * n_vec;
    const uint64_t v_blc_sz        = 1;
    const uint64_t n_blc           = n_vec / v_blc_sz;
    const uint64_t n_trd           = v_blc_sz * warp_sz;

    std::cout << "-- File: " << data_file << "\n";
    std::cout << "-- File size: " << file_size << " bytes\n";
    std::cout << "-- Total elements: " << total_elements << "\n";
    std::cout << "-- Processing elements: " << n_tup << "\n";
    std::cout << "-- Number of vectors: " << n_vec << "\n";

    // Allocate memory
    auto* h_org_arr        = new uint32_t[n_tup];
    auto* h_encoded_bp     = new uint32_t[n_tup];  // BitPack encoded
    auto* h_encoded_for    = new uint32_t[n_tup];  // FOR encoded
    auto* h_subtracted     = new uint32_t[n_tup];  // Data after subtracting base
    auto* h_decoded_arr    = new uint32_t[n_tup];
    auto* h_base_arr       = new uint32_t[n_vec];  // FOR base values
    auto* h_bitwidths      = new uint8_t[n_vec];   // FOR per-block bitwidths

    uint32_t* d_decoded_arr   = nullptr;
    uint32_t* d_encoded_bp    = nullptr;
    uint32_t* d_encoded_for   = nullptr;
    uint32_t* d_base_arr      = nullptr;

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint32_t) * n_tup));

    std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
    std::cout << "------------------------------------ \n";
    std::cout << "-- Load data:  \n";

    file.read(reinterpret_cast<char*>(h_org_arr), n_tup * sizeof(uint32_t));
    file.close();

    uint64_t original_size = n_tup * sizeof(uint32_t);

    // ==================== BitPack ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- [BitPack] Encode:  \n";

    uint8_t bp_bitwidth = compute_bitwidth_32(h_org_arr, n_tup);
    std::cout << "-- BitPack bitwidth: " << (int)bp_bitwidth << "\n";

    auto bp_encode_start = std::chrono::high_resolution_clock::now();
    auto* in_ptr = h_org_arr;
    auto* out_ptr = h_encoded_bp;
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bp_bitwidth);
        in_ptr  += vec_sz;
        out_ptr += (bp_bitwidth * vec_sz / 32);
    }
    auto bp_encode_end = std::chrono::high_resolution_clock::now();
    double bp_encode_time_ms = std::chrono::duration<double, std::milli>(bp_encode_end - bp_encode_start).count();

    uint64_t bp_compressed_size = n_vec * (bp_bitwidth * vec_sz / 32) * sizeof(uint32_t);
    double bp_compression_ratio = (double)original_size / bp_compressed_size;
    double bp_encode_throughput = (original_size / 1e9) / (bp_encode_time_ms / 1000.0);

    std::cout << "-- Compressed size: " << (bp_compressed_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Compression ratio: " << bp_compression_ratio << "x\n";
    std::cout << "-- Encode time: " << bp_encode_time_ms << " ms\n";
    std::cout << "-- Encode throughput: " << bp_encode_throughput << " GB/s\n";

    // Load to GPU
    d_encoded_bp = fastlanes::gpu::load_arr(h_encoded_bp, bp_compressed_size);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Decode BitPack
    std::cout << "------------------------------------ \n";
    std::cout << "-- [BitPack] Decode:  \n";

    unpack_global<<<n_blc, n_trd>>>(d_encoded_bp, d_decoded_arr, bp_bitwidth);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    const int num_runs = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        unpack_global<<<n_blc, n_trd>>>(d_encoded_bp, d_decoded_arr, bp_bitwidth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float bp_decode_time_ms = 0;
    cudaEventElapsedTime(&bp_decode_time_ms, start, stop);
    bp_decode_time_ms /= num_runs;
    double bp_decode_throughput = (original_size / 1e9) / (bp_decode_time_ms / 1000.0);

    std::cout << "-- Decode time: " << bp_decode_time_ms << " ms\n";
    std::cout << "-- Decode throughput: " << bp_decode_throughput << " GB/s\n";

    // Verify BitPack
    CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
    int bp_errors = 0;
    for (uint64_t i = 0; i < n_tup; i++) {
        if (h_org_arr[i] != h_decoded_arr[i]) bp_errors++;
    }
    std::cout << "-- Verification: " << (bp_errors == 0 ? "PASSED" : "FAILED") << "\n";

    // ==================== FOR (Frame of Reference) ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- [FOR] Encode:  \n";

    // Compute FOR parameters
    compute_for_params_32(h_org_arr, n_vec, vec_sz, h_base_arr, h_bitwidths);

    // Calculate average bitwidth and total compressed size
    uint64_t total_bits = 0;
    for (uint64_t i = 0; i < n_vec; i++) {
        total_bits += h_bitwidths[i] * vec_sz;
    }
    double avg_bitwidth = (double)total_bits / n_tup;
    std::cout << "-- FOR average bitwidth: " << avg_bitwidth << "\n";

    // Use uniform bitwidth (max of all blocks) for simplicity
    uint8_t for_bitwidth = 0;
    for (uint64_t i = 0; i < n_vec; i++) {
        if (h_bitwidths[i] > for_bitwidth) for_bitwidth = h_bitwidths[i];
    }
    std::cout << "-- FOR uniform bitwidth: " << (int)for_bitwidth << "\n";

    // Subtract base and pack
    auto for_encode_start = std::chrono::high_resolution_clock::now();
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        uint32_t base_val = h_base_arr[vec_idx];
        for (uint64_t i = 0; i < vec_sz; i++) {
            h_subtracted[vec_idx * vec_sz + i] = h_org_arr[vec_idx * vec_sz + i] - base_val;
        }
    }

    in_ptr = h_subtracted;
    out_ptr = h_encoded_for;
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        generated::pack::fallback::scalar::pack(in_ptr, out_ptr, for_bitwidth);
        in_ptr  += vec_sz;
        out_ptr += (for_bitwidth * vec_sz / 32);
    }
    auto for_encode_end = std::chrono::high_resolution_clock::now();
    double for_encode_time_ms = std::chrono::duration<double, std::milli>(for_encode_end - for_encode_start).count();

    uint64_t for_data_size = n_vec * (for_bitwidth * vec_sz / 32) * sizeof(uint32_t);
    uint64_t for_base_size = n_vec * sizeof(uint32_t);
    uint64_t for_total_size = for_data_size + for_base_size;
    double for_compression_ratio = (double)original_size / for_total_size;
    double for_encode_throughput = (original_size / 1e9) / (for_encode_time_ms / 1000.0);

    std::cout << "-- Compressed data: " << (for_data_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Base array: " << (for_base_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Total size: " << (for_total_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Compression ratio: " << for_compression_ratio << "x\n";
    std::cout << "-- Encode time: " << for_encode_time_ms << " ms\n";
    std::cout << "-- Encode throughput: " << for_encode_throughput << " GB/s\n";

    // Load FOR data to GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_for, for_data_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_encoded_for, h_encoded_for, for_data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base_arr, for_base_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_base_arr, h_base_arr, for_base_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Decode FOR
    std::cout << "------------------------------------ \n";
    std::cout << "-- [FOR] Decode:  \n";

    unpack_for_global<<<n_blc, n_trd>>>(d_encoded_for, d_decoded_arr, d_base_arr, for_bitwidth);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        unpack_for_global<<<n_blc, n_trd>>>(d_encoded_for, d_decoded_arr, d_base_arr, for_bitwidth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float for_decode_time_ms = 0;
    cudaEventElapsedTime(&for_decode_time_ms, start, stop);
    for_decode_time_ms /= num_runs;
    double for_decode_throughput = (original_size / 1e9) / (for_decode_time_ms / 1000.0);

    std::cout << "-- Decode time: " << for_decode_time_ms << " ms\n";
    std::cout << "-- Decode throughput: " << for_decode_throughput << " GB/s\n";

    // Verify FOR
    CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint32_t) * n_tup, cudaMemcpyDeviceToHost));
    int for_errors = 0;
    for (uint64_t i = 0; i < n_tup; i++) {
        if (h_org_arr[i] != h_decoded_arr[i]) {
            for_errors++;
            if (for_errors <= 5) {
                std::cout << "ERROR: idx " << i << " : " << h_org_arr[i] << " != " << h_decoded_arr[i] << "\n";
            }
        }
    }
    std::cout << "-- Verification: " << (for_errors == 0 ? "PASSED" : "FAILED") << "\n";

    // ==================== Summary ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- Summary (uint32):  \n";
    std::cout << "-- Original size: " << (original_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- | Algorithm | Bitwidth | Comp.Ratio | Enc.Time(ms) | Dec.Time(ms) | Dec.Throughput(GB/s) |\n";
    std::cout << "-- |-----------|----------|------------|--------------|--------------|----------------------|\n";
    std::cout << "-- | BitPack   | " << (int)bp_bitwidth << " | " << bp_compression_ratio << "x | "
              << bp_encode_time_ms << " | " << bp_decode_time_ms << " | " << bp_decode_throughput << " |\n";
    std::cout << "-- | FOR       | " << (int)for_bitwidth << " | " << for_compression_ratio << "x | "
              << for_encode_time_ms << " | " << for_decode_time_ms << " | " << for_decode_throughput << " |\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_org_arr;
    delete[] h_encoded_bp;
    delete[] h_encoded_for;
    delete[] h_subtracted;
    delete[] h_decoded_arr;
    delete[] h_base_arr;
    delete[] h_bitwidths;

    CUDA_SAFE_CALL(cudaFree(d_decoded_arr));
    CUDA_SAFE_CALL(cudaFree(d_encoded_bp));
    CUDA_SAFE_CALL(cudaFree(d_encoded_for));
    CUDA_SAFE_CALL(cudaFree(d_base_arr));

    return (bp_errors == 0 && for_errors == 0) ? 0 : -1;
}

// ==================== uint64 基准测试 ====================

int run_benchmark_uint64(const char* data_file) {
    std::cout << "------------------------------------ \n";
    std::cout << "-- Init (uint64):  \n";
    cudaDeviceSynchronize();

    std::ifstream file(data_file, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << data_file << std::endl;
        return -1;
    }
    uint64_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    const uint64_t warp_sz         = 32;
    const uint64_t vec_sz          = 1024;
    uint64_t       total_elements  = file_size / sizeof(uint64_t);
    const uint64_t n_vec           = (total_elements / vec_sz);
    const uint64_t n_tup           = vec_sz * n_vec;
    const uint64_t v_blc_sz        = 1;
    const uint64_t n_blc           = n_vec / v_blc_sz;
    const uint64_t n_trd           = v_blc_sz * warp_sz;

    std::cout << "-- File: " << data_file << "\n";
    std::cout << "-- File size: " << file_size << " bytes\n";
    std::cout << "-- Total elements: " << total_elements << "\n";
    std::cout << "-- Processing elements: " << n_tup << "\n";
    std::cout << "-- Number of vectors: " << n_vec << "\n";

    // Allocate memory
    auto* h_org_arr        = new uint64_t[n_tup];
    auto* h_encoded_bp     = new uint64_t[n_tup];
    auto* h_encoded_for    = new uint64_t[n_tup];
    auto* h_subtracted     = new uint64_t[n_tup];
    auto* h_decoded_arr    = new uint64_t[n_tup];
    auto* h_base_arr       = new uint64_t[n_vec];
    auto* h_bitwidths      = new uint8_t[n_vec];

    uint64_t* d_decoded_arr   = nullptr;
    uint64_t* d_encoded_bp    = nullptr;
    uint64_t* d_encoded_for   = nullptr;
    uint64_t* d_base_arr      = nullptr;

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_decoded_arr, sizeof(uint64_t) * n_tup));

    std::cout << fastlanes::debug::green << "-- successful ! " << fastlanes::debug::def << '\n';
    std::cout << "------------------------------------ \n";
    std::cout << "-- Load data:  \n";

    file.read(reinterpret_cast<char*>(h_org_arr), n_tup * sizeof(uint64_t));
    file.close();

    uint64_t original_size = n_tup * sizeof(uint64_t);

    // ==================== BitPack ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- [BitPack] Encode:  \n";

    uint8_t bp_bitwidth = compute_bitwidth_64(h_org_arr, n_tup);
    std::cout << "-- BitPack bitwidth: " << (int)bp_bitwidth << "\n";

    auto bp_encode_start = std::chrono::high_resolution_clock::now();
    auto* in_ptr = h_org_arr;
    auto* out_ptr = h_encoded_bp;
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        generated::pack::fallback::scalar::pack(in_ptr, out_ptr, bp_bitwidth);
        in_ptr  += vec_sz;
        out_ptr += (bp_bitwidth * vec_sz / 64);
    }
    auto bp_encode_end = std::chrono::high_resolution_clock::now();
    double bp_encode_time_ms = std::chrono::duration<double, std::milli>(bp_encode_end - bp_encode_start).count();

    uint64_t bp_compressed_size = n_vec * (bp_bitwidth * vec_sz / 64) * sizeof(uint64_t);
    double bp_compression_ratio = (double)original_size / bp_compressed_size;
    double bp_encode_throughput = (original_size / 1e9) / (bp_encode_time_ms / 1000.0);

    std::cout << "-- Compressed size: " << (bp_compressed_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Compression ratio: " << bp_compression_ratio << "x\n";
    std::cout << "-- Encode time: " << bp_encode_time_ms << " ms\n";
    std::cout << "-- Encode throughput: " << bp_encode_throughput << " GB/s\n";

    // Load to GPU
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_bp, bp_compressed_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_encoded_bp, h_encoded_bp, bp_compressed_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Decode BitPack
    std::cout << "------------------------------------ \n";
    std::cout << "-- [BitPack] Decode:  \n";

    unpack_global_64<<<n_blc, n_trd>>>(d_encoded_bp, d_decoded_arr, bp_bitwidth);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    const int num_runs = 10;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        unpack_global_64<<<n_blc, n_trd>>>(d_encoded_bp, d_decoded_arr, bp_bitwidth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float bp_decode_time_ms = 0;
    cudaEventElapsedTime(&bp_decode_time_ms, start, stop);
    bp_decode_time_ms /= num_runs;
    double bp_decode_throughput = (original_size / 1e9) / (bp_decode_time_ms / 1000.0);

    std::cout << "-- Decode time: " << bp_decode_time_ms << " ms\n";
    std::cout << "-- Decode throughput: " << bp_decode_throughput << " GB/s\n";

    // Verify BitPack
    CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
    int bp_errors = 0;
    for (uint64_t i = 0; i < n_tup; i++) {
        if (h_org_arr[i] != h_decoded_arr[i]) bp_errors++;
    }
    std::cout << "-- Verification: " << (bp_errors == 0 ? "PASSED" : "FAILED") << "\n";

    // ==================== FOR ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- [FOR] Encode:  \n";

    compute_for_params_64(h_org_arr, n_vec, vec_sz, h_base_arr, h_bitwidths);

    uint8_t for_bitwidth = 0;
    for (uint64_t i = 0; i < n_vec; i++) {
        if (h_bitwidths[i] > for_bitwidth) for_bitwidth = h_bitwidths[i];
    }
    std::cout << "-- FOR uniform bitwidth: " << (int)for_bitwidth << "\n";

    auto for_encode_start = std::chrono::high_resolution_clock::now();
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        uint64_t base_val = h_base_arr[vec_idx];
        for (uint64_t i = 0; i < vec_sz; i++) {
            h_subtracted[vec_idx * vec_sz + i] = h_org_arr[vec_idx * vec_sz + i] - base_val;
        }
    }

    in_ptr = h_subtracted;
    out_ptr = h_encoded_for;
    for (uint64_t vec_idx = 0; vec_idx < n_vec; vec_idx++) {
        generated::pack::fallback::scalar::pack(in_ptr, out_ptr, for_bitwidth);
        in_ptr  += vec_sz;
        out_ptr += (for_bitwidth * vec_sz / 64);
    }
    auto for_encode_end = std::chrono::high_resolution_clock::now();
    double for_encode_time_ms = std::chrono::duration<double, std::milli>(for_encode_end - for_encode_start).count();

    uint64_t for_data_size = n_vec * (for_bitwidth * vec_sz / 64) * sizeof(uint64_t);
    uint64_t for_base_size = n_vec * sizeof(uint64_t);
    uint64_t for_total_size = for_data_size + for_base_size;
    double for_compression_ratio = (double)original_size / for_total_size;
    double for_encode_throughput = (original_size / 1e9) / (for_encode_time_ms / 1000.0);

    std::cout << "-- Compressed data: " << (for_data_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Base array: " << (for_base_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Total size: " << (for_total_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- Compression ratio: " << for_compression_ratio << "x\n";
    std::cout << "-- Encode time: " << for_encode_time_ms << " ms\n";
    std::cout << "-- Encode throughput: " << for_encode_throughput << " GB/s\n";

    CUDA_SAFE_CALL(cudaMalloc((void**)&d_encoded_for, for_data_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_encoded_for, h_encoded_for, for_data_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_base_arr, for_base_size));
    CUDA_SAFE_CALL(cudaMemcpy(d_base_arr, h_base_arr, for_base_size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Decode FOR
    std::cout << "------------------------------------ \n";
    std::cout << "-- [FOR] Decode:  \n";

    unpack_for_global_64<<<n_blc, n_trd>>>(d_encoded_for, d_decoded_arr, d_base_arr, for_bitwidth);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaEventRecord(start);
    for (int run = 0; run < num_runs; run++) {
        unpack_for_global_64<<<n_blc, n_trd>>>(d_encoded_for, d_decoded_arr, d_base_arr, for_bitwidth);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float for_decode_time_ms = 0;
    cudaEventElapsedTime(&for_decode_time_ms, start, stop);
    for_decode_time_ms /= num_runs;
    double for_decode_throughput = (original_size / 1e9) / (for_decode_time_ms / 1000.0);

    std::cout << "-- Decode time: " << for_decode_time_ms << " ms\n";
    std::cout << "-- Decode throughput: " << for_decode_throughput << " GB/s\n";

    // Verify FOR
    CUDA_SAFE_CALL(cudaMemcpy(h_decoded_arr, d_decoded_arr, sizeof(uint64_t) * n_tup, cudaMemcpyDeviceToHost));
    int for_errors = 0;
    for (uint64_t i = 0; i < n_tup; i++) {
        if (h_org_arr[i] != h_decoded_arr[i]) {
            for_errors++;
            if (for_errors <= 5) {
                std::cout << "ERROR: idx " << i << " : " << h_org_arr[i] << " != " << h_decoded_arr[i] << "\n";
            }
        }
    }
    std::cout << "-- Verification: " << (for_errors == 0 ? "PASSED" : "FAILED") << "\n";

    // ==================== Summary ====================
    std::cout << "------------------------------------ \n";
    std::cout << "-- Summary (uint64):  \n";
    std::cout << "-- Original size: " << (original_size / 1024.0 / 1024.0) << " MB\n";
    std::cout << "-- | Algorithm | Bitwidth | Comp.Ratio | Enc.Time(ms) | Dec.Time(ms) | Dec.Throughput(GB/s) |\n";
    std::cout << "-- |-----------|----------|------------|--------------|--------------|----------------------|\n";
    std::cout << "-- | BitPack   | " << (int)bp_bitwidth << " | " << bp_compression_ratio << "x | "
              << bp_encode_time_ms << " | " << bp_decode_time_ms << " | " << bp_decode_throughput << " |\n";
    std::cout << "-- | FOR       | " << (int)for_bitwidth << " | " << for_compression_ratio << "x | "
              << for_encode_time_ms << " | " << for_decode_time_ms << " | " << for_decode_throughput << " |\n";

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] h_org_arr;
    delete[] h_encoded_bp;
    delete[] h_encoded_for;
    delete[] h_subtracted;
    delete[] h_decoded_arr;
    delete[] h_base_arr;
    delete[] h_bitwidths;

    CUDA_SAFE_CALL(cudaFree(d_decoded_arr));
    CUDA_SAFE_CALL(cudaFree(d_encoded_bp));
    CUDA_SAFE_CALL(cudaFree(d_encoded_for));
    CUDA_SAFE_CALL(cudaFree(d_base_arr));

    return (bp_errors == 0 && for_errors == 0) ? 0 : -1;
}

int main(int argc, char** argv) {
    bool use_uint64 = false;
    const char* data_file = "/root/autodl-tmp/code/L3/data/sosd/5-books_200M_uint32.bin";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--uint64") == 0) {
            use_uint64 = true;
        } else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) {
            data_file = argv[++i];
        }
    }

    std::cout << "FastLanes BitPack + FOR Benchmark\n";
    std::cout << "Data type: " << (use_uint64 ? "uint64" : "uint32") << "\n";

    if (use_uint64) {
        return run_benchmark_uint64(data_file);
    } else {
        return run_benchmark_uint32(data_file);
    }
}
