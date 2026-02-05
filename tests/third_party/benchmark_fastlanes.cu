/**
 * FastLanesGPU Benchmark - Using Library Kernels
 *
 * This benchmark directly uses the FLS-GPU library's kernels and functions.
 * Based on fastlanes/example/fastlanes_bench_bitpack.cu and fastlanes_bench_delta.cu
 *
 * Key characteristics (FLS-GPU native format):
 * - Vector size: 1024 values
 * - Threads per block: 32
 * - Values per thread: 32
 *
 * BitPack: No transpose needed, direct pack/unpack
 * Delta: Requires transpose (g2g 输出线性顺序)
 */

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <system_error>

// FLS-GPU library headers
#include "fls_gen/pack/pack.hpp"
#include "fls_gen/unpack/unpack.cuh"
#include "fls_gen/unpack/unpack_64.cuh"
// unpack.cuh and unpack_fused.cuh both define global wrappers named `unpack_device`/`unpack_global`.
// Rename the fused wrappers to avoid ODR violations in this TU.
#define unpack_global unpack_global_fused
#define unpack_device unpack_device_fused
#include "fls_gen/unpack/unpack_fused.cuh"
#undef unpack_global
#undef unpack_device
#include "fls_gen/rsum/rsum.cuh"
#include "fls_gen/unrsum/unrsum.cuh"  // GPU prefix sum for delta decoding
#include "fls_gen/transpose/transpose.hpp"
#include "fls_gen/unrsum/unrsum.hpp"  // CPU unrsum for delta encoding

// ============================================================================
// Constants
// ============================================================================

constexpr int FLS_VEC_SIZE = 1024;
constexpr int FLS_THREADS = 32;

// FastLanes unified transposed layout offsets for 32 values per thread
// Bit-reversal permutation: start offsets are 0, 64, 32, 96 (bit-reversal of 0,1,2,3)
// Each group of 8 has step=128: start, start+128, start+256, ...
__device__ __constant__ int FLS_OFFSETS_32[32] = {
    0, 128, 256, 384, 512, 640, 768, 896,    // start=0
    64, 192, 320, 448, 576, 704, 832, 960,   // start=64
    32, 160, 288, 416, 544, 672, 800, 928,   // start=32
    96, 224, 352, 480, 608, 736, 864, 992    // start=96
};

// Host version for CPU encoding
constexpr int FLS_OFFSETS_32_HOST[32] = {
    0, 128, 256, 384, 512, 640, 768, 896,    // start=0
    64, 192, 320, 448, 576, 704, 832, 960,   // start=64
    32, 160, 288, 416, 544, 672, 800, 928,   // start=32
    96, 224, 352, 480, 608, 736, 864, 992    // start=96
};

// ============================================================================
// Inverse Transpose Kernels
// ============================================================================

/**
 * Inverse transpose for FastLanes "04261537" interleaved format back to linear order.
 *
 * Delta decode output layout:
 *   transposed[t + FLS_OFFSETS_32[j]] == original[(segment_id * 32) + j]
 * where:
 *   segment_id = rotl1_5bit(t)  (i.e., inverse of the thread permutation used by transpose_i)
 *
 * This kernel restores the original linear order so Delta can be fairly compared to BitPack.
 */
__global__ void inverse_transpose_32(
    const uint32_t* __restrict__ in_transposed,
    uint32_t* __restrict__ out_linear,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t sm[FLS_VEC_SIZE];

    const size_t base = vec_idx * FLS_VEC_SIZE;

    // Coalesced load
    for (int i = threadIdx.x; i < FLS_VEC_SIZE; i += blockDim.x) {
        sm[i] = in_transposed[base + static_cast<size_t>(i)];
    }
    __syncthreads();

    // Coalesced store to linear order
    for (int k = threadIdx.x; k < FLS_VEC_SIZE; k += blockDim.x) {
        const int segment_id = k >> 5;  // /32
        const int j = k & 31;

        // thread_id = rotr1_5bit(segment_id)
        const int thread_id = (segment_id >> 1) | ((segment_id & 1) << 4);
        const int transposed_idx = thread_id + FLS_OFFSETS_32[j];

        out_linear[base + static_cast<size_t>(k)] = sm[transposed_idx];
    }
}

__global__ void inverse_transpose_64(
    const uint64_t* __restrict__ in_transposed,
    uint64_t* __restrict__ out_linear,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint64_t sm[FLS_VEC_SIZE];

    const size_t base = vec_idx * FLS_VEC_SIZE;

    // Coalesced load
    for (int i = threadIdx.x; i < FLS_VEC_SIZE; i += blockDim.x) {
        sm[i] = in_transposed[base + static_cast<size_t>(i)];
    }
    __syncthreads();

    // Coalesced store to linear order (64-bit path uses direct thread mapping, no rotation)
    for (int k = threadIdx.x; k < FLS_VEC_SIZE; k += blockDim.x) {
        const int t = k >> 5;  // /32
        const int j = k & 31;
        const int transposed_idx = t + FLS_OFFSETS_32[j];
        out_linear[base + static_cast<size_t>(k)] = sm[transposed_idx];
    }
}

// ============================================================================
// Batch Unpack Kernels - Single kernel launch for all vectors
// ============================================================================

/**
 * Batch unpack kernel for 32-bit data
 * Each block processes one vector, reading its bitwidth and offset from arrays
 */
__global__ void batch_unpack_32(
    const uint32_t* __restrict__ encoded_data,
    uint32_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    uint8_t bw = bitwidths[vec_idx];
    if (bw == 0) {
        uint32_t* out = output + vec_idx * 1024;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            out[threadIdx.x + 32 * j] = 0U;
        }
        return;
    }

    const uint32_t* in = encoded_data + vec_offsets[vec_idx];
    uint32_t* out = output + vec_idx * 1024;

    // Call the device unpack function
    generated::unpack::cuda::normal::unpack(in, out, bw);
}

/**
 * Batch unpack + FOR (Frame-of-Reference) kernel for 32-bit data.
 * Each block: unpack residuals to shared memory, add per-vector base, write to global.
 */
__global__ void batch_unpack_for_32(
    const uint32_t* __restrict__ encoded_data,
    const uint32_t* __restrict__ base_values,  // one min-value per vector
    uint32_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    uint8_t bw = bitwidths[vec_idx];
    uint32_t base = base_values[vec_idx];
    uint32_t* out = output + vec_idx * 1024;

    if (bw == 0) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            out[threadIdx.x + 32 * j] = base;
        }
        return;
    }

    const uint32_t* in = encoded_data + vec_offsets[vec_idx];

    __shared__ uint32_t sm[1024];
    generated::unpack::cuda::normal::unpack(in, sm, bw);
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int idx = threadIdx.x + 32 * j;
        out[idx] = sm[idx] + base;
    }
}

/**
 * Batch unpack kernel for 64-bit data
 * Each block processes one vector, reading its bitwidth and offset from arrays
 */
__global__ void batch_unpack_64(
    const uint64_t* __restrict__ encoded_data,
    uint64_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    uint8_t bw = bitwidths[vec_idx];
    if (bw == 0) {
        uint64_t* out = output + vec_idx * 1024;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            out[threadIdx.x + 32 * j] = 0ULL;
        }
        return;
    }

    const uint64_t* in = encoded_data + vec_offsets[vec_idx];
    uint64_t* out = output + vec_idx * 1024;

    __shared__ uint64_t sm[1024];
    generated::unpack::cuda::normal_64::unpack(in, sm, bw);
    __syncthreads();

    // inverse transpose: transposed[t + FLS_OFFSETS_32[j]] -> linear[t*32 + j]
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const int k = threadIdx.x + 32 * j;
        const int t = k >> 5;  // /32
        const int jj = k & 31;
        out[k] = sm[t + FLS_OFFSETS_32[jj]];
    }
}

// ============================================================================
// Paper-aligned aggregation kernels (register/shared -> scalar)
// ============================================================================

__device__ __forceinline__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
    // Assumes 32-thread warp.
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__global__ void bitpack_32_sum_fused_kernel(
    const uint32_t* __restrict__ encoded_data,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs,
    unsigned long long* __restrict__ out_sum
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    uint8_t bw = bitwidths[vec_idx];
    const uint32_t* in = encoded_data + vec_offsets[vec_idx];

    uint32_t values[32];
    unpack_device_fused(in, values, bw);

    unsigned long long thread_sum = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        thread_sum += static_cast<unsigned long long>(values[j]);
    }

    unsigned long long warp_sum = warp_reduce_sum_ull(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, warp_sum);
    }
}

__global__ void delta_32_sum_fused_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs,
    unsigned long long* __restrict__ out_sum
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t unpacked[1024];
    __shared__ uint32_t rsumed[1024];

    uint8_t bw = bitwidths[vec_idx];
    const uint32_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint32_t* vec_base = base + vec_idx * 32;

    unsigned long long thread_sum = 0;

    if (bw == 0) {
        // All deltas are zero => all values are the base value for this lane.
        thread_sum = static_cast<unsigned long long>(vec_base[threadIdx.x]) * 32ULL;
    } else {
        uint32_t deltas[32];
        unpack_device_fused(vec_encoded, deltas, bw);

        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unpacked[threadIdx.x + 32 * j] = deltas[j];
        }
        __syncthreads();

        // Decode delta via running sum (FastLanes order), write to shared
        d_rsum_32(unpacked, rsumed, vec_base);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 32; j++) {
            thread_sum += static_cast<unsigned long long>(rsumed[threadIdx.x + 32 * j]);
        }
    }

    unsigned long long warp_sum = warp_reduce_sum_ull(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, warp_sum);
    }
}

__global__ void bitpack_64_sum_kernel(
    const uint64_t* __restrict__ encoded_data,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs,
    unsigned long long* __restrict__ out_sum
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    uint8_t bw = bitwidths[vec_idx];
    if (bw == 0) return;

    const uint64_t* in = encoded_data + vec_offsets[vec_idx];

    __shared__ uint64_t values[1024];
    generated::unpack::cuda::normal_64::unpack(in, values, bw);
    __syncthreads();

    unsigned long long thread_sum = 0;
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        thread_sum += static_cast<unsigned long long>(values[threadIdx.x + 32 * j]);
    }

    unsigned long long warp_sum = warp_reduce_sum_ull(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, warp_sum);
    }
}

__global__ void delta_64_sum_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,
    size_t num_vecs,
    unsigned long long* __restrict__ out_sum
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint64_t deltas[1024];

    uint8_t bw = bitwidths[vec_idx];
    const uint64_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint64_t* vec_base = base + vec_idx * 32;

    unsigned long long thread_sum = 0;
    uint64_t running = vec_base[threadIdx.x];

    if (bw == 0) {
        // All deltas are zero => all values are the base value for this lane.
        thread_sum = static_cast<unsigned long long>(running) * 32ULL;
    } else {
        generated::unpack::cuda::normal_64::unpack(vec_encoded, deltas, bw);
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 32; j++) {
            running += deltas[threadIdx.x + FLS_OFFSETS_32[j]];
            thread_sum += static_cast<unsigned long long>(running);
        }
    }

    unsigned long long warp_sum = warp_reduce_sum_ull(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(out_sum, warp_sum);
    }
}

// ============================================================================
// Data Structures
// ============================================================================

struct BenchmarkResult {
    std::string framework;
    std::string algorithm;
    std::string dataset;
    size_t original_size;
    size_t compressed_size;
    size_t estimated_hbm_bytes;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double compress_throughput_gbps;
    double decompress_throughput_gbps;
    double estimated_hbm_throughput_gbps;
    bool verified;
};

// ============================================================================
// Utility Functions
// ============================================================================

bool has_sosd_header(const char* filename, size_t file_size, size_t element_size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    uint64_t header_count = 0;
    file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));
    file.close();

    size_t expected_elements = (file_size - sizeof(uint64_t)) / element_size;
    return header_count == expected_elements;
}

template<typename T>
T* load_binary_file(const char* filename, size_t& count) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    bool sosd_format = has_sosd_header(filename, file_size, sizeof(T));

    if (sosd_format) {
        uint64_t header_count;
        file.read(reinterpret_cast<char*>(&header_count), sizeof(uint64_t));
        count = header_count;

        T* data = new T[count];
        file.read(reinterpret_cast<char*>(data), count * sizeof(T));
        file.close();
        return data;
    } else {
        count = file_size / sizeof(T);
        T* data = new T[count];
        file.read(reinterpret_cast<char*>(data), file_size);
        file.close();
        return data;
    }
}

std::string extract_dataset_name(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    size_t last_dot = filename.find_last_of(".");
    return (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
}

bool is_64bit_dataset(const std::string& filename) {
    std::string lower_filename = filename;
    std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);
    return (lower_filename.find("uint64") != std::string::npos ||
            lower_filename.find("int64") != std::string::npos ||
            lower_filename.find("_u64") != std::string::npos ||
            lower_filename.find("_i64") != std::string::npos);
}

void write_csv_header(std::ofstream& csv) {
    csv << "framework,algorithm,dataset,data_size_bytes,compressed_size_bytes,"
        << "compression_ratio,compress_time_ms,decompress_time_ms,"
        << "compress_throughput_gbps,decompress_throughput_gbps,"
        << "estimated_hbm_bytes,estimated_hbm_throughput_gbps,verified\n";
}

void write_csv_result(std::ofstream& csv, const BenchmarkResult& result) {
    csv << std::fixed << std::setprecision(4)
        << result.framework << "," << result.algorithm << "," << result.dataset << ","
        << result.original_size << "," << result.compressed_size << ","
        << result.compression_ratio << "," << result.compress_time_ms << ","
        << result.decompress_time_ms << ","
        << result.compress_throughput_gbps << "," << result.decompress_throughput_gbps << ","
        << result.estimated_hbm_bytes << "," << result.estimated_hbm_throughput_gbps << ","
        << (result.verified ? "true" : "false") << "\n";
}

template<typename T>
T* load_to_gpu(T* src, size_t count) {
    T* dest = nullptr;
    cudaMalloc(&dest, count * sizeof(T));
    cudaMemcpy(dest, src, count * sizeof(T), cudaMemcpyHostToDevice);
    return dest;
}

enum class BenchmarkMode {
    kGlobalToGlobal,
    kAggregate
};

BenchmarkMode parse_mode(const std::string& mode) {
    if (mode == "g2g" || mode == "global" || mode == "global_to_global") return BenchmarkMode::kGlobalToGlobal;
    if (mode == "agg" || mode == "aggregate" || mode == "paper") return BenchmarkMode::kAggregate;
    std::cerr << "Unknown mode: " << mode << " (expected: g2g|agg)\n";
    return BenchmarkMode::kGlobalToGlobal;
}

template<typename T>
unsigned long long cpu_sum_u64(const T* data, size_t count) {
    unsigned long long sum = 0;
    for (size_t i = 0; i < count; i++) {
        sum += static_cast<unsigned long long>(data[i]);
    }
    return sum;
}

template<typename T>
bool verify_device_output_linear(
    const T* __restrict__ d_output,
    const T* __restrict__ expected,
    size_t count,
    size_t& first_error_idx,
    T& first_error_expected,
    T& first_error_actual
) {
    constexpr size_t kChunkBytes = 64ULL * 1024ULL * 1024ULL;
    const size_t chunk_elems = std::max<size_t>(1, kChunkBytes / sizeof(T));

    std::vector<T> buf(chunk_elems);
    for (size_t offset = 0; offset < count; offset += chunk_elems) {
        const size_t n = std::min(chunk_elems, count - offset);
        cudaMemcpy(buf.data(), d_output + offset, n * sizeof(T), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < n; i++) {
            const size_t idx = offset + i;
            if (buf[i] != expected[idx]) {
                first_error_idx = idx;
                first_error_expected = expected[idx];
                first_error_actual = buf[i];
                return false;
            }
        }
    }
    return true;
}

// ============================================================================
// BitPack Encoding (CPU) - No transpose needed
// ============================================================================

// Calculate required bitwidth for a vector
uint8_t calculate_bitwidth_32(const uint32_t* data, size_t count) {
    uint32_t max_val = 0;
    for (size_t i = 0; i < count; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    if (max_val == 0) return 0;
    return (uint8_t)(32 - __builtin_clz(max_val));
}

// ============================================================================
// 32-bit Custom Pack Function (consistent with 64-bit pack_64_for_gpu)
// ============================================================================

// Custom pack function for 32-bit that matches GPU unpack layout
// GPU unpack outputs in FastLanes interleaved format: thread t outputs to positions t + FLS_OFFSETS_32_HOST[j]
// To get correct output after unpack, we read from the same interleaved positions
size_t pack_32_for_gpu(const uint32_t* in, uint32_t* out, uint8_t bw) {
    if (bw == 0) return 0;
    if (bw == 32) {
        // Direct copy: thread t handles values at positions t + FLS_OFFSETS_32_HOST[0..31]
        for (int t = 0; t < 32; t++) {
            for (int j = 0; j < 32; j++) {
                out[t * 32 + j] = in[t + FLS_OFFSETS_32_HOST[j]];
            }
        }
        return 1024;
    }

    // Calculate output size: 1024 values * bw bits / 32 bits per word
    size_t out_words = (1024ULL * bw + 31) / 32;
    memset(out, 0, out_words * sizeof(uint32_t));

    const uint32_t mask = (bw == 32) ? ~0U : ((1U << bw) - 1);

    // Pack in the order that GPU unpack expects with interleaved output
    // GPU thread t outputs to positions: t + FLS_OFFSETS_32_HOST[0..31]
    // So we pack values from those same positions to get correct output
    size_t bit_pos = 0;
    for (int t = 0; t < 32; t++) {
        for (int j = 0; j < 32; j++) {
            uint32_t val = in[t + FLS_OFFSETS_32_HOST[j]] & mask;

            size_t word_idx = bit_pos / 32;
            int bit_offset = bit_pos % 32;

            if (bit_offset + bw <= 32) {
                out[word_idx] |= (val << bit_offset);
            } else {
                // Cross word boundary
                out[word_idx] |= (val << bit_offset);
                out[word_idx + 1] |= (val >> (32 - bit_offset));
            }
            bit_pos += bw;
        }
    }

    return out_words;
}

// Pack function for BitPack using linear layout (step 32)
// Matches the library's pack/unpack format:
// out word i contains values from in[i], in[i+32], in[i+64], ..., in[i+992]
size_t pack_32_linear_for_gpu(const uint32_t* in, uint32_t* out, uint8_t bw) {
    if (bw == 0) return 0;
    if (bw == 32) {
        // Direct copy: word i gets value from position i + j*32
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                out[i * 32 + j] = in[i + j * 32];
            }
        }
        return 1024;
    }

    // Calculate output size: 1024 values * bw bits / 32 bits per word
    size_t out_words = (1024ULL * bw + 31) / 32;
    memset(out, 0, out_words * sizeof(uint32_t));

    const uint32_t mask = (1U << bw) - 1;

    // Pack in the order matching library format:
    // For each output word position i, pack 32 values from positions i, i+32, i+64, ...
    // These 32 values get packed sequentially into the bit stream starting at word i
    for (int i = 0; i < 32; i++) {
        size_t bit_pos = i * bw * 32;  // Starting bit position for this "thread"
        for (int j = 0; j < 32; j++) {
            uint32_t val = in[i + j * 32] & mask;

            size_t word_idx = bit_pos / 32;
            int bit_offset = bit_pos % 32;

            if (bit_offset + bw <= 32) {
                out[word_idx] |= (val << bit_offset);
            } else {
                // Cross word boundary
                out[word_idx] |= (val << bit_offset);
                out[word_idx + 1] |= (val >> (32 - bit_offset));
            }
            bit_pos += bw;
        }
    }

    return out_words;
}

// Encode one vector (1024 values) with bitpack (uses library pack function)
// Returns packed size in uint32_t words
size_t encode_bitpack_vector_32(const uint32_t* in, uint32_t* out, uint8_t& bitwidth) {
    bitwidth = calculate_bitwidth_32(in, FLS_VEC_SIZE);
    if (bitwidth > 0) {
        generated::pack::fallback::scalar::pack(in, out, bitwidth);
        return (bitwidth * FLS_VEC_SIZE) / 32;
    }
    return 0;
}

// ============================================================================
// 32-bit Benchmark Functions
// ============================================================================

BenchmarkResult benchmark_bitpack_32(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = (mode == BenchmarkMode::kAggregate) ? "BitPackAgg" : "BitPack";
    result.dataset = extract_dataset_name(data_file);

    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    // Align to FLS_VEC_SIZE
    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    // Allocate encoded data buffer
    size_t max_encoded_size = aligned_n_tup;  // Worst case
    uint32_t* encoded_data = new uint32_t[max_encoded_size]();
    uint8_t* bitwidths = new uint8_t[num_vecs];
    size_t* vec_offsets = new size_t[num_vecs];

    // Encode
    auto encode_start = std::chrono::high_resolution_clock::now();

    size_t encoded_offset = 0;
    for (size_t v = 0; v < num_vecs; v++) {
        vec_offsets[v] = encoded_offset;
        size_t packed_size = encode_bitpack_vector_32(
            aligned_data + v * FLS_VEC_SIZE,
            encoded_data + encoded_offset,
            bitwidths[v]
        );
        encoded_offset += packed_size;
    }

    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Upload to GPU
    uint32_t* d_encoded;
    size_t* d_offsets;
    uint8_t* d_bitwidths;

    cudaMalloc(&d_encoded, encoded_offset * sizeof(uint32_t));
    cudaMalloc(&d_offsets, num_vecs * sizeof(size_t));
    cudaMalloc(&d_bitwidths, num_vecs * sizeof(uint8_t));

    cudaMemcpy(d_encoded, encoded_data, encoded_offset * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, vec_offsets, num_vecs * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitwidths, bitwidths, num_vecs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float avg_decode_time_ms = 0.0f;
    bool correct = true;

    if (mode == BenchmarkMode::kAggregate) {
        unsigned long long* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(unsigned long long));
        cudaMemset(d_sum, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        for (int i = 0; i < warmup_count; i++) {
            bitpack_32_sum_fused_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int trial = 0; trial < num_trials; trial++) {
            bitpack_32_sum_fused_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        avg_decode_time_ms = total_time / num_trials;

        if (verify) {
            const unsigned long long expected = cpu_sum_u64(aligned_data, aligned_n_tup);

            cudaMemset(d_sum, 0, sizeof(unsigned long long));
            bitpack_32_sum_fused_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
            cudaDeviceSynchronize();

            unsigned long long actual = 0;
            cudaMemcpy(&actual, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            correct = (actual == expected);
            if (!correct) {
                std::cerr << "BitPackAgg sum mismatch: expected " << expected
                          << ", got " << actual << std::endl;
            }
        }

        cudaFree(d_sum);
    } else {
        uint32_t* d_output = nullptr;
        cudaMalloc(&d_output, aligned_n_tup * sizeof(uint32_t));
        cudaMemset(d_output, 0, aligned_n_tup * sizeof(uint32_t));  // Initialize for bw=0 vectors
        cudaDeviceSynchronize();

        for (int i = 0; i < warmup_count; i++) {
            batch_unpack_32<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_output, d_offsets, d_bitwidths, num_vecs);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(start);
        for (int trial = 0; trial < num_trials; trial++) {
            batch_unpack_32<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_output, d_offsets, d_bitwidths, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        avg_decode_time_ms = total_time / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint32_t first_error_expected = 0, first_error_actual = 0;
            correct = verify_device_output_linear<uint32_t>(
                d_output, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!correct) {
                std::cerr << "BitPack verify failed at index " << first_error_idx
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Results
    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = encoded_offset * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = (mode == BenchmarkMode::kAggregate)
            ? sizeof(unsigned long long)
            : aligned_n_tup * sizeof(uint32_t);
        result.estimated_hbm_bytes = result.compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_decode_time_ms / 1000.0);
    }
    result.verified = correct;

    // Cleanup
    cudaFree(d_encoded);
    cudaFree(d_offsets);
    cudaFree(d_bitwidths);
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] bitwidths;
    delete[] vec_offsets;

    return result;
}

// ============================================================================
// 32-bit BitPack + FOR (Frame-of-Reference) Benchmark
// ============================================================================

BenchmarkResult benchmark_bitpack_for_32(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = "BitPackFOR";
    result.dataset = extract_dataset_name(data_file);

    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t max_encoded_size = aligned_n_tup;
    uint32_t* encoded_data = new uint32_t[max_encoded_size]();
    uint32_t* base_values = new uint32_t[num_vecs];
    uint8_t* bitwidths = new uint8_t[num_vecs];
    size_t* vec_offsets = new size_t[num_vecs + 1];

    // Encode: FOR (subtract min) + BitPack
    auto encode_start = std::chrono::high_resolution_clock::now();

    uint32_t residuals[FLS_VEC_SIZE];
    size_t encoded_offset = 0;
    for (size_t v = 0; v < num_vecs; v++) {
        vec_offsets[v] = encoded_offset;
        const uint32_t* vec_in = aligned_data + v * FLS_VEC_SIZE;

        // Find min value in this vector
        uint32_t min_val = vec_in[0];
        for (int i = 1; i < FLS_VEC_SIZE; i++) {
            if (vec_in[i] < min_val) min_val = vec_in[i];
        }
        base_values[v] = min_val;

        // Compute residuals = value - min
        uint32_t max_residual = 0;
        for (int i = 0; i < FLS_VEC_SIZE; i++) {
            residuals[i] = vec_in[i] - min_val;
            if (residuals[i] > max_residual) max_residual = residuals[i];
        }

        uint8_t bw = (max_residual == 0) ? 0 : (32 - __builtin_clz(max_residual));
        bitwidths[v] = bw;

        if (bw > 0) {
            generated::pack::fallback::scalar::pack(residuals, encoded_data + encoded_offset, bw);
            encoded_offset += (bw * FLS_VEC_SIZE) / 32;
        }
    }
    vec_offsets[num_vecs] = encoded_offset;

    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Print bitwidth distribution
    int bw_counts[33] = {0};
    for (size_t v = 0; v < num_vecs; v++) {
        bw_counts[bitwidths[v]]++;
    }
    std::cout << "  Bitwidth distribution: ";
    for (int bw = 0; bw <= 32; bw++) {
        if (bw_counts[bw] > 0) {
            std::cout << "bw" << bw << "=" << bw_counts[bw] << " ";
        }
    }
    std::cout << std::endl;

    // Upload to GPU
    uint32_t* d_encoded = nullptr;
    uint32_t* d_base = nullptr;
    size_t* d_offsets = nullptr;
    uint8_t* d_bitwidths = nullptr;

    cudaMalloc(&d_encoded, encoded_offset * sizeof(uint32_t));
    cudaMalloc(&d_base, num_vecs * sizeof(uint32_t));
    cudaMalloc(&d_offsets, (num_vecs + 1) * sizeof(size_t));
    cudaMalloc(&d_bitwidths, num_vecs * sizeof(uint8_t));

    cudaMemcpy(d_encoded, encoded_data, encoded_offset * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base_values, num_vecs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, vec_offsets, (num_vecs + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitwidths, bitwidths, num_vecs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float avg_decode_time_ms = 0.0f;
    bool correct = true;

    {
        uint32_t* d_output_transposed = nullptr;
        uint32_t* d_output_linear = nullptr;
        cudaMalloc(&d_output_transposed, aligned_n_tup * sizeof(uint32_t));
        cudaMalloc(&d_output_linear, aligned_n_tup * sizeof(uint32_t));
        cudaDeviceSynchronize();

        for (int i = 0; i < warmup_count; i++) {
            batch_unpack_for_32<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_base, d_output_transposed, d_offsets, d_bitwidths, num_vecs);
            inverse_transpose_32<<<num_vecs, 256>>>(
                d_output_transposed, d_output_linear, num_vecs);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int trial = 0; trial < num_trials; trial++) {
            batch_unpack_for_32<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_base, d_output_transposed, d_offsets, d_bitwidths, num_vecs);
            inverse_transpose_32<<<num_vecs, 256>>>(
                d_output_transposed, d_output_linear, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        avg_decode_time_ms = total_time / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint32_t first_error_expected = 0, first_error_actual = 0;
            correct = verify_device_output_linear<uint32_t>(
                d_output_linear, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!correct) {
                std::cerr << "BitPackFOR verify failed at index " << first_error_idx
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output_transposed);
        cudaFree(d_output_linear);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = encoded_offset * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = aligned_n_tup * sizeof(uint32_t);
        result.estimated_hbm_bytes = result.compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_decode_time_ms / 1000.0);
    }
    result.verified = correct;

    cudaFree(d_encoded);
    cudaFree(d_base);
    cudaFree(d_offsets);
    cudaFree(d_bitwidths);
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] base_values;
    delete[] bitwidths;
    delete[] vec_offsets;

    return result;
}

// ============================================================================
// 32-bit Generic Unpack Template (consistent with 64-bit unpack_generic_64)
// ============================================================================

/**
 * Generic 32-bit unpack function - template version for precompiled dispatch
 * Handles non-word-aligned thread starts correctly
 * Output uses FastLanes interleaved layout
 */
template<int BIT_WIDTH>
__device__ void unpack_generic_32(const uint32_t* __restrict__ a_in_p, uint32_t* __restrict__ a_out_p) {
    static_assert(BIT_WIDTH >= 1 && BIT_WIDTH <= 31, "BIT_WIDTH must be 1-31");

    auto out = a_out_p;
    auto in = a_in_p;

    constexpr uint32_t MASK = (1U << BIT_WIDTH) - 1;
    constexpr int BITS_PER_WORD = 32;
    constexpr int VALUES_PER_THREAD = 32;

    // Total bits needed per thread
    constexpr int TOTAL_BITS = BIT_WIDTH * VALUES_PER_THREAD;
    // Number of 32-bit words needed per thread (rounded up, +1 for cross-boundary)
    constexpr int WORDS_NEEDED = (TOTAL_BITS + BITS_PER_WORD - 1) / BITS_PER_WORD;

    int i = threadIdx.x; // THREAD INDEX

    // Calculate starting word for this thread
    int thread_start_bit = i * BIT_WIDTH * VALUES_PER_THREAD;
    int thread_start_word = thread_start_bit / BITS_PER_WORD;

    // Load input words for this thread
    uint32_t regs[WORDS_NEEDED + 1]; // +1 for cross-boundary access
    #pragma unroll
    for (int w = 0; w <= WORDS_NEEDED; w++) {
        regs[w] = in[thread_start_word + w];
    }

    // Extract each value and output in interleaved order
    #pragma unroll
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
        // Calculate global bit position and convert to local indices
        int global_bit_pos = thread_start_bit + j * BIT_WIDTH;
        int global_word = global_bit_pos / BITS_PER_WORD;
        int word_idx = global_word - thread_start_word;
        int bit_offset = global_bit_pos % BITS_PER_WORD;

        uint32_t val;
        if (bit_offset + BIT_WIDTH <= BITS_PER_WORD) {
            // No cross-boundary
            val = (regs[word_idx] >> bit_offset) & MASK;
        } else {
            // Cross-boundary: combine bits from two words
            int bits_from_first = BITS_PER_WORD - bit_offset;
            val = (regs[word_idx] >> bit_offset) | ((regs[word_idx + 1] << bits_from_first) & MASK);
            val &= MASK;
        }
        // Output in interleaved order using FLS_OFFSETS_32
        out[i + FLS_OFFSETS_32[j]] = val;
    }
}

/**
 * TRUE RUNTIME unpack function for 32-bit values
 * bitwidth is a runtime variable, NOT compile-time template parameter
 */
__device__ void unpack_runtime_32(
    int bw,  // Runtime bitwidth
    const uint32_t* __restrict__ encoded,
    uint32_t* __restrict__ output
) {
    if (bw <= 0) {
        // bw=0: all zeros
        int tid = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            output[tid + FLS_OFFSETS_32[j]] = 0;
        }
        return;
    }

    if (bw == 32) {
        // bw=32: Special case - data is stored as encoded[tid * 32 + j]
        // No bit packing, just reorder from linear to interleaved format
        int tid = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            output[tid + FLS_OFFSETS_32[j]] = encoded[tid * 32 + j];
        }
        return;
    }

    // Runtime mask computation
    uint32_t mask = (1U << bw) - 1U;

    int tid = threadIdx.x;  // Thread index (0-31)
    constexpr int VALUES_PER_THREAD = 32;

    // Calculate starting bit position for this thread
    int thread_start_bit = tid * bw * VALUES_PER_THREAD;

    // Extract 32 values for this thread
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
        int global_bit_pos = thread_start_bit + j * bw;
        int global_word = global_bit_pos / 32;
        int bit_offset = global_bit_pos % 32;

        uint32_t val;
        if (bit_offset + bw <= 32) {
            // No cross-boundary
            val = (__ldg(&encoded[global_word]) >> bit_offset) & mask;
        } else {
            // Cross-boundary: combine bits from two words
            int bits_from_first = 32 - bit_offset;
            val = (__ldg(&encoded[global_word]) >> bit_offset) |
                  ((__ldg(&encoded[global_word + 1]) << bits_from_first) & mask);
            val &= mask;
        }
        // Output in interleaved order using FLS_OFFSETS_32
        output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

/**
 * Batch delta decode kernel with TRUE RUNTIME bitwidth for 32-bit
 * Each block handles one vector, reading its bitwidth from d_bitwidths array
 */
__global__ void delta_decode_32_batch_runtime_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,  // Per-vector bitwidths
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t deltas[1056];  // padded 32*33 for bank-conflict-free transpose
    __shared__ uint32_t values_transposed[1024];

    // Read this vector's bitwidth at RUNTIME
    int bw = bitwidths[vec_idx];

    const uint32_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint32_t* vec_base = base + vec_idx * 32;
    uint32_t* vec_output = output + vec_idx * 1024;

    // Handle bw=0 case (all deltas are zero): linear output is base value per lane.
    if (bw == 0) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const int k = threadIdx.x + 32 * j;
            const int segment_id = k >> 5;  // /32
            const int thread_id = (segment_id >> 1) | ((segment_id & 1) << 4);  // rotr1_5bit
            vec_output[k] = vec_base[thread_id];
        }
        return;
    }

    // Step 1: Unpack to shared memory using LIBRARY function (fair comparison with BitPack)
    // Output format: stride=32 (thread i writes to i, i+32, i+64, ...)
    generated::unpack::cuda::normal::unpack(vec_encoded, deltas, bw);

    __syncthreads();

    // Step 2: Prefix sum in FastLanes transposed order to shared memory
    d_rsum_32(deltas, values_transposed, vec_base);
    __syncthreads();

    // Step 3: Inverse transpose - bank-conflict-free 3-phase approach
    // OLD code had 32-way bank conflict: FLS_OFFSETS_32 are all multiples of 32,
    // so all threads hit the same bank when reading values_transposed[thread_id + FLS_OFFSETS_32[tid]].
    //
    // Phase A: Each thread reads its own column from shared memory (bank-conflict-free).
    //   Thread tid reads: values_transposed[tid + FLS_OFFSETS_32[j]] for j=0..31
    //   Banks: (tid + multiple_of_32) % 32 = tid → all threads hit different banks.
    uint32_t regs[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        regs[j] = values_transposed[threadIdx.x + FLS_OFFSETS_32[j]];
    }

    // Phase B: Write to padded shared memory (stride=33 eliminates bank conflicts).
    //   Thread tid maps to segment_id = rotl1_5bit(tid).
    //   values_transposed[tid + FLS_OFFSETS_32[j]] == original[segment_id * 32 + j]
    const int segment_id = ((threadIdx.x << 1) & 31) | (threadIdx.x >> 4);  // rotl1_5bit
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        deltas[segment_id * 33 + j] = regs[j];  // padded stride 33: bank = (seg*33+j)%32 = (seg+j)%32
    }

    __syncthreads();

    // Phase C: Coalesced read from padded smem + coalesced write to global memory.
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        vec_output[threadIdx.x + 32 * j] = deltas[j * 33 + threadIdx.x];  // bank = (j*33+tid)%32 = (j+tid)%32
    }
}

/**
 * Uniform-bitwidth Delta decode kernel: all vectors share the same bitwidth.
 * No per-vector metadata arrays needed — offset is computed arithmetically.
 */
__global__ void delta_decode_32_uniform_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output,
    int bw,
    size_t words_per_vec,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t deltas[1056];  // padded 32*33 for bank-conflict-free transpose
    __shared__ uint32_t values_transposed[1024];

    const uint32_t* vec_encoded = encoded + vec_idx * words_per_vec;
    const uint32_t* vec_base = base + vec_idx * 32;
    uint32_t* vec_output = output + vec_idx * 1024;

    // Handle bw=0 case
    if (bw == 0) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const int k = threadIdx.x + 32 * j;
            const int segment_id = k >> 5;
            const int thread_id = (segment_id >> 1) | ((segment_id & 1) << 4);
            vec_output[k] = vec_base[thread_id];
        }
        return;
    }

    // Step 1: Unpack
    generated::unpack::cuda::normal::unpack(vec_encoded, deltas, bw);
    __syncthreads();

    // Step 2: Prefix sum
    d_rsum_32(deltas, values_transposed, vec_base);
    __syncthreads();

    // Step 3: Inverse transpose (bank-conflict-free 3-phase)
    uint32_t regs[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        regs[j] = values_transposed[threadIdx.x + FLS_OFFSETS_32[j]];
    }

    const int segment_id = ((threadIdx.x << 1) & 31) | (threadIdx.x >> 4);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        deltas[segment_id * 33 + j] = regs[j];
    }
    __syncthreads();

    #pragma unroll
    for (int j = 0; j < 32; j++) {
        vec_output[threadIdx.x + 32 * j] = deltas[j * 33 + threadIdx.x];
    }
}

// ============================================================================
// 32-bit Delta Decoding Kernels (precompiled template version)
// ============================================================================

/**
 * Specialized kernel for bitwidth 0 (all zeros - only base values)
 * Output in interleaved format
 */
__global__ void delta_decode_32_bw0_kernel(
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const uint32_t* vec_base = base + vec_idx * 32;
    uint32_t* vec_output = output + vec_idx * 1024;

    int tid = threadIdx.x % 32;
    uint32_t val = vec_base[tid];

    // All values in this lane are the same (base value)
    // Output in interleaved format using FLS_OFFSETS_32
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        vec_output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

// Single-vector bw0 kernel for verification
__global__ void delta_decode_32_bw0_single_kernel(
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output
) {
    int tid = threadIdx.x % 32;
    uint32_t val = base[tid];

    // Output in interleaved format
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

/**
 * Specialized kernel for bitwidth 32 (full width - no compression, just copy)
 * Encoded data is stored as: thread t's values are at out[t*32 ... t*32+31]
 * Output in interleaved format after delta decoding
 */
__global__ void delta_decode_32_bw32_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t sm_arr[1024];

    const uint32_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint32_t* vec_base = base + vec_idx * 32;
    uint32_t* vec_output = output + vec_idx * 1024;

    int tid = threadIdx.x % 32;

    // Step 1: Copy from encoded to shared memory in interleaved format
    // Encoded layout: thread t's values are at vec_encoded[t*32 + j] for j=0..31
    // Output to interleaved format: sm_arr[tid + FLS_OFFSETS_32[j]]
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        sm_arr[tid + FLS_OFFSETS_32[j]] = vec_encoded[tid * 32 + j];
    }

    __syncthreads();

    // Step 2: Prefix sum (delta decoding)
    d_unrsum_32(sm_arr, vec_output, vec_base);
}

// Single-vector bw32 kernel for verification
__global__ void delta_decode_32_bw32_single_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output
) {
    __shared__ uint32_t sm_arr[1024];

    int tid = threadIdx.x % 32;

    // Copy from encoded to shared memory in interleaved format
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        sm_arr[tid + FLS_OFFSETS_32[j]] = encoded[tid * 32 + j];
    }

    __syncthreads();

    // Prefix sum (delta decoding)
    d_unrsum_32(sm_arr, output, base);
}

/**
 * Template delta decode kernel for 32-bit data (precompiled)
 * Each block processes one 1024-element vector
 */
template<int BIT_WIDTH>
__global__ void delta_decode_32_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint32_t sm_arr[1024];

    // Get encoded data pointer for this vector
    const uint32_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint32_t* vec_base = base + vec_idx * 32;
    uint32_t* vec_output = output + vec_idx * 1024;

    // Step 1: Unpack to shared memory using template function
    unpack_generic_32<BIT_WIDTH>(vec_encoded, sm_arr);

    __syncthreads();

    // Step 2: Prefix sum (delta decoding) from shared memory to output
    d_unrsum_32(sm_arr, vec_output, vec_base);
}

/**
 * Single-vector template delta decode kernel for verification
 */
template<int BIT_WIDTH>
__global__ void delta_decode_32_single_kernel(
    const uint32_t* __restrict__ encoded,
    const uint32_t* __restrict__ base,
    uint32_t* __restrict__ output
) {
    __shared__ uint32_t sm_arr[1024];

    // Step 1: Unpack to shared memory using template function
    unpack_generic_32<BIT_WIDTH>(encoded, sm_arr);

    __syncthreads();

    // Step 2: Prefix sum (delta decoding) from shared memory to output
    d_unrsum_32(sm_arr, output, base);
}

// Dispatch function to select appropriate kernel based on bitwidth (precompiled)
void launch_delta_decode_32(
    int bw,
    const uint32_t* encoded,
    const uint32_t* base,
    uint32_t* output,
    const size_t* vec_offsets,
    size_t num_vecs,
    cudaStream_t stream = 0
) {
    dim3 grid(num_vecs);
    dim3 block(32);

    switch (bw) {
        case 0:  delta_decode_32_bw0_kernel<<<grid, block, 0, stream>>>(base, output, num_vecs); break;
        case 1:  delta_decode_32_kernel<1><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 2:  delta_decode_32_kernel<2><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 3:  delta_decode_32_kernel<3><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 4:  delta_decode_32_kernel<4><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 5:  delta_decode_32_kernel<5><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 6:  delta_decode_32_kernel<6><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 7:  delta_decode_32_kernel<7><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 8:  delta_decode_32_kernel<8><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 9:  delta_decode_32_kernel<9><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 10: delta_decode_32_kernel<10><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 11: delta_decode_32_kernel<11><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 12: delta_decode_32_kernel<12><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 13: delta_decode_32_kernel<13><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 14: delta_decode_32_kernel<14><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 15: delta_decode_32_kernel<15><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 16: delta_decode_32_kernel<16><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 17: delta_decode_32_kernel<17><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 18: delta_decode_32_kernel<18><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 19: delta_decode_32_kernel<19><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 20: delta_decode_32_kernel<20><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 21: delta_decode_32_kernel<21><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 22: delta_decode_32_kernel<22><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 23: delta_decode_32_kernel<23><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 24: delta_decode_32_kernel<24><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 25: delta_decode_32_kernel<25><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 26: delta_decode_32_kernel<26><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 27: delta_decode_32_kernel<27><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 28: delta_decode_32_kernel<28><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 29: delta_decode_32_kernel<29><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 30: delta_decode_32_kernel<30><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 31: delta_decode_32_kernel<31><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 32: delta_decode_32_bw32_kernel<<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        default: break;
    }
}

BenchmarkResult benchmark_delta_32(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = (mode == BenchmarkMode::kAggregate) ? "DeltaAgg" : "Delta";
    result.dataset = extract_dataset_name(data_file);

    std::cout << "  Starting Delta32 benchmark..." << std::endl;

    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        std::cerr << "  Failed to load data file" << std::endl;
        result.verified = false;
        return result;
    }
    std::cout << "  Loaded " << n_tup << " tuples" << std::endl;

    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    // Allocate buffers
    // Worst case: bw=32 => 1024 uint32_t words per vector => aligned_n_tup words total
    size_t max_encoded_size = aligned_n_tup;
    uint32_t* encoded_data = new uint32_t[max_encoded_size]();
    uint32_t* base_values = new uint32_t[32 * num_vecs]();
    uint8_t* bitwidths = new uint8_t[num_vecs];
    size_t* vec_offsets = new size_t[num_vecs + 1];
    // NOTE: Delta encoding needs a per-vector transposed buffer, but we don't need
    // to keep the full transposed reference for verification (we verify against aligned_data).

    // Encode using official FLS library functions (same as fastlanes_bench_delta.cu):
    // transpose_i -> unrsum -> pack
    auto encode_start = std::chrono::high_resolution_clock::now();

    size_t encoded_offset = 0;
    for (size_t v = 0; v < num_vecs; v++) {
        vec_offsets[v] = encoded_offset;

        uint32_t* vec_in = aligned_data + v * FLS_VEC_SIZE;
        uint32_t* vec_base = base_values + v * 32;

        // Step 1: Transpose using official library function
        // Output is in "04261537" interleaved format (same as FLS_OFFSETS_32)
        uint32_t vec_transposed[FLS_VEC_SIZE];
        generated::transpose::fallback::scalar::transpose_i(vec_in, vec_transposed);

        // Step 2: Compute deltas using official library function
        // Input/output both in "04261537" format
        uint32_t deltas[FLS_VEC_SIZE];
        generated::unrsum::fallback::scalar::unrsum(vec_transposed, deltas);

        // Step 3: Extract base values (first 32 values of transposed array)
        // These are the values at positions 0..31 which are the first values in each lane
        std::memcpy(vec_base, vec_transposed, sizeof(uint32_t) * 32);

        // Calculate bitwidth from deltas
        uint32_t max_delta = 0;
        for (int i = 0; i < FLS_VEC_SIZE; i++) {
            if (deltas[i] > max_delta) max_delta = deltas[i];
        }
        uint8_t bw = (max_delta == 0) ? 0 : (32 - __builtin_clz(max_delta));
        bitwidths[v] = bw;

        // Step 4: Pack using official library function
        if (bw > 0) {
            generated::pack::fallback::scalar::pack(deltas, encoded_data + encoded_offset, bw);
            encoded_offset += (bw * FLS_VEC_SIZE) / 32;
        }
        // bw == 0: no data needed
    }
    vec_offsets[num_vecs] = encoded_offset;

    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Upload to GPU
    uint32_t* d_encoded;
    uint32_t* d_base;
    size_t* d_offsets;

    cudaMalloc(&d_encoded, encoded_offset * sizeof(uint32_t));
    cudaMalloc(&d_base, 32 * num_vecs * sizeof(uint32_t));
    cudaMalloc(&d_offsets, (num_vecs + 1) * sizeof(size_t));

    cudaMemcpy(d_encoded, encoded_data, encoded_offset * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base_values, 32 * num_vecs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, vec_offsets, (num_vecs + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Print bitwidth distribution for debugging
    int bw_counts[33] = {0};
    for (size_t v = 0; v < num_vecs; v++) {
        bw_counts[bitwidths[v]]++;
    }
    std::cout << "  Bitwidth distribution: ";
    for (int bw = 0; bw <= 32; bw++) {
        if (bw_counts[bw] > 0) {
            std::cout << "bw" << bw << "=" << bw_counts[bw] << " ";
        }
    }
    std::cout << std::endl;

    // Upload bitwidths to GPU for batch runtime kernel
    uint8_t* d_bitwidths;
    cudaMalloc(&d_bitwidths, num_vecs * sizeof(uint8_t));
    cudaMemcpy(d_bitwidths, bitwidths, num_vecs * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(num_vecs);
    dim3 block(32);
    double avg_time_ms = 0.0;
    bool verified = true;

    if (mode == BenchmarkMode::kAggregate) {
        unsigned long long* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(unsigned long long));
        cudaMemset(d_sum, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        for (int w = 0; w < warmup_count; w++) {
            delta_32_sum_fused_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int t = 0; t < num_trials; t++) {
            delta_32_sum_fused_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        avg_time_ms = elapsed_ms / num_trials;

        if (verify) {
            const unsigned long long expected = cpu_sum_u64(aligned_data, aligned_n_tup);

            cudaMemset(d_sum, 0, sizeof(unsigned long long));
            delta_32_sum_fused_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
            cudaDeviceSynchronize();

            unsigned long long actual = 0;
            cudaMemcpy(&actual, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            verified = (actual == expected);
            if (!verified) {
                std::cerr << "DeltaAgg sum mismatch: expected " << expected
                          << ", got " << actual << std::endl;
            }
        }

        cudaFree(d_sum);
    } else {
        uint32_t* d_output_linear = nullptr;
        cudaMalloc(&d_output_linear, aligned_n_tup * sizeof(uint32_t));
        cudaDeviceSynchronize();

        for (int w = 0; w < warmup_count; w++) {
            delta_decode_32_batch_runtime_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear, d_offsets, d_bitwidths, num_vecs);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int t = 0; t < num_trials; t++) {
            delta_decode_32_batch_runtime_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear, d_offsets, d_bitwidths, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        avg_time_ms = elapsed_ms / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint32_t first_error_expected = 0, first_error_actual = 0;
            verified = verify_device_output_linear<uint32_t>(
                d_output_linear, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!verified) {
                size_t bad_vec_idx = first_error_idx / 1024;
                std::cerr << "Delta32 verify failed at index " << first_error_idx
                          << " (vec " << bad_vec_idx << ", bw=" << (int)bitwidths[bad_vec_idx] << ")"
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output_linear);
    }

    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = (encoded_offset + 32 * num_vecs) * sizeof(uint32_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = (mode == BenchmarkMode::kAggregate)
            ? sizeof(unsigned long long)
            : aligned_n_tup * sizeof(uint32_t);
        result.estimated_hbm_bytes = result.compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_time_ms / 1000.0);
    }
    result.verified = verified;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] base_values;
    delete[] bitwidths;
    delete[] vec_offsets;
    cudaFree(d_encoded);
    cudaFree(d_base);
    cudaFree(d_offsets);
    cudaFree(d_bitwidths);

    return result;
}

// ============================================================================
// 32-bit Delta Uniform Bitwidth Benchmark
// ============================================================================

BenchmarkResult benchmark_delta_32_uniform(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = "DeltaUniform";
    result.dataset = extract_dataset_name(data_file);

    std::cout << "  Starting DeltaUniform32 benchmark..." << std::endl;

    size_t n_tup = 0;
    uint32_t* original_data = load_binary_file<uint32_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        std::cerr << "  Failed to load data file" << std::endl;
        result.verified = false;
        return result;
    }
    std::cout << "  Loaded " << n_tup << " tuples" << std::endl;

    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint32_t* aligned_data = new uint32_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint32_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    uint32_t* base_values = new uint32_t[32 * num_vecs]();

    // First pass: compute per-vector bitwidths and find global max
    uint8_t global_bw = 0;
    {
        uint32_t vec_transposed[FLS_VEC_SIZE];
        uint32_t deltas[FLS_VEC_SIZE];
        for (size_t v = 0; v < num_vecs; v++) {
            uint32_t* vec_in = aligned_data + v * FLS_VEC_SIZE;
            uint32_t* vec_base = base_values + v * 32;

            generated::transpose::fallback::scalar::transpose_i(vec_in, vec_transposed);
            generated::unrsum::fallback::scalar::unrsum(vec_transposed, deltas);
            std::memcpy(vec_base, vec_transposed, sizeof(uint32_t) * 32);

            uint32_t max_delta = 0;
            for (int i = 0; i < FLS_VEC_SIZE; i++) {
                if (deltas[i] > max_delta) max_delta = deltas[i];
            }
            uint8_t bw = (max_delta == 0) ? 0 : (32 - __builtin_clz(max_delta));
            if (bw > global_bw) global_bw = bw;
        }
    }
    std::cout << "  Global bitwidth: " << (int)global_bw << std::endl;

    // Second pass: encode all vectors with the global bitwidth
    size_t words_per_vec = (global_bw > 0) ? ((size_t)global_bw * FLS_VEC_SIZE / 32) : 0;
    size_t total_encoded_words = words_per_vec * num_vecs;
    uint32_t* encoded_data = new uint32_t[total_encoded_words > 0 ? total_encoded_words : 1]();

    auto encode_start = std::chrono::high_resolution_clock::now();
    {
        uint32_t vec_transposed[FLS_VEC_SIZE];
        uint32_t deltas[FLS_VEC_SIZE];
        for (size_t v = 0; v < num_vecs; v++) {
            uint32_t* vec_in = aligned_data + v * FLS_VEC_SIZE;

            generated::transpose::fallback::scalar::transpose_i(vec_in, vec_transposed);
            generated::unrsum::fallback::scalar::unrsum(vec_transposed, deltas);

            if (global_bw > 0) {
                generated::pack::fallback::scalar::pack(
                    deltas, encoded_data + v * words_per_vec, global_bw);
            }
        }
    }
    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Upload to GPU
    uint32_t* d_encoded = nullptr;
    uint32_t* d_base = nullptr;
    if (total_encoded_words > 0) {
        cudaMalloc(&d_encoded, total_encoded_words * sizeof(uint32_t));
        cudaMemcpy(d_encoded, encoded_data, total_encoded_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&d_base, 32 * num_vecs * sizeof(uint32_t));
    cudaMemcpy(d_base, base_values, 32 * num_vecs * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 grid(num_vecs);
    dim3 block(32);
    double avg_time_ms = 0.0;
    bool verified = true;

    // Only g2g mode for uniform
    {
        uint32_t* d_output_linear = nullptr;
        cudaMalloc(&d_output_linear, aligned_n_tup * sizeof(uint32_t));
        cudaDeviceSynchronize();

        for (int w = 0; w < warmup_count; w++) {
            delta_decode_32_uniform_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear,
                (int)global_bw, words_per_vec, num_vecs);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int t = 0; t < num_trials; t++) {
            delta_decode_32_uniform_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear,
                (int)global_bw, words_per_vec, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        avg_time_ms = elapsed_ms / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint32_t first_error_expected = 0, first_error_actual = 0;
            verified = verify_device_output_linear<uint32_t>(
                d_output_linear, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!verified) {
                size_t bad_vec_idx = first_error_idx / 1024;
                std::cerr << "DeltaUniform32 verify failed at index " << first_error_idx
                          << " (vec " << bad_vec_idx << ", global_bw=" << (int)global_bw << ")"
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output_linear);
    }

    size_t compressed_size = (total_encoded_words + 32 * num_vecs) * sizeof(uint32_t);
    result.original_size = n_tup * sizeof(uint32_t);
    result.compressed_size = compressed_size;
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = aligned_n_tup * sizeof(uint32_t);
        result.estimated_hbm_bytes = compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_time_ms / 1000.0);
    }
    result.verified = verified;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] base_values;
    if (d_encoded) cudaFree(d_encoded);
    cudaFree(d_base);

    return result;
}

// ============================================================================
// 64-bit Delta Decoding Kernels
// ============================================================================

/**
 * Generic 64-bit delta decoding kernel (unpack + prefix sum)
 * Each block processes one 1024-element vector
 */
template<int BIT_WIDTH>
__global__ void delta_decode_64_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint64_t sm_arr[1024];

    // Get encoded data pointer for this vector
    const uint64_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint64_t* vec_base = base + vec_idx * 32;
    uint64_t* vec_output = output + vec_idx * 1024;

    // Step 1: Unpack to shared memory
    generated::unpack::cuda::normal_64::unpack_generic_64<BIT_WIDTH>(vec_encoded, sm_arr);

    __syncthreads();

    // Step 2: Prefix sum (delta decoding) from shared memory to output
    d_unrsum_64(sm_arr, vec_output, vec_base);
}

/**
 * Single-vector delta decoding kernel for per-vector verification
 * encoded/base/output point directly to the single vector's data
 */
template<int BIT_WIDTH>
__global__ void delta_decode_64_single_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output
) {
    __shared__ uint64_t sm_arr[1024];

    // Step 1: Unpack to shared memory
    generated::unpack::cuda::normal_64::unpack_generic_64<BIT_WIDTH>(encoded, sm_arr);

    __syncthreads();

    // Step 2: Prefix sum (delta decoding) from shared memory to output
    d_unrsum_64(sm_arr, output, base);
}

// ============================================================================
// Runtime bitwidth batch kernel for FAIR benchmarking
// Each block reads its own bitwidth from d_bitwidths and does runtime switch
// ============================================================================

/**
 * TRUE RUNTIME unpack function for 64-bit values
 * bitwidth is a runtime variable, NOT compile-time template parameter
 * This is the FAIR version - no template specialization
 */
__device__ void unpack_runtime_64(
    int bw,  // Runtime bitwidth
    const uint64_t* __restrict__ encoded,
    uint64_t* __restrict__ output
) {
    if (bw <= 0) {
        // bw=0: all zeros
        int tid = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            output[tid + FLS_OFFSETS_32[j]] = 0ULL;
        }
        return;
    }

    if (bw == 64) {
        // bw=64: Special case - data is stored as encoded[tid * 32 + j]
        // No bit packing, just reorder from linear to interleaved format
        int tid = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            output[tid + FLS_OFFSETS_32[j]] = encoded[tid * 32 + j];
        }
        return;
    }

    // Runtime mask computation
    uint64_t mask = (bw >= 64) ? ~0ULL : ((1ULL << bw) - 1ULL);

    int tid = threadIdx.x;  // Thread index (0-31)
    constexpr int VALUES_PER_THREAD = 32;

    // Calculate starting bit position for this thread
    int64_t thread_start_bit = static_cast<int64_t>(tid) * bw * VALUES_PER_THREAD;

    // Reinterpret encoded data as uint32_t for easier bit manipulation
    const uint32_t* encoded32 = reinterpret_cast<const uint32_t*>(encoded);

    // Extract 32 values for this thread
    for (int j = 0; j < VALUES_PER_THREAD; j++) {
        int64_t bit_pos = thread_start_bit + static_cast<int64_t>(j) * bw;
        int word_idx = bit_pos >> 5;  // bit_pos / 32
        int bit_offset = bit_pos & 31;  // bit_pos % 32

        // Load two consecutive 32-bit words and combine into 64-bit
        uint64_t combined = (static_cast<uint64_t>(__ldg(&encoded32[word_idx + 1])) << 32) |
                           __ldg(&encoded32[word_idx]);

        uint64_t val = (combined >> bit_offset) & mask;

        // Handle 3-word case for bw > 32 and misaligned
        if (bw > 32 && bit_offset > 0 && (64 - bit_offset) < bw) {
            int bits_from_first_two = 64 - bit_offset;
            uint64_t third_word = __ldg(&encoded32[word_idx + 2]);
            val |= (third_word << bits_from_first_two) & mask;
        }

        // Output in interleaved order using FLS_OFFSETS_32
        output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

/**
 * Fair batch delta decode kernel with TRUE RUNTIME bitwidth
 * Each block handles one vector, reading its bitwidth from d_bitwidths array
 * Uses runtime unpack function - NO template specialization
 */
__global__ void delta_decode_64_batch_runtime_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    const uint8_t* __restrict__ bitwidths,  // Per-vector bitwidths
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint64_t deltas[1024];
    __shared__ uint64_t values_transposed[1024];

    // Read this vector's bitwidth at RUNTIME
    int bw = bitwidths[vec_idx];

    const uint64_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint64_t* vec_base = base + vec_idx * 32;
    uint64_t* vec_output = output + vec_idx * 1024;

    // Handle bw=0 case (all deltas are zero): linear output is base value per lane.
    if (bw == 0) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const int k = threadIdx.x + 32 * j;
            const int t = k >> 5;  // /32
            vec_output[k] = vec_base[t];
        }
        return;
    }

    // Step 1: Unpack to shared memory using LIBRARY function (fair comparison with BitPack)
    generated::unpack::cuda::normal_64::unpack(vec_encoded, deltas, bw);

    __syncthreads();

    // Step 2: Prefix sum (delta decoding) in FastLanes transposed order to shared memory
    d_unrsum_64(deltas, values_transposed, vec_base);
    __syncthreads();

    // Step 3: Inverse transpose to linear order and store to global memory
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const int k = threadIdx.x + 32 * j;
        const int t = k >> 5;  // /32
        const int jj = k & 31;
        vec_output[k] = values_transposed[t + FLS_OFFSETS_32[jj]];
    }
}

// ============================================================================
// End of runtime batch kernel
// ============================================================================

// Specialized kernel for bitwidth 0 (all zeros - only base values)
// Output in interleaved format
__global__ void delta_decode_64_bw0_kernel(
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const uint64_t* vec_base = base + vec_idx * 32;
    uint64_t* vec_output = output + vec_idx * 1024;

    int tid = threadIdx.x % 32;
    uint64_t val = vec_base[tid];

    // All values in this lane are the same (base value)
    // Output in interleaved format using FLS_OFFSETS_32
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        vec_output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

// Single-vector bw0 kernel for verification
__global__ void delta_decode_64_bw0_single_kernel(
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output
) {
    int tid = threadIdx.x % 32;
    uint64_t val = base[tid];

    // Output in interleaved format
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        output[tid + FLS_OFFSETS_32[j]] = val;
    }
}

/**
 * Specialized kernel for bitwidth 64 (full width - no compression, just copy)
 * Encoded data is stored as: thread t's values are at out[t*32 ... t*32+31]
 * Output in interleaved format after delta decoding
 */
__global__ void delta_decode_64_bw64_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output,
    const size_t* __restrict__ vec_offsets,
    size_t num_vecs
) {
    size_t vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    __shared__ uint64_t sm_arr[1024];

    const uint64_t* vec_encoded = encoded + vec_offsets[vec_idx];
    const uint64_t* vec_base = base + vec_idx * 32;
    uint64_t* vec_output = output + vec_idx * 1024;

    int tid = threadIdx.x % 32;

    // Step 1: Copy from encoded to shared memory in interleaved format
    // Encoded layout: thread t's values are at vec_encoded[t*32 + j] for j=0..31
    // Output to interleaved format: sm_arr[tid + FLS_OFFSETS_32[j]]
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        sm_arr[tid + FLS_OFFSETS_32[j]] = vec_encoded[tid * 32 + j];
    }

    __syncthreads();

    // Step 2: Prefix sum (delta decoding)
    d_unrsum_64(sm_arr, vec_output, vec_base);
}

// Single-vector bw64 kernel for verification
__global__ void delta_decode_64_bw64_single_kernel(
    const uint64_t* __restrict__ encoded,
    const uint64_t* __restrict__ base,
    uint64_t* __restrict__ output
) {
    __shared__ uint64_t sm_arr[1024];

    int tid = threadIdx.x % 32;

    // Copy from encoded to shared memory in interleaved format
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        sm_arr[tid + FLS_OFFSETS_32[j]] = encoded[tid * 32 + j];
    }

    __syncthreads();

    // Prefix sum (delta decoding)
    d_unrsum_64(sm_arr, output, base);
}

// Function pointer type for dispatch
typedef void (*DeltaDecode64Func)(const uint64_t*, const uint64_t*, uint64_t*, const size_t*, size_t);

// Dispatch function to select appropriate kernel based on bitwidth
void launch_delta_decode_64(
    int bw,
    const uint64_t* encoded,
    const uint64_t* base,
    uint64_t* output,
    const size_t* vec_offsets,
    size_t num_vecs,
    cudaStream_t stream = 0
) {
    dim3 grid(num_vecs);
    dim3 block(32);

    switch (bw) {
        case 0:  delta_decode_64_bw0_kernel<<<grid, block, 0, stream>>>(base, output, num_vecs); break;
        case 1:  delta_decode_64_kernel<1><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 2:  delta_decode_64_kernel<2><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 3:  delta_decode_64_kernel<3><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 4:  delta_decode_64_kernel<4><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 5:  delta_decode_64_kernel<5><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 6:  delta_decode_64_kernel<6><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 7:  delta_decode_64_kernel<7><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 8:  delta_decode_64_kernel<8><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 9:  delta_decode_64_kernel<9><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 10: delta_decode_64_kernel<10><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 11: delta_decode_64_kernel<11><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 12: delta_decode_64_kernel<12><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 13: delta_decode_64_kernel<13><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 14: delta_decode_64_kernel<14><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 15: delta_decode_64_kernel<15><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 16: delta_decode_64_kernel<16><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 17: delta_decode_64_kernel<17><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 18: delta_decode_64_kernel<18><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 19: delta_decode_64_kernel<19><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 20: delta_decode_64_kernel<20><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 21: delta_decode_64_kernel<21><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 22: delta_decode_64_kernel<22><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 23: delta_decode_64_kernel<23><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 24: delta_decode_64_kernel<24><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 25: delta_decode_64_kernel<25><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 26: delta_decode_64_kernel<26><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 27: delta_decode_64_kernel<27><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 28: delta_decode_64_kernel<28><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 29: delta_decode_64_kernel<29><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 30: delta_decode_64_kernel<30><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 31: delta_decode_64_kernel<31><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 32: delta_decode_64_kernel<32><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 33: delta_decode_64_kernel<33><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 34: delta_decode_64_kernel<34><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 35: delta_decode_64_kernel<35><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 36: delta_decode_64_kernel<36><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 37: delta_decode_64_kernel<37><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 38: delta_decode_64_kernel<38><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 39: delta_decode_64_kernel<39><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 40: delta_decode_64_kernel<40><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 41: delta_decode_64_kernel<41><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 42: delta_decode_64_kernel<42><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 43: delta_decode_64_kernel<43><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 44: delta_decode_64_kernel<44><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 45: delta_decode_64_kernel<45><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 46: delta_decode_64_kernel<46><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 47: delta_decode_64_kernel<47><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 48: delta_decode_64_kernel<48><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 49: delta_decode_64_kernel<49><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 50: delta_decode_64_kernel<50><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 51: delta_decode_64_kernel<51><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 52: delta_decode_64_kernel<52><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 53: delta_decode_64_kernel<53><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 54: delta_decode_64_kernel<54><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 55: delta_decode_64_kernel<55><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 56: delta_decode_64_kernel<56><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 57: delta_decode_64_kernel<57><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 58: delta_decode_64_kernel<58><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 59: delta_decode_64_kernel<59><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 60: delta_decode_64_kernel<60><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 61: delta_decode_64_kernel<61><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 62: delta_decode_64_kernel<62><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 63: delta_decode_64_kernel<63><<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        case 64: delta_decode_64_bw64_kernel<<<grid, block, 0, stream>>>(encoded, base, output, vec_offsets, num_vecs); break;
        default: break;
    }
}

// ============================================================================
// 64-bit Benchmark Functions
// ============================================================================

uint8_t calculate_bitwidth_64(const uint64_t* data, size_t count) {
    uint64_t max_val = 0;
    for (size_t i = 0; i < count; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    if (max_val == 0) return 0;
    return (uint8_t)(64 - __builtin_clzll(max_val));
}

// Custom pack function for 64-bit that matches GPU unpack layout
// GPU unpack outputs in FastLanes interleaved format: thread t outputs to positions t + FLS_OFFSETS_32_HOST[j]
// To get correct output after unpack, we read from the same interleaved positions
size_t pack_64_for_gpu(const uint64_t* in, uint64_t* out, uint8_t bw) {
    if (bw == 0) return 0;
    if (bw == 64) {
        // Direct copy: thread t handles values at positions t + FLS_OFFSETS_32_HOST[0..31]
        for (int t = 0; t < 32; t++) {
            for (int j = 0; j < 32; j++) {
                out[t * 32 + j] = in[t + FLS_OFFSETS_32_HOST[j]];
            }
        }
        return 1024;
    }

    // Calculate output size: 1024 values * bw bits / 64 bits per word
    size_t out_words = (1024ULL * bw + 63) / 64;
    memset(out, 0, out_words * sizeof(uint64_t));

    const uint64_t mask = (bw == 64) ? ~0ULL : ((1ULL << bw) - 1);

    // Pack in the order that GPU unpack expects with interleaved output
    // GPU thread t outputs to positions: t + FLS_OFFSETS_32_HOST[0..31]
    // So we pack values from those same positions to get correct output
    size_t bit_pos = 0;
    for (int t = 0; t < 32; t++) {
        for (int j = 0; j < 32; j++) {
            uint64_t val = in[t + FLS_OFFSETS_32_HOST[j]] & mask;

            size_t word_idx = bit_pos / 64;
            size_t bit_offset = bit_pos % 64;

            if (bit_offset + bw <= 64) {
                out[word_idx] |= (val << bit_offset);
            } else {
                // Cross word boundary
                out[word_idx] |= (val << bit_offset);
                out[word_idx + 1] |= (val >> (64 - bit_offset));
            }
            bit_pos += bw;
        }
    }

    return (1024ULL * bw + 63) / 64;
}

size_t encode_bitpack_vector_64(const uint64_t* in, uint64_t* out, uint8_t& bitwidth) {
    bitwidth = calculate_bitwidth_64(in, FLS_VEC_SIZE);
    if (bitwidth == 0) return 0;

    // 64-bit unpack outputs in FastLanes "04261537" interleaved format, so we must
    // transpose input to that format before packing to make decode comparable to BitPack32.
    uint64_t vec_transposed[FLS_VEC_SIZE];
    for (int t = 0; t < 32; t++) {
        for (int j = 0; j < 32; j++) {
            vec_transposed[t + FLS_OFFSETS_32_HOST[j]] = in[t * 32 + j];
        }
    }

    return pack_64_for_gpu(vec_transposed, out, bitwidth);
}

BenchmarkResult benchmark_bitpack_64(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = (mode == BenchmarkMode::kAggregate) ? "BitPack64Agg" : "BitPack64";
    result.dataset = extract_dataset_name(data_file);

    size_t n_tup = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        result.verified = false;
        return result;
    }

    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint64_t* aligned_data = new uint64_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint64_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    size_t max_encoded_size = aligned_n_tup;
    uint64_t* encoded_data = new uint64_t[max_encoded_size]();
    uint8_t* bitwidths = new uint8_t[num_vecs];
    size_t* vec_offsets = new size_t[num_vecs];

    auto encode_start = std::chrono::high_resolution_clock::now();

    size_t encoded_offset = 0;
    for (size_t v = 0; v < num_vecs; v++) {
        vec_offsets[v] = encoded_offset;
        size_t packed_size = encode_bitpack_vector_64(
            aligned_data + v * FLS_VEC_SIZE,
            encoded_data + encoded_offset,
            bitwidths[v]
        );
        encoded_offset += packed_size;
    }

    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Upload to GPU
    uint64_t* d_encoded;
    size_t* d_offsets;
    uint8_t* d_bitwidths;

    cudaMalloc(&d_encoded, encoded_offset * sizeof(uint64_t));
    cudaMalloc(&d_offsets, num_vecs * sizeof(size_t));
    cudaMalloc(&d_bitwidths, num_vecs * sizeof(uint8_t));

    cudaMemcpy(d_encoded, encoded_data, encoded_offset * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, vec_offsets, num_vecs * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitwidths, bitwidths, num_vecs * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float avg_decode_time_ms = 0.0f;
    bool correct = true;

    if (mode == BenchmarkMode::kAggregate) {
        unsigned long long* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(unsigned long long));
        cudaMemset(d_sum, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        for (int i = 0; i < warmup_count; i++) {
            bitpack_64_sum_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int trial = 0; trial < num_trials; trial++) {
            bitpack_64_sum_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        avg_decode_time_ms = total_time / num_trials;

        if (verify) {
            const unsigned long long expected = cpu_sum_u64(aligned_data, aligned_n_tup);

            cudaMemset(d_sum, 0, sizeof(unsigned long long));
            bitpack_64_sum_kernel<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_offsets, d_bitwidths, num_vecs, d_sum);
            cudaDeviceSynchronize();

            unsigned long long actual = 0;
            cudaMemcpy(&actual, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            correct = (actual == expected);
            if (!correct) {
                std::cerr << "BitPack64Agg sum mismatch: expected " << expected
                          << ", got " << actual << std::endl;
            }
        }

        cudaFree(d_sum);
    } else {
        uint64_t* d_output_linear = nullptr;
        cudaMalloc(&d_output_linear, aligned_n_tup * sizeof(uint64_t));
        cudaDeviceSynchronize();

        for (int i = 0; i < warmup_count; i++) {
            batch_unpack_64<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_output_linear, d_offsets, d_bitwidths, num_vecs);
            cudaDeviceSynchronize();
        }

        cudaEventRecord(start);
        for (int trial = 0; trial < num_trials; trial++) {
            batch_unpack_64<<<num_vecs, FLS_THREADS>>>(
                d_encoded, d_output_linear, d_offsets, d_bitwidths, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_time;
        cudaEventElapsedTime(&total_time, start, stop);
        avg_decode_time_ms = total_time / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint64_t first_error_expected = 0, first_error_actual = 0;
            correct = verify_device_output_linear<uint64_t>(
                d_output_linear, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!correct) {
                std::cerr << "BitPack64 verify failed at index " << first_error_idx
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output_linear);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.original_size = n_tup * sizeof(uint64_t);
    result.compressed_size = encoded_offset * sizeof(uint64_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_decode_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_decode_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = (mode == BenchmarkMode::kAggregate)
            ? sizeof(unsigned long long)
            : aligned_n_tup * sizeof(uint64_t);
        result.estimated_hbm_bytes = result.compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_decode_time_ms / 1000.0);
    }
    result.verified = correct;

    // Cleanup
    cudaFree(d_encoded);
    cudaFree(d_offsets);
    cudaFree(d_bitwidths);
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] bitwidths;
    delete[] vec_offsets;

    return result;
}

BenchmarkResult benchmark_delta_64(
    const std::string& data_file,
    int num_trials,
    int warmup_count,
    BenchmarkMode mode,
    bool verify
) {
    BenchmarkResult result;
    result.framework = "FLS-GPU";
    result.algorithm = (mode == BenchmarkMode::kAggregate) ? "Delta64Agg" : "Delta64";
    result.dataset = extract_dataset_name(data_file);

    std::cout << "  Starting Delta64 benchmark..." << std::endl;

    size_t n_tup = 0;
    uint64_t* original_data = load_binary_file<uint64_t>(data_file.c_str(), n_tup);
    if (!original_data) {
        std::cerr << "  Failed to load data file" << std::endl;
        result.verified = false;
        return result;
    }
    std::cout << "  Loaded " << n_tup << " tuples" << std::endl;

    size_t aligned_n_tup = ((n_tup + FLS_VEC_SIZE - 1) / FLS_VEC_SIZE) * FLS_VEC_SIZE;
    size_t num_vecs = aligned_n_tup / FLS_VEC_SIZE;

    uint64_t* aligned_data = new uint64_t[aligned_n_tup]();
    memcpy(aligned_data, original_data, n_tup * sizeof(uint64_t));
    for (size_t i = n_tup; i < aligned_n_tup; i++) {
        aligned_data[i] = original_data[n_tup - 1];
    }

    // Allocate buffers
    // Worst case: bw=64 => 1024 uint64_t words per vector => aligned_n_tup words total
    size_t max_encoded_size = aligned_n_tup;
    uint64_t* encoded_data = new uint64_t[max_encoded_size]();
    uint64_t* base_values = new uint64_t[32 * num_vecs]();
    uint8_t* bitwidths = new uint8_t[num_vecs];
    size_t* vec_offsets = new size_t[num_vecs + 1];

    // Encode: transpose -> diff -> pack
    auto encode_start = std::chrono::high_resolution_clock::now();

    size_t encoded_offset = 0;
    for (size_t v = 0; v < num_vecs; v++) {
        vec_offsets[v] = encoded_offset;

        uint64_t* vec_in = aligned_data + v * FLS_VEC_SIZE;
        uint64_t* vec_base = base_values + v * 32;

        // Step 1: Transpose using FastLanes interleaved layout
        // Each thread t gets values from input[t*32 ... t*32+31]
        // Output to interleaved positions: t + FLS_OFFSETS_32_HOST[j]
        uint64_t vec_transposed[FLS_VEC_SIZE];
        for (int t = 0; t < 32; t++) {
            for (int j = 0; j < 32; j++) {
                vec_transposed[t + FLS_OFFSETS_32_HOST[j]] = vec_in[t * 32 + j];
            }
        }

        // Step 2: Extract base values and compute deltas in interleaved order
        uint64_t deltas[FLS_VEC_SIZE];
        uint64_t max_delta = 0;
        for (int t = 0; t < 32; t++) {
            // Base value is first value in interleaved order (at offset 0)
            vec_base[t] = vec_transposed[t + FLS_OFFSETS_32_HOST[0]];  // = vec_transposed[t]
            deltas[t + FLS_OFFSETS_32_HOST[0]] = 0;  // First delta is 0

            for (int j = 1; j < 32; j++) {
                int curr_idx = t + FLS_OFFSETS_32_HOST[j];
                int prev_idx = t + FLS_OFFSETS_32_HOST[j - 1];
                uint64_t delta = vec_transposed[curr_idx] - vec_transposed[prev_idx];
                deltas[curr_idx] = delta;
                if (delta > max_delta) max_delta = delta;
            }
        }

        // Calculate bitwidth
        uint8_t bw = (max_delta == 0) ? 0 : (64 - __builtin_clzll(max_delta));
        bitwidths[v] = bw;

        // Step 3: Pack deltas - read in order matching GPU unpack
        // GPU thread t unpacks values j=0..31 sequentially, outputting to t + FLS_OFFSETS_32_HOST[j]
        // So we pack: for each thread t, pack deltas[t + FLS_OFFSETS_32_HOST[0..31]] in sequence
        if (bw > 0 && bw < 64) {
            size_t packed_words = (1024ULL * bw + 63) / 64;
            memset(encoded_data + encoded_offset, 0, packed_words * sizeof(uint64_t));

            const uint64_t mask = (1ULL << bw) - 1;
            size_t bit_pos = 0;
            for (int t = 0; t < 32; t++) {
                for (int j = 0; j < 32; j++) {
                    uint64_t val = deltas[t + FLS_OFFSETS_32_HOST[j]] & mask;
                    size_t word_idx = bit_pos / 64;
                    int bit_offset = bit_pos % 64;

                    encoded_data[encoded_offset + word_idx] |= (val << bit_offset);
                    if (bit_offset + bw > 64) {
                        encoded_data[encoded_offset + word_idx + 1] |= (val >> (64 - bit_offset));
                    }
                    bit_pos += bw;
                }
            }
            encoded_offset += packed_words;
        } else if (bw == 64) {
            // For 64-bit, pack in the order matching unpack read order
            for (int t = 0; t < 32; t++) {
                for (int j = 0; j < 32; j++) {
                    encoded_data[encoded_offset + t * 32 + j] = deltas[t + FLS_OFFSETS_32_HOST[j]];
                }
            }
            encoded_offset += FLS_VEC_SIZE;
        }
        // bw == 0: no data needed
    }
    vec_offsets[num_vecs] = encoded_offset;

    auto encode_end = std::chrono::high_resolution_clock::now();
    double encode_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();

    // Upload to GPU
    uint64_t* d_encoded;
    uint64_t* d_base;
    size_t* d_offsets;

    cudaMalloc(&d_encoded, encoded_offset * sizeof(uint64_t));
    cudaMalloc(&d_base, 32 * num_vecs * sizeof(uint64_t));
    cudaMalloc(&d_offsets, (num_vecs + 1) * sizeof(size_t));

    cudaMemcpy(d_encoded, encoded_data, encoded_offset * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base_values, 32 * num_vecs * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, vec_offsets, (num_vecs + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Upload bitwidths to GPU for FAIR runtime benchmark
    uint8_t* d_bitwidths;
    cudaMalloc(&d_bitwidths, num_vecs * sizeof(uint8_t));
    cudaMemcpy(d_bitwidths, bitwidths, num_vecs * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Print bitwidth distribution for debugging
    int bw_counts[65] = {0};
    for (size_t v = 0; v < num_vecs; v++) {
        bw_counts[bitwidths[v]]++;
    }
    std::cout << "  Bitwidth distribution: ";
    for (int bw = 0; bw <= 64; bw++) {
        if (bw_counts[bw] > 0) {
            std::cout << "bw" << bw << "=" << bw_counts[bw] << " ";
        }
    }
    std::cout << std::endl;

    // Warmup with FAIR runtime kernel (each block reads its own bitwidth)
    dim3 grid(num_vecs);
    dim3 block(32);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double avg_time_ms = 0.0;
    bool verified = true;

    if (mode == BenchmarkMode::kAggregate) {
        unsigned long long* d_sum = nullptr;
        cudaMalloc(&d_sum, sizeof(unsigned long long));
        cudaMemset(d_sum, 0, sizeof(unsigned long long));
        cudaDeviceSynchronize();

        for (int w = 0; w < warmup_count; w++) {
            delta_64_sum_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int t = 0; t < num_trials; t++) {
            delta_64_sum_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        avg_time_ms = elapsed_ms / num_trials;

        if (verify) {
            const unsigned long long expected = cpu_sum_u64(aligned_data, aligned_n_tup);

            cudaMemset(d_sum, 0, sizeof(unsigned long long));
            delta_64_sum_kernel<<<grid, block>>>(
                d_encoded, d_base, d_offsets, d_bitwidths, num_vecs, d_sum);
            cudaDeviceSynchronize();

            unsigned long long actual = 0;
            cudaMemcpy(&actual, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            verified = (actual == expected);
            if (!verified) {
                std::cerr << "Delta64Agg sum mismatch: expected " << expected
                          << ", got " << actual << std::endl;
            }
        }

        cudaFree(d_sum);
    } else {
        uint64_t* d_output_linear = nullptr;
        cudaMalloc(&d_output_linear, aligned_n_tup * sizeof(uint64_t));
        cudaDeviceSynchronize();

        for (int w = 0; w < warmup_count; w++) {
            delta_decode_64_batch_runtime_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear, d_offsets, d_bitwidths, num_vecs);
        }
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        for (int t = 0; t < num_trials; t++) {
            delta_decode_64_batch_runtime_kernel<<<grid, block>>>(
                d_encoded, d_base, d_output_linear, d_offsets, d_bitwidths, num_vecs);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        avg_time_ms = elapsed_ms / num_trials;

        if (verify) {
            size_t first_error_idx = 0;
            uint64_t first_error_expected = 0, first_error_actual = 0;
            verified = verify_device_output_linear<uint64_t>(
                d_output_linear, aligned_data, n_tup,
                first_error_idx, first_error_expected, first_error_actual);
            if (!verified) {
                size_t bad_vec_idx = first_error_idx / 1024;
                std::cerr << "Delta64 verify failed at index " << first_error_idx
                          << " (vec " << bad_vec_idx << ", bw=" << (int)bitwidths[bad_vec_idx] << ")"
                          << ": expected " << first_error_expected
                          << ", got " << first_error_actual << std::endl;
            }
        }

        cudaFree(d_output_linear);
    }

    result.original_size = n_tup * sizeof(uint64_t);
    result.compressed_size = (encoded_offset + 32 * num_vecs) * sizeof(uint64_t);
    result.compression_ratio = (double)result.original_size / result.compressed_size;
    result.compress_time_ms = encode_time_ms;
    result.decompress_time_ms = avg_time_ms;
    result.compress_throughput_gbps = (result.original_size / 1e9) / (encode_time_ms / 1000.0);
    result.decompress_throughput_gbps = (result.original_size / 1e9) / (avg_time_ms / 1000.0);
    {
        const size_t timed_output_bytes = (mode == BenchmarkMode::kAggregate)
            ? sizeof(unsigned long long)
            : aligned_n_tup * sizeof(uint64_t);
        result.estimated_hbm_bytes = result.compressed_size + timed_output_bytes;
        result.estimated_hbm_throughput_gbps =
            (result.estimated_hbm_bytes / 1e9) / (avg_time_ms / 1000.0);
    }
    result.verified = verified;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup
    delete[] original_data;
    delete[] aligned_data;
    delete[] encoded_data;
    delete[] base_values;
    delete[] bitwidths;
    delete[] vec_offsets;
    cudaFree(d_encoded);
    cudaFree(d_base);
    cudaFree(d_offsets);
    cudaFree(d_bitwidths);

    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -d, --data_dir <path>    Data directory\n"
              << "  -o, --output <file>      Output CSV file (default: reports/fastlanes_results.csv)\n"
              << "  -a, --algorithm <alg>    Algorithm: all|bitpack|delta (default: all)\n"
              << "  -m, --mode <mode>        Mode: g2g|agg (default: g2g)\n"
              << "      --no-verify          Skip verification\n"
              << "  -f, --file <filename>    Specific dataset file\n"
              << "  -n, --trials <num>       Number of trials (default: 10)\n"
              << "  -w, --warmup <num>       Warmup iterations (default: 3)\n"
              << "  -g, --gpu <id>           GPU device ID (default: 0)\n"
              << "  -b, --bits <32|64>       Force data type (default: auto-detect)\n"
              << "  -h, --help               Show this help\n";
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/data/sosd/";
    std::string output_file = "reports/fastlanes_results.csv";
    std::string algorithm = "all";
    std::string mode_str = "g2g";
    std::vector<std::string> specific_files;
    int num_trials = 10;
    int warmup_count = 3;
    int gpu_id = 0;
    int force_bits = 0;
    bool verify = true;
    bool gpu_id_provided = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-d" || arg == "--data_dir") && i + 1 < argc) {
            data_dir = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_file = argv[++i];
        } else if ((arg == "-a" || arg == "--algorithm") && i + 1 < argc) {
            algorithm = argv[++i];
        } else if ((arg == "-m" || arg == "--mode") && i + 1 < argc) {
            mode_str = argv[++i];
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            specific_files.push_back(argv[++i]);
        } else if ((arg == "-n" || arg == "--trials") && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup_count = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
            gpu_id_provided = true;
        } else if ((arg == "-b" || arg == "--bits") && i + 1 < argc) {
            force_bits = std::stoi(argv[++i]);
        } else if (arg == "--no-verify") {
            verify = false;
        }
    }

    if (!gpu_id_provided) {
        const char* env_device = std::getenv("DEVICE");
        if (env_device && std::strlen(env_device) > 0) {
            try {
                gpu_id = std::stoi(env_device);
            } catch (...) {
                std::cerr << "Invalid DEVICE env var: " << env_device << "\n";
                return 1;
            }
        }
    }

    BenchmarkMode mode = parse_mode(mode_str);

    cudaSetDevice(gpu_id);

    {
        std::filesystem::path out_path(output_file);
        if (out_path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(out_path.parent_path(), ec);
            if (ec) {
                std::cerr << "Failed to create output directory: " << out_path.parent_path()
                          << " (" << ec.message() << ")\n";
                return 1;
            }
        }
    }

    std::vector<std::string> datasets;
    if (!specific_files.empty()) {
        datasets = specific_files;
    } else {
        std::vector<std::string> default_datasets = {
            "linear_200M_uint32.bin",
            "normal_200M_uint32.bin"
        };
        for (const auto& ds : default_datasets) {
            datasets.push_back(data_dir + ds);
        }
    }

    std::ofstream csv_file;
    bool file_exists = std::ifstream(output_file).good();
    csv_file.open(output_file, std::ios::app);

    if (!file_exists) {
        write_csv_header(csv_file);
    }

    std::cout << "FLS-GPU Benchmark (Using Library Kernels)\n";
    std::cout << "==========================================\n";
    std::cout << "Vector size: " << FLS_VEC_SIZE << " values\n";
    std::cout << "Threads/block: " << FLS_THREADS << "\n";
    std::cout << "Algorithms: " << algorithm << "\n";
    std::cout << "Mode: " << mode_str << (verify ? " (verify)" : " (no-verify)") << "\n";
    std::cout << "Trials: " << num_trials << ", Warmup: " << warmup_count << "\n\n";

    for (const auto& data_file : datasets) {
        std::cout << "Processing: " << data_file << std::endl;

        std::ifstream test_file(data_file);
        if (!test_file.good()) {
            std::cerr << "  File not found, skipping.\n";
            continue;
        }
        test_file.close();

        bool use_64bit = (force_bits == 64) || (force_bits == 0 && is_64bit_dataset(data_file));

        if (use_64bit) {
            std::cout << "  [64-bit mode]" << std::endl;

            if (algorithm == "all" || algorithm == "bitpack") {
                std::cout << "  Running BitPack64..." << std::flush;
                BenchmarkResult result = benchmark_bitpack_64(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
            }

            if (algorithm == "all" || algorithm == "delta") {
                std::cout << "  Running Delta64..." << std::flush;
                BenchmarkResult result = benchmark_delta_64(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
            }
        } else {
            std::cout << "  [32-bit mode]" << std::endl;

            if (algorithm == "all" || algorithm == "bitpack") {
                std::cout << "  Running BitPack..." << std::flush;
                BenchmarkResult result = benchmark_bitpack_32(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
            }

            if (algorithm == "all" || algorithm == "bitpackfor") {
                std::cout << "  Running BitPackFOR..." << std::flush;
                BenchmarkResult result = benchmark_bitpack_for_32(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << std::fixed << std::setprecision(2)
                          << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << " [" << (result.verified ? "OK" : "FAIL") << "]" << std::endl;
            }

            if (algorithm == "all" || algorithm == "delta") {
                std::cout << "  Running Delta..." << std::flush;
                BenchmarkResult result = benchmark_delta_32(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << std::fixed << std::setprecision(2)
                          << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << " [" << (result.verified ? "OK" : "FAIL") << "]" << std::endl;
                if (!result.verified) {
                    std::cerr << "ERROR: Delta32 verification failed! Stopping." << std::endl;
                    csv_file.close();
                    return 1;
                }
            }

            if (algorithm == "all" || algorithm == "deltau") {
                std::cout << "  Running DeltaUniform..." << std::flush;
                BenchmarkResult result = benchmark_delta_32_uniform(data_file, num_trials, warmup_count, mode, verify);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp(eff): " << std::fixed << std::setprecision(2)
                          << result.decompress_throughput_gbps << " GB/s"
                          << ", HBM(est): " << result.estimated_hbm_throughput_gbps << " GB/s"
                          << " [" << (result.verified ? "OK" : "FAIL") << "]" << std::endl;
                if (!result.verified) {
                    std::cerr << "ERROR: DeltaUniform32 verification failed! Stopping." << std::endl;
                    csv_file.close();
                    return 1;
                }
            }
        }
    }

    csv_file.close();
    std::cout << "\nResults written to: " << output_file << std::endl;

    return 0;
}
