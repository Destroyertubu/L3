/**
 * L3 String Compression Benchmark (Large Dataset)
 *
 * Goals:
 * - Benchmark L3 string compression/decompression kernels on large line-based datasets
 * - Avoid std::vector<std::string> to keep CPU memory overhead manageable
 * - Emit machine-readable JSON per dataset for downstream parsing (paper appendix)
 *
 * NOTE:
 * - This benchmark measures GPU kernel time (CUDA events) and reports bytes-based throughput.
 * - It compares encoded-value roundtrip correctness (encoded == decoded) without reconstructing full strings.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "L3_string_format.hpp"
#include "L3_string_utils.hpp"

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// ============================================================================
// Kernel launch declarations (implemented in L3 sources)
// ============================================================================

void launchEncodeStringsToUint64(const char* d_strings, const int32_t* d_string_offsets,
                                 const int8_t* d_string_lengths, int32_t num_strings, int32_t min_char,
                                 int32_t shift_bits, int32_t max_length, uint64_t* d_encoded_values,
                                 int8_t* d_original_lengths, cudaStream_t stream);

void launchEncodeStringsToUint128(const char* d_strings, const int32_t* d_string_offsets,
                                  const int8_t* d_string_lengths, int32_t num_strings, int32_t min_char,
                                  int32_t shift_bits, int32_t max_length, uint128_gpu* d_encoded_values,
                                  int8_t* d_original_lengths, cudaStream_t stream);

void launchEncodeStringsToUint256(const char* d_strings, const int32_t* d_string_offsets,
                                  const int8_t* d_string_lengths, int32_t num_strings, int32_t min_char,
                                  int32_t shift_bits, int32_t max_length, uint256_gpu* d_encoded_values,
                                  int8_t* d_original_lengths, cudaStream_t stream);

void launchFitStringModel(const uint64_t* d_encoded_values, int32_t* d_start_indices, int32_t* d_end_indices,
                          int32_t* d_model_types, double* d_model_params, int32_t* d_delta_bits,
                          int64_t* d_error_bounds, int num_partitions, int64_t* d_total_bits,
                          cudaStream_t stream);

void launchFitStringModel128(const uint128_gpu* d_encoded_values, int32_t* d_start_indices, int32_t* d_end_indices,
                             int32_t* d_model_types, double* d_model_params, int32_t* d_delta_bits,
                             uint128_gpu* d_error_bounds, int num_partitions, int64_t* d_total_bits,
                             cudaStream_t stream);

void launchFitStringModel256(const uint256_gpu* d_encoded_values, int32_t* d_start_indices, int32_t* d_end_indices,
                             int32_t* d_model_types, double* d_model_params, int32_t* d_delta_bits,
                             uint256_gpu* d_error_bounds, int num_partitions, int64_t* d_total_bits,
                             cudaStream_t stream);

void launchSetStringBitOffsets(const int32_t* d_start_indices, const int32_t* d_end_indices,
                               const int32_t* d_delta_bits, int64_t* d_bit_offsets, int num_partitions,
                               cudaStream_t stream);

void launchPackStringDeltas(const uint64_t* d_encoded_values, const int32_t* d_start_indices,
                            const int32_t* d_end_indices, const int32_t* d_model_types,
                            const double* d_model_params, const int32_t* d_delta_bits,
                            const int64_t* d_bit_offsets, int num_partitions, int total_elements,
                            uint32_t* delta_array, cudaStream_t stream);

void launchPackStringDeltas128(const uint128_gpu* d_encoded_values, const int32_t* d_start_indices,
                               const int32_t* d_end_indices, const int32_t* d_model_types,
                               const double* d_model_params, const int32_t* d_delta_bits,
                               const int64_t* d_bit_offsets, int num_partitions, int total_elements,
                               uint32_t* delta_array, cudaStream_t stream);

void launchPackStringDeltas256(const uint256_gpu* d_encoded_values, const int32_t* d_start_indices,
                               const int32_t* d_end_indices, const int32_t* d_model_types,
                               const double* d_model_params, const int32_t* d_delta_bits,
                               const int64_t* d_bit_offsets, int num_partitions, int total_elements,
                               uint32_t* delta_array, cudaStream_t stream);

void launchDecompressToEncodedValues(const uint32_t* delta_array, const int32_t* d_start_indices,
                                     const int32_t* d_end_indices, const int32_t* d_model_types,
                                     const double* d_model_params, const int32_t* d_delta_bits,
                                     const int64_t* d_bit_offsets, int num_partitions, uint64_t* d_encoded_values,
                                     cudaStream_t stream);

void launchDecompressToEncodedValues128(const uint32_t* delta_array, const int32_t* d_start_indices,
                                        const int32_t* d_end_indices, const int32_t* d_model_types,
                                        const double* d_model_params, const int32_t* d_delta_bits,
                                        const int64_t* d_bit_offsets, int num_partitions,
                                        uint128_gpu* d_encoded_values, cudaStream_t stream);

void launchDecompressToEncodedValues256(const uint32_t* delta_array, const int32_t* d_start_indices,
                                        const int32_t* d_end_indices, const int32_t* d_model_types,
                                        const double* d_model_params, const int32_t* d_delta_bits,
                                        const int64_t* d_bit_offsets, int num_partitions,
                                        uint256_gpu* d_encoded_values, cudaStream_t stream);

// ============================================================================
// Local utilities
// ============================================================================

struct Options {
    std::string data_dir = "/root/autodl-tmp/code/L3/tests/strings/data/";
    int iterations = 10;
    int partition_size = 4096;
    int max_strings = 0; // 0 = no limit
    int random_queries = 1000000;
    bool extract_prefix = true;
    bool verify_encoded_roundtrip = true;
};

static std::string ensureTrailingSlash(std::string s) {
    if (!s.empty() && s.back() != '/') s.push_back('/');
    return s;
}

static bool parseIntArg(const char* arg, const char* key, int* out) {
    const size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) != 0) return false;
    const char* value = arg + key_len;
    if (value[0] == '\0') return false;
    *out = std::stoi(value);
    return true;
}

static bool parseBoolArg(const char* arg, const char* key, bool* out) {
    const size_t key_len = std::strlen(key);
    if (std::strncmp(arg, key, key_len) != 0) return false;
    std::string value(arg + key_len);
    if (value == "1" || value == "true" || value == "TRUE") {
        *out = true;
        return true;
    }
    if (value == "0" || value == "false" || value == "FALSE") {
        *out = false;
        return true;
    }
    return false;
}

static Options parseOptions(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "-h" || a == "--help") {
            std::cout << "Usage: bench_l3_strings_large [--data_dir=DIR] [--iterations=N] [--partition_size=N]\n"
                         "                               [--max_strings=N] [--random_queries=N]\n"
                         "                               [--extract_prefix=0|1] [--verify=0|1]\n";
            std::exit(0);
        }
        if (a.rfind("--data_dir=", 0) == 0) {
            opt.data_dir = a.substr(std::string("--data_dir=").size());
            continue;
        }
        if (parseIntArg(argv[i], "--iterations=", &opt.iterations)) continue;
        if (parseIntArg(argv[i], "--partition_size=", &opt.partition_size)) continue;
        if (parseIntArg(argv[i], "--max_strings=", &opt.max_strings)) continue;
        if (parseIntArg(argv[i], "--random_queries=", &opt.random_queries)) continue;
        if (parseBoolArg(argv[i], "--extract_prefix=", &opt.extract_prefix)) continue;
        if (parseBoolArg(argv[i], "--verify=", &opt.verify_encoded_roundtrip)) continue;
        std::cerr << "Unknown argument: " << a << "\n";
        std::exit(2);
    }
    opt.data_dir = ensureTrailingSlash(opt.data_dir);
    if (opt.iterations <= 0) opt.iterations = 1;
    if (opt.partition_size <= 0) opt.partition_size = 4096;
    if (opt.max_strings < 0) opt.max_strings = 0;
    if (opt.random_queries < 0) opt.random_queries = 0;
    return opt;
}

static std::vector<char> readFileBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        std::ostringstream oss;
        oss << "Cannot open file: " << path;
        throw std::runtime_error(oss.str());
    }
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (size > 0 && !f.read(buf.data(), size)) {
        std::ostringstream oss;
        oss << "Failed reading file: " << path;
        throw std::runtime_error(oss.str());
    }
    return buf;
}

static inline void rtrimLine(const std::vector<char>& buf, size_t start, size_t* end_inout) {
    size_t end = *end_inout;
    while (end > start) {
        const char c = buf[end - 1];
        if (c == '\r' || c == ' ') {
            --end;
            continue;
        }
        break;
    }
    *end_inout = end;
}

static std::string computeCommonPrefix(const std::vector<char>& buf) {
    std::string prefix;
    bool init = false;
    size_t start = 0;
    for (size_t i = 0; i <= buf.size(); ++i) {
        if (i == buf.size() || buf[i] == '\n') {
            size_t end = i;
            rtrimLine(buf, start, &end);
            if (end > start) {
                const size_t len = end - start;
                if (!init) {
                    prefix.assign(buf.data() + start, len);
                    init = true;
                } else if (!prefix.empty()) {
                    const size_t max_common = std::min(prefix.size(), len);
                    size_t common = 0;
                    while (common < max_common && prefix[common] == buf[start + common]) ++common;
                    prefix.resize(common);
                    if (prefix.empty()) return prefix;
                }
            }
            start = i + 1;
        }
    }
    return prefix;
}

struct ParsedStrings {
    std::vector<char> suffix_chars;        // concatenated suffix bytes (after prefix removal, possibly truncated)
    std::vector<int32_t> offsets;          // per-string offset into suffix_chars
    std::vector<int8_t> lengths;           // per-string suffix length (after truncation)
    int64_t num_strings = 0;
    int64_t payload_bytes = 0;             // sum of original string lengths (excluding '\n' and trimmed)
    int64_t suffix_bytes = 0;              // suffix_chars.size()
    int32_t max_suffix_len = 0;
    int32_t min_char = 255;
    int32_t max_char = 0;
    int64_t empty_lines = 0;
    int64_t truncated_by_maxlen = 0;
    int64_t truncated_by_int8 = 0;
};

static ParsedStrings parseLineBasedStrings(const std::vector<char>& buf, const std::string& common_prefix,
                                          int32_t truncate_suffix_len, int max_strings) {
    ParsedStrings out;
    const int32_t prefix_len = static_cast<int32_t>(common_prefix.size());
    {
        int32_t suffix_cap = 127;
        if (truncate_suffix_len > 0) suffix_cap = std::min(suffix_cap, truncate_suffix_len);
        if (max_strings > 0) {
            const size_t reserve_bytes = std::min(buf.size(), static_cast<size_t>(max_strings) * static_cast<size_t>(suffix_cap));
            out.suffix_chars.reserve(reserve_bytes);
        } else {
            out.suffix_chars.reserve(buf.size());
        }
    }
    {
        if (max_strings > 0) {
            out.offsets.reserve(static_cast<size_t>(max_strings));
            out.lengths.reserve(static_cast<size_t>(max_strings));
        } else {
            // Avoid massive reallocations on large datasets (e.g., 95M lines).
            int64_t approx_lines = 0;
            for (char c : buf) {
                if (c == '\n') approx_lines++;
            }
            if (!buf.empty() && buf.back() != '\n') approx_lines++;
            if (approx_lines > 0) {
                out.offsets.reserve(static_cast<size_t>(approx_lines));
                out.lengths.reserve(static_cast<size_t>(approx_lines));
            }
        }
    }

    size_t start = 0;
    int64_t cur_offset = 0;
    for (size_t i = 0; i <= buf.size(); ++i) {
        if (i == buf.size() || buf[i] == '\n') {
            size_t end = i;
            rtrimLine(buf, start, &end);
            if (end <= start) {
                out.empty_lines++;
                start = i + 1;
                continue;
            }

            const int32_t line_len = static_cast<int32_t>(end - start);
            out.payload_bytes += line_len;

            const int32_t effective_prefix = std::min(prefix_len, line_len);
            const size_t suffix_start = start + static_cast<size_t>(effective_prefix);
            int32_t suffix_len = line_len - effective_prefix;

            if (truncate_suffix_len > 0 && suffix_len > truncate_suffix_len) {
                out.truncated_by_maxlen++;
                suffix_len = truncate_suffix_len;
            }

            if (suffix_len > 127) {
                out.truncated_by_int8++;
                suffix_len = 127;
            }

            if (cur_offset > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error("Suffix buffer exceeds int32 offset range (>2GB).");
            }
            out.offsets.push_back(static_cast<int32_t>(cur_offset));
            out.lengths.push_back(static_cast<int8_t>(suffix_len));
            out.num_strings++;
            out.max_suffix_len = std::max(out.max_suffix_len, suffix_len);

            for (int32_t k = 0; k < suffix_len; ++k) {
                const uint8_t c = static_cast<uint8_t>(buf[suffix_start + static_cast<size_t>(k)]);
                out.min_char = std::min(out.min_char, static_cast<int32_t>(c));
                out.max_char = std::max(out.max_char, static_cast<int32_t>(c));
                out.suffix_chars.push_back(static_cast<char>(c));
            }

            cur_offset += suffix_len;
            start = i + 1;

            if (max_strings > 0 && out.num_strings >= max_strings) {
                break;
            }
        }
    }

    out.suffix_bytes = static_cast<int64_t>(out.suffix_chars.size());
    if (out.min_char > out.max_char) {
        out.min_char = 0;
        out.max_char = 255;
    }
    return out;
}

struct DeviceInput {
    char* d_suffix_chars = nullptr;
    int32_t* d_offsets = nullptr;
    int8_t* d_lengths = nullptr;
    int32_t num_strings = 0;
    size_t suffix_bytes = 0;

    void free() {
        if (d_suffix_chars) CUDA_CHECK(cudaFree(d_suffix_chars));
        if (d_offsets) CUDA_CHECK(cudaFree(d_offsets));
        if (d_lengths) CUDA_CHECK(cudaFree(d_lengths));
        d_suffix_chars = nullptr;
        d_offsets = nullptr;
        d_lengths = nullptr;
        num_strings = 0;
        suffix_bytes = 0;
    }
};

static DeviceInput uploadToDevice(const ParsedStrings& parsed, cudaStream_t stream) {
    DeviceInput in;
    in.num_strings = static_cast<int32_t>(parsed.num_strings);
    in.suffix_bytes = parsed.suffix_chars.size();
    if (in.num_strings <= 0) {
        throw std::runtime_error("No strings parsed (num_strings=0).");
    }
    if (in.suffix_bytes == 0) {
        in.suffix_bytes = 1;
    }
    CUDA_CHECK(cudaMalloc(&in.d_suffix_chars, in.suffix_bytes));
    CUDA_CHECK(cudaMalloc(&in.d_offsets, parsed.offsets.size() * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&in.d_lengths, parsed.lengths.size() * sizeof(int8_t)));

    CUDA_CHECK(cudaMemcpyAsync(in.d_suffix_chars, parsed.suffix_chars.data(), parsed.suffix_chars.size(),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(in.d_offsets, parsed.offsets.data(), parsed.offsets.size() * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(in.d_lengths, parsed.lengths.data(), parsed.lengths.size() * sizeof(int8_t),
                               cudaMemcpyHostToDevice, stream));
    return in;
}

struct CudaEventPair {
    cudaEvent_t start{};
    cudaEvent_t stop{};
    CudaEventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    ~CudaEventPair() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    float elapsedMs(cudaStream_t stream, const std::function<void()>& fn, int iterations) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) fn();
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / static_cast<float>(iterations);
    }
};

// ============================================================================
// Encoded roundtrip verification
// ============================================================================

static __device__ __forceinline__ bool equal_val(uint64_t a, uint64_t b) { return a == b; }
static __device__ __forceinline__ bool equal_val(const uint128_gpu& a, const uint128_gpu& b) {
    return a.low == b.low && a.high == b.high;
}
static __device__ __forceinline__ bool equal_val(const uint256_gpu& a, const uint256_gpu& b) {
    return a.words[0] == b.words[0] && a.words[1] == b.words[1] && a.words[2] == b.words[2] && a.words[3] == b.words[3];
}

template <typename T>
__global__ void countMismatchesKernel(const T* a, const T* b, int n, unsigned long long* out) {
    unsigned long long local = 0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        if (!equal_val(a[idx], b[idx])) local++;
    }
    if (local) atomicAdd(out, local);
}

template <typename T>
static unsigned long long countMismatchesDevice(const T* a, const T* b, int n, cudaStream_t stream) {
    unsigned long long* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemsetAsync(d_out, 0, sizeof(unsigned long long), stream));
    const int block = 256;
    const int grid = std::min(65535, (n + block - 1) / block);
    countMismatchesKernel<<<grid, block, 0, stream>>>(a, b, n, d_out);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h_out = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(d_out));
    return h_out;
}

// ============================================================================
// Random access decoding kernels (encoded-value only)
// ============================================================================

static __device__ __forceinline__ int64_t extractDelta64(const uint32_t* delta_array, int64_t bit_offset,
                                                        int32_t delta_bits) {
    if (delta_bits <= 0) return 0;
    int word_idx = static_cast<int>(bit_offset / 32);
    int offset_in_word = static_cast<int>(bit_offset % 32);
    uint64_t raw_value = 0;
    if (delta_bits <= 32) {
        uint64_t word0 = delta_array[word_idx];
        raw_value = word0 >> offset_in_word;
        if (offset_in_word + delta_bits > 32) {
            uint64_t word1 = delta_array[word_idx + 1];
            raw_value |= word1 << (32 - offset_in_word);
        }
        raw_value &= (1ULL << delta_bits) - 1;
    } else {
        int bits_read = 0;
        int current_word = word_idx;
        int current_offset = offset_in_word;
        while (bits_read < delta_bits) {
            int bits_in_word = min(delta_bits - bits_read, 32 - current_offset);
            uint64_t word = delta_array[current_word];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> current_offset) & mask;
            raw_value |= part << bits_read;
            bits_read += bits_in_word;
            current_word++;
            current_offset = 0;
        }
    }
    if (delta_bits < 64) {
        uint64_t sign_bit = 1ULL << (delta_bits - 1);
        if (raw_value & sign_bit) {
            raw_value |= ~((1ULL << delta_bits) - 1);
        }
    }
    return static_cast<int64_t>(raw_value);
}

static __device__ __forceinline__ uint128_gpu extractDelta128(const uint32_t* delta_array, int64_t bit_offset,
                                                             int32_t delta_bits) {
    if (delta_bits <= 0) return uint128_gpu(0);
    uint128_gpu result;
    int word_idx = static_cast<int>(bit_offset / 32);
    int offset_in_word = static_cast<int>(bit_offset % 32);
    int bits_read = 0;
    uint64_t low = 0;
    int bits_for_low = min(delta_bits, 64);
    while (bits_read < bits_for_low) {
        int bits_in_word = min(bits_for_low - bits_read, 32 - offset_in_word);
        uint64_t word = delta_array[word_idx];
        uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
        uint64_t part = (word >> offset_in_word) & mask;
        low |= part << bits_read;
        bits_read += bits_in_word;
        word_idx++;
        offset_in_word = 0;
    }
    result.low = low;
    if (delta_bits > 64) {
        uint64_t high = 0;
        int bits_for_high = delta_bits - 64;
        int high_bits_read = 0;
        while (high_bits_read < bits_for_high) {
            int bits_in_word = min(bits_for_high - high_bits_read, 32 - offset_in_word);
            uint64_t word = delta_array[word_idx];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> offset_in_word) & mask;
            high |= part << high_bits_read;
            high_bits_read += bits_in_word;
            word_idx++;
            offset_in_word = 0;
        }
        result.high = high;
    }
    return result;
}

static __device__ __forceinline__ uint256_gpu extractDelta256(const uint32_t* delta_array, int64_t bit_offset,
                                                             int32_t delta_bits) {
    if (delta_bits <= 0) return uint256_gpu(0);
    uint256_gpu result;
    int word_idx = static_cast<int>(bit_offset / 32);
    int offset_in_word = static_cast<int>(bit_offset % 32);
    int bits_read = 0;
    for (int w = 0; w < 4; ++w) {
        uint64_t word_val = 0;
        int bits_for_word = min(delta_bits - bits_read, 64);
        if (bits_for_word <= 0) {
            result.words[w] = 0;
            continue;
        }
        int word_bits_read = 0;
        while (word_bits_read < bits_for_word) {
            int bits_in_word = min(bits_for_word - word_bits_read, 32 - offset_in_word);
            uint64_t word = delta_array[word_idx];
            uint64_t mask = (bits_in_word == 32) ? ~0ULL : ((1ULL << bits_in_word) - 1);
            uint64_t part = (word >> offset_in_word) & mask;
            word_val |= part << word_bits_read;
            word_bits_read += bits_in_word;
            bits_read += bits_in_word;
            word_idx++;
            offset_in_word = 0;
        }
        result.words[w] = word_val;
    }
    return result;
}

__global__ void randomAccessDecode64Kernel(const uint32_t* delta_array, const double* d_model_params,
                                           const int32_t* d_delta_bits, const int64_t* d_bit_offsets,
                                           int32_t partition_size_aligned, int32_t num_strings,
                                           const int32_t* query_indices, int32_t num_queries,
                                           uint64_t* out_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    int32_t idx = query_indices[tid];
    if (idx < 0 || idx >= num_strings) {
        out_values[tid] = 0;
        return;
    }
    int32_t part = idx / partition_size_aligned;
    int32_t local = idx - part * partition_size_aligned;
    double theta0 = d_model_params[part * 4];
    double theta1 = d_model_params[part * 4 + 1];
    int32_t delta_bits = d_delta_bits[part];
    int64_t bit_offset = d_bit_offsets[part] + static_cast<int64_t>(local) * delta_bits;
    int64_t delta = extractDelta64(delta_array, bit_offset, delta_bits);
    double predicted = fma(theta1, static_cast<double>(local), theta0);
    int64_t pred_int = static_cast<int64_t>(round(predicted));
    out_values[tid] = static_cast<uint64_t>(pred_int + delta);
}

__global__ void randomAccessDecode128Kernel(const uint32_t* delta_array, const double* d_model_params,
                                            const int32_t* d_delta_bits, const int64_t* d_bit_offsets,
                                            int32_t partition_size_aligned, int32_t num_strings,
                                            const int32_t* query_indices, int32_t num_queries,
                                            uint128_gpu* out_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    int32_t idx = query_indices[tid];
    if (idx < 0 || idx >= num_strings) {
        out_values[tid] = uint128_gpu(0);
        return;
    }
    int32_t part = idx / partition_size_aligned;
    int32_t local = idx - part * partition_size_aligned;
    double theta0 = d_model_params[part * 4];
    double theta1 = d_model_params[part * 4 + 1];
    int32_t delta_bits = d_delta_bits[part];
    int64_t bit_offset = d_bit_offsets[part] + static_cast<int64_t>(local) * delta_bits;
    uint128_gpu delta = extractDelta128(delta_array, bit_offset, delta_bits);
    double predicted = fma(theta1, static_cast<double>(local), theta0);
    uint128_gpu pred_128;
    if (predicted >= 0) {
        pred_128.high = static_cast<uint64_t>(predicted / 18446744073709551616.0);
        pred_128.low = static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0));
    }
    out_values[tid] = pred_128 + delta;
}

__global__ void randomAccessDecode256Kernel(const uint32_t* delta_array, const double* d_model_params,
                                            const int32_t* d_delta_bits, const int64_t* d_bit_offsets,
                                            int32_t partition_size_aligned, int32_t num_strings,
                                            const int32_t* query_indices, int32_t num_queries,
                                            uint256_gpu* out_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;
    int32_t idx = query_indices[tid];
    if (idx < 0 || idx >= num_strings) {
        out_values[tid] = uint256_gpu(0);
        return;
    }
    int32_t part = idx / partition_size_aligned;
    int32_t local = idx - part * partition_size_aligned;
    double theta0 = d_model_params[part * 4];
    double theta1 = d_model_params[part * 4 + 1];
    int32_t delta_bits = d_delta_bits[part];
    int64_t bit_offset = d_bit_offsets[part] + static_cast<int64_t>(local) * delta_bits;
    uint256_gpu delta = extractDelta256(delta_array, bit_offset, delta_bits);
    double predicted = fma(theta1, static_cast<double>(local), theta0);
    uint256_gpu pred_256;
    if (predicted >= 0 && predicted < 1e77) {
        pred_256 = uint256_gpu(static_cast<uint64_t>(fmod(predicted, 18446744073709551616.0)));
    }
    out_values[tid] = pred_256 + delta;
}

// ============================================================================
// Benchmark per dataset
// ============================================================================

struct BenchResult {
    std::string dataset;
    int64_t file_bytes = 0;
    int64_t payload_bytes = 0;
    int64_t suffix_bytes = 0;
    int64_t num_strings = 0;
    int32_t max_suffix_len = 0;
    int32_t common_prefix_len = 0;
    int32_t min_char = 0;
    int32_t max_char = 255;
    int32_t shift_bits = 0;
    int32_t partition_size = 0;
    int32_t num_partitions = 0;
    std::string encoding_type;
    int64_t total_bits = 0;
    int64_t delta_array_words = 0;
    int64_t compressed_bytes = 0;
    double compression_ratio = 0.0;
    unsigned long long encoded_mismatches = 0;
    int64_t random_queries = 0;
    float random_access_ms = 0.0f;
    double random_access_mqps = 0.0;
    double random_access_gbps = 0.0;

    float h2d_ms = 0.0f;
    float encode_ms = 0.0f;
    float fit_ms = 0.0f;
    float bit_offsets_ms = 0.0f;
    float delta_zero_ms = 0.0f;
    float pack_ms = 0.0f;
    float decompress_ms = 0.0f;

    float compression_total_ms() const {
        return encode_ms + fit_ms + bit_offsets_ms + delta_zero_ms + pack_ms;
    }
};

static int32_t alignPartitionSize(int32_t partition_size) {
    return ((partition_size + 31) / 32) * 32;
}

static void createPartitions(int32_t num_strings, int32_t partition_size, std::vector<int32_t>& start,
                             std::vector<int32_t>& end) {
    const int32_t aligned = alignPartitionSize(partition_size);
    const int32_t num_partitions = (num_strings + aligned - 1) / aligned;
    start.resize(num_partitions);
    end.resize(num_partitions);
    for (int32_t p = 0; p < num_partitions; ++p) {
        start[p] = p * aligned;
        end[p] = std::min((p + 1) * aligned, num_strings);
    }
}

static std::string encodingTypeName(StringEncodingType t) {
    switch (t) {
        case STRING_ENCODING_64: return "64-bit";
        case STRING_ENCODING_128: return "128-bit";
        case STRING_ENCODING_256: return "256-bit";
        default: return "unknown";
    }
}

static BenchResult runDataset(const Options& opt, const std::string& dataset_name, const std::string& filepath) {
    BenchResult res;
    res.dataset = dataset_name;
    std::cout << "\n== Running dataset: " << dataset_name << " ==\n";

    std::vector<char> file_bytes = readFileBytes(filepath);
    res.file_bytes = static_cast<int64_t>(file_bytes.size());
    std::cout << "  File bytes: " << res.file_bytes << "\n";

    const std::string common_prefix = opt.extract_prefix ? computeCommonPrefix(file_bytes) : std::string();
    res.common_prefix_len = static_cast<int32_t>(common_prefix.size());
    std::cout << "  Common prefix len: " << res.common_prefix_len << "\n";

    // First parse without truncation to decide shift bits / encoding type / max length
    ParsedStrings parsed_full = parseLineBasedStrings(file_bytes, common_prefix, -1, opt.max_strings);
    res.num_strings = parsed_full.num_strings;
    res.payload_bytes = parsed_full.payload_bytes;
    res.suffix_bytes = parsed_full.suffix_bytes;
    res.max_suffix_len = parsed_full.max_suffix_len;
    res.min_char = parsed_full.min_char;
    res.max_char = parsed_full.max_char;

    res.shift_bits = calculateShiftBits(res.max_char - res.min_char + 1);
    const StringEncodingType enc_type = selectEncodingType(res.max_suffix_len, res.shift_bits);
    res.encoding_type = encodingTypeName(enc_type);
    std::cout << "  Strings: " << res.num_strings << " (max_strings=" << opt.max_strings << ")\n";
    std::cout << "  Payload bytes: " << res.payload_bytes << ", suffix bytes: " << res.suffix_bytes << "\n";
    std::cout << "  Max suffix len: " << res.max_suffix_len << ", encoding: " << res.encoding_type << "\n";

    const int32_t max_supported = getMaxStringLength(enc_type, res.shift_bits);
    const int32_t max_encoded_length = std::min(res.max_suffix_len, max_supported);

    ParsedStrings parsed = parsed_full;
    if (max_encoded_length < res.max_suffix_len) {
        parsed = parseLineBasedStrings(file_bytes, common_prefix, max_encoded_length, opt.max_strings);
        res.suffix_bytes = parsed.suffix_bytes;
        res.max_suffix_len = parsed.max_suffix_len;
        res.min_char = parsed.min_char;
        res.max_char = parsed.max_char;
        res.shift_bits = calculateShiftBits(res.max_char - res.min_char + 1);
    }

    // CUDA resources
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Upload inputs (strings+offsets+lengths)
    CudaEventPair timer;
    DeviceInput dev_in;
    res.h2d_ms = timer.elapsedMs(stream, [&]() { dev_in = uploadToDevice(parsed, stream); CUDA_CHECK(cudaStreamSynchronize(stream)); }, 1);
    // NOTE: elapsedMs() already divided by iterations, so here iterations=1 and includes sync.

    // Partitions
    res.partition_size = opt.partition_size;
    std::vector<int32_t> h_start, h_end;
    createPartitions(dev_in.num_strings, opt.partition_size, h_start, h_end);
    res.num_partitions = static_cast<int32_t>(h_start.size());
    std::cout << "  Partitions: " << res.num_partitions << " (partition_size=" << res.partition_size << ")\n";

    int32_t* d_start = nullptr;
    int32_t* d_end = nullptr;
    int32_t* d_model_types = nullptr;
    double* d_model_params = nullptr;
    int32_t* d_delta_bits = nullptr;
    int64_t* d_bit_offsets = nullptr;
    int64_t* d_total_bits = nullptr;
    int8_t* d_out_lengths = nullptr;

    CUDA_CHECK(cudaMalloc(&d_start, res.num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_end, res.num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_model_types, res.num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_model_params, res.num_partitions * 4 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_delta_bits, res.num_partitions * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_bit_offsets, res.num_partitions * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_total_bits, sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_out_lengths, dev_in.num_strings * sizeof(int8_t)));

    CUDA_CHECK(cudaMemcpyAsync(d_start, h_start.data(), h_start.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_end, h_end.data(), h_end.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    // Allocate type-dependent buffers
    uint32_t* d_delta_array = nullptr;
    uint64_t* d_encoded64 = nullptr;
    uint64_t* d_decoded64 = nullptr;
    uint128_gpu* d_encoded128 = nullptr;
    uint128_gpu* d_decoded128 = nullptr;
    uint256_gpu* d_encoded256 = nullptr;
    uint256_gpu* d_decoded256 = nullptr;

    int64_t h_total_bits = 0;

    // Helper lambdas for each type
    auto setup64 = [&]() {
        CUDA_CHECK(cudaMalloc(&d_encoded64, dev_in.num_strings * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_decoded64, dev_in.num_strings * sizeof(uint64_t)));
        int64_t* d_error_bounds = nullptr;
        CUDA_CHECK(cudaMalloc(&d_error_bounds, res.num_partitions * sizeof(int64_t)));

        // Encode + fit once to get total bits and allocate delta array
        launchEncodeStringsToUint64(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                    res.min_char, res.shift_bits, max_encoded_length, d_encoded64, d_out_lengths,
                                    stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
        launchFitStringModel(d_encoded64, d_start, d_end, d_model_types, d_model_params, d_delta_bits, d_error_bounds,
                             res.num_partitions, d_total_bits, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&h_total_bits, d_total_bits, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_error_bounds));
    };

    auto setup128 = [&]() {
        CUDA_CHECK(cudaMalloc(&d_encoded128, dev_in.num_strings * sizeof(uint128_gpu)));
        CUDA_CHECK(cudaMalloc(&d_decoded128, dev_in.num_strings * sizeof(uint128_gpu)));
        uint128_gpu* d_error_bounds = nullptr;
        CUDA_CHECK(cudaMalloc(&d_error_bounds, res.num_partitions * sizeof(uint128_gpu)));

        launchEncodeStringsToUint128(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                     res.min_char, res.shift_bits, max_encoded_length, d_encoded128, d_out_lengths,
                                     stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
        launchFitStringModel128(d_encoded128, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                d_error_bounds, res.num_partitions, d_total_bits, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&h_total_bits, d_total_bits, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_error_bounds));
    };

    auto setup256 = [&]() {
        CUDA_CHECK(cudaMalloc(&d_encoded256, dev_in.num_strings * sizeof(uint256_gpu)));
        CUDA_CHECK(cudaMalloc(&d_decoded256, dev_in.num_strings * sizeof(uint256_gpu)));
        uint256_gpu* d_error_bounds = nullptr;
        CUDA_CHECK(cudaMalloc(&d_error_bounds, res.num_partitions * sizeof(uint256_gpu)));

        launchEncodeStringsToUint256(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                     res.min_char, res.shift_bits, max_encoded_length, d_encoded256, d_out_lengths,
                                     stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
        launchFitStringModel256(d_encoded256, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                d_error_bounds, res.num_partitions, d_total_bits, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(&h_total_bits, d_total_bits, sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_error_bounds));
    };

    if (enc_type == STRING_ENCODING_64) {
        setup64();
    } else if (enc_type == STRING_ENCODING_128) {
        setup128();
    } else {
        setup256();
    }

    res.total_bits = h_total_bits;
    res.delta_array_words = (h_total_bits + 31) / 32;
    if (res.delta_array_words < 1) res.delta_array_words = 1;
    CUDA_CHECK(cudaMalloc(&d_delta_array, static_cast<size_t>(res.delta_array_words) * sizeof(uint32_t)));

    // Compute bit offsets once (depends on delta bits from fit)
    launchSetStringBitOffsets(d_start, d_end, d_delta_bits, d_bit_offsets, res.num_partitions, stream);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Warmup pack+decompress once
    CUDA_CHECK(cudaMemsetAsync(d_delta_array, 0, static_cast<size_t>(res.delta_array_words) * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (enc_type == STRING_ENCODING_64) {
        launchPackStringDeltas(d_encoded64, d_start, d_end, d_model_types, d_model_params, d_delta_bits, d_bit_offsets,
                               res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launchDecompressToEncodedValues(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                        d_bit_offsets, res.num_partitions, d_decoded64, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else if (enc_type == STRING_ENCODING_128) {
        launchPackStringDeltas128(d_encoded128, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                  d_bit_offsets, res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launchDecompressToEncodedValues128(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                           d_bit_offsets, res.num_partitions, d_decoded128, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        launchPackStringDeltas256(d_encoded256, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                  d_bit_offsets, res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        launchDecompressToEncodedValues256(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                           d_bit_offsets, res.num_partitions, d_decoded256, stream);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Optional correctness check: encoded == decoded (on device)
    if (opt.verify_encoded_roundtrip) {
        if (enc_type == STRING_ENCODING_64) {
            res.encoded_mismatches = countMismatchesDevice(d_encoded64, d_decoded64, dev_in.num_strings, stream);
        } else if (enc_type == STRING_ENCODING_128) {
            res.encoded_mismatches = countMismatchesDevice(d_encoded128, d_decoded128, dev_in.num_strings, stream);
        } else {
            res.encoded_mismatches = countMismatchesDevice(d_encoded256, d_decoded256, dev_in.num_strings, stream);
        }
    }

    // Benchmark timings (kernel-only, using CUDA events)
    const int iters = opt.iterations;

    if (enc_type == STRING_ENCODING_64) {
        res.encode_ms = timer.elapsedMs(stream, [&]() {
            launchEncodeStringsToUint64(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                        res.min_char, res.shift_bits, max_encoded_length, d_encoded64, d_out_lengths,
                                        stream);
        }, iters);

        res.fit_ms = timer.elapsedMs(stream, [&]() {
            CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
            // Allocate error bounds once per iteration? Here we reuse a scratch buffer to avoid alloc in timing.
            // Use d_bit_offsets as scratch? Not safe. Allocate a dedicated scratch outside timing would be better.
        }, 1); // placeholder, will be overwritten below
    }

    // Allocate per-type error bounds scratch for fit benchmark (avoid cudaMalloc in timing)
    int64_t* d_error_bounds64 = nullptr;
    uint128_gpu* d_error_bounds128 = nullptr;
    uint256_gpu* d_error_bounds256 = nullptr;
    if (enc_type == STRING_ENCODING_64) {
        CUDA_CHECK(cudaMalloc(&d_error_bounds64, res.num_partitions * sizeof(int64_t)));
    } else if (enc_type == STRING_ENCODING_128) {
        CUDA_CHECK(cudaMalloc(&d_error_bounds128, res.num_partitions * sizeof(uint128_gpu)));
    } else {
        CUDA_CHECK(cudaMalloc(&d_error_bounds256, res.num_partitions * sizeof(uint256_gpu)));
    }

    if (enc_type == STRING_ENCODING_64) {
        res.fit_ms = timer.elapsedMs(stream, [&]() {
            CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
            launchFitStringModel(d_encoded64, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                 d_error_bounds64, res.num_partitions, d_total_bits, stream);
        }, iters);
    } else if (enc_type == STRING_ENCODING_128) {
        res.encode_ms = timer.elapsedMs(stream, [&]() {
            launchEncodeStringsToUint128(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                         res.min_char, res.shift_bits, max_encoded_length, d_encoded128, d_out_lengths,
                                         stream);
        }, iters);

        res.fit_ms = timer.elapsedMs(stream, [&]() {
            CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
            launchFitStringModel128(d_encoded128, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                    d_error_bounds128, res.num_partitions, d_total_bits, stream);
        }, iters);
    } else {
        res.encode_ms = timer.elapsedMs(stream, [&]() {
            launchEncodeStringsToUint256(dev_in.d_suffix_chars, dev_in.d_offsets, dev_in.d_lengths, dev_in.num_strings,
                                         res.min_char, res.shift_bits, max_encoded_length, d_encoded256, d_out_lengths,
                                         stream);
        }, iters);

        res.fit_ms = timer.elapsedMs(stream, [&]() {
            CUDA_CHECK(cudaMemsetAsync(d_total_bits, 0, sizeof(int64_t), stream));
            launchFitStringModel256(d_encoded256, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                    d_error_bounds256, res.num_partitions, d_total_bits, stream);
        }, iters);
    }

    res.bit_offsets_ms = timer.elapsedMs(stream, [&]() {
        launchSetStringBitOffsets(d_start, d_end, d_delta_bits, d_bit_offsets, res.num_partitions, stream);
    }, iters);

    res.delta_zero_ms = timer.elapsedMs(stream, [&]() {
        CUDA_CHECK(cudaMemsetAsync(d_delta_array, 0, static_cast<size_t>(res.delta_array_words) * sizeof(uint32_t),
                                   stream));
    }, iters);

    if (enc_type == STRING_ENCODING_64) {
        res.pack_ms = timer.elapsedMs(stream, [&]() {
            launchPackStringDeltas(d_encoded64, d_start, d_end, d_model_types, d_model_params, d_delta_bits, d_bit_offsets,
                                   res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        }, iters);
        res.decompress_ms = timer.elapsedMs(stream, [&]() {
            launchDecompressToEncodedValues(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                            d_bit_offsets, res.num_partitions, d_decoded64, stream);
        }, iters);
    } else if (enc_type == STRING_ENCODING_128) {
        res.pack_ms = timer.elapsedMs(stream, [&]() {
            launchPackStringDeltas128(d_encoded128, d_start, d_end, d_model_types, d_model_params, d_delta_bits, d_bit_offsets,
                                      res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        }, iters);
        res.decompress_ms = timer.elapsedMs(stream, [&]() {
            launchDecompressToEncodedValues128(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                               d_bit_offsets, res.num_partitions, d_decoded128, stream);
        }, iters);
    } else {
        res.pack_ms = timer.elapsedMs(stream, [&]() {
            launchPackStringDeltas256(d_encoded256, d_start, d_end, d_model_types, d_model_params, d_delta_bits, d_bit_offsets,
                                      res.num_partitions, dev_in.num_strings, d_delta_array, stream);
        }, iters);
        res.decompress_ms = timer.elapsedMs(stream, [&]() {
            launchDecompressToEncodedValues256(d_delta_array, d_start, d_end, d_model_types, d_model_params, d_delta_bits,
                                               d_bit_offsets, res.num_partitions, d_decoded256, stream);
        }, iters);
    }

    // Compressed size estimate (align with L3_string_codec.cpp for 64-bit; extend conservatively for 128/256)
    const int64_t per_partition_base =
        static_cast<int64_t>(sizeof(int32_t) * 2 + sizeof(int32_t) + sizeof(double) * 4 + sizeof(int32_t) + sizeof(int64_t)); // + bit offset
    int64_t error_bound_bytes = 0;
    if (enc_type == STRING_ENCODING_64) error_bound_bytes = sizeof(int64_t);
    if (enc_type == STRING_ENCODING_128) error_bound_bytes = sizeof(uint128_gpu);
    if (enc_type == STRING_ENCODING_256) error_bound_bytes = sizeof(uint256_gpu);

    res.compressed_bytes =
        res.num_partitions * (per_partition_base + error_bound_bytes) +
        res.delta_array_words * static_cast<int64_t>(sizeof(uint32_t)) +
        res.num_strings * static_cast<int64_t>(sizeof(int8_t)) +
        res.common_prefix_len;
    res.compression_ratio = res.compressed_bytes > 0 ? static_cast<double>(res.payload_bytes) / static_cast<double>(res.compressed_bytes) : 0.0;

    // Random access benchmark (encoded-value only, query indices on GPU)
    if (opt.random_queries > 0 && res.num_strings > 0) {
        const int64_t queries = std::min<int64_t>(opt.random_queries, res.num_strings);
        if (queries > 0) {
            std::vector<int32_t> h_indices(static_cast<size_t>(queries));
            std::mt19937 rng(42);
            std::uniform_int_distribution<int32_t> dist(0, static_cast<int32_t>(res.num_strings - 1));
            for (int64_t i = 0; i < queries; ++i) {
                h_indices[static_cast<size_t>(i)] = dist(rng);
            }
            int32_t* d_indices = nullptr;
            CUDA_CHECK(cudaMalloc(&d_indices, static_cast<size_t>(queries) * sizeof(int32_t)));
            CUDA_CHECK(cudaMemcpyAsync(d_indices, h_indices.data(), static_cast<size_t>(queries) * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, stream));
            const int threads = 256;
            const int blocks = static_cast<int>((queries + threads - 1) / threads);
            const int32_t aligned_partition = alignPartitionSize(res.partition_size);
            if (enc_type == STRING_ENCODING_64) {
                uint64_t* d_out = nullptr;
                CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(queries) * sizeof(uint64_t)));
                res.random_access_ms = timer.elapsedMs(stream, [&]() {
                    randomAccessDecode64Kernel<<<blocks, threads, 0, stream>>>(
                        d_delta_array, d_model_params, d_delta_bits, d_bit_offsets,
                        aligned_partition, dev_in.num_strings, d_indices, static_cast<int32_t>(queries), d_out);
                }, iters);
                CUDA_CHECK(cudaFree(d_out));
            } else if (enc_type == STRING_ENCODING_128) {
                uint128_gpu* d_out = nullptr;
                CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(queries) * sizeof(uint128_gpu)));
                res.random_access_ms = timer.elapsedMs(stream, [&]() {
                    randomAccessDecode128Kernel<<<blocks, threads, 0, stream>>>(
                        d_delta_array, d_model_params, d_delta_bits, d_bit_offsets,
                        aligned_partition, dev_in.num_strings, d_indices, static_cast<int32_t>(queries), d_out);
                }, iters);
                CUDA_CHECK(cudaFree(d_out));
            } else {
                uint256_gpu* d_out = nullptr;
                CUDA_CHECK(cudaMalloc(&d_out, static_cast<size_t>(queries) * sizeof(uint256_gpu)));
                res.random_access_ms = timer.elapsedMs(stream, [&]() {
                    randomAccessDecode256Kernel<<<blocks, threads, 0, stream>>>(
                        d_delta_array, d_model_params, d_delta_bits, d_bit_offsets,
                        aligned_partition, dev_in.num_strings, d_indices, static_cast<int32_t>(queries), d_out);
                }, iters);
                CUDA_CHECK(cudaFree(d_out));
            }
            CUDA_CHECK(cudaFree(d_indices));
            res.random_queries = queries;
            if (res.random_access_ms > 0.0f) {
                res.random_access_mqps = (static_cast<double>(queries) / 1.0e6) / (res.random_access_ms / 1000.0);
                const double avg_bytes = res.payload_bytes > 0 ? static_cast<double>(res.payload_bytes) / static_cast<double>(res.num_strings) : 0.0;
                res.random_access_gbps =
                    (avg_bytes * static_cast<double>(queries) / 1.0e9) / (res.random_access_ms / 1000.0);
            }
        }
    }

    // Cleanup
    if (d_error_bounds64) CUDA_CHECK(cudaFree(d_error_bounds64));
    if (d_error_bounds128) CUDA_CHECK(cudaFree(d_error_bounds128));
    if (d_error_bounds256) CUDA_CHECK(cudaFree(d_error_bounds256));

    if (d_encoded64) CUDA_CHECK(cudaFree(d_encoded64));
    if (d_decoded64) CUDA_CHECK(cudaFree(d_decoded64));
    if (d_encoded128) CUDA_CHECK(cudaFree(d_encoded128));
    if (d_decoded128) CUDA_CHECK(cudaFree(d_decoded128));
    if (d_encoded256) CUDA_CHECK(cudaFree(d_encoded256));
    if (d_decoded256) CUDA_CHECK(cudaFree(d_decoded256));
    if (d_delta_array) CUDA_CHECK(cudaFree(d_delta_array));

    if (d_start) CUDA_CHECK(cudaFree(d_start));
    if (d_end) CUDA_CHECK(cudaFree(d_end));
    if (d_model_types) CUDA_CHECK(cudaFree(d_model_types));
    if (d_model_params) CUDA_CHECK(cudaFree(d_model_params));
    if (d_delta_bits) CUDA_CHECK(cudaFree(d_delta_bits));
    if (d_bit_offsets) CUDA_CHECK(cudaFree(d_bit_offsets));
    if (d_total_bits) CUDA_CHECK(cudaFree(d_total_bits));
    if (d_out_lengths) CUDA_CHECK(cudaFree(d_out_lengths));

    dev_in.free();
    CUDA_CHECK(cudaStreamDestroy(stream));
    return res;
}

static void printJsonLine(const Options& opt, const BenchResult& r) {
    // Single-line JSON for easy parsing
    std::cout << "L3_BENCH_JSON: {"
              << "\"framework\":\"L3\""
              << ",\"dataset\":\"" << r.dataset << "\""
              << ",\"file_bytes\":" << r.file_bytes
              << ",\"payload_bytes\":" << r.payload_bytes
              << ",\"suffix_bytes\":" << r.suffix_bytes
              << ",\"num_strings\":" << r.num_strings
              << ",\"max_suffix_len\":" << r.max_suffix_len
              << ",\"common_prefix_len\":" << r.common_prefix_len
              << ",\"min_char\":" << r.min_char
              << ",\"max_char\":" << r.max_char
              << ",\"shift_bits\":" << r.shift_bits
              << ",\"partition_size\":" << r.partition_size
              << ",\"num_partitions\":" << r.num_partitions
              << ",\"encoding_type\":\"" << r.encoding_type << "\""
              << ",\"iterations\":" << opt.iterations
              << ",\"total_bits\":" << r.total_bits
              << ",\"delta_array_words\":" << r.delta_array_words
              << ",\"compressed_bytes\":" << r.compressed_bytes
              << ",\"compression_ratio\":" << std::fixed << std::setprecision(4) << r.compression_ratio
              << ",\"encoded_mismatches\":" << r.encoded_mismatches
              << ",\"random_queries\":" << r.random_queries
              << ",\"random_access_ms\":" << std::setprecision(6) << r.random_access_ms
              << ",\"random_access_mqps\":" << std::setprecision(6) << r.random_access_mqps
              << ",\"random_access_gbps\":" << std::setprecision(6) << r.random_access_gbps
              << ",\"h2d_ms\":" << std::setprecision(6) << r.h2d_ms
              << ",\"encode_ms\":" << r.encode_ms
              << ",\"fit_ms\":" << r.fit_ms
              << ",\"bit_offsets_ms\":" << r.bit_offsets_ms
              << ",\"delta_zero_ms\":" << r.delta_zero_ms
              << ",\"pack_ms\":" << r.pack_ms
              << ",\"decompress_ms\":" << r.decompress_ms
              << "}" << std::endl;
}

static void printHumanSummary(const BenchResult& r) {
    auto gbps = [](int64_t bytes, float ms) -> double {
        if (ms <= 0) return 0.0;
        return (static_cast<double>(bytes) / 1.0e9) / (static_cast<double>(ms) / 1000.0);
    };

    std::cout << "\n[Dataset: " << r.dataset << "]\n";
    std::cout << "  Strings: " << r.num_strings << "\n";
    std::cout << "  Payload bytes: " << r.payload_bytes << "\n";
    std::cout << "  Suffix bytes:  " << r.suffix_bytes << "\n";
    std::cout << "  Prefix len:    " << r.common_prefix_len << "\n";
    std::cout << "  Charset:       [" << r.min_char << "," << r.max_char << "] (" << (r.max_char - r.min_char + 1)
              << "), shift_bits=" << r.shift_bits << "\n";
    std::cout << "  Encoding:      " << r.encoding_type << "\n";
    std::cout << "  Partitions:    " << r.num_partitions << " (partition_size=" << r.partition_size << ")\n";
    std::cout << "  Compressed:    " << r.compressed_bytes << " bytes, ratio=" << std::fixed << std::setprecision(2)
              << r.compression_ratio << "x\n";
    std::cout << "  Enc mismatches:" << r.encoded_mismatches << "\n";
    if (r.random_queries > 0) {
        std::cout << "  RandomAccess:  " << std::setprecision(3) << r.random_access_ms << " ms | "
                  << std::setprecision(2) << r.random_access_mqps << " M q/s | "
                  << std::setprecision(2) << r.random_access_gbps << " GB/s (payload)\n";
    }

    std::cout << "  Encode:        " << std::setprecision(3) << r.encode_ms << " ms | "
              << std::setprecision(2) << gbps(r.payload_bytes, r.encode_ms) << " GB/s (payload)\n";
    std::cout << "  Fit:           " << r.fit_ms << " ms\n";
    std::cout << "  BitOffsets:    " << r.bit_offsets_ms << " ms\n";
    std::cout << "  DeltaZero:     " << r.delta_zero_ms << " ms\n";
    std::cout << "  Pack:          " << r.pack_ms << " ms\n";
    std::cout << "  CompressTotal: " << r.compression_total_ms() << " ms | "
              << gbps(r.payload_bytes, r.compression_total_ms()) << " GB/s (payload)\n";
    std::cout << "  Decompress:    " << r.decompress_ms << " ms | " << gbps(r.payload_bytes, r.decompress_ms)
              << " GB/s (payload)\n";
}

int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);
    Options opt = parseOptions(argc, argv);

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "L3 large string benchmark\n";
    std::cout << "  GPU: " << prop.name << "\n";
    std::cout << "  CC: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  GlobalMem: " << (prop.totalGlobalMem / (1024ull * 1024ull * 1024ull)) << " GB\n";
    std::cout << "  L2: " << (prop.l2CacheSize / (1024ull * 1024ull)) << " MB\n";
    std::cout << "  data_dir: " << opt.data_dir << "\n";
    std::cout << "  iterations: " << opt.iterations << "\n";
    std::cout << "  partition_size: " << opt.partition_size << "\n";
    std::cout << "  max_strings: " << opt.max_strings << "\n";
    std::cout << "  random_queries: " << opt.random_queries << "\n";
    std::cout << "  extract_prefix: " << (opt.extract_prefix ? 1 : 0) << "\n";
    std::cout << "  verify: " << (opt.verify_encoded_roundtrip ? 1 : 0) << "\n";

    const std::vector<std::string> datasets = {"email_leco_30k.txt", "hex.txt", "words.txt", "mix_500m.txt"};
    for (const auto& name : datasets) {
        const std::string path = opt.data_dir + name;
        try {
            BenchResult r = runDataset(opt, name, path);
            printHumanSummary(r);
            printJsonLine(opt, r);
        } catch (const std::exception& e) {
            std::cerr << "Dataset failed: " << name << " (" << e.what() << ")\n";
        }
    }

    return 0;
}
