/**
 * Test: Transposed vs Vertical Layout Performance Comparison
 *
 * This test compares the performance of:
 *   - Vertical (Lane-Major): Strided memory access
 *   - Transposed (Word-Interleaved): Coalesced memory access
 *
 * Expected Results:
 *   - Similar or better decompression performance for Transposed
 *   - 4x fewer L1 cache sectors for Transposed (NCU analysis)
 *   - Perfect memory coalescing for Transposed
 *
 * Date: 2025-12-16
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

// Include both Vertical and Transposed implementations
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "L3_Transposed_format.hpp"
#include "../src/kernels/compression/encoder_Vertical_opt.cu"
#include "../src/kernels/decompression/decoder_Vertical_opt.cu"
#include "../src/kernels/compression/encoder_Transposed.cu"
#include "../src/kernels/decompression/decoder_Transposed.cu"

// ============================================================================
// Test Configuration
// ============================================================================

struct TestConfig {
    int64_t num_values;
    int partition_size;
    int num_warmup;
    int num_runs;
    bool verify;
    bool verbose;
    bool for4bit_no_tail;
    bool naive_access_bench;
    int naive_bit_width;
    int naive_block_threads;
    int naive_values_per_thread;  // Configurable VALUES_PER_THREAD for naive benchmark
};

// ============================================================================
// Timer Utility
// ============================================================================

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    float stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// ============================================================================
// Naive Access Benchmark (Isolate Memory Access Pattern)
// ============================================================================

__device__ __forceinline__ uint32_t warpXorReduce(uint32_t v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v ^= __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__global__ void naiveAccessVerticalKernel(const uint32_t* __restrict__ data,
                                         int words_per_lane,
                                         int64_t num_mini_vectors,
                                         uint32_t* __restrict__ out) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int64_t words_per_mv = static_cast<int64_t>(words_per_lane) * 32;
    for (int64_t mv = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
         mv < num_mini_vectors;
         mv += static_cast<int64_t>(gridDim.x) * warps_per_block) {

        const int64_t mv_base = mv * words_per_mv;
        const int64_t lane_word_base = mv_base + static_cast<int64_t>(lane_id) * words_per_lane;

        uint32_t acc = 0;
        for (int w = 0; w < words_per_lane; w++) {
            acc ^= data[lane_word_base + w];
        }

        acc = warpXorReduce(acc);
        if (lane_id == 0) out[mv] = acc;
    }
}

__global__ void naiveAccessTransposedKernel(const uint32_t* __restrict__ data,
                                           int words_per_lane,
                                           int64_t num_mini_vectors,
                                           uint32_t* __restrict__ out) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;

    const int64_t words_per_mv = static_cast<int64_t>(words_per_lane) * 32;
    for (int64_t mv = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_id;
         mv < num_mini_vectors;
         mv += static_cast<int64_t>(gridDim.x) * warps_per_block) {

        const int64_t mv_base = mv * words_per_mv;

        uint32_t acc = 0;
        for (int w = 0; w < words_per_lane; w++) {
            const int64_t addr = mv_base + static_cast<int64_t>(w) * 32 + lane_id;
            acc ^= data[addr];
        }

        acc = warpXorReduce(acc);
        if (lane_id == 0) out[mv] = acc;
    }
}

static void generateSyntheticPackedWords(std::vector<uint32_t>& vertical_words,
                                        std::vector<uint32_t>& transposed_words,
                                        int64_t num_mini_vectors,
                                        int words_per_lane) {
    if (num_mini_vectors <= 0) {
        throw std::invalid_argument("num_mini_vectors must be > 0");
    }
    if (words_per_lane <= 0) {
        throw std::invalid_argument("words_per_lane must be > 0");
    }

    const int64_t words_per_mv = static_cast<int64_t>(words_per_lane) * 32;
    const int64_t total_words = num_mini_vectors * words_per_mv;

    vertical_words.resize(total_words);
    transposed_words.resize(total_words);

    // Deterministic, high-entropy-ish pattern to avoid trivial compression/caching artifacts.
    auto mix32 = [](uint32_t x) {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    };

    for (int64_t mv = 0; mv < num_mini_vectors; mv++) {
        const int64_t mv_base = mv * words_per_mv;
        for (int lane = 0; lane < 32; lane++) {
            for (int w = 0; w < words_per_lane; w++) {
                const uint32_t v = mix32(static_cast<uint32_t>(mv) * 0x9e3779b9U ^
                                         static_cast<uint32_t>(lane) * 0x85ebca6bU ^
                                         static_cast<uint32_t>(w) * 0xc2b2ae35U);

                vertical_words[mv_base + static_cast<int64_t>(lane) * words_per_lane + w] = v;
                transposed_words[mv_base + static_cast<int64_t>(w) * 32 + lane] = v;
            }
        }
    }
}

static void runNaiveAccessBenchmark(const TestConfig& config) {
    if (config.naive_bit_width <= 0 || config.naive_bit_width > 32) {
        throw std::invalid_argument("naive_bit_width must be in [1, 32]");
    }
    if (config.num_values <= 0) {
        throw std::invalid_argument("num_values must be > 0");
    }
    if (config.naive_block_threads <= 0 || (config.naive_block_threads % 32) != 0) {
        throw std::invalid_argument("naive_block_threads must be a positive multiple of 32");
    }
    if (config.naive_values_per_thread <= 0) {
        throw std::invalid_argument("naive_values_per_thread must be > 0");
    }

    // Use configurable values_per_thread for naive benchmark
    const int values_per_thread = config.naive_values_per_thread;
    const int mini_vector_size = values_per_thread * 32;  // 32 threads per warp

    if ((config.num_values % mini_vector_size) != 0) {
        throw std::invalid_argument("num_values must be a multiple of (values_per_thread * 32)");
    }

    const int64_t num_mini_vectors = config.num_values / mini_vector_size;
    const int words_per_lane = (values_per_thread * config.naive_bit_width + 31) / 32;
    const int64_t words_per_mv = static_cast<int64_t>(words_per_lane) * 32;
    const int64_t total_words = num_mini_vectors * words_per_mv;
    const double bytes_read = static_cast<double>(total_words) * sizeof(uint32_t);

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          Naive Memory Access Benchmark (Read-Only)               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\nConfig:" << std::endl;
    std::cout << "  MINI_VECTOR_SIZE: " << mini_vector_size << " (configurable)" << std::endl;
    std::cout << "  VALUES_PER_THREAD: " << values_per_thread << " (configurable)" << std::endl;
    std::cout << "  Bit width: " << config.naive_bit_width << std::endl;
    std::cout << "  Mini-vectors: " << num_mini_vectors << std::endl;
    std::cout << "  Words per lane: " << words_per_lane << std::endl;
    std::cout << "  Words per mini-vector: " << words_per_mv << std::endl;
    std::cout << "  Total bytes read per run: " << std::fixed << std::setprecision(2)
              << (bytes_read / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Threads per block: " << config.naive_block_threads << std::endl;

    std::cout << "\nGenerating synthetic packed words..." << std::endl;
    std::vector<uint32_t> h_vertical_words;
    std::vector<uint32_t> h_transposed_words;
    generateSyntheticPackedWords(h_vertical_words, h_transposed_words, num_mini_vectors, words_per_lane);

    uint32_t* d_vertical_words = nullptr;
    uint32_t* d_transposed_words = nullptr;
    uint32_t* d_out_vertical = nullptr;
    uint32_t* d_out_transposed = nullptr;

    cudaMalloc(&d_vertical_words, static_cast<size_t>(total_words) * sizeof(uint32_t));
    cudaMalloc(&d_transposed_words, static_cast<size_t>(total_words) * sizeof(uint32_t));
    cudaMalloc(&d_out_vertical, static_cast<size_t>(num_mini_vectors) * sizeof(uint32_t));
    cudaMalloc(&d_out_transposed, static_cast<size_t>(num_mini_vectors) * sizeof(uint32_t));

    cudaMemcpy(d_vertical_words, h_vertical_words.data(),
               static_cast<size_t>(total_words) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_transposed_words, h_transposed_words.data(),
               static_cast<size_t>(total_words) * sizeof(uint32_t), cudaMemcpyHostToDevice);

    const int threads = config.naive_block_threads;
    const int warps_per_block = threads / 32;
    int blocks = static_cast<int>((num_mini_vectors + warps_per_block - 1) / warps_per_block);
    blocks = std::min(blocks, 65535);

    CudaTimer timer;

    // Warmup
    for (int i = 0; i < config.num_warmup; i++) {
        naiveAccessVerticalKernel<<<blocks, threads>>>(d_vertical_words, words_per_lane, num_mini_vectors, d_out_vertical);
        naiveAccessTransposedKernel<<<blocks, threads>>>(d_transposed_words, words_per_lane, num_mini_vectors, d_out_transposed);
    }
    cudaDeviceSynchronize();

    // Vertical
    timer.start();
    for (int i = 0; i < config.num_runs; i++) {
        naiveAccessVerticalKernel<<<blocks, threads>>>(d_vertical_words, words_per_lane, num_mini_vectors, d_out_vertical);
    }
    float vertical_ms = timer.stop() / config.num_runs;

    // Transposed
    timer.start();
    for (int i = 0; i < config.num_runs; i++) {
        naiveAccessTransposedKernel<<<blocks, threads>>>(d_transposed_words, words_per_lane, num_mini_vectors, d_out_transposed);
    }
    float transposed_ms = timer.stop() / config.num_runs;

    const double vertical_gbps = (bytes_read / 1e9) / (vertical_ms / 1000.0);
    const double transposed_gbps = (bytes_read / 1e9) / (transposed_ms / 1000.0);

    std::cout << "\nResults (effective read bandwidth):" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Vertical (Lane-Major, strided):     " << vertical_ms << " ms, "
              << std::setprecision(2) << vertical_gbps << " GB/s" << std::endl;
    std::cout << "  Transposed (Word-Interleaved):      " << std::setprecision(4) << transposed_ms << " ms, "
              << std::setprecision(2) << transposed_gbps << " GB/s" << std::endl;
    std::cout << "  Speedup (Transposed/Vertical):      " << std::setprecision(2)
              << (vertical_ms / transposed_ms) << "x" << std::endl;

    cudaFree(d_vertical_words);
    cudaFree(d_transposed_words);
    cudaFree(d_out_vertical);
    cudaFree(d_out_transposed);
}

// ============================================================================
// Test Functions
// ============================================================================

template<typename T>
void generateLinearNoiseData(std::vector<T>& data, int64_t n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<T> dist(0, 10000);

    data.resize(n);

    // Generate linear data with noise (good for compression)
    for (int64_t i = 0; i < n; i++) {
        data[i] = static_cast<T>(i * 100 + dist(gen));
    }
}

template<typename T>
void generateFor4bitNoTailData(std::vector<T>& data, int64_t n, int partition_size) {
    if (partition_size <= 0) {
        throw std::invalid_argument("partition_size must be > 0");
    }
    if (n <= 0) {
        throw std::invalid_argument("num_values must be > 0");
    }
    if ((n % partition_size) != 0) {
        throw std::invalid_argument("num_values must be a multiple of partition_size (no tail partitions)");
    }
    if ((partition_size % MINI_VECTOR_SIZE) != 0) {
        throw std::invalid_argument("partition_size must be a multiple of MINI_VECTOR_SIZE (no tail mini-vectors)");
    }

    data.resize(n);

    constexpr uint32_t kDeltaMask = 0xF;  // 4-bit
    const int64_t num_partitions = n / partition_size;

    for (int64_t pid = 0; pid < num_partitions; pid++) {
        // Ensure per-partition range is exactly [base, base + 15]
        const T base = static_cast<T>(1000 + pid * 17);
        const int64_t start = pid * partition_size;
        for (int i = 0; i < partition_size; i++) {
            const uint32_t delta = static_cast<uint32_t>(i) & kDeltaMask;  // 0..15, includes 0 and 15
            data[start + i] = static_cast<T>(base + static_cast<T>(delta));
        }
    }
}

template<typename Compressed>
bool validateNoTailAndDeltaBits(const Compressed& compressed,
                               int32_t expected_delta_bits,
                               const char* label) {
    if (compressed.num_partitions <= 0) {
        std::cout << "  [" << label << "] Invalid num_partitions=" << compressed.num_partitions << std::endl;
        return false;
    }

    std::vector<int32_t> h_delta_bits(compressed.num_partitions);
    std::vector<int32_t> h_tail_sizes(compressed.num_partitions);

    cudaMemcpy(h_delta_bits.data(), compressed.d_delta_bits,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tail_sizes.data(), compressed.d_tail_sizes,
               compressed.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int32_t mismatched_bits = 0;
    int32_t nonzero_tails = 0;
    for (int i = 0; i < compressed.num_partitions; i++) {
        if (h_delta_bits[i] != expected_delta_bits) mismatched_bits++;
        if (h_tail_sizes[i] != 0) nonzero_tails++;
    }

    if (mismatched_bits == 0 && nonzero_tails == 0) {
        std::cout << "  [" << label << "] Format OK: delta_bits=" << expected_delta_bits
                  << ", tail_size=0 (partitions=" << compressed.num_partitions << ")" << std::endl;
        return true;
    }

    std::cout << "  [" << label << "] Format FAIL:"
              << " mismatched_delta_bits=" << mismatched_bits
              << ", nonzero_tail_sizes=" << nonzero_tails
              << " (expected delta_bits=" << expected_delta_bits << ", tail_size=0)" << std::endl;
    return false;
}

template<typename T>
bool verifyResults(const T* h_original, const T* h_decoded, int64_t n, bool verbose = false) {
    int errors = 0;
    int64_t first_error_idx = -1;

    for (int64_t i = 0; i < n; i++) {
        if (h_original[i] != h_decoded[i]) {
            if (first_error_idx < 0) first_error_idx = i;
            errors++;
            if (verbose && errors <= 10) {
                std::cout << "  Mismatch at index " << i
                          << ": expected " << h_original[i]
                          << ", got " << h_decoded[i] << std::endl;
            }
        }
    }

    if (errors > 0) {
        std::cout << "  Total errors: " << errors << " / " << n << std::endl;
        return false;
    }
    return true;
}

template<typename T>
void runTest(const TestConfig& config) {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║       Transposed vs Vertical Layout Performance Test             ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;

    // Generate test data
    std::cout << "\nGenerating test data..." << std::endl;
    std::vector<T> h_data;
    try {
        if (config.for4bit_no_tail) {
            generateFor4bitNoTailData<T>(h_data, config.num_values, config.partition_size);
        } else {
            generateLinearNoiseData<T>(h_data, config.num_values);
        }
    } catch (const std::exception& e) {
        std::cerr << "Data generation error: " << e.what() << std::endl;
        return;
    }

    double uncompressed_mb = config.num_values * sizeof(T) / 1024.0 / 1024.0;
    std::cout << "  Values: " << config.num_values << std::endl;
    std::cout << "  Size: " << std::fixed << std::setprecision(2) << uncompressed_mb << " MB" << std::endl;
    std::cout << "  Partition size: " << config.partition_size << std::endl;

    // ========== Vertical Encoding ==========
    std::cout << "\n=== Encoding with Vertical (Lane-Major) ===" << std::endl;

    VerticalConfig vertical_config = VerticalConfig::defaultConfig();
    vertical_config.partition_size_hint = config.partition_size;
    if (config.for4bit_no_tail) {
        vertical_config.partitioning_strategy = PartitioningStrategy::FIXED;
        vertical_config.use_variable_partition = false;
        vertical_config.enable_adaptive_selection = false;
        vertical_config.fixed_model_type = MODEL_FOR_BITPACK;
        vertical_config.max_delta_bits = 4;
    } else {
        vertical_config.enable_adaptive_selection = true;
    }

    auto vertical_compressed = Vertical_encoder::encodeVerticalGPU<T>(
        h_data, config.partition_size, vertical_config);

    double vertical_compressed_mb = vertical_compressed.interleaved_delta_words * sizeof(uint32_t) / 1024.0 / 1024.0;
    std::cout << "  Compressed size: " << vertical_compressed_mb << " MB" << std::endl;
    std::cout << "  Compression ratio: " << (uncompressed_mb / vertical_compressed_mb) << "x" << std::endl;
    std::cout << "  Encoding time: " << vertical_compressed.kernel_time_ms << " ms" << std::endl;

    // ========== Transposed Encoding ==========
    std::cout << "\n=== Encoding with Transposed (Word-Interleaved) ===" << std::endl;

    TransposedConfig transposed_config = TransposedConfig::defaultConfig();
    transposed_config.partition_size_hint = config.partition_size;
    if (config.for4bit_no_tail) {
        transposed_config.partitioning_strategy = PartitioningStrategy::FIXED;
        transposed_config.use_variable_partition = false;
        transposed_config.enable_adaptive_selection = false;
        transposed_config.fixed_model_type = MODEL_FOR_BITPACK;
        transposed_config.max_delta_bits = 4;
    } else {
        transposed_config.enable_adaptive_selection = true;
    }

    auto transposed_compressed = Transposed_encoder::encodeTransposedGPU<T>(
        h_data, config.partition_size, transposed_config);

    double transposed_compressed_mb = transposed_compressed.transposed_delta_words * sizeof(uint32_t) / 1024.0 / 1024.0;
    std::cout << "  Compressed size: " << transposed_compressed_mb << " MB" << std::endl;
    std::cout << "  Compression ratio: " << (uncompressed_mb / transposed_compressed_mb) << "x" << std::endl;
    std::cout << "  Encoding time: " << transposed_compressed.kernel_time_ms << " ms" << std::endl;

    if (config.for4bit_no_tail) {
        std::cout << "\n=== Validating Format Constraints (4-bit deltas, no tails) ===" << std::endl;
        bool vertical_ok = validateNoTailAndDeltaBits(vertical_compressed, /*expected_delta_bits=*/4, "Vertical");
        bool transposed_ok = validateNoTailAndDeltaBits(transposed_compressed, /*expected_delta_bits=*/4, "Transposed");
        if (!vertical_ok || !transposed_ok) {
            std::cerr << "Format validation failed; aborting benchmark to avoid misleading numbers." << std::endl;
            Vertical_encoder::freeCompressedData(vertical_compressed);
            Transposed_encoder::freeCompressedData(transposed_compressed);
            return;
        }
    }

    // Allocate output buffers
    T* d_output_vertical;
    T* d_output_transposed;
    cudaMalloc(&d_output_vertical, config.num_values * sizeof(T));
    cudaMalloc(&d_output_transposed, config.num_values * sizeof(T));

    CudaTimer timer;

    // ========== Vertical Decoding Benchmark ==========
    std::cout << "\n=== Benchmarking Vertical Decode ===" << std::endl;

    // Warmup
    for (int i = 0; i < config.num_warmup; i++) {
        Vertical_decoder::decompressAll<T>(vertical_compressed, d_output_vertical,
                                           DecompressMode::INTERLEAVED);
    }
    cudaDeviceSynchronize();

    // Benchmark
    timer.start();
    for (int i = 0; i < config.num_runs; i++) {
        Vertical_decoder::decompressAll<T>(vertical_compressed, d_output_vertical,
                                           DecompressMode::INTERLEAVED);
    }
    float vertical_decode_ms = timer.stop() / config.num_runs;

    double vertical_throughput = (config.num_values * sizeof(T) / 1e9) / (vertical_decode_ms / 1000.0);
    std::cout << "  Decode time: " << std::fixed << std::setprecision(4) << vertical_decode_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << vertical_throughput << " GB/s" << std::endl;

    // ========== Transposed Decoding Benchmark ==========
    std::cout << "\n=== Benchmarking Transposed Decode ===" << std::endl;

    // Warmup
    for (int i = 0; i < config.num_warmup; i++) {
        Transposed_decoder::decompressAll<T>(transposed_compressed, d_output_transposed);
    }
    cudaDeviceSynchronize();

    // Benchmark
    timer.start();
    for (int i = 0; i < config.num_runs; i++) {
        Transposed_decoder::decompressAll<T>(transposed_compressed, d_output_transposed);
    }
    float transposed_decode_ms = timer.stop() / config.num_runs;

    double transposed_throughput = (config.num_values * sizeof(T) / 1e9) / (transposed_decode_ms / 1000.0);
    std::cout << "  Decode time: " << std::fixed << std::setprecision(4) << transposed_decode_ms << " ms" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(2) << transposed_throughput << " GB/s" << std::endl;

    // ========== Verification ==========
    if (config.verify) {
        std::cout << "\n=== Verifying Correctness ===" << std::endl;

        std::vector<T> h_output_vertical(config.num_values);
        std::vector<T> h_output_transposed(config.num_values);

        cudaMemcpy(h_output_vertical.data(), d_output_vertical,
                   config.num_values * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_output_transposed.data(), d_output_transposed,
                   config.num_values * sizeof(T), cudaMemcpyDeviceToHost);

        bool vertical_ok = verifyResults(h_data.data(), h_output_vertical.data(),
                                          config.num_values, config.verbose);
        bool transposed_ok = verifyResults(h_data.data(), h_output_transposed.data(),
                                            config.num_values, config.verbose);

        std::cout << "  Vertical:   " << (vertical_ok ? "PASS" : "FAIL") << std::endl;
        std::cout << "  Transposed: " << (transposed_ok ? "PASS" : "FAIL") << std::endl;
    }

    // ========== Summary ==========
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                      PERFORMANCE SUMMARY                          ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Layout              │ Decode (ms) │ Throughput   │ Speedup       ║" << std::endl;
    std::cout << "╠═══════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Vertical (Strided)  │ " << std::setw(11) << std::fixed << std::setprecision(4) << vertical_decode_ms
              << " │ " << std::setw(7) << std::setprecision(2) << vertical_throughput << " GB/s"
              << " │   1.00x       ║" << std::endl;
    std::cout << "║  Transposed (Coalesce│ " << std::setw(11) << std::fixed << std::setprecision(4) << transposed_decode_ms
              << " │ " << std::setw(7) << std::setprecision(2) << transposed_throughput << " GB/s"
              << " │ " << std::setw(6) << std::setprecision(2) << (vertical_decode_ms / transposed_decode_ms) << "x       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════════════╝" << std::endl;

    std::cout << "\n┌─────────────────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│                      MEMORY ACCESS COMPARISON                   │" << std::endl;
    std::cout << "├─────────────────────────────────────────────────────────────────┤" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ Vertical (Lane-Major):                                          │" << std::endl;
    std::cout << "│   Thread 0 reads word[0], word[1], ...                          │" << std::endl;
    std::cout << "│   Thread 1 reads word[W], word[W+1], ...  (stride = W)          │" << std::endl;
    std::cout << "│   → Strided access, multiple memory transactions                │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ Transposed (Word-Interleaved):                                  │" << std::endl;
    std::cout << "│   Thread 0 reads word[0], word[32], ...                         │" << std::endl;
    std::cout << "│   Thread 1 reads word[1], word[33], ...                         │" << std::endl;
    std::cout << "│   → Consecutive addresses, single 128B coalesced transaction!   │" << std::endl;
    std::cout << "│                                                                 │" << std::endl;
    std::cout << "│ Use NCU to verify:                                              │" << std::endl;
    std::cout << "│   ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  │" << std::endl;
    std::cout << "│   Transposed should show 4x fewer L1 sectors than Vertical      │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────────────────┘" << std::endl;

    // Cleanup
    cudaFree(d_output_vertical);
    cudaFree(d_output_transposed);
    Vertical_encoder::freeCompressedData(vertical_compressed);
    Transposed_encoder::freeCompressedData(transposed_compressed);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Default configuration
    TestConfig config;
    config.num_values = 256 * 1024 * 100;  // ~100 MB uncompressed
    config.partition_size = 4096;
    config.num_warmup = 3;
    config.num_runs = 10;
    config.verify = true;
    config.verbose = false;
    config.for4bit_no_tail = false;
    config.naive_access_bench = false;
    config.naive_bit_width = 4;
    config.naive_block_threads = 256;
    config.naive_values_per_thread = 32;  // Default: 32 values per thread (MV_SIZE=1024)

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--size" && i + 1 < argc) {
            config.num_values = std::stoll(argv[++i]);
        } else if (arg == "--partition" && i + 1 < argc) {
            config.partition_size = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            config.num_runs = std::stoi(argv[++i]);
        } else if (arg == "--no-verify") {
            config.verify = false;
        } else if (arg == "--verbose") {
            config.verbose = true;
        } else if (arg == "--for4bit") {
            config.for4bit_no_tail = true;
        } else if (arg == "--naive") {
            config.naive_access_bench = true;
            config.verify = false;
        } else if (arg == "--bit-width" && i + 1 < argc) {
            config.naive_bit_width = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.naive_block_threads = std::stoi(argv[++i]);
        } else if (arg == "--values-per-thread" && i + 1 < argc) {
            config.naive_values_per_thread = std::stoi(argv[++i]);
        } else if (arg == "--large") {
            // Large scale test to exceed L2 cache
            config.num_values = 256LL * 4096000;  // ~4GB
            config.verify = false;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --size N              Number of values (default: 25.6M)\n"
                      << "  --partition N         Partition size (default: 4096)\n"
                      << "  --runs N              Number of benchmark runs (default: 10)\n"
                      << "  --for4bit             Generate per-partition data with 4-bit FOR deltas, and require no tails\n"
                      << "  --naive               Run naive read-only kernels to isolate layout memory access\n"
                      << "  --bit-width N         Bit width for --naive (default: 4)\n"
                      << "  --threads N           Threads per block for --naive (default: 256, must be multiple of 32)\n"
                      << "  --values-per-thread N Values per thread for --naive (default: 32, MV_SIZE = N * 32)\n"
                      << "  --no-verify           Skip verification\n"
                      << "  --verbose             Verbose error output\n"
                      << "  --large               Large scale test (~4GB)\n"
                      << "  --help                Show this help\n";
            return 0;
        }
    }

    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << " (CC " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "L2 Cache: " << (prop.l2CacheSize / 1024 / 1024) << " MB" << std::endl;

    try {
        if (config.naive_access_bench) {
            runNaiveAccessBenchmark(config);
        } else {
            runTest<uint32_t>(config);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
