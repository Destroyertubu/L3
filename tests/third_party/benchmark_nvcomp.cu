/**
 * nvcomp Benchmark (nvcomp 5.x API)
 *
 * Tests all nvcomp compression algorithms: ANS, Bitcomp, Cascaded, Deflate,
 * GDeflate, LZ4, Snappy, Zstd
 *
 * Measures: compression ratio, compression throughput, decompression throughput
 *
 * Usage: benchmark_nvcomp [options]
 *   -d, --data_dir <path>    Data directory (default: /root/autodl-tmp/test/data/sosd/)
 *   -o, --output <file>      Output CSV file (default: reports/nvcomp_results.csv)
 *   -a, --algorithm <alg>    Algorithm: all|ans|bitcomp|cascaded|deflate|gdeflate|lz4|snappy|zstd
 *   -p, --chunk_size <size>  Chunk size in bytes (default: 65536)
 *   -n, --trials <num>       Number of trials (default: 10)
 *   -w, --warmup <num>       Warmup iterations (default: 1)
 *   -g, --gpu <id>           GPU device ID (default: 0)
 */

#include <cuda_runtime.h>
#include <nvcomp.h>
#include <nvcomp/ans.h>
#include <nvcomp/bitcomp.h>
#include <nvcomp/cascaded.h>
#include <nvcomp/deflate.h>
#include <nvcomp/gdeflate.h>
#include <nvcomp/lz4.h>
#include <nvcomp/snappy.h>
#include <nvcomp/zstd.h>

#include <fstream>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <system_error>

// ============================================================================
// Macros and Utilities
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while(0)

#define NVCOMP_CHECK(call) do { \
    nvcompStatus_t status = call; \
    if (status != nvcompSuccess) { \
        std::cerr << "nvcomp Error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        return; \
    } \
} while(0)

struct BenchmarkResult {
    std::string framework;
    std::string algorithm;
    std::string dataset;
    size_t original_size;
    size_t compressed_size;
    double compression_ratio;
    double compress_time_ms;
    double decompress_time_ms;
    double compress_throughput_gbps;
    double decompress_throughput_gbps;
    bool verified;
};

// ============================================================================
// Utility Functions
// ============================================================================

// Detect data type from filename
int detect_element_size(const std::string& filename) {
    if (filename.find("uint64") != std::string::npos ||
        filename.find("int64") != std::string::npos) {
        return 8;
    } else if (filename.find("uint32") != std::string::npos ||
               filename.find("int32") != std::string::npos) {
        return 4;
    }
    // Default: check file size to guess
    return 4;  // Assume 32-bit if unknown
}

// Load binary file as raw bytes
char* load_binary_file_raw(const char* filename, size_t& total_bytes) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    total_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    char* data = new char[total_bytes];
    file.read(data, total_bytes);
    file.close();

    return data;
}

template<typename T>
T* load_binary_file(const char* filename, size_t& count) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return nullptr;
    }

    size_t size = file.tellg();
    count = size / sizeof(T);
    file.seekg(0, std::ios::beg);

    T* data = new T[count];
    file.read(reinterpret_cast<char*>(data), size);
    file.close();

    return data;
}

std::string extract_dataset_name(const std::string& path) {
    size_t last_slash = path.find_last_of("/\\");
    std::string filename = (last_slash == std::string::npos) ? path : path.substr(last_slash + 1);
    size_t last_dot = filename.find_last_of(".");
    return (last_dot == std::string::npos) ? filename : filename.substr(0, last_dot);
}

void write_csv_header(std::ofstream& csv) {
    csv << "framework,algorithm,dataset,data_size_bytes,compressed_size_bytes,"
        << "compression_ratio,compress_time_ms,decompress_time_ms,"
        << "compress_throughput_gbps,decompress_throughput_gbps,verified\n";
}

void write_csv_result(std::ofstream& csv, const BenchmarkResult& result) {
    csv << std::fixed << std::setprecision(4)
        << result.framework << "," << result.algorithm << "," << result.dataset << ","
        << result.original_size << "," << result.compressed_size << ","
        << result.compression_ratio << "," << result.compress_time_ms << ","
        << result.decompress_time_ms << ","
        << result.compress_throughput_gbps << "," << result.decompress_throughput_gbps << ","
        << (result.verified ? "true" : "false") << "\n";
    csv.flush();
}

// ============================================================================
// BatchData class for managing GPU memory
// ============================================================================

class BatchData {
public:
    bool valid = true;  // Track if construction succeeded

    BatchData(const std::vector<std::vector<char>>& host_data) {
        m_size = host_data.size();

        // Calculate total size and prefixsum
        std::vector<size_t> prefixsum(m_size + 1, 0);
        size_t max_chunk = 0;
        for (size_t i = 0; i < m_size; ++i) {
            if (host_data[i].size() > max_chunk) {
                max_chunk = host_data[i].size();
            }
            // Align to 8 bytes
            prefixsum[i + 1] = ((prefixsum[i] + host_data[i].size() + 7) / 8) * 8;
        }

        m_total_size = prefixsum.back();

        // Allocate device memory with error checking
        cudaError_t err;
        err = cudaMalloc(&m_data, m_total_size);
        if (err != cudaSuccess) { valid = false; return; }

        err = cudaMalloc(&m_ptrs, m_size * sizeof(void*));
        if (err != cudaSuccess) { valid = false; cudaFree(m_data); m_data = nullptr; return; }

        err = cudaMalloc(&m_sizes, m_size * sizeof(size_t));
        if (err != cudaSuccess) { valid = false; cudaFree(m_data); cudaFree(m_ptrs); m_data = nullptr; m_ptrs = nullptr; return; }

        // Build pointers array
        std::vector<void*> h_ptrs(m_size);
        std::vector<size_t> h_sizes(m_size);
        for (size_t i = 0; i < m_size; ++i) {
            h_ptrs[i] = static_cast<void*>(m_data + prefixsum[i]);
            h_sizes[i] = host_data[i].size();
        }

        cudaMemcpy(m_ptrs, h_ptrs.data(), m_size * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(m_sizes, h_sizes.data(), m_size * sizeof(size_t), cudaMemcpyHostToDevice);

        // Copy data to GPU
        for (size_t i = 0; i < m_size; ++i) {
            cudaMemcpy(h_ptrs[i], host_data[i].data(), host_data[i].size(), cudaMemcpyHostToDevice);
        }
    }

    BatchData(size_t max_output_size, size_t batch_size) {
        m_size = batch_size;
        m_total_size = max_output_size * batch_size;

        cudaError_t err;
        err = cudaMalloc(&m_data, m_total_size);
        if (err != cudaSuccess) { valid = false; return; }

        err = cudaMalloc(&m_ptrs, batch_size * sizeof(void*));
        if (err != cudaSuccess) { valid = false; cudaFree(m_data); m_data = nullptr; return; }

        err = cudaMalloc(&m_sizes, batch_size * sizeof(size_t));
        if (err != cudaSuccess) { valid = false; cudaFree(m_data); cudaFree(m_ptrs); m_data = nullptr; m_ptrs = nullptr; return; }

        std::vector<void*> h_ptrs(batch_size);
        std::vector<size_t> h_sizes(batch_size, max_output_size);
        for (size_t i = 0; i < batch_size; ++i) {
            h_ptrs[i] = static_cast<void*>(m_data + max_output_size * i);
        }

        cudaMemcpy(m_ptrs, h_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);
        cudaMemcpy(m_sizes, h_sizes.data(), batch_size * sizeof(size_t), cudaMemcpyHostToDevice);
    }

    ~BatchData() {
        if (m_data) cudaFree(m_data);
        if (m_ptrs) cudaFree(m_ptrs);
        if (m_sizes) cudaFree(m_sizes);
    }

    void** ptrs() { return m_ptrs; }
    size_t* sizes() { return m_sizes; }
    uint8_t* data() { return m_data; }
    size_t total_size() const { return m_total_size; }
    size_t size() const { return m_size; }

private:
    void** m_ptrs = nullptr;
    size_t* m_sizes = nullptr;
    uint8_t* m_data = nullptr;
    size_t m_size = 0;
    size_t m_total_size = 0;
};

// ============================================================================
// Split data into chunks
// ============================================================================

std::vector<std::vector<char>> split_into_chunks(const void* data, size_t total_bytes, size_t chunk_size) {
    std::vector<std::vector<char>> chunks;
    const char* bytes = static_cast<const char*>(data);

    for (size_t offset = 0; offset < total_bytes; offset += chunk_size) {
        size_t this_chunk = std::min(chunk_size, total_bytes - offset);
        chunks.emplace_back(bytes + offset, bytes + offset + this_chunk);
    }

    return chunks;
}

// ============================================================================
// LZ4 Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_lz4(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "LZ4";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    // Load data as raw bytes
    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    // Split into chunks
    auto chunks = split_into_chunks(raw_data, total_bytes, chunk_size);
    size_t batch_size = chunks.size();

    // Create input BatchData
    BatchData input_data(chunks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Get input sizes on host
    std::vector<size_t> h_input_sizes(batch_size);
    cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    // nvcomp 5.x: nvcompBatchedLZ4Opts_t
    nvcompBatchedLZ4Opts_t comp_opts = nvcompBatchedLZ4DefaultOpts;

    // Get temp size - nvcomp 5.x async version
    size_t comp_temp_bytes;
    nvcompBatchedLZ4CompressGetTempSizeEx(batch_size, chunk_size, comp_opts, &comp_temp_bytes, total_bytes);

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0) {
        cudaMalloc(&d_comp_temp, comp_temp_bytes);
    }

    // Get max output size
    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, comp_opts, &max_out_bytes);

    // Create output BatchData
    BatchData compress_data(max_out_bytes, batch_size);

    // Allocate status arrays (required by nvcomp 5.x)
    nvcompStatus_t* d_comp_statuses;
    cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

    // Warmup
    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedLZ4CompressAsync(
            (const void* const*)input_data.ptrs(),
            input_data.sizes(),
            chunk_size,
            batch_size,
            d_comp_temp,
            comp_temp_bytes,
            compress_data.ptrs(),
            compress_data.sizes(),
            comp_opts, stream);
        cudaStreamSynchronize(stream);
    }

    // Measure compression
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_comp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedLZ4CompressAsync(
            (const void* const*)input_data.ptrs(),
            input_data.sizes(),
            chunk_size,
            batch_size,
            d_comp_temp,
            comp_temp_bytes,
            compress_data.ptrs(),
            compress_data.sizes(),
            comp_opts, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_comp_time += ms;
    }
    result.compress_time_ms = total_comp_time / num_trials;

    // Get compressed sizes
    std::vector<size_t> h_comp_sizes(batch_size);
    cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t total_compressed = 0;
    for (size_t i = 0; i < batch_size; i++) {
        total_compressed += h_comp_sizes[i];
    }
    result.compressed_size = total_compressed;

    // Check for compression failure
    if (total_compressed == 0) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (d_comp_temp) cudaFree(d_comp_temp);
        cudaFree(d_comp_statuses);
        cudaStreamDestroy(stream);
        delete[] raw_data;
        return result;
    }
    result.compression_ratio = (double)total_bytes / total_compressed;

    // Decompression - nvcomp 5.x API

    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSizeEx(batch_size, chunk_size, &decomp_temp_bytes, total_bytes);

    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) {
        cudaMalloc(&d_decomp_temp, decomp_temp_bytes);
    }

    // Allocate output buffers
    std::vector<void*> h_decomp_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
    }
    void** d_decomp_ptrs;
    cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
    cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t* d_decomp_sizes;
    cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));

    nvcompStatus_t* d_decomp_statuses;
    cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

    // Warmup decompression
    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedLZ4DecompressAsync(
            (const void* const*)compress_data.ptrs(),
            compress_data.sizes(),
            input_data.sizes(),
            d_decomp_sizes,
            batch_size,
            d_decomp_temp,
            decomp_temp_bytes,
            d_decomp_ptrs, d_decomp_statuses,
            stream);
        cudaStreamSynchronize(stream);
    }

    // Measure decompression
    float total_decomp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedLZ4DecompressAsync(
            (const void* const*)compress_data.ptrs(),
            compress_data.sizes(),
            input_data.sizes(),
            d_decomp_sizes,
            batch_size,
            d_decomp_temp,
            decomp_temp_bytes,
            d_decomp_ptrs, d_decomp_statuses,
            stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_decomp_time += ms;
    }
    result.decompress_time_ms = total_decomp_time / num_trials;

    // Verify
    result.verified = true;
    std::vector<char> decomp_chunk(chunk_size);
    for (size_t i = 0; i < batch_size && result.verified; i++) {
        cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
        if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) {
            result.verified = false;
        }
    }

    // Calculate throughput
    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    if (d_comp_temp) cudaFree(d_comp_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
    cudaFree(d_decomp_ptrs);
    cudaFree(d_decomp_sizes);
    cudaFree(d_comp_statuses);
    cudaFree(d_decomp_statuses);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// Snappy Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_snappy(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "Snappy";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    auto chunks = split_into_chunks(raw_data, total_bytes, chunk_size);
    size_t batch_size = chunks.size();

    BatchData input_data(chunks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<size_t> h_input_sizes(batch_size);
    cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    // nvcomp 5.x
    nvcompBatchedSnappyOpts_t comp_opts = nvcompBatchedSnappyDefaultOpts;

    size_t comp_temp_bytes;
    nvcompBatchedSnappyCompressGetTempSizeEx(batch_size, chunk_size, comp_opts, &comp_temp_bytes, total_bytes);

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0) cudaMalloc(&d_comp_temp, comp_temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedSnappyCompressGetMaxOutputChunkSize(chunk_size, comp_opts, &max_out_bytes);

    BatchData compress_data(max_out_bytes, batch_size);

    nvcompStatus_t* d_comp_statuses;
    cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

    // Warmup and measure compression
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedSnappyCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaStreamSynchronize(stream);
    }

    float total_comp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedSnappyCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_comp_time += ms;
    }
    result.compress_time_ms = total_comp_time / num_trials;

    std::vector<size_t> h_comp_sizes(batch_size);
    cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t total_compressed = 0;
    for (auto s : h_comp_sizes) total_compressed += s;
    result.compressed_size = total_compressed;

    // Check for compression failure
    if (total_compressed == 0) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (d_comp_temp) cudaFree(d_comp_temp);
        cudaFree(d_comp_statuses);
        cudaStreamDestroy(stream);
        delete[] raw_data;
        return result;
    }
    result.compression_ratio = (double)total_bytes / total_compressed;

    // Decompression

    size_t decomp_temp_bytes;
    nvcompBatchedSnappyDecompressGetTempSizeEx(batch_size, chunk_size, &decomp_temp_bytes, total_bytes);

    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

    std::vector<void*> h_decomp_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
    void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
    cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
    nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedSnappyDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaStreamSynchronize(stream);
    }

    float total_decomp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedSnappyDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_decomp_time += ms;
    }
    result.decompress_time_ms = total_decomp_time / num_trials;

    // Verify
    result.verified = true;
    std::vector<char> decomp_chunk(chunk_size);
    for (size_t i = 0; i < batch_size && result.verified; i++) {
        cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
        if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) result.verified = false;
    }

    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (d_comp_temp) cudaFree(d_comp_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
    cudaFree(d_decomp_ptrs); cudaFree(d_decomp_sizes);
    cudaFree(d_comp_statuses); cudaFree(d_decomp_statuses);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// Zstd Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_zstd(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "Zstd";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    // For large files (>2GB), split into segments and process each independently
    // This ensures memory is fully released between segments
    const size_t SEGMENT_SIZE = 2UL * 1024 * 1024 * 1024;  // 2GB per segment
    size_t num_segments = (total_bytes + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

    if (total_bytes > 1e9) {
        std::cerr << "\n    [Zstd] Processing " << (total_bytes / 1e9) << " GB in "
                  << num_segments << " segments" << std::endl;
    }

    float total_comp_time = 0;
    float total_decomp_time = 0;
    size_t total_compressed = 0;
    bool all_verified = true;

    nvcompBatchedZstdOpts_t comp_opts = nvcompBatchedZstdDefaultOpts;

    for (size_t seg_idx = 0; seg_idx < num_segments; seg_idx++) {
        size_t seg_start = seg_idx * SEGMENT_SIZE;
        size_t seg_end = std::min(seg_start + SEGMENT_SIZE, total_bytes);
        size_t seg_size = seg_end - seg_start;

        // Split this segment into chunks
        auto chunks = split_into_chunks(raw_data + seg_start, seg_size, chunk_size);
        size_t batch_size = chunks.size();

        if (total_bytes > 1e9) {
            std::cerr << "    [Zstd] Segment " << seg_idx << "/" << num_segments
                      << ": " << batch_size << " chunks" << std::endl;
        }

        // Create CUDA resources for this segment
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Input data
        BatchData input_data(chunks);
        if (!input_data.valid) {
            std::cerr << "    [Zstd] Segment " << seg_idx << " input_data allocation failed" << std::endl;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            delete[] raw_data;
            return result;
        }

        std::vector<size_t> h_input_sizes(batch_size);
        cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        // Get temp buffer size
        size_t comp_temp_bytes;
        nvcompBatchedZstdCompressGetTempSizeEx(batch_size, chunk_size, comp_opts, &comp_temp_bytes, seg_size);

        void* d_comp_temp = nullptr;
        if (comp_temp_bytes > 0) {
            cudaError_t err = cudaMalloc(&d_comp_temp, comp_temp_bytes);
            if (err != cudaSuccess) {
                std::cerr << "    [Zstd] Segment " << seg_idx << " comp_temp allocation failed" << std::endl;
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                cudaStreamDestroy(stream);
                delete[] raw_data;
                return result;
            }
        }

        // Get max output size
        size_t max_out_bytes;
        nvcompBatchedZstdCompressGetMaxOutputChunkSize(chunk_size, comp_opts, &max_out_bytes);

        BatchData compress_data(max_out_bytes, batch_size);
        if (!compress_data.valid) {
            std::cerr << "    [Zstd] Segment " << seg_idx << " compress_data allocation failed" << std::endl;
            if (d_comp_temp) cudaFree(d_comp_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            delete[] raw_data;
            return result;
        }

        // Warmup (only on first segment)
        if (seg_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedZstdCompressAsync(
                    (const void* const*)input_data.ptrs(), input_data.sizes(),
                    chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                    compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Compression timing
        float seg_comp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedZstdCompressAsync(
                (const void* const*)input_data.ptrs(), input_data.sizes(),
                chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            seg_comp_time += ms;
        }
        total_comp_time += seg_comp_time / num_trials;

        // Check compression status and get sizes
        std::vector<size_t> h_comp_sizes(batch_size);
        cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        size_t seg_compressed = 0;
        for (size_t i = 0; i < batch_size; i++) {
            seg_compressed += h_comp_sizes[i];
        }

        if (seg_compressed == 0) {
            std::cerr << "    [Zstd] Segment " << seg_idx << " compression failed (size=0)" << std::endl;
            if (d_comp_temp) cudaFree(d_comp_temp);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            cudaStreamDestroy(stream);
            delete[] raw_data;
            return result;
        }

        total_compressed += seg_compressed;

        // Decompression
        size_t decomp_temp_bytes;
        nvcompBatchedZstdDecompressGetTempSizeEx(batch_size, chunk_size, &decomp_temp_bytes, seg_size);

        void* d_decomp_temp = nullptr;
        if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

        std::vector<void*> h_decomp_ptrs(batch_size);
        for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
        void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
        cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

        size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
        nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

        // Warmup decompression (only on first segment)
        if (seg_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedZstdDecompressAsync(
                    (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                    input_data.sizes(), d_decomp_sizes, batch_size,
                    d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Decompression timing
        float seg_decomp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedZstdDecompressAsync(
                (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                input_data.sizes(), d_decomp_sizes, batch_size,
                d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            seg_decomp_time += ms;
        }
        total_decomp_time += seg_decomp_time / num_trials;

        // Check decompression status
        std::vector<nvcompStatus_t> h_decomp_statuses(batch_size);
        cudaMemcpy(h_decomp_statuses.data(), d_decomp_statuses, batch_size * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost);

        bool decomp_ok = true;
        for (size_t i = 0; i < batch_size; i++) {
            if (h_decomp_statuses[i] != nvcompSuccess) {
                decomp_ok = false;
                break;
            }
        }

        if (!decomp_ok) {
            all_verified = false;
        } else {
            // Verify decompression
            std::vector<char> decomp_chunk(chunk_size);
            for (size_t i = 0; i < batch_size && all_verified; i++) {
                cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
                if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) {
                    all_verified = false;
                }
            }
        }

        // Cleanup segment resources
        if (d_comp_temp) cudaFree(d_comp_temp);
        if (d_decomp_temp) cudaFree(d_decomp_temp);
        for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
        cudaFree(d_decomp_ptrs);
        cudaFree(d_decomp_sizes);
        cudaFree(d_decomp_statuses);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);

        // Force GPU memory cleanup between segments
        cudaDeviceSynchronize();
    }

    // Set results
    result.compressed_size = total_compressed;
    result.compression_ratio = (double)total_bytes / total_compressed;
    result.compress_time_ms = total_comp_time;
    result.decompress_time_ms = total_decomp_time;
    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);

    if (all_verified && total_decomp_time > 0.1) {
        result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);
        result.verified = true;
    } else {
        result.decompress_throughput_gbps = 0;
        result.verified = false;
    }

    delete[] raw_data;
    return result;
}

// ============================================================================
// Bitcomp Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_bitcomp(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "Bitcomp";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    auto chunks = split_into_chunks(raw_data, total_bytes, chunk_size);
    size_t batch_size = chunks.size();

    BatchData input_data(chunks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<size_t> h_input_sizes(batch_size);
    cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    nvcompBatchedBitcompFormatOpts comp_opts = nvcompBatchedBitcompDefaultOpts;
    comp_opts.data_type = NVCOMP_TYPE_UINT;

    size_t comp_temp_bytes;
    nvcompBatchedBitcompCompressGetTempSizeEx(batch_size, chunk_size, comp_opts, &comp_temp_bytes, total_bytes);

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0) cudaMalloc(&d_comp_temp, comp_temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedBitcompCompressGetMaxOutputChunkSize(chunk_size, comp_opts, &max_out_bytes);

    BatchData compress_data(max_out_bytes, batch_size);

    nvcompStatus_t* d_comp_statuses;
    cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedBitcompCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaStreamSynchronize(stream);
    }

    float total_comp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedBitcompCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_comp_time += ms;
    }
    result.compress_time_ms = total_comp_time / num_trials;

    std::vector<size_t> h_comp_sizes(batch_size);
    cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t total_compressed = 0;
    for (auto s : h_comp_sizes) total_compressed += s;
    result.compressed_size = total_compressed;

    // Check for compression failure
    if (total_compressed == 0) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (d_comp_temp) cudaFree(d_comp_temp);
        cudaFree(d_comp_statuses);
        cudaStreamDestroy(stream);
        delete[] raw_data;
        return result;
    }
    result.compression_ratio = (double)total_bytes / total_compressed;


    size_t decomp_temp_bytes;
    nvcompBatchedBitcompDecompressGetTempSizeEx(batch_size, chunk_size, &decomp_temp_bytes, total_bytes);

    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

    std::vector<void*> h_decomp_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
    void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
    cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
    nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedBitcompDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaStreamSynchronize(stream);
    }

    float total_decomp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedBitcompDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_decomp_time += ms;
    }
    result.decompress_time_ms = total_decomp_time / num_trials;

    result.verified = true;
    std::vector<char> decomp_chunk(chunk_size);
    for (size_t i = 0; i < batch_size && result.verified; i++) {
        cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
        if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) result.verified = false;
    }

    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (d_comp_temp) cudaFree(d_comp_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
    cudaFree(d_decomp_ptrs); cudaFree(d_decomp_sizes);
    cudaFree(d_comp_statuses); cudaFree(d_decomp_statuses);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// Cascaded Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_cascaded(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "Cascaded";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    // Use smaller chunk size for Cascaded (recommended 512-16384)
    // But for very large files, use larger chunks to limit batch size
    size_t cascaded_chunk_size = std::min(chunk_size, (size_t)4096);
    size_t max_batch_size = 500000;  // Limit to 500K chunks to avoid memory/timeout issues
    if (total_bytes / cascaded_chunk_size > max_batch_size) {
        cascaded_chunk_size = (total_bytes / max_batch_size) + 1;
        // Round up to power of 2 for better alignment
        size_t power = 1;
        while (power < cascaded_chunk_size) power *= 2;
        cascaded_chunk_size = power;
    }

    auto chunks = split_into_chunks(raw_data, total_bytes, cascaded_chunk_size);
    size_t batch_size = chunks.size();

    BatchData input_data(chunks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<size_t> h_input_sizes(batch_size);
    cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    nvcompBatchedCascadedOpts_t comp_opts = nvcompBatchedCascadedDefaultOpts;
    comp_opts.chunk_size = cascaded_chunk_size;
    comp_opts.type = NVCOMP_TYPE_UINT;
    comp_opts.num_RLEs = 2;
    comp_opts.num_deltas = 1;
    comp_opts.use_bp = 1;

    size_t comp_temp_bytes;
    nvcompBatchedCascadedCompressGetTempSizeEx(batch_size, cascaded_chunk_size, comp_opts, &comp_temp_bytes, total_bytes);

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0) cudaMalloc(&d_comp_temp, comp_temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedCascadedCompressGetMaxOutputChunkSize(cascaded_chunk_size, comp_opts, &max_out_bytes);

    BatchData compress_data(max_out_bytes, batch_size);

    nvcompStatus_t* d_comp_statuses;
    cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedCascadedCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            cascaded_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaStreamSynchronize(stream);
    }

    float total_comp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedCascadedCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            cascaded_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_comp_time += ms;
    }
    result.compress_time_ms = total_comp_time / num_trials;

    std::vector<size_t> h_comp_sizes(batch_size);
    cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t total_compressed = 0;
    for (auto s : h_comp_sizes) total_compressed += s;
    result.compressed_size = total_compressed;

    // Check for compression failure
    if (total_compressed == 0) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (d_comp_temp) cudaFree(d_comp_temp);
        cudaFree(d_comp_statuses);
        cudaStreamDestroy(stream);
        delete[] raw_data;
        return result;
    }
    result.compression_ratio = (double)total_bytes / total_compressed;

    // Decompression setup

    size_t decomp_temp_bytes;
    nvcompBatchedCascadedDecompressGetTempSizeEx(batch_size, cascaded_chunk_size, &decomp_temp_bytes, total_bytes);

    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

    std::vector<void*> h_decomp_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
    void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
    cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
    nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedCascadedDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaStreamSynchronize(stream);
    }

    float total_decomp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedCascadedDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_decomp_time += ms;
    }
    result.decompress_time_ms = total_decomp_time / num_trials;

    result.verified = true;
    std::vector<char> decomp_chunk(cascaded_chunk_size);
    for (size_t i = 0; i < batch_size && result.verified; i++) {
        cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
        if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) result.verified = false;
    }

    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (d_comp_temp) cudaFree(d_comp_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
    cudaFree(d_decomp_ptrs); cudaFree(d_decomp_sizes);
    cudaFree(d_comp_statuses); cudaFree(d_decomp_statuses);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// GDeflate Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_gdeflate(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "GDeflate";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    // GDeflate max chunk size is 64KB
    size_t gdeflate_chunk_size = std::min(chunk_size, (size_t)65536);

    auto all_chunks = split_into_chunks(raw_data, total_bytes, gdeflate_chunk_size);
    size_t total_chunks = all_chunks.size();

    // Limit batch size to avoid huge temp buffer (each chunk needs ~360KB temp)
    // 30GB temp buffer / 360KB per chunk ≈ 80K chunks max
    const size_t MAX_BATCH_SIZE = 50000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvcompBatchedGdeflateOpts_t comp_opts = nvcompBatchedGdeflateDefaultOpts;

    // Collect all compressed data for each chunk
    std::vector<std::vector<char>> all_compressed(total_chunks);
    std::vector<size_t> all_comp_sizes(total_chunks);

    float total_comp_time = 0;
    float total_decomp_time = 0;
    size_t total_compressed = 0;
    bool all_verified = true;

    // Process in batches
    size_t num_batches = (total_chunks + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        size_t batch_start = batch_idx * MAX_BATCH_SIZE;
        size_t batch_end = std::min(batch_start + MAX_BATCH_SIZE, total_chunks);
        size_t batch_size = batch_end - batch_start;

        // Extract chunks for this batch
        std::vector<std::vector<char>> batch_chunks(all_chunks.begin() + batch_start,
                                                     all_chunks.begin() + batch_end);

        BatchData input_data(batch_chunks);

        std::vector<size_t> h_input_sizes(batch_size);
        cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        // Get temp buffer size for this batch
        size_t comp_temp_bytes;
        nvcompBatchedGdeflateCompressGetTempSizeEx(batch_size, gdeflate_chunk_size, comp_opts, &comp_temp_bytes, 0);

        void* d_comp_temp = nullptr;
        if (comp_temp_bytes > 0) {
            cudaError_t err = cudaMalloc(&d_comp_temp, comp_temp_bytes);
            if (err != cudaSuccess) {
                // Memory allocation failed, skip this algorithm for this dataset
                cudaStreamDestroy(stream);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                delete[] raw_data;
                return result;
            }
        }

        size_t max_out_bytes;
        nvcompBatchedGdeflateCompressGetMaxOutputChunkSize(gdeflate_chunk_size, comp_opts, &max_out_bytes);

        BatchData compress_data(max_out_bytes, batch_size);

        nvcompStatus_t* d_comp_statuses;
        cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

        // Warmup (only on first batch)
        if (batch_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedGdeflateCompressAsync(
                    (const void* const*)input_data.ptrs(), input_data.sizes(),
                    gdeflate_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                    compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Compression timing
        float batch_comp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedGdeflateCompressAsync(
                (const void* const*)input_data.ptrs(), input_data.sizes(),
                gdeflate_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            batch_comp_time += ms;
        }
        total_comp_time += batch_comp_time / num_trials;

        // Check compression status and get sizes
        std::vector<size_t> h_comp_sizes(batch_size);
        cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        bool batch_comp_ok = true;
        size_t batch_compressed = 0;
        for (size_t i = 0; i < batch_size; i++) {
            batch_compressed += h_comp_sizes[i];
            all_comp_sizes[batch_start + i] = h_comp_sizes[i];
        }
        total_compressed += batch_compressed;

        if (!batch_comp_ok || batch_compressed == 0) {
            std::cerr << "\n    [DEBUG] batch " << batch_idx << "/" << num_batches
                      << " comp failed (size=0)" << std::endl;
            if (d_comp_temp) cudaFree(d_comp_temp);
            cudaFree(d_comp_statuses);
            cudaStreamDestroy(stream);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            delete[] raw_data;
            return result;
        }

        // Copy compressed data to host for later decompression
        std::vector<void*> h_comp_ptrs(batch_size);
        cudaMemcpy(h_comp_ptrs.data(), compress_data.ptrs(), batch_size * sizeof(void*), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < batch_size; i++) {
            all_compressed[batch_start + i].resize(h_comp_sizes[i]);
            cudaMemcpy(all_compressed[batch_start + i].data(), h_comp_ptrs[i], h_comp_sizes[i], cudaMemcpyDeviceToHost);
        }

        // Decompression
        size_t decomp_temp_bytes;
        nvcompBatchedGdeflateDecompressGetTempSizeEx(batch_size, gdeflate_chunk_size, &decomp_temp_bytes, 0);

        void* d_decomp_temp = nullptr;
        if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

        std::vector<void*> h_decomp_ptrs(batch_size);
        for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
        void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
        cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

        size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
        nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

        // Warmup decompression (only on first batch)
        if (batch_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedGdeflateDecompressAsync(
                    (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                    input_data.sizes(), d_decomp_sizes, batch_size,
                    d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Decompression timing
        float batch_decomp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedGdeflateDecompressAsync(
                (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                input_data.sizes(), d_decomp_sizes, batch_size,
                d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            batch_decomp_time += ms;
        }
        total_decomp_time += batch_decomp_time / num_trials;

        // Verify decompression
        std::vector<char> decomp_chunk(gdeflate_chunk_size);
        for (size_t i = 0; i < batch_size && all_verified; i++) {
            cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
            if (memcmp(decomp_chunk.data(), batch_chunks[i].data(), h_input_sizes[i]) != 0) {
                all_verified = false;
            }
        }

        // Cleanup batch resources
        if (d_comp_temp) cudaFree(d_comp_temp);
        if (d_decomp_temp) cudaFree(d_decomp_temp);
        for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
        cudaFree(d_decomp_ptrs);
        cudaFree(d_decomp_sizes);
        cudaFree(d_comp_statuses);
        cudaFree(d_decomp_statuses);
    }

    // Set results
    result.compressed_size = total_compressed;
    result.compression_ratio = (double)total_bytes / total_compressed;
    result.compress_time_ms = total_comp_time;
    result.decompress_time_ms = total_decomp_time;
    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);
    result.verified = all_verified;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// Deflate Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_deflate(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "Deflate";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    // Deflate max chunk size is 64KB
    size_t deflate_chunk_size = std::min(chunk_size, (size_t)65536);

    auto all_chunks = split_into_chunks(raw_data, total_bytes, deflate_chunk_size);
    size_t total_chunks = all_chunks.size();

    // Limit batch size to avoid huge temp buffer
    const size_t MAX_BATCH_SIZE = 50000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    nvcompBatchedDeflateOpts_t comp_opts = nvcompBatchedDeflateDefaultOpts;

    // Collect all compressed data for each chunk
    std::vector<std::vector<char>> all_compressed(total_chunks);
    std::vector<size_t> all_comp_sizes(total_chunks);

    float total_comp_time = 0;
    float total_decomp_time = 0;
    size_t total_compressed = 0;
    bool all_verified = true;

    // Process in batches
    size_t num_batches = (total_chunks + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;

    for (size_t batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        size_t batch_start = batch_idx * MAX_BATCH_SIZE;
        size_t batch_end = std::min(batch_start + MAX_BATCH_SIZE, total_chunks);
        size_t batch_size = batch_end - batch_start;

        // Extract chunks for this batch
        std::vector<std::vector<char>> batch_chunks(all_chunks.begin() + batch_start,
                                                     all_chunks.begin() + batch_end);

        BatchData input_data(batch_chunks);

        std::vector<size_t> h_input_sizes(batch_size);
        cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        // Get temp buffer size for this batch
        size_t comp_temp_bytes;
        nvcompBatchedDeflateCompressGetTempSizeEx(batch_size, deflate_chunk_size, comp_opts, &comp_temp_bytes, 0);

        void* d_comp_temp = nullptr;
        if (comp_temp_bytes > 0) {
            cudaError_t err = cudaMalloc(&d_comp_temp, comp_temp_bytes);
            if (err != cudaSuccess) {
                // Memory allocation failed, skip this algorithm for this dataset
                cudaStreamDestroy(stream);
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                delete[] raw_data;
                return result;
            }
        }

        size_t max_out_bytes;
        nvcompBatchedDeflateCompressGetMaxOutputChunkSize(deflate_chunk_size, comp_opts, &max_out_bytes);

        BatchData compress_data(max_out_bytes, batch_size);

        nvcompStatus_t* d_comp_statuses;
        cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

        // Warmup (only on first batch)
        if (batch_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedDeflateCompressAsync(
                    (const void* const*)input_data.ptrs(), input_data.sizes(),
                    deflate_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                    compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Compression timing
        float batch_comp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedDeflateCompressAsync(
                (const void* const*)input_data.ptrs(), input_data.sizes(),
                deflate_chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
                compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            batch_comp_time += ms;
        }
        total_comp_time += batch_comp_time / num_trials;

        // Check compression status and get sizes
        std::vector<size_t> h_comp_sizes(batch_size);
        cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

        bool batch_comp_ok = true;
        size_t batch_compressed = 0;
        for (size_t i = 0; i < batch_size; i++) {
            batch_compressed += h_comp_sizes[i];
            all_comp_sizes[batch_start + i] = h_comp_sizes[i];
        }
        total_compressed += batch_compressed;

        if (!batch_comp_ok || batch_compressed == 0) {
            std::cerr << "\n    [DEBUG] batch " << batch_idx << "/" << num_batches
                      << " comp failed (size=0)" << std::endl;
            if (d_comp_temp) cudaFree(d_comp_temp);
            cudaFree(d_comp_statuses);
            cudaStreamDestroy(stream);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            delete[] raw_data;
            return result;
        }

        // Copy compressed data to host for later decompression
        std::vector<void*> h_comp_ptrs(batch_size);
        cudaMemcpy(h_comp_ptrs.data(), compress_data.ptrs(), batch_size * sizeof(void*), cudaMemcpyDeviceToHost);
        for (size_t i = 0; i < batch_size; i++) {
            all_compressed[batch_start + i].resize(h_comp_sizes[i]);
            cudaMemcpy(all_compressed[batch_start + i].data(), h_comp_ptrs[i], h_comp_sizes[i], cudaMemcpyDeviceToHost);
        }

        // Decompression
        size_t decomp_temp_bytes;
        nvcompBatchedDeflateDecompressGetTempSizeEx(batch_size, deflate_chunk_size, &decomp_temp_bytes, 0);

        void* d_decomp_temp = nullptr;
        if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

        std::vector<void*> h_decomp_ptrs(batch_size);
        for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
        void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
        cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

        size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
        nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

        // Warmup decompression (only on first batch)
        if (batch_idx == 0) {
            for (int i = 0; i < warmup_count; i++) {
                nvcompBatchedDeflateDecompressAsync(
                    (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                    input_data.sizes(), d_decomp_sizes, batch_size,
                    d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Decompression timing
        float batch_decomp_time = 0;
        for (int trial = 0; trial < num_trials; trial++) {
            cudaEventRecord(start, stream);
            nvcompBatchedDeflateDecompressAsync(
                (const void* const*)compress_data.ptrs(), compress_data.sizes(),
                input_data.sizes(), d_decomp_sizes, batch_size,
                d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            batch_decomp_time += ms;
        }
        total_decomp_time += batch_decomp_time / num_trials;

        // Verify decompression
        std::vector<char> decomp_chunk(deflate_chunk_size);
        for (size_t i = 0; i < batch_size && all_verified; i++) {
            cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
            if (memcmp(decomp_chunk.data(), batch_chunks[i].data(), h_input_sizes[i]) != 0) {
                all_verified = false;
            }
        }

        // Cleanup batch resources
        if (d_comp_temp) cudaFree(d_comp_temp);
        if (d_decomp_temp) cudaFree(d_decomp_temp);
        for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
        cudaFree(d_decomp_ptrs);
        cudaFree(d_decomp_sizes);
        cudaFree(d_comp_statuses);
        cudaFree(d_decomp_statuses);
    }

    // Set results
    result.compressed_size = total_compressed;
    result.compression_ratio = (double)total_bytes / total_compressed;
    result.compress_time_ms = total_comp_time;
    result.decompress_time_ms = total_decomp_time;
    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);
    result.verified = all_verified;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// ANS Benchmark (nvcomp 5.x API)
// ============================================================================

BenchmarkResult benchmark_ans(const std::string& data_file, size_t chunk_size, int num_trials, int warmup_count) {
    BenchmarkResult result;
    result.framework = "nvcomp";
    result.algorithm = "ANS";
    result.dataset = extract_dataset_name(data_file);
    result.verified = false;
    result.original_size = 0;
    result.compressed_size = 0;
    result.compression_ratio = 0;
    result.compress_time_ms = 0;
    result.decompress_time_ms = 0;
    result.compress_throughput_gbps = 0;
    result.decompress_throughput_gbps = 0;

    size_t total_bytes = 0;
    char* raw_data = load_binary_file_raw(data_file.c_str(), total_bytes);
    if (!raw_data || total_bytes == 0) return result;

    result.original_size = total_bytes;

    auto chunks = split_into_chunks(raw_data, total_bytes, chunk_size);
    size_t batch_size = chunks.size();

    BatchData input_data(chunks);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::vector<size_t> h_input_sizes(batch_size);
    cudaMemcpy(h_input_sizes.data(), input_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    nvcompBatchedANSOpts_t comp_opts = nvcompBatchedANSDefaultOpts;

    size_t comp_temp_bytes;
    nvcompBatchedANSCompressGetTempSizeEx(batch_size, chunk_size, comp_opts, &comp_temp_bytes, total_bytes);

    void* d_comp_temp = nullptr;
    if (comp_temp_bytes > 0) cudaMalloc(&d_comp_temp, comp_temp_bytes);

    size_t max_out_bytes;
    nvcompBatchedANSCompressGetMaxOutputChunkSize(chunk_size, comp_opts, &max_out_bytes);

    BatchData compress_data(max_out_bytes, batch_size);

    nvcompStatus_t* d_comp_statuses;
    cudaMalloc(&d_comp_statuses, batch_size * sizeof(nvcompStatus_t));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedANSCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaStreamSynchronize(stream);
    }

    float total_comp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedANSCompressAsync(
            (const void* const*)input_data.ptrs(), input_data.sizes(),
            chunk_size, batch_size, d_comp_temp, comp_temp_bytes,
            compress_data.ptrs(), compress_data.sizes(), comp_opts, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_comp_time += ms;
    }
    result.compress_time_ms = total_comp_time / num_trials;

    std::vector<size_t> h_comp_sizes(batch_size);
    cudaMemcpy(h_comp_sizes.data(), compress_data.sizes(), batch_size * sizeof(size_t), cudaMemcpyDeviceToHost);

    size_t total_compressed = 0;
    for (auto s : h_comp_sizes) total_compressed += s;
    result.compressed_size = total_compressed;

    // Check for compression failure
    if (total_compressed == 0) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if (d_comp_temp) cudaFree(d_comp_temp);
        cudaFree(d_comp_statuses);
        cudaStreamDestroy(stream);
        delete[] raw_data;
        return result;
    }
    result.compression_ratio = (double)total_bytes / total_compressed;


    size_t decomp_temp_bytes;
    nvcompBatchedANSDecompressGetTempSizeEx(batch_size, chunk_size, &decomp_temp_bytes, total_bytes);

    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) cudaMalloc(&d_decomp_temp, decomp_temp_bytes);

    std::vector<void*> h_decomp_ptrs(batch_size);
    for (size_t i = 0; i < batch_size; i++) cudaMalloc(&h_decomp_ptrs[i], h_input_sizes[i]);
    void** d_decomp_ptrs; cudaMalloc(&d_decomp_ptrs, batch_size * sizeof(void*));
    cudaMemcpy(d_decomp_ptrs, h_decomp_ptrs.data(), batch_size * sizeof(void*), cudaMemcpyHostToDevice);

    size_t* d_decomp_sizes; cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t));
    nvcompStatus_t* d_decomp_statuses; cudaMalloc(&d_decomp_statuses, batch_size * sizeof(nvcompStatus_t));

    for (int i = 0; i < warmup_count; i++) {
        nvcompBatchedANSDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaStreamSynchronize(stream);
    }

    float total_decomp_time = 0;
    for (int trial = 0; trial < num_trials; trial++) {
        cudaEventRecord(start, stream);
        nvcompBatchedANSDecompressAsync(
            (const void* const*)compress_data.ptrs(), compress_data.sizes(),
            input_data.sizes(), d_decomp_sizes, batch_size,
            d_decomp_temp, decomp_temp_bytes, d_decomp_ptrs, d_decomp_statuses, stream);
        cudaEventRecord(stop, stream);
        cudaStreamSynchronize(stream);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_decomp_time += ms;
    }
    result.decompress_time_ms = total_decomp_time / num_trials;

    result.verified = true;
    std::vector<char> decomp_chunk(chunk_size);
    for (size_t i = 0; i < batch_size && result.verified; i++) {
        cudaMemcpy(decomp_chunk.data(), h_decomp_ptrs[i], h_input_sizes[i], cudaMemcpyDeviceToHost);
        if (memcmp(decomp_chunk.data(), chunks[i].data(), h_input_sizes[i]) != 0) result.verified = false;
    }

    result.compress_throughput_gbps = (total_bytes / 1e9) / (result.compress_time_ms / 1000.0);
    result.decompress_throughput_gbps = (total_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    if (d_comp_temp) cudaFree(d_comp_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    for (auto ptr : h_decomp_ptrs) cudaFree(ptr);
    cudaFree(d_decomp_ptrs); cudaFree(d_decomp_sizes);
    cudaFree(d_comp_statuses); cudaFree(d_decomp_statuses);
    cudaStreamDestroy(stream);
    delete[] raw_data;

    return result;
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n"
              << "Options:\n"
              << "  -d, --data_dir <path>    Data directory (default: /root/autodl-tmp/test/data/sosd/)\n"
              << "  -o, --output <file>      Output CSV file (default: reports/nvcomp_results.csv)\n"
              << "  -a, --algorithm <alg>    Algorithm: all|ans|bitcomp|cascaded|deflate|gdeflate|lz4|snappy|zstd\n"
              << "  -f, --file <filename>    Specific dataset file (can use multiple times)\n"
              << "  -p, --chunk_size <size>  Chunk size in bytes (default: 65536)\n"
              << "  -n, --trials <num>       Number of trials (default: 10)\n"
              << "  -w, --warmup <num>       Warmup iterations (default: 1)\n"
              << "  -g, --gpu <id>           GPU device ID (default: 0)\n"
              << "  -h, --help               Show this help message\n";
}

int main(int argc, char** argv) {
    std::string data_dir = "/root/autodl-tmp/test/data/sosd/";
    std::string output_file = "reports/nvcomp_results.csv";
    std::string algorithm = "all";
    std::vector<std::string> specific_files;
    size_t chunk_size = 65536;
    int num_trials = 10;
    int warmup_count = 1;
    int gpu_id = 0;

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
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            specific_files.push_back(argv[++i]);
        } else if ((arg == "-p" || arg == "--chunk_size") && i + 1 < argc) {
            chunk_size = std::stoull(argv[++i]);
        } else if ((arg == "-n" || arg == "--trials") && i + 1 < argc) {
            num_trials = std::stoi(argv[++i]);
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup_count = std::stoi(argv[++i]);
        } else if ((arg == "-g" || arg == "--gpu") && i + 1 < argc) {
            gpu_id = std::stoi(argv[++i]);
        }
    }

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
            "1-linear_200M_uint32.bin",
            "2-normal_200M_uint32.bin",
            "5-books_200M_uint32.bin",
            "9-movieid_uint32.bin",
            "14-cosmos_int32.bin",
            "18-site_250k_uint32.bin",
            "19-weight_25k_uint32.bin",
            "20-adult_30k_uint32.bin"
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

    std::cout << "nvcomp Benchmark (nvcomp 5.x API)\n";
    std::cout << "=================================\n";
    std::cout << "Algorithms: " << algorithm << "\n";
    std::cout << "Chunk size: " << chunk_size << "\n";
    std::cout << "Trials: " << num_trials << "\n";
    std::cout << "GPU: " << gpu_id << "\n";
    std::cout << "Datasets: " << datasets.size() << "\n\n";

    for (const auto& data_file : datasets) {
        std::cout << "Processing: " << data_file << std::endl;

        std::ifstream test_file(data_file);
        if (!test_file.good()) {
            std::cerr << "  File not found, skipping.\n";
            continue;
        }
        test_file.close();

        if (algorithm == "all" || algorithm == "lz4") {
            std::cout << "  Running LZ4..." << std::flush;
            BenchmarkResult result = benchmark_lz4(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "snappy") {
            std::cout << "  Running Snappy..." << std::flush;
            BenchmarkResult result = benchmark_snappy(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "zstd") {
            std::cout << "  Running Zstd..." << std::flush;
            BenchmarkResult result = benchmark_zstd(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "bitcomp") {
            std::cout << "  Running Bitcomp..." << std::flush;
            BenchmarkResult result = benchmark_bitcomp(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        if (algorithm == "all" || algorithm == "cascaded") {
            std::cout << "  Running Cascaded..." << std::flush;
            BenchmarkResult result = benchmark_cascaded(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }

        bool gdeflate_failed = false;
        if (algorithm == "all" || algorithm == "gdeflate") {
            std::cout << "  Running GDeflate..." << std::flush;
            BenchmarkResult result = benchmark_gdeflate(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
            gdeflate_failed = !result.verified;
            if (gdeflate_failed) {
                // Just sync the device to ensure clean state, don't reset
                cudaDeviceSynchronize();
            }
        }

        if (algorithm == "all" || algorithm == "deflate") {
            if (gdeflate_failed && algorithm == "all") {
                // Skip Deflate when GDeflate failed on same data (nvcomp internal state issue)
                std::cout << "  Running Deflate... SKIPPED (GDeflate failed)" << std::endl;
                BenchmarkResult result;
                result.framework = "nvcomp";
                result.algorithm = "Deflate";
                result.dataset = extract_dataset_name(data_file);
                result.verified = false;
                write_csv_result(csv_file, result);
            } else {
                std::cout << "  Running Deflate..." << std::flush;
                BenchmarkResult result = benchmark_deflate(data_file, chunk_size, num_trials, warmup_count);
                write_csv_result(csv_file, result);
                std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                          << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                          << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
            }
        }

        if (algorithm == "all" || algorithm == "ans") {
            std::cout << "  Running ANS..." << std::flush;
            BenchmarkResult result = benchmark_ans(data_file, chunk_size, num_trials, warmup_count);
            write_csv_result(csv_file, result);
            std::cout << " Ratio: " << std::fixed << std::setprecision(2) << result.compression_ratio
                      << ", Decomp: " << result.decompress_throughput_gbps << " GB/s"
                      << (result.verified ? " [OK]" : " [FAIL]") << std::endl;
        }
    }

    csv_file.close();
    std::cout << "\nResults written to: " << output_file << std::endl;

    return 0;
}
