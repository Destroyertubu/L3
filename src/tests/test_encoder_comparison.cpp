/**
 * L3 Encoder Comparison Test
 *
 * 对比两种编码器：
 * 1. 固定长度分区 (Fixed-Length) - 现有encoder
 * 2. 变长分区 (Variable-Length) - L32方法
 *
 * 测试数据：data/fb_200M_uint64.bin
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include "L3_codec.hpp"
#include "L3_format.hpp"

// Forward declaration for variable-length encoder
template<typename T>
std::vector<PartitionInfo> createPartitionsVariableLength(
    const std::vector<T>& data,
    int base_partition_size,
    int* num_partitions_out,
    cudaStream_t stream = 0,
    int variance_block_multiplier = 8,
    int num_thresholds = 3);

// 外部声明Phase 2 Bucket解压缩函数
template<typename T>
void decompressL3_Phase2_Bucket(
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const uint8_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    const uint32_t* delta_array,
    int num_partitions,
    int total_elements,
    T* output);

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 加载二进制数据
std::vector<uint64_t> loadBinaryFile(const char* filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error: Failed to open file: " << filename << std::endl;
        exit(1);
    }

    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = file_size / sizeof(uint64_t);
    if (max_elements > 0 && num_elements > max_elements) {
        num_elements = max_elements;
    }

    std::vector<uint64_t> data(num_elements);
    if (!file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint64_t))) {
        std::cerr << "Error: Failed to read data from file" << std::endl;
        exit(1);
    }

    return data;
}

// 验证解压缩正确性
bool verifyDecompression(const std::vector<uint64_t>& original,
                        const std::vector<uint64_t>& decompressed,
                        const char* method_name) {
    if (original.size() != decompressed.size()) {
        std::cerr << "[" << method_name << "] Size mismatch: original=" << original.size()
                  << ", decompressed=" << decompressed.size() << std::endl;
        return false;
    }

    size_t errors = 0;
    for (size_t i = 0; i < original.size(); i++) {
        if (original[i] != decompressed[i]) {
            if (errors < 5) {
                std::cerr << "[" << method_name << "] Mismatch at index " << i << ": original="
                          << original[i] << ", decompressed=" << decompressed[i] << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cerr << "[" << method_name << "] Total errors: " << errors << " / " << original.size() << std::endl;
        return false;
    }

    return true;
}

// 测试一个配置
struct TestResult {
    std::string method_name;
    int partition_size;
    size_t original_bytes;
    size_t compressed_bytes;
    double compression_ratio;
    int num_partitions;
    double compress_time_ms;
    double decompress_time_ms;
    double decompress_throughput_gbps;
    bool correctness_pass;
};

TestResult testFixedLength(const std::vector<uint64_t>& data, int partition_size) {
    TestResult result;
    result.method_name = "Fixed-Length";
    result.partition_size = partition_size;
    result.original_bytes = data.size() * sizeof(uint64_t);

    // 压缩
    auto compress_start = std::chrono::high_resolution_clock::now();
    CompressionStats comp_stats;
    auto* compressed = compressData(data, partition_size, &comp_stats);
    auto compress_end = std::chrono::high_resolution_clock::now();

    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        result.correctness_pass = false;
        return result;
    }

    result.compress_time_ms = std::chrono::duration<double, std::milli>(
        compress_end - compress_start).count();
    result.compressed_bytes = comp_stats.compressed_bytes;
    result.compression_ratio = comp_stats.compression_ratio;
    result.num_partitions = comp_stats.num_partitions;

    // 准备解压缩
    std::vector<uint8_t> h_delta_bits(compressed->num_partitions);
    std::vector<int32_t> h_delta_bits_i32(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits_i32.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < compressed->num_partitions; ++i) {
        h_delta_bits[i] = static_cast<uint8_t>(h_delta_bits_i32[i]);
    }

    uint8_t* d_delta_bits_u8;
    CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits.data(),
                         compressed->num_partitions * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));

    // 解压缩 (多次测试取平均)
    std::vector<float> decompress_times;
    const int num_iters = 20;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < num_iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressL3_Phase2_Bucket<uint64_t>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        decompress_times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 计算中位数解压缩时间
    std::sort(decompress_times.begin(), decompress_times.end());
    result.decompress_time_ms = decompress_times[decompress_times.size() / 2];
    result.decompress_throughput_gbps = (result.original_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    // 验证正确性
    std::vector<uint64_t> decompressed(data.size());
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    result.correctness_pass = verifyDecompression(data, decompressed, result.method_name.c_str());

    // 清理
    CUDA_CHECK(cudaFree(d_delta_bits_u8));
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return result;
}

// 测试变长分区方法
TestResult testVariableLength(const std::vector<uint64_t>& data, int base_partition_size) {
    TestResult result;
    result.method_name = "Variable-Length";
    result.partition_size = base_partition_size;
    result.original_bytes = data.size() * sizeof(uint64_t);

    // 使用变长编码器创建分区
    auto compress_start = std::chrono::high_resolution_clock::now();

    int num_partitions = 0;
    std::vector<PartitionInfo> partitions;

    try {
        partitions = createPartitionsVariableLength(data, base_partition_size, &num_partitions);
    } catch (const std::exception& e) {
        std::cerr << "Variable-length partitioning failed: " << e.what() << std::endl;
        result.correctness_pass = false;
        return result;
    }

    if (partitions.empty()) {
        std::cerr << "No partitions created!" << std::endl;
        result.correctness_pass = false;
        return result;
    }

    result.num_partitions = num_partitions;

    // 使用预先计算好的变长分区进行压缩
    CompressionStats comp_stats;
    auto* compressed = compressDataWithPartitions(data, partitions, &comp_stats);

    auto compress_end = std::chrono::high_resolution_clock::now();

    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        result.correctness_pass = false;
        return result;
    }

    result.compress_time_ms = std::chrono::duration<double, std::milli>(
        compress_end - compress_start).count();
    result.compressed_bytes = comp_stats.compressed_bytes;
    result.compression_ratio = comp_stats.compression_ratio;

    // 准备解压缩
    std::vector<uint8_t> h_delta_bits(compressed->num_partitions);
    std::vector<int32_t> h_delta_bits_i32(compressed->num_partitions);
    CUDA_CHECK(cudaMemcpy(h_delta_bits_i32.data(), compressed->d_delta_bits,
                         compressed->num_partitions * sizeof(int32_t),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < compressed->num_partitions; ++i) {
        h_delta_bits[i] = static_cast<uint8_t>(h_delta_bits_i32[i]);
    }

    uint8_t* d_delta_bits_u8;
    CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits.data(),
                         compressed->num_partitions * sizeof(uint8_t),
                         cudaMemcpyHostToDevice));

    uint64_t* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, data.size() * sizeof(uint64_t)));

    // 解压缩 (多次测试取平均)
    std::vector<float> decompress_times;
    const int num_iters = 20;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < num_iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));

        decompressL3_Phase2_Bucket<uint64_t>(
            compressed->d_start_indices,
            compressed->d_end_indices,
            compressed->d_model_types,
            compressed->d_model_params,
            d_delta_bits_u8,
            compressed->d_delta_array_bit_offsets,
            compressed->delta_array,
            compressed->num_partitions,
            compressed->total_values,
            d_output
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        decompress_times.push_back(ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // 计算中位数解压缩时间
    std::sort(decompress_times.begin(), decompress_times.end());
    result.decompress_time_ms = decompress_times[decompress_times.size() / 2];
    result.decompress_throughput_gbps = (result.original_bytes / 1e9) / (result.decompress_time_ms / 1000.0);

    // 验证正确性
    std::vector<uint64_t> decompressed(data.size());
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_output,
                         data.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    result.correctness_pass = verifyDecompression(data, decompressed, result.method_name.c_str());

    // 清理
    CUDA_CHECK(cudaFree(d_delta_bits_u8));
    CUDA_CHECK(cudaFree(d_output));
    freeCompressedData(compressed);

    return result;
}

void printResult(const TestResult& result) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Method: " << result.method_name << std::endl;
    std::cout << "Partition Size: " << result.partition_size << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Original Size:        " << std::setw(12) << result.original_bytes
              << " bytes (" << std::fixed << std::setprecision(2)
              << (result.original_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Compressed Size:      " << std::setw(12) << result.compressed_bytes
              << " bytes (" << std::fixed << std::setprecision(2)
              << (result.compressed_bytes / 1024.0 / 1024.0) << " MB)" << std::endl;
    std::cout << "Compression Ratio:    " << std::setw(12) << std::fixed
              << std::setprecision(2) << result.compression_ratio << "x" << std::endl;
    std::cout << "Partitions:           " << std::setw(12) << result.num_partitions << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Compression Time:     " << std::setw(12) << std::fixed
              << std::setprecision(3) << result.compress_time_ms << " ms" << std::endl;
    std::cout << "Decompression Time:   " << std::setw(12) << std::fixed
              << std::setprecision(3) << result.decompress_time_ms << " ms" << std::endl;
    std::cout << "Decompression Thru:   " << std::setw(12) << std::fixed
              << std::setprecision(2) << result.decompress_throughput_gbps << " GB/s" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Correctness:          " << (result.correctness_pass ? "✅ PASS" : "❌ FAIL") << std::endl;
    std::cout << std::string(80, '=') << std::endl;
}

void printComparison(const std::vector<TestResult>& results) {
    std::cout << "\n\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPRESSION METHOD COMPARISON" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::cout << std::left << std::setw(20) << "Method"
              << std::right << std::setw(10) << "Part.Size"
              << std::setw(12) << "Ratio"
              << std::setw(10) << "Parts"
              << std::setw(12) << "Comp(ms)"
              << std::setw(12) << "Decomp(GB/s)"
              << "  " << "OK" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (const auto& r : results) {
        std::cout << std::left << std::setw(20) << r.method_name
                  << std::right << std::setw(10) << r.partition_size
                  << std::setw(12) << std::fixed << std::setprecision(2) << r.compression_ratio
                  << std::setw(10) << r.num_partitions
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.compress_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << r.decompress_throughput_gbps
                  << "  " << (r.correctness_pass ? "✅" : "❌") << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
}

int main(int argc, char** argv) {
    const char* default_file = "data/fb_200M_uint64.bin";
    const char* input_file = (argc > 1) ? argv[1] : default_file;
    size_t max_elements = (argc > 2) ? std::stoull(argv[2]) : 0;

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "L3 Encoder Comparison Test" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Input file: " << input_file << std::endl;
    if (max_elements > 0) {
        std::cout << "Max elements: " << max_elements << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;

    // 加载数据
    std::cout << "\nLoading data..." << std::flush;
    auto data = loadBinaryFile(input_file, max_elements);
    std::cout << " done (" << data.size() << " elements, "
              << (data.size() * sizeof(uint64_t) / 1024.0 / 1024.0) << " MB)" << std::endl;

    std::vector<TestResult> results;

    // 测试不同的固定长度分区大小
    std::vector<int> partition_sizes = {2048, 4096, 8192, 16384};

    for (int ps : partition_sizes) {
        std::cout << "\n>>> Testing Fixed-Length with partition_size=" << ps << "..." << std::endl;
        TestResult result = testFixedLength(data, ps);
        printResult(result);
        results.push_back(result);
    }

    // 测试变长分区方法
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Testing Variable-Length Encoder" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    std::vector<int> vl_base_sizes = {2048, 4096};
    for (int bs : vl_base_sizes) {
        std::cout << "\n>>> Testing Variable-Length with base_size=" << bs << "..." << std::endl;
        TestResult result = testVariableLength(data, bs);
        printResult(result);
        results.push_back(result);
    }

    // 打印对比表
    printComparison(results);

    std::cout << "\n✅ Encoder comparison test complete!\n" << std::endl;

    return 0;
}
