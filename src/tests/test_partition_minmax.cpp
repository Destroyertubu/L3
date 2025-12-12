/**
 * 测试分区Min/Max值是否正确记录
 *
 * 验证：
 * 1. 固定长度分区的min/max
 * 2. 变长分区的min/max
 * 3. 两者都应该反映实际数据的min/max
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cstdint>
#include "L3_codec.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Forward declaration
template<typename T>
std::vector<PartitionInfo> createPartitionsVariableLength(
    const std::vector<T>& data,
    int base_partition_size,
    int* num_partitions_out,
    cudaStream_t stream = 0,
    int variance_block_multiplier = 8,
    int num_thresholds = 3);

void testMinMax(const std::vector<uint64_t>& data, const char* test_name) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Test: " << test_name << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 压缩数据
    CompressionStats stats;
    CompressedDataL3<uint64_t>* compressed = nullptr;

    if (std::string(test_name).find("Fixed") != std::string::npos) {
        compressed = compressData(data, 2048, &stats);
    } else {
        int num_partitions = 0;
        auto partitions = createPartitionsVariableLength(data, 2048, &num_partitions);
        compressed = compressDataWithPartitions(data, partitions, &stats);
    }

    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return;
    }

    int num_partitions = compressed->num_partitions;
    std::cout << "Number of partitions: " << num_partitions << std::endl;

    // 下载分区边界和min/max
    std::vector<int32_t> h_start(num_partitions);
    std::vector<int32_t> h_end(num_partitions);
    std::vector<uint64_t> h_min(num_partitions);
    std::vector<uint64_t> h_max(num_partitions);

    CUDA_CHECK(cudaMemcpy(h_start.data(), compressed->d_start_indices,
                         num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end.data(), compressed->d_end_indices,
                         num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_min.data(), compressed->d_partition_min_values,
                         num_partitions * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_max.data(), compressed->d_partition_max_values,
                         num_partitions * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // 验证前5个分区
    int check_count = std::min(5, num_partitions);
    std::cout << "\nVerifying first " << check_count << " partitions:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    bool all_correct = true;
    for (int i = 0; i < check_count; i++) {
        int start = h_start[i];
        int end = h_end[i];

        // 从原始数据计算真实的min/max
        uint64_t true_min = data[start];
        uint64_t true_max = data[start];
        for (int j = start; j < end; j++) {
            if (data[j] < true_min) true_min = data[j];
            if (data[j] > true_max) true_max = data[j];
        }

        uint64_t recorded_min = h_min[i];
        uint64_t recorded_max = h_max[i];

        bool min_match = (recorded_min == true_min);
        bool max_match = (recorded_max == true_max);

        std::cout << "Partition " << i << ": [" << start << ", " << end << ") "
                  << "len=" << (end - start) << std::endl;
        std::cout << "  MIN: recorded=" << recorded_min << " true=" << true_min
                  << (min_match ? " ✅" : " ❌") << std::endl;
        std::cout << "  MAX: recorded=" << recorded_max << " true=" << true_max
                  << (max_match ? " ✅" : " ❌") << std::endl;

        if (!min_match || !max_match) {
            all_correct = false;
        }
    }

    std::cout << std::string(80, '-') << std::endl;
    std::cout << "Result: " << (all_correct ? "✅ ALL CORRECT" : "❌ ERRORS FOUND") << std::endl;

    freeCompressedData(compressed);
}

int main() {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Partition Min/Max Verification Test" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    // 创建测试数据：有明显的分段特征
    const int N = 20000;
    std::vector<uint64_t> data(N);

    for (int i = 0; i < N; i++) {
        // 创建分段数据，每个段有不同的值范围
        int segment = i / 2000;
        uint64_t base = segment * 1000000;
        data[i] = base + (i % 2000);
    }

    std::cout << "\nTest data: " << N << " elements with segmented values" << std::endl;
    std::cout << "Segment pattern: each 2000 elements have base value offset by 1000000" << std::endl;

    // 测试固定长度分区
    testMinMax(data, "Fixed-Length Partitions");

    // 测试变长分区
    testMinMax(data, "Variable-Length Partitions");

    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "✅ Min/Max verification test complete!" << std::endl;
    std::cout << std::string(80, '=') << std::endl;

    return 0;
}
