/**
 * Full test on OSM 800M with Cost-Aware partitioning
 */

#include "L3_codec.hpp"
#include "L3_format.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <cmath>

template<typename T>
std::vector<PartitionInfo> createPartitionsCostAware(
    const std::vector<T>& data, int base_partition_size, int* num_partitions_out,
    cudaStream_t stream);

template<typename T>
bool loadBinaryFile(const std::string& filename, std::vector<T>& data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(T);
    file.seekg(0, std::ios::beg);
    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "OSM 800M Full Dataset Test" << std::endl;
    std::cout << "========================================" << std::endl;

    std::vector<uint64_t> data;
    if (!loadBinaryFile("/root/autodl-tmp/test/data/sosd/osm_cellids_800M_uint64.bin", data)) {
        std::cerr << "Failed to load data" << std::endl;
        return 1;
    }

    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Original size: " << (data.size() * sizeof(uint64_t) / 1024.0 / 1024.0) << " MB" << std::endl;

    // Test Fixed Partitioning
    std::cout << "\n[1. Fixed Partitioning]" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        CompressionStats stats;
        L3Config config = L3Config::fixedPartitioning(2048);
        auto* compressed = compressDataWithConfig(data, config, &stats);
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "  Partitions: " << stats.num_partitions << std::endl;
        std::cout << "  Avg delta bits: " << stats.avg_delta_bits << std::endl;
        std::cout << "  Compressed: " << (stats.compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Ratio: " << stats.compression_ratio << "x" << std::endl;
        std::cout << "  Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

        // Verify decompression
        std::vector<uint64_t> decompressed;
        decompressData(compressed, decompressed);
        bool correct = (decompressed.size() == data.size());
        for (size_t i = 0; i < data.size() && correct; i++) {
            if (data[i] != decompressed[i]) correct = false;
        }
        std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

        freeCompressedData(compressed);
    }

    // Test Cost-Aware Partitioning
    std::cout << "\n[2. Cost-Aware Partitioning]" << std::endl;
    {
        auto start = std::chrono::high_resolution_clock::now();
        int num_partitions;
        auto partitions = createPartitionsCostAware<uint64_t>(data, 2048, &num_partitions, 0);
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate statistics
        int64_t total_bits = 0;
        int64_t weighted_delta_bits = 0;
        int64_t total_elements = 0;

        for (const auto& p : partitions) {
            int n = p.end_idx - p.start_idx;
            total_bits += (int64_t)p.delta_bits * n;
            weighted_delta_bits += (int64_t)p.delta_bits * n;
            total_elements += n;
        }

        const double METADATA_PER_PARTITION = 64.0;
        double metadata_bytes = partitions.size() * METADATA_PER_PARTITION;
        double delta_bytes = total_bits / 8.0;
        double compressed_bytes = metadata_bytes + delta_bytes;
        double original_bytes = data.size() * sizeof(uint64_t);
        double ratio = original_bytes / compressed_bytes;
        double avg_delta_bits = (double)weighted_delta_bits / total_elements;

        std::cout << "  Partitions: " << partitions.size() << std::endl;
        std::cout << "  Avg partition size: " << (total_elements / partitions.size()) << std::endl;
        std::cout << "  Avg delta bits: " << avg_delta_bits << std::endl;
        std::cout << "  Metadata: " << (metadata_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Delta array: " << (delta_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total compressed: " << (compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Ratio: " << ratio << "x" << std::endl;
        std::cout << "  Partition time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

        // Verify by compressing with these partitions
        CompressionStats stats;
        auto* compressed = compressDataWithPartitions(data, partitions, &stats);
        std::cout << "  Actual compressed: " << (stats.compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Actual ratio: " << stats.compression_ratio << "x" << std::endl;

        // Verify decompression
        std::vector<uint64_t> decompressed;
        decompressData(compressed, decompressed);
        bool correct = (decompressed.size() == data.size());
        for (size_t i = 0; i < data.size() && correct; i++) {
            if (data[i] != decompressed[i]) correct = false;
        }
        std::cout << "  Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

        freeCompressedData(compressed);
    }

    // Debug: Check a few partitions
    std::cout << "\n[3. Debug: Sample partition analysis]" << std::endl;
    {
        int num_partitions;
        auto partitions = createPartitionsCostAware<uint64_t>(data, 2048, &num_partitions, 0);

        for (int i = 0; i < 5 && i < (int)partitions.size(); i++) {
            const auto& p = partitions[i];
            std::cout << "  P" << i << ": [" << p.start_idx << ", " << p.end_idx << ")"
                      << " len=" << (p.end_idx - p.start_idx)
                      << " delta_bits=" << p.delta_bits
                      << " theta0=" << p.model_params[0]
                      << " theta1=" << p.model_params[1]
                      << std::endl;

            // Manually verify delta bits for first partition
            if (i == 0) {
                int n = p.end_idx - p.start_idx;
                long long max_err = 0;
                for (int j = 0; j < n; j++) {
                    double predicted = p.model_params[0] + p.model_params[1] * j;
                    uint64_t pred_u = static_cast<uint64_t>(round(predicted));
                    long long delta = static_cast<long long>(data[p.start_idx + j]) - static_cast<long long>(pred_u);
                    max_err = std::max(max_err, std::llabs(delta));
                }
                int computed_bits = (max_err > 0) ? (64 - __builtin_clzll(max_err) + 1) : 0;
                std::cout << "    Manual check: max_err=" << max_err << " computed_bits=" << computed_bits << std::endl;
            }
        }
    }

    return 0;
}
