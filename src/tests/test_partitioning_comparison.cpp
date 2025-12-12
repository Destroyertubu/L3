/**
 * Test Cost-Aware Partitioning vs Original Variable-Length vs Fixed
 */

#include "L3_codec.hpp"
#include "L3_format.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

// External functions
template<typename T>
std::vector<PartitionInfo> createPartitionsVariableLength(
    const std::vector<T>& data, int base_partition_size, int* num_partitions_out,
    cudaStream_t stream, int variance_block_multiplier, int num_thresholds);

template<typename T>
std::vector<PartitionInfo> createPartitionsCostAware(
    const std::vector<T>& data, int base_partition_size, int* num_partitions_out,
    cudaStream_t stream);

template<typename T>
bool loadBinaryFile(const std::string& filename, std::vector<T>& data, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(T);
    if (max_elements > 0 && num_elements > max_elements) num_elements = max_elements;
    file.seekg(0, std::ios::beg);
    data.resize(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    return true;
}

template<typename T>
void analyzePartitions(const std::string& name, const std::vector<PartitionInfo>& partitions,
                       size_t data_size) {
    if (partitions.empty()) {
        std::cout << name << ": No partitions" << std::endl;
        return;
    }

    int original_bits = sizeof(T) * 8;
    int64_t total_bits = 0;
    int64_t total_elements = 0;
    int64_t weighted_delta_bits = 0;

    for (const auto& p : partitions) {
        int n = p.end_idx - p.start_idx;
        total_bits += (int64_t)p.delta_bits * n;
        total_elements += n;
        weighted_delta_bits += (int64_t)p.delta_bits * n;
    }

    // Calculate compressed size
    const double METADATA_PER_PARTITION = 64.0;  // bytes
    double metadata_bytes = partitions.size() * METADATA_PER_PARTITION;
    double delta_bytes = total_bits / 8.0;
    double compressed_bytes = metadata_bytes + delta_bytes;
    double original_bytes = data_size * sizeof(T);
    double ratio = original_bytes / compressed_bytes;
    double avg_delta_bits = (double)weighted_delta_bits / total_elements;

    std::cout << name << ":" << std::endl;
    std::cout << "  Partitions: " << partitions.size() << std::endl;
    std::cout << "  Avg partition size: " << (total_elements / partitions.size()) << std::endl;
    std::cout << "  Avg delta bits: " << std::fixed << std::setprecision(2) << avg_delta_bits << std::endl;
    std::cout << "  Metadata: " << std::fixed << std::setprecision(2) << (metadata_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Delta array: " << std::fixed << std::setprecision(2) << (delta_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Total compressed: " << std::fixed << std::setprecision(2) << (compressed_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Original: " << std::fixed << std::setprecision(2) << (original_bytes / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Compression ratio: " << std::fixed << std::setprecision(4) << ratio << "x" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Partitioning Strategy Comparison Test" << std::endl;
    std::cout << "========================================" << std::endl;

    // Test datasets
    std::vector<std::pair<std::string, std::string>> datasets = {
        {"/root/autodl-tmp/test/data/sosd/osm_cellids_800M_uint64.bin", "osm_800M (random)"},
        {"/root/autodl-tmp/test/data/sosd/books_200M_uint32.bin", "books_200M (sorted)"},
        {"/root/autodl-tmp/test/data/sosd/fb_200M_uint64.bin", "fb_200M (semi-sorted)"}
    };

    for (const auto& ds : datasets) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Dataset: " << ds.second << std::endl;
        std::cout << "========================================" << std::endl;

        // Try to load as uint64 first, then uint32
        std::vector<uint64_t> data64;
        std::vector<uint32_t> data32;

        bool is_64bit = ds.first.find("uint64") != std::string::npos;

        if (is_64bit) {
            if (!loadBinaryFile(ds.first, data64, 10000000)) {  // Limit to 10M for quick test
                std::cerr << "Failed to load " << ds.first << std::endl;
                continue;
            }
            std::cout << "Loaded " << data64.size() << " uint64 elements" << std::endl;

            // Test Fixed Partitioning
            int num_fixed = (data64.size() + 2047) / 2048;
            std::vector<PartitionInfo> fixed_parts;
            fixed_parts.reserve(num_fixed);
            for (int i = 0; i < num_fixed; i++) {
                PartitionInfo p;
                p.start_idx = i * 2048;
                p.end_idx = std::min((size_t)(i + 1) * 2048, data64.size());
                p.delta_bits = 64;  // Worst case
                fixed_parts.push_back(p);
            }

            // Use compressData to get actual delta bits for fixed
            CompressionStats stats;
            L3Config fixed_config = L3Config::fixedPartitioning(2048);
            auto* compressed = compressDataWithConfig(data64, fixed_config, &stats);
            std::cout << "\n[Fixed Partitioning (actual)]" << std::endl;
            std::cout << "  Partitions: " << stats.num_partitions << std::endl;
            std::cout << "  Avg delta bits: " << stats.avg_delta_bits << std::endl;
            std::cout << "  Compressed: " << (stats.compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  Ratio: " << stats.compression_ratio << "x" << std::endl;
            freeCompressedData(compressed);

            // Test Original Variable-Length
            std::cout << "\n[Original Variable-Length]" << std::endl;
            int num_var;
            auto var_parts = createPartitionsVariableLength<uint64_t>(data64, 2048, &num_var, 0, 8, 3);
            analyzePartitions<uint64_t>("Original Variable", var_parts, data64.size());

            // Test Cost-Aware
            std::cout << "[Cost-Aware Variable-Length]" << std::endl;
            int num_cost;
            auto cost_parts = createPartitionsCostAware<uint64_t>(data64, 2048, &num_cost, 0);
            analyzePartitions<uint64_t>("Cost-Aware", cost_parts, data64.size());

        } else {
            if (!loadBinaryFile(ds.first, data32, 10000000)) {
                std::cerr << "Failed to load " << ds.first << std::endl;
                continue;
            }
            // Skip header for books dataset
            if (ds.first.find("books") != std::string::npos && data32.size() > 1) {
                data32.erase(data32.begin());
            }
            std::cout << "Loaded " << data32.size() << " uint32 elements" << std::endl;

            // Use compressData to get actual stats for fixed
            CompressionStats stats;
            L3Config fixed_config = L3Config::fixedPartitioning(2048);
            auto* compressed = compressDataWithConfig(data32, fixed_config, &stats);
            std::cout << "\n[Fixed Partitioning (actual)]" << std::endl;
            std::cout << "  Partitions: " << stats.num_partitions << std::endl;
            std::cout << "  Avg delta bits: " << stats.avg_delta_bits << std::endl;
            std::cout << "  Compressed: " << (stats.compressed_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
            std::cout << "  Ratio: " << stats.compression_ratio << "x" << std::endl;
            freeCompressedData(compressed);

            // Test Original Variable-Length
            std::cout << "\n[Original Variable-Length]" << std::endl;
            int num_var;
            auto var_parts = createPartitionsVariableLength<uint32_t>(data32, 2048, &num_var, 0, 8, 3);
            analyzePartitions<uint32_t>("Original Variable", var_parts, data32.size());

            // Test Cost-Aware
            std::cout << "[Cost-Aware Variable-Length]" << std::endl;
            int num_cost;
            auto cost_parts = createPartitionsCostAware<uint32_t>(data32, 2048, &num_cost, 0);
            analyzePartitions<uint32_t>("Cost-Aware", cost_parts, data32.size());
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test completed." << std::endl;

    return 0;
}
