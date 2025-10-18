/**
 * Example: Using Different Partition Strategies
 *
 * This example demonstrates how to use L3's partition strategy interface
 * to choose between fixed-size and variable-length partitioning.
 *
 * Compile:
 *   mkdir build && cd build
 *   cmake .. -DBUILD_EXAMPLES=ON
 *   make
 *
 * Run:
 *   ./examples/01_partition_strategies
 */

#include "l3/partitioner.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

// Generate test data with varying distribution
std::vector<int64_t> generateTestData(size_t size) {
    std::vector<int64_t> data(size);
    std::random_device rd;
    std::mt19937_64 gen(rd());

    // Create data with mixed patterns:
    // - First half: low variance (sorted)
    // - Second half: high variance (random)
    for (size_t i = 0; i < size / 2; i++) {
        data[i] = i * 100;  // Low variance
    }
    std::uniform_int_distribution<int64_t> dist(0, 1000000);
    for (size_t i = size / 2; i < size; i++) {
        data[i] = dist(gen);  // High variance
    }

    return data;
}

// Print partition statistics
void printPartitionStats(const std::vector<l3::PartitionInfo>& partitions,
                        const char* strategy_name) {
    std::cout << "\n========================================\n";
    std::cout << "Strategy: " << strategy_name << "\n";
    std::cout << "========================================\n";
    std::cout << "Number of partitions: " << partitions.size() << "\n";

    // Calculate statistics
    std::vector<int> sizes;
    sizes.reserve(partitions.size());
    for (const auto& p : partitions) {
        sizes.push_back(p.size());
    }

    int min_size = *std::min_element(sizes.begin(), sizes.end());
    int max_size = *std::max_element(sizes.begin(), sizes.end());
    double avg_size = 0.0;
    for (int s : sizes) avg_size += s;
    avg_size /= sizes.size();

    std::cout << "Partition size - Min: " << min_size
              << ", Max: " << max_size
              << ", Avg: " << std::fixed << std::setprecision(1) << avg_size << "\n";

    // Show first few partitions
    std::cout << "\nFirst 5 partitions:\n";
    for (size_t i = 0; i < std::min<size_t>(5, partitions.size()); i++) {
        std::cout << "  [" << i << "] start=" << partitions[i].start
                  << ", end=" << partitions[i].end
                  << ", size=" << partitions[i].size() << "\n";
    }
}

int main() {
    std::cout << "L3 Partition Strategy Example\n";
    std::cout << "===============================\n\n";

    // Generate test data
    const size_t DATA_SIZE = 100000;
    std::cout << "Generating test data (" << DATA_SIZE << " elements)...\n";
    auto data = generateTestData(DATA_SIZE);

    // ========================================================================
    // Method 1: Using FixedSizePartitioner directly
    // ========================================================================
    {
        std::cout << "\n[Method 1] Direct instantiation - Fixed-size partitions\n";

        l3::FixedSizePartitioner partitioner(4096);  // 4K elements per partition
        auto partitions = partitioner.partition(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        printPartitionStats(partitions, partitioner.getName());
    }

    // ========================================================================
    // Method 2: Using VariableLengthPartitioner directly
    // ========================================================================
    {
        std::cout << "\n[Method 2] Direct instantiation - Variable-length partitions\n";

        l3::VariableLengthPartitioner partitioner(
            1024,  // base_size
            8,     // variance_multiplier
            3      // num_thresholds
        );

        auto partitions = partitioner.partition(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        printPartitionStats(partitions, partitioner.getName());
    }

    // ========================================================================
    // Method 3: Using Factory with FIXED_SIZE strategy
    // ========================================================================
    {
        std::cout << "\n[Method 3] Factory pattern - FIXED_SIZE\n";

        l3::PartitionConfig config;
        config.base_size = 2048;

        auto partitioner = l3::PartitionerFactory::create(
            l3::PartitionerFactory::FIXED_SIZE,
            config
        );

        auto partitions = partitioner->partition(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        printPartitionStats(partitions, partitioner->getName());
    }

    // ========================================================================
    // Method 4: Using Factory with VARIABLE_LENGTH strategy
    // ========================================================================
    {
        std::cout << "\n[Method 4] Factory pattern - VARIABLE_LENGTH\n";

        l3::PartitionConfig config;
        config.base_size = 1024;
        config.variance_multiplier = 16;
        config.num_thresholds = 5;

        auto partitioner = l3::PartitionerFactory::create(
            l3::PartitionerFactory::VARIABLE_LENGTH,
            config
        );

        auto partitions = partitioner->partition(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        printPartitionStats(partitions, partitioner->getName());
    }

    // ========================================================================
    // Method 5: Using Factory with AUTO strategy
    // ========================================================================
    {
        std::cout << "\n[Method 5] Factory pattern - AUTO (automatic selection)\n";

        auto partitioner = l3::PartitionerFactory::createAuto(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        auto partitions = partitioner->partition(
            data.data(),
            data.size(),
            sizeof(int64_t)
        );

        printPartitionStats(partitions, partitioner->getName());
    }

    // ========================================================================
    // Comparison
    // ========================================================================
    std::cout << "\n========================================\n";
    std::cout << "Summary\n";
    std::cout << "========================================\n";
    std::cout << "Fixed-size partitioning:\n";
    std::cout << "  + Simple and fast\n";
    std::cout << "  + Predictable partition sizes\n";
    std::cout << "  + Good for uniform data\n";
    std::cout << "  - May not be optimal for non-uniform data\n\n";

    std::cout << "Variable-length partitioning:\n";
    std::cout << "  + Adaptive to data distribution\n";
    std::cout << "  + Better compression for non-uniform data\n";
    std::cout << "  + High-variance → small partitions\n";
    std::cout << "  + Low-variance → large partitions\n";
    std::cout << "  - More complex and slower partitioning\n\n";

    std::cout << "Choose the strategy based on your data characteristics!\n";

    return 0;
}
