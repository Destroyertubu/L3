/**
 * L3 Partition Strategy Interface
 *
 * This file defines the unified partitioning interface for L3 compression.
 * Users can choose between different partitioning strategies or implement custom ones.
 *
 * Available strategies:
 * - FixedSizePartitioner: Fixed-size partitions (simple, predictable)
 * - VariableLengthPartitioner: Adaptive partitioning based on data variance
 *
 * Example usage:
 *   auto config = l3::CompressionConfig();
 *   config.partition_strategy = l3::PartitionerFactory::FIXED_SIZE;
 *   config.partition_size_hint = 4096;
 *   auto* compressed = l3::compress(data, size, config);
 */

#ifndef L3_PARTITIONER_HPP
#define L3_PARTITIONER_HPP

#include <vector>
#include <memory>
#include <cstdint>
#include <cuda_runtime.h>

namespace l3 {

/**
 * Partition information structure
 * Represents a single partition with start and end indices
 */
struct PartitionInfo {
    int32_t start;  // Inclusive start index
    int32_t end;    // Exclusive end index

    PartitionInfo() : start(0), end(0) {}
    PartitionInfo(int32_t s, int32_t e) : start(s), end(e) {}

    int32_t size() const { return end - start; }
};

/**
 * Partition configuration
 */
struct PartitionConfig {
    // Common parameters
    int base_size = 4096;           // Base partition size

    // Variable-length partitioner parameters
    int variance_multiplier = 8;    // Variance block size multiplier
    int num_thresholds = 3;         // Number of variance thresholds

    // CUDA parameters
    cudaStream_t stream = 0;        // CUDA stream for async operations
};

/**
 * Abstract base class for partition strategies
 *
 * All partition strategies must inherit from this class and implement
 * the partition() method.
 */
class PartitionStrategy {
public:
    virtual ~PartitionStrategy() = default;

    /**
     * Partition the data
     *
     * @param data Pointer to data (can be on host or device)
     * @param size Number of elements
     * @param element_size Size of each element in bytes
     * @return Vector of PartitionInfo describing each partition
     */
    virtual std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) = 0;

    /**
     * Get strategy name
     */
    virtual const char* getName() const = 0;

    /**
     * Get strategy configuration
     */
    virtual PartitionConfig getConfig() const = 0;

    /**
     * Estimate number of partitions (optional, for pre-allocation)
     */
    virtual size_t estimatePartitions(size_t data_size) const {
        return (data_size + getConfig().base_size - 1) / getConfig().base_size;
    }
};

/**
 * Fixed-size partitioner
 *
 * Creates partitions of fixed size. Simple and predictable.
 *
 * Characteristics:
 * - All partitions have the same size (except possibly the last one)
 * - Fast partitioning (O(1) computation)
 * - Good for uniformly distributed data
 * - Predictable memory access patterns
 *
 * Best for:
 * - Data with uniform distribution
 * - When predictable partition sizes are needed
 * - When partitioning speed is critical
 */
class FixedSizePartitioner : public PartitionStrategy {
public:
    /**
     * Constructor
     *
     * @param partition_size Size of each partition
     * @param stream CUDA stream for operations
     */
    explicit FixedSizePartitioner(
        int partition_size = 4096,
        cudaStream_t stream = 0
    );

    std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) override;

    const char* getName() const override {
        return "FixedSize";
    }

    PartitionConfig getConfig() const override;

    size_t estimatePartitions(size_t data_size) const override;

private:
    int partition_size_;
    cudaStream_t stream_;
};

/**
 * Variable-length partitioner
 *
 * Creates adaptive partitions based on data variance.
 * High-variance regions get smaller partitions for better compression.
 * Low-variance regions get larger partitions for faster processing.
 *
 * Algorithm:
 * 1. Analyze data variance in blocks
 * 2. Sort variances and compute thresholds
 * 3. Assign partition sizes based on variance buckets:
 *    - High variance → small partitions
 *    - Low variance → large partitions
 *
 * Characteristics:
 * - Adaptive partition sizes
 * - Better compression for non-uniform data
 * - More complex partitioning (O(n) analysis + sorting)
 *
 * Best for:
 * - Data with non-uniform distribution
 * - When compression ratio is more important than speed
 * - Time series with varying patterns
 */
class VariableLengthPartitioner : public PartitionStrategy {
public:
    /**
     * Constructor
     *
     * @param base_size Base partition size (geometric progression base)
     * @param variance_multiplier Multiplier for variance analysis block size
     * @param num_thresholds Number of variance thresholds (buckets = thresholds + 1)
     * @param stream CUDA stream for operations
     */
    explicit VariableLengthPartitioner(
        int base_size = 1024,
        int variance_multiplier = 8,
        int num_thresholds = 3,
        cudaStream_t stream = 0
    );

    ~VariableLengthPartitioner();

    std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) override;

    const char* getName() const override {
        return "VariableLength";
    }

    PartitionConfig getConfig() const override;

private:
    int base_size_;
    int variance_multiplier_;
    int num_thresholds_;
    cudaStream_t stream_;

    // Internal implementation pointer (to hide CUDA details)
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Partition strategy factory
 *
 * Creates partition strategies by type with optional configuration.
 */
class PartitionerFactory {
public:
    /**
     * Strategy type enumeration
     */
    enum Strategy {
        FIXED_SIZE,       // Fixed-size partitions
        VARIABLE_LENGTH,  // Variable-length adaptive partitions
        AUTO              // Automatically choose best strategy
    };

    /**
     * Create a partitioner by strategy type
     *
     * @param strategy Strategy type
     * @param config Configuration parameters
     * @return Unique pointer to partition strategy
     */
    static std::unique_ptr<PartitionStrategy> create(
        Strategy strategy,
        const PartitionConfig& config = PartitionConfig()
    );

    /**
     * Automatically select best strategy based on data characteristics
     *
     * Analyzes data distribution and selects the most appropriate strategy:
     * - Uniform distribution → FixedSizePartitioner
     * - Non-uniform distribution → VariableLengthPartitioner
     *
     * @param data Pointer to data
     * @param size Number of elements
     * @param element_size Size of each element
     * @return Unique pointer to selected strategy
     */
    static std::unique_ptr<PartitionStrategy> createAuto(
        const void* data,
        size_t size,
        size_t element_size
    );

    /**
     * Get strategy name from enum
     */
    static const char* getStrategyName(Strategy strategy);
};

} // namespace l3

#endif // L3_PARTITIONER_HPP
