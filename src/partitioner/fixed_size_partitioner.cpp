/**
 * Fixed-Size Partitioner Implementation
 *
 * Creates uniform-sized partitions for L3 compression.
 * Simple, fast, and predictable.
 */

#include "l3/partitioner.hpp"
#include <algorithm>
#include <stdexcept>

namespace l3 {

FixedSizePartitioner::FixedSizePartitioner(int partition_size, cudaStream_t stream)
    : partition_size_(partition_size), stream_(stream)
{
    if (partition_size_ <= 0) {
        throw std::invalid_argument("Partition size must be positive");
    }
}

std::vector<PartitionInfo> FixedSizePartitioner::partition(
    const void* data,
    size_t size,
    size_t element_size)
{
    if (size == 0) {
        return std::vector<PartitionInfo>();
    }

    // Calculate number of partitions
    int num_partitions = (size + partition_size_ - 1) / partition_size_;

    std::vector<PartitionInfo> partitions;
    partitions.reserve(num_partitions);

    // Create partitions
    for (int i = 0; i < num_partitions; i++) {
        int32_t start = i * partition_size_;
        int32_t end = std::min((i + 1) * partition_size_, static_cast<int>(size));

        partitions.emplace_back(start, end);
    }

    return partitions;
}

PartitionConfig FixedSizePartitioner::getConfig() const {
    PartitionConfig config;
    config.base_size = partition_size_;
    config.stream = stream_;
    return config;
}

size_t FixedSizePartitioner::estimatePartitions(size_t data_size) const {
    return (data_size + partition_size_ - 1) / partition_size_;
}

} // namespace l3
