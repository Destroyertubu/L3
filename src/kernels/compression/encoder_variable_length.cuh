/**
 * Variable-Length Encoder Header for GLECO
 */

#ifndef ENCODER_VARIABLE_LENGTH_CUH
#define ENCODER_VARIABLE_LENGTH_CUH

#include <vector>
#include <cuda_runtime.h>
#include "L3_format.hpp"

// Forward declaration of the Variable-Length Partitioner class
template<typename T>
class GPUVariableLengthPartitionerV6;

/**
 * Create partitions using variable-length adaptive partitioning.
 *
 * @param data Input data vector
 * @param base_partition_size Base partition size (will be adapted based on variance)
 * @param num_partitions_out Output: number of partitions created
 * @param stream CUDA stream (optional)
 * @param variance_block_multiplier Multiplier for variance analysis block size
 * @param num_thresholds Number of variance thresholds to use
 * @return Vector of PartitionInfo structures
 */
template<typename T>
std::vector<PartitionInfo> createPartitionsVariableLength(
    const std::vector<T>& data,
    int base_partition_size,
    int* num_partitions_out,
    cudaStream_t stream = 0,
    int variance_block_multiplier = 8,
    int num_thresholds = 3);

#endif // ENCODER_VARIABLE_LENGTH_CUH
