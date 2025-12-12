/**
 * L3 Unified Codec Interface
 *
 * This header provides a clean, simplified API for L3 compression and
 * decompression that ensures encoder and decoder remain perfectly matched.
 *
 * PORT DATE: 2025-10-14
 * FORMAT VERSION: 1.0.0
 *
 * PRINCIPLE: Self-consistent codec
 * - Encoder and decoder use identical algorithms
 * - Any bitstream produced by encoder MUST be decodable by decoder
 * - Roundtrip correctness: decompress(compress(data)) == data
 */

#ifndef L3_CODEC_HPP
#define L3_CODEC_HPP

#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "L3_format.hpp"

// ============================================================================
// Compression Result Structure
// ============================================================================

/**
 * Compression statistics and metadata
 */
struct CompressionStats {
    int64_t original_bytes;
    int64_t compressed_bytes;
    double compression_ratio;
    double compression_time_ms;
    double compression_throughput_gbps;
    int num_partitions;
    int avg_delta_bits;
    int64_t total_bits_used;
};

// ============================================================================
// Encoder API
// ============================================================================

/**
 * Compress data on GPU using L3 algorithm
 *
 * PARAMETERS:
 * - h_data: Host array of data to compress (vector or pointer + size)
 * - partition_size: Target partition size (adaptive) - typical: 1024-4096
 * - stats: Optional output for compression statistics
 *
 * RETURNS:
 * - CompressedDataL3<T> structure with all metadata and delta arrays on device
 *
 * ALGORITHM:
 * 1. Upload data to GPU
 * 2. Partition data (fixed-size or adaptive)
 * 3. Fit models per partition (linear, polynomial, or direct copy)
 * 4. Compute deltas and determine bit width per partition
 * 5. Pack deltas into bit array
 * 6. Return compressed structure (device memory)
 *
 * MEMORY: Caller must free returned structure using freeCompressedData()
 *
 * INVARIANT: Output bitstream MUST be decodable by decompressData()
 */
template<typename T>
CompressedDataL3<T>* compressData(
    const std::vector<T>& h_data,
    int partition_size = 2048,
    CompressionStats* stats = nullptr
);

template<typename T>
CompressedDataL3<T>* compressData(
    const T* h_data,
    int num_elements,
    int partition_size = 2048,
    CompressionStats* stats = nullptr
);

// PartitionInfo is defined in L3_format.hpp

/**
 * Compress data using precomputed variable-length partitions
 *
 * This function accepts partitions that have already been created by the
 * variable-length encoder (with fitted models and computed delta bits).
 * It skips the partition creation and model fitting steps.
 */
template<typename T>
CompressedDataL3<T>* compressDataWithPartitions(
    const std::vector<T>& h_data,
    const std::vector<PartitionInfo>& partitions,
    CompressionStats* stats = nullptr
);

template<typename T>
CompressedDataL3<T>* compressDataWithPartitions(
    const T* h_data,
    int num_elements,
    const std::vector<PartitionInfo>& partitions,
    CompressionStats* stats = nullptr
);

/**
 * Free compressed data structure (both host and device memory)
 */
template<typename T>
void freeCompressedData(CompressedDataL3<T>* compressed);

// ============================================================================
// Decoder API
// ============================================================================

/**
 * Decompression statistics
 */
struct DecompressionStats {
    int64_t compressed_bytes;
    int64_t decompressed_bytes;
    double decompression_time_ms;
    double decompression_throughput_gbps;
    int total_elements;
};

/**
 * Decompress data on GPU
 *
 * PARAMETERS:
 * - compressed: CompressedDataL3 structure (device memory)
 * - h_output: Host buffer to receive decompressed data (must be pre-allocated)
 * - stats: Optional output for decompression statistics
 *
 * RETURNS:
 * - Number of elements decompressed
 *
 * ALGORITHM:
 * 1. Allocate device output array
 * 2. Launch decompression kernel (1 thread per element)
 * 3. Download decompressed data to host
 * 4. Free device output array
 *
 * INVARIANT: For all valid compressed data:
 *   decompress(compress(data)) == data (bit-for-bit)
 */
template<typename T>
int decompressData(
    const CompressedDataL3<T>* compressed,
    std::vector<T>& h_output,
    DecompressionStats* stats = nullptr
);

template<typename T>
int decompressData(
    const CompressedDataL3<T>* compressed,
    T* h_output,
    int output_capacity,
    DecompressionStats* stats = nullptr
);

// ============================================================================
// Roundtrip Test Utilities
// ============================================================================

/**
 * Test roundtrip correctness: compress → decompress → verify
 *
 * RETURNS: true if bit-for-bit identical, false otherwise
 *
 * USAGE:
 *   std::vector<uint32_t> data = {...};
 *   bool correct = testRoundtrip(data);
 *   assert(correct);
 */
template<typename T>
bool testRoundtrip(
    const std::vector<T>& original_data,
    int partition_size = 2048,
    bool verbose = false
);

/**
 * Verify compressed data integrity
 *
 * Checks:
 * - Metadata consistency (partitions non-overlapping, sorted)
 * - Bit offset calculation correctness
 * - Delta array size sufficient
 *
 * RETURNS: true if valid, false if corrupted
 */
template<typename T>
bool verifyCompressedData(const CompressedDataL3<T>* compressed);

// ============================================================================
// Partitioning Strategies
// ============================================================================

/**
 * Fixed-size partitioning (simple, predictable)
 *
 * PARAMETERS:
 * - d_data: Device array
 * - num_elements: Total elements
 * - partition_size: Elements per partition
 * - d_start_indices, d_end_indices: Output arrays (pre-allocated)
 *
 * RETURNS: num_partitions
 */
template<typename T>
int createFixedSizePartitions(
    const T* d_data,
    int num_elements,
    int partition_size,
    int32_t* d_start_indices,
    int32_t* d_end_indices
);

/**
 * Adaptive partitioning based on variance (better compression)
 *
 * PARAMETERS:
 * - Similar to fixed-size
 * - variance_threshold: Split partition if variance exceeds threshold
 *
 * RETURNS: num_partitions (variable)
 */
template<typename T>
int createAdaptivePartitions(
    const T* d_data,
    int num_elements,
    int target_partition_size,
    double variance_threshold,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int max_partitions
);

// ============================================================================
// Debug and Inspection
// ============================================================================

/**
 * Print compressed data metadata (for debugging)
 */
template<typename T>
void printCompressedMetadata(const CompressedDataL3<T>* compressed);

/**
 * Dump compressed data to file (binary format)
 */
template<typename T>
bool saveCompressedDataToFile(
    const CompressedDataL3<T>* compressed,
    const char* filename
);

/**
 * Load compressed data from file
 */
template<typename T>
CompressedDataL3<T>* loadCompressedDataFromFile(const char* filename);

// ============================================================================
// Configuration
// ============================================================================

/**
 * Partitioning strategy enumeration
 */
enum class PartitioningStrategy {
    FIXED,              // Fixed-size partitions (fastest compression)
    VARIANCE_ADAPTIVE,  // Variance-based adaptive (legacy, can cause partition explosion)
    COST_OPTIMAL        // Cost-optimal (NEW: delta-bits driven, prevents explosion)
};

/**
 * Compression configuration
 */
struct L3Config {
    int partition_size;              // Base partition size (1024-4096 typical)
    bool use_adaptive_partitioning;  // Use variance-based adaptive partitioning (legacy)
    int max_delta_bits;              // Maximum bits per delta (64 max)
    double variance_threshold;       // For adaptive partitioning (legacy)
    int model_type;                  // Force model type (-1 = auto)

    // Adaptive partitioning parameters (for VARIANCE_ADAPTIVE strategy)
    int variance_block_multiplier;   // Variance analysis block = multiplier × base_size
    int num_variance_thresholds;     // Number of variance thresholds (creates n+1 buckets)

    // Partitioning strategy (new unified interface)
    PartitioningStrategy partitioning_strategy;

    // Cost-optimal partitioning parameters (for COST_OPTIMAL strategy)
    int cost_analysis_block_size;    // Size of blocks for delta-bits analysis
    int cost_min_partition_size;     // Minimum partition size (warp-aligned)
    int cost_max_partition_size;     // Maximum partition size
    int cost_target_partition_size;  // Preferred partition size
    int cost_breakpoint_threshold;   // Delta-bits change to trigger breakpoint
    float cost_merge_benefit_threshold; // Minimum benefit (5%) to merge
    int cost_max_merge_rounds;       // Maximum merge iterations
    bool cost_enable_merging;        // Enable cost-based merging
    bool enable_polynomial_models;   // Enable POLY2/POLY3 model selection

    // Default config - cost-optimal partitioning ENABLED by default
    L3Config()
        : partition_size(2048),
          use_adaptive_partitioning(false),  // Deprecated, use partitioning_strategy
          max_delta_bits(64),
          variance_threshold(1e6),
          model_type(-1),  // Auto-select
          variance_block_multiplier(8),
          num_variance_thresholds(3),
          partitioning_strategy(PartitioningStrategy::COST_OPTIMAL),  // NEW DEFAULT
          cost_analysis_block_size(2048),
          cost_min_partition_size(256),
          cost_max_partition_size(8192),
          cost_target_partition_size(2048),
          cost_breakpoint_threshold(2),
          cost_merge_benefit_threshold(0.05f),
          cost_max_merge_rounds(4),
          cost_enable_merging(true),
          enable_polynomial_models(false)  // Default: off for backward compatibility
    {}

    // Factory method for fixed-size partitioning (legacy behavior)
    static L3Config fixedPartitioning(int partition_size = 2048) {
        L3Config config;
        config.partition_size = partition_size;
        config.use_adaptive_partitioning = false;
        config.partitioning_strategy = PartitioningStrategy::FIXED;
        return config;
    }

    // Factory method for high compression (cost-optimal with smaller partitions)
    static L3Config highCompression() {
        L3Config config;
        config.partition_size = 1024;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_target_partition_size = 1024;
        config.cost_min_partition_size = 256;
        config.cost_max_partition_size = 4096;
        config.cost_breakpoint_threshold = 1;  // More sensitive
        config.cost_enable_merging = true;
        return config;
    }

    // Factory method for high throughput (cost-optimal with larger partitions)
    static L3Config highThroughput() {
        L3Config config;
        config.partition_size = 4096;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_target_partition_size = 4096;
        config.cost_min_partition_size = 512;
        config.cost_max_partition_size = 16384;
        config.cost_breakpoint_threshold = 3;  // Less sensitive
        config.cost_enable_merging = false;    // Skip merging for speed
        return config;
    }

    // Factory method for variance-based adaptive (legacy, may cause partition explosion)
    static L3Config varianceAdaptive(int partition_size = 2048) {
        L3Config config;
        config.partition_size = partition_size;
        config.use_adaptive_partitioning = true;
        config.partitioning_strategy = PartitioningStrategy::VARIANCE_ADAPTIVE;
        config.variance_block_multiplier = 8;
        config.num_variance_thresholds = 3;
        return config;
    }

    // Factory method for cost-optimal partitioning (recommended default)
    static L3Config costOptimal(int target_partition_size = 2048) {
        L3Config config;
        config.partition_size = target_partition_size;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_target_partition_size = target_partition_size;
        return config;
    }
};

/**
 * Compress with custom configuration
 */
template<typename T>
CompressedDataL3<T>* compressDataWithConfig(
    const std::vector<T>& h_data,
    const L3Config& config,
    CompressionStats* stats = nullptr
);

// ============================================================================
// Encoder/Decoder Launch Functions (Low-Level)
// ============================================================================

/**
 * Launch model fitting kernel
 * (See encoder.cu for implementation)
 * Now also stores actual partition min/max values for tight predicate pushdown bounds
 */
template<typename T>
void launchModelFittingKernel(
    const T* d_values,
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_model_types,
    double* d_model_params,
    int32_t* d_delta_bits,
    int64_t* d_error_bounds,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions,
    int64_t* d_total_bits,
    cudaStream_t stream = 0
);

/**
 * Launch bit offset calculation kernel
 */
void launchSetBitOffsetsKernel(
    int32_t* d_start_indices,
    int32_t* d_end_indices,
    int32_t* d_delta_bits,
    int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    cudaStream_t stream = 0
);

/**
 * Launch delta packing kernel
 */
template<typename T>
void launchDeltaPackingKernel(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    const int32_t* d_model_types,
    const double* d_model_params,
    const int32_t* d_delta_bits,
    const int64_t* d_delta_array_bit_offsets,
    int num_partitions,
    uint32_t* delta_array,
    int total_elements,
    cudaStream_t stream = 0
);

/**
 * Launch decompression kernel (from decompression_kernels.cu)
 */
template<typename T>
void launchDecompressionKernel(
    const CompressedDataL3<T>* d_compressed,
    T* d_output,
    int num_elements,
    cudaStream_t stream = 0
);

/**
 * Compute partition min/max bounds (for predicate pushdown optimization)
 */
template<typename T>
void launchComputePartitionBounds(
    const T* d_values,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    T* d_partition_min,
    T* d_partition_max,
    int num_partitions,
    cudaStream_t stream = 0
);

#endif // L3_CODEC_HPP
