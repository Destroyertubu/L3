/**
 * Cost-Optimal Partitioning Algorithm for L3
 *
 * Key innovations:
 * 1. Delta-bits driven breakpoint detection (not variance-based)
 * 2. Cost-based partition evaluation and merging
 * 3. Parallel odd-even merging strategy
 * 4. Warp-aligned partition sizes for GPU efficiency
 */

#ifndef ENCODER_COST_OPTIMAL_CUH
#define ENCODER_COST_OPTIMAL_CUH

#include <vector>
#include <cuda_runtime.h>
#include "L3_format.hpp"

/**
 * Configuration for cost-optimal partitioning
 */
struct CostOptimalConfig {
    int analysis_block_size = 2048;      // Size of blocks for delta-bits analysis
    int min_partition_size = 256;         // Initial partition size (warp-aligned), merge from here
    int max_partition_size = 8192;        // Maximum partition size after merging
    int breakpoint_threshold = 2;         // Delta-bits change to trigger breakpoint
    float merge_benefit_threshold = 0.05f; // Minimum benefit (5%) to merge
    int max_merge_rounds = 4;             // Maximum merge iterations
    bool enable_merging = true;           // Enable cost-based merging
    bool enable_rle = true;               // Enable RLE/CONSTANT model selection

    // Polynomial model selection configuration (Stage 2)
    bool enable_polynomial_models = false; // Enable POLY2/POLY3 model selection (default: off for compatibility)
    int polynomial_min_size = 10;          // Minimum partition size for POLY2 consideration
    int cubic_min_size = 20;               // Minimum partition size for POLY3 consideration
    float polynomial_cost_threshold = 0.95f; // Require 5% improvement to select higher-order model

    // Factory methods
    static CostOptimalConfig balanced() {
        CostOptimalConfig config;
        config.min_partition_size = 256;
        config.max_partition_size = 8192;
        return config;
    }

    static CostOptimalConfig highCompression() {
        CostOptimalConfig config;
        config.min_partition_size = 128;
        config.max_partition_size = 4096;
        config.breakpoint_threshold = 1;  // More sensitive
        return config;
    }

    static CostOptimalConfig highThroughput() {
        CostOptimalConfig config;
        config.min_partition_size = 512;
        config.max_partition_size = 16384;
        config.breakpoint_threshold = 3;  // Less sensitive
        config.enable_merging = false;    // Skip merging for speed
        return config;
    }

    static CostOptimalConfig polynomialEnabled() {
        CostOptimalConfig config;
        config.enable_polynomial_models = true;
        config.polynomial_min_size = 10;
        config.cubic_min_size = 20;
        config.polynomial_cost_threshold = 0.95f;
        return config;
    }
};

/**
 * Create partitions using cost-optimal algorithm.
 *
 * @param data Input data vector
 * @param config Configuration parameters
 * @param num_partitions_out Output: number of partitions created
 * @param stream CUDA stream (optional)
 * @return Vector of PartitionInfo structures
 */
template<typename T>
std::vector<PartitionInfo> createPartitionsCostOptimal(
    const std::vector<T>& data,
    const CostOptimalConfig& config,
    int* num_partitions_out,
    cudaStream_t stream = 0);

#endif // ENCODER_COST_OPTIMAL_CUH
