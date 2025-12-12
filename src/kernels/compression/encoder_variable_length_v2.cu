/**
 * Variable-Length Encoder V2 - Cost-Aware Adaptive Partitioning
 *
 * Key improvement over V1:
 * - Cost-aware partitioning: only use small partitions when compression benefit > metadata cost
 * - Post-fit partition merging: merge adjacent partitions with no compression benefit
 * - Guaranteed: compression ratio >= fixed partitioning
 *
 * Strategy:
 * 1. Create initial partitions based on variance (like V1)
 * 2. Fit models and compute delta_bits for each partition
 * 3. Calculate net benefit for each partition: compression_savings - metadata_overhead
 * 4. Merge adjacent partitions with negative net benefit
 * 5. Refit merged partitions
 */

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include "L3_format.hpp"
#include "L3_codec.hpp"

// Constants
#define MIN_PARTITION_SIZE 128
#define MAX_PARTITION_SIZE 8192
#define PARTITION_METADATA_BYTES 64.0  // Approximate metadata per partition

// ============================================================================
// Helper Functions (same as V1)
// ============================================================================

template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;
    } else {
        const double MAX_SAFE_DOUBLE = 9007199254740992.0;
        return static_cast<double>(value) > MAX_SAFE_DOUBLE;
    }
}

template<typename T>
__device__ __host__ inline long long calculateDeltaV2(T actual, T predicted) {
    return static_cast<long long>(actual) - static_cast<long long>(predicted);
}

// ============================================================================
// Cost-Aware Partition Info
// ============================================================================

struct PartitionCostInfo {
    int start_idx;
    int end_idx;
    int model_type;
    double theta0;
    double theta1;
    int delta_bits;
    long long max_error;
    double net_benefit;  // compression_savings - metadata_cost
    bool should_merge;   // Flag for merge candidates
};

// ============================================================================
// Host-side Model Fitting (for flexibility and accuracy)
// ============================================================================

template<typename T>
void fitPartitionModel(const std::vector<T>& data, PartitionCostInfo& info) {
    int start = info.start_idx;
    int end = info.end_idx;
    int n = end - start;

    if (n <= 0) {
        info.model_type = MODEL_DIRECT_COPY;
        info.theta0 = 0;
        info.theta1 = 0;
        info.delta_bits = sizeof(T) * 8;
        info.max_error = 0;
        return;
    }

    // Check for overflow values
    bool has_overflow = false;
    for (int i = 0; i < n && !has_overflow; i++) {
        if (mightOverflowDoublePrecision(data[start + i])) {
            has_overflow = true;
        }
    }

    if (has_overflow) {
        info.model_type = MODEL_DIRECT_COPY;
        info.theta0 = 0;
        info.theta1 = 0;
        info.delta_bits = sizeof(T) * 8;
        info.max_error = 0;
        return;
    }

    // Linear regression using Kahan summation for numerical stability
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    double c_x = 0.0, c_y = 0.0, c_xx = 0.0, c_xy = 0.0;

    for (int i = 0; i < n; i++) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);

        // Kahan summation
        double y_x = x - c_x;
        double t_x = sum_x + y_x;
        c_x = (t_x - sum_x) - y_x;
        sum_x = t_x;

        double y_y = y - c_y;
        double t_y = sum_y + y_y;
        c_y = (t_y - sum_y) - y_y;
        sum_y = t_y;

        double y_xx = x * x - c_xx;
        double t_xx = sum_xx + y_xx;
        c_xx = (t_xx - sum_xx) - y_xx;
        sum_xx = t_xx;

        double y_xy = x * y - c_xy;
        double t_xy = sum_xy + y_xy;
        c_xy = (t_xy - sum_xy) - y_xy;
        sum_xy = t_xy;
    }

    double dn = static_cast<double>(n);
    double determinant = dn * sum_xx - sum_x * sum_x;

    if (std::fabs(determinant) > 1e-10) {
        info.theta1 = (dn * sum_xy - sum_x * sum_y) / determinant;
        info.theta0 = (sum_y - info.theta1 * sum_x) / dn;
    } else {
        info.theta1 = 0.0;
        info.theta0 = sum_y / dn;
    }

    info.model_type = MODEL_LINEAR;

    // Calculate max error
    // Use std::llrint for consistency with GPU's __double2ll_rn (banker's rounding)
    long long max_error = 0;
    for (int i = 0; i < n; i++) {
        double predicted = info.theta0 + info.theta1 * i;
        T pred_T = static_cast<T>(std::llrint(predicted));
        long long delta = calculateDeltaV2(data[start + i], pred_T);
        long long abs_error = (delta < 0) ? -delta : delta;
        if (abs_error > max_error) max_error = abs_error;
    }

    info.max_error = max_error;

    // Calculate delta bits
    int delta_bits = 0;
    if (max_error > 0) {
        delta_bits = 64 - __builtin_clzll(static_cast<unsigned long long>(max_error)) + 1;
    }

    // For 64-bit types: if delta_bits > 32, we need special handling
    // The decoder's 64-bit path (delta_bits==64) expects raw values, not deltas
    // So if delta_bits > 32, we must use MODEL_DIRECT_COPY with full bit width
    const int original_bits = sizeof(T) * 8;
    if (delta_bits > 32) {
        info.model_type = MODEL_DIRECT_COPY;
        info.theta0 = 0.0;
        info.theta1 = 0.0;
        info.delta_bits = original_bits;
        info.max_error = 0;
    } else {
        info.delta_bits = delta_bits;
    }
}

// ============================================================================
// Calculate partition net benefit
// ============================================================================

template<typename T>
double calculateNetBenefit(const PartitionCostInfo& info) {
    int n = info.end_idx - info.start_idx;
    int original_bits = sizeof(T) * 8;

    // Bytes saved by compression
    double bits_saved = (double)(original_bits - info.delta_bits) * n;
    double bytes_saved = bits_saved / 8.0;

    // Net benefit = compression savings - metadata overhead
    return bytes_saved - PARTITION_METADATA_BYTES;
}

// ============================================================================
// Cost-Aware Partitioner Class
// ============================================================================

template<typename T>
class CostAwarePartitioner {
private:
    const std::vector<T>& data;
    int base_partition_size;

public:
    CostAwarePartitioner(const std::vector<T>& input_data, int base_size = 2048)
        : data(input_data), base_partition_size(base_size) {}

    std::vector<PartitionInfo> partition() {
        if (data.empty()) return {};

        int data_size = data.size();

        // Step 1: Create initial fixed-size partitions
        std::vector<PartitionCostInfo> partitions;
        int num_initial = (data_size + base_partition_size - 1) / base_partition_size;
        partitions.reserve(num_initial);

        for (int i = 0; i < num_initial; i++) {
            PartitionCostInfo info;
            info.start_idx = i * base_partition_size;
            info.end_idx = std::min((i + 1) * base_partition_size, data_size);
            info.should_merge = false;
            partitions.push_back(info);
        }

        // Step 2: Fit models for all partitions
        for (auto& p : partitions) {
            fitPartitionModel(data, p);
            p.net_benefit = calculateNetBenefit<T>(p);
        }

        // Step 3: Try splitting partitions that might benefit from smaller sizes
        std::vector<PartitionCostInfo> refined_partitions;
        refined_partitions.reserve(partitions.size() * 2);

        for (const auto& p : partitions) {
            int n = p.end_idx - p.start_idx;

            // Only try splitting if partition is large enough and has some compression potential
            if (n >= MIN_PARTITION_SIZE * 2 && p.delta_bits < sizeof(T) * 8 - 4) {
                // Try splitting in half
                int mid = p.start_idx + n / 2;

                PartitionCostInfo left, right;
                left.start_idx = p.start_idx;
                left.end_idx = mid;
                left.should_merge = false;

                right.start_idx = mid;
                right.end_idx = p.end_idx;
                right.should_merge = false;

                fitPartitionModel(data, left);
                fitPartitionModel(data, right);
                left.net_benefit = calculateNetBenefit<T>(left);
                right.net_benefit = calculateNetBenefit<T>(right);

                // Calculate total benefit of split vs original
                double split_benefit = left.net_benefit + right.net_benefit;

                // Only split if it improves total benefit
                if (split_benefit > p.net_benefit + PARTITION_METADATA_BYTES * 0.5) {
                    refined_partitions.push_back(left);
                    refined_partitions.push_back(right);
                } else {
                    refined_partitions.push_back(p);
                }
            } else {
                refined_partitions.push_back(p);
            }
        }

        // Step 4: Merge adjacent partitions with negative net benefit
        std::vector<PartitionCostInfo> merged_partitions;
        merged_partitions.reserve(refined_partitions.size());

        size_t i = 0;
        while (i < refined_partitions.size()) {
            PartitionCostInfo current = refined_partitions[i];

            // Try to merge with subsequent partitions if both have negative benefit
            while (i + 1 < refined_partitions.size()) {
                const auto& next = refined_partitions[i + 1];

                // Both partitions have negative or near-zero benefit - candidate for merge
                if (current.net_benefit < PARTITION_METADATA_BYTES * 0.1 &&
                    next.net_benefit < PARTITION_METADATA_BYTES * 0.1) {

                    // Check if merged partition would be too large
                    int merged_size = next.end_idx - current.start_idx;
                    if (merged_size > MAX_PARTITION_SIZE) break;

                    // Create merged partition
                    PartitionCostInfo merged;
                    merged.start_idx = current.start_idx;
                    merged.end_idx = next.end_idx;
                    merged.should_merge = false;

                    fitPartitionModel(data, merged);
                    merged.net_benefit = calculateNetBenefit<T>(merged);

                    // Check if merging is beneficial
                    // Merging saves one partition's metadata overhead
                    double merge_savings = PARTITION_METADATA_BYTES;
                    double merge_cost = merged.net_benefit - (current.net_benefit + next.net_benefit);

                    if (merge_cost + merge_savings >= 0) {
                        // Merging is beneficial or neutral
                        current = merged;
                        i++;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            merged_partitions.push_back(current);
            i++;
        }

        // Step 5: Convert to PartitionInfo format
        std::vector<PartitionInfo> result;
        result.reserve(merged_partitions.size());

        for (const auto& p : merged_partitions) {
            PartitionInfo info;
            info.start_idx = p.start_idx;
            info.end_idx = p.end_idx;
            info.model_type = p.model_type;
            info.model_params[0] = p.theta0;
            info.model_params[1] = p.theta1;
            info.model_params[2] = 0.0;
            info.model_params[3] = 0.0;
            info.delta_bits = p.delta_bits;
            info.delta_array_bit_offset = 0;
            info.error_bound = p.max_error;
            result.push_back(info);
        }

        // Ensure complete coverage
        if (!result.empty()) {
            result[0].start_idx = 0;
            result.back().end_idx = data_size;

            for (size_t j = 0; j < result.size() - 1; j++) {
                if (result[j].end_idx != result[j + 1].start_idx) {
                    result[j].end_idx = result[j + 1].start_idx;
                }
            }
        }

        return result;
    }
};

// ============================================================================
// GPU-Accelerated Cost-Aware Partitioner
// ============================================================================

// Kernel to compute variance per block
template<typename T>
__global__ void computeBlockVariance(
    const T* __restrict__ data,
    int data_size,
    int block_size,
    double* __restrict__ block_means,
    double* __restrict__ block_variances,
    int num_blocks)
{
    int bid = blockIdx.x;
    if (bid >= num_blocks) return;

    int start = bid * block_size;
    int end = min(start + block_size, data_size);
    int n = end - start;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            block_means[bid] = 0;
            block_variances[bid] = 0;
        }
        return;
    }

    // Compute mean using block reduction
    __shared__ double s_sum[256];
    __shared__ double s_sum_sq[256];

    double local_sum = 0;
    double local_sum_sq = 0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        double val = static_cast<double>(data[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        double mean = s_sum[0] / n;
        double variance = s_sum_sq[0] / n - mean * mean;
        block_means[bid] = mean;
        block_variances[bid] = max(0.0, variance);
    }
}

// Warp reduction helpers
__device__ __forceinline__ double warpReduceSumDouble(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ long long warpReduceMaxLL(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ double blockReduceSumDouble(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSumDouble(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0;
    if (wid == 0) val = warpReduceSumDouble(val);

    return val;
}

__device__ __forceinline__ long long blockReduceMaxLL(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMaxLL(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMaxLL(val);

    return val;
}

// Kernel to fit linear model for each partition
template<typename T>
__global__ void fitPartitionModels(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    double* __restrict__ net_benefits,
    int num_partitions,
    int original_bits)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = partition_starts[pid];
    int end = partition_ends[pid];
    int n = end - start;

    __shared__ double s_theta0, s_theta1;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0;
            theta1_array[pid] = 0;
            delta_bits_array[pid] = original_bits;
            max_errors[pid] = 0;
            net_benefits[pid] = -PARTITION_METADATA_BYTES;
        }
        return;
    }

    // Compute sums for linear regression
    double local_sum_x = 0, local_sum_y = 0, local_sum_xx = 0, local_sum_xy = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        local_sum_x += x;
        local_sum_y += y;
        local_sum_xx += x * x;
        local_sum_xy += x * y;
    }

    // Block reduction for sums
    double sum_x = blockReduceSumDouble(local_sum_x);
    double sum_y = blockReduceSumDouble(local_sum_y);
    double sum_xx = blockReduceSumDouble(local_sum_xx);
    double sum_xy = blockReduceSumDouble(local_sum_xy);

    // Compute model parameters (single thread)
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double det = dn * sum_xx - sum_x * sum_x;

        if (fabs(det) > 1e-10) {
            s_theta1 = (dn * sum_xy - sum_x * sum_y) / det;
            s_theta0 = (sum_y - s_theta1 * sum_x) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }

        theta0_array[pid] = s_theta0;
        theta1_array[pid] = s_theta1;
        model_types[pid] = MODEL_LINEAR;
    }
    __syncthreads();

    // Compute max error
    double theta0 = s_theta0;
    double theta1 = s_theta1;
    long long local_max = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double predicted = theta0 + theta1 * i;
        // Use __double2ll_rn for consistency with decoder (banker's rounding)
        T pred_T = static_cast<T>(__double2ll_rn(predicted));
        long long delta = static_cast<long long>(data[start + i]) - static_cast<long long>(pred_T);
        local_max = max(local_max, llabs(delta));
    }

    long long max_err = blockReduceMaxLL(local_max);

    if (threadIdx.x == 0) {
        max_errors[pid] = max_err;

        int dbits = 0;
        if (max_err > 0) {
            dbits = 64 - __clzll(static_cast<unsigned long long>(max_err)) + 1;
        }

        // For 64-bit types: if delta_bits > 32, we need special handling
        // The decoder's 64-bit path (delta_bits==64) expects raw values, not deltas
        // So if delta_bits is between 33-63, we must round to 64 AND use MODEL_DIRECT_COPY
        if (dbits > 32) {
            // Fall back to direct copy - decoder handles this correctly
            dbits = original_bits;  // Use full bit width
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            max_errors[pid] = 0;
        }

        delta_bits_array[pid] = dbits;

        // Calculate net benefit
        double bits_saved = (double)(original_bits - dbits) * n;
        double bytes_saved = bits_saved / 8.0;
        net_benefits[pid] = bytes_saved - PARTITION_METADATA_BYTES;
    }
}

// ============================================================================
// Main Cost-Aware Variable-Length Partitioner (GPU version)
// ============================================================================

template<typename T>
class GPUCostAwarePartitioner {
private:
    T* d_data;
    const std::vector<T>& h_data;
    int data_size;
    int base_partition_size;
    cudaStream_t stream;

public:
    GPUCostAwarePartitioner(const std::vector<T>& data, int base_size = 2048, cudaStream_t s = 0)
        : h_data(data), data_size(data.size()), base_partition_size(base_size), stream(s)
    {
        cudaMalloc(&d_data, data_size * sizeof(T));
        cudaMemcpyAsync(d_data, data.data(), data_size * sizeof(T), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    ~GPUCostAwarePartitioner() {
        if (d_data) cudaFree(d_data);
    }

    std::vector<PartitionInfo> partition() {
        if (data_size == 0) return {};

        const int original_bits = sizeof(T) * 8;

        // Step 1: Create initial partitions using base size
        int num_partitions = (data_size + base_partition_size - 1) / base_partition_size;

        // Allocate GPU arrays
        int* d_starts;
        int* d_ends;
        int* d_model_types;
        double* d_theta0;
        double* d_theta1;
        int* d_delta_bits;
        long long* d_max_errors;
        double* d_net_benefits;

        cudaMalloc(&d_starts, num_partitions * sizeof(int));
        cudaMalloc(&d_ends, num_partitions * sizeof(int));
        cudaMalloc(&d_model_types, num_partitions * sizeof(int));
        cudaMalloc(&d_theta0, num_partitions * sizeof(double));
        cudaMalloc(&d_theta1, num_partitions * sizeof(double));
        cudaMalloc(&d_delta_bits, num_partitions * sizeof(int));
        cudaMalloc(&d_max_errors, num_partitions * sizeof(long long));
        cudaMalloc(&d_net_benefits, num_partitions * sizeof(double));

        // Initialize partition boundaries
        std::vector<int> h_starts(num_partitions), h_ends(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            h_starts[i] = i * base_partition_size;
            h_ends[i] = std::min((i + 1) * base_partition_size, data_size);
        }

        cudaMemcpyAsync(d_starts, h_starts.data(), num_partitions * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_ends, h_ends.data(), num_partitions * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        // Step 2: Fit models on GPU
        int threads = 256;
        fitPartitionModels<T><<<num_partitions, threads, 0, stream>>>(
            d_data, d_starts, d_ends, d_model_types,
            d_theta0, d_theta1, d_delta_bits, d_max_errors, d_net_benefits,
            num_partitions, original_bits);
        cudaStreamSynchronize(stream);

        // Copy results back to host
        std::vector<int> h_model_types(num_partitions);
        std::vector<double> h_theta0(num_partitions), h_theta1(num_partitions);
        std::vector<int> h_delta_bits(num_partitions);
        std::vector<long long> h_max_errors(num_partitions);
        std::vector<double> h_net_benefits(num_partitions);

        cudaMemcpy(h_model_types.data(), d_model_types, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_theta0.data(), d_theta0, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_theta1.data(), d_theta1, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta_bits.data(), d_delta_bits, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_max_errors.data(), d_max_errors, num_partitions * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_net_benefits.data(), d_net_benefits, num_partitions * sizeof(double), cudaMemcpyDeviceToHost);

        // Free GPU arrays
        cudaFree(d_starts);
        cudaFree(d_ends);
        cudaFree(d_model_types);
        cudaFree(d_theta0);
        cudaFree(d_theta1);
        cudaFree(d_delta_bits);
        cudaFree(d_max_errors);
        cudaFree(d_net_benefits);

        // Step 3: Merge partitions with negative net benefit (CPU side for flexibility)
        std::vector<PartitionInfo> result;
        result.reserve(num_partitions);

        int i = 0;
        while (i < num_partitions) {
            PartitionInfo current;
            current.start_idx = h_starts[i];
            current.end_idx = h_ends[i];
            current.model_type = h_model_types[i];
            current.model_params[0] = h_theta0[i];
            current.model_params[1] = h_theta1[i];
            current.model_params[2] = 0;
            current.model_params[3] = 0;
            current.delta_bits = h_delta_bits[i];
            current.error_bound = h_max_errors[i];

            double current_benefit = h_net_benefits[i];

            // Try to merge with subsequent partitions if benefit is low
            while (i + 1 < num_partitions &&
                   current_benefit < PARTITION_METADATA_BYTES * 0.2 &&
                   h_net_benefits[i + 1] < PARTITION_METADATA_BYTES * 0.2) {

                int merged_size = h_ends[i + 1] - current.start_idx;
                if (merged_size > MAX_PARTITION_SIZE) break;

                // Compute merged partition parameters
                PartitionCostInfo merged_info;
                merged_info.start_idx = current.start_idx;
                merged_info.end_idx = h_ends[i + 1];
                fitPartitionModel(h_data, merged_info);

                double merged_benefit = calculateNetBenefit<T>(merged_info);

                // Merging is beneficial if we save metadata without losing much compression
                double total_before = current_benefit + h_net_benefits[i + 1];
                double merge_gain = PARTITION_METADATA_BYTES + (merged_benefit - total_before);

                if (merge_gain >= 0) {
                    // Accept merge
                    current.end_idx = merged_info.end_idx;
                    current.model_params[0] = merged_info.theta0;
                    current.model_params[1] = merged_info.theta1;
                    current.delta_bits = merged_info.delta_bits;
                    current.error_bound = merged_info.max_error;
                    current_benefit = merged_benefit;
                    i++;
                } else {
                    break;
                }
            }

            current.delta_array_bit_offset = 0;
            result.push_back(current);
            i++;
        }

        // Ensure complete coverage
        if (!result.empty()) {
            result[0].start_idx = 0;
            result.back().end_idx = data_size;
        }

        return result;
    }
};

// ============================================================================
// Public API
// ============================================================================

template<typename T>
std::vector<PartitionInfo> createPartitionsCostAware(
    const std::vector<T>& data,
    int base_partition_size,
    int* num_partitions_out,
    cudaStream_t stream)
{
    GPUCostAwarePartitioner<T> partitioner(data, base_partition_size, stream);
    std::vector<PartitionInfo> result = partitioner.partition();

    if (num_partitions_out) {
        *num_partitions_out = static_cast<int>(result.size());
    }

    return result;
}

// Explicit instantiations
template std::vector<PartitionInfo> createPartitionsCostAware<uint64_t>(
    const std::vector<uint64_t>&, int, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostAware<int64_t>(
    const std::vector<int64_t>&, int, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostAware<uint32_t>(
    const std::vector<uint32_t>&, int, int*, cudaStream_t);
template std::vector<PartitionInfo> createPartitionsCostAware<int32_t>(
    const std::vector<int32_t>&, int, int*, cudaStream_t);
