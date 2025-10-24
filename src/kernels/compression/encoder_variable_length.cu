/**
 * Variable-Length Encoder for GLECO
 *
 * Implements adaptive partitioning based on data variance.
 * Extracted and adapted from L32.cu
 */

#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include "L3_format.hpp"
#include "L3_codec.hpp"

// Constants
#define MIN_PARTITION_SIZE 128
#define PARTITION_MODEL_SIZE_BYTES 64.0

// ============================================================================
// Helper Functions
// ============================================================================

template<typename T>
__device__ __host__ inline bool mightOverflowDoublePrecision(T value) {
    if (std::is_signed<T>::value) {
        return false;
    } else {
        const double MAX_SAFE_DOUBLE = 9007199254740992.0; // 2^53
        return static_cast<double>(value) > MAX_SAFE_DOUBLE;
    }
}

template<typename T>
__device__ __host__ inline long long calculateDelta(T actual, T predicted) {
    if (std::is_signed<T>::value) {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    } else {
        return static_cast<long long>(actual) - static_cast<long long>(predicted);
    }
}

__device__ double warpReduceSum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ long long warpReduceMax(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ double blockReduceSum(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);

    return val;
}

__device__ long long blockReduceMax(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceMax(val);

    return val;
}

// ============================================================================
// Variance Analysis Kernel
// ============================================================================

template<typename T>
__global__ void analyzeDataVarianceFast(
    const T* __restrict__ data,
    int data_size,
    int block_size,
    float* __restrict__ variances,
    int num_blocks) {

    for (int bid = blockIdx.x; bid < num_blocks; bid += gridDim.x) {
        int start = bid * block_size;
        int end = min(start + block_size, data_size);
        int n = end - start;

        if (n <= 0) continue;

        double sum = 0.0;
        double sum_sq = 0.0;
        double c1 = 0.0, c2 = 0.0;

        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            double val = static_cast<double>(data[i]);

            // Kahan summation
            double y1 = val - c1;
            double t1 = sum + y1;
            c1 = (t1 - sum) - y1;
            sum = t1;

            double y2 = val * val - c2;
            double t2 = sum_sq + y2;
            c2 = (t2 - sum_sq) - y2;
            sum_sq = t2;
        }

        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);

        if ((threadIdx.x & 31) == 0) {
            atomicAdd(&variances[bid], static_cast<float>(sum_sq / n - (sum / n) * (sum / n)));
        }
    }
}

// ============================================================================
// Partition Creation Kernels
// ============================================================================

template<typename T>
__global__ void countPartitionsPerBlock(
    int data_size,
    int base_size,
    const float* __restrict__ variances,
    int num_variance_blocks,
    int* __restrict__ partition_counts_per_block,
    const float* __restrict__ variance_thresholds,
    const int* __restrict__ partition_sizes_for_buckets,
    int num_thresholds,
    int variance_block_multiplier)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_variance_blocks) return;

    float var = variances[i];
    int block_start = i * base_size * variance_block_multiplier;
    int block_end = min(block_start + base_size * variance_block_multiplier, data_size);

    int partition_size = partition_sizes_for_buckets[num_thresholds];
    for (int j = 0; j < num_thresholds; ++j) {
        if (var < variance_thresholds[j]) {
            partition_size = partition_sizes_for_buckets[j];
            break;
        }
    }

    int count = 0;
    if (partition_size > 0) {
        for (int j = block_start; j < block_end; j += partition_size) {
            if (j < data_size) {
                count++;
            }
        }
    }

    partition_counts_per_block[i] = count;
}

template<typename T>
__global__ void writePartitionsOrdered(
    int data_size,
    int base_size,
    const float* __restrict__ variances,
    int num_variance_blocks,
    const int* __restrict__ partition_offsets,
    int* __restrict__ partition_starts,
    int* __restrict__ partition_ends,
    const float* __restrict__ variance_thresholds,
    const int* __restrict__ partition_sizes_for_buckets,
    int num_thresholds,
    int variance_block_multiplier)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_variance_blocks) return;

    float var = variances[i];
    int block_start = i * base_size * variance_block_multiplier;
    int block_end = min(block_start + base_size * variance_block_multiplier, data_size);

    int partition_size = partition_sizes_for_buckets[num_thresholds];
    for (int j = 0; j < num_thresholds; ++j) {
        if (var < variance_thresholds[j]) {
            partition_size = partition_sizes_for_buckets[j];
            break;
        }
    }

    if (partition_size <= 0) return;

    int write_pos = partition_offsets[i];

    int local_idx = 0;
    for (int j = block_start; j < block_end; j += partition_size) {
        if (j < data_size) {
            partition_starts[write_pos + local_idx] = j;
            partition_ends[write_pos + local_idx] = min(j + partition_size, data_size);
            local_idx++;
        }
    }
}

// ============================================================================
// Model Fitting Kernel
// ============================================================================

template<typename T>
__global__ void fitPartitionsBatched_Optimized(
    const T* __restrict__ data,
    const int* __restrict__ partition_starts,
    const int* __restrict__ partition_ends,
    int* __restrict__ model_types,
    double* __restrict__ theta0_array,
    double* __restrict__ theta1_array,
    int* __restrict__ delta_bits_array,
    long long* __restrict__ max_errors,
    double* __restrict__ costs,
    int num_partitions)
{
    const int pid = blockIdx.x;
    if (pid >= num_partitions) {
        return;
    }

    __shared__ double s_theta0;
    __shared__ double s_theta1;
    __shared__ int s_has_overflow_flag;

    const int start = partition_starts[pid];
    const int end = partition_ends[pid];
    const int n = end - start;

    if (threadIdx.x == 0) {
        s_has_overflow_flag = false;
    }
    __syncthreads();

    if (n <= 0) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            costs[pid] = 0.0;
        }
        return;
    }

    bool local_overflow = false;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (mightOverflowDoublePrecision(data[start + i])) {
            local_overflow = true;
            break;
        }
    }

    if (local_overflow) {
        atomicExch(&s_has_overflow_flag, true);
    }
    __syncthreads();

    if (s_has_overflow_flag) {
        if (threadIdx.x == 0) {
            model_types[pid] = MODEL_DIRECT_COPY;
            theta0_array[pid] = 0.0;
            theta1_array[pid] = 0.0;
            delta_bits_array[pid] = sizeof(T) * 8;
            max_errors[pid] = 0;
            costs[pid] = PARTITION_MODEL_SIZE_BYTES + n * sizeof(T);
        }
        return;
    }

    // Fast linear regression
    double sum_x = 0.0, sum_y = 0.0, sum_xx = 0.0, sum_xy = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double x = static_cast<double>(i);
        double y = static_cast<double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        sum_xx = fma(x, x, sum_xx);
        sum_xy = fma(x, y, sum_xy);
    }

    sum_x = blockReduceSum(sum_x);
    sum_y = blockReduceSum(sum_y);
    sum_xx = blockReduceSum(sum_xx);
    sum_xy = blockReduceSum(sum_xy);

    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);
        double determinant = fma(dn, sum_xx, -(sum_x * sum_x));

        if (fabs(determinant) > 1e-10) {
            s_theta1 = fma(dn, sum_xy, -(sum_x * sum_y)) / determinant;
            s_theta0 = fma(-s_theta1, sum_x, sum_y) / dn;
        } else {
            s_theta1 = 0.0;
            s_theta0 = sum_y / dn;
        }
        model_types[pid] = MODEL_LINEAR;
        theta0_array[pid] = s_theta0;
        theta1_array[pid] = s_theta1;
    }
    __syncthreads();

    double theta0 = theta0_array[pid];
    double theta1 = theta1_array[pid];
    long long local_max_error = 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        double predicted = fma(theta1, static_cast<double>(i), theta0);
        T pred_T = static_cast<T>(round(predicted));
        long long delta = calculateDelta(data[start + i], pred_T);
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long partition_max_error = blockReduceMax(local_max_error);

    if (threadIdx.x == 0) {
        max_errors[pid] = partition_max_error;

        int delta_bits = 0;
        if (partition_max_error > 0) {
            delta_bits = 64 - __clzll(static_cast<unsigned long long>(partition_max_error)) + 1;
        }
        delta_bits_array[pid] = delta_bits;

        double delta_array_bytes = static_cast<double>(n) * delta_bits / 8.0;
        costs[pid] = PARTITION_MODEL_SIZE_BYTES + delta_array_bytes;
    }
}

// ============================================================================
// Variable-Length Partitioner Class
// ============================================================================

template<typename T>
class GPUVariableLengthPartitionerV6 {
private:
    T* d_data;
    int data_size;
    int base_partition_size;
    cudaStream_t stream;
    int variance_block_multiplier;
    int num_thresholds;

public:
    GPUVariableLengthPartitionerV6(const std::vector<T>& data,
                                   int base_size = 2048,
                                   cudaStream_t cuda_stream = 0,
                                   int multiplier = 8,
                                   int thresholds = 3)
        : data_size(data.size()),
          base_partition_size(base_size),
          stream(cuda_stream),
          variance_block_multiplier(multiplier),
          num_thresholds(thresholds)
    {
        if (this->num_thresholds < 1) {
            this->num_thresholds = 1;
        }
        cudaMalloc(&d_data, data_size * sizeof(T));
        cudaMemcpyAsync(d_data, data.data(), data_size * sizeof(T),
                       cudaMemcpyHostToDevice, stream);
    }

    ~GPUVariableLengthPartitionerV6() {
        if (d_data) cudaFree(d_data);
    }

    std::vector<PartitionInfo> partition() {
        if (data_size == 0) return std::vector<PartitionInfo>();

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        int sm_count = prop.multiProcessorCount;

        int variance_block_size = base_partition_size * variance_block_multiplier;
        int num_variance_blocks = (data_size + variance_block_size - 1) / variance_block_size;
        float* d_variances;
        float* d_variance_thresholds;

        cudaMalloc(&d_variances, num_variance_blocks * sizeof(float));
        cudaMalloc(&d_variance_thresholds, num_thresholds * sizeof(float));
        cudaMemsetAsync(d_variances, 0, num_variance_blocks * sizeof(float), stream);

        int threads = 128;
        int blocks = min(num_variance_blocks, sm_count * 4);

        analyzeDataVarianceFast<T><<<blocks, threads, 0, stream>>>(
            d_data, data_size, variance_block_size, d_variances, num_variance_blocks);

        thrust::device_ptr<float> var_ptr(d_variances);
        if (num_variance_blocks > 1) {
           thrust::sort(var_ptr, var_ptr + num_variance_blocks);
        }

        std::vector<float> h_thresholds(num_thresholds);
        for (int i = 0; i < num_thresholds; ++i) {
            long long idx = (long long)(i + 1) * num_variance_blocks / (num_thresholds + 1);
            if (idx >= num_variance_blocks) idx = num_variance_blocks - 1;
            if (idx < 0) idx = 0;
            h_thresholds[i] = (num_variance_blocks > 0) ? var_ptr[idx] : 0.0f;
        }

        cudaMemcpyAsync(d_variance_thresholds, h_thresholds.data(),
                       num_thresholds * sizeof(float), cudaMemcpyHostToDevice, stream);

        std::vector<int> h_partition_sizes_for_buckets(num_thresholds + 1);
        int min_partition_size_val = base_partition_size;
        for (int i = 0; i <= num_thresholds; ++i) {
            int shift = (num_thresholds / 2) - i;
            h_partition_sizes_for_buckets[i] = std::max(MIN_PARTITION_SIZE, base_partition_size << shift);
            if (h_partition_sizes_for_buckets[i] < min_partition_size_val) {
                min_partition_size_val = h_partition_sizes_for_buckets[i];
            }
        }

        int* d_partition_sizes_for_buckets;
        cudaMalloc(&d_partition_sizes_for_buckets, (num_thresholds + 1) * sizeof(int));
        cudaMemcpyAsync(d_partition_sizes_for_buckets, h_partition_sizes_for_buckets.data(),
                       (num_thresholds + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

        int estimated_partitions = (min_partition_size_val > 0) ?
                                  (data_size / min_partition_size_val + 1) * 2 :
                                  data_size / MIN_PARTITION_SIZE;
        int* d_partition_starts;
        int* d_partition_ends;

        cudaMalloc(&d_partition_starts, estimated_partitions * sizeof(int));
        cudaMalloc(&d_partition_ends, estimated_partitions * sizeof(int));

        blocks = min((num_variance_blocks + threads - 1) / threads, sm_count * 2);

        int* d_partition_counts;
        int* d_partition_offsets;
        cudaMalloc(&d_partition_counts, num_variance_blocks * sizeof(int));
        cudaMalloc(&d_partition_offsets, (num_variance_blocks + 1) * sizeof(int));

        countPartitionsPerBlock<T><<<blocks, threads, 0, stream>>>(
            data_size, base_partition_size, d_variances, num_variance_blocks,
            d_partition_counts,
            d_variance_thresholds, d_partition_sizes_for_buckets,
            num_thresholds, variance_block_multiplier);

        thrust::device_ptr<int> counts_ptr(d_partition_counts);
        thrust::device_ptr<int> offsets_ptr(d_partition_offsets);
        thrust::exclusive_scan(counts_ptr, counts_ptr + num_variance_blocks, offsets_ptr);

        int h_num_partitions;
        int h_last_count;
        cudaMemcpyAsync(&h_num_partitions,
                       d_partition_offsets + num_variance_blocks - 1,
                       sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_last_count,
                       d_partition_counts + num_variance_blocks - 1,
                       sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_num_partitions += h_last_count;

        if (h_num_partitions > estimated_partitions) {
            std::cerr << "Warning: Partition count " << h_num_partitions
                      << " exceeds estimate " << estimated_partitions << std::endl;
            h_num_partitions = estimated_partitions;
        }

        writePartitionsOrdered<T><<<blocks, threads, 0, stream>>>(
            data_size, base_partition_size, d_variances, num_variance_blocks,
            d_partition_offsets,
            d_partition_starts, d_partition_ends,
            d_variance_thresholds, d_partition_sizes_for_buckets,
            num_thresholds, variance_block_multiplier);

        cudaFree(d_partition_counts);
        cudaFree(d_partition_offsets);

        int* d_model_types;
        double* d_theta0;
        double* d_theta1;
        int* d_delta_bits;
        long long* d_max_errors;
        double* d_costs;

        cudaMalloc(&d_model_types, h_num_partitions * sizeof(int));
        cudaMalloc(&d_theta0, h_num_partitions * sizeof(double));
        cudaMalloc(&d_theta1, h_num_partitions * sizeof(double));
        cudaMalloc(&d_delta_bits, h_num_partitions * sizeof(int));
        cudaMalloc(&d_max_errors, h_num_partitions * sizeof(long long));
        cudaMalloc(&d_costs, h_num_partitions * sizeof(double));

        int threads_per_block = 256;
        int grid_size = h_num_partitions;

        if (grid_size > 0) {
            fitPartitionsBatched_Optimized<T><<<grid_size, threads_per_block, 0, stream>>>(
                d_data,
                d_partition_starts,
                d_partition_ends,
                d_model_types,
                d_theta0,
                d_theta1,
                d_delta_bits,
                d_max_errors,
                d_costs,
                h_num_partitions
            );
        }

        std::vector<int> h_starts(h_num_partitions);
        std::vector<int> h_ends(h_num_partitions);
        std::vector<int> h_model_types(h_num_partitions);
        std::vector<double> h_theta0(h_num_partitions);
        std::vector<double> h_theta1(h_num_partitions);
        std::vector<int> h_delta_bits(h_num_partitions);
        std::vector<long long> h_max_errors(h_num_partitions);

        if (h_num_partitions > 0) {
            cudaMemcpyAsync(h_starts.data(), d_partition_starts,
                           h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_ends.data(), d_partition_ends,
                           h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_model_types.data(), d_model_types,
                           h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_theta0.data(), d_theta0,
                           h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_theta1.data(), d_theta1,
                           h_num_partitions * sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_delta_bits.data(), d_delta_bits,
                           h_num_partitions * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(h_max_errors.data(), d_max_errors,
                           h_num_partitions * sizeof(long long), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);

        std::vector<PartitionInfo> result;
        result.reserve(h_num_partitions);

        for (int i = 0; i < h_num_partitions; i++) {
            PartitionInfo info;
            info.start_idx = h_starts[i];
            info.end_idx = h_ends[i];
            info.model_type = h_model_types[i];
            info.model_params[0] = h_theta0[i];
            info.model_params[1] = h_theta1[i];
            info.model_params[2] = 0.0;
            info.model_params[3] = 0.0;
            info.delta_bits = h_delta_bits[i];
            info.delta_array_bit_offset = 0;
            info.error_bound = h_max_errors[i];
            result.push_back(info);
        }

        if (!result.empty()) {
            std::sort(result.begin(), result.end(),
                     [](const PartitionInfo& a, const PartitionInfo& b) {
                         return a.start_idx < b.start_idx;
                     });

            result[0].start_idx = 0;
            result.back().end_idx = data_size;

            for (size_t i = 0; i < result.size() - 1; i++) {
                if (result[i].end_idx != result[i + 1].start_idx) {
                    result[i].end_idx = result[i + 1].start_idx;
                }
            }
        }

        // Cleanup
        cudaFree(d_variances);
        cudaFree(d_variance_thresholds);
        cudaFree(d_partition_sizes_for_buckets);
        cudaFree(d_partition_starts);
        cudaFree(d_partition_ends);
        cudaFree(d_model_types);
        cudaFree(d_theta0);
        cudaFree(d_theta1);
        cudaFree(d_delta_bits);
        cudaFree(d_max_errors);
        cudaFree(d_costs);

        return result;
    }
};

// ============================================================================
// Public API
// ============================================================================

template<typename T>
std::vector<PartitionInfo> createPartitionsVariableLength(
    const std::vector<T>& data,
    int base_partition_size,
    int* num_partitions_out,
    cudaStream_t stream,
    int variance_block_multiplier,
    int num_thresholds)
{
    GPUVariableLengthPartitionerV6<T> partitioner(
        data, base_partition_size, stream,
        variance_block_multiplier, num_thresholds);

    std::vector<PartitionInfo> result = partitioner.partition();

    if (num_partitions_out) {
        *num_partitions_out = static_cast<int>(result.size());
    }

    return result;
}

// Explicit template instantiation
template class GPUVariableLengthPartitionerV6<uint64_t>;
template class GPUVariableLengthPartitionerV6<int64_t>;
template class GPUVariableLengthPartitionerV6<uint32_t>;
template class GPUVariableLengthPartitionerV6<int32_t>;

template std::vector<PartitionInfo> createPartitionsVariableLength<uint64_t>(
    const std::vector<uint64_t>&, int, int*, cudaStream_t, int, int);
template std::vector<PartitionInfo> createPartitionsVariableLength<int64_t>(
    const std::vector<int64_t>&, int, int*, cudaStream_t, int, int);
template std::vector<PartitionInfo> createPartitionsVariableLength<uint32_t>(
    const std::vector<uint32_t>&, int, int*, cudaStream_t, int, int);
template std::vector<PartitionInfo> createPartitionsVariableLength<int32_t>(
    const std::vector<int32_t>&, int, int*, cudaStream_t, int, int);
