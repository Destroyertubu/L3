/**
 * Benchmark: Multi-Model vs LINEAR-Only Merge Evaluation
 *
 * This test compares two merge evaluation strategies:
 * - V2 (LINEAR-Only): Current implementation using only LINEAR model for merge cost estimation
 * - V3 (Multi-Model): Evaluates FOR, LINEAR, POLY2, POLY3 and selects the best
 *
 * Goal: Verify if multi-model evaluation produces different/better results on SSB data
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iomanip>

// ============================================================================
// Configuration
// ============================================================================

constexpr int BLOCK_SIZE = 256;
constexpr int TARGET_PARTITION_SIZE = 4096;
constexpr int MIN_PARTITION_SIZE = 256;
constexpr int MAX_PARTITION_SIZE = 8192;
constexpr float MERGE_BENEFIT_THRESHOLD = 0.05f;
constexpr int MAX_MERGE_ROUNDS = 4;

constexpr int POLY_MIN_SIZE = 10;
constexpr int CUBIC_MIN_SIZE = 20;
constexpr float MODEL_SELECTION_THRESHOLD = 0.95f;  // 5% improvement required

// Model types
constexpr int MODEL_FOR_BITPACK = 4;
constexpr int MODEL_LINEAR = 1;
constexpr int MODEL_POLYNOMIAL2 = 2;
constexpr int MODEL_POLYNOMIAL3 = 3;

// Model overhead bytes
constexpr float FOR_OVERHEAD = 4.0f;      // Just base value
constexpr float LINEAR_OVERHEAD = 16.0f;  // theta0, theta1
constexpr float POLY2_OVERHEAD = 24.0f;   // theta0, theta1, theta2
constexpr float POLY3_OVERHEAD = 32.0f;   // theta0, theta1, theta2, theta3

// ============================================================================
// GPU Helper Functions
// ============================================================================

__device__ __forceinline__ int computeBitsForValue(unsigned long long val) {
    if (val == 0) return 0;
    return 64 - __clzll(val);
}

template<typename T>
__device__ T blockReduceMin(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] :
          (sizeof(T) == 8 ? (T)LLONG_MAX : (T)INT_MAX);

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] :
          (sizeof(T) == 8 ? (T)LLONG_MIN : (T)INT_MIN);

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ double blockReduceSumDouble(double val) {
    __shared__ double shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ long long blockReduceMaxLL(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : LLONG_MIN;

    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ============================================================================
// 4x4 Linear System Solver (Gaussian Elimination with Partial Pivoting)
// ============================================================================

__device__ bool solve4x4(double A[16], double b[4], double x[4]) {
    // Forward elimination with partial pivoting
    for (int k = 0; k < 4; k++) {
        // Find pivot
        int max_row = k;
        double max_val = fabs(A[k * 4 + k]);
        for (int i = k + 1; i < 4; i++) {
            if (fabs(A[i * 4 + k]) > max_val) {
                max_val = fabs(A[i * 4 + k]);
                max_row = i;
            }
        }

        // Swap rows
        if (max_row != k) {
            for (int j = 0; j < 4; j++) {
                double tmp = A[k * 4 + j];
                A[k * 4 + j] = A[max_row * 4 + j];
                A[max_row * 4 + j] = tmp;
            }
            double tmp = b[k];
            b[k] = b[max_row];
            b[max_row] = tmp;
        }

        // Check for singular matrix
        if (fabs(A[k * 4 + k]) < 1e-12) {
            return false;
        }

        // Eliminate
        for (int i = k + 1; i < 4; i++) {
            double factor = A[i * 4 + k] / A[k * 4 + k];
            for (int j = k; j < 4; j++) {
                A[i * 4 + j] -= factor * A[k * 4 + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int i = 3; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < 4; j++) {
            x[i] -= A[i * 4 + j] * x[j];
        }
        x[i] /= A[i * 4 + i];
    }

    return true;
}

// Power sum formulas for scaled coordinates
__device__ double sumXScaled(double n) { return n * (n - 1.0) / 2.0; }
__device__ double sumX2Scaled(double n) { return n * (n - 1.0) * (2.0 * n - 1.0) / 6.0; }
__device__ double sumX3Scaled(double n) { double s = n * (n - 1.0) / 2.0; return s * s; }
__device__ double sumX4Scaled(double n) { return n * (n - 1.0) * (2.0 * n - 1.0) * (3.0 * n * n - 3.0 * n - 1.0) / 30.0; }
__device__ double sumX5Scaled(double n) {
    double nn = n * n;
    double nm1 = n - 1.0;
    return nn * nm1 * nm1 * (2.0 * nn - 2.0 * n - 1.0) / 12.0;
}
__device__ double sumX6Scaled(double n) {
    double m = n - 1.0;
    double m2 = m * m;
    double m3 = m2 * m;
    double m4 = m2 * m2;
    return m * (m + 1.0) * (2.0 * m + 1.0) * (3.0 * m4 + 6.0 * m3 - 3.0 * m + 1.0) / 42.0;
}

// ============================================================================
// Multi-Model Merge Evaluation Kernel (V3)
// ============================================================================

template<typename T>
__global__ void evaluateMergeMultiModelKernel(
    const T* __restrict__ data,
    const int* __restrict__ starts,
    const int* __restrict__ ends,
    const float* __restrict__ costs,
    // Outputs
    float* __restrict__ merge_benefits_v3,
    int* __restrict__ best_model_types,
    int* __restrict__ best_delta_bits,
    // For comparison with V2
    float* __restrict__ merge_benefits_v2,
    int* __restrict__ linear_delta_bits,
    // Config
    int num_partitions,
    int max_partition_size)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions - 1) return;

    int start_a = starts[pid];
    int end_a = ends[pid];
    int start_b = starts[pid + 1];
    int end_b = ends[pid + 1];

    int n_a = end_a - start_a;
    int n_b = end_b - start_b;
    int n_c = n_a + n_b;

    // Check size constraint
    if (n_c > max_partition_size) {
        if (threadIdx.x == 0) {
            merge_benefits_v3[pid] = -1.0f;
            merge_benefits_v2[pid] = -1.0f;
            best_model_types[pid] = MODEL_FOR_BITPACK;
            best_delta_bits[pid] = 0;
            linear_delta_bits[pid] = 0;
        }
        return;
    }

    int merged_start = start_a;
    int merged_end = end_b;
    int n = n_c;
    double dn = (double)n;

    // ========== Phase 1: Collect statistics ==========
    double local_sum_y = 0.0, local_sum_xy = 0.0;
    double local_sum_x2y = 0.0, local_sum_x3y = 0.0, local_sum_x4y = 0.0;
    T local_min = (sizeof(T) == 8) ? (T)LLONG_MAX : (T)INT_MAX;
    T local_max = (sizeof(T) == 8) ? (T)LLONG_MIN : (T)INT_MIN;

    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        int local_idx = i - merged_start;
        double x = (double)local_idx;
        double x2 = x * x;
        double x3 = x2 * x;
        T val = data[i];
        double y = (double)val;

        local_sum_y += y;
        local_sum_xy += x * y;
        local_sum_x2y += x2 * y;
        local_sum_x3y += x3 * y;
        local_sum_x4y += x2 * x2 * y;  // Σx⁴y for POLY3

        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    // Block reductions
    double sum_y = blockReduceSumDouble(local_sum_y);
    __syncthreads();
    double sum_xy = blockReduceSumDouble(local_sum_xy);
    __syncthreads();
    double sum_x2y = blockReduceSumDouble(local_sum_x2y);
    __syncthreads();
    double sum_x3y = blockReduceSumDouble(local_sum_x3y);
    __syncthreads();
    double sum_x4y = blockReduceSumDouble(local_sum_x4y);
    __syncthreads();
    T global_min = blockReduceMin(local_min);
    __syncthreads();
    T global_max = blockReduceMax(local_max);
    __syncthreads();

    // ========== Phase 2: Compute model parameters (thread 0) ==========
    __shared__ double s_linear_params[2];  // theta0, theta1
    __shared__ double s_poly2_params[3];   // theta0, theta1, theta2
    __shared__ double s_poly3_params[4];   // theta0, theta1, theta2, theta3
    __shared__ T s_global_min, s_global_max;

    if (threadIdx.x == 0) {
        s_global_min = global_min;
        s_global_max = global_max;

        // Sum formulas for x^k
        double sx = dn * (dn - 1.0) / 2.0;
        double sx2 = dn * (dn - 1.0) * (2.0 * dn - 1.0) / 6.0;
        double sx3 = sx * sx;
        double sx4 = dn * (dn - 1.0) * (2.0 * dn - 1.0) * (3.0 * dn * dn - 3.0 * dn - 1.0) / 30.0;

        // LINEAR fit
        double det = dn * sx2 - sx * sx;
        if (fabs(det) > 1e-10) {
            s_linear_params[1] = (dn * sum_xy - sx * sum_y) / det;
            s_linear_params[0] = (sum_y - s_linear_params[1] * sx) / dn;
        } else {
            s_linear_params[1] = 0.0;
            s_linear_params[0] = sum_y / dn;
        }

        // POLY2 fit (3x3 system)
        if (n >= POLY_MIN_SIZE) {
            double a00 = dn, a01 = sx, a02 = sx2;
            double a10 = sx, a11 = sx2, a12 = sx3;
            double a20 = sx2, a21 = sx3, a22 = sx4;
            double b0 = sum_y, b1 = sum_xy, b2 = sum_x2y;

            double det2 = a00 * (a11 * a22 - a12 * a21) -
                          a01 * (a10 * a22 - a12 * a20) +
                          a02 * (a10 * a21 - a11 * a20);

            if (fabs(det2) > 1e-10) {
                s_poly2_params[0] = (b0 * (a11 * a22 - a12 * a21) -
                                     a01 * (b1 * a22 - a12 * b2) +
                                     a02 * (b1 * a21 - a11 * b2)) / det2;
                s_poly2_params[1] = (a00 * (b1 * a22 - a12 * b2) -
                                     b0 * (a10 * a22 - a12 * a20) +
                                     a02 * (a10 * b2 - b1 * a20)) / det2;
                s_poly2_params[2] = (a00 * (a11 * b2 - b1 * a21) -
                                     a01 * (a10 * b2 - b1 * a20) +
                                     b0 * (a10 * a21 - a11 * a20)) / det2;
            } else {
                s_poly2_params[0] = s_linear_params[0];
                s_poly2_params[1] = s_linear_params[1];
                s_poly2_params[2] = 0.0;
            }
        } else {
            s_poly2_params[0] = s_linear_params[0];
            s_poly2_params[1] = s_linear_params[1];
            s_poly2_params[2] = 0.0;
        }

        // POLY3 fit using scaled coordinates (4x4 system)
        if (n >= CUBIC_MIN_SIZE) {
            // Scale factor: x' = x / scale, where scale = n-1
            // This maps x from [0, n-1] to [0, 1] for numerical stability
            double scale = dn - 1.0;
            double s2 = scale * scale;
            double s3 = s2 * scale;
            double s4 = s2 * s2;
            double s5 = s4 * scale;
            double s6 = s3 * s3;

            // Scaled power sums
            double sx_sc = sumXScaled(dn) / scale;
            double sx2_sc = sumX2Scaled(dn) / s2;
            double sx3_sc = sumX3Scaled(dn) / s3;
            double sx4_sc = sumX4Scaled(dn) / s4;
            double sx5_sc = sumX5Scaled(dn) / s5;
            double sx6_sc = sumX6Scaled(dn) / s6;

            // Scaled data sums
            double sum_xpy_sc = sum_xy / scale;
            double sum_x2py_sc = sum_x2y / s2;
            double sum_x3py_sc = sum_x3y / s3;

            // Build 4x4 Vandermonde normal equations matrix in scaled coordinates
            double A[16] = {
                dn,      sx_sc,   sx2_sc,  sx3_sc,
                sx_sc,   sx2_sc,  sx3_sc,  sx4_sc,
                sx2_sc,  sx3_sc,  sx4_sc,  sx5_sc,
                sx3_sc,  sx4_sc,  sx5_sc,  sx6_sc
            };
            double b[4] = {sum_y, sum_xpy_sc, sum_x2py_sc, sum_x3py_sc};
            double alpha[4];

            if (solve4x4(A, b, alpha)) {
                // Transform back to original coordinates:
                // y = α₀ + α₁(x/s) + α₂(x/s)² + α₃(x/s)³
                //   = α₀ + (α₁/s)x + (α₂/s²)x² + (α₃/s³)x³
                s_poly3_params[0] = alpha[0];
                s_poly3_params[1] = alpha[1] / scale;
                s_poly3_params[2] = alpha[2] / s2;
                s_poly3_params[3] = alpha[3] / s3;
            } else {
                // Fallback to POLY2
                s_poly3_params[0] = s_poly2_params[0];
                s_poly3_params[1] = s_poly2_params[1];
                s_poly3_params[2] = s_poly2_params[2];
                s_poly3_params[3] = 0.0;
            }
        } else {
            // Not enough points for stable cubic fit
            s_poly3_params[0] = s_poly2_params[0];
            s_poly3_params[1] = s_poly2_params[1];
            s_poly3_params[2] = s_poly2_params[2];
            s_poly3_params[3] = 0.0;
        }
    }
    __syncthreads();

    // ========== Phase 3: Compute max errors for each model ==========
    long long linear_max_err = 0, poly2_max_err = 0, poly3_max_err = 0;

    for (int i = merged_start + threadIdx.x; i < merged_end; i += blockDim.x) {
        int local_idx = i - merged_start;
        double x = (double)local_idx;
        T val = data[i];

        // LINEAR error
        double pred_linear = s_linear_params[0] + s_linear_params[1] * x;
        T pv_linear = (T)__double2ll_rn(pred_linear);
        long long err_linear = (val >= pv_linear) ? (long long)(val - pv_linear) : -(long long)(pv_linear - val);
        linear_max_err = max(linear_max_err, llabs(err_linear));

        // POLY2 error
        double pred_poly2 = s_poly2_params[0] + x * (s_poly2_params[1] + x * s_poly2_params[2]);
        T pv_poly2 = (T)__double2ll_rn(pred_poly2);
        long long err_poly2 = (val >= pv_poly2) ? (long long)(val - pv_poly2) : -(long long)(pv_poly2 - val);
        poly2_max_err = max(poly2_max_err, llabs(err_poly2));

        // POLY3 error (Horner's method: a0 + x*(a1 + x*(a2 + x*a3)))
        double pred_poly3 = s_poly3_params[0] + x * (s_poly3_params[1] + x * (s_poly3_params[2] + x * s_poly3_params[3]));
        T pv_poly3 = (T)__double2ll_rn(pred_poly3);
        long long err_poly3 = (val >= pv_poly3) ? (long long)(val - pv_poly3) : -(long long)(pv_poly3 - val);
        poly3_max_err = max(poly3_max_err, llabs(err_poly3));
    }

    linear_max_err = blockReduceMaxLL(linear_max_err);
    __syncthreads();
    poly2_max_err = blockReduceMaxLL(poly2_max_err);
    __syncthreads();
    poly3_max_err = blockReduceMaxLL(poly3_max_err);
    __syncthreads();

    // ========== Phase 4: Select best model and compute benefits ==========
    if (threadIdx.x == 0) {
        // Compute bits for each model
        int linear_bits = (linear_max_err > 0) ? computeBitsForValue((unsigned long long)linear_max_err) + 1 : 0;
        int poly2_bits = (poly2_max_err > 0) ? computeBitsForValue((unsigned long long)poly2_max_err) + 1 : 0;
        int poly3_bits = (poly3_max_err > 0) ? computeBitsForValue((unsigned long long)poly3_max_err) + 1 : 0;

        // FOR model: range = max - min
        uint64_t range = (uint64_t)s_global_max - (uint64_t)s_global_min;
        int for_bits = (range > 0) ? computeBitsForValue(range) : 0;

        // Compute costs
        float fn = (float)n;
        float for_cost = FOR_OVERHEAD + fn * for_bits / 8.0f;
        float linear_cost = LINEAR_OVERHEAD + fn * linear_bits / 8.0f;
        float poly2_cost = POLY2_OVERHEAD + fn * poly2_bits / 8.0f;
        float poly3_cost = POLY3_OVERHEAD + fn * poly3_bits / 8.0f;

        // Select best model
        int best_model = MODEL_FOR_BITPACK;
        float best_cost = for_cost;
        int best_bits = for_bits;

        if (linear_cost < best_cost * MODEL_SELECTION_THRESHOLD) {
            best_model = MODEL_LINEAR;
            best_cost = linear_cost;
            best_bits = linear_bits;
        }

        if (n >= POLY_MIN_SIZE && poly2_cost < best_cost * MODEL_SELECTION_THRESHOLD) {
            best_model = MODEL_POLYNOMIAL2;
            best_cost = poly2_cost;
            best_bits = poly2_bits;
        }

        if (n >= CUBIC_MIN_SIZE && poly3_cost < best_cost * MODEL_SELECTION_THRESHOLD) {
            best_model = MODEL_POLYNOMIAL3;
            best_cost = poly3_cost;
            best_bits = poly3_bits;
        }

        // Compute merge benefits
        float separate_cost = costs[pid] + costs[pid + 1];

        // V3: Multi-Model benefit
        float benefit_v3 = (separate_cost - best_cost) / separate_cost;
        merge_benefits_v3[pid] = benefit_v3;
        best_model_types[pid] = best_model;
        best_delta_bits[pid] = best_bits;

        // V2: LINEAR-Only benefit (for comparison)
        float benefit_v2 = (separate_cost - linear_cost) / separate_cost;
        merge_benefits_v2[pid] = benefit_v2;
        linear_delta_bits[pid] = linear_bits;
    }
}

// ============================================================================
// Initial Partitioning Kernel (create fixed-size partitions)
// ============================================================================

template<typename T>
__global__ void computePartitionCostsKernel(
    const T* __restrict__ data,
    const int* __restrict__ starts,
    const int* __restrict__ ends,
    float* __restrict__ costs,
    int* __restrict__ delta_bits,
    int* __restrict__ model_types,
    int num_partitions)
{
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = starts[pid];
    int end = ends[pid];
    int n = end - start;

    // Find min/max for FOR model
    T local_min = (sizeof(T) == 8) ? (T)LLONG_MAX : (T)INT_MAX;
    T local_max = (sizeof(T) == 8) ? (T)LLONG_MIN : (T)INT_MIN;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        local_min = min(local_min, val);
        local_max = max(local_max, val);
    }

    T global_min = blockReduceMin(local_min);
    __syncthreads();
    T global_max = blockReduceMax(local_max);
    __syncthreads();

    if (threadIdx.x == 0) {
        uint64_t range = (uint64_t)global_max - (uint64_t)global_min;
        int bits = (range > 0) ? computeBitsForValue(range) : 0;

        delta_bits[pid] = bits;
        model_types[pid] = MODEL_FOR_BITPACK;
        costs[pid] = FOR_OVERHEAD + (float)n * bits / 8.0f;
    }
}

// ============================================================================
// Host Code: Load SSB Data
// ============================================================================

std::vector<uint32_t> loadSSBColumn(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open " << filepath << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t num_elements = size / sizeof(uint32_t);
    std::vector<uint32_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}

// ============================================================================
// Comparison Test Function
// ============================================================================

struct ComparisonResult {
    std::string column_name;
    size_t num_values;
    int initial_partitions;
    int v2_final_partitions;
    int v3_final_partitions;
    float v2_compress_time_ms;
    float v3_compress_time_ms;
    float v2_total_size_mb;
    float v3_total_size_mb;
    int merge_decisions_same;
    int merge_decisions_diff;
    int v3_for_count;
    int v3_linear_count;
    int v3_poly2_count;
    int v3_poly3_count;  // NEW: POLY3 count
};

template<typename T>
ComparisonResult compareStrategies(const std::vector<T>& h_data, const std::string& column_name) {
    ComparisonResult result;
    result.column_name = column_name;
    result.num_values = h_data.size();

    size_t n = h_data.size();

    // Allocate device memory
    T* d_data;
    cudaMalloc(&d_data, n * sizeof(T));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(T), cudaMemcpyHostToDevice);

    // Create initial fixed-size partitions
    int num_partitions = (n + TARGET_PARTITION_SIZE - 1) / TARGET_PARTITION_SIZE;
    result.initial_partitions = num_partitions;

    std::vector<int> h_starts(num_partitions), h_ends(num_partitions);
    for (int i = 0; i < num_partitions; i++) {
        h_starts[i] = i * TARGET_PARTITION_SIZE;
        h_ends[i] = std::min((i + 1) * TARGET_PARTITION_SIZE, (int)n);
    }

    int *d_starts, *d_ends;
    cudaMalloc(&d_starts, num_partitions * sizeof(int));
    cudaMalloc(&d_ends, num_partitions * sizeof(int));
    cudaMemcpy(d_starts, h_starts.data(), num_partitions * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ends, h_ends.data(), num_partitions * sizeof(int), cudaMemcpyHostToDevice);

    float *d_costs;
    int *d_delta_bits, *d_model_types;
    cudaMalloc(&d_costs, num_partitions * sizeof(float));
    cudaMalloc(&d_delta_bits, num_partitions * sizeof(int));
    cudaMalloc(&d_model_types, num_partitions * sizeof(int));

    // Compute initial partition costs
    computePartitionCostsKernel<<<num_partitions, BLOCK_SIZE>>>(
        d_data, d_starts, d_ends, d_costs, d_delta_bits, d_model_types, num_partitions);
    cudaDeviceSynchronize();

    // Allocate merge evaluation outputs
    float *d_benefits_v2, *d_benefits_v3;
    int *d_best_models, *d_best_bits, *d_linear_bits;
    cudaMalloc(&d_benefits_v2, (num_partitions - 1) * sizeof(float));
    cudaMalloc(&d_benefits_v3, (num_partitions - 1) * sizeof(float));
    cudaMalloc(&d_best_models, (num_partitions - 1) * sizeof(int));
    cudaMalloc(&d_best_bits, (num_partitions - 1) * sizeof(int));
    cudaMalloc(&d_linear_bits, (num_partitions - 1) * sizeof(int));

    // Time V3 (Multi-Model) evaluation
    cudaEvent_t start_v3, stop_v3;
    cudaEventCreate(&start_v3);
    cudaEventCreate(&stop_v3);

    cudaEventRecord(start_v3);
    evaluateMergeMultiModelKernel<<<num_partitions - 1, BLOCK_SIZE>>>(
        d_data, d_starts, d_ends, d_costs,
        d_benefits_v3, d_best_models, d_best_bits,
        d_benefits_v2, d_linear_bits,
        num_partitions, MAX_PARTITION_SIZE);
    cudaEventRecord(stop_v3);
    cudaEventSynchronize(stop_v3);

    float v3_time_ms;
    cudaEventElapsedTime(&v3_time_ms, start_v3, stop_v3);
    result.v3_compress_time_ms = v3_time_ms;
    result.v2_compress_time_ms = v3_time_ms;  // V2 runs in same kernel for fair comparison

    // Copy results back
    std::vector<float> h_benefits_v2(num_partitions - 1), h_benefits_v3(num_partitions - 1);
    std::vector<int> h_best_models(num_partitions - 1), h_best_bits(num_partitions - 1);
    std::vector<int> h_linear_bits(num_partitions - 1);
    std::vector<float> h_costs(num_partitions);
    std::vector<int> h_delta_bits(num_partitions);

    cudaMemcpy(h_benefits_v2.data(), d_benefits_v2, (num_partitions - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_benefits_v3.data(), d_benefits_v3, (num_partitions - 1) * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_models.data(), d_best_models, (num_partitions - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_best_bits.data(), d_best_bits, (num_partitions - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_linear_bits.data(), d_linear_bits, (num_partitions - 1) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_costs.data(), d_costs, num_partitions * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta_bits.data(), d_delta_bits, num_partitions * sizeof(int), cudaMemcpyDeviceToHost);

    // Analyze merge decisions
    int same_decisions = 0, diff_decisions = 0;
    int v2_merges = 0, v3_merges = 0;
    int for_count = 0, linear_count = 0, poly2_count = 0, poly3_count = 0;

    for (int i = 0; i < num_partitions - 1; i++) {
        bool v2_would_merge = h_benefits_v2[i] >= MERGE_BENEFIT_THRESHOLD;
        bool v3_would_merge = h_benefits_v3[i] >= MERGE_BENEFIT_THRESHOLD;

        if (v2_would_merge == v3_would_merge) {
            same_decisions++;
        } else {
            diff_decisions++;
        }

        if (v2_would_merge) v2_merges++;
        if (v3_would_merge) v3_merges++;

        // Count model selections
        if (h_best_models[i] == MODEL_FOR_BITPACK) for_count++;
        else if (h_best_models[i] == MODEL_LINEAR) linear_count++;
        else if (h_best_models[i] == MODEL_POLYNOMIAL2) poly2_count++;
        else if (h_best_models[i] == MODEL_POLYNOMIAL3) poly3_count++;
    }

    result.merge_decisions_same = same_decisions;
    result.merge_decisions_diff = diff_decisions;
    result.v3_for_count = for_count;
    result.v3_linear_count = linear_count;
    result.v3_poly2_count = poly2_count;
    result.v3_poly3_count = poly3_count;

    // Estimate final partition counts (simplified - just count potential merges)
    result.v2_final_partitions = num_partitions - v2_merges;
    result.v3_final_partitions = num_partitions - v3_merges;

    // Calculate total compressed sizes
    float v2_total = 0, v3_total = 0;
    for (int i = 0; i < num_partitions; i++) {
        v2_total += h_costs[i];
    }
    // V3 uses best model costs
    for (int i = 0; i < num_partitions - 1; i++) {
        if (h_benefits_v3[i] >= MERGE_BENEFIT_THRESHOLD) {
            // Would merge: subtract old costs, add merged cost
            float merged_cost = FOR_OVERHEAD + (float)(h_ends[i+1] - h_starts[i]) * h_best_bits[i] / 8.0f;
            v3_total -= h_costs[i] + h_costs[i+1];
            v3_total += merged_cost;
        }
    }
    v3_total += v2_total;  // Base cost

    result.v2_total_size_mb = v2_total / (1024.0f * 1024.0f);
    result.v3_total_size_mb = v3_total / (1024.0f * 1024.0f);

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_starts);
    cudaFree(d_ends);
    cudaFree(d_costs);
    cudaFree(d_delta_bits);
    cudaFree(d_model_types);
    cudaFree(d_benefits_v2);
    cudaFree(d_benefits_v3);
    cudaFree(d_best_models);
    cudaFree(d_best_bits);
    cudaFree(d_linear_bits);
    cudaEventDestroy(start_v3);
    cudaEventDestroy(stop_v3);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::string ssb_data_dir = "data/ssb";
    if (argc > 1) {
        ssb_data_dir = argv[1];
    }

    std::cout << "========== Merge Strategy Comparison: V2 (LINEAR) vs V3 (Multi-Model) ==========\n\n";
    std::cout << "Configuration:\n";
    std::cout << "  Target partition size: " << TARGET_PARTITION_SIZE << "\n";
    std::cout << "  Max partition size: " << MAX_PARTITION_SIZE << "\n";
    std::cout << "  Merge benefit threshold: " << (MERGE_BENEFIT_THRESHOLD * 100) << "%\n";
    std::cout << "  Model selection threshold: " << ((1.0f - MODEL_SELECTION_THRESHOLD) * 100) << "% improvement required\n\n";

    // Test columns with different characteristics
    struct TestColumn {
        std::string filename;
        std::string description;
    };

    std::vector<TestColumn> columns = {
        {"LINEORDER8.bin", "lo_quantity (range 1-50, ~6 bits)"},
        {"LINEORDER11.bin", "lo_discount (range 0-10, ~4 bits)"},
        {"LINEORDER4.bin", "lo_suppkey (range 1-20000, ~15 bits)"},
        {"LINEORDER2.bin", "lo_custkey (range 1-300000, ~19 bits)"},
        {"LINEORDER3.bin", "lo_partkey (range 1-800000, ~20 bits)"},
        {"LINEORDER5.bin", "lo_orderdate (date range, ~16 bits)"},
        {"LINEORDER12.bin", "lo_revenue (large range, ~16 bits)"},
        {"LINEORDER13.bin", "lo_supplycost (medium range, ~10 bits)"},
        {"LINEORDER9.bin", "lo_extendedprice (large range, ~16 bits)"}
    };

    std::vector<ComparisonResult> results;

    for (const auto& col : columns) {
        std::string filepath = ssb_data_dir + "/" + col.filename;
        auto data = loadSSBColumn(filepath);

        if (data.empty()) {
            std::cerr << "Skipping " << col.filename << " (not found)\n";
            continue;
        }

        std::cout << "Processing: " << col.description << " (" << data.size() << " values)\n";

        auto result = compareStrategies(data, col.description);
        results.push_back(result);

        std::cout << "  Initial partitions: " << result.initial_partitions << "\n";
        std::cout << "  V2 would merge to: " << result.v2_final_partitions << " partitions\n";
        std::cout << "  V3 would merge to: " << result.v3_final_partitions << " partitions\n";
        std::cout << "  Same merge decisions: " << result.merge_decisions_same
                  << " (" << (100.0f * result.merge_decisions_same / (result.merge_decisions_same + result.merge_decisions_diff)) << "%)\n";
        std::cout << "  Different decisions: " << result.merge_decisions_diff << "\n";
        std::cout << "  V3 best model distribution:\n";
        std::cout << "    FOR: " << result.v3_for_count << " ("
                  << (100.0f * result.v3_for_count / (result.initial_partitions - 1)) << "%)\n";
        std::cout << "    LINEAR: " << result.v3_linear_count << " ("
                  << (100.0f * result.v3_linear_count / (result.initial_partitions - 1)) << "%)\n";
        std::cout << "    POLY2: " << result.v3_poly2_count << " ("
                  << (100.0f * result.v3_poly2_count / (result.initial_partitions - 1)) << "%)\n";
        std::cout << "    POLY3: " << result.v3_poly3_count << " ("
                  << (100.0f * result.v3_poly3_count / (result.initial_partitions - 1)) << "%)\n";
        std::cout << "  Evaluation time: " << result.v3_compress_time_ms << " ms\n\n";
    }

    // Print summary table
    std::cout << "\n========== Summary Table ==========\n\n";
    std::cout << std::setw(30) << "Column"
              << std::setw(12) << "Init Part"
              << std::setw(12) << "V2 Final"
              << std::setw(12) << "V3 Final"
              << std::setw(12) << "Same %"
              << std::setw(12) << "FOR %"
              << std::setw(12) << "LINEAR %"
              << std::setw(12) << "POLY3 %"
              << "\n";
    std::cout << std::string(114, '-') << "\n";

    for (const auto& r : results) {
        float same_pct = 100.0f * r.merge_decisions_same / (r.merge_decisions_same + r.merge_decisions_diff);
        float for_pct = 100.0f * r.v3_for_count / (r.initial_partitions - 1);
        float linear_pct = 100.0f * r.v3_linear_count / (r.initial_partitions - 1);
        float poly3_pct = 100.0f * r.v3_poly3_count / (r.initial_partitions - 1);

        std::cout << std::setw(30) << r.column_name.substr(0, 28)
                  << std::setw(12) << r.initial_partitions
                  << std::setw(12) << r.v2_final_partitions
                  << std::setw(12) << r.v3_final_partitions
                  << std::setw(11) << std::fixed << std::setprecision(1) << same_pct << "%"
                  << std::setw(11) << std::fixed << std::setprecision(1) << for_pct << "%"
                  << std::setw(11) << std::fixed << std::setprecision(1) << linear_pct << "%"
                  << std::setw(11) << std::fixed << std::setprecision(1) << poly3_pct << "%"
                  << "\n";
    }

    std::cout << "\n========== Conclusion ==========\n\n";

    // Analyze overall results
    int total_same = 0, total_diff = 0;
    for (const auto& r : results) {
        total_same += r.merge_decisions_same;
        total_diff += r.merge_decisions_diff;
    }

    float overall_same_pct = 100.0f * total_same / (total_same + total_diff);

    std::cout << "Overall merge decision agreement: " << std::fixed << std::setprecision(1)
              << overall_same_pct << "%\n";

    if (overall_same_pct > 95.0f) {
        std::cout << "=> V2 (LINEAR-only) and V3 (Multi-Model) produce nearly identical merge decisions.\n";
        std::cout << "=> For SSB data, the simpler V2 approach is sufficient.\n";
    } else if (overall_same_pct > 80.0f) {
        std::cout << "=> V2 and V3 have some differences, but mostly agree.\n";
        std::cout << "=> V3 may provide marginal benefits for certain columns.\n";
    } else {
        std::cout << "=> V2 and V3 produce significantly different merge decisions.\n";
        std::cout << "=> V3 (Multi-Model) should be considered for better accuracy.\n";
    }

    return 0;
}
