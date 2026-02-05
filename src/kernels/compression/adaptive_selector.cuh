/**
 * Adaptive Model Selector for L3 Compression
 *
 * Automatically selects the best model from:
 *   - MODEL_LINEAR (1):      y = θ₀ + θ₁·x
 *   - MODEL_POLYNOMIAL2 (2): y = θ₀ + θ₁·x + θ₂·x²
 *   - MODEL_POLYNOMIAL3 (3): y = θ₀ + θ₁·x + θ₂·x² + θ₃·x³
 *   - MODEL_FOR_BITPACK (4): y = base + delta (base = min)
 *
 * Key Features:
 * 1. Cost-based selection: chooses model with lowest storage cost
 * 2. Polynomial fitting via normal equations
 * 3. Actual residual computation for accurate bit width estimation
 *
 * Cost Formula:
 *   cost_linear = 16 bytes + n × bits / 8
 *   cost_poly2  = 24 bytes + n × bits / 8
 *   cost_poly3  = 32 bytes + n × bits / 8
 *   cost_for    = sizeof(T) + n × bits / 8
 *
 * Date: 2025-12-05
 */

#ifndef ADAPTIVE_SELECTOR_CUH
#define ADAPTIVE_SELECTOR_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <type_traits>
#include "L3_format.hpp"
#include "L3_Vertical_format.hpp"
#include "../utils/finite_diff_shared.cuh"

namespace adaptive_selector {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr double CONSTANT_METADATA_BYTES = 8.0;   // 1 value (the constant)
constexpr double LINEAR_METADATA_BYTES = 16.0;    // 2 doubles
constexpr double POLY2_METADATA_BYTES = 24.0;     // 3 doubles
constexpr double POLY3_METADATA_BYTES = 32.0;     // 4 doubles
constexpr double COST_THRESHOLD = 0.95;           // Must be 5% better to choose complex model

// Maximum integer that can be exactly represented as double (2^53)
// Values larger than this will lose precision when converted to double
constexpr uint64_t DOUBLE_PRECISION_MAX = 9007199254740992ULL;  // 2^53

// ============================================================================
// Decision Result Structure
// ============================================================================

template<typename T>
struct ModelDecision {
    int32_t model_type;      // MODEL_LINEAR, MODEL_POLYNOMIAL2, MODEL_POLYNOMIAL3, or MODEL_FOR_BITPACK
    double params[4];        // θ₀, θ₁, θ₂, θ₃ (for polynomial models)
    T min_val;               // Partition minimum (base for FOR)
    T max_val;               // Partition maximum
    int32_t delta_bits;      // Bits needed for deltas
    double estimated_cost;   // Estimated storage cost in bytes
};

// ============================================================================
// Warp-level Reduction Primitives
// ============================================================================

template<typename T>
__device__ __forceinline__ T warpReduceMin(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other < val) ? other : val;
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// int64_t warp reduce functions for precise residual computation
__device__ __forceinline__ int64_t warpReduceMax_i64(int64_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int64_t other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other > val) ? other : val;
    }
    return val;
}

__device__ __forceinline__ int64_t warpReduceMin_i64(int64_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int64_t other = __shfl_down_sync(0xffffffff, val, offset);
        val = (other < val) ? other : val;
    }
    return val;
}

// ============================================================================
// Block-level Reduction Primitives
// ============================================================================

// Helper to get correct identity for min reduction (works for both signed and unsigned)
template<typename T>
__device__ __forceinline__ T getMinIdentity() {
    if (std::is_unsigned<T>::value) {
        // For unsigned: use max value
        return static_cast<T>(~0ULL);  // All 1s = max unsigned
    } else {
        // For signed: use max positive
        if (sizeof(T) == 4) {
            return static_cast<T>(0x7FFFFFFF);
        } else {
            return static_cast<T>(0x7FFFFFFFFFFFFFFFLL);
        }
    }
}

// Helper to get correct identity for max reduction (works for both signed and unsigned)
template<typename T>
__device__ __forceinline__ T getMaxIdentity() {
    if (std::is_unsigned<T>::value) {
        // For unsigned: use 0 (min value)
        return static_cast<T>(0);
    } else {
        // For signed: use min negative
        if (sizeof(T) == 4) {
            return static_cast<T>(0x80000000);
        } else {
            return static_cast<T>(0x8000000000000000LL);
        }
    }
}

template<typename T>
__device__ T blockReduceMin(T val) {
    __shared__ T shared_min[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMin(val);
    if (lane == 0) shared_min[wid] = val;
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shared_min[lane];
    } else {
        val = getMinIdentity<T>();
    }

    if (wid == 0) val = warpReduceMin(val);
    __syncthreads();  // Ensure all threads see the result
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val) {
    __shared__ T shared_max[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMax(val);
    if (lane == 0) shared_max[wid] = val;
    __syncthreads();

    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shared_max[lane];
    } else {
        val = getMaxIdentity<T>();
    }

    if (wid == 0) val = warpReduceMax(val);
    __syncthreads();  // Ensure all threads see the result
    return val;
}

__device__ inline double blockReduceSum(double val) {
    __shared__ double shared_sum[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared_sum[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared_sum[lane] : 0.0;
    if (wid == 0) val = warpReduceSum(val);
    __syncthreads();  // Ensure all threads see the result
    return val;
}

// ============================================================================
// Bit Width Calculation
// ============================================================================

template<typename T>
__device__ __forceinline__ int computeBitWidth(T range) {
    if (range == 0) return 0;

    if (sizeof(T) == 4) {
        return 32 - __clz(static_cast<unsigned int>(range));
    } else {
        return 64 - __clzll(static_cast<unsigned long long>(range));
    }
}

inline int computeBitWidthCPU(uint64_t range) {
    if (range == 0) return 0;
    return 64 - __builtin_clzll(range);
}

// ============================================================================
// Pre-computed Power Sums (Σxᵏ for x = 0..n-1)
// ============================================================================

// Σx = n(n-1)/2
inline double sumX(double n) {
    return n * (n - 1.0) / 2.0;
}

// Σx² = n(n-1)(2n-1)/6
inline double sumX2(double n) {
    return n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
}

// Σx³ = [n(n-1)/2]²
inline double sumX3(double n) {
    double s = n * (n - 1.0) / 2.0;
    return s * s;
}

// Σx⁴ = n(n-1)(2n-1)(3n²-3n-1)/30
inline double sumX4(double n) {
    return n * (n - 1.0) * (2.0 * n - 1.0) * (3.0 * n * n - 3.0 * n - 1.0) / 30.0;
}

// Σx⁵ = n²(n-1)²(2n²-2n-1)/12
inline double sumX5(double n) {
    double nn = n * n;
    double nm1 = n - 1.0;
    return nn * nm1 * nm1 * (2.0 * nn - 2.0 * n - 1.0) / 12.0;
}

// Σx⁶ = m(m+1)(2m+1)(3m⁴+6m³-3m+1)/42 where m = n-1
inline double sumX6(double n) {
    // Using Faulhaber's formula for Σᵢ₌₁ᵐ i⁶ with m = n-1
    double m = n - 1.0;
    double m2 = m * m;
    double m3 = m2 * m;
    double m4 = m2 * m2;
    return m * (m + 1.0) * (2.0 * m + 1.0) * (3.0 * m4 + 6.0 * m3 - 3.0 * m + 1.0) / 42.0;
}

// ============================================================================
// Gaussian Elimination for Small Systems
// ============================================================================

/**
 * Solve Ax = b using Gaussian elimination with partial pivoting
 * For small systems (2x2, 3x3, 4x4)
 */
inline bool solveLinearSystem(double* A, double* b, double* x, int n) {
    // Forward elimination with partial pivoting
    for (int k = 0; k < n; k++) {
        // Find pivot
        int max_row = k;
        double max_val = std::fabs(A[k * n + k]);
        for (int i = k + 1; i < n; i++) {
            if (std::fabs(A[i * n + k]) > max_val) {
                max_val = std::fabs(A[i * n + k]);
                max_row = i;
            }
        }

        // Swap rows
        if (max_row != k) {
            for (int j = 0; j < n; j++) {
                std::swap(A[k * n + j], A[max_row * n + j]);
            }
            std::swap(b[k], b[max_row]);
        }

        // Check for singular matrix
        if (std::fabs(A[k * n + k]) < 1e-12) {
            return false;
        }

        // Eliminate
        for (int i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
            b[i] -= factor * b[k];
        }
    }

    // Back substitution
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i * n + j] * x[j];
        }
        x[i] /= A[i * n + i];
    }

    return true;
}

// ============================================================================
// Polynomial Fitting Functions
// ============================================================================

/**
 * Fit linear model: y = a₀ + a₁·x
 * Returns coefficients in params[0], params[1]
 */
template<typename T>
inline void fitLinear(
    const T* data, int start, int end,
    double sum_y, double sum_xy,
    double* params)
{
    int n = end - start;
    double dn = static_cast<double>(n);

    double sx = sumX(dn);
    double sx2 = sumX2(dn);

    double denom = dn * sx2 - sx * sx;
    if (std::fabs(denom) < 1e-10) {
        params[0] = sum_y / dn;
        params[1] = 0.0;
    } else {
        params[1] = (dn * sum_xy - sx * sum_y) / denom;
        // CRITICAL FIX: The formula params[0] = (sum_y - params[1] * sx) / dn
        // suffers from catastrophic cancellation when values are large (~10^16+).
        // Instead, anchor theta0 at the first data point: at x=0, pred = theta0 = data[0]
        params[0] = static_cast<double>(data[start]);
    }
    params[2] = 0.0;
    params[3] = 0.0;
}

/**
 * Fit quadratic model: y = a₀ + a₁·x + a₂·x²
 * Requires Σx²y from data scan
 */
template<typename T>
inline void fitQuadratic(
    const T* data, int start, int end,
    double sum_y, double sum_xy, double sum_x2y,
    double* params)
{
    int n = end - start;
    double dn = static_cast<double>(n);

    // Build normal equations matrix
    double sx = sumX(dn);
    double sx2 = sumX2(dn);
    double sx3 = sumX3(dn);
    double sx4 = sumX4(dn);

    // Matrix: [n, sx, sx2; sx, sx2, sx3; sx2, sx3, sx4]
    double A[9] = {
        dn,  sx,  sx2,
        sx,  sx2, sx3,
        sx2, sx3, sx4
    };
    double b[3] = {sum_y, sum_xy, sum_x2y};
    double x[3];

    if (solveLinearSystem(A, b, x, 3)) {
        // CRITICAL FIX: anchor theta0 at first data point to avoid catastrophic cancellation
        params[0] = static_cast<double>(data[start]);
        params[1] = x[1];
        params[2] = x[2];
    } else {
        // Fallback to linear
        fitLinear(data, start, end, sum_y, sum_xy, params);
    }
    params[3] = 0.0;
}

/**
 * Fit cubic model: y = a₀ + a₁·x + a₂·x² + a₃·x³
 * Requires Σx²y, Σx³y from data scan
 *
 * Uses scaled coordinates x' = x/(n-1) ∈ [0,1] to avoid numerical instability
 * in the normal equations matrix. The original approach fails because sumX6
 * can be ~1e22 while n is ~1e3, causing condition numbers ~1e16.
 */
template<typename T>
inline void fitCubic(
    const T* data, int start, int end,
    double sum_y, double sum_xy, double sum_x2y, double sum_x3y,
    double* params)
{
    int n = end - start;
    double dn = static_cast<double>(n);

    if (n <= 20) {
        // Not enough points for stable cubic fit, fallback to quadratic
        fitQuadratic(data, start, end, sum_y, sum_xy, sum_x2y, params);
        return;
    }

    // Scale factor: x' = x / scale, where scale = n-1
    // This maps x from [0, n-1] to [0, 1]
    double scale = dn - 1.0;
    double s2 = scale * scale;
    double s3 = s2 * scale;
    double s4 = s2 * s2;
    double s5 = s4 * scale;
    double s6 = s3 * s3;

    // Scaled power sums: Σx'ᵏ = Σxᵏ / sᵏ
    double sx_scaled = sumX(dn) / scale;
    double sx2_scaled = sumX2(dn) / s2;
    double sx3_scaled = sumX3(dn) / s3;
    double sx4_scaled = sumX4(dn) / s4;
    double sx5_scaled = sumX5(dn) / s5;
    double sx6_scaled = sumX6(dn) / s6;

    // Scaled data sums: Σx'ᵏy = Σxᵏy / sᵏ
    double sum_xpy_scaled = sum_xy / scale;
    double sum_x2py_scaled = sum_x2y / s2;
    double sum_x3py_scaled = sum_x3y / s3;

    // Build normal equations matrix in scaled coordinates
    double A[16] = {
        dn,          sx_scaled,   sx2_scaled,  sx3_scaled,
        sx_scaled,   sx2_scaled,  sx3_scaled,  sx4_scaled,
        sx2_scaled,  sx3_scaled,  sx4_scaled,  sx5_scaled,
        sx3_scaled,  sx4_scaled,  sx5_scaled,  sx6_scaled
    };
    double b[4] = {sum_y, sum_xpy_scaled, sum_x2py_scaled, sum_x3py_scaled};
    double alpha[4];  // Coefficients in scaled coordinates

    if (solveLinearSystem(A, b, alpha, 4)) {
        // Transform back to original coordinates:
        // y = α₀ + α₁(x/s) + α₂(x/s)² + α₃(x/s)³
        //   = α₀ + (α₁/s)x + (α₂/s²)x² + (α₃/s³)x³
        // CRITICAL FIX: anchor theta0 at first data point to avoid catastrophic cancellation
        params[0] = static_cast<double>(data[start]);
        params[1] = alpha[1] / scale;
        params[2] = alpha[2] / s2;
        params[3] = alpha[3] / s3;
    } else {
        // Fallback to quadratic
        fitQuadratic(data, start, end, sum_y, sum_xy, sum_x2y, params);
    }
}

// ============================================================================
// Residual Computation
// ============================================================================

/**
 * Compute residual range for a polynomial model and adjust intercept for unsigned encoding.
 * This enables unsigned delta encoding which saves 1 bit per value compared to signed encoding.
 *
 * @param data Input data
 * @param start Start index
 * @param end End index
 * @param params Model parameters (params[0] will be adjusted)
 * @param model_type Model type
 * @param max_unsigned_delta Output: maximum unsigned delta (range)
 */
template<typename T>
inline void computeResidualRangeUnsigned(
    const T* data, int start, int end,
    double* params, int model_type,
    uint64_t& max_unsigned_delta)
{
    int64_t min_residual = 0;
    int64_t max_residual = 0;

    const int num_values = end - start;
    const int num_mini_vectors = (num_values + MINI_VECTOR_SIZE - 1) / MINI_VECTOR_SIZE;
    constexpr int WARP_SIZE = 32;

    for (int mv = 0; mv < num_mini_vectors; mv++) {
        int mv_base = mv * MINI_VECTOR_SIZE;

        for (int lane = 0; lane < WARP_SIZE; lane++) {
            int start_local_idx = mv_base + lane;

            double pred_y = 0.0, pred_d1 = 0.0, pred_d2 = 0.0, pred_d3 = 0.0;

            switch (model_type) {
                case MODEL_LINEAR:
                    FiniteDiff::computeLinearHost(params, start_local_idx, WARP_SIZE, pred_y, pred_d1);
                    break;
                case MODEL_POLYNOMIAL2:
                    FiniteDiff::computePoly2Host(params, start_local_idx, WARP_SIZE, pred_y, pred_d1, pred_d2);
                    break;
                case MODEL_POLYNOMIAL3:
                    FiniteDiff::computePoly3Host(params, start_local_idx, WARP_SIZE, pred_y, pred_d1, pred_d2, pred_d3);
                    break;
                default:
                    pred_y = params[0];
                    break;
            }

            for (int v = 0; v < VALUES_PER_THREAD; v++) {
                int local_idx = start_local_idx + v * WARP_SIZE;
                if (local_idx >= num_values) break;

                int64_t predicted_int = FiniteDiff::fp64_to_int_host(pred_y);
                T val = data[start + local_idx];
                int64_t residual = static_cast<int64_t>(val) - predicted_int;

                if (residual > max_residual) max_residual = residual;
                if (residual < min_residual) min_residual = residual;

                pred_y += pred_d1;
                pred_d1 += pred_d2;
                pred_d2 += pred_d3;
            }
        }
    }

    // +1 safety margin for potential CPU/GPU FMA differences
    if (max_residual > 0) max_residual += 1;
    if (min_residual < 0) min_residual -= 1;

    // Adjust intercept to make all residuals non-negative (unsigned encoding)
    params[0] += static_cast<double>(min_residual);

    // Max unsigned delta = max_residual - min_residual
    max_unsigned_delta = static_cast<uint64_t>(max_residual - min_residual);
}

/**
 * Compute bits needed for unsigned residual range (no sign bit needed)
 */
inline int computeResidualBitsUnsigned(uint64_t max_unsigned_delta, int max_bits) {
    // Perfect fit: no residuals needed
    if (max_unsigned_delta == 0) {
        return 0;
    }

    int bits = computeBitWidthCPU(max_unsigned_delta);
    return std::min(bits, max_bits);
}

// Signed residual range using finite difference to match GPU prediction path
template<typename T>
inline void computeResidualRange(
    const T* data, int start, int end,
    const double* params, int model_type,
    int64_t& max_positive, int64_t& max_negative)
{
    max_positive = 0;
    max_negative = 0;

    const int num_values = end - start;
    const int num_mini_vectors = (num_values + MINI_VECTOR_SIZE - 1) / MINI_VECTOR_SIZE;
    constexpr int WARP_SIZE = 32;

    for (int mv = 0; mv < num_mini_vectors; mv++) {
        int mv_base = mv * MINI_VECTOR_SIZE;

        for (int lane = 0; lane < WARP_SIZE; lane++) {
            int start_local_idx = mv_base + lane;

            double pred_y = 0.0, pred_d1 = 0.0, pred_d2 = 0.0, pred_d3 = 0.0;

            switch (model_type) {
                case MODEL_LINEAR:
                    FiniteDiff::computeLinearHost(params, start_local_idx, WARP_SIZE, pred_y, pred_d1);
                    break;
                case MODEL_POLYNOMIAL2:
                    FiniteDiff::computePoly2Host(params, start_local_idx, WARP_SIZE, pred_y, pred_d1, pred_d2);
                    break;
                case MODEL_POLYNOMIAL3:
                    FiniteDiff::computePoly3Host(params, start_local_idx, WARP_SIZE, pred_y, pred_d1, pred_d2, pred_d3);
                    break;
                default:
                    pred_y = params[0];
                    break;
            }

            for (int v = 0; v < VALUES_PER_THREAD; v++) {
                int local_idx = start_local_idx + v * WARP_SIZE;
                if (local_idx >= num_values) break;

                int64_t predicted_int = FiniteDiff::fp64_to_int_host(pred_y);
                T val = data[start + local_idx];
                int64_t residual = static_cast<int64_t>(val) - predicted_int;

                if (residual > max_positive) max_positive = residual;
                if (residual < max_negative) max_negative = residual;

                // Finite difference advance - matches GPU exactly
                pred_y += pred_d1;
                pred_d1 += pred_d2;
                pred_d2 += pred_d3;
            }
        }
    }

    // +1 safety margin for potential CPU/GPU FMA differences
    if (max_positive > 0) max_positive += 1;
    if (max_negative < 0) max_negative -= 1;
}

inline int computeResidualBits(int64_t max_positive, int64_t max_negative, int max_bits) {
    // Perfect fit: no residuals needed
    if (max_positive == 0 && max_negative == 0) {
        return 0;
    }

    uint64_t range = static_cast<uint64_t>(max_positive) + static_cast<uint64_t>(-max_negative);

    int bits = computeBitWidthCPU(range) + 1;  // +1 for sign
    return std::min(bits, max_bits);
}

// ============================================================================
// GPU Device Functions for Polynomial Fitting
// ============================================================================

// Pre-computed power sums (GPU version)
__device__ __forceinline__ double sumX_d(double n) {
    return n * (n - 1.0) / 2.0;
}

__device__ __forceinline__ double sumX2_d(double n) {
    return n * (n - 1.0) * (2.0 * n - 1.0) / 6.0;
}

__device__ __forceinline__ double sumX3_d(double n) {
    double s = n * (n - 1.0) / 2.0;
    return s * s;
}

__device__ __forceinline__ double sumX4_d(double n) {
    return n * (n - 1.0) * (2.0 * n - 1.0) * (3.0 * n * n - 3.0 * n - 1.0) / 30.0;
}

__device__ __forceinline__ double sumX5_d(double n) {
    double nn = n * n;
    double nm1 = n - 1.0;
    return nn * nm1 * nm1 * (2.0 * nn - 2.0 * n - 1.0) / 12.0;
}

__device__ __forceinline__ double sumX6_d(double n) {
    // Using Faulhaber's formula for Σᵢ₌₁ᵐ i⁶ with m = n-1
    double m = n - 1.0;
    double m2 = m * m;
    double m3 = m2 * m;
    double m4 = m2 * m2;
    return m * (m + 1.0) * (2.0 * m + 1.0) * (3.0 * m4 + 6.0 * m3 - 3.0 * m + 1.0) / 42.0;
}

// GPU Gaussian elimination for 2x2 system
__device__ __forceinline__ void solveLinear2x2_d(
    double a00, double a01, double a10, double a11,
    double b0, double b1,
    double& x0, double& x1)
{
    double det = a00 * a11 - a01 * a10;
    if (fabs(det) < 1e-12) {
        x0 = b0 / fmax(a00, 1e-12);
        x1 = 0.0;
    } else {
        x0 = (b0 * a11 - b1 * a01) / det;
        x1 = (a00 * b1 - a10 * b0) / det;
    }
}

// GPU Gaussian elimination for 3x3 system - fully register-based
// Solves: [a00 a01 a02][x0]   [b0]
//         [a10 a11 a12][x1] = [b1]
//         [a20 a21 a22][x2]   [b2]
__device__ __forceinline__ void solveLinear3x3_registers(
    double a00, double a01, double a02,
    double a10, double a11, double a12,
    double a20, double a21, double a22,
    double b0, double b1, double b2,
    double& x0, double& x1, double& x2)
{
    // Gaussian elimination with partial pivoting - all in registers

    // Column 0: find pivot
    double abs0 = fabs(a00), abs1 = fabs(a10), abs2 = fabs(a20);
    if (abs1 > abs0 && abs1 >= abs2) {
        // Swap row 0 and row 1
        double t;
        t = a00; a00 = a10; a10 = t;
        t = a01; a01 = a11; a11 = t;
        t = a02; a02 = a12; a12 = t;
        t = b0; b0 = b1; b1 = t;
    } else if (abs2 > abs0 && abs2 > abs1) {
        // Swap row 0 and row 2
        double t;
        t = a00; a00 = a20; a20 = t;
        t = a01; a01 = a21; a21 = t;
        t = a02; a02 = a22; a22 = t;
        t = b0; b0 = b2; b2 = t;
    }

    // Eliminate column 0
    if (fabs(a00) > 1e-12) {
        double f1 = a10 / a00;
        a11 -= f1 * a01;
        a12 -= f1 * a02;
        b1 -= f1 * b0;

        double f2 = a20 / a00;
        a21 -= f2 * a01;
        a22 -= f2 * a02;
        b2 -= f2 * b0;
    }

    // Column 1: find pivot between rows 1 and 2
    if (fabs(a21) > fabs(a11)) {
        double t;
        t = a11; a11 = a21; a21 = t;
        t = a12; a12 = a22; a22 = t;
        t = b1; b1 = b2; b2 = t;
    }

    // Eliminate column 1
    if (fabs(a11) > 1e-12) {
        double f2 = a21 / a11;
        a22 -= f2 * a12;
        b2 -= f2 * b1;
    }

    // Back substitution
    x2 = (fabs(a22) > 1e-12) ? b2 / a22 : 0.0;
    x1 = (fabs(a11) > 1e-12) ? (b1 - a12 * x2) / a11 : 0.0;
    x0 = (fabs(a00) > 1e-12) ? (b0 - a01 * x1 - a02 * x2) / a00 : 0.0;
}

// GPU Gaussian elimination for 4x4 system - fully register-based
// Uses unrolled loops to keep everything in registers
__device__ __forceinline__ void solveLinear4x4_registers(
    double a00, double a01, double a02, double a03,
    double a10, double a11, double a12, double a13,
    double a20, double a21, double a22, double a23,
    double a30, double a31, double a32, double a33,
    double b0, double b1, double b2, double b3,
    double& x0, double& x1, double& x2, double& x3)
{
    // Column 0: find pivot
    double abs0 = fabs(a00), abs1 = fabs(a10), abs2 = fabs(a20), abs3 = fabs(a30);
    int maxrow = 0;
    double maxabs = abs0;
    if (abs1 > maxabs) { maxrow = 1; maxabs = abs1; }
    if (abs2 > maxabs) { maxrow = 2; maxabs = abs2; }
    if (abs3 > maxabs) { maxrow = 3; }

    // Swap row 0 with pivot row
    if (maxrow == 1) {
        double t;
        t = a00; a00 = a10; a10 = t;
        t = a01; a01 = a11; a11 = t;
        t = a02; a02 = a12; a12 = t;
        t = a03; a03 = a13; a13 = t;
        t = b0; b0 = b1; b1 = t;
    } else if (maxrow == 2) {
        double t;
        t = a00; a00 = a20; a20 = t;
        t = a01; a01 = a21; a21 = t;
        t = a02; a02 = a22; a22 = t;
        t = a03; a03 = a23; a23 = t;
        t = b0; b0 = b2; b2 = t;
    } else if (maxrow == 3) {
        double t;
        t = a00; a00 = a30; a30 = t;
        t = a01; a01 = a31; a31 = t;
        t = a02; a02 = a32; a32 = t;
        t = a03; a03 = a33; a33 = t;
        t = b0; b0 = b3; b3 = t;
    }

    // Eliminate column 0
    if (fabs(a00) > 1e-12) {
        double f1 = a10 / a00;
        a11 -= f1 * a01; a12 -= f1 * a02; a13 -= f1 * a03; b1 -= f1 * b0;
        double f2 = a20 / a00;
        a21 -= f2 * a01; a22 -= f2 * a02; a23 -= f2 * a03; b2 -= f2 * b0;
        double f3 = a30 / a00;
        a31 -= f3 * a01; a32 -= f3 * a02; a33 -= f3 * a03; b3 -= f3 * b0;
    }

    // Column 1: find pivot among rows 1-3
    abs1 = fabs(a11); abs2 = fabs(a21); abs3 = fabs(a31);
    maxrow = 1; maxabs = abs1;
    if (abs2 > maxabs) { maxrow = 2; maxabs = abs2; }
    if (abs3 > maxabs) { maxrow = 3; }

    if (maxrow == 2) {
        double t;
        t = a11; a11 = a21; a21 = t;
        t = a12; a12 = a22; a22 = t;
        t = a13; a13 = a23; a23 = t;
        t = b1; b1 = b2; b2 = t;
    } else if (maxrow == 3) {
        double t;
        t = a11; a11 = a31; a31 = t;
        t = a12; a12 = a32; a32 = t;
        t = a13; a13 = a33; a33 = t;
        t = b1; b1 = b3; b3 = t;
    }

    // Eliminate column 1
    if (fabs(a11) > 1e-12) {
        double f2 = a21 / a11;
        a22 -= f2 * a12; a23 -= f2 * a13; b2 -= f2 * b1;
        double f3 = a31 / a11;
        a32 -= f3 * a12; a33 -= f3 * a13; b3 -= f3 * b1;
    }

    // Column 2: find pivot between rows 2-3
    if (fabs(a32) > fabs(a22)) {
        double t;
        t = a22; a22 = a32; a32 = t;
        t = a23; a23 = a33; a33 = t;
        t = b2; b2 = b3; b3 = t;
    }

    // Eliminate column 2
    if (fabs(a22) > 1e-12) {
        double f3 = a32 / a22;
        a33 -= f3 * a23; b3 -= f3 * b2;
    }

    // Back substitution
    x3 = (fabs(a33) > 1e-12) ? b3 / a33 : 0.0;
    x2 = (fabs(a22) > 1e-12) ? (b2 - a23 * x3) / a22 : 0.0;
    x1 = (fabs(a11) > 1e-12) ? (b1 - a12 * x2 - a13 * x3) / a11 : 0.0;
    x0 = (fabs(a00) > 1e-12) ? (b0 - a01 * x1 - a02 * x2 - a03 * x3) / a00 : 0.0;
}

// GPU polynomial fitting functions
__device__ inline void fitLinear_d(int n, double sum_y, double sum_xy, double* params) {
    double dn = static_cast<double>(n);
    double sx = sumX_d(dn);
    double sx2 = sumX2_d(dn);

    solveLinear2x2_d(dn, sx, sx, sx2, sum_y, sum_xy, params[0], params[1]);
    params[2] = 0.0;
    params[3] = 0.0;
}

__device__ __forceinline__ void fitQuadratic_d(int n, double sum_y, double sum_xy, double sum_x2y, double* params) {
    double dn = static_cast<double>(n);
    double sx = sumX_d(dn);
    double sx2 = sumX2_d(dn);
    double sx3 = sumX3_d(dn);
    double sx4 = sumX4_d(dn);

    // Use register-based solver instead of array-based
    double x0, x1, x2;
    solveLinear3x3_registers(
        dn,  sx,  sx2,
        sx,  sx2, sx3,
        sx2, sx3, sx4,
        sum_y, sum_xy, sum_x2y,
        x0, x1, x2
    );
    params[0] = x0;
    params[1] = x1;
    params[2] = x2;
    params[3] = 0.0;
}

/**
 * GPU version of cubic fitting with scaled coordinates.
 * Uses x' = x/(n-1) ∈ [0,1] to avoid numerical instability.
 */
__device__ __forceinline__ void fitCubic_d(int n, double sum_y, double sum_xy, double sum_x2y, double sum_x3y, double* params) {
    double dn = static_cast<double>(n);

    if (n <= 20) {
        // Not enough points, fallback to quadratic
        fitQuadratic_d(n, sum_y, sum_xy, sum_x2y, params);
        return;
    }

    // Scale factor: x' = x / scale
    double scale = dn - 1.0;
    double s2 = scale * scale;
    double s3 = s2 * scale;
    double s4 = s2 * s2;
    double s5 = s4 * scale;
    double s6 = s3 * s3;

    // Scaled power sums
    double sx_scaled = sumX_d(dn) / scale;
    double sx2_scaled = sumX2_d(dn) / s2;
    double sx3_scaled = sumX3_d(dn) / s3;
    double sx4_scaled = sumX4_d(dn) / s4;
    double sx5_scaled = sumX5_d(dn) / s5;
    double sx6_scaled = sumX6_d(dn) / s6;

    // Scaled data sums
    double sum_xpy_scaled = sum_xy / scale;
    double sum_x2py_scaled = sum_x2y / s2;
    double sum_x3py_scaled = sum_x3y / s3;

    // Solve in scaled coordinates
    double alpha0, alpha1, alpha2, alpha3;
    solveLinear4x4_registers(
        dn,          sx_scaled,   sx2_scaled,  sx3_scaled,
        sx_scaled,   sx2_scaled,  sx3_scaled,  sx4_scaled,
        sx2_scaled,  sx3_scaled,  sx4_scaled,  sx5_scaled,
        sx3_scaled,  sx4_scaled,  sx5_scaled,  sx6_scaled,
        sum_y, sum_xpy_scaled, sum_x2py_scaled, sum_x3py_scaled,
        alpha0, alpha1, alpha2, alpha3
    );

    // Transform back to original coordinates
    params[0] = alpha0;
    params[1] = alpha1 / scale;
    params[2] = alpha2 / s2;
    params[3] = alpha3 / s3;
}

// GPU bit width for signed residuals (int64_t version)
__device__ __forceinline__ int computeResidualBits_d(int64_t max_pos, int64_t max_neg, int max_bits) {
    // Perfect fit: no residuals needed
    if (max_pos == 0 && max_neg == 0) {
        return 0;
    }

    uint64_t range = static_cast<uint64_t>(max_pos) + static_cast<uint64_t>(-max_neg);
    int bits = (64 - __clzll(range)) + 1;  // +1 for sign
    return min(bits, max_bits);
}

// GPU bit width for signed residuals (double version - avoids overflow for uint64 > INT64_MAX)
__device__ __forceinline__ int computeResidualBits_d(double max_pos, double max_neg, int max_bits) {
    // Perfect fit: no residuals needed (with small epsilon for floating point)
    if (fabs(max_pos) < 0.5 && fabs(max_neg) < 0.5) {
        return 0;
    }

    // Compute range as double to avoid overflow
    double range = max_pos - max_neg;  // max_pos >= 0, max_neg <= 0, so this is positive
    if (range <= 0.0) {
        return 0;
    }

    // Compute bits needed: ceil(log2(range + 1)) + 1 for sign
    int bits = static_cast<int>(ceil(log2(range + 1.0))) + 1;
    return min(bits, max_bits);
}

// ============================================================================
// Full GPU Kernel with Polynomial Model Selection (LINEAR, POLY2, POLY3, FOR)
// ============================================================================

template<typename T>
__global__ void computeStatsAndDecideFullPolynomial(
    const T* __restrict__ data,
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    ModelDecision<T>* __restrict__ decisions,
    bool enable_rle = true  // NEW: Control RLE/CONSTANT model selection
) {
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = start_indices[pid];
    int end = end_indices[pid];
    int n = end - start;
    const int max_bits = sizeof(T) * 8;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            decisions[pid].model_type = MODEL_FOR_BITPACK;
            decisions[pid].delta_bits = 0;
            decisions[pid].min_val = 0;
            decisions[pid].max_val = 0;
            decisions[pid].params[0] = 0;
            decisions[pid].params[1] = 0;
            decisions[pid].params[2] = 0;
            decisions[pid].params[3] = 0;
            decisions[pid].estimated_cost = 0;
        }
        return;
    }

    // =========================================================================
    // Phase 1: Compute all statistics in parallel
    // =========================================================================
    T local_min = getMinIdentity<T>();
    T local_max = getMaxIdentity<T>();

    double local_sum_y = 0.0;
    double local_sum_xy = 0.0;
    double local_sum_x2y = 0.0;
    double local_sum_x3y = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);
        double x2 = x * x;
        double y = static_cast<double>(val);

        local_min = (val < local_min) ? val : local_min;
        local_max = (val > local_max) ? val : local_max;

        local_sum_y += y;
        local_sum_xy += x * y;
        local_sum_x2y += x2 * y;
        local_sum_x3y += x2 * x * y;
    }

    // Block reduce all statistics with explicit synchronization between calls
    T global_min = blockReduceMin(local_min);
    __syncthreads();  // Ensure blockReduceMin completes before next reduction
    T global_max = blockReduceMax(local_max);
    __syncthreads();  // Ensure blockReduceMax completes before next reduction
    double sum_y = blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xy = blockReduceSum(local_sum_xy);
    __syncthreads();
    double sum_x2y = blockReduceSum(local_sum_x2y);
    __syncthreads();
    double sum_x3y = blockReduceSum(local_sum_x3y);
    __syncthreads();  // Final sync before Phase 2

    // =========================================================================
    // Phase 2: Thread 0 computes all model coefficients
    // =========================================================================
    __shared__ double s_linear_params[4];
    __shared__ double s_poly2_params[4];
    __shared__ double s_poly3_params[4];
    __shared__ T s_global_min;
    __shared__ T s_global_max;
    __shared__ int s_for_bits;

    if (threadIdx.x == 0) {
        s_global_min = global_min;
        s_global_max = global_max;

        // FOR bits
        T range = global_max - global_min;
        s_for_bits = computeBitWidth(range);

        // Fit LINEAR
        fitLinear_d(n, sum_y, sum_xy, s_linear_params);
        // CRITICAL FIX: The standard least-squares formula computes theta0 using
        // Cramer's rule: theta0 = (sum_y * sx2 - sum_xy * sx) / det
        // This suffers from catastrophic cancellation when values are large (~10^16+).
        // Instead, anchor theta0 at the first data point: at x=0, pred = theta0 = data[0]
        s_linear_params[0] = static_cast<double>(data[start]);

        // Fit POLY2 (only if n > 10)
        if (n > 10) {
            fitQuadratic_d(n, sum_y, sum_xy, sum_x2y, s_poly2_params);
            // CRITICAL FIX: anchor theta0 at first data point to avoid catastrophic cancellation
            s_poly2_params[0] = static_cast<double>(data[start]);
        } else {
            for (int i = 0; i < 4; i++) s_poly2_params[i] = s_linear_params[i];
        }

        // Fit POLY3 (only if n > 20)
        if (n > 20) {
            fitCubic_d(n, sum_y, sum_xy, sum_x2y, sum_x3y, s_poly3_params);
            // CRITICAL FIX: anchor theta0 at first data point to avoid catastrophic cancellation
            s_poly3_params[0] = static_cast<double>(data[start]);
        } else {
            for (int i = 0; i < 4; i++) s_poly3_params[i] = s_poly2_params[i];
        }
    }
    __syncthreads();

    // =========================================================================
    // Phase 3: Parallel computation of residual ranges for all models
    // =========================================================================

    // Declare shared memory for reductions - use int64_t for precise residuals
    // Note: For uint64_t data > 2^53, we force FOR model (see force_for_bitpack check),
    // so int64_t is safe here for polynomial residual computation
    __shared__ int64_t s_reduce_max[32];
    __shared__ int64_t s_reduce_min[32];
    __shared__ int64_t s_linear_max_pos_final, s_linear_max_neg_final;
    __shared__ int64_t s_poly2_max_pos_final, s_poly2_max_neg_final;
    __shared__ int64_t s_poly3_max_pos_final, s_poly3_max_neg_final;

    int64_t linear_max_pos = INT64_MIN, linear_max_neg = INT64_MAX;
    int64_t poly2_max_pos = INT64_MIN, poly2_max_neg = INT64_MAX;
    int64_t poly3_max_pos = INT64_MIN, poly3_max_neg = INT64_MAX;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);

        // LINEAR residual: use __double2ll_rn for consistency with encoder/decoder
        double pred_linear_d = s_linear_params[0] + s_linear_params[1] * x;
        int64_t pred_linear = __double2ll_rn(pred_linear_d);
        int64_t res_linear = static_cast<int64_t>(val) - pred_linear;
        linear_max_pos = max(linear_max_pos, res_linear);
        linear_max_neg = min(linear_max_neg, res_linear);

        // POLY2 residual - use __dadd_rn/__dmul_rn for consistency with encoder/decoder
        double x_sq = __dmul_rn(x, x);
        double pred_poly2_d = __dadd_rn(__dadd_rn(s_poly2_params[0], __dmul_rn(s_poly2_params[1], x)), __dmul_rn(s_poly2_params[2], x_sq));
        int64_t pred_poly2 = __double2ll_rn(pred_poly2_d);
        int64_t res_poly2 = static_cast<int64_t>(val) - pred_poly2;
        poly2_max_pos = max(poly2_max_pos, res_poly2);
        poly2_max_neg = min(poly2_max_neg, res_poly2);

        // POLY3 residual - use __dadd_rn/__dmul_rn for consistency with encoder/decoder
        double x_cu = __dmul_rn(x_sq, x);
        double pred_poly3_d = __dadd_rn(__dadd_rn(__dadd_rn(s_poly3_params[0], __dmul_rn(s_poly3_params[1], x)), __dmul_rn(s_poly3_params[2], x_sq)), __dmul_rn(s_poly3_params[3], x_cu));
        int64_t pred_poly3 = __double2ll_rn(pred_poly3_d);
        int64_t res_poly3 = static_cast<int64_t>(val) - pred_poly3;
        poly3_max_pos = max(poly3_max_pos, res_poly3);
        poly3_max_neg = min(poly3_max_neg, res_poly3);
    }

    // Handle case where thread didn't process any data
    if (start + threadIdx.x >= end) {
        linear_max_pos = INT64_MIN; linear_max_neg = INT64_MAX;
        poly2_max_pos = INT64_MIN; poly2_max_neg = INT64_MAX;
        poly3_max_pos = INT64_MIN; poly3_max_neg = INT64_MAX;
    }

    // Block reduce residual ranges with proper synchronization (using int64_t)
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    // Initialize shared memory
    if (threadIdx.x < 32) {
        s_reduce_max[threadIdx.x] = INT64_MIN;
        s_reduce_min[threadIdx.x] = INT64_MAX;
    }
    __syncthreads();

    // Reduce LINEAR max_pos and max_neg together
    linear_max_pos = warpReduceMax_i64(linear_max_pos);
    linear_max_neg = warpReduceMin_i64(linear_max_neg);
    if (lane == 0) {
        s_reduce_max[wid] = linear_max_pos;
        s_reduce_min[wid] = linear_max_neg;
    }
    __syncthreads();

    if (threadIdx.x < num_warps) {
        linear_max_pos = s_reduce_max[threadIdx.x];
        linear_max_neg = s_reduce_min[threadIdx.x];
    } else {
        linear_max_pos = INT64_MIN;
        linear_max_neg = INT64_MAX;
    }
    if (wid == 0) {
        linear_max_pos = warpReduceMax_i64(linear_max_pos);
        linear_max_neg = warpReduceMin_i64(linear_max_neg);
    }
    if (threadIdx.x == 0) {
        s_linear_max_pos_final = linear_max_pos;
        s_linear_max_neg_final = linear_max_neg;
    }
    __syncthreads();

    // Reduce POLY2 max_pos and max_neg together
    poly2_max_pos = warpReduceMax_i64(poly2_max_pos);
    poly2_max_neg = warpReduceMin_i64(poly2_max_neg);
    if (lane == 0) {
        s_reduce_max[wid] = poly2_max_pos;
        s_reduce_min[wid] = poly2_max_neg;
    }
    __syncthreads();

    if (threadIdx.x < num_warps) {
        poly2_max_pos = s_reduce_max[threadIdx.x];
        poly2_max_neg = s_reduce_min[threadIdx.x];
    } else {
        poly2_max_pos = INT64_MIN;
        poly2_max_neg = INT64_MAX;
    }
    if (wid == 0) {
        poly2_max_pos = warpReduceMax_i64(poly2_max_pos);
        poly2_max_neg = warpReduceMin_i64(poly2_max_neg);
    }
    if (threadIdx.x == 0) {
        s_poly2_max_pos_final = poly2_max_pos;
        s_poly2_max_neg_final = poly2_max_neg;
    }
    __syncthreads();

    // Reduce POLY3 max_pos and max_neg together
    poly3_max_pos = warpReduceMax_i64(poly3_max_pos);
    poly3_max_neg = warpReduceMin_i64(poly3_max_neg);
    if (lane == 0) {
        s_reduce_max[wid] = poly3_max_pos;
        s_reduce_min[wid] = poly3_max_neg;
    }
    __syncthreads();

    if (threadIdx.x < num_warps) {
        poly3_max_pos = s_reduce_max[threadIdx.x];
        poly3_max_neg = s_reduce_min[threadIdx.x];
    } else {
        poly3_max_pos = INT64_MIN;
        poly3_max_neg = INT64_MAX;
    }
    if (wid == 0) {
        poly3_max_pos = warpReduceMax_i64(poly3_max_pos);
        poly3_max_neg = warpReduceMin_i64(poly3_max_neg);
    }
    if (threadIdx.x == 0) {
        s_poly3_max_pos_final = poly3_max_pos;
        s_poly3_max_neg_final = poly3_max_neg;
    }
    __syncthreads();

    // Read back final values for thread 0
    linear_max_pos = s_linear_max_pos_final;
    linear_max_neg = s_linear_max_neg_final;
    poly2_max_pos = s_poly2_max_pos_final;
    poly2_max_neg = s_poly2_max_neg_final;
    poly3_max_pos = s_poly3_max_pos_final;
    poly3_max_neg = s_poly3_max_neg_final;

    // =========================================================================
    // Phase 4: Thread 0 compares costs and selects best model
    // =========================================================================
    if (threadIdx.x == 0) {
        double dn = static_cast<double>(n);

        // CRITICAL: For 64-bit unsigned types with values > 2^53, polynomial models
        // cannot be used because double precision is insufficient. Force FOR+BitPack.
        bool force_for_bitpack = false;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (static_cast<uint64_t>(s_global_max) > DOUBLE_PRECISION_MAX) {
                force_for_bitpack = true;
            }
        }

        // FOR cost
        double cost_for = sizeof(T) + dn * s_for_bits / 8.0;

        // LINEAR cost - now using int64_t residuals with llrint(), no safety margin needed
        int linear_bits = computeResidualBits_d(linear_max_pos, linear_max_neg, max_bits);
        double cost_linear = LINEAR_METADATA_BYTES + dn * linear_bits / 8.0;

        // POLY2 cost - add ±1 safety margin for FP64 diff accumulation rounding errors
        // (selector uses direct polynomial eval, encoder/decoder use FP64 finite diff accumulation)
        int poly2_bits = computeResidualBits_d(poly2_max_pos + 1, poly2_max_neg - 1, max_bits);
        double cost_poly2 = POLY2_METADATA_BYTES + dn * poly2_bits / 8.0;

        // POLY3 cost - add ±1 safety margin for FP64 diff accumulation rounding errors
        int poly3_bits = computeResidualBits_d(poly3_max_pos + 1, poly3_max_neg - 1, max_bits);
        double cost_poly3 = POLY3_METADATA_BYTES + dn * poly3_bits / 8.0;

        // Select best model (must be COST_THRESHOLD better to choose more complex model)
        int best_model = MODEL_FOR_BITPACK;
        double best_cost = cost_for;
        int best_bits = s_for_bits;
        // For FOR model: use bit-pattern copy to preserve precision for large integers (>2^53)
        // This stores min_val's exact bit pattern in params[0] as double
        double best_params[4];
        if (sizeof(T) == 8) {
            // 64-bit: use __longlong_as_double to preserve exact bit pattern
            best_params[0] = __longlong_as_double(static_cast<long long>(s_global_min));
        } else {
            // 32-bit: safe to use direct cast
            best_params[0] = static_cast<double>(s_global_min);
        }
        best_params[1] = 0.0;
        best_params[2] = 0.0;
        best_params[3] = 0.0;

        // If force_for_bitpack is set, skip polynomial model comparisons
        if (!force_for_bitpack) {
            // Check CONSTANT first: if all values are the same AND RLE is enabled
            //
            // NOTE: MODEL_CONSTANT uses the unified CONSTANT/RLE parameter layout:
            //   params[0] = num_runs (1 for pure constant)
            //   params[1] = base_value (bit-pattern preserved for 64-bit)
            //   params[2] = value_bits (reserved; 0 for pure constant)
            //   params[3] = count_bits (reserved; 0 for pure constant)
            if (enable_rle && s_global_min == s_global_max) {
                best_model = MODEL_CONSTANT;
                best_cost = CONSTANT_METADATA_BYTES;
                best_bits = 0;  // delta_bits==0 => single-run CONSTANT
                best_params[0] = 1.0;  // num_runs
                if (sizeof(T) == 8) {
                    best_params[1] = __longlong_as_double(static_cast<long long>(s_global_min));
                } else {
                    best_params[1] = static_cast<double>(s_global_min);
                }
                best_params[2] = 0.0;
                best_params[3] = 0.0;
            } else {
                // Check LINEAR
                if (cost_linear < best_cost * COST_THRESHOLD) {
                    best_model = MODEL_LINEAR;
                    best_cost = cost_linear;
                    best_bits = linear_bits;
                    for (int i = 0; i < 4; i++) best_params[i] = s_linear_params[i];
                }

                // Check POLY2 (only if n > 10)
                if (n > 10 && cost_poly2 < best_cost * COST_THRESHOLD) {
                    best_model = MODEL_POLYNOMIAL2;
                    best_cost = cost_poly2;
                    best_bits = poly2_bits;
                    for (int i = 0; i < 4; i++) best_params[i] = s_poly2_params[i];
                }

                // Check POLY3 (only if n > 20)
                if (n > 20 && cost_poly3 < best_cost * COST_THRESHOLD) {
                    best_model = MODEL_POLYNOMIAL3;
                    best_cost = cost_poly3;
                    best_bits = poly3_bits;
                    for (int i = 0; i < 4; i++) best_params[i] = s_poly3_params[i];
                }
            }
        }

        // DEBUG: Print POLY3 residual info for partitions with potential issues
        if (best_model == MODEL_POLYNOMIAL3 && best_bits == 0) {
            printf("[DEBUG pid=%d] POLY3 selected with bits=0! poly3_max_pos=%lld, poly3_max_neg=%lld, "
                   "poly3_bits=%d, +1/-1 adjusted: max_pos=%lld, max_neg=%lld\n",
                   pid, (long long)poly3_max_pos, (long long)poly3_max_neg, poly3_bits,
                   (long long)(poly3_max_pos + 1), (long long)(poly3_max_neg - 1));
        }

        // Write decision
        ModelDecision<T> decision;
        decision.model_type = best_model;
        decision.min_val = s_global_min;
        decision.max_val = s_global_max;
        decision.delta_bits = best_bits;
        decision.estimated_cost = best_cost;
        for (int i = 0; i < 4; i++) decision.params[i] = best_params[i];

        decisions[pid] = decision;
    }
}

// Launch wrapper for full polynomial selector
template<typename T>
void launchAdaptiveSelectorFullPolynomial(
    const T* d_data,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    int num_partitions,
    ModelDecision<T>* d_decisions,
    int block_size = 256,
    cudaStream_t stream = 0,
    bool enable_rle = true  // NEW: Control RLE/CONSTANT model selection
) {
    if (num_partitions == 0) return;

    computeStatsAndDecideFullPolynomial<T><<<num_partitions, block_size, 0, stream>>>(
        d_data,
        d_start_indices,
        d_end_indices,
        num_partitions,
        d_decisions,
        enable_rle
    );
}

// ============================================================================
// Main GPU Kernel (Simplified - LINEAR and FOR only for performance)
// ============================================================================

template<typename T>
__global__ void computeStatsAndDecide(
    const T* __restrict__ data,
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    ModelDecision<T>* __restrict__ decisions
) {
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = start_indices[pid];
    int end = end_indices[pid];
    int n = end - start;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            decisions[pid].model_type = MODEL_FOR_BITPACK;
            decisions[pid].delta_bits = 0;
            decisions[pid].min_val = 0;
            decisions[pid].max_val = 0;
            decisions[pid].params[0] = 0;
            decisions[pid].params[1] = 0;
            decisions[pid].params[2] = 0;
            decisions[pid].params[3] = 0;
            decisions[pid].estimated_cost = 0;
        }
        return;
    }

    // Initialize with extreme values
    T local_min = getMinIdentity<T>();
    T local_max = getMaxIdentity<T>();

    double local_sum_y = 0.0;
    double local_sum_xy = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        int local_idx = i - start;

        local_min = (val < local_min) ? val : local_min;
        local_max = (val > local_max) ? val : local_max;

        double y = static_cast<double>(val);
        local_sum_y += y;
        local_sum_xy += static_cast<double>(local_idx) * y;
    }

    // Block reduce all statistics with explicit synchronization between calls
    T global_min = blockReduceMin(local_min);
    __syncthreads();  // Ensure blockReduceMin completes before next reduction
    T global_max = blockReduceMax(local_max);
    __syncthreads();  // Ensure blockReduceMax completes before next reduction
    double sum_y = blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xy = blockReduceSum(local_sum_xy);
    __syncthreads();  // Final sync before thread 0 uses results

    if (threadIdx.x == 0) {
        T range = global_max - global_min;

        // FOR cost
        int for_bits = computeBitWidth(range);
        double cost_for = sizeof(T) + static_cast<double>(n) * for_bits / 8.0;

        // LINEAR cost (estimated)
        double dn = static_cast<double>(n);
        double sx = dn * (dn - 1.0) / 2.0;
        double sx2 = dn * (dn - 1.0) * (2.0 * dn - 1.0) / 6.0;

        double denom = dn * sx2 - sx * sx;
        double slope = 0.0;
        double intercept = sum_y / dn;

        if (fabs(denom) > 1e-10) {
            slope = (dn * sum_xy - sx * sum_y) / denom;
            intercept = (sum_y - slope * sx) / dn;
        }

        double trend_span = fabs(slope) * (dn - 1.0);
        double est_residual_range = (trend_span >= static_cast<double>(range) * 0.9) ?
            static_cast<double>(range) * 0.1 :
            static_cast<double>(range) - trend_span;
        est_residual_range = fmax(est_residual_range, 1.0);

        int linear_bits;
        if (sizeof(T) == 4) {
            linear_bits = computeBitWidth(static_cast<uint32_t>(est_residual_range)) + 1;
        } else {
            linear_bits = computeBitWidth(static_cast<uint64_t>(est_residual_range)) + 1;
        }

        double cost_linear = LINEAR_METADATA_BYTES + static_cast<double>(n) * linear_bits / 8.0;

        ModelDecision<T> decision;
        decision.min_val = global_min;
        decision.max_val = global_max;

        // Check CONSTANT first: if all values are the same
        if (global_min == global_max) {
            decision.model_type = MODEL_CONSTANT;
            decision.params[0] = 1.0;  // num_runs
            if (sizeof(T) == 8) {
                decision.params[1] = __longlong_as_double(static_cast<long long>(global_min));
            } else {
                decision.params[1] = static_cast<double>(global_min);
            }
            decision.params[2] = 0.0;
            decision.params[3] = 0.0;
            decision.delta_bits = 0;
            decision.estimated_cost = CONSTANT_METADATA_BYTES;
        } else if (cost_linear < cost_for * COST_THRESHOLD) {
            decision.model_type = MODEL_LINEAR;
            decision.params[0] = intercept;
            decision.params[1] = slope;
            decision.params[2] = 0.0;
            decision.params[3] = 0.0;
            decision.delta_bits = linear_bits;
            decision.estimated_cost = cost_linear;
        } else {
            decision.model_type = MODEL_FOR_BITPACK;
            decision.params[0] = static_cast<double>(global_min);
            decision.params[1] = 0.0;
            decision.params[2] = 0.0;
            decision.params[3] = 0.0;
            decision.delta_bits = for_bits;
            decision.estimated_cost = cost_for;
        }

        decisions[pid] = decision;
    }
}

// ============================================================================
// Host Helper Functions
// ============================================================================

template<typename T>
void launchAdaptiveSelector(
    const T* d_data,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    int num_partitions,
    ModelDecision<T>* d_decisions,
    int block_size = 256,
    cudaStream_t stream = 0
) {
    if (num_partitions == 0) return;

    computeStatsAndDecide<T><<<num_partitions, block_size, 0, stream>>>(
        d_data,
        d_start_indices,
        d_end_indices,
        num_partitions,
        d_decisions
    );
}

// ============================================================================
// CPU Version with Full Polynomial Support
// ============================================================================

/**
 * CPU version with LINEAR, POLYNOMIAL2, POLYNOMIAL3, and FOR_BITPACK
 * Uses actual residual computation for accurate bit width estimation
 */
template<typename T>
ModelDecision<T> computeDecisionCPU(
    const T* data,
    int start,
    int end
) {
    int n = end - start;
    ModelDecision<T> decision = {};

    if (n <= 0) {
        decision.model_type = MODEL_FOR_BITPACK;
        return decision;
    }

    const int max_bits = sizeof(T) * 8;

    // =========================================================================
    // Phase 1: Compute all statistics in a single pass
    // =========================================================================
    T min_val = data[start];
    T max_val = data[start];
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2y = 0.0;
    double sum_x3y = 0.0;

    for (int i = start; i < end; i++) {
        T val = data[i];
        int local_idx = i - start;

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;

        double x = static_cast<double>(local_idx);
        double y = static_cast<double>(val);

        sum_y += y;
        sum_xy += x * y;
        sum_x2y += x * x * y;
        sum_x3y += x * x * x * y;
    }

    T range = max_val - min_val;
    decision.min_val = min_val;
    decision.max_val = max_val;

    // =========================================================================
    // Phase 2: Compute FOR+BitPacking cost (baseline)
    // =========================================================================
    int for_bits = 0;
    if (range > 0) {
        for_bits = computeBitWidthCPU(static_cast<uint64_t>(range));
    }
    double cost_for = sizeof(T) + static_cast<double>(n) * for_bits / 8.0;

    // Initialize best model as FOR
    int best_model = MODEL_FOR_BITPACK;
    double best_cost = cost_for;
    int best_bits = for_bits;
    double best_params[4] = {0.0, 0.0, 0.0, 0.0};

    // For FOR model: use bit-pattern copy to preserve precision for large integers (>2^53)
    if (sizeof(T) == 8) {
        // 64-bit: use memcpy to preserve exact bit pattern
        uint64_t min_bits = static_cast<uint64_t>(min_val);
        std::memcpy(&best_params[0], &min_bits, sizeof(double));
    } else {
        // 32-bit: safe to use direct cast
        best_params[0] = static_cast<double>(min_val);
    }

    // =========================================================================
    // Phase 2.5: Check CONSTANT model (if all values are the same)
    // =========================================================================
    if (min_val == max_val) {
        // All values are identical, use CONSTANT model (single-run).
        //
        // MODEL_CONSTANT uses the unified CONSTANT/RLE parameter layout:
        //   params[0] = num_runs (1 for pure constant)
        //   params[1] = base_value (bit-pattern preserved for 64-bit)
        //   params[2] = value_bits (reserved; 0 for pure constant)
        //   params[3] = count_bits (reserved; 0 for pure constant)
        best_model = MODEL_CONSTANT;
        best_cost = CONSTANT_METADATA_BYTES;
        best_bits = 0;
        best_params[0] = 1.0;  // num_runs
        if (sizeof(T) == 8) {
            // Preserve exact integer bit pattern for 64-bit values (>2^53 safe)
            uint64_t base_bits = static_cast<uint64_t>(min_val);
            std::memcpy(&best_params[1], &base_bits, sizeof(double));
        } else {
            best_params[1] = static_cast<double>(min_val);
        }
        best_params[2] = 0.0;
        best_params[3] = 0.0;

        // Set decision and return early
        decision.model_type = best_model;
        decision.delta_bits = best_bits;
        decision.estimated_cost = best_cost;
        std::copy(best_params, best_params + 4, decision.params);
        return decision;
    }

    // =========================================================================
    // Phase 3: Try LINEAR model
    // =========================================================================
    {
        double params[4];
        fitLinear(data, start, end, sum_y, sum_xy, params);

        int64_t max_pos, max_neg;
        computeResidualRange(data, start, end, params, MODEL_LINEAR, max_pos, max_neg);

        int bits = computeResidualBits(max_pos, max_neg, max_bits);
        double cost = LINEAR_METADATA_BYTES + static_cast<double>(n) * bits / 8.0;

        if (cost < best_cost * COST_THRESHOLD) {
            best_model = MODEL_LINEAR;
            best_cost = cost;
            best_bits = bits;
            std::copy(params, params + 4, best_params);
        }
    }

    // =========================================================================
    // Phase 4: Try POLYNOMIAL2 model (only if n > 10 for stability)
    // =========================================================================
    if (n > 10) {
        double params[4];
        fitQuadratic(data, start, end, sum_y, sum_xy, sum_x2y, params);

        int64_t max_pos, max_neg;
        computeResidualRange(data, start, end, params, MODEL_POLYNOMIAL2, max_pos, max_neg);

        int bits = computeResidualBits(max_pos, max_neg, max_bits);
        double cost = POLY2_METADATA_BYTES + static_cast<double>(n) * bits / 8.0;

        // Must be significantly better than current best
        if (cost < best_cost * COST_THRESHOLD) {
            best_model = MODEL_POLYNOMIAL2;
            best_cost = cost;
            best_bits = bits;
            std::copy(params, params + 4, best_params);
        }
    }

    // =========================================================================
    // Phase 5: Try POLYNOMIAL3 model (only if n > 20 for stability)
    // =========================================================================
    if (n > 20) {
        double params[4];
        fitCubic(data, start, end, sum_y, sum_xy, sum_x2y, sum_x3y, params);

        int64_t max_pos, max_neg;
        computeResidualRange(data, start, end, params, MODEL_POLYNOMIAL3, max_pos, max_neg);

        int bits = computeResidualBits(max_pos, max_neg, max_bits);
        double cost = POLY3_METADATA_BYTES + static_cast<double>(n) * bits / 8.0;

        // Must be significantly better than current best
        if (cost < best_cost * COST_THRESHOLD) {
            best_model = MODEL_POLYNOMIAL3;
            best_cost = cost;
            best_bits = bits;
            std::copy(params, params + 4, best_params);
        }
    }

    // =========================================================================
    // Phase 6: Set decision result
    // =========================================================================
    decision.model_type = best_model;
    decision.delta_bits = best_bits;
    decision.estimated_cost = best_cost;
    std::copy(best_params, best_params + 4, decision.params);

    return decision;
}

// ============================================================================
// Model Name Helper
// ============================================================================

inline const char* modelTypeName(int model_type) {
    switch (model_type) {
        case MODEL_CONSTANT: return "CONSTANT";
        case MODEL_LINEAR: return "LINEAR";
        case MODEL_POLYNOMIAL2: return "POLY2";
        case MODEL_POLYNOMIAL3: return "POLY3";
        case MODEL_FOR_BITPACK: return "FOR+BP";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Fixed Model Selection (Force specific model for all partitions)
// ============================================================================

/**
 * CPU version - compute metadata using a fixed model type
 *
 * @param data Input data
 * @param start Start index
 * @param end End index
 * @param fixed_model_type Model type to use (MODEL_LINEAR, MODEL_POLYNOMIAL2, etc.)
 */
template<typename T>
ModelDecision<T> computeFixedModelCPU(
    const T* data,
    int start,
    int end,
    int fixed_model_type
) {
    int n = end - start;
    ModelDecision<T> decision = {};

    if (n <= 0) {
        decision.model_type = fixed_model_type;
        return decision;
    }

    const int max_bits = sizeof(T) * 8;

    // Compute statistics
    T min_val = data[start];
    T max_val = data[start];
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_x2y = 0.0;
    double sum_x3y = 0.0;

    for (int i = start; i < end; i++) {
        T val = data[i];
        int local_idx = i - start;

        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;

        double x = static_cast<double>(local_idx);
        double y = static_cast<double>(val);

        sum_y += y;
        sum_xy += x * y;
        sum_x2y += x * x * y;
        sum_x3y += x * x * x * y;
    }

    decision.min_val = min_val;
    decision.max_val = max_val;
    decision.model_type = fixed_model_type;

    double params[4] = {0.0, 0.0, 0.0, 0.0};
    int64_t max_pos = 0, max_neg = 0;

    switch (fixed_model_type) {
        case MODEL_LINEAR:
            fitLinear(data, start, end, sum_y, sum_xy, params);
            computeResidualRange(data, start, end, params, MODEL_LINEAR, max_pos, max_neg);
            decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
            break;

        case MODEL_POLYNOMIAL2:
            if (n > 10) {
                fitQuadratic(data, start, end, sum_y, sum_xy, sum_x2y, params);
                computeResidualRange(data, start, end, params, MODEL_POLYNOMIAL2, max_pos, max_neg);
                decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
            } else {
                // Fallback to linear for small partitions
                fitLinear(data, start, end, sum_y, sum_xy, params);
                computeResidualRange(data, start, end, params, MODEL_LINEAR, max_pos, max_neg);
                decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
                decision.model_type = MODEL_LINEAR;
            }
            break;

        case MODEL_POLYNOMIAL3:
            if (n > 20) {
                fitCubic(data, start, end, sum_y, sum_xy, sum_x2y, sum_x3y, params);
                computeResidualRange(data, start, end, params, MODEL_POLYNOMIAL3, max_pos, max_neg);
                decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
            } else if (n > 10) {
                // Fallback to poly2 for medium partitions
                fitQuadratic(data, start, end, sum_y, sum_xy, sum_x2y, params);
                computeResidualRange(data, start, end, params, MODEL_POLYNOMIAL2, max_pos, max_neg);
                decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
                decision.model_type = MODEL_POLYNOMIAL2;
            } else {
                // Fallback to linear for small partitions
                fitLinear(data, start, end, sum_y, sum_xy, params);
                computeResidualRange(data, start, end, params, MODEL_LINEAR, max_pos, max_neg);
                decision.delta_bits = computeResidualBits(max_pos, max_neg, max_bits);
                decision.model_type = MODEL_LINEAR;
            }
            break;

        case MODEL_FOR_BITPACK:
        default: {
            T range = max_val - min_val;
            decision.delta_bits = (range > 0) ? computeBitWidthCPU(static_cast<uint64_t>(range)) : 0;
            // Store min_val as base, preserving bit pattern for 64-bit types
            if (sizeof(T) == 8) {
                uint64_t min_bits = static_cast<uint64_t>(min_val);
                std::memcpy(&params[0], &min_bits, sizeof(double));
            } else {
                params[0] = static_cast<double>(min_val);
            }
            decision.model_type = MODEL_FOR_BITPACK;
            break;
        }
    }

    std::copy(params, params + 4, decision.params);
    return decision;
}

/**
 * GPU Kernel - compute metadata using a fixed model type
 */
template<typename T>
__global__ void computeFixedModelKernel(
    const T* __restrict__ data,
    const int32_t* __restrict__ start_indices,
    const int32_t* __restrict__ end_indices,
    int num_partitions,
    int fixed_model_type,
    ModelDecision<T>* __restrict__ decisions
) {
    int pid = blockIdx.x;
    if (pid >= num_partitions) return;

    int start = start_indices[pid];
    int end = end_indices[pid];
    int n = end - start;
    const int max_bits = sizeof(T) * 8;

    if (n <= 0) {
        if (threadIdx.x == 0) {
            decisions[pid].model_type = fixed_model_type;
            decisions[pid].delta_bits = 0;
            decisions[pid].min_val = 0;
            decisions[pid].max_val = 0;
            for (int i = 0; i < 4; i++) decisions[pid].params[i] = 0;
        }
        return;
    }

    // Phase 1: Compute statistics
    T local_min = getMinIdentity<T>();
    T local_max = getMaxIdentity<T>();
    double local_sum_y = 0.0;
    double local_sum_xy = 0.0;
    double local_sum_x2y = 0.0;
    double local_sum_x3y = 0.0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        T val = data[i];
        int local_idx = i - start;
        double x = static_cast<double>(local_idx);
        double y = static_cast<double>(val);

        local_min = (val < local_min) ? val : local_min;
        local_max = (val > local_max) ? val : local_max;

        local_sum_y += y;
        local_sum_xy += x * y;
        local_sum_x2y += x * x * y;
        local_sum_x3y += x * x * x * y;
    }

    // Block reduce all statistics with explicit synchronization between calls
    T global_min = blockReduceMin(local_min);
    __syncthreads();  // Ensure blockReduceMin completes before next reduction
    T global_max = blockReduceMax(local_max);
    __syncthreads();  // Ensure blockReduceMax completes before next reduction
    double sum_y = blockReduceSum(local_sum_y);
    __syncthreads();
    double sum_xy = blockReduceSum(local_sum_xy);
    __syncthreads();
    double sum_x2y = blockReduceSum(local_sum_x2y);
    __syncthreads();
    double sum_x3y = blockReduceSum(local_sum_x3y);
    __syncthreads();  // Final sync before Phase 2

    // Phase 2: Thread 0 fits the specified model
    __shared__ double s_params[4];
    __shared__ T s_global_min;
    __shared__ T s_global_max;
    __shared__ int s_actual_model;

    if (threadIdx.x == 0) {
        s_global_min = global_min;
        s_global_max = global_max;

        // CRITICAL: For 64-bit unsigned types with values > 2^53, polynomial models
        // cannot be used because double precision is insufficient. Force FOR+BitPack.
        int effective_model_type = fixed_model_type;
        if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
            if (static_cast<uint64_t>(global_max) > DOUBLE_PRECISION_MAX) {
                effective_model_type = MODEL_FOR_BITPACK;
            }
        }
        s_actual_model = effective_model_type;

        switch (effective_model_type) {
            case MODEL_LINEAR:
                fitLinear_d(n, sum_y, sum_xy, s_params);
                // CRITICAL FIX: anchor theta0 at first data point to avoid catastrophic cancellation
                s_params[0] = static_cast<double>(data[start]);
                break;
            case MODEL_POLYNOMIAL2:
                if (n > 10) {
                    fitQuadratic_d(n, sum_y, sum_xy, sum_x2y, s_params);
                    // CRITICAL FIX: anchor theta0 at first data point
                    s_params[0] = static_cast<double>(data[start]);
                } else {
                    fitLinear_d(n, sum_y, sum_xy, s_params);
                    s_params[0] = static_cast<double>(data[start]);
                    s_actual_model = MODEL_LINEAR;
                }
                break;
            case MODEL_POLYNOMIAL3:
                if (n > 20) {
                    fitCubic_d(n, sum_y, sum_xy, sum_x2y, sum_x3y, s_params);
                    // CRITICAL FIX: anchor theta0 at first data point
                    s_params[0] = static_cast<double>(data[start]);
                } else if (n > 10) {
                    fitQuadratic_d(n, sum_y, sum_xy, sum_x2y, s_params);
                    s_params[0] = static_cast<double>(data[start]);
                    s_actual_model = MODEL_POLYNOMIAL2;
                } else {
                    fitLinear_d(n, sum_y, sum_xy, s_params);
                    s_params[0] = static_cast<double>(data[start]);
                    s_actual_model = MODEL_LINEAR;
                }
                break;
            case MODEL_FOR_BITPACK:
            default:
                if (sizeof(T) == 8) {
                    s_params[0] = __longlong_as_double(static_cast<long long>(global_min));
                } else {
                    s_params[0] = static_cast<double>(global_min);
                }
                s_params[1] = 0.0;
                s_params[2] = 0.0;
                s_params[3] = 0.0;
                s_actual_model = MODEL_FOR_BITPACK;
                break;
        }
    }
    __syncthreads();

    // Phase 3: Compute residuals for the chosen model
    int actual_model = s_actual_model;

    if (actual_model == MODEL_FOR_BITPACK) {
        // FOR model: compute range-based bits
        if (threadIdx.x == 0) {
            T range = s_global_max - s_global_min;
            int for_bits = computeBitWidth(range);

            ModelDecision<T> decision;
            decision.model_type = MODEL_FOR_BITPACK;
            decision.min_val = s_global_min;
            decision.max_val = s_global_max;
            decision.delta_bits = for_bits;
            for (int i = 0; i < 4; i++) decision.params[i] = s_params[i];
            decisions[pid] = decision;
        }
    } else {
        // Polynomial models: compute actual residuals using int64_t for precision
        int64_t local_max_pos = INT64_MIN;
        int64_t local_max_neg = INT64_MAX;

        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            T val = data[i];
            int local_idx = i - start;
            double x = static_cast<double>(local_idx);

            double predicted_d;
            switch (actual_model) {
                case MODEL_LINEAR:
                    predicted_d = s_params[0] + s_params[1] * x;
                    break;
                case MODEL_POLYNOMIAL2:
                    predicted_d = s_params[0] + x * (s_params[1] + x * s_params[2]);
                    break;
                case MODEL_POLYNOMIAL3:
                    predicted_d = s_params[0] + x * (s_params[1] + x * (s_params[2] + x * s_params[3]));
                    break;
                default:
                    predicted_d = s_params[0];
                    break;
            }

            int64_t predicted = __double2ll_rn(predicted_d);
            int64_t residual = static_cast<int64_t>(val) - predicted;
            local_max_pos = max(local_max_pos, residual);
            local_max_neg = min(local_max_neg, residual);
        }

        // Handle threads that didn't process any data
        if (start + threadIdx.x >= end) {
            local_max_pos = INT64_MIN;
            local_max_neg = INT64_MAX;
        }

        // Reduce residual ranges with proper synchronization using int64_t
        __shared__ int64_t s_reduce_i64[32];
        __shared__ int64_t s_max_pos_final_i64, s_max_neg_final_i64;
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;
        int num_warps = (blockDim.x + 31) >> 5;

        // Reduce max_pos
        local_max_pos = warpReduceMax_i64(local_max_pos);
        if (lane == 0) s_reduce_i64[wid] = local_max_pos;
        __syncthreads();
        if (threadIdx.x < num_warps) local_max_pos = s_reduce_i64[threadIdx.x];
        else local_max_pos = INT64_MIN;
        if (wid == 0) {
            local_max_pos = warpReduceMax_i64(local_max_pos);
        }
        if (threadIdx.x == 0) s_max_pos_final_i64 = local_max_pos;
        __syncthreads();

        // Reduce max_neg (min)
        local_max_neg = warpReduceMin_i64(local_max_neg);
        if (lane == 0) s_reduce_i64[wid] = local_max_neg;
        __syncthreads();
        if (threadIdx.x < num_warps) local_max_neg = s_reduce_i64[threadIdx.x];
        else local_max_neg = INT64_MAX;
        if (wid == 0) {
            local_max_neg = warpReduceMin_i64(local_max_neg);
        }
        if (threadIdx.x == 0) s_max_neg_final_i64 = local_max_neg;
        __syncthreads();

        // Thread 0 writes final decision
        if (threadIdx.x == 0) {
            int bits = computeResidualBits_d(s_max_pos_final_i64, s_max_neg_final_i64, max_bits);

            ModelDecision<T> decision;
            decision.model_type = actual_model;
            decision.min_val = s_global_min;
            decision.max_val = s_global_max;
            decision.delta_bits = bits;
            for (int i = 0; i < 4; i++) decision.params[i] = s_params[i];
            decisions[pid] = decision;
        }
    }
}

/**
 * Launch wrapper for fixed model selector (GPU)
 */
template<typename T>
void launchFixedModelSelector(
    const T* d_data,
    const int32_t* d_start_indices,
    const int32_t* d_end_indices,
    int num_partitions,
    int fixed_model_type,
    ModelDecision<T>* d_decisions,
    int block_size = 256,
    cudaStream_t stream = 0
) {
    if (num_partitions == 0) return;

    computeFixedModelKernel<T><<<num_partitions, block_size, 0, stream>>>(
        d_data,
        d_start_indices,
        d_end_indices,
        num_partitions,
        fixed_model_type,
        d_decisions
    );
}

}  // namespace adaptive_selector

#endif // ADAPTIVE_SELECTOR_CUH
