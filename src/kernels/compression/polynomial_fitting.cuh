/**
 * Polynomial Fitting Utilities for L3 Compression
 *
 * Implements numerically stable polynomial regression using:
 * 1. Data centering: x' = 2(x - x_center) / (n-1) ∈ [-1, 1]
 * 2. Orthogonal polynomial basis (Legendre) for fitting
 * 3. Coefficient transformation back to original coordinates
 *
 * This approach avoids numerical overflow when computing Σx^k for large partitions.
 *
 * Date: 2025-12-05
 */

#ifndef POLYNOMIAL_FITTING_CUH
#define POLYNOMIAL_FITTING_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

namespace polynomial_fitting {

// ============================================================================
// Constants
// ============================================================================

constexpr int MAX_POLYNOMIAL_DEGREE = 3;  // Up to cubic

// ============================================================================
// Legendre Polynomial Evaluation
// ============================================================================

/**
 * Evaluate Legendre polynomial P_k(x) for x ∈ [-1, 1]
 *
 * P_0(x) = 1
 * P_1(x) = x
 * P_2(x) = (3x² - 1) / 2
 * P_3(x) = (5x³ - 3x) / 2
 */
__device__ __forceinline__
double legendreP(int k, double x) {
    switch (k) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return (3.0 * x * x - 1.0) * 0.5;
        case 3: return (5.0 * x * x * x - 3.0 * x) * 0.5;
        default: return 0.0;
    }
}

/**
 * Evaluate all Legendre polynomials up to degree d
 */
__device__ __forceinline__
void legendreAll(double x, int degree, double* P) {
    P[0] = 1.0;
    if (degree >= 1) P[1] = x;
    if (degree >= 2) P[2] = (3.0 * x * x - 1.0) * 0.5;
    if (degree >= 3) P[3] = (5.0 * x * x * x - 3.0 * x) * 0.5;
}

// ============================================================================
// Coordinate Transformation
// ============================================================================

/**
 * Transform from original index to normalized coordinate
 * x' = 2 * (x - center) / (n - 1) ∈ [-1, 1]
 *
 * For x ∈ [0, n-1], center = (n-1)/2
 */
__device__ __forceinline__
double toNormalized(int local_idx, int n) {
    if (n <= 1) return 0.0;
    double center = (n - 1) * 0.5;
    double scale = 2.0 / (n - 1);
    return (local_idx - center) * scale;
}

/**
 * Transform Legendre coefficients to standard polynomial coefficients
 *
 * Given: y = c0*P0(x') + c1*P1(x') + c2*P2(x') + c3*P3(x')
 * Where: x' = scale * (x - center), scale = 2/(n-1), center = (n-1)/2
 *
 * Compute: y = a0 + a1*x + a2*x² + a3*x³
 *
 * Derivation:
 *   P0(x') = 1
 *   P1(x') = x' = s(x - c)
 *   P2(x') = (3x'² - 1)/2
 *   P3(x') = (5x'³ - 3x')/2
 *
 * Substituting x' = s*x - s*c = s*x + d where d = -s*c = -1:
 *
 * For degree 1: y = c0 + c1*(s*x + d)
 *               a0 = c0 + c1*d = c0 - c1
 *               a1 = c1*s
 *
 * For degree 2: P2 = (3*(s*x+d)² - 1)/2
 *               = (3*s²*x² + 6*s*d*x + 3*d² - 1)/2
 *               = 1.5*s²*x² + 3*s*d*x + (3*d² - 1)/2
 *               Since d = -1: (3*1 - 1)/2 = 1
 *
 * For degree 3: P3 = (5*(s*x+d)³ - 3*(s*x+d))/2
 */
__device__ __forceinline__
void legendreToStandard(const double* c, int degree, int n, double* a) {
    if (n <= 1) {
        a[0] = c[0];
        for (int i = 1; i <= MAX_POLYNOMIAL_DEGREE; i++) a[i] = 0.0;
        return;
    }

    double s = 2.0 / (n - 1);      // scale
    double d = -1.0;               // offset = -s * center = -1
    double s2 = s * s;
    double s3 = s2 * s;
    double d2 = d * d;             // = 1
    double d3 = d * d * d;         // = -1

    // Initialize
    a[0] = a[1] = a[2] = a[3] = 0.0;

    // c0 * P0 = c0 * 1
    a[0] += c[0];

    if (degree >= 1) {
        // c1 * P1 = c1 * (s*x + d)
        a[0] += c[1] * d;
        a[1] += c[1] * s;
    }

    if (degree >= 2) {
        // c2 * P2 = c2 * (3*(s*x+d)² - 1)/2
        // = c2 * (3*s²*x² + 6*s*d*x + 3*d² - 1) / 2
        // = c2 * (1.5*s²*x² + 3*s*d*x + (3*d² - 1)/2)
        double p2_const = (3.0 * d2 - 1.0) * 0.5;  // = 1
        double p2_x1 = 3.0 * s * d;                 // = -3s
        double p2_x2 = 1.5 * s2;

        a[0] += c[2] * p2_const;
        a[1] += c[2] * p2_x1;
        a[2] += c[2] * p2_x2;
    }

    if (degree >= 3) {
        // c3 * P3 = c3 * (5*(s*x+d)³ - 3*(s*x+d)) / 2
        // (s*x+d)³ = s³x³ + 3s²dx² + 3sd²x + d³
        // = c3 * (5*s³*x³ + 15*s²*d*x² + 15*s*d²*x + 5*d³ - 3*s*x - 3*d) / 2
        double p3_const = (5.0 * d3 - 3.0 * d) * 0.5;     // = (-5 + 3)/2 = -1
        double p3_x1 = (15.0 * s * d2 - 3.0 * s) * 0.5;   // = (15s - 3s)/2 = 6s
        double p3_x2 = (15.0 * s2 * d) * 0.5;             // = -7.5s²
        double p3_x3 = (5.0 * s3) * 0.5;                  // = 2.5s³

        a[0] += c[3] * p3_const;
        a[1] += c[3] * p3_x1;
        a[2] += c[3] * p3_x2;
        a[3] += c[3] * p3_x3;
    }
}

// ============================================================================
// Block-level Reduction Helpers
// ============================================================================

__device__ __forceinline__
double warpReduceSumDouble(double val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__
double blockReduceSumDouble(double val) {
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

__device__ __forceinline__
long long warpReduceMaxLL(long long val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__
long long blockReduceMaxLL(long long val) {
    __shared__ long long shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warpReduceMaxLL(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : LLONG_MIN;
    if (wid == 0) val = warpReduceMaxLL(val);

    return val;
}

// ============================================================================
// Legendre Coefficient Computation (Correct Least Squares)
// ============================================================================

/**
 * Solve a small linear system using Gaussian elimination with partial pivoting
 * Solves G * c = b where G is (d+1) x (d+1)
 *
 * @param G     Gram matrix (modified in place)
 * @param b     Right-hand side (modified to contain solution)
 * @param d     Polynomial degree (system size is d+1)
 */
__device__ __forceinline__
void solveLinearSystem(double G[4][4], double b[4], int d) {
    int size = d + 1;

    // Forward elimination with partial pivoting
    for (int k = 0; k < size; k++) {
        // Find pivot
        int maxRow = k;
        double maxVal = fabs(G[k][k]);
        for (int i = k + 1; i < size; i++) {
            if (fabs(G[i][k]) > maxVal) {
                maxVal = fabs(G[i][k]);
                maxRow = i;
            }
        }

        // Swap rows
        if (maxRow != k) {
            for (int j = k; j < size; j++) {
                double tmp = G[k][j];
                G[k][j] = G[maxRow][j];
                G[maxRow][j] = tmp;
            }
            double tmp = b[k];
            b[k] = b[maxRow];
            b[maxRow] = tmp;
        }

        // Eliminate column
        if (fabs(G[k][k]) > 1e-15) {
            for (int i = k + 1; i < size; i++) {
                double factor = G[i][k] / G[k][k];
                for (int j = k; j < size; j++) {
                    G[i][j] -= factor * G[k][j];
                }
                b[i] -= factor * b[k];
            }
        }
    }

    // Back substitution
    for (int k = size - 1; k >= 0; k--) {
        if (fabs(G[k][k]) > 1e-15) {
            for (int j = k + 1; j < size; j++) {
                b[k] -= G[k][j] * b[j];
            }
            b[k] /= G[k][k];
        } else {
            b[k] = 0.0;
        }
    }
}

/**
 * Compute Legendre coefficients using CORRECT least squares
 *
 * Solves the normal equations: G * c = b
 * where G[i,j] = Σ P_i(x') * P_j(x')  (Gram matrix)
 *       b[k] = Σ y * P_k(x')
 *
 * This handles the non-orthogonality of discrete Legendre polynomials correctly!
 *
 * @param data       Input data array
 * @param start      Start index in data
 * @param end        End index (exclusive)
 * @param degree     Polynomial degree (1, 2, or 3)
 * @param coeffs     Output: Standard polynomial coefficients [a0, a1, a2, a3]
 * @param max_error  Output: Maximum absolute error
 * @param delta_bits Output: Bits needed for deltas
 */
template<typename T>
__device__ void fitLegendreDirect(
    const T* __restrict__ data,
    int start, int end,
    int degree,
    double* coeffs,      // [4] Standard polynomial coefficients
    long long* max_error,
    int* delta_bits
) {
    int n = end - start;
    if (n <= 0) {
        coeffs[0] = coeffs[1] = coeffs[2] = coeffs[3] = 0.0;
        *max_error = 0;
        *delta_bits = 0;
        return;
    }

    // Shared memory for Gram matrix and RHS vector
    __shared__ double s_G[4][4];      // Gram matrix G[i,j] = Σ P_i * P_j
    __shared__ double s_b[4];         // RHS vector b[k] = Σ y * P_k
    __shared__ double s_coeffs[4];    // Legendre coefficients (solution)

    // Initialize shared memory
    if (threadIdx.x < 16) {
        int i = threadIdx.x / 4;
        int j = threadIdx.x % 4;
        s_G[i][j] = 0.0;
    }
    if (threadIdx.x < 4) {
        s_b[threadIdx.x] = 0.0;
        s_coeffs[threadIdx.x] = 0.0;
    }
    __syncthreads();

    // Thread-local accumulators for Gram matrix and RHS
    double local_G[4][4] = {{0}};
    double local_b[4] = {0};

    // Compute Gram matrix and RHS in parallel
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double xp = toNormalized(local_idx, n);  // x' ∈ [-1, 1]
        double y = static_cast<double>(data[i]);

        double P[4];
        legendreAll(xp, degree, P);

        // Accumulate Gram matrix entries
        for (int j = 0; j <= degree; j++) {
            local_b[j] += y * P[j];
            for (int k = 0; k <= degree; k++) {
                local_G[j][k] += P[j] * P[k];
            }
        }
    }

    // Block-level reduction for Gram matrix
    for (int j = 0; j <= degree; j++) {
        for (int k = 0; k <= degree; k++) {
            double sum = blockReduceSumDouble(local_G[j][k]);
            if (threadIdx.x == 0) {
                s_G[j][k] = sum;
            }
        }
        double sum_b = blockReduceSumDouble(local_b[j]);
        if (threadIdx.x == 0) {
            s_b[j] = sum_b;
        }
    }
    __syncthreads();

    // Solve the linear system G * c = b (only thread 0)
    if (threadIdx.x == 0) {
        // Copy to working arrays (will be modified by solver)
        double G_work[4][4];
        double b_work[4];
        for (int j = 0; j <= degree; j++) {
            b_work[j] = s_b[j];
            for (int k = 0; k <= degree; k++) {
                G_work[j][k] = s_G[j][k];
            }
        }

        // Solve using Gaussian elimination
        solveLinearSystem(G_work, b_work, degree);

        // Store Legendre coefficients
        for (int k = 0; k <= degree; k++) {
            s_coeffs[k] = b_work[k];
        }
        for (int k = degree + 1; k < 4; k++) {
            s_coeffs[k] = 0.0;
        }
    }
    __syncthreads();

    // Compute max error with Legendre prediction
    long long local_max_error = 0;

    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        int local_idx = i - start;
        double xp = toNormalized(local_idx, n);

        // Evaluate polynomial in Legendre basis
        double P[4];
        legendreAll(xp, degree, P);

        double predicted = 0.0;
        for (int k = 0; k <= degree; k++) {
            predicted += s_coeffs[k] * P[k];
        }

        T pred_val = static_cast<T>(__double2ll_rn(predicted));
        long long delta;
        if (data[i] >= pred_val) {
            delta = static_cast<long long>(data[i] - pred_val);
        } else {
            delta = -static_cast<long long>(pred_val - data[i]);
        }
        local_max_error = max(local_max_error, llabs(delta));
    }

    long long global_max_error = blockReduceMaxLL(local_max_error);

    // Output results
    if (threadIdx.x == 0) {
        // Convert Legendre coefficients to standard polynomial coefficients
        legendreToStandard(s_coeffs, degree, n, coeffs);

        *max_error = global_max_error;

        // Compute bits needed
        if (global_max_error == 0) {
            *delta_bits = 0;
        } else {
            *delta_bits = 64 - __clzll(static_cast<unsigned long long>(global_max_error)) + 1;
        }
    }
}

// ============================================================================
// Multi-Model Fitting (Compare Linear, Quadratic, Cubic)
// ============================================================================

/**
 * Fit multiple polynomial models and select the best one based on cost
 *
 * Cost = metadata_bytes + n * delta_bits / 8
 *
 * Metadata overhead per model:
 *   Linear (d=1):    16 bytes (2 doubles: a0, a1)
 *   Quadratic (d=2): 24 bytes (3 doubles: a0, a1, a2)
 *   Cubic (d=3):     32 bytes (4 doubles: a0, a1, a2, a3)
 *
 * @param data        Input data
 * @param start       Start index
 * @param end         End index (exclusive)
 * @param max_degree  Maximum degree to try (1, 2, or 3)
 * @param best_model  Output: Best model type (MODEL_LINEAR, MODEL_POLYNOMIAL2, MODEL_POLYNOMIAL3)
 * @param coeffs      Output: Best model coefficients [a0, a1, a2, a3]
 * @param best_bits   Output: Delta bits for best model
 * @param best_error  Output: Max error for best model
 * @param best_cost   Output: Cost for best model
 */
template<typename T>
__device__ void fitBestPolynomial(
    const T* __restrict__ data,
    int start, int end,
    int max_degree,
    int* best_model,
    double* coeffs,
    int* best_bits,
    long long* best_error,
    float* best_cost
) {
    int n = end - start;

    // Shared memory for each model's results
    __shared__ double s_coeffs[3][4];     // Coefficients for degree 1, 2, 3
    __shared__ int s_bits[3];              // Delta bits for each degree
    __shared__ long long s_errors[3];      // Max errors for each degree
    __shared__ float s_costs[3];           // Costs for each degree

    // Model metadata overhead (bytes)
    const float metadata_bytes[3] = {16.0f, 24.0f, 32.0f};  // d=1, d=2, d=3

    // Try each degree
    for (int d = 1; d <= max_degree; d++) {
        double local_coeffs[4];
        long long local_error;
        int local_bits;

        fitLegendreDirect<T>(data, start, end, d,
                             local_coeffs, &local_error, &local_bits);

        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; i++) {
                s_coeffs[d-1][i] = local_coeffs[i];
            }
            s_bits[d-1] = local_bits;
            s_errors[d-1] = local_error;
            s_costs[d-1] = metadata_bytes[d-1] + static_cast<float>(n) * local_bits / 8.0f;
        }
        __syncthreads();
    }

    // Select best model
    if (threadIdx.x == 0) {
        int best_d = 1;
        float min_cost = s_costs[0];

        for (int d = 2; d <= max_degree; d++) {
            if (s_costs[d-1] < min_cost) {
                min_cost = s_costs[d-1];
                best_d = d;
            }
        }

        // Map degree to model type
        // MODEL_LINEAR = 1, MODEL_POLYNOMIAL2 = 2, MODEL_POLYNOMIAL3 = 3
        *best_model = best_d;

        for (int i = 0; i < 4; i++) {
            coeffs[i] = s_coeffs[best_d-1][i];
        }
        *best_bits = s_bits[best_d-1];
        *best_error = s_errors[best_d-1];
        *best_cost = min_cost;
    }
}

// ============================================================================
// Standard Polynomial Evaluation (for decoder)
// ============================================================================

/**
 * Evaluate standard polynomial using Horner's method
 *
 * y = a0 + a1*x + a2*x² + a3*x³
 *   = a0 + x*(a1 + x*(a2 + x*a3))
 *
 * This is numerically stable and efficient (d multiplications + d additions)
 */
__device__ __forceinline__
double evaluatePolynomialHorner(const double* a, int degree, double x) {
    double result = a[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * x + a[i];
    }
    return result;
}

/**
 * Evaluate polynomial with model type dispatch
 * Optimized version without branching in inner loop
 */
template<typename T>
__device__ __forceinline__
T evaluateModel(int model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    // Use switch for model type (compiler optimizes this well)
    switch (model_type) {
        case 0:  // MODEL_CONSTANT
            predicted = params[0];
            break;
        case 1:  // MODEL_LINEAR
            predicted = params[0] + params[1] * x;
            break;
        case 2:  // MODEL_POLYNOMIAL2 (Horner)
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case 3:  // MODEL_POLYNOMIAL3 (Horner)
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        default:
            predicted = params[0] + params[1] * x;  // Fallback to linear
            break;
    }

    return static_cast<T>(__double2ll_rn(predicted));
}

}  // namespace polynomial_fitting

#endif // POLYNOMIAL_FITTING_CUH
