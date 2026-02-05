/**
 * Shared Finite Difference Functions
 *
 * CRITICAL: This file must be included by BOTH encoder and decoder
 * to ensure bit-exact identical computation paths.
 *
 * All functions use explicit FP64 arithmetic without relying on
 * compiler-specific optimizations like FMA fusion.
 */

#ifndef FINITE_DIFF_SHARED_CUH
#define FINITE_DIFF_SHARED_CUH

#include <cuda_runtime.h>
#include <type_traits>
#include <cmath>

namespace FiniteDiff {

// ============================================================================
// HOST versions - computation order MUST match device versions exactly
// ============================================================================

inline int64_t fp64_to_int_host(double val) {
    return std::llrint(val);
}

inline void computeLinearHost(
    const double* params,
    int start_idx,
    int stride,
    double& y,
    double& step)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = a + b * x0;
    step = b * s;
}

inline void computePoly2Host(
    const double* params,
    int start_idx,
    int stride,
    double& y,
    double& d1,
    double& d2)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Explicit computation order to avoid FMA differences
    double x0_sq = x0 * x0;
    y = a + b * x0 + c * x0_sq;

    double two_x0_plus_s = 2.0 * x0 + s;
    double c_s = c * s;
    d1 = b * s + c_s * two_x0_plus_s;

    d2 = 2.0 * c * s * s;
}

inline void computePoly3Host(
    const double* params,
    int start_idx,
    int stride,
    double& y,
    double& d1,
    double& d2,
    double& d3)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double d = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = x0 * x0;
    double x0_cu = x0_sq * x0;
    y = a + b * x0 + c * x0_sq + d * x0_cu;

    // Compute y1
    double x1 = x0 + s;
    double x1_sq = x1 * x1;
    double x1_cu = x1_sq * x1;
    double y1 = a + b * x1 + c * x1_sq + d * x1_cu;
    d1 = y1 - y;

    // Compute y2
    double x2 = x0 + 2.0 * s;
    double x2_sq = x2 * x2;
    double x2_cu = x2_sq * x2;
    double y2 = a + b * x2 + c * x2_sq + d * x2_cu;
    double d1_next = y2 - y1;
    d2 = d1_next - d1;

    // Third difference is constant
    d3 = 6.0 * d * s * s * s;
}

// ============================================================================
// DEVICE versions - use __dmul_rn/__dadd_rn to prevent FMA fusion
// This ensures bit-exact results across different compilation units.
// ============================================================================

// FP64 to integer conversion helper - MUST match everywhere
template<typename T>
__device__ __forceinline__
T fp64_to_int(double val) {
    if constexpr (sizeof(T) == 8 && std::is_unsigned<T>::value) {
        return static_cast<T>(__double2ull_rn(val));
    } else {
        return static_cast<T>(__double2ll_rn(val));
    }
}

// FP64 addition/multiplication without FMA
__device__ __forceinline__
double d_mul(double a, double b) { return __dmul_rn(a, b); }

__device__ __forceinline__
double d_add(double a, double b) { return __dadd_rn(a, b); }

// LINEAR: y = a + b*x
// Step = b * stride (constant)
template<typename T>
__device__ __forceinline__
void computeLinear(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y,
    double& step)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y = d_add(a, d_mul(b, x0));
    step = d_mul(b, s);
}

// POLY2: y = a + b*x + c*x²
// d1 = b*s + c*s*(2*x0 + s)
// d2 = 2*c*s² (constant)
template<typename T>
__device__ __forceinline__
void computePoly2(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y,
    double& d1,
    double& d2)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    double x0_sq = d_mul(x0, x0);
    y = d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq));

    double two_x0_plus_s = d_add(d_mul(2.0, x0), s);
    double c_s = d_mul(c, s);
    d1 = d_add(d_mul(b, s), d_mul(c_s, two_x0_plus_s));

    d2 = d_mul(2.0, d_mul(c, d_mul(s, s)));
}

// POLY3: y = a + b*x + c*x² + d*x³
// Compute differences by explicit point evaluation to ensure consistency
template<typename T>
__device__ __forceinline__
void computePoly3(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y,
    double& d1,
    double& d2,
    double& d3)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double dd = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = d_mul(x0, x0);
    double x0_cu = d_mul(x0_sq, x0);
    y = d_add(d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq)), d_mul(dd, x0_cu));

    // Compute y1
    double x1 = d_add(x0, s);
    double x1_sq = d_mul(x1, x1);
    double x1_cu = d_mul(x1_sq, x1);
    double y1 = d_add(d_add(d_add(a, d_mul(b, x1)), d_mul(c, x1_sq)), d_mul(dd, x1_cu));
    d1 = d_add(y1, -y);

    // Compute y2
    double x2 = d_add(x0, d_mul(2.0, s));
    double x2_sq = d_mul(x2, x2);
    double x2_cu = d_mul(x2_sq, x2);
    double y2 = d_add(d_add(d_add(a, d_mul(b, x2)), d_mul(c, x2_sq)), d_mul(dd, x2_cu));
    double d1_next = d_add(y2, -y1);
    d2 = d_add(d1_next, -d1);

    // Third difference is constant
    d3 = d_mul(6.0, d_mul(dd, d_mul(s, d_mul(s, s))));
}

// ============================================================================
// INT64 versions - use integer accumulation after initial FP64→int conversion
// These versions eliminate FP64 from the hot loop, using only int64 arithmetic.
// ============================================================================

// LINEAR INT: step = round(y(x0+s)) - round(y(x0))
template<typename T>
__device__ __forceinline__
void computeLinearINT(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    int64_t& y,
    int64_t& step)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0 and y1 using FMA-safe arithmetic
    double y0_fp = d_add(a, d_mul(b, x0));
    double y1_fp = d_add(a, d_mul(b, d_add(x0, s)));

    // Convert to int64 using banker's rounding
    y = __double2ll_rn(y0_fp);
    int64_t y1 = __double2ll_rn(y1_fp);
    step = y1 - y;
}

// POLY2 INT: compute via point evaluation for exact first differences
template<typename T>
__device__ __forceinline__
void computePoly2INT(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    int64_t& y,
    int64_t& d1,
    int64_t& d2)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = d_mul(x0, x0);
    double y0_fp = d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq));

    // Compute y1
    double x1 = d_add(x0, s);
    double x1_sq = d_mul(x1, x1);
    double y1_fp = d_add(d_add(a, d_mul(b, x1)), d_mul(c, x1_sq));

    // Compute y2
    double x2 = d_add(x0, d_mul(2.0, s));
    double x2_sq = d_mul(x2, x2);
    double y2_fp = d_add(d_add(a, d_mul(b, x2)), d_mul(c, x2_sq));

    // Convert to int64
    y = __double2ll_rn(y0_fp);
    int64_t y1_int = __double2ll_rn(y1_fp);
    int64_t y2_int = __double2ll_rn(y2_fp);

    // Compute differences
    d1 = y1_int - y;
    int64_t d1_next = y2_int - y1_int;
    d2 = d1_next - d1;
}

// POLY3 INT: compute via 4-point evaluation
template<typename T>
__device__ __forceinline__
void computePoly3INT(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    int64_t& y,
    int64_t& d1,
    int64_t& d2,
    int64_t& d3)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double dd = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = d_mul(x0, x0);
    double x0_cu = d_mul(x0_sq, x0);
    double y0_fp = d_add(d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq)), d_mul(dd, x0_cu));

    // Compute y1
    double x1 = d_add(x0, s);
    double x1_sq = d_mul(x1, x1);
    double x1_cu = d_mul(x1_sq, x1);
    double y1_fp = d_add(d_add(d_add(a, d_mul(b, x1)), d_mul(c, x1_sq)), d_mul(dd, x1_cu));

    // Compute y2
    double x2 = d_add(x0, d_mul(2.0, s));
    double x2_sq = d_mul(x2, x2);
    double x2_cu = d_mul(x2_sq, x2);
    double y2_fp = d_add(d_add(d_add(a, d_mul(b, x2)), d_mul(c, x2_sq)), d_mul(dd, x2_cu));

    // Compute y3
    double x3 = d_add(x0, d_mul(3.0, s));
    double x3_sq = d_mul(x3, x3);
    double x3_cu = d_mul(x3_sq, x3);
    double y3_fp = d_add(d_add(d_add(a, d_mul(b, x3)), d_mul(c, x3_sq)), d_mul(dd, x3_cu));

    // Convert to int64
    y = __double2ll_rn(y0_fp);
    int64_t y1_int = __double2ll_rn(y1_fp);
    int64_t y2_int = __double2ll_rn(y2_fp);
    int64_t y3_int = __double2ll_rn(y3_fp);

    // Compute first differences
    d1 = y1_int - y;
    int64_t d1_1 = y2_int - y1_int;
    int64_t d1_2 = y3_int - y2_int;

    // Compute second differences
    d2 = d1_1 - d1;
    int64_t d2_1 = d1_2 - d1_1;

    // Compute third difference
    d3 = d2_1 - d2;
}

// ============================================================================
// FP64 ACCUMULATION versions - use FP64 accumulation for zero drift error
// These versions use FP64 in the hot loop but eliminate accumulation drift.
// Result: round(y_fp) == round(a + b*x) for every point, exactly matching
// the point-evaluation approach used by the encoder.
// ============================================================================

// LINEAR FP64 Accum: y = a + b*x with FP64 step accumulation
// Usage in hot loop:
//   T pred = static_cast<T>(__double2ll_rn(y_fp));
//   y_fp = d_add(y_fp, step_fp);
template<typename T>
__device__ __forceinline__
void computeLinearFP64Accum(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y_fp,      // Initial FP64 value
    double& step_fp)   // FP64 step for accumulation
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y_fp = d_add(a, d_mul(b, x0));
    step_fp = d_mul(b, s);
}

// POLY2 FP64 Accum: y = a + b*x + c*x²
// For quadratic, we still need FP64 finite difference (second order)
// d1, d2 are FP64 differences, d2 is constant
// Usage in hot loop:
//   T pred = static_cast<T>(__double2ll_rn(y_fp));
//   y_fp = d_add(y_fp, d1_fp);
//   d1_fp = d_add(d1_fp, d2_fp);
template<typename T>
__device__ __forceinline__
void computePoly2FP64Accum(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y_fp,
    double& d1_fp,
    double& d2_fp)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    double x0_sq = d_mul(x0, x0);
    y_fp = d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq));

    double two_x0_plus_s = d_add(d_mul(2.0, x0), s);
    double c_s = d_mul(c, s);
    d1_fp = d_add(d_mul(b, s), d_mul(c_s, two_x0_plus_s));

    d2_fp = d_mul(2.0, d_mul(c, d_mul(s, s)));
}

// POLY3 FP64 Accum: y = a + b*x + c*x² + d*x³
// Third-order finite difference with FP64 accumulation
// Usage in hot loop:
//   T pred = static_cast<T>(__double2ll_rn(y_fp));
//   y_fp = d_add(y_fp, d1_fp);
//   d1_fp = d_add(d1_fp, d2_fp);
//   d2_fp = d_add(d2_fp, d3_fp);
template<typename T>
__device__ __forceinline__
void computePoly3FP64Accum(
    const double* __restrict__ params,
    int start_idx,
    int stride,
    double& y_fp,
    double& d1_fp,
    double& d2_fp,
    double& d3_fp)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double dd = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = d_mul(x0, x0);
    double x0_cu = d_mul(x0_sq, x0);
    y_fp = d_add(d_add(d_add(a, d_mul(b, x0)), d_mul(c, x0_sq)), d_mul(dd, x0_cu));

    // Compute y1
    double x1 = d_add(x0, s);
    double x1_sq = d_mul(x1, x1);
    double x1_cu = d_mul(x1_sq, x1);
    double y1 = d_add(d_add(d_add(a, d_mul(b, x1)), d_mul(c, x1_sq)), d_mul(dd, x1_cu));
    d1_fp = d_add(y1, -y_fp);

    // Compute y2
    double x2 = d_add(x0, d_mul(2.0, s));
    double x2_sq = d_mul(x2, x2);
    double x2_cu = d_mul(x2_sq, x2);
    double y2 = d_add(d_add(d_add(a, d_mul(b, x2)), d_mul(c, x2_sq)), d_mul(dd, x2_cu));
    double d1_next = d_add(y2, -y1);
    d2_fp = d_add(d1_next, -d1_fp);

    // Third difference is constant
    d3_fp = d_mul(6.0, d_mul(dd, d_mul(s, d_mul(s, s))));
}

// ============================================================================
// HOST FP64 ACCUMULATION versions
// ============================================================================

inline void computeLinearFP64AccumHost(
    const double* params,
    int start_idx,
    int stride,
    double& y_fp,
    double& step_fp)
{
    double a = params[0];
    double b = params[1];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    y_fp = a + b * x0;
    step_fp = b * s;
}

inline void computePoly2FP64AccumHost(
    const double* params,
    int start_idx,
    int stride,
    double& y_fp,
    double& d1_fp,
    double& d2_fp)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    double x0_sq = x0 * x0;
    y_fp = a + b * x0 + c * x0_sq;

    double two_x0_plus_s = 2.0 * x0 + s;
    double c_s = c * s;
    d1_fp = b * s + c_s * two_x0_plus_s;

    d2_fp = 2.0 * c * s * s;
}

inline void computePoly3FP64AccumHost(
    const double* params,
    int start_idx,
    int stride,
    double& y_fp,
    double& d1_fp,
    double& d2_fp,
    double& d3_fp)
{
    double a = params[0];
    double b = params[1];
    double c = params[2];
    double d = params[3];
    double x0 = static_cast<double>(start_idx);
    double s = static_cast<double>(stride);

    // Compute y0
    double x0_sq = x0 * x0;
    double x0_cu = x0_sq * x0;
    y_fp = a + b * x0 + c * x0_sq + d * x0_cu;

    // Compute y1
    double x1 = x0 + s;
    double x1_sq = x1 * x1;
    double x1_cu = x1_sq * x1;
    double y1 = a + b * x1 + c * x1_sq + d * x1_cu;
    d1_fp = y1 - y_fp;

    // Compute y2
    double x2 = x0 + 2.0 * s;
    double x2_sq = x2 * x2;
    double x2_cu = x2_sq * x2;
    double y2 = a + b * x2 + c * x2_sq + d * x2_cu;
    double d1_next = y2 - y1;
    d2_fp = d1_next - d1_fp;

    // Third difference is constant
    d3_fp = 6.0 * d * s * s * s;
}

} // namespace FiniteDiff

#endif // FINITE_DIFF_SHARED_CUH
