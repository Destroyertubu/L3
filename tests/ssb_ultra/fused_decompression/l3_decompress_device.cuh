/**
 * @file l3_decompress_device.cuh
 * @brief L3 Device Decompression Functions for True Fused Query Execution
 *
 * This header provides device functions to decompress L3 compressed values
 * directly in registers, enabling true fused decompression where data is
 * decompressed inside the query kernel without intermediate global memory writes.
 *
 * Key Functions:
 * - decompressValue_L3(): Decompress a single value in registers
 * - decompressTile_L3(): Decompress a tile of values (for block processing)
 *
 * Usage Pattern:
 *   // Inside query kernel
 *   uint32_t orderdate = decompressValue_L3<uint32_t>(
 *       delta_array, model_type, params, delta_bits, bit_offset_base, local_idx);
 *   // Immediately use for filtering/computation
 *   if (orderdate >= 19930101 && orderdate <= 19931231) { ... }
 *
 * Date: 2025-12-09
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

namespace l3_fused {

// ============================================================================
// Constants (from L3_format.hpp)
// ============================================================================

enum ModelType : int32_t {
    MODEL_CONSTANT = 0,      // f(x) = θ₀
    MODEL_LINEAR = 1,        // f(x) = θ₀ + θ₁·x
    MODEL_POLYNOMIAL2 = 2,   // f(x) = θ₀ + θ₁·x + θ₂·x²
    MODEL_POLYNOMIAL3 = 3,   // f(x) = θ₀ + θ₁·x + θ₂·x² + θ₃·x³
    MODEL_FOR_BITPACK = 4,   // FOR + BitPacking
    MODEL_DIRECT_COPY = 5    // Direct copy
};

// ============================================================================
// Helper Functions (inline to avoid linking issues)
// ============================================================================

/**
 * @brief Runtime mask generation for 64-bit values
 */
__device__ __forceinline__
uint64_t mask64_rt(int k) {
    if (k <= 0) return 0ULL;
    if (k >= 64) return ~0ULL;
    return (1ULL << k) - 1ULL;
}

/**
 * @brief Sign extend a value from bit_width to 64 bits
 */
__device__ __forceinline__
int64_t sign_extend_64(uint64_t value, int bit_width) {
    if (bit_width >= 64) return static_cast<int64_t>(value);
    if (bit_width <= 0) return 0;

    const uint64_t sign_bit = value >> (bit_width - 1);
    const uint64_t sign_mask = -sign_bit;
    const uint64_t extend_mask = ~mask64_rt(bit_width);

    return static_cast<int64_t>(value | (sign_mask & extend_mask));
}

/**
 * @brief Branchless 64-bit extraction from bit-packed array
 *
 * Always loads two words and stitches them - no branch for boundary cases.
 * This eliminates warp divergence.
 */
__device__ __forceinline__
uint64_t extract_branchless_64_rt(
    const uint32_t* __restrict__ words,
    int64_t start_bit,
    int bits)
{
    if (bits <= 0) return 0ULL;
    if (bits > 64) bits = 64;

    // Always use 64-bit path for runtime
    const uint64_t word64_idx = start_bit >> 6;
    const int bit_offset = start_bit & 63;

    const uint64_t* __restrict__ p64 = reinterpret_cast<const uint64_t*>(words);
    const uint64_t lo = __ldg(&p64[word64_idx]);
    const uint64_t hi = __ldg(&p64[word64_idx + 1]);

    // Branchless stitch
    const uint64_t shifted_lo = lo >> bit_offset;
    const uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));

    return (shifted_lo | shifted_hi) & mask64_rt(bits);
}

// ============================================================================
// Polynomial Prediction
// ============================================================================

/**
 * @brief Compute polynomial prediction using Horner's method
 *
 * Supports:
 *   MODEL_CONSTANT (0):    y = a0
 *   MODEL_LINEAR (1):      y = a0 + a1*x
 *   MODEL_POLYNOMIAL2 (2): y = a0 + x*(a1 + x*a2)
 *   MODEL_POLYNOMIAL3 (3): y = a0 + x*(a1 + x*(a2 + x*a3))
 *   MODEL_FOR_BITPACK (4): Returns base value (params[0])
 */
template<typename T>
__device__ __forceinline__
T computePredictionPoly(int32_t model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:  // 0
            predicted = params[0];
            break;
        case MODEL_LINEAR:    // 1
            predicted = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:  // 2 - Horner: a0 + x*(a1 + x*a2)
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:  // 3 - Horner: a0 + x*(a1 + x*(a2 + x*a3))
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        case MODEL_FOR_BITPACK:  // 4 - FOR: base is params[0]
        case MODEL_DIRECT_COPY:  // 5 - Direct: params[0] is 0
            if (sizeof(T) == 8) {
                return static_cast<T>(__double_as_longlong(params[0]));
            } else {
                return static_cast<T>(__double2ll_rn(params[0]));
            }
        default:
            // Fallback to linear for unknown types
            predicted = params[0] + params[1] * x;
            break;
    }

    // Use __double2ll_rn for banker's rounding - matches encoder's rounding
    return static_cast<T>(__double2ll_rn(predicted));
}

// ============================================================================
// Core Decompression Function
// ============================================================================

/**
 * @brief Decompress a single L3 value directly in registers
 *
 * This is the core function for true fused decompression. It:
 * 1. Computes polynomial prediction
 * 2. Extracts delta from bit-packed array
 * 3. Sign extends delta
 * 4. Returns predicted + delta
 *
 * All operations happen in registers - no global memory write for intermediate result.
 *
 * @param delta_array      Bit-packed delta array
 * @param model_type       Model type for this partition
 * @param model_params     Polynomial params [theta0, theta1, theta2, theta3]
 * @param delta_bits       Bit width for this partition
 * @param bit_offset_base  Starting bit offset for this partition
 * @param local_idx        Index within partition [0, partition_size)
 * @return Decompressed value
 */
template<typename T>
__device__ __forceinline__
T decompressValue_L3(
    const uint32_t* __restrict__ delta_array,
    int32_t model_type,
    const double* model_params,
    int32_t delta_bits,
    int64_t bit_offset_base,
    int local_idx)
{
    // Handle FOR+BitPack and DIRECT_COPY specially
    if (model_type == MODEL_FOR_BITPACK || model_type == MODEL_DIRECT_COPY) {
        // Get base value
        T base;
        if (sizeof(T) == 8) {
            base = static_cast<T>(__double_as_longlong(model_params[0]));
        } else {
            base = static_cast<T>(__double2ll_rn(model_params[0]));
        }

        bool is_direct_copy = (model_params[0] == 0.0 && model_params[1] == 0.0);

        if (delta_bits == 0) {
            return base;
        } else if (is_direct_copy) {
            // DIRECT_COPY: delta IS the actual value (need sign extend)
            int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t extracted = extract_branchless_64_rt(delta_array, bit_offset, delta_bits);

            if constexpr (std::is_signed<T>::value) {
                int64_t signed_val = sign_extend_64(extracted, delta_bits);
                return static_cast<T>(signed_val);
            } else {
                return static_cast<T>(extracted);
            }
        } else {
            // FOR+BitPack: val = base + unsigned_delta
            int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
            uint64_t delta = extract_branchless_64_rt(delta_array, bit_offset, delta_bits);
            return base + static_cast<T>(delta);
        }
    }

    // Polynomial models (CONSTANT, LINEAR, POLYNOMIAL2, POLYNOMIAL3)

    // 1. Compute prediction using polynomial model
    T predicted = computePredictionPoly<T>(model_type, model_params, local_idx);

    // 2. Handle zero-bit case (perfect prediction)
    if (delta_bits == 0) {
        return predicted;
    }

    // 3. Extract delta from bit-packed array
    int64_t bit_offset = bit_offset_base + static_cast<int64_t>(local_idx) * delta_bits;
    uint64_t extracted = extract_branchless_64_rt(delta_array, bit_offset, delta_bits);

    // 4. Sign extend and combine
    if constexpr (std::is_signed<T>::value) {
        int64_t delta = sign_extend_64(extracted, delta_bits);
        return predicted + static_cast<T>(delta);
    } else {
        // For unsigned types, use sign extension anyway (deltas can be negative)
        int64_t delta = sign_extend_64(extracted, delta_bits);
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

// ============================================================================
// Partition Metadata Cache (for efficient block processing)
// ============================================================================

#ifndef L3_FUSED_PARTITION_METADATA_CACHE_DEFINED
/**
 * @brief Cache structure for partition metadata
 *
 * Loaded once per block, then reused for all elements in the partition.
 * Note: If fused_kernel_common.cuh is included first, that version is used instead.
 */
struct PartitionMetadataCache {
    int32_t model_type;
    double model_params[4];
    int32_t delta_bits;
    int64_t bit_offset_base;
    int32_t start_idx;
    int32_t end_idx;

    /**
     * @brief Load partition metadata from device arrays
     */
    __device__ __forceinline__
    void load(
        int partition_id,
        const int32_t* __restrict__ d_model_types,
        const double* __restrict__ d_model_params,
        const int32_t* __restrict__ d_delta_bits,
        const int64_t* __restrict__ d_bit_offsets,
        const int32_t* __restrict__ d_start_indices,
        const int32_t* __restrict__ d_end_indices)
    {
        model_type = d_model_types[partition_id];
        model_params[0] = d_model_params[partition_id * 4];
        model_params[1] = d_model_params[partition_id * 4 + 1];
        model_params[2] = d_model_params[partition_id * 4 + 2];
        model_params[3] = d_model_params[partition_id * 4 + 3];
        delta_bits = d_delta_bits[partition_id];
        bit_offset_base = d_bit_offsets[partition_id];
        start_idx = d_start_indices[partition_id];
        end_idx = d_end_indices[partition_id];
    }

    /**
     * @brief Get partition size
     */
    __device__ __forceinline__
    int size() const { return end_idx - start_idx; }

    /**
     * @brief Decompress a value using cached metadata
     */
    template<typename T>
    __device__ __forceinline__
    T decompress(const uint32_t* __restrict__ delta_array, int local_idx) const {
        return decompressValue_L3<T>(
            delta_array, model_type, model_params, delta_bits, bit_offset_base, local_idx);
    }
};
#endif // L3_FUSED_PARTITION_METADATA_CACHE_DEFINED

// ============================================================================
// Multi-Column Decompression Helper
// ============================================================================

/**
 * @brief Compressed column accessor for fused decompression
 *
 * Wraps all metadata arrays for a single compressed column.
 */
struct CompressedColumnFused {
    const uint32_t* delta_array;
    const int32_t* model_types;
    const double* model_params;
    const int32_t* delta_bits;
    const int64_t* bit_offsets;
    const int32_t* start_indices;
    const int32_t* end_indices;
    int num_partitions;

    /**
     * @brief Decompress a single value by global index
     *
     * Note: This requires finding the partition for each element (expensive).
     * Prefer partition-based processing when possible.
     */
    template<typename T>
    __device__ __forceinline__
    T decompressByGlobalIdx(int global_idx) const {
        // Binary search for partition
        int left = 0, right = num_partitions - 1;
        int pid = -1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (global_idx >= start_indices[mid] && global_idx < end_indices[mid]) {
                pid = mid;
                break;
            } else if (global_idx < start_indices[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }

        if (pid < 0) return T(0);

        int local_idx = global_idx - start_indices[pid];
        double params[4] = {
            model_params[pid * 4],
            model_params[pid * 4 + 1],
            model_params[pid * 4 + 2],
            model_params[pid * 4 + 3]
        };

        return decompressValue_L3<T>(
            delta_array, model_types[pid], params, delta_bits[pid],
            bit_offsets[pid], local_idx);
    }
};

} // namespace l3_fused
