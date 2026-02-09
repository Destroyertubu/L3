/**
 * V15 Common - L3 V4 Vectorized Decoder with Dynamic Bit-widths
 *
 * Key differences from V12:
 * 1. Dynamic bit-widths (read from L3 metadata) instead of hardcoded
 * 2. V4 vectorized extraction (4 values at once) for better throughput
 * 3. Supports all model types (LINEAR, POLY2, POLY3, FOR_BITPACK)
 * 4. MINI_VECTOR_SIZE = 2048, VALUES_PER_THREAD = 64 (matches libL3_kernels.a)
 *
 * Key differences from V13:
 * 5. CONSTANT model fix: params[0]=num_runs, params[1]=base_value
 *
 * Query plan remains unchanged from V12.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

// L3 Vertical format configuration - configurable via build flags
// Supported: L3_VERTICAL_256_CONFIG, L3_VERTICAL_512_CONFIG, L3_VERTICAL_1024_CONFIG, L3_VERTICAL_2048_CONFIG
// Default to 2048 if not specified
#if !defined(L3_VERTICAL_256_CONFIG) && !defined(L3_VERTICAL_512_CONFIG) && !defined(L3_VERTICAL_1024_CONFIG) && !defined(L3_VERTICAL_2048_CONFIG)
#define L3_VERTICAL_2048_CONFIG
#endif

#include "L3_Vertical_format.hpp"
#include "crystal_hash_v7.cuh"
#include "crystal_hash_build_v7.cuh"

namespace v15 {

// ============================================================================
// V15 Configuration - Uses compile-time MINI_VECTOR_SIZE from L3_Vertical_format.hpp
// Supported configurations:
//   - L3_VERTICAL_256_CONFIG:  256 values (8 per thread)
//   - L3_VERTICAL_512_CONFIG:  512 values (16 per thread)
//   - L3_VERTICAL_1024_CONFIG: 1024 values (32 per thread)
//   - L3_VERTICAL_2048_CONFIG: 2048 values (64 per thread)
// ============================================================================
#if defined(L3_VERTICAL_256_CONFIG)
static_assert(MINI_VECTOR_SIZE == 256, "L3_VERTICAL_256_CONFIG requires MINI_VECTOR_SIZE=256");
static_assert(VALUES_PER_THREAD == 8, "L3_VERTICAL_256_CONFIG requires VALUES_PER_THREAD=8");
#elif defined(L3_VERTICAL_512_CONFIG)
static_assert(MINI_VECTOR_SIZE == 512, "L3_VERTICAL_512_CONFIG requires MINI_VECTOR_SIZE=512");
static_assert(VALUES_PER_THREAD == 16, "L3_VERTICAL_512_CONFIG requires VALUES_PER_THREAD=16");
#elif defined(L3_VERTICAL_1024_CONFIG)
static_assert(MINI_VECTOR_SIZE == 1024, "L3_VERTICAL_1024_CONFIG requires MINI_VECTOR_SIZE=1024");
static_assert(VALUES_PER_THREAD == 32, "L3_VERTICAL_1024_CONFIG requires VALUES_PER_THREAD=32");
#else
static_assert(MINI_VECTOR_SIZE == 2048, "L3_VERTICAL_2048_CONFIG requires MINI_VECTOR_SIZE=2048");
static_assert(VALUES_PER_THREAD == 64, "L3_VERTICAL_2048_CONFIG requires VALUES_PER_THREAD=64");
#endif

constexpr int V15_MINI_VECTOR_SIZE = MINI_VECTOR_SIZE;
constexpr int V15_VALUES_PER_THREAD = VALUES_PER_THREAD;
constexpr int V15_WARP_SIZE = 32;
constexpr int V15_THREADS_PER_BLOCK = 32;  // One warp per partition

// ============================================================================
// Model type constants (from L3_format.hpp)
// ============================================================================
// MODEL_CONSTANT = 0
// MODEL_LINEAR = 1
// MODEL_POLYNOMIAL2 = 2
// MODEL_POLYNOMIAL3 = 3
// MODEL_FOR_BITPACK = 4

// ============================================================================
// Utility Functions
// ============================================================================

__device__ __forceinline__ int64_t sign_extend_v15(uint64_t val, int bits) {
    if (bits <= 0 || bits >= 64) return static_cast<int64_t>(val);
    uint64_t sign_bit = 1ULL << (bits - 1);
    if (val & sign_bit) {
        return static_cast<int64_t>(val | (~0ULL << bits));
    }
    return static_cast<int64_t>(val);
}

// ============================================================================
// V4 Vectorized Bit Extraction - 4 values at once (32-bit optimized)
// ============================================================================
__device__ __forceinline__ void extract_vectorized_4_v15(
    const uint32_t* __restrict__ words,
    uint64_t start_bit,
    int bits,
    uint32_t& v0, uint32_t& v1, uint32_t& v2, uint32_t& v3)
{
    if (bits <= 0) {
        v0 = v1 = v2 = v3 = 0U;
        return;
    }
    if (bits > 32) bits = 32;

    const uint64_t MASK = (bits == 32) ? 0xFFFFFFFFULL : ((1ULL << bits) - 1ULL);

    auto extract_one = [&](int value_idx) -> uint32_t {
        uint64_t bit_pos = start_bit + static_cast<uint64_t>(value_idx) * bits;
        uint32_t word_idx = static_cast<uint32_t>(bit_pos >> 5);
        int bit_in_word = static_cast<int>(bit_pos & 31);

        uint32_t lo = __ldg(&words[word_idx]);
        uint32_t hi = __ldg(&words[word_idx + 1]);
        uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;

        return static_cast<uint32_t>((combined >> bit_in_word) & MASK);
    };

    v0 = extract_one(0);
    v1 = extract_one(1);
    v2 = extract_one(2);
    v3 = extract_one(3);
}

// Single value extraction for tail handling
__device__ __forceinline__ uint32_t extract_single_v15(
    const uint32_t* __restrict__ words,
    int64_t bit_offset,
    int bits)
{
    if (bits <= 0) return 0U;
    if (bits > 32) bits = 32;

    const uint32_t MASK = (bits == 32) ? 0xFFFFFFFFU : ((1U << bits) - 1U);

    int64_t word_idx = bit_offset >> 5;
    int bit_in_word = bit_offset & 31;

    uint32_t lo = __ldg(&words[word_idx]);
    uint32_t hi = __ldg(&words[word_idx + 1]);
    uint64_t combined = (static_cast<uint64_t>(hi) << 32) | lo;

    return static_cast<uint32_t>((combined >> bit_in_word) & MASK);
}

// ============================================================================
// Polynomial Prediction (for regression models)
// ============================================================================
template<typename T>
__device__ __forceinline__
T computePredictionV15(int32_t model_type, const double* params, int local_idx) {
    double x = static_cast<double>(local_idx);
    double predicted;

    switch (model_type) {
        case MODEL_CONSTANT:
            predicted = params[1];
            break;
        case MODEL_LINEAR:
            predicted = params[0] + params[1] * x;
            break;
        case MODEL_POLYNOMIAL2:
            predicted = params[0] + x * (params[1] + x * params[2]);
            break;
        case MODEL_POLYNOMIAL3:
            predicted = params[0] + x * (params[1] + x * (params[2] + x * params[3]));
            break;
        default:
            predicted = params[0] + params[1] * x;
            break;
    }

    return static_cast<T>(__double2ll_rn(predicted));
}

template<typename T>
__device__ __forceinline__
T applyDeltaV15(T predicted, int64_t delta) {
    return static_cast<T>(static_cast<int64_t>(predicted) + delta);
}

// ============================================================================
// V15 Fused Decode Function - Decodes a partition into thread-local array
// This is the core function used by all SSB queries
// ============================================================================
__device__ __forceinline__ void decode_l3_partition_v15(
    const uint32_t* __restrict__ interleaved_data,
    int64_t partition_word_offset,
    int partition_size,
    int delta_bits,
    int model_type,
    const double* params,
    int lane_id,
    uint32_t* out)  // [V15_VALUES_PER_THREAD]
{
    int num_mv = partition_size / V15_MINI_VECTOR_SIZE;

    // Pre-compute base value for FOR_BITPACK and CONSTANT
    uint32_t base_value = 0;
    if (model_type == MODEL_FOR_BITPACK) {
        base_value = static_cast<uint32_t>(__double2ll_rn(params[0]));
    } else if (model_type == MODEL_CONSTANT) {
        base_value = static_cast<uint32_t>(__double2ll_rn(params[1]));
    }

    // Handle no interleaved data case (delta_bits == 0 or word_offset < 0)
    // This happens when all values can be perfectly predicted by the model
    if (partition_word_offset < 0 || delta_bits == 0) {
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            int local_idx = v * V15_WARP_SIZE + lane_id;
            if (model_type == MODEL_FOR_BITPACK || model_type == MODEL_CONSTANT) {
                out[v] = base_value;
            } else {
                out[v] = computePredictionV15<uint32_t>(model_type, params, local_idx);
            }
        }
        return;
    }

    // Process complete mini-vectors (only when delta_bits > 0)
    for (int mv_idx = 0; mv_idx < num_mv; mv_idx++) {
        int64_t mv_bit_base = (partition_word_offset << 5) +
                             static_cast<int64_t>(mv_idx) * V15_MINI_VECTOR_SIZE * delta_bits;
        int64_t lane_bit_start = mv_bit_base + static_cast<int64_t>(lane_id) * V15_VALUES_PER_THREAD * delta_bits;
        int64_t local_bit = lane_bit_start;

        // Vectorized extraction - 4 values at a time
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v += 4) {
            uint32_t vals[4];
            extract_vectorized_4_v15(interleaved_data, static_cast<uint64_t>(local_bit), delta_bits,
                                    vals[0], vals[1], vals[2], vals[3]);

            #pragma unroll
            for (int i = 0; i < 4; i++) {
                int vv = v + i;
                int local_idx_in_partition = mv_idx * V15_MINI_VECTOR_SIZE + vv * V15_WARP_SIZE + lane_id;

                if (model_type == MODEL_FOR_BITPACK) {
                    out[vv] = base_value + vals[i];
                } else {
                    uint32_t predicted = computePredictionV15<uint32_t>(model_type, params, local_idx_in_partition);
                    int64_t delta = sign_extend_v15(vals[i], delta_bits);
                    out[vv] = applyDeltaV15(predicted, delta);
                }
            }
            local_bit += 4 * delta_bits;
        }
    }
}

// Simplified decode for single column with CompressedDataVertical
__device__ __forceinline__ void decode_column_v15(
    const CompressedDataVertical<uint32_t>& col,
    int partition_idx,
    int lane_id,
    uint32_t* out)  // [V15_VALUES_PER_THREAD]
{
    int delta_bits = col.d_delta_bits[partition_idx];
    int model_type = col.d_model_types[partition_idx];
    int64_t word_offset = col.d_interleaved_offsets[partition_idx];
    int start_idx = col.d_start_indices[partition_idx];
    int end_idx = col.d_end_indices[partition_idx];
    int partition_size = end_idx - start_idx;

    // Load parameters
    double params[4];
    params[0] = col.d_model_params[partition_idx * 4];
    if (model_type != MODEL_FOR_BITPACK) {
        params[1] = col.d_model_params[partition_idx * 4 + 1];
        params[2] = col.d_model_params[partition_idx * 4 + 2];
        params[3] = col.d_model_params[partition_idx * 4 + 3];
    }

    decode_l3_partition_v15(
        col.d_interleaved_deltas,
        word_offset,
        partition_size,
        delta_bits,
        model_type,
        params,
        lane_id,
        out);
}

// ============================================================================
// Tail value decode (for last incomplete partition)
// ============================================================================
__device__ __forceinline__ uint32_t decode_tail_value_v15(
    const CompressedDataVertical<uint32_t>& col,
    int partition_idx,
    int value_idx_in_partition)
{
    int delta_bits = col.d_delta_bits[partition_idx];
    int model_type = col.d_model_types[partition_idx];
    int64_t word_offset = col.d_interleaved_offsets[partition_idx];
    int num_mv = col.d_num_mini_vectors[partition_idx];

    double params[4];
    params[0] = col.d_model_params[partition_idx * 4];
    if (model_type != MODEL_FOR_BITPACK) {
        params[1] = col.d_model_params[partition_idx * 4 + 1];
        params[2] = col.d_model_params[partition_idx * 4 + 2];
        params[3] = col.d_model_params[partition_idx * 4 + 3];
    }

    uint32_t base_value = 0;
    if (model_type == MODEL_FOR_BITPACK) {
        base_value = static_cast<uint32_t>(__double2ll_rn(params[0]));
    } else if (model_type == MODEL_CONSTANT) {
        base_value = static_cast<uint32_t>(__double2ll_rn(params[1]));
    }

    if (delta_bits == 0) {
        if (model_type == MODEL_FOR_BITPACK || model_type == MODEL_CONSTANT) {
            return base_value;
        }
        return computePredictionV15<uint32_t>(model_type, params, value_idx_in_partition);
    }

    // Calculate bit offset for tail value
    int64_t tail_bit_base = (word_offset << 5) +
                           static_cast<int64_t>(num_mv) * V15_MINI_VECTOR_SIZE * delta_bits;
    int tail_local_idx = value_idx_in_partition - num_mv * V15_MINI_VECTOR_SIZE;
    int64_t bit_offset = tail_bit_base + static_cast<int64_t>(tail_local_idx) * delta_bits;

    uint32_t extracted = extract_single_v15(col.d_interleaved_deltas, bit_offset, delta_bits);

    if (model_type == MODEL_FOR_BITPACK || model_type == MODEL_CONSTANT) {
        return base_value + extracted;
    }

    uint32_t predicted = computePredictionV15<uint32_t>(model_type, params, value_idx_in_partition);
    int64_t delta = sign_extend_v15(extracted, delta_bits);
    return applyDeltaV15(predicted, delta);
}

// ============================================================================
// SSB Column Bit-Widths (Uniform across all 117,188 partitions)
// This allows direct offset calculation without memory access
// ============================================================================
constexpr int BW_ORDERDATE = 16;
constexpr int BW_QUANTITY = 6;
constexpr int BW_DISCOUNT = 4;
constexpr int BW_EXTENDEDPRICE = 16;
constexpr int BW_REVENUE = 16;
constexpr int BW_SUPPLYCOST = 10;
constexpr int BW_CUSTKEY = 20;
constexpr int BW_PARTKEY = 20;
constexpr int BW_SUPPKEY = 16;

// ============================================================================
// Compile-time mask generation (same as V12)
// ============================================================================
template<int BIT_WIDTH>
__device__ __forceinline__ constexpr uint32_t mask32_v15() {
    if constexpr (BIT_WIDTH == 0) return 0;
    else if constexpr (BIT_WIDTH >= 32) return 0xFFFFFFFFU;
    else return (1U << BIT_WIDTH) - 1U;
}

// ============================================================================
// V15 Register-based Decode (Original approach)
// - Each thread pre-loads its data range into registers
// - Direct offset calculation for known bit-widths
// - Base value passed as parameter (read once in kernel)
//
// Note: Memory access is strided across warp (not coalesced), but L2 cache
// helps mitigate this. Alternative approaches (shared memory, shuffle) were
// tested but proved slower due to bank conflicts and synchronization overhead.
// ============================================================================
template<int BIT_WIDTH>
__device__ __forceinline__ void decode_fls_v15(
    const uint32_t* __restrict__ data,
    int partition_idx,
    int lane_id,
    uint32_t base_value,
    uint32_t* out)  // [V15_VALUES_PER_THREAD]
{
    constexpr uint32_t MASK = mask32_v15<BIT_WIDTH>();

    // Handle zero bit-width case
    if constexpr (BIT_WIDTH == 0) {
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
            out[v] = base_value;
        }
        return;
    }

    // Direct offset calculation
    constexpr int BITS_PER_PARTITION = V15_MINI_VECTOR_SIZE * BIT_WIDTH;
    int64_t partition_bit_base = static_cast<int64_t>(partition_idx) * BITS_PER_PARTITION;

    constexpr int BITS_PER_LANE = V15_VALUES_PER_THREAD * BIT_WIDTH;
    int64_t lane_bit_start = partition_bit_base + static_cast<int64_t>(lane_id) * BITS_PER_LANE;
    int64_t lane_word_start = lane_bit_start >> 5;

    // Pre-load all data into registers
    constexpr int bits_per_lane = V15_VALUES_PER_THREAD * BIT_WIDTH;
    constexpr int words_per_lane = (bits_per_lane + 63) / 32;
    constexpr int MAX_WORDS = 34;
    constexpr int actual_words = (words_per_lane < MAX_WORDS) ? words_per_lane : MAX_WORDS;

    uint32_t lane_words[MAX_WORDS];
    #pragma unroll
    for (int i = 0; i < MAX_WORDS; i++) {
        lane_words[i] = (i < actual_words) ? __ldg(&data[lane_word_start + i]) : 0;
    }

    int local_bit = lane_bit_start & 31;

    // Extract values from pre-loaded registers
    #pragma unroll
    for (int v = 0; v < V15_VALUES_PER_THREAD; v++) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint32_t extracted = static_cast<uint32_t>((combined >> bit_in_word) & MASK);

        out[v] = base_value + extracted;
        local_bit += BIT_WIDTH;
    }
}

// Tail value decode with raw pointer
template<int BIT_WIDTH>
__device__ __forceinline__ uint32_t decode_fls_single_v15(
    const uint32_t* __restrict__ data,
    int partition_idx,
    int value_idx_in_partition,
    uint32_t base_value)
{
    constexpr uint32_t MASK = mask32_v15<BIT_WIDTH>();

    if constexpr (BIT_WIDTH == 0) {
        return base_value;
    }

    constexpr int BITS_PER_PARTITION = V15_MINI_VECTOR_SIZE * BIT_WIDTH;
    int64_t partition_bit_base = static_cast<int64_t>(partition_idx) * BITS_PER_PARTITION;
    int64_t bit_offset = partition_bit_base + static_cast<int64_t>(value_idx_in_partition) * BIT_WIDTH;
    int64_t word_idx = bit_offset >> 5;
    int bit_in_word = bit_offset & 31;

    uint32_t w0 = __ldg(&data[word_idx]);
    uint32_t w1 = __ldg(&data[word_idx + 1]);
    uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
    uint32_t extracted = static_cast<uint32_t>((combined >> bit_in_word) & MASK);

    return base_value + extracted;
}

// Single value decode for TRANSPOSED layout (used within full mini-vectors)
// Data layout: Lane 0's 32 values first, then Lane 1's 32 values, etc.
// value_idx = v * WARP_SIZE + lane_id maps to bit offset = lane * 32 * BW + v * BW
template<int BIT_WIDTH>
__device__ __forceinline__ uint32_t decode_fls_single_transposed_v15(
    const uint32_t* __restrict__ data,
    int partition_idx,
    int value_idx_in_partition,
    uint32_t base_value)
{
    constexpr uint32_t MASK = mask32_v15<BIT_WIDTH>();

    if constexpr (BIT_WIDTH == 0) {
        return base_value;
    }

    constexpr int BITS_PER_PARTITION = V15_MINI_VECTOR_SIZE * BIT_WIDTH;
    int64_t partition_bit_base = static_cast<int64_t>(partition_idx) * BITS_PER_PARTITION;

    // Transposed layout: value at (v, lane) is stored at lane * 32 * BW + v * BW
    int lane = value_idx_in_partition % V15_WARP_SIZE;  // 0-31
    int v = value_idx_in_partition / V15_WARP_SIZE;      // 0-31
    int64_t bit_offset = partition_bit_base +
                         static_cast<int64_t>(lane) * V15_VALUES_PER_THREAD * BIT_WIDTH +
                         static_cast<int64_t>(v) * BIT_WIDTH;

    int64_t word_idx = bit_offset >> 5;
    int bit_in_word = bit_offset & 31;

    uint32_t w0 = __ldg(&data[word_idx]);
    uint32_t w1 = __ldg(&data[word_idx + 1]);
    uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
    uint32_t extracted = static_cast<uint32_t>((combined >> bit_in_word) & MASK);

    return base_value + extracted;
}

// Legacy wrapper for backward compatibility (reads base from struct)
template<int BIT_WIDTH>
__device__ __forceinline__ void decode_column_v15_opt(
    const CompressedDataVertical<uint32_t>& col,
    int partition_idx,
    int lane_id,
    uint32_t* out)
{
    int model_type = col.d_model_types[partition_idx];
    if (model_type == MODEL_CONSTANT) {
        uint32_t val = static_cast<uint32_t>(col.d_model_params[partition_idx * 4 + 1]);
        #pragma unroll
        for (int v = 0; v < V15_VALUES_PER_THREAD; v++) out[v] = val;
        return;
    }
    uint32_t base_value = static_cast<uint32_t>(col.d_model_params[partition_idx * 4]);
    decode_fls_v15<BIT_WIDTH>(col.d_interleaved_deltas, partition_idx, lane_id, base_value, out);
}

template<int BIT_WIDTH>
__device__ __forceinline__ uint32_t decode_tail_value_v15_opt(
    const CompressedDataVertical<uint32_t>& col,
    int partition_idx,
    int value_idx_in_partition)
{
    if (col.d_model_types[partition_idx] == MODEL_CONSTANT) {
        return static_cast<uint32_t>(col.d_model_params[partition_idx * 4 + 1]);
    }
    uint32_t base_value = static_cast<uint32_t>(col.d_model_params[partition_idx * 4]);
    return decode_fls_single_v15<BIT_WIDTH>(col.d_interleaved_deltas, partition_idx, value_idx_in_partition, base_value);
}

// ============================================================================
// Date filter constants (same as V12)
// ============================================================================
constexpr uint32_t YEAR_1992_START = 19920101;
constexpr uint32_t YEAR_1992_END   = 19921231;
constexpr uint32_t YEAR_1993_START = 19930101;
constexpr uint32_t YEAR_1993_END   = 19931231;
constexpr uint32_t YEAR_1994_START = 19940101;
constexpr uint32_t YEAR_1994_END   = 19941231;
constexpr uint32_t YEAR_1997_START = 19970101;
constexpr uint32_t YEAR_1997_END   = 19971231;
constexpr uint32_t YEAR_1998_START = 19980101;
constexpr uint32_t YEAR_1998_END   = 19981231;

// Q1.2: yearmonthnum = 199401
constexpr uint32_t MONTH_199401_START = 19940101;
constexpr uint32_t MONTH_199401_END   = 19940131;

// Q1.3: weeknuminyear = 6 in 1994 (Feb 6-12, 1994)
constexpr uint32_t WEEK6_1994_START = 19940206;
constexpr uint32_t WEEK6_1994_END   = 19940212;

// Q3.4: December 1997
constexpr uint32_t DEC_1997_START = 19971201;
constexpr uint32_t DEC_1997_END   = 19971231;

// ============================================================================
// SSB Filter Constants (same as V12)
// ============================================================================
constexpr uint32_t REGION_AMERICA = 1;
constexpr uint32_t REGION_ASIA = 2;
constexpr uint32_t REGION_EUROPE = 3;

constexpr uint32_t NATION_US = 24;

constexpr uint32_t CATEGORY_MFGR12 = 12;
constexpr uint32_t CATEGORY_MFGR14 = 13;

constexpr uint32_t BRAND_MFGR2221 = 260;
constexpr uint32_t BRAND_MFGR2228 = 267;
constexpr uint32_t BRAND_MFGR2239 = 278;

constexpr uint32_t MFGR1 = 1;
constexpr uint32_t MFGR2 = 2;

} // namespace v15

// ============================================================================
// Hash Table Build Helper Functions (copied from V12)
// ============================================================================
namespace v15 {

using namespace l3_crystal;

inline void buildSupplierHashTable_Region(
    const uint32_t* d_suppkey, const uint32_t* d_region, int num_rows,
    uint32_t** ht, int* ht_len, uint32_t filter_region)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * sizeof(uint32_t));
    build_supplier_ht_region_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_suppkey, d_region, num_rows, filter_region, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildSupplierHashTable_Nation(
    const uint32_t* d_suppkey, const uint32_t* d_nation, const uint32_t* d_city,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_nation)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_supplier_ht_nation_city_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_suppkey, d_nation, d_city, num_rows, filter_nation, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildSupplierHashTable_Cities(
    const uint32_t* d_suppkey, const uint32_t* d_city, int num_rows,
    uint32_t** ht, int* ht_len, uint32_t city1, uint32_t city2)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_supplier_ht_city_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_suppkey, d_city, num_rows, city1, city2, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildSupplierHashTable_RegionNation(
    const uint32_t* d_suppkey, const uint32_t* d_region, const uint32_t* d_nation,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_region)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_supplier_ht_region_nation_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_suppkey, d_region, d_nation, num_rows, filter_region, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_Category(
    const uint32_t* d_partkey, const uint32_t* d_category, const uint32_t* d_brand1,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_category)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_part_ht_category_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_category, d_brand1, num_rows, filter_category, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_BrandRange(
    const uint32_t* d_partkey, const uint32_t* d_brand1,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t brand_min, uint32_t brand_max)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_part_ht_brand_range_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_brand1, num_rows, brand_min, brand_max, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_BrandExact(
    const uint32_t* d_partkey, const uint32_t* d_brand1,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t target_brand)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_part_ht_brand_exact_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_brand1, num_rows, target_brand, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_Mfgr(
    const uint32_t* d_partkey, const uint32_t* d_mfgr,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t mfgr1, uint32_t mfgr2)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * sizeof(uint32_t));
    build_part_ht_mfgr_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_mfgr, num_rows, mfgr1, mfgr2, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_MfgrCategory(
    const uint32_t* d_partkey, const uint32_t* d_mfgr, const uint32_t* d_category,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t mfgr1, uint32_t mfgr2)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_part_ht_mfgr_category_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_mfgr, d_category, num_rows, mfgr1, mfgr2, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildPartHashTable_CategoryBrand(
    const uint32_t* d_partkey, const uint32_t* d_category, const uint32_t* d_brand1,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_category)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_part_ht_category_brand_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_partkey, d_category, d_brand1, num_rows, filter_category, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildCustomerHashTable_Region(
    const uint32_t* d_custkey, const uint32_t* d_region, const uint32_t* d_nation,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_region)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_customer_ht_region_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_custkey, d_region, d_nation, num_rows, filter_region, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildCustomerHashTable_RegionKeys(
    const uint32_t* d_custkey, const uint32_t* d_region,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_region)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * sizeof(uint32_t));
    build_customer_ht_region_keys_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_custkey, d_region, num_rows, filter_region, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildCustomerHashTable_Nation(
    const uint32_t* d_custkey, const uint32_t* d_nation, const uint32_t* d_city,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_nation)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_customer_ht_nation_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_custkey, d_nation, d_city, num_rows, filter_nation, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildCustomerHashTable_Cities(
    const uint32_t* d_custkey, const uint32_t* d_city, int num_rows,
    uint32_t** ht, int* ht_len, uint32_t city1, uint32_t city2)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * 2 * sizeof(uint32_t));
    build_customer_ht_city_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_custkey, d_city, num_rows, city1, city2, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildDateHashTable_All(
    const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
    uint32_t** ht, int* ht_len)
{
    *ht_len = DATE_HT_LEN;
    cudaMalloc(ht, DATE_HT_LEN * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, DATE_HT_LEN * 2 * sizeof(uint32_t));
    build_date_ht_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_datekey, d_year, num_rows, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildDateHashTable_YearRange(
    const uint32_t* d_datekey, const uint32_t* d_year, int num_rows,
    uint32_t** ht, int* ht_len, uint32_t year_min, uint32_t year_max)
{
    *ht_len = DATE_HT_LEN;
    cudaMalloc(ht, DATE_HT_LEN * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, DATE_HT_LEN * 2 * sizeof(uint32_t));
    build_date_ht_year_range_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_datekey, d_year, num_rows, year_min, year_max, *ht, *ht_len);
    cudaDeviceSynchronize();
}

inline void buildDateHashTable_YearMonth(
    const uint32_t* d_datekey, const uint32_t* d_year,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_yearmonthnum)
{
    uint32_t year = filter_yearmonthnum / 100;
    uint32_t month = filter_yearmonthnum % 100;
    uint32_t start_date = year * 10000 + month * 100 + 1;
    uint32_t end_date = year * 10000 + month * 100 + 31;

    *ht_len = DATE_HT_LEN;
    cudaMalloc(ht, DATE_HT_LEN * 2 * sizeof(uint32_t));
    cudaMemset(*ht, 0, DATE_HT_LEN * 2 * sizeof(uint32_t));
    build_date_ht_yearmonth_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_datekey, d_year, num_rows, start_date, end_date, *ht, *ht_len);
    cudaDeviceSynchronize();
}

// Alias functions for backward compatibility
inline void buildPartHashTable_Brand(
    const uint32_t* d_partkey, const uint32_t* d_brand1,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t target_brand)
{
    buildPartHashTable_BrandExact(d_partkey, d_brand1, num_rows, ht, ht_len, target_brand);
}

inline void buildPartHashTable_MfgrRange(
    const uint32_t* d_partkey, const uint32_t* d_mfgr,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t mfgr1, uint32_t mfgr2)
{
    buildPartHashTable_Mfgr(d_partkey, d_mfgr, num_rows, ht, ht_len, mfgr1, mfgr2);
}

inline void buildSupplierHashTable_NationCity(
    const uint32_t* d_suppkey, const uint32_t* d_nation, const uint32_t* d_city,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_nation)
{
    buildSupplierHashTable_Nation(d_suppkey, d_nation, d_city, num_rows, ht, ht_len, filter_nation);
}

inline void buildCustomerHashTable_RegionOnly(
    const uint32_t* d_custkey, const uint32_t* d_region,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_region)
{
    buildCustomerHashTable_RegionKeys(d_custkey, d_region, num_rows, ht, ht_len, filter_region);
}

__global__ void build_customer_ht_nation_keys_crystal(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_nation,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_nation[idx] != filter_nation) return;

    uint32_t key = c_custkey[idx];
    int slot = key % ht_len;

    atomicCAS(&ht[slot], 0, key);
}

inline void buildCustomerHashTable_NationOnly(
    const uint32_t* d_custkey, const uint32_t* d_nation,
    int num_rows, uint32_t** ht, int* ht_len, uint32_t filter_nation)
{
    *ht_len = num_rows;
    cudaMalloc(ht, num_rows * sizeof(uint32_t));
    cudaMemset(*ht, 0, num_rows * sizeof(uint32_t));
    build_customer_ht_nation_keys_crystal<<<(num_rows + 255) / 256, 256>>>(
        d_custkey, d_nation, num_rows, filter_nation, *ht, *ht_len);
    cudaDeviceSynchronize();
}

} // namespace v15
