#ifndef BITPACK_UTILS_Vertical_CUH
#define BITPACK_UTILS_Vertical_CUH

/**
 * Vertical-Optimized Bit-Packing Utilities
 *
 * Key optimizations inspired by Vertical GPU paper:
 *
 * 1. BRANCHLESS UNPACKING:
 *    - Always read two words and stitch, eliminating boundary conditionals
 *    - Uniform execution path regardless of bit alignment
 *    - Reduces warp divergence to zero
 *
 * 2. REGISTER BUFFERING:
 *    - Prefetch multiple words into registers before extraction
 *    - Amortizes memory latency across multiple extractions
 *    - Reduces L1 cache pressure
 *
 * 3. WARP-COOPERATIVE LOADING:
 *    - 32 threads cooperatively load 32 consecutive words
 *    - Coalesced 128-byte cache line reads
 *    - Shared memory staging for random access patterns
 *
 * 4. TEMPLATE SPECIALIZATION:
 *    - Compile-time constants for common bit widths
 *    - Eliminates runtime branching and mask computation
 *
 * Platform: SM 8.0+ (Ampere and later)
 * Date: 2025-12-04
 */

#include <cuda_runtime.h>
#include <cstdint>
#include "L3_Vertical_format.hpp"

namespace Vertical {

// ============================================================================
// Constants
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int CACHE_LINE_BYTES = 128;
constexpr int CACHE_LINE_WORDS = 32;  // 128 bytes / 4 bytes per word

// ============================================================================
// Compile-Time Mask Generation
// ============================================================================

template<int K>
__device__ __host__ __forceinline__ constexpr uint64_t mask64() {
    static_assert(K >= 0 && K <= 64, "Bit width out of range");
    if constexpr (K == 0) return 0ULL;
    if constexpr (K == 64) return ~0ULL;
    return (1ULL << K) - 1ULL;
}

template<int K>
__device__ __host__ __forceinline__ constexpr uint32_t mask32() {
    static_assert(K >= 0 && K <= 32, "Bit width out of range");
    if constexpr (K == 0) return 0U;
    if constexpr (K == 32) return ~0U;
    return (1U << K) - 1U;
}

// Runtime mask (host + device for flexibility)
__device__ __host__ __forceinline__ uint64_t mask64_rt(int k) {
    if (k <= 0) return 0ULL;
    if (k >= 64) return ~0ULL;
    return (1ULL << k) - 1ULL;
}

__device__ __host__ __forceinline__ uint32_t mask32_rt(int k) {
    if (k <= 0) return 0U;
    if (k >= 32) return ~0U;
    return (1U << k) - 1U;
}

// ============================================================================
// SIGN EXTENSION (Forward declarations for use before full definition)
// ============================================================================

__device__ __forceinline__
int32_t sign_extend_32(uint32_t value, int bit_width);

__device__ __forceinline__
int64_t sign_extend_64(uint64_t value, int bit_width);

// ============================================================================
// BRANCHLESS BIT EXTRACTION (Core Innovation)
// ============================================================================

/**
 * Branchless 64-bit extraction from 128-bit window
 *
 * ALGORITHM:
 * 1. Load two consecutive 64-bit words (forming 128-bit window)
 * 2. Always perform shift+OR (no boundary check)
 * 3. Apply mask
 *
 * KEY INSIGHT: Even when value doesn't cross word boundary,
 * the OR with shifted high word is harmless (just ORs zeros)
 *
 * PERFORMANCE: Eliminates 1 branch per extraction
 *
 * @param words     Base pointer to uint32_t array
 * @param start_bit Starting bit position
 * @return Extracted value (0-64 bits)
 */
template<int BITS>
__device__ __forceinline__ uint64_t extract_branchless_64(
    const uint32_t* __restrict__ words,
    uint64_t start_bit)
{
    static_assert(BITS >= 0 && BITS <= 64, "BITS out of range [0, 64]");

    if constexpr (BITS == 0) {
        return 0ULL;
    }

    // Compute 64-bit word index and bit offset within word
    const uint64_t word64_idx = start_bit >> 6;    // start_bit / 64
    const int bit_offset = start_bit & 63;         // start_bit % 64

    // Reinterpret as uint64_t array
    const uint64_t* __restrict__ p64 = reinterpret_cast<const uint64_t*>(words);

    // Load 128-bit window (ALWAYS load both words - branchless)
    const uint64_t lo = __ldg(&p64[word64_idx]);
    const uint64_t hi = __ldg(&p64[word64_idx + 1]);

    // BRANCHLESS stitch: shift and combine
    // When bit_offset == 0: (lo >> 0) | (hi << 64) = lo | 0 = lo (correct)
    // When bit_offset > 0:  (lo >> offset) | (hi << (64 - offset)) (correct)
    //
    // Note: (hi << 64) is undefined in C++, but CUDA handles it as 0
    // We use conditional to be safe, but it's a compile-time constant
    uint64_t result;
    if constexpr (BITS <= 32) {
        // For small widths, optimize with 32-bit operations
        const uint32_t* p32 = words;
        const uint32_t word32_idx = start_bit >> 5;
        const int bit32_offset = start_bit & 31;

        const uint32_t w0 = __ldg(&p32[word32_idx]);
        const uint32_t w1 = __ldg(&p32[word32_idx + 1]);

        // Branchless: combine and shift
        const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
        result = (combined >> bit32_offset) & mask64<BITS>();
    } else {
        // For wider values, use 64-bit path
        const uint64_t shifted_lo = lo >> bit_offset;
        const uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));
        result = (shifted_lo | shifted_hi) & mask64<BITS>();
    }

    return result;
}

/**
 * Branchless extraction with runtime bit width
 */
__device__ __forceinline__ uint64_t extract_branchless_64_rt(
    const uint32_t* __restrict__ words,
    uint64_t start_bit,
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

/**
 * Branchless 32-bit extraction (for 1-32 bit values)
 *
 * Optimized path for common case where delta fits in 32 bits
 */
template<int BITS>
__device__ __forceinline__ uint32_t extract_branchless_32(
    const uint32_t* __restrict__ words,
    uint64_t start_bit)
{
    static_assert(BITS >= 0 && BITS <= 32, "BITS out of range [0, 32]");

    if constexpr (BITS == 0) {
        return 0U;
    }

    const uint32_t word_idx = start_bit >> 5;
    const int bit_offset = start_bit & 31;

    // Always load two words (branchless)
    const uint32_t w0 = __ldg(&words[word_idx]);
    const uint32_t w1 = __ldg(&words[word_idx + 1]);

    // Combine into 64-bit for easy extraction
    const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;

    // Extract and mask
    return static_cast<uint32_t>((combined >> bit_offset) & mask64<BITS>());
}

__device__ __forceinline__ uint32_t extract_branchless_32_rt(
    const uint32_t* __restrict__ words,
    uint64_t start_bit,
    int bits)
{
    if (bits <= 0) return 0U;
    if (bits > 32) bits = 32;

    const uint32_t word_idx = start_bit >> 5;
    const int bit_offset = start_bit & 31;

    const uint32_t w0 = __ldg(&words[word_idx]);
    const uint32_t w1 = __ldg(&words[word_idx + 1]);

    const uint64_t combined = (static_cast<uint64_t>(w1) << 32) | w0;
    return static_cast<uint32_t>((combined >> bit_offset) & mask32_rt(bits));
}

// ============================================================================
// REGISTER BUFFERING (Reduces Memory Transactions)
// ============================================================================

/**
 * Register Buffer for Sequential Extraction
 *
 * Prefetches BUFFER_SIZE words into registers, then extracts values
 * from the buffer. Reduces global memory accesses by factor of BUFFER_SIZE.
 *
 * USAGE:
 *   RegisterBuffer<4> buffer(words, start_word);
 *   for (int i = 0; i < N; i++) {
 *       uint64_t val = buffer.extract(bit_offset, bit_width);
 *       bit_offset += bit_width;
 *       buffer.advance_if_needed(bit_offset);
 *   }
 */
template<int BUFFER_SIZE = 4>
struct RegisterBuffer {
    static_assert(BUFFER_SIZE >= 2 && BUFFER_SIZE <= 8, "Buffer size must be 2-8");

    uint32_t regs[BUFFER_SIZE];
    const uint32_t* __restrict__ base_ptr;
    int64_t buffer_start_word;

    __device__ __forceinline__
    RegisterBuffer(const uint32_t* __restrict__ words, int64_t start_word)
        : base_ptr(words), buffer_start_word(start_word)
    {
        // Initial load
        #pragma unroll
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            regs[i] = __ldg(&base_ptr[start_word + i]);
        }
    }

    /**
     * Extract bits from buffer (branchless)
     */
    __device__ __forceinline__
    uint64_t extract(int64_t bit_offset, int bit_width) const {
        // Convert global bit offset to buffer-local
        int64_t local_bit = bit_offset - (buffer_start_word << 5);
        int word_in_buffer = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        // Safety check (should not happen with proper advance)
        if (word_in_buffer < 0 || word_in_buffer >= BUFFER_SIZE - 1) {
            return 0ULL;  // Or could reload
        }

        // Branchless extraction from buffer
        uint64_t combined = (static_cast<uint64_t>(regs[word_in_buffer + 1]) << 32) |
                           regs[word_in_buffer];
        return (combined >> bit_in_word) & mask64_rt(bit_width);
    }

    /**
     * Advance buffer if needed (when approaching end)
     */
    __device__ __forceinline__
    void advance_if_needed(int64_t current_bit_offset) {
        int64_t current_word = current_bit_offset >> 5;
        int64_t buffer_end_word = buffer_start_word + BUFFER_SIZE - 2;  // Need 2 words margin

        if (current_word >= buffer_end_word) {
            // Slide buffer forward
            int64_t new_start = current_word;
            buffer_start_word = new_start;

            #pragma unroll
            for (int i = 0; i < BUFFER_SIZE; ++i) {
                regs[i] = __ldg(&base_ptr[new_start + i]);
            }
        }
    }

    /**
     * Prefetch next chunk (for software pipelining)
     */
    __device__ __forceinline__
    void prefetch_next() const {
        #pragma unroll
        for (int i = 0; i < BUFFER_SIZE; ++i) {
            __ldg(&base_ptr[buffer_start_word + BUFFER_SIZE + i]);
        }
    }
};

// ============================================================================
// WARP-COOPERATIVE LOADING (Coalesced Access)
// ============================================================================

/**
 * Warp cooperatively loads N words into shared memory
 *
 * Each thread loads one word, achieving coalesced 128-byte reads
 */
template<int N_WORDS>
__device__ __forceinline__
void warp_load_to_shared(
    const uint32_t* __restrict__ global_data,
    int64_t global_word_offset,
    uint32_t* __restrict__ shared_buffer,
    int lane_id)
{
    static_assert(N_WORDS % WARP_SIZE == 0 || N_WORDS < WARP_SIZE,
                  "N_WORDS should be multiple of warp size for efficiency");

    constexpr int ITERATIONS = (N_WORDS + WARP_SIZE - 1) / WARP_SIZE;

    #pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        int shared_idx = iter * WARP_SIZE + lane_id;
        if (shared_idx < N_WORDS) {
            shared_buffer[shared_idx] = __ldg(&global_data[global_word_offset + shared_idx]);
        }
    }
}

/**
 * Warp cooperatively loads and extracts 32 values (one per thread)
 *
 * Optimized for warp-level decompression where each thread gets one value
 */
__device__ __forceinline__
int32_t warp_extract_one(
    const uint32_t* __restrict__ global_data,
    int64_t base_bit_offset,
    int bit_width,
    int lane_id,
    uint32_t* __restrict__ smem_buffer)  // Shared memory: >= (bit_width + 31) / 32 * 32 + 32 words
{
    // Calculate which words this warp needs
    int64_t min_bit = base_bit_offset;
    int64_t max_bit = base_bit_offset + (WARP_SIZE - 1) * bit_width + bit_width - 1;

    int64_t start_word = min_bit >> 5;
    int64_t end_word = (max_bit >> 5) + 1;
    int num_words = static_cast<int>(end_word - start_word + 1);

    // Cooperative load
    if (num_words <= 64) {  // Reasonable limit for shared memory
        warp_load_to_shared<64>(global_data, start_word, smem_buffer, lane_id);
        __syncwarp();

        // Each thread extracts its value
        int64_t my_bit = base_bit_offset + lane_id * bit_width;
        int64_t local_bit = my_bit - (start_word << 5);
        int word_in_buffer = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(smem_buffer[word_in_buffer + 1]) << 32) |
                           smem_buffer[word_in_buffer];
        uint32_t extracted = static_cast<uint32_t>((combined >> bit_in_word) & mask32_rt(bit_width));

        // Sign extend
        return sign_extend_32(extracted, bit_width);
    } else {
        // Fallback: direct global memory access
        int64_t my_bit = base_bit_offset + lane_id * bit_width;
        uint32_t extracted = extract_branchless_32_rt(global_data, my_bit, bit_width);
        return sign_extend_32(extracted, bit_width);
    }
}

// ============================================================================
// SIGN EXTENSION
// ============================================================================

/**
 * Sign extend extracted value (branchless)
 */
__device__ __forceinline__
int32_t sign_extend_32(uint32_t value, int bit_width) {
    if (bit_width >= 32) return static_cast<int32_t>(value);
    if (bit_width <= 0) return 0;

    // Branchless sign extension
    const uint32_t sign_bit = value >> (bit_width - 1);
    const uint32_t sign_mask = -sign_bit;  // All 1s if sign bit set, all 0s otherwise
    const uint32_t extend_mask = ~mask32_rt(bit_width);

    return static_cast<int32_t>(value | (sign_mask & extend_mask));
}

__device__ __forceinline__
int64_t sign_extend_64(uint64_t value, int bit_width) {
    if (bit_width >= 64) return static_cast<int64_t>(value);
    if (bit_width <= 0) return 0;

    const uint64_t sign_bit = value >> (bit_width - 1);
    const uint64_t sign_mask = -sign_bit;
    const uint64_t extend_mask = ~mask64_rt(bit_width);

    return static_cast<int64_t>(value | (sign_mask & extend_mask));
}

// Template version for compile-time width
template<int BITS>
__device__ __forceinline__
int64_t sign_extend_64_ct(uint64_t value) {
    static_assert(BITS >= 0 && BITS <= 64, "BITS out of range");

    if constexpr (BITS == 0) return 0;
    if constexpr (BITS == 64) return static_cast<int64_t>(value);
    if constexpr (BITS == 32) {
        return static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(value)));
    }

    const uint64_t sign_bit = value >> (BITS - 1);
    const uint64_t sign_mask = -sign_bit;
    const uint64_t extend_mask = ~mask64<BITS>();

    return static_cast<int64_t>(value | (sign_mask & extend_mask));
}

// ============================================================================
// INTERLEAVED EXTRACTION (Mini-Vector Format)
// ============================================================================

/**
 * Extract value from interleaved mini-vector format
 *
 * Layout: Values are distributed round-robin across 32 lanes
 * Lane L contains values at indices: L, L+32, L+64, L+96, L+128, L+160, L+192, L+224
 *
 * @param interleaved_data  Pointer to interleaved mini-vector data
 * @param mini_vector_base  Word offset to start of mini-vector
 * @param lane_id           Lane (0-31) to extract from
 * @param value_idx         Value index within lane (0-7)
 * @param bit_width         Bits per value
 */
__device__ __forceinline__
int64_t extract_interleaved_value(
    const uint32_t* __restrict__ interleaved_data,
    int64_t mini_vector_word_base,
    int lane_id,
    int value_idx,
    int bit_width)
{
    // Calculate bit offset within mini-vector
    // Each lane has 8 * bit_width bits
    // Lane L's data starts at bit: L * 8 * bit_width
    // Value V within lane at bit: L * 8 * bit_width + V * bit_width

    int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * bit_width;
    int64_t value_bit_offset = static_cast<int64_t>(value_idx) * bit_width;
    int64_t total_bit_offset = (mini_vector_word_base << 5) + lane_bit_start + value_bit_offset;

    // Branchless extraction
    uint64_t extracted = extract_branchless_64_rt(interleaved_data, total_bit_offset, bit_width);
    return sign_extend_64(extracted, bit_width);
}

/**
 * Warp extracts 32 consecutive values from interleaved mini-vector
 *
 * Each thread extracts one value, mapping:
 *   thread T extracts value at global index: mini_vector_idx * 256 + value_row * 32 + T
 *
 * @param interleaved_data  Pointer to interleaved data
 * @param mini_vector_base  Word offset to mini-vector start
 * @param value_row         Which row (0-7) within mini-vector
 * @param bit_width         Bits per value
 * @param lane_id           Thread's lane ID
 * @return Extracted and sign-extended value
 */
__device__ __forceinline__
int64_t warp_extract_interleaved_row(
    const uint32_t* __restrict__ interleaved_data,
    int64_t mini_vector_word_base,
    int value_row,
    int bit_width,
    int lane_id)
{
    // In interleaved format, all values in a row are in different lanes
    // Thread lane_id gets value at global index: row * 32 + lane_id
    // This maps to: lane = lane_id, value_idx = row

    return extract_interleaved_value(
        interleaved_data,
        mini_vector_word_base,
        lane_id,    // lane = thread's lane_id
        value_row,  // value_idx = row
        bit_width
    );
}

// ============================================================================
// BATCH EXTRACTION (8 Values per Thread)
// ============================================================================

/**
 * Thread extracts 8 values from interleaved mini-vector
 *
 * In interleaved format, thread with lane_id L owns values at global indices:
 *   L, L+32, L+64, L+96, L+128, L+160, L+192, L+224
 *
 * All 8 values are stored consecutively in the thread's lane data,
 * making extraction efficient.
 */
__device__ __forceinline__
void thread_extract_8_interleaved(
    const uint32_t* __restrict__ interleaved_data,
    int64_t mini_vector_word_base,
    int bit_width,
    int lane_id,
    int64_t output[8])
{
    // Lane L's data starts at bit: L * 8 * bit_width
    int64_t lane_bit_start = (mini_vector_word_base << 5) +
                            static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * bit_width;

    // Use register buffer for 8 sequential extractions from lane data
    // Calculate words needed: (8 * bit_width + 31) / 32
    int bits_per_lane = VALUES_PER_THREAD * bit_width;
    int64_t lane_word_start = lane_bit_start >> 5;

    // Load lane data into registers (branchless)
    uint32_t lane_words[4];  // Enough for 8 * 16 = 128 bits max (typically)

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        lane_words[i] = __ldg(&interleaved_data[lane_word_start + i]);
    }

    // Extract 8 values
    int local_bit = lane_bit_start & 31;

    #pragma unroll
    for (int v = 0; v < VALUES_PER_THREAD; ++v) {
        int word_idx = local_bit >> 5;
        int bit_in_word = local_bit & 31;

        uint64_t combined = (static_cast<uint64_t>(lane_words[word_idx + 1]) << 32) |
                           lane_words[word_idx];
        uint64_t extracted = (combined >> bit_in_word) & mask64_rt(bit_width);
        output[v] = sign_extend_64(extracted, bit_width);

        local_bit += bit_width;
    }
}

// ============================================================================
// PACKING UTILITIES (For Encoder)
// ============================================================================

/**
 * Pack value into bit stream (branchless)
 */
__device__ __forceinline__
void pack_branchless_64(
    uint32_t* __restrict__ words,
    uint64_t start_bit,
    uint64_t value,
    int bits)
{
    if (bits <= 0) return;

    const uint64_t masked_value = value & mask64_rt(bits);
    const uint64_t word64_idx = start_bit >> 6;
    const int bit_offset = start_bit & 63;

    uint64_t* __restrict__ p64 = reinterpret_cast<uint64_t*>(words);

    // Atomic OR for thread safety (if needed)
    // For single-threaded encoder, direct assignment is faster
    atomicOr(reinterpret_cast<unsigned long long*>(&p64[word64_idx]),
             static_cast<unsigned long long>(masked_value << bit_offset));

    // Handle overflow to next word (branchless - only writes if needed)
    int overflow_bits = bit_offset + bits - 64;
    if (overflow_bits > 0) {
        atomicOr(reinterpret_cast<unsigned long long*>(&p64[word64_idx + 1]),
                 static_cast<unsigned long long>(masked_value >> (bits - overflow_bits)));
    }
}

/**
 * Host-side packing (non-atomic, for CPU encoder)
 */
__host__ inline
void pack_sequential_host(
    uint32_t* words,
    uint64_t start_bit,
    uint64_t value,
    int bits)
{
    if (bits <= 0) return;

    const uint64_t masked_value = value & ((bits == 64) ? ~0ULL : ((1ULL << bits) - 1));
    const uint32_t word_idx = start_bit >> 5;
    const int bit_offset = start_bit & 31;

    // Pack into 32-bit words
    words[word_idx] |= static_cast<uint32_t>(masked_value << bit_offset);

    int remaining = bits - (32 - bit_offset);
    if (remaining > 0) {
        words[word_idx + 1] |= static_cast<uint32_t>(masked_value >> (32 - bit_offset));

        if (remaining > 32) {
            words[word_idx + 2] |= static_cast<uint32_t>(masked_value >> (64 - bit_offset));
        }
    }
}

// ============================================================================
// INTERLEAVED PACKING (For Encoder)
// ============================================================================

/**
 * Pack value into interleaved mini-vector format
 *
 * @param interleaved_data  Output buffer
 * @param mini_vector_base  Word offset to mini-vector start
 * @param lane_id           Target lane (0-31)
 * @param value_idx         Value index within lane (0-7)
 * @param value             Value to pack
 * @param bit_width         Bits per value
 */
__host__ inline
void pack_interleaved_host(
    uint32_t* interleaved_data,
    int64_t mini_vector_word_base,
    int lane_id,
    int value_idx,
    uint64_t value,
    int bit_width)
{
    int64_t lane_bit_start = static_cast<int64_t>(lane_id) * VALUES_PER_THREAD * bit_width;
    int64_t value_bit_offset = static_cast<int64_t>(value_idx) * bit_width;
    int64_t total_bit_offset = (mini_vector_word_base << 5) + lane_bit_start + value_bit_offset;

    pack_sequential_host(interleaved_data, total_bit_offset, value, bit_width);
}

/**
 * Convert sequential layout to interleaved mini-vector layout (host)
 *
 * @param sequential_deltas   Input: sequential bit-packed deltas
 * @param seq_bit_offset      Starting bit offset in sequential
 * @param interleaved_deltas  Output: interleaved bit-packed deltas
 * @param int_word_offset     Word offset in interleaved output
 * @param num_values          Number of values (should be 256 for full mini-vector)
 * @param bit_width           Bits per value
 */
__host__ inline
void sequential_to_interleaved_host(
    const uint32_t* sequential_deltas,
    int64_t seq_bit_offset,
    uint32_t* interleaved_deltas,
    int64_t int_word_offset,
    int num_values,
    int bit_width)
{
    // Clear output region
    int64_t total_bits = static_cast<int64_t>(num_values) * bit_width;
    int64_t total_words = (total_bits + 31) / 32;
    for (int64_t i = 0; i < total_words; ++i) {
        interleaved_deltas[int_word_offset + i] = 0;
    }

    // Reorder values
    for (int i = 0; i < num_values; ++i) {
        // Extract from sequential
        uint64_t bit_pos = seq_bit_offset + static_cast<int64_t>(i) * bit_width;
        uint64_t word64_idx = bit_pos >> 6;
        int bit_offset = bit_pos & 63;

        const uint64_t* seq64 = reinterpret_cast<const uint64_t*>(sequential_deltas);
        uint64_t lo = seq64[word64_idx];
        uint64_t hi = seq64[word64_idx + 1];

        uint64_t shifted_lo = lo >> bit_offset;
        uint64_t shifted_hi = (bit_offset == 0) ? 0ULL : (hi << (64 - bit_offset));
        uint64_t value = (shifted_lo | shifted_hi) & mask64_rt(bit_width);

        // Calculate interleaved position
        int lane_id = i % LANES_PER_MINI_VECTOR;
        int value_idx = i / LANES_PER_MINI_VECTOR;

        // Pack into interleaved
        pack_interleaved_host(interleaved_deltas, int_word_offset, lane_id, value_idx, value, bit_width);
    }
}

}  // namespace Vertical

#endif // BITPACK_UTILS_Vertical_CUH
