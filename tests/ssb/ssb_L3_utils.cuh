#ifndef SSB_L3_UTILS_CUH
#define SSB_L3_UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Partition metadata structure for optimized random access
struct PartitionMetaOpt {
    int32_t start_idx;
    int32_t model_type;
    int32_t delta_bits;
    int32_t partition_len;
    double theta0;
    double theta1;
    int64_t bit_offset_base;
};

// Helper template for applying delta to prediction
template<typename T>
__device__ __host__ inline T applyDelta(T predicted, long long delta) {
    // For unsigned types, use unsigned arithmetic to handle wraparound correctly
    if (sizeof(T) == 8) {
        // For 64-bit unsigned types
        unsigned long long pred_ull = static_cast<unsigned long long>(predicted);
        unsigned long long delta_ull = static_cast<unsigned long long>(delta);
        return static_cast<T>(pred_ull + delta_ull);
    } else {
        // For 32-bit and smaller types
        unsigned int pred_uint = static_cast<unsigned int>(predicted);
        unsigned int delta_uint = static_cast<unsigned int>(delta);
        return static_cast<T>(pred_uint + delta_uint);
    }
}

// Optimized delta extraction function with reduced branching
template<typename T>
__device__ inline long long extractDelta_Optimized(const uint32_t* __restrict__ delta_array,
                                                   int64_t bit_offset,
                                                   int delta_bits) {
    if (delta_bits <= 0) return 0;

    // Use bit operations instead of division/modulo
    int word_idx = bit_offset >> 5;  // bit_offset / 32
    int bit_offset_in_word = bit_offset & 31;  // bit_offset % 32

    if (delta_bits <= 32) {
        // Always read two words to avoid branching
        uint32_t w1 = __ldg(&delta_array[word_idx]);
        uint32_t w2 = __ldg(&delta_array[word_idx + 1]);

        // Combine words into 64-bit value for branchless extraction
        uint64_t combined = (static_cast<uint64_t>(w2) << 32) | static_cast<uint64_t>(w1);
        uint32_t extracted_bits = (combined >> bit_offset_in_word) & ((1U << delta_bits) - 1U);

        // Branchless sign extension
        if (delta_bits < 32) {
            uint32_t sign_mask = 1U << (delta_bits - 1);
            uint32_t sign_ext_mask = ~((1U << delta_bits) - 1U);
            int32_t sign_extended = (extracted_bits & sign_mask) ?
                                    static_cast<int32_t>(extracted_bits | sign_ext_mask) :
                                    static_cast<int32_t>(extracted_bits);
            return static_cast<long long>(sign_extended);
        } else {
            return static_cast<long long>(static_cast<int32_t>(extracted_bits));
        }
    } else {
        // Handle 33-64 bit deltas
        uint64_t extracted_val_64 = 0;
        int shift = 0;
        int bits_remaining = delta_bits;
        int offset_in_word = bit_offset_in_word;

        while (bits_remaining > 0 && shift < 64) {
            int bits_in_this_word = min(bits_remaining, 32 - offset_in_word);
            uint32_t mask = (bits_in_this_word == 32) ? ~0U : ((1U << bits_in_this_word) - 1U);
            uint32_t word_val = (__ldg(&delta_array[word_idx]) >> offset_in_word) & mask;
            extracted_val_64 |= (static_cast<uint64_t>(word_val) << shift);

            shift += bits_in_this_word;
            bits_remaining -= bits_in_this_word;
            word_idx++;
            offset_in_word = 0;
        }

        // Sign extension for 64-bit deltas
        if (delta_bits < 64) {
            uint64_t sign_mask_64 = 1ULL << (delta_bits - 1);
            if (extracted_val_64 & sign_mask_64) {
                uint64_t sign_ext_mask_64 = ~((1ULL << delta_bits) - 1ULL);
                return static_cast<long long>(extracted_val_64 | sign_ext_mask_64);
            } else {
                return static_cast<long long>(extracted_val_64);
            }
        } else {
            return static_cast<long long>(extracted_val_64);
        }
    }
}

#endif // SSB_L3_UTILS_CUH
