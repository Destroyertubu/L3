#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Partition metadata structure for optimized access
struct PartitionMetaOpt {
    int32_t start_idx;
    int32_t model_type;
    int32_t delta_bits;
    int32_t partition_len;
    double theta0;
    double theta1;
    int64_t bit_offset_base;
};

// Apply delta to predicted value
template<typename T>
__device__ __host__ inline T applyDelta(T predicted, long long delta) {
    if (std::is_signed<T>::value) {
        // For signed types, simple addition
        return predicted + static_cast<T>(delta);
    } else {
        // For unsigned types, cast to signed to handle negative deltas correctly
        return static_cast<T>(static_cast<int64_t>(predicted) + delta);
    }
}

// Optimized delta extraction with __ldg and branchless sign extension
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
            uint32_t sign_bit = extracted_bits >> (delta_bits - 1);
            uint32_t sign_mask = -sign_bit;  // All 1s if sign bit set, 0 otherwise
            uint32_t extend_mask = ~((1U << delta_bits) - 1U);
            extracted_bits |= (sign_mask & extend_mask);
        }

        return static_cast<long long>(static_cast<int32_t>(extracted_bits));
    } else {
        // Handle > 32 bit deltas with optimized loop
        uint64_t extracted_val_64 = 0;

        // First word
        uint32_t first_word = __ldg(&delta_array[word_idx]);
        int bits_from_first = 32 - bit_offset_in_word;
        extracted_val_64 = (first_word >> bit_offset_in_word);

        int remaining_bits = delta_bits - bits_from_first;
        int current_word = word_idx + 1;
        int bits_accumulated = bits_from_first;

        // Read subsequent words
        while (remaining_bits > 0) {
            uint32_t word = __ldg(&delta_array[current_word]);
            int bits_to_take = (remaining_bits < 32) ? remaining_bits : 32;
            uint64_t mask = (1ULL << bits_to_take) - 1ULL;
            extracted_val_64 |= ((word & mask) << bits_accumulated);

            bits_accumulated += bits_to_take;
            remaining_bits -= bits_to_take;
            current_word++;
        }

        // Sign extension for > 32 bit values
        if (delta_bits < 64) {
            uint64_t sign_bit = extracted_val_64 >> (delta_bits - 1);
            if (sign_bit) {
                uint64_t extend_mask = ~((1ULL << delta_bits) - 1ULL);
                extracted_val_64 |= extend_mask;
            }
        }

        return static_cast<long long>(extracted_val_64);
    }
}
