#include <bench/gtsst-prepare.cuh>
#include <cassert>
namespace gtsst::bench {
    size_t check_buffer_required_length(const size_t data_len,
                                        const CompressionConfiguration& compression_configuration) {
        assert(compression_configuration.block_size % compression_configuration.min_alignment_input == 0);

        size_t additional_padding_padding = 0;

        // If we have to pad the block we need additional room, this will also automatically pad the alignment.
        // If we only have to pad the alignment, take that into account
        if (compression_configuration.must_pad_block) {
            const size_t block_remaining = data_len % compression_configuration.block_size;
            additional_padding_padding =
                (compression_configuration.block_size - block_remaining) % compression_configuration.block_size;
        } else if (compression_configuration.must_pad_alignment) {
            const size_t word_remaining = data_len % compression_configuration.min_alignment_input;
            additional_padding_padding =
                (compression_configuration.min_alignment_input - word_remaining) % compression_configuration.min_alignment_input;
        }

        return data_len + additional_padding_padding;
    }

    bool is_valid(const uint8_t* src, const size_t data_len,
                  const CompressionConfiguration& compression_configuration) {
        const bool pointer_aligned = (uintptr_t)src % compression_configuration.min_alignment_input == 0;
        const bool block_aligned =
            !compression_configuration.must_pad_block || data_len % compression_configuration.block_size == 0;
        const bool word_aligned = !compression_configuration.must_pad_alignment ||
            data_len % compression_configuration.min_alignment_input == 0;

        return pointer_aligned && block_aligned && word_aligned;
    }

    bool is_fixable(const uint8_t* src, const size_t data_len, const size_t buf_len,
                    const CompressionConfiguration& compression_configuration) {
        assert(compression_configuration.block_size % compression_configuration.min_alignment_input == 0);

        // Pointer alignment checks
        const bool pointer_aligned = (uintptr_t)src % compression_configuration.min_alignment_input == 0;

        // Block alignment checks
        const bool block_aligned = data_len % compression_configuration.block_size == 0;
        const bool block_fixable =
            !compression_configuration.must_pad_block || block_aligned || compression_configuration.padding_enabled;

        const bool word_aligned = data_len % compression_configuration.min_alignment_input == 0;
        const bool word_fixable =
            !compression_configuration.must_pad_alignment || word_aligned || compression_configuration.padding_enabled;

        // Padding budget check
        const bool enough_reserve = check_buffer_required_length(data_len, compression_configuration) <= buf_len;

        /*
         * In order to be fixable:
         *  - The base pointer needs to be aligned correctly
         *  - The data must already be block-aligned, or there must be enough space to pad it (and padding must be
         * enabled)
         */
        return pointer_aligned && block_fixable && word_fixable && enough_reserve;
    }

    void add_block_padding(uint8_t* src, const size_t data_len, const size_t buf_len, size_t* modified_len,
                           const size_t block_size, const uint8_t padding) {
        const size_t block_remaining = data_len % block_size;
        const size_t additional_block_padding = (block_size - block_remaining) % block_size;

        // Sanity check
        if (data_len + additional_block_padding > buf_len) {
            return;
        }

        for (int i = 0; i < additional_block_padding; i++) {
            src[data_len + i] = padding;
        }

        *modified_len += additional_block_padding;
    }

    bool fix_buffer(uint8_t* src, const size_t data_len, const size_t buf_len, size_t* modified_len,
                    const CompressionConfiguration& compression_configuration) {
        assert(compression_configuration.block_size % compression_configuration.min_alignment_input == 0);
        if (!is_fixable(src, data_len, buf_len, compression_configuration)) {
            return false;
        }

        // Keep track of active length
        size_t current_len = data_len;

        // First fix block level (which will also fix word alignment if needed)
        if (compression_configuration.must_pad_block) {
            add_block_padding(src, current_len, buf_len, &current_len, compression_configuration.block_size,
                              compression_configuration.padding_symbol);
        }

        // Then fix word level (if needed)
        if (compression_configuration.must_pad_alignment) {
            add_block_padding(src, current_len, buf_len, &current_len, compression_configuration.min_alignment_input,
                              compression_configuration.padding_symbol);
        }

        // Update length
        *modified_len = current_len;

        // Then check if the buffer is now valid
        return is_valid(src, current_len, compression_configuration);
    }

    size_t next_block_size(const size_t data_len, const size_t block_size) {
        return data_len % block_size == 0 ? data_len : (data_len / block_size + 1) * block_size;
    }

    uint8_t* recreate_buffer(const uint8_t* src, const size_t data_len,
                             const CompressionConfiguration& compression_configuration) {
        assert(compression_configuration.block_size % compression_configuration.min_alignment_input == 0);
        const size_t next_aligned_size = next_block_size(data_len, compression_configuration.block_size);
        auto new_buf = (uint8_t*)malloc(next_aligned_size);

        // Add padding
        memset(new_buf + data_len, compression_configuration.padding_symbol, next_aligned_size - data_len);
        memcpy(new_buf, src, data_len);

        return new_buf;
    }
} // namespace gtsst::bench
