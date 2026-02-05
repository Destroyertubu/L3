#ifndef GTSST_PREPARE_CUH
#define GTSST_PREPARE_CUH
#include <cstdint>
#include <gtsst/gtsst.cuh>

namespace gtsst::bench {
    inline size_t check_smaller_buffer(const size_t data_len,
                                       const CompressionConfiguration& compression_configuration) {
        return data_len / compression_configuration.block_size * compression_configuration.block_size;
    }

    size_t check_buffer_required_length(size_t data_len, const CompressionConfiguration& compression_configuration);

    bool is_valid(const uint8_t* src, size_t data_len, const CompressionConfiguration& compression_configuration);
    bool is_fixable(const uint8_t* src, size_t data_len, size_t buf_len,
                    const CompressionConfiguration& compression_configuration);

    bool fix_buffer(uint8_t* src, size_t data_len, size_t buf_len, size_t* modified_len,
                    const CompressionConfiguration& compression_configuration);
    uint8_t* recreate_buffer(const uint8_t* src, size_t data_len,
                             const CompressionConfiguration& compression_configuration);
} // namespace gtsst::bench

#endif // GTSST_PREPARE_CUH
