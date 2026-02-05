#pragma once

#include <cstdint>

#include "gpu_ic/utils/cuda_utils.hpp" // extract()
#include "ssb/gpu_bp_column.hpp"

namespace ssb {
namespace bp {

__device__ __forceinline__ uint32_t decode_u32(const DeviceColumn& col, uint32_t row_idx) {
    constexpr uint32_t block_size = static_cast<uint32_t>(kBlockSize);

    const uint32_t block    = row_idx / block_size;
    const uint32_t in_block = row_idx - block * block_size;

    const uint32_t offset0 = col.offsets[block];
    const uint32_t offset1 = col.offsets[block + 1];

    const uint32_t bit_size = (offset1 - offset0) * 32 / block_size;
    return extract(col.payload + offset0, static_cast<size_t>(in_block) * bit_size, bit_size);
}

} // namespace bp
} // namespace ssb

