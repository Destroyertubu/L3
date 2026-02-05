#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "gpu_ic/cuda_bp.cuh"
#include "gpu_ic/utils/cuda_utils.hpp"

namespace ssb {
namespace bp {

constexpr std::size_t kBlockSize = 32; // must match cuda_bp default

struct DeviceColumn {
    const uint32_t* offsets = nullptr; // (num_blocks + 1) entries, in 32-bit words
    const uint32_t* payload = nullptr; // packed payload as 32-bit words
};

struct EncodedColumn {
    std::size_t          n          = 0;
    std::size_t          num_blocks = 0;
    std::size_t          header_bytes = 0;
    std::vector<uint8_t> bytes;

    // Device storage (owns the allocation).
    uint8_t* d_bytes = nullptr;

    DeviceColumn device_view() const {
        DeviceColumn view;
        view.offsets = reinterpret_cast<const uint32_t*>(d_bytes);
        view.payload = reinterpret_cast<const uint32_t*>(d_bytes + header_bytes);
        return view;
    }

    void upload() {
        if (bytes.empty()) {
            throw std::runtime_error("EncodedColumn::upload: empty bytes");
        }
        if (d_bytes != nullptr) {
            return; // already uploaded
        }
        CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&d_bytes), bytes.size() * sizeof(uint8_t)));
        CUDA_CHECK_ERROR(cudaMemcpy(d_bytes, bytes.data(), bytes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));
    }

    void reset_device() {
        if (d_bytes != nullptr) {
            cudaFree(d_bytes);
            d_bytes = nullptr;
        }
    }
};

inline EncodedColumn encode_u32(const uint32_t* values, std::size_t n) {
    EncodedColumn col;
    col.n          = n;
    col.num_blocks = (n + (kBlockSize - 1)) / kBlockSize;
    col.header_bytes = 4 * (col.num_blocks + 1);

    // Worst-case payload is 32 bits/value => 4*n bytes, plus header.
    col.bytes.resize(col.header_bytes + n * sizeof(uint32_t));
    const std::size_t compressed_bytes = cuda_bp::encode<kBlockSize>(col.bytes.data(), values, n);
    col.bytes.resize(compressed_bytes);
    col.bytes.shrink_to_fit();
    return col;
}

} // namespace bp
} // namespace ssb
