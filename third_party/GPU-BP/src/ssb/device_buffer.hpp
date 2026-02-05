#pragma once

#include <cstddef>
#include <sstream>
#include <utility>

#include <cuda_runtime.h>

#include "gpu_ic/utils/cuda_utils.hpp"

namespace ssb {

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t size) { allocate(size); }

    ~DeviceBuffer() { reset(); }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        ptr_  = other.ptr_;
        size_ = other.size_;
        other.ptr_  = nullptr;
        other.size_ = 0;
        return *this;
    }

    void allocate(std::size_t size) {
        reset();
        size_ = size;
        CUDA_CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * size));
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = 0;
    }

    T*       data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

private:
    T*          ptr_  = nullptr;
    std::size_t size_ = 0;
};

} // namespace ssb

