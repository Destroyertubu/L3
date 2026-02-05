#pragma once

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "common/cuda_utils.hpp"

namespace planner {

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size) { resize(size); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept { *this = std::move(other); }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
        return *this;
    }

    ~DeviceBuffer() { reset(); }

    void resize(std::size_t size) {
        if (size == size_) {
            return;
        }
        reset();
        if (size == 0) {
            return;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), sizeof(T) * size));
        size_ = size;
    }

    void reset() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return size_; }

   private:
    T* ptr_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace planner

