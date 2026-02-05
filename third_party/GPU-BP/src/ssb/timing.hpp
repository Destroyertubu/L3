#pragma once

#include <cuda_runtime.h>

#include "gpu_ic/utils/cuda_utils.hpp"

namespace ssb {

class CudaEventTimer {
   public:
    CudaEventTimer() {
        CUDA_CHECK_ERROR(cudaEventCreate(&start_));
        CUDA_CHECK_ERROR(cudaEventCreate(&stop_));
    }

    ~CudaEventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) { CUDA_CHECK_ERROR(cudaEventRecord(start_, stream)); }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK_ERROR(cudaEventRecord(stop_, stream));
        CUDA_CHECK_ERROR(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }

   private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

} // namespace ssb

