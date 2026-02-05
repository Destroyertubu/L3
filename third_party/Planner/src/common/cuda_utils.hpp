#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace planner {

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) {
        return;
    }
    std::ostringstream oss;
    oss << "CUDA error: " << cudaGetErrorString(err) << " (" << static_cast<int>(err) << ")\n"
        << "  expr: " << expr << "\n"
        << "  at  : " << file << ":" << line;
    throw std::runtime_error(oss.str());
}

} // namespace planner

#define CUDA_CHECK(expr) ::planner::cuda_check((expr), #expr, __FILE__, __LINE__)

