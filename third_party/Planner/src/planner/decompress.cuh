#pragma once

#include <cstddef>
#include <cstdint>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "common/cuda_utils.hpp"
#include "common/device_buffer.hpp"
#include "planner/encoded_column.hpp"

namespace planner {

namespace detail {

__device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p, int byte_width) {
    if (byte_width == 1) return static_cast<uint32_t>(p[0]);
    if (byte_width == 2) return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8);
    if (byte_width == 3)
        return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16);
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

static __global__ void ns_decode_kernel(const uint8_t* in_bytes, int byte_width, int n, int* out_ints) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n) return;
    const uint8_t* p = in_bytes + static_cast<std::size_t>(idx) * static_cast<std::size_t>(byte_width);
    out_ints[idx] = static_cast<int>(load_u32_le(p, byte_width));
}

static __global__ void add_base_kernel(int n, int base, int* data) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n) return;
    data[idx] += base;
}

static __global__ void delta_add_first_kernel(int n, int first, const int* prefix_deltas, int* out) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n) return;
    out[idx] = first + prefix_deltas[idx];
}

static __global__ void set_first_kernel(int* out, int first) { out[0] = first; }

static __global__ void rle_scatter_boundaries_kernel(const int* boundary_pos, int runs, int* boundary_flags, int n) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= runs - 1) return; // last boundary_pos == n (out of range)
    const int pos = boundary_pos[idx];
    if (pos >= 0 && pos < n) {
        boundary_flags[pos] = 1;
    }
}

static __global__ void rle_gather_values_kernel(const int* run_values, const int* run_ids, int n, int* out) {
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (idx >= n) return;
    const int run_id = run_ids[idx];
    out[idx] = run_values[run_id];
}

} // namespace detail

struct DeltaWorkspace {
    DeviceBuffer<int> deltas;
    DeviceBuffer<uint8_t> scan_tmp;
    std::size_t scan_tmp_bytes = 0;
};

struct RleWorkspace {
    DeviceBuffer<int> boundary_pos;
    DeviceBuffer<int> boundary_flags;
    DeviceBuffer<uint8_t> scan_tmp1;
    DeviceBuffer<uint8_t> scan_tmp2;
    std::size_t scan_tmp1_bytes = 0;
    std::size_t scan_tmp2_bytes = 0;
};

inline void ns_decode_into(const EncodedColumn& col, int* out, cudaStream_t stream = 0) {
    const int threads = 256;
    const int blocks = (static_cast<int>(col.n) + threads - 1) / threads;
    detail::ns_decode_kernel<<<blocks, threads, 0, stream>>>(col.d_bytes_ptr(), col.byte_width, static_cast<int>(col.n), out);
    CUDA_CHECK(cudaGetLastError());
}

inline void for_ns_decode_into(const EncodedColumn& col, int* out, cudaStream_t stream = 0) {
    ns_decode_into(col, out, stream);
    const int threads = 256;
    const int blocks = (static_cast<int>(col.n) + threads - 1) / threads;
    detail::add_base_kernel<<<blocks, threads, 0, stream>>>(static_cast<int>(col.n), col.base, out);
    CUDA_CHECK(cudaGetLastError());
}

inline void delta_ns_decode_into(const EncodedColumn& col, int* out, DeltaWorkspace& ws, cudaStream_t stream = 0) {
    if (col.n == 0) return;
    if (col.n == 1) {
        CUDA_CHECK(cudaMemcpyAsync(out, &col.first, sizeof(int), cudaMemcpyHostToDevice, stream));
        return;
    }

    const std::size_t deltas_n = col.n - 1;
    ws.deltas.resize(deltas_n);
    {
        const int threads = 256;
        const int blocks = (static_cast<int>(deltas_n) + threads - 1) / threads;
        detail::ns_decode_kernel<<<blocks, threads, 0, stream>>>(
            col.d_bytes_ptr(), col.byte_width, static_cast<int>(deltas_n), ws.deltas.data());
        CUDA_CHECK(cudaGetLastError());
    }

    // inclusive scan on deltas -> prefix_deltas (in-place)
    std::size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, ws.deltas.data(), ws.deltas.data(), deltas_n, stream));
    if (tmp_bytes != ws.scan_tmp_bytes) {
        ws.scan_tmp_bytes = tmp_bytes;
        ws.scan_tmp.resize(tmp_bytes);
    }
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(ws.scan_tmp.data(), ws.scan_tmp_bytes, ws.deltas.data(), ws.deltas.data(), deltas_n, stream));

    // out[0] = first, out[i]=first+prefix_deltas[i-1]
    detail::set_first_kernel<<<1, 1, 0, stream>>>(out, col.first);
    CUDA_CHECK(cudaGetLastError());
    const int threads = 256;
    const int blocks = (static_cast<int>(deltas_n) + threads - 1) / threads;
    detail::delta_add_first_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int>(deltas_n), col.first, ws.deltas.data(), out + 1);
    CUDA_CHECK(cudaGetLastError());
}

// DELTA+FOR+NS: 3-layer cascading decompression
// Pass 1: NS decode packed bytes -> deltas (with FOR base removed)
// Pass 2: Add delta_base to get actual deltas
// Pass 3: Prefix sum on deltas
// Pass 4: Add first value
inline void delta_for_ns_decode_into(const EncodedColumn& col, int* out, DeltaWorkspace& ws, cudaStream_t stream = 0) {
    if (col.n == 0) return;
    if (col.n == 1) {
        CUDA_CHECK(cudaMemcpyAsync(out, &col.first, sizeof(int), cudaMemcpyHostToDevice, stream));
        return;
    }

    const std::size_t deltas_n = col.n - 1;
    ws.deltas.resize(deltas_n);

    // Pass 1: NS decode
    {
        const int threads = 256;
        const int blocks = (static_cast<int>(deltas_n) + threads - 1) / threads;
        detail::ns_decode_kernel<<<blocks, threads, 0, stream>>>(
            col.d_bytes_ptr(), col.byte_width, static_cast<int>(deltas_n), ws.deltas.data());
        CUDA_CHECK(cudaGetLastError());
    }

    // Pass 2: Add delta_base (FOR base on deltas)
    {
        const int threads = 256;
        const int blocks = (static_cast<int>(deltas_n) + threads - 1) / threads;
        detail::add_base_kernel<<<blocks, threads, 0, stream>>>(static_cast<int>(deltas_n), col.delta_base, ws.deltas.data());
        CUDA_CHECK(cudaGetLastError());
    }

    // Pass 3: Inclusive scan on deltas
    std::size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, tmp_bytes, ws.deltas.data(), ws.deltas.data(), deltas_n, stream));
    if (tmp_bytes != ws.scan_tmp_bytes) {
        ws.scan_tmp_bytes = tmp_bytes;
        ws.scan_tmp.resize(tmp_bytes);
    }
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(ws.scan_tmp.data(), ws.scan_tmp_bytes, ws.deltas.data(), ws.deltas.data(), deltas_n, stream));

    // Pass 4: out[0] = first, out[i] = first + prefix_deltas[i-1]
    detail::set_first_kernel<<<1, 1, 0, stream>>>(out, col.first);
    CUDA_CHECK(cudaGetLastError());
    const int threads = 256;
    const int blocks = (static_cast<int>(deltas_n) + threads - 1) / threads;
    detail::delta_add_first_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<int>(deltas_n), col.first, ws.deltas.data(), out + 1);
    CUDA_CHECK(cudaGetLastError());
}

inline void rle_decode_into(const EncodedColumn& col, int* out, RleWorkspace& ws, cudaStream_t stream = 0) {
    const int runs = static_cast<int>(col.rle_runs());
    if (col.n == 0 || runs == 0) return;

    ws.boundary_pos.resize(runs);
    ws.boundary_flags.resize(col.n);

    // Step 1: inclusive scan on run lengths => boundary_pos
    std::size_t tmp1 = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, tmp1, col.d_rle_lengths_ptr(), ws.boundary_pos.data(), runs, stream));
    if (tmp1 != ws.scan_tmp1_bytes) {
        ws.scan_tmp1_bytes = tmp1;
        ws.scan_tmp1.resize(tmp1);
    }
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(ws.scan_tmp1.data(),
                                            ws.scan_tmp1_bytes,
                                            col.d_rle_lengths_ptr(),
                                            ws.boundary_pos.data(),
                                            runs,
                                            stream));

    // Step 2: boundary_flags[boundary_pos[i]] = 1 for i in [0, runs-2]
    CUDA_CHECK(cudaMemsetAsync(ws.boundary_flags.data(), 0, col.n * sizeof(int), stream));
    {
        const int threads = 256;
        const int blocks = (runs + threads - 1) / threads;
        detail::rle_scatter_boundaries_kernel<<<blocks, threads, 0, stream>>>(
            ws.boundary_pos.data(), runs, ws.boundary_flags.data(), static_cast<int>(col.n));
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: inclusive scan on boundary_flags => run_ids (in-place)
    std::size_t tmp2 = 0;
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr,
                                            tmp2,
                                            ws.boundary_flags.data(),
                                            ws.boundary_flags.data(),
                                            col.n,
                                            stream));
    if (tmp2 != ws.scan_tmp2_bytes) {
        ws.scan_tmp2_bytes = tmp2;
        ws.scan_tmp2.resize(tmp2);
    }
    CUDA_CHECK(cub::DeviceScan::InclusiveSum(ws.scan_tmp2.data(),
                                            ws.scan_tmp2_bytes,
                                            ws.boundary_flags.data(),
                                            ws.boundary_flags.data(),
                                            col.n,
                                            stream));

    // Step 4: gather run_values[run_id] => out
    {
        const int threads = 256;
        const int blocks = (static_cast<int>(col.n) + threads - 1) / threads;
        detail::rle_gather_values_kernel<<<blocks, threads, 0, stream>>>(
            col.d_rle_values_ptr(), ws.boundary_flags.data(), static_cast<int>(col.n), out);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace planner
