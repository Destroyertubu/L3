#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "common/device_buffer.hpp"

namespace planner {

enum class Scheme {
    Uncompressed,
    NS,           // fixed-length byte-aligned null suppression
    FOR_NS,       // FOR (base) + NS on offsets
    DELTA_NS,     // DELTA + NS on deltas (requires non-decreasing input)
    DELTA_FOR_NS, // DELTA + FOR + NS (3-layer cascading, requires non-decreasing input)
    RLE,          // RLE (values + run lengths, both int32)
};

inline const char* scheme_name(Scheme s) {
    switch (s) {
        case Scheme::Uncompressed: return "UNCOMPRESSED";
        case Scheme::NS: return "NS";
        case Scheme::FOR_NS: return "FOR+NS";
        case Scheme::DELTA_NS: return "DELTA+NS";
        case Scheme::DELTA_FOR_NS: return "DELTA+FOR+NS";
        case Scheme::RLE: return "RLE";
    }
    return "UNKNOWN";
}

struct ColumnStats {
    int min_v = std::numeric_limits<int>::max();
    int max_v = std::numeric_limits<int>::min();
    bool non_decreasing = true;
    std::size_t runs = 0;
    int max_delta = 0;
};

inline int bytes_needed_u32(uint32_t v) {
    if (v <= 0xFFu) return 1;
    if (v <= 0xFFFFu) return 2;
    if (v <= 0xFFFFFFu) return 3;
    return 4;
}

inline ColumnStats compute_stats(const int* values, std::size_t n) {
    ColumnStats st;
    if (n == 0) {
        st.min_v = 0;
        st.max_v = 0;
        st.non_decreasing = true;
        st.runs = 0;
        st.max_delta = 0;
        return st;
    }
    st.min_v = values[0];
    st.max_v = values[0];
    st.runs = 1;
    int prev = values[0];
    for (std::size_t i = 1; i < n; ++i) {
        const int v = values[i];
        st.min_v = std::min(st.min_v, v);
        st.max_v = std::max(st.max_v, v);
        if (v < prev) {
            st.non_decreasing = false;
        } else {
            st.max_delta = std::max(st.max_delta, v - prev);
        }
        if (v != prev) {
            ++st.runs;
            prev = v;
        }
    }
    return st;
}

inline double avg_run_length(std::size_t n, std::size_t runs) {
    if (runs == 0) return 0.0;
    return static_cast<double>(n) / static_cast<double>(runs);
}

struct HostRle {
    std::vector<int> values;
    std::vector<int> lengths;
};

inline HostRle rle_encode(const int* values, std::size_t n) {
    HostRle out;
    if (n == 0) {
        return out;
    }
    out.values.reserve(n / 4 + 1);
    out.lengths.reserve(n / 4 + 1);

    int cur = values[0];
    int len = 1;
    for (std::size_t i = 1; i < n; ++i) {
        const int v = values[i];
        if (v == cur) {
            ++len;
        } else {
            out.values.push_back(cur);
            out.lengths.push_back(len);
            cur = v;
            len = 1;
        }
    }
    out.values.push_back(cur);
    out.lengths.push_back(len);
    return out;
}

inline void ns_pack_u32(const uint32_t* values, std::size_t n, int byte_width, std::vector<uint8_t>& out_bytes) {
    out_bytes.resize(n * static_cast<std::size_t>(byte_width));
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t v = values[i];
        uint8_t* dst = out_bytes.data() + i * static_cast<std::size_t>(byte_width);
        dst[0] = static_cast<uint8_t>(v & 0xFFu);
        if (byte_width >= 2) dst[1] = static_cast<uint8_t>((v >> 8) & 0xFFu);
        if (byte_width >= 3) dst[2] = static_cast<uint8_t>((v >> 16) & 0xFFu);
        if (byte_width >= 4) dst[3] = static_cast<uint8_t>((v >> 24) & 0xFFu);
    }
}

struct EncodedColumn {
    Scheme scheme = Scheme::Uncompressed;
    std::size_t n = 0;

    // NS / FOR+NS / DELTA+NS / DELTA+FOR+NS payload.
    int byte_width = 4;
    int base = 0;        // FOR base
    int first = 0;       // DELTA first value
    int delta_base = 0;  // FOR base on deltas (for DELTA+FOR+NS)
    std::vector<uint8_t> bytes;

    // RLE payload (values/lengths are int32).
    std::vector<int> rle_values;
    std::vector<int> rle_lengths;

    // Device copies.
    DeviceBuffer<uint8_t> d_bytes;
    DeviceBuffer<int> d_ints; // UNCOMPRESSED
    DeviceBuffer<int> d_rle_values;
    DeviceBuffer<int> d_rle_lengths;

    const uint8_t* d_bytes_ptr() const { return d_bytes.data(); }
    const int* d_ints_ptr() const { return d_ints.data(); }
    const int* d_rle_values_ptr() const { return d_rle_values.data(); }
    const int* d_rle_lengths_ptr() const { return d_rle_lengths.data(); }
    std::size_t rle_runs() const { return rle_values.size(); }

    void upload(const int* host_values) {
        if (scheme == Scheme::Uncompressed) {
            d_ints.resize(n);
            CUDA_CHECK(cudaMemcpy(d_ints.data(), host_values, n * sizeof(int), cudaMemcpyHostToDevice));
            return;
        }
        if (scheme == Scheme::RLE) {
            d_rle_values.resize(rle_values.size());
            d_rle_lengths.resize(rle_lengths.size());
            CUDA_CHECK(cudaMemcpy(d_rle_values.data(),
                                  rle_values.data(),
                                  rle_values.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_rle_lengths.data(),
                                  rle_lengths.data(),
                                  rle_lengths.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));
            return;
        }
        d_bytes.resize(bytes.size());
        CUDA_CHECK(cudaMemcpy(d_bytes.data(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
    }
};

inline EncodedColumn encode_column_planner(const int* values, std::size_t n) {
    EncodedColumn col;
    col.n = n;
    if (n == 0) {
        col.scheme = Scheme::Uncompressed;
        return col;
    }

    const ColumnStats st = compute_stats(values, n);

    const std::size_t size_uncompressed = n * sizeof(int);
    std::size_t best_size = size_uncompressed;
    Scheme best_scheme = Scheme::Uncompressed;

    const int bytes_value = bytes_needed_u32(static_cast<uint32_t>(st.max_v));
    const std::size_t size_ns = static_cast<std::size_t>(bytes_value) * n;
    if (size_ns < best_size) {
        best_size = size_ns;
        best_scheme = Scheme::NS;
    }

    const uint32_t range = static_cast<uint32_t>(st.max_v - st.min_v);
    const int bytes_range = bytes_needed_u32(range);
    const std::size_t size_for = sizeof(int) + static_cast<std::size_t>(bytes_range) * n;
    if (size_for < best_size) {
        best_size = size_for;
        best_scheme = Scheme::FOR_NS;
    }

    if (st.non_decreasing && n >= 2) {
        const int bytes_delta = bytes_needed_u32(static_cast<uint32_t>(st.max_delta));
        const std::size_t size_delta = sizeof(int) + static_cast<std::size_t>(bytes_delta) * (n - 1);
        if (size_delta < best_size) {
            best_size = size_delta;
            best_scheme = Scheme::DELTA_NS;
        }

        // DELTA+FOR+NS: compute min/max of deltas and apply FOR on deltas
        std::vector<int> deltas(n - 1);
        int min_delta = st.max_delta;
        int max_delta_v = 0;
        int prev = values[0];
        for (std::size_t i = 1; i < n; ++i) {
            const int d = values[i] - prev;
            deltas[i - 1] = d;
            min_delta = std::min(min_delta, d);
            max_delta_v = std::max(max_delta_v, d);
            prev = values[i];
        }
        const uint32_t delta_range = static_cast<uint32_t>(max_delta_v - min_delta);
        const int bytes_delta_for = bytes_needed_u32(delta_range);
        const std::size_t size_delta_for = 2 * sizeof(int) + static_cast<std::size_t>(bytes_delta_for) * (n - 1);
        if (size_delta_for < best_size) {
            best_size = size_delta_for;
            best_scheme = Scheme::DELTA_FOR_NS;
        }
    }

    const double arl = avg_run_length(n, st.runs);
    if (st.non_decreasing && arl >= 4.0) {
        const std::size_t runs = st.runs;
        const std::size_t size_rle = runs * sizeof(int) * 2;
        if (size_rle < best_size) {
            best_size = size_rle;
            best_scheme = Scheme::RLE;
        }
    }

    col.scheme = best_scheme;

    if (best_scheme == Scheme::RLE) {
        const auto rle = rle_encode(values, n);
        col.rle_values = rle.values;
        col.rle_lengths = rle.lengths;
        return col;
    }

    if (best_scheme == Scheme::NS) {
        col.byte_width = bytes_value;
        std::vector<uint32_t> tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            tmp[i] = static_cast<uint32_t>(values[i]);
        }
        ns_pack_u32(tmp.data(), n, col.byte_width, col.bytes);
    } else if (best_scheme == Scheme::FOR_NS) {
        col.base = st.min_v;
        col.byte_width = bytes_range;
        std::vector<uint32_t> tmp(n);
        for (std::size_t i = 0; i < n; ++i) {
            tmp[i] = static_cast<uint32_t>(values[i] - col.base);
        }
        ns_pack_u32(tmp.data(), n, col.byte_width, col.bytes);
    } else if (best_scheme == Scheme::DELTA_NS) {
        col.first = values[0];
        const int bytes_delta = bytes_needed_u32(static_cast<uint32_t>(st.max_delta));
        col.byte_width = bytes_delta;
        std::vector<uint32_t> tmp(n - 1);
        int prev = values[0];
        for (std::size_t i = 1; i < n; ++i) {
            const int v = values[i];
            tmp[i - 1] = static_cast<uint32_t>(v - prev);
            prev = v;
        }
        ns_pack_u32(tmp.data(), n - 1, col.byte_width, col.bytes);
    } else if (best_scheme == Scheme::DELTA_FOR_NS) {
        col.first = values[0];
        // Compute deltas and their min/max
        std::vector<int> deltas(n - 1);
        int min_delta = std::numeric_limits<int>::max();
        int max_delta_v = std::numeric_limits<int>::min();
        int prev = values[0];
        for (std::size_t i = 1; i < n; ++i) {
            const int d = values[i] - prev;
            deltas[i - 1] = d;
            min_delta = std::min(min_delta, d);
            max_delta_v = std::max(max_delta_v, d);
            prev = values[i];
        }
        col.delta_base = min_delta;
        const uint32_t delta_range = static_cast<uint32_t>(max_delta_v - min_delta);
        col.byte_width = bytes_needed_u32(delta_range);
        std::vector<uint32_t> tmp(n - 1);
        for (std::size_t i = 0; i < n - 1; ++i) {
            tmp[i] = static_cast<uint32_t>(deltas[i] - min_delta);
        }
        ns_pack_u32(tmp.data(), n - 1, col.byte_width, col.bytes);
    }

    return col;
}

} // namespace planner
