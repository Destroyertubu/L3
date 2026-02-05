/**
 * LeCo (Learned Compression) CPU Implementation - Core Codec
 * Based on SIGMOD'24 paper - Cost-Optimal Encoder
 */

#include "benchmark_cpu_leco.h"
#include <iostream>
#include <cassert>
#include <cstring>

namespace leco {

// ============================================================================
// Bit Packing Implementation
// ============================================================================

namespace internal {

void packDelta(std::vector<uint8_t>& packed_data, int64_t& bit_offset,
               int64_t delta, int32_t delta_bits) {
    if (delta_bits == 0) return;

    // Encode sign in MSB: 1 = positive/zero, 0 = negative
    bool sign = (delta >= 0);
    uint64_t abs_delta = sign ? static_cast<uint64_t>(delta)
                              : static_cast<uint64_t>(-delta);

    // Value = abs_delta | (sign << (delta_bits - 1))
    uint64_t encoded = (abs_delta & ((1ULL << (delta_bits - 1)) - 1))
                     | (static_cast<uint64_t>(sign) << (delta_bits - 1));

    // Pack bits
    int64_t byte_idx = bit_offset / 8;
    int bit_in_byte = static_cast<int>(bit_offset % 8);

    // Ensure buffer is large enough
    int64_t bytes_needed = (bit_offset + delta_bits + 7) / 8;
    if (static_cast<int64_t>(packed_data.size()) < bytes_needed) {
        packed_data.resize(bytes_needed + 16, 0);  // Extra padding
    }

    // Write bits (little-endian)
    int bits_remaining = delta_bits;
    while (bits_remaining > 0) {
        int bits_to_write = std::min(8 - bit_in_byte, bits_remaining);
        uint8_t mask = ((1U << bits_to_write) - 1) << bit_in_byte;
        uint8_t bits = static_cast<uint8_t>((encoded & ((1ULL << bits_to_write) - 1))
                                            << bit_in_byte);
        packed_data[byte_idx] = (packed_data[byte_idx] & ~mask) | bits;

        encoded >>= bits_to_write;
        bits_remaining -= bits_to_write;
        bit_in_byte = 0;
        byte_idx++;
    }

    bit_offset += delta_bits;
}

int64_t unpackDelta(const std::vector<uint8_t>& packed_data, int64_t bit_offset,
                    int32_t delta_bits) {
    if (delta_bits == 0) return 0;

    int64_t byte_idx = bit_offset / 8;
    int bit_in_byte = static_cast<int>(bit_offset % 8);

    // Read enough bytes to cover the value
    uint64_t decoded = 0;
    int bits_read = 0;

    while (bits_read < delta_bits) {
        int bits_to_read = std::min(8 - bit_in_byte, delta_bits - bits_read);
        uint8_t byte_val = (byte_idx < static_cast<int64_t>(packed_data.size()))
                             ? packed_data[byte_idx] : 0;
        uint64_t bits = (byte_val >> bit_in_byte) & ((1U << bits_to_read) - 1);
        decoded |= (bits << bits_read);

        bits_read += bits_to_read;
        bit_in_byte = 0;
        byte_idx++;
    }

    // Extract sign (MSB of delta_bits)
    bool sign = (decoded >> (delta_bits - 1)) & 1;
    int64_t abs_val = static_cast<int64_t>(decoded & ((1ULL << (delta_bits - 1)) - 1));

    return sign ? abs_val : -abs_val;
}

// ============================================================================
// Linear Regression with Theta0 Centering
// ============================================================================

template<typename T>
void fitLinearModel(const T* data, int32_t start, int32_t length,
                    long double& theta0, long double& theta1, int32_t& delta_bits,
                    T& max_delta) {
    if (length <= 0) {
        theta0 = theta1 = 0;
        delta_bits = 0;
        max_delta = 0;
        return;
    }

    if (length == 1) {
        theta0 = static_cast<long double>(data[start]);
        theta1 = 0;
        delta_bits = 0;
        max_delta = 0;
        return;
    }

    // Step 1: Least squares regression using long double for precision
    // y = theta0 + theta1 * x where x = 0, 1, 2, ..., length-1
    long double sum_x = 0, sum_y = 0, sum_xx = 0, sum_xy = 0;
    for (int32_t i = 0; i < length; i++) {
        long double x = static_cast<long double>(i);
        long double y = static_cast<long double>(data[start + i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    long double n = static_cast<long double>(length);
    long double denom = n * sum_xx - sum_x * sum_x;

    long double ld_theta0, ld_theta1;
    if (std::abs(denom) < 1e-10L) {
        // Constant data
        ld_theta0 = sum_y / n;
        ld_theta1 = 0;
    } else {
        ld_theta1 = (n * sum_xy - sum_x * sum_y) / denom;
        ld_theta0 = (sum_y - ld_theta1 * sum_x) / n;
    }

    // Step 2: Center theta0 to minimize max absolute error
    // Find min/max error to center the predictions
    // Use __int128 for large uint64 values to avoid overflow
    __int128 max_error = std::numeric_limits<int64_t>::min();
    __int128 min_error = std::numeric_limits<int64_t>::max();

    for (int32_t i = 0; i < length; i++) {
        __int128 pred = static_cast<__int128>(std::llroundl(ld_theta0 + ld_theta1 * static_cast<long double>(i)));
        __int128 actual = static_cast<__int128>(data[start + i]);
        __int128 error = actual - pred;
        max_error = std::max(max_error, error);
        min_error = std::min(min_error, error);
    }

    // Center theta0
    ld_theta0 += static_cast<long double>(max_error + min_error) / 2.0L;

    // Step 3: Compute final delta bits after centering
    T final_max_delta = 0;
    for (int32_t i = 0; i < length; i++) {
        __int128 pred = static_cast<__int128>(std::llroundl(ld_theta0 + ld_theta1 * static_cast<long double>(i)));
        __int128 actual = static_cast<__int128>(data[start + i]);
        __int128 delta = actual - pred;
        __int128 abs_delta = delta >= 0 ? delta : -delta;
        if (abs_delta > static_cast<__int128>(final_max_delta)) {
            final_max_delta = static_cast<T>(abs_delta);
        }
    }

    max_delta = final_max_delta;
    if (final_max_delta == 0) {
        delta_bits = 0;
    } else {
        delta_bits = static_cast<int32_t>(bitsRequired(final_max_delta)) + 1;  // +1 for sign
    }

    // Cap at type size
    if (delta_bits >= static_cast<int32_t>(sizeof(T) * 8)) {
        delta_bits = static_cast<int32_t>(sizeof(T) * 8);
    }

    // Store as long double for precision
    theta0 = ld_theta0;
    theta1 = ld_theta1;
}

// ============================================================================
// Segment Size Calculation
// ============================================================================

template<typename T>
uint64_t calculateSegmentSize(const T* data, int32_t start, int32_t end) {
    int32_t length = end - start + 1;

    // Special cases
    if (length == 1) {
        // start_index(4) + marker(1) + value(sizeof(T))
        return 5 + sizeof(T);
    }
    if (length == 2) {
        // start_index(4) + marker(1) + 2*value(sizeof(T))
        return 5 + 2 * sizeof(T);
    }

    // General case: fit model and calculate
    long double theta0, theta1;
    int32_t delta_bits;
    T max_delta;
    fitLinearModel(data, start, length, theta0, theta1, delta_bits, max_delta);

    // If delta_bits >= sizeof(T)*8, store raw
    if (delta_bits >= static_cast<int32_t>(sizeof(T) * 8)) {
        return 5 + sizeof(T) * length;
    }

    // Overhead: start_index(4) + bits(1) + theta0(4) + theta1(4, optional)
    uint64_t overhead = 5 + 4;  // start_index + bits + theta0
    if (std::abs(theta1) >= 1e-8) {
        overhead += 4;  // theta1 (float)
    }

    // Delta storage
    uint64_t delta_size = static_cast<uint64_t>(std::ceil(delta_bits * length / 8.0));
    return overhead + delta_size;
}

// ============================================================================
// Cost-Optimal Partitioning (Greedy Merge Algorithm)
// ============================================================================

template<typename T>
std::vector<int32_t> costOptimalPartition(const T* data, size_t length,
                                           const LeCoConfig& config) {
    if (length <= 2) {
        return {0};
    }

    // Phase 1: Initial segmentation based on second-order deltas
    // Create doubly-linked list of segments

    MergeSegment head(0, 0, 0, 0, 0, 10000);
    MergeSegment tail(0, 0, 0, 0, 0, 0);
    MergeSegment* current = &head;

    int min_second_bit = 10000;
    int max_second_bit = -1;

    int64_t delta_prev = static_cast<int64_t>(data[1]) - static_cast<int64_t>(data[0]);

    std::vector<MergeSegment*> allocated_segments;

    for (size_t i = 1; i < length - 1; i++) {
        int64_t delta = static_cast<int64_t>(data[i + 1]) - static_cast<int64_t>(data[i]);
        int second_delta_bit = calcBitsForRange<T>(delta_prev, delta);

        min_second_bit = std::min(min_second_bit, second_delta_bit);
        max_second_bit = std::max(max_second_bit, second_delta_bit);

        MergeSegment* newseg = new MergeSegment(
            static_cast<int>(i - 1), static_cast<int>(i - 1),
            0, 0, delta_prev, second_delta_bit);
        allocated_segments.push_back(newseg);

        current->next = newseg;
        newseg->prev = current;
        current = newseg;
        delta_prev = delta;
    }

    // Add last segment
    MergeSegment* lastseg = new MergeSegment(
        static_cast<int>(length - 2), static_cast<int>(length - 2),
        0, 0, delta_prev, 10000);
    allocated_segments.push_back(lastseg);

    current->next = lastseg;
    lastseg->prev = current;
    current = lastseg;
    current->next = &tail;
    tail.prev = current;

    // Phase 2: Greedy merge ordered by second_delta_next
    for (int aim_bit = min_second_bit; aim_bit <= max_second_bit; aim_bit++) {
        current = head.next;
        while (current != &tail && current->next != &tail) {
            if (current->double_delta_next == aim_bit) {
                MergeSegment* next_seg = current->next;
                int former_index = current->start_index;
                int start_index = next_seg->start_index;
                int now_index = next_seg->end_index;

                int left_bit_origin = calcBitsForRange<T>(current->min_delta, current->max_delta);
                int right_bit_origin = calcBitsForRange<T>(next_seg->min_delta, next_seg->max_delta);

                int64_t new_max_delta = std::max(current->max_delta, next_seg->max_delta);
                new_max_delta = std::max(new_max_delta, current->next_delta);
                int64_t new_min_delta = std::min(current->min_delta, next_seg->min_delta);
                new_min_delta = std::min(new_min_delta, current->next_delta);

                int new_bit = calcBitsForRange<T>(new_min_delta, new_max_delta);

                int origin_cost = (start_index - former_index) * left_bit_origin +
                                  (now_index - start_index + 1) * right_bit_origin;
                int merged_cost = new_bit * (now_index - former_index + 1);

                if (merged_cost - origin_cost < config.overhead) {
                    // Merge
                    current->end_index = now_index;
                    current->next = next_seg->next;
                    next_seg->next->prev = current;
                    current->next_delta = next_seg->next_delta;
                    current->max_delta = new_max_delta;
                    current->min_delta = new_min_delta;
                    current->double_delta_next = next_seg->double_delta_next;

                    // Lookback: try merging with left neighbor
                    MergeSegment* prev_seg = current->prev;
                    while (prev_seg != &head && prev_seg->prev != &head) {
                        int left_index = prev_seg->start_index;
                        int64_t left_max_delta = std::max(prev_seg->max_delta, current->max_delta);
                        left_max_delta = std::max(left_max_delta, prev_seg->next_delta);
                        int64_t left_min_delta = std::min(prev_seg->min_delta, current->min_delta);
                        left_min_delta = std::min(left_min_delta, prev_seg->next_delta);

                        int new_bit_left = calcBitsForRange<T>(left_min_delta, left_max_delta);
                        int origin_left_bit = calcBitsForRange<T>(prev_seg->min_delta, prev_seg->max_delta);
                        int origin_right_bit = calcBitsForRange<T>(current->min_delta, current->max_delta);

                        int origin_cost_left = (current->start_index - left_index) * origin_left_bit +
                                               (current->end_index - current->start_index + 1) * origin_right_bit;
                        int merged_cost_left = new_bit_left * (current->end_index - left_index + 1);

                        if (merged_cost_left - origin_cost_left < config.overhead) {
                            current->start_index = left_index;
                            current->prev = prev_seg->prev;
                            prev_seg->prev->next = current;
                            current->min_delta = left_min_delta;
                            current->max_delta = left_max_delta;
                            prev_seg = current->prev;
                        } else {
                            break;
                        }
                    }

                    // Lookahead: try merging with right neighbor
                    next_seg = current->next;
                    while (next_seg != &tail && next_seg->next != &tail) {
                        int right_index = next_seg->end_index;
                        int64_t right_max_delta = std::max(next_seg->max_delta, current->max_delta);
                        right_max_delta = std::max(right_max_delta, current->next_delta);
                        int64_t right_min_delta = std::min(next_seg->min_delta, current->min_delta);
                        right_min_delta = std::min(right_min_delta, current->next_delta);

                        int new_bit_right = calcBitsForRange<T>(right_min_delta, right_max_delta);
                        int origin_left_bit = calcBitsForRange<T>(current->min_delta, current->max_delta);
                        int origin_right_bit = calcBitsForRange<T>(next_seg->min_delta, next_seg->max_delta);

                        int origin_cost_right = (right_index - next_seg->start_index + 1) * origin_right_bit +
                                                (next_seg->start_index - current->start_index) * origin_left_bit;
                        int merged_cost_right = new_bit_right * (right_index - current->start_index + 1);

                        if (merged_cost_right - origin_cost_right < config.overhead) {
                            current->end_index = right_index;
                            current->next = next_seg->next;
                            next_seg->next->prev = current;
                            current->max_delta = right_max_delta;
                            current->min_delta = right_min_delta;
                            current->double_delta_next = next_seg->double_delta_next;
                            current->next_delta = next_seg->next_delta;
                            next_seg = current->next;
                        } else {
                            break;
                        }
                    }

                    current = current->next;
                } else {
                    current = current->next;
                }
            } else {
                current = current->next;
            }
        }
    }

    // Extract segment indices
    std::vector<int32_t> segment_indices;
    current = head.next;
    while (current != &tail) {
        segment_indices.push_back(current->start_index);
        current = current->next;
    }

    // Calculate segment sizes
    segment_indices.push_back(static_cast<int32_t>(length));
    std::vector<uint64_t> segment_sizes;
    for (size_t i = 0; i < segment_indices.size() - 1; i++) {
        uint64_t size = calculateSegmentSize(data, segment_indices[i],
                                             segment_indices[i + 1] - 1);
        segment_sizes.push_back(size);
    }
    segment_indices.pop_back();

    // Phase 3: Iterative refinement (merge_both_direction)
    uint64_t total_byte = 0;
    for (auto s : segment_sizes) total_byte += s;

    int iter = 0;
    uint64_t cost_decline = total_byte;

    while (cost_decline > 0) {
        iter++;
        cost_decline = total_byte;

        // merge_both_direction
        std::vector<int32_t> new_segment_indices;
        std::vector<uint64_t> new_segment_sizes;
        uint64_t totalbyte_after_merge = 0;

        int32_t total_segments = static_cast<int32_t>(segment_indices.size());
        segment_indices.push_back(static_cast<int32_t>(length));

        new_segment_indices.push_back(segment_indices[0]);
        new_segment_sizes.push_back(segment_sizes[0]);
        totalbyte_after_merge += segment_sizes[0];

        int segment_num = 1;
        while (segment_num < total_segments) {
            if (segment_num == total_segments - 1) {
                // Only can try merging with former one
                int last_merged = static_cast<int>(new_segment_sizes.size()) - 1;
                uint64_t init_cost_front = segment_sizes[segment_num] + new_segment_sizes[last_merged];
                uint64_t merge_cost_front = calculateSegmentSize(
                    data, new_segment_indices[last_merged], segment_indices[segment_num + 1] - 1);
                int64_t saved_cost_front = static_cast<int64_t>(init_cost_front) -
                                           static_cast<int64_t>(merge_cost_front);

                if (saved_cost_front > 0) {
                    totalbyte_after_merge -= new_segment_sizes.back();
                    new_segment_sizes.back() = merge_cost_front;
                    totalbyte_after_merge += merge_cost_front;
                } else {
                    new_segment_indices.push_back(segment_indices[segment_num]);
                    new_segment_sizes.push_back(segment_sizes[segment_num]);
                    totalbyte_after_merge += segment_sizes[segment_num];
                }
                segment_num++;
                break;
            }

            int last_merged = static_cast<int>(new_segment_sizes.size()) - 1;

            // Cost of merging with front (previous)
            uint64_t init_cost_front = segment_sizes[segment_num] + new_segment_sizes[last_merged];
            uint64_t merge_cost_front = calculateSegmentSize(
                data, new_segment_indices[last_merged], segment_indices[segment_num + 1] - 1);
            int64_t saved_cost_front = static_cast<int64_t>(init_cost_front) -
                                       static_cast<int64_t>(merge_cost_front);

            // Cost of merging with back (next)
            uint64_t init_cost_back = segment_sizes[segment_num] + segment_sizes[segment_num + 1];
            uint64_t merge_cost_back = calculateSegmentSize(
                data, segment_indices[segment_num], segment_indices[segment_num + 2] - 1);
            int64_t saved_cost_back = static_cast<int64_t>(init_cost_back) -
                                      static_cast<int64_t>(merge_cost_back);

            int64_t saved_cost = std::max(saved_cost_front, saved_cost_back);

            if (saved_cost <= 0) {
                // Do not merge
                new_segment_indices.push_back(segment_indices[segment_num]);
                new_segment_sizes.push_back(segment_sizes[segment_num]);
                totalbyte_after_merge += segment_sizes[segment_num];
                segment_num++;
            } else if (saved_cost_back > saved_cost_front) {
                // Merge with back
                new_segment_indices.push_back(segment_indices[segment_num]);
                new_segment_sizes.push_back(merge_cost_back);
                totalbyte_after_merge += merge_cost_back;
                segment_num += 2;
            } else {
                // Merge with front
                totalbyte_after_merge -= new_segment_sizes.back();
                new_segment_sizes.back() = merge_cost_front;
                totalbyte_after_merge += merge_cost_front;
                segment_num++;
            }
        }

        total_byte = totalbyte_after_merge;
        segment_indices.pop_back();
        segment_indices = std::move(new_segment_indices);
        segment_sizes = std::move(new_segment_sizes);

        cost_decline = cost_decline - total_byte;
        double cost_decline_percent = static_cast<double>(cost_decline) * 100.0 /
                                      (sizeof(T) * length);
        if (cost_decline_percent < config.cost_decline_threshold * 100) {
            break;
        }
    }

    // Clean up allocated segments
    for (auto* seg : allocated_segments) {
        delete seg;
    }

    return segment_indices;
}

// Explicit instantiations for internal functions
template void fitLinearModel<uint32_t>(const uint32_t*, int32_t, int32_t,
                                        long double&, long double&, int32_t&, uint32_t&);
template void fitLinearModel<uint64_t>(const uint64_t*, int32_t, int32_t,
                                        long double&, long double&, int32_t&, uint64_t&);
template uint64_t calculateSegmentSize<uint32_t>(const uint32_t*, int32_t, int32_t);
template uint64_t calculateSegmentSize<uint64_t>(const uint64_t*, int32_t, int32_t);
template std::vector<int32_t> costOptimalPartition<uint32_t>(const uint32_t*, size_t, const LeCoConfig&);
template std::vector<int32_t> costOptimalPartition<uint64_t>(const uint64_t*, size_t, const LeCoConfig&);

}  // namespace internal

// ============================================================================
// Encoder Implementation
// ============================================================================

template<typename T>
LeCoCompressedBlock<T> lecoEncode(const T* data, size_t length,
                                   const LeCoConfig& config) {
    LeCoCompressedBlock<T> result;
    result.total_values = static_cast<int32_t>(length);
    result.original_bytes = static_cast<int64_t>(length * sizeof(T));

    if (length == 0) {
        result.num_segments = 0;
        result.total_bits = 0;
        result.compressed_bytes = 0;
        result.compression_ratio = 1.0;
        return result;
    }

    // Step 1: Get optimal partition
    std::vector<int32_t> partition = internal::costOptimalPartition(data, length, config);

    // Handle edge case: single segment
    if (partition.empty()) {
        partition.push_back(0);
    }

    // Add end marker
    partition.push_back(static_cast<int32_t>(length));

    result.num_segments = static_cast<int32_t>(partition.size() - 1);
    result.segments.resize(result.num_segments);

    // Step 2: Calculate total bits and prepare segments
    int64_t current_bit_offset = 0;
    int64_t total_metadata_bytes = 0;

    for (int32_t i = 0; i < result.num_segments; i++) {
        int32_t start = partition[i];
        int32_t end = partition[i + 1] - 1;
        int32_t seg_length = end - start + 1;

        LeCoSegment<T>& seg = result.segments[i];
        seg.start_idx = start;
        seg.end_idx = partition[i + 1];

        if (seg_length == 1) {
            // Single point: store directly
            seg.theta0 = static_cast<long double>(data[start]);
            seg.theta1 = 0;
            seg.delta_bits = 255;  // Marker for single point
            seg.bit_offset = current_bit_offset;
            total_metadata_bytes += 5 + sizeof(T);
        } else if (seg_length == 2) {
            // Two points: store directly
            seg.theta0 = static_cast<long double>(data[start]);
            seg.theta1 = static_cast<long double>(data[start + 1]);  // Store second value here
            seg.delta_bits = 254;  // Marker for two points
            seg.bit_offset = current_bit_offset;
            total_metadata_bytes += 5 + 2 * sizeof(T);
        } else {
            // General case: linear regression
            T max_delta;
            internal::fitLinearModel(data, start, seg_length,
                                     seg.theta0, seg.theta1, seg.delta_bits, max_delta);
            seg.bit_offset = current_bit_offset;

            if (seg.delta_bits >= static_cast<int32_t>(sizeof(T) * 8)) {
                // Store raw
                seg.delta_bits = static_cast<int32_t>(sizeof(T) * 8);
                total_metadata_bytes += 5 + sizeof(T) * seg_length;
            } else {
                // Overhead: start_index(4) + bits(1) + theta0(4) + theta1(4, optional)
                total_metadata_bytes += 5 + 4;
                if (std::abs(seg.theta1) >= 1e-8) {
                    total_metadata_bytes += 4;
                }
                current_bit_offset += static_cast<int64_t>(seg.delta_bits) * seg_length;
            }
        }
    }

    result.total_bits = current_bit_offset;

    // Step 3: Allocate and pack deltas
    int64_t delta_bytes = (current_bit_offset + 7) / 8 + 16;  // Extra padding
    result.packed_data.resize(delta_bytes, 0);

    current_bit_offset = 0;
    for (int32_t i = 0; i < result.num_segments; i++) {
        const LeCoSegment<T>& seg = result.segments[i];
        int32_t start = seg.start_idx;
        int32_t end = seg.end_idx;
        int32_t seg_length = end - start;

        if (seg.delta_bits == 255 || seg.delta_bits == 254 ||
            seg.delta_bits >= static_cast<int32_t>(sizeof(T) * 8)) {
            // Raw storage handled separately
            continue;
        }

        if (seg.delta_bits == 0) {
            continue;  // Perfect linear fit, no deltas needed
        }

        // Pack deltas
        for (int32_t j = 0; j < seg_length; j++) {
            __int128 pred = static_cast<__int128>(
                std::llroundl(seg.theta0 + seg.theta1 * static_cast<long double>(j)));
            __int128 actual = static_cast<__int128>(data[start + j]);
            __int128 delta = actual - pred;
            internal::packDelta(result.packed_data, current_bit_offset,
                               static_cast<int64_t>(delta), seg.delta_bits);
        }
    }

    // Calculate compressed size
    result.compressed_bytes = total_metadata_bytes + (result.total_bits + 7) / 8;
    result.compression_ratio = static_cast<double>(result.original_bytes) /
                                static_cast<double>(result.compressed_bytes);

    return result;
}

template<typename T>
LeCoCompressedBlock<T> lecoEncode(const std::vector<T>& data, const LeCoConfig& config) {
    return lecoEncode(data.data(), data.size(), config);
}

// ============================================================================
// Decoder Implementation
// ============================================================================

template<typename T>
std::vector<T> lecoDecode(const LeCoCompressedBlock<T>& compressed) {
    std::vector<T> result(compressed.total_values);

    for (int32_t i = 0; i < compressed.num_segments; i++) {
        const LeCoSegment<T>& seg = compressed.segments[i];
        int32_t start = seg.start_idx;
        int32_t end = seg.end_idx;
        int32_t seg_length = end - start;

        if (seg.delta_bits == 255) {
            // Single point
            result[start] = static_cast<T>(seg.theta0);
        } else if (seg.delta_bits == 254) {
            // Two points
            result[start] = static_cast<T>(seg.theta0);
            result[start + 1] = static_cast<T>(seg.theta1);
        } else if (seg.delta_bits == 0) {
            // Perfect fit
            for (int32_t j = 0; j < seg_length; j++) {
                result[start + j] = static_cast<T>(
                    std::llroundl(seg.theta0 + seg.theta1 * static_cast<long double>(j)));
            }
        } else {
            // General case with deltas
            int64_t bit_offset = seg.bit_offset;
            for (int32_t j = 0; j < seg_length; j++) {
                __int128 pred = static_cast<__int128>(
                    std::llroundl(seg.theta0 + seg.theta1 * static_cast<long double>(j)));
                int64_t delta = internal::unpackDelta(compressed.packed_data, bit_offset,
                                                       seg.delta_bits);
                result[start + j] = static_cast<T>(pred + delta);
                bit_offset += seg.delta_bits;
            }
        }
    }

    return result;
}

// ============================================================================
// Random Access Decoder
// ============================================================================

template<typename T>
T lecoDecodeAt(const LeCoCompressedBlock<T>& compressed, int32_t index) {
    // Binary search for segment
    int32_t left = 0;
    int32_t right = compressed.num_segments - 1;
    int32_t seg_idx = 0;

    while (left <= right) {
        int32_t mid = left + (right - left) / 2;
        if (compressed.segments[mid].start_idx <= index &&
            index < compressed.segments[mid].end_idx) {
            seg_idx = mid;
            break;
        } else if (compressed.segments[mid].start_idx > index) {
            right = mid - 1;
        } else {
            left = mid + 1;
            seg_idx = mid;
        }
    }

    const LeCoSegment<T>& seg = compressed.segments[seg_idx];
    int32_t local_idx = index - seg.start_idx;

    if (seg.delta_bits == 255) {
        return static_cast<T>(seg.theta0);
    } else if (seg.delta_bits == 254) {
        return local_idx == 0 ? static_cast<T>(seg.theta0) : static_cast<T>(seg.theta1);
    } else if (seg.delta_bits == 0) {
        return static_cast<T>(
            std::llroundl(seg.theta0 + seg.theta1 * static_cast<long double>(local_idx)));
    } else {
        int64_t bit_offset = seg.bit_offset + static_cast<int64_t>(local_idx) * seg.delta_bits;
        __int128 pred = static_cast<__int128>(
            std::llroundl(seg.theta0 + seg.theta1 * static_cast<long double>(local_idx)));
        int64_t delta = internal::unpackDelta(compressed.packed_data, bit_offset, seg.delta_bits);
        return static_cast<T>(pred + delta);
    }
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template LeCoCompressedBlock<uint32_t> lecoEncode(const uint32_t*, size_t, const LeCoConfig&);
template LeCoCompressedBlock<uint64_t> lecoEncode(const uint64_t*, size_t, const LeCoConfig&);
template LeCoCompressedBlock<uint32_t> lecoEncode(const std::vector<uint32_t>&, const LeCoConfig&);
template LeCoCompressedBlock<uint64_t> lecoEncode(const std::vector<uint64_t>&, const LeCoConfig&);
template std::vector<uint32_t> lecoDecode(const LeCoCompressedBlock<uint32_t>&);
template std::vector<uint64_t> lecoDecode(const LeCoCompressedBlock<uint64_t>&);
template uint32_t lecoDecodeAt(const LeCoCompressedBlock<uint32_t>&, int32_t);
template uint64_t lecoDecodeAt(const LeCoCompressedBlock<uint64_t>&, int32_t);

}  // namespace leco
