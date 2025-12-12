/**
 * Detailed trace of CPU version's merge process
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#include "L3_format.hpp"
#include "encoder_cost_optimal.cuh"

// Need to access internal structures
// We'll manually trace what the kernel does

template<typename T>
std::vector<T> generateSortedData(size_t size) {
    std::vector<T> data(size);
    for (size_t i = 0; i < size; i++) {
        data[i] = static_cast<T>(i);
    }
    return data;
}

// Simulate the exact kernel logic
void tracePartitionCreation(int data_size, const CostOptimalConfig& config) {
    int analysis_block_size = config.analysis_block_size;
    int num_analysis_blocks = (data_size + analysis_block_size - 1) / analysis_block_size;

    std::cout << "\n=== Stage 1: Analysis blocks ===" << std::endl;
    std::cout << "num_analysis_blocks: " << num_analysis_blocks << std::endl;

    // For sorted data, delta_bits = 0 for all blocks
    std::vector<int> delta_bits(num_analysis_blocks, 0);

    std::cout << "\n=== Stage 2: Breakpoint detection ===" << std::endl;
    std::vector<int> breakpoint_positions;
    for (int i = 0; i < num_analysis_blocks; i++) {
        bool is_bp = (i == 0) || (abs(delta_bits[i] - delta_bits[i-1]) >= config.breakpoint_threshold);
        if (is_bp) {
            breakpoint_positions.push_back(i * analysis_block_size);
            std::cout << "  Block " << i << " is breakpoint at position " << (i * analysis_block_size) << std::endl;
        }
    }

    if (breakpoint_positions.empty()) {
        breakpoint_positions.push_back(0);
    }

    std::cout << "\n=== Stage 3: Partition creation ===" << std::endl;
    int num_breakpoints = breakpoint_positions.size();

    std::vector<int> partition_starts, partition_ends;

    for (int seg_idx = 0; seg_idx < num_breakpoints; seg_idx++) {
        int seg_start = breakpoint_positions[seg_idx];
        int seg_end = (seg_idx + 1 < num_breakpoints) ?
                      breakpoint_positions[seg_idx + 1] : data_size;
        int seg_len = seg_end - seg_start;

        std::cout << "  Segment " << seg_idx << ": [" << seg_start << ", " << seg_end
                  << ") len=" << seg_len << std::endl;

        if (seg_len <= 0) continue;

        // Same logic as kernel
        int part_size = config.target_partition_size;
        int num_parts = (seg_len + part_size - 1) / part_size;
        if (num_parts > 0) {
            part_size = (seg_len + num_parts - 1) / num_parts;
            part_size = ((part_size + 31) / 32) * 32;  // Align to warp
            part_size = std::max(config.min_partition_size,
                                std::min(config.max_partition_size, part_size));
        }

        std::cout << "    -> part_size=" << part_size << ", num_parts=" << num_parts << std::endl;

        for (int pos = seg_start; pos < seg_end; pos += part_size) {
            partition_starts.push_back(pos);
            partition_ends.push_back(std::min(pos + part_size, seg_end));
        }
    }

    int num_partitions = partition_starts.size();
    std::cout << "\nInitial partitions: " << num_partitions << std::endl;
    for (size_t i = 0; i < partition_starts.size(); i++) {
        std::cout << "  [" << i << "] [" << partition_starts[i] << ", " << partition_ends[i]
                  << ") size=" << (partition_ends[i] - partition_starts[i]) << std::endl;
    }

    // For sorted data, all costs are MODEL_OVERHEAD_BYTES (32 bytes) + 0 delta bytes
    std::vector<float> costs(num_partitions, 32.0f);  // MODEL_OVERHEAD_BYTES = 32

    std::cout << "\n=== Stage 4-5: Merge simulation ===" << std::endl;

    for (int round = 0; round < config.max_merge_rounds; round++) {
        std::cout << "\n--- Round " << round << " ---" << std::endl;

        // Evaluate merge benefits
        std::vector<float> merge_benefits(num_partitions, -1.0f);
        for (int i = 0; i < num_partitions - 1; i++) {
            int merged_size = partition_ends[i + 1] - partition_starts[i];
            if (merged_size > config.max_partition_size) {
                merge_benefits[i] = -1.0f;
                std::cout << "  Pair (" << i << "," << (i+1) << "): merged_size=" << merged_size
                          << " > max=" << config.max_partition_size << " -> CANNOT" << std::endl;
            } else {
                // For sorted data, delta_bits = 0, so merged cost = MODEL_OVERHEAD_BYTES
                float separate_cost = costs[i] + costs[i + 1];
                float merged_cost = 32.0f;  // MODEL_OVERHEAD_BYTES
                float benefit = (separate_cost - merged_cost) / separate_cost;
                merge_benefits[i] = benefit;
                std::cout << "  Pair (" << i << "," << (i+1) << "): merged_size=" << merged_size
                          << ", benefit=" << benefit << std::endl;
            }
        }

        // Mark merges - even phase
        std::vector<int> merge_flags(num_partitions, 0);
        std::cout << "\n  Even phase marking:" << std::endl;
        for (int idx = 0; idx * 2 < num_partitions - 1; idx++) {
            int pid = idx * 2;
            if (merge_benefits[pid] >= config.merge_benefit_threshold) {
                if (merge_flags[pid] == 0 && merge_flags[pid + 1] == 0) {
                    merge_flags[pid] = 1;
                    std::cout << "    Marked " << pid << " to merge with " << (pid+1) << std::endl;
                }
            }
        }

        // Mark merges - odd phase
        std::cout << "  Odd phase marking:" << std::endl;
        for (int idx = 0; idx * 2 + 1 < num_partitions - 1; idx++) {
            int pid = idx * 2 + 1;
            if (merge_benefits[pid] >= config.merge_benefit_threshold) {
                if (merge_flags[pid] == 0 && merge_flags[pid + 1] == 0) {
                    merge_flags[pid] = 1;
                    std::cout << "    Marked " << pid << " to merge with " << (pid+1) << std::endl;
                }
            }
        }

        // Count merges
        int merge_count = 0;
        for (int i = 0; i < num_partitions; i++) {
            if (merge_flags[i]) merge_count++;
        }

        if (merge_count == 0) {
            std::cout << "  No merges possible, stopping." << std::endl;
            break;
        }

        // Apply merges - CPU style with i++ skip
        std::cout << "\n  Applying merges:" << std::endl;
        std::vector<int> new_starts, new_ends;
        std::vector<float> new_costs;

        for (int i = 0; i < num_partitions; i++) {
            if (merge_flags[i] && i + 1 < num_partitions) {
                // Merge i with i+1
                new_starts.push_back(partition_starts[i]);
                new_ends.push_back(partition_ends[i + 1]);
                new_costs.push_back(32.0f);  // MODEL_OVERHEAD_BYTES for merged
                std::cout << "    Merged [" << partition_starts[i] << ", " << partition_ends[i + 1]
                          << ") size=" << (partition_ends[i + 1] - partition_starts[i]) << std::endl;
                i++;  // Skip next partition
            } else if (i > 0 && merge_flags[i - 1]) {
                // Already merged with previous, skip
                continue;
            } else {
                // Keep as is
                new_starts.push_back(partition_starts[i]);
                new_ends.push_back(partition_ends[i]);
                new_costs.push_back(costs[i]);
                std::cout << "    Kept [" << partition_starts[i] << ", " << partition_ends[i]
                          << ")" << std::endl;
            }
        }

        // Update
        partition_starts = new_starts;
        partition_ends = new_ends;
        costs = new_costs;
        num_partitions = partition_starts.size();

        std::cout << "\n  After merge: " << num_partitions << " partitions" << std::endl;
        for (size_t i = 0; i < partition_starts.size(); i++) {
            std::cout << "    [" << i << "] [" << partition_starts[i] << ", " << partition_ends[i]
                      << ") size=" << (partition_ends[i] - partition_starts[i]) << std::endl;
        }

        if (num_partitions <= 1) break;
    }

    std::cout << "\n=== Final Result ===" << std::endl;
    std::cout << num_partitions << " partitions:" << std::endl;
    for (size_t i = 0; i < partition_starts.size(); i++) {
        std::cout << "  [" << partition_starts[i] << ", " << partition_ends[i]
                  << ") size=" << (partition_ends[i] - partition_starts[i]) << std::endl;
    }
}

int main() {
    std::cout << "=== CPU Merge Process Trace ===" << std::endl;

    CostOptimalConfig config = CostOptimalConfig::balanced();

    std::cout << "Config:" << std::endl;
    std::cout << "  analysis_block_size: " << config.analysis_block_size << std::endl;
    std::cout << "  target_partition_size: " << config.target_partition_size << std::endl;
    std::cout << "  min_partition_size: " << config.min_partition_size << std::endl;
    std::cout << "  max_partition_size: " << config.max_partition_size << std::endl;
    std::cout << "  merge_benefit_threshold: " << config.merge_benefit_threshold << std::endl;
    std::cout << "  max_merge_rounds: " << config.max_merge_rounds << std::endl;

    tracePartitionCreation(10000, config);

    // Also run the actual CPU version
    std::cout << "\n\n=== Actual CPU Version Result ===" << std::endl;
    auto data = generateSortedData<uint32_t>(10000);
    auto result = createPartitionsCostOptimal(data, config, nullptr);
    std::cout << result.size() << " partitions:" << std::endl;
    for (const auto& p : result) {
        std::cout << "  [" << p.start_idx << ", " << p.end_idx
                  << ") size=" << (p.end_idx - p.start_idx) << std::endl;
    }

    return 0;
}
