#!/usr/bin/env python3
"""
Dataset Distribution Analysis Script

Analyzes the distribution characteristics of SOSD datasets to understand
compression potential and identify optimization opportunities.
"""

import numpy as np
import sys
import os
from collections import Counter

def load_binary_uint64(filepath, max_elements=None):
    """Load binary uint64 data file"""
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint64)
    if max_elements:
        data = data[:max_elements]
    return data

def analyze_basic_stats(data, name):
    """Basic statistical analysis"""
    print(f"\n{'='*80}")
    print(f"Dataset: {name}")
    print(f"{'='*80}")

    n = len(data)
    print(f"\n[1] Basic Statistics")
    print(f"    Elements: {n:,}")
    print(f"    Size: {n * 8 / 1024 / 1024:.2f} MB")
    print(f"    Min value: {data.min()}")
    print(f"    Max value: {data.max()}")
    print(f"    Range: {data.max() - data.min()}")

    # Bit width for range
    range_val = data.max() - data.min()
    if range_val > 0:
        range_bits = int(np.ceil(np.log2(float(range_val) + 1)))
    else:
        range_bits = 0
    print(f"    Bits for range: {range_bits}")

    # Check if sorted
    is_sorted = np.all(data[:-1] <= data[1:])
    print(f"    Is sorted: {is_sorted}")

    return range_bits, is_sorted

def analyze_value_distribution(data):
    """Analyze the distribution of values"""
    print(f"\n[2] Value Distribution")

    n = len(data)

    # Convert to float for statistical analysis (may lose precision for very large values)
    data_f = data.astype(np.float64)

    print(f"    Mean: {np.mean(data_f):.2e}")
    print(f"    Std Dev: {np.std(data_f):.2e}")
    print(f"    Median: {np.median(data_f):.2e}")

    # Percentiles
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    print(f"\n    Percentiles:")
    for p in percentiles:
        val = np.percentile(data_f, p)
        print(f"      {p:3d}%: {val:.2e}")

    # Unique values
    unique_count = len(np.unique(data))
    print(f"\n    Unique values: {unique_count:,} ({100*unique_count/n:.2f}%)")

def analyze_delta_distribution(data):
    """Analyze differences between consecutive values"""
    print(f"\n[3] Delta (Consecutive Difference) Analysis")

    # Compute deltas (signed)
    deltas = np.diff(data.astype(np.int64))

    print(f"    Delta count: {len(deltas):,}")
    print(f"    Delta min: {deltas.min()}")
    print(f"    Delta max: {deltas.max()}")
    print(f"    Delta mean: {np.mean(deltas):.2e}")
    print(f"    Delta std: {np.std(deltas):.2e}")

    # Absolute deltas
    abs_deltas = np.abs(deltas)
    print(f"\n    Absolute delta stats:")
    print(f"      Max abs delta: {abs_deltas.max()}")
    print(f"      Mean abs delta: {np.mean(abs_deltas):.2e}")

    # Bits needed for deltas
    max_abs_delta = abs_deltas.max()
    if max_abs_delta > 0:
        delta_bits = int(np.ceil(np.log2(float(max_abs_delta) + 1))) + 1  # +1 for sign
    else:
        delta_bits = 1
    print(f"      Bits for max delta (signed): {delta_bits}")

    # Histogram of delta magnitudes
    print(f"\n    Delta magnitude distribution:")
    magnitude_bins = [0, 1, 10, 100, 1000, 10000, 100000, 1000000, 1e9, 1e12, 1e15, 1e18, float('inf')]
    for i in range(len(magnitude_bins) - 1):
        count = np.sum((abs_deltas >= magnitude_bins[i]) & (abs_deltas < magnitude_bins[i+1]))
        pct = 100 * count / len(deltas)
        if count > 0:
            print(f"      [{magnitude_bins[i]:.0e}, {magnitude_bins[i+1]:.0e}): {count:,} ({pct:.2f}%)")

    return delta_bits

def analyze_partitioned_compression(data, partition_sizes=[1024, 2048, 4096]):
    """Analyze compression potential with different partition sizes"""
    print(f"\n[4] Partition-based Compression Analysis")

    n = len(data)

    for ps in partition_sizes:
        print(f"\n    Partition size: {ps}")
        num_partitions = (n + ps - 1) // ps

        # Analyze FOR (Frame of Reference) compression
        for_bits_list = []
        linear_bits_list = []

        for i in range(min(num_partitions, 10000)):  # Analyze first 10000 partitions
            start = i * ps
            end = min(start + ps, n)
            partition = data[start:end]

            if len(partition) == 0:
                continue

            # FOR bits: log2(max - min)
            p_min = partition.min()
            p_max = partition.max()
            p_range = p_max - p_min
            if p_range > 0:
                for_bits = int(np.ceil(np.log2(float(p_range) + 1)))
            else:
                for_bits = 0
            for_bits_list.append(for_bits)

            # Linear regression bits
            x = np.arange(len(partition), dtype=np.float64)
            y = partition.astype(np.float64)

            # Fit linear model
            n_p = len(partition)
            sum_x = np.sum(x)
            sum_y = np.sum(y)
            sum_xx = np.sum(x * x)
            sum_xy = np.sum(x * y)

            denom = n_p * sum_xx - sum_x * sum_x
            if abs(denom) > 1e-10:
                slope = (n_p * sum_xy - sum_x * sum_y) / denom
                intercept = (sum_y - slope * sum_x) / n_p
            else:
                slope = 0
                intercept = sum_y / n_p

            # Compute residuals
            predictions = intercept + slope * x
            residuals = y - predictions
            max_residual = np.max(np.abs(residuals))

            if max_residual > 0:
                linear_bits = int(np.ceil(np.log2(max_residual + 1))) + 1  # +1 for sign
            else:
                linear_bits = 0
            linear_bits_list.append(linear_bits)

        avg_for_bits = np.mean(for_bits_list)
        avg_linear_bits = np.mean(linear_bits_list)

        # Theoretical compression ratios
        original_bits = 64
        for_ratio = original_bits / avg_for_bits if avg_for_bits > 0 else float('inf')
        linear_ratio = original_bits / avg_linear_bits if avg_linear_bits > 0 else float('inf')

        print(f"      Partitions analyzed: {len(for_bits_list)}")
        print(f"      FOR avg bits: {avg_for_bits:.1f} -> ratio: {for_ratio:.2f}x")
        print(f"      LINEAR avg bits: {avg_linear_bits:.1f} -> ratio: {linear_ratio:.2f}x")
        print(f"      Best theoretical: {max(for_ratio, linear_ratio):.2f}x")

        # Distribution of bits
        print(f"      FOR bits distribution:")
        for_bits_arr = np.array(for_bits_list)
        for bits_range in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 64)]:
            count = np.sum((for_bits_arr >= bits_range[0]) & (for_bits_arr < bits_range[1]))
            pct = 100 * count / len(for_bits_list)
            if count > 0:
                print(f"        [{bits_range[0]:2d}, {bits_range[1]:2d}): {count:,} ({pct:.1f}%)")

def analyze_sorted_data_specifics(data):
    """Special analysis for sorted data"""
    print(f"\n[5] Sorted Data Analysis")

    n = len(data)

    # Check monotonicity
    deltas = np.diff(data.astype(np.int64))

    increasing = np.sum(deltas > 0)
    decreasing = np.sum(deltas < 0)
    equal = np.sum(deltas == 0)

    print(f"    Increasing steps: {increasing:,} ({100*increasing/(n-1):.2f}%)")
    print(f"    Decreasing steps: {decreasing:,} ({100*decreasing/(n-1):.2f}%)")
    print(f"    Equal steps: {equal:,} ({100*equal/(n-1):.2f}%)")

    if increasing > decreasing:
        # Mostly increasing - analyze positive deltas
        pos_deltas = deltas[deltas > 0]
        if len(pos_deltas) > 0:
            print(f"\n    Positive delta analysis:")
            print(f"      Count: {len(pos_deltas):,}")
            print(f"      Min: {pos_deltas.min()}")
            print(f"      Max: {pos_deltas.max()}")
            print(f"      Mean: {np.mean(pos_deltas):.2e}")
            print(f"      Std: {np.std(pos_deltas):.2e}")

            # Bits for positive deltas
            max_pos = pos_deltas.max()
            bits_needed = int(np.ceil(np.log2(float(max_pos) + 1)))
            print(f"      Bits for max positive delta: {bits_needed}")

def analyze_entropy(data, sample_size=1000000):
    """Estimate entropy of the data"""
    print(f"\n[6] Entropy Analysis")

    # Sample for efficiency
    if len(data) > sample_size:
        indices = np.random.choice(len(data), sample_size, replace=False)
        sample = data[indices]
    else:
        sample = data

    # Byte-level entropy
    bytes_data = sample.tobytes()
    byte_counts = Counter(bytes_data)
    total_bytes = len(bytes_data)

    entropy = 0
    for count in byte_counts.values():
        p = count / total_bytes
        entropy -= p * np.log2(p)

    print(f"    Byte-level entropy: {entropy:.2f} bits/byte")
    print(f"    Theoretical compression: {8/entropy:.2f}x")

    # Delta entropy (for sorted data)
    if len(sample) > 1:
        deltas = np.diff(sample.astype(np.int64))
        unique_deltas = len(np.unique(deltas))
        print(f"    Unique delta values (in sample): {unique_deltas:,}")

def theoretical_compression_analysis(data):
    """Calculate theoretical best compression"""
    print(f"\n[7] Theoretical Compression Limits")

    n = len(data)
    original_size = n * 8  # bytes

    # Method 1: Global FOR
    global_range = data.max() - data.min()
    if global_range > 0:
        global_for_bits = int(np.ceil(np.log2(float(global_range) + 1)))
    else:
        global_for_bits = 0
    global_for_size = 8 + (n * global_for_bits) // 8  # base + packed deltas
    global_for_ratio = original_size / global_for_size

    print(f"    Global FOR:")
    print(f"      Range: {global_range}")
    print(f"      Bits per value: {global_for_bits}")
    print(f"      Compression ratio: {global_for_ratio:.2f}x")

    # Method 2: Delta encoding (for sorted data)
    deltas = np.diff(data.astype(np.int64))
    if len(deltas) > 0:
        # Check if all deltas are non-negative
        if np.all(deltas >= 0):
            max_delta = deltas.max()
            if max_delta > 0:
                delta_bits = int(np.ceil(np.log2(float(max_delta) + 1)))
            else:
                delta_bits = 0
            delta_size = 8 + (n * delta_bits) // 8  # first value + packed deltas
            delta_ratio = original_size / delta_size

            print(f"\n    Delta Encoding (unsigned):")
            print(f"      Max delta: {max_delta}")
            print(f"      Bits per delta: {delta_bits}")
            print(f"      Compression ratio: {delta_ratio:.2f}x")
        else:
            # Signed deltas
            max_abs_delta = np.max(np.abs(deltas))
            if max_abs_delta > 0:
                delta_bits = int(np.ceil(np.log2(float(max_abs_delta) + 1))) + 1
            else:
                delta_bits = 1
            delta_size = 8 + (n * delta_bits) // 8
            delta_ratio = original_size / delta_size

            print(f"\n    Delta Encoding (signed):")
            print(f"      Max abs delta: {max_abs_delta}")
            print(f"      Bits per delta: {delta_bits}")
            print(f"      Compression ratio: {delta_ratio:.2f}x")

def main():
    if len(sys.argv) < 2:
        # Default to normal dataset
        data_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin"
    else:
        data_file = sys.argv[1]

    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        return 1

    print(f"Loading: {data_file}")
    data = load_binary_uint64(data_file)

    name = os.path.basename(data_file)

    # Run all analyses
    range_bits, is_sorted = analyze_basic_stats(data, name)
    analyze_value_distribution(data)
    delta_bits = analyze_delta_distribution(data)
    analyze_partitioned_compression(data)

    if is_sorted:
        analyze_sorted_data_specifics(data)

    analyze_entropy(data)
    theoretical_compression_analysis(data)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: {name}")
    print(f"Is sorted: {is_sorted}")
    print(f"Global range bits: {range_bits}")
    print(f"Delta encoding bits: {delta_bits}")

    if is_sorted:
        print(f"\nFor sorted data, delta encoding should achieve:")
        print(f"  Theoretical ratio: {64/delta_bits:.2f}x")
        print(f"\nIf current L3 achieves only ~2x, check:")
        print(f"  1. Is the model fitting correct?")
        print(f"  2. Are residuals being computed correctly?")
        print(f"  3. Is FOR model being selected correctly for sorted data?")

    return 0

if __name__ == "__main__":
    sys.exit(main())
