#!/usr/bin/env python3
"""
Check if the bits distribution varies across the dataset
"""

import numpy as np
import sys

def load_binary_uint64(filepath, max_elements=None):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint64)
    if max_elements:
        data = data[:max_elements]
    return data

def analyze_segment(data, segment_name, partition_size=1024):
    """Analyze compression potential for a data segment"""
    n = len(data)
    num_partitions = n // partition_size

    linear_bits_list = []
    poly2_bits_list = []

    for i in range(min(num_partitions, 10000)):
        start = i * partition_size
        end = start + partition_size
        partition = data[start:end]

        x = np.arange(partition_size, dtype=np.float64)
        y = partition.astype(np.float64)

        # Linear fit
        coeffs1 = np.polyfit(x, y, 1)
        pred1 = np.polyval(coeffs1, x)
        res1 = np.max(np.abs(y - pred1))
        linear_bits = int(np.ceil(np.log2(res1 + 1))) + 1 if res1 > 0 else 1
        linear_bits_list.append(linear_bits)

        # Quadratic fit
        coeffs2 = np.polyfit(x, y, 2)
        pred2 = np.polyval(coeffs2, x)
        res2 = np.max(np.abs(y - pred2))
        poly2_bits = int(np.ceil(np.log2(res2 + 1))) + 1 if res2 > 0 else 1
        poly2_bits_list.append(poly2_bits)

    avg_linear = np.mean(linear_bits_list)
    avg_poly2 = np.mean(poly2_bits_list)

    print(f"{segment_name}:")
    print(f"  Partitions: {len(linear_bits_list)}")
    print(f"  LINEAR avg bits: {avg_linear:.1f} -> {64/avg_linear:.2f}x")
    print(f"  POLY2 avg bits: {avg_poly2:.1f} -> {64/avg_poly2:.2f}x")
    print()

def main():
    data_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin"
    print(f"Loading: {data_file}")
    data = load_binary_uint64(data_file)
    n = len(data)

    print(f"Total elements: {n}\n")

    # Analyze different segments
    segment_size = 10_000_000  # 10M

    # First 10M
    analyze_segment(data[:segment_size], "First 10M")

    # Middle 10M
    mid_start = n // 2 - segment_size // 2
    analyze_segment(data[mid_start:mid_start + segment_size], "Middle 10M")

    # Last 10M
    analyze_segment(data[-segment_size:], "Last 10M")

    # Full dataset (sample)
    analyze_segment(data, "Full 200M (first 10k partitions)")

if __name__ == "__main__":
    main()
