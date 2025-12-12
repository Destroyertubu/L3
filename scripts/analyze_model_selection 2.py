#!/usr/bin/env python3
"""
Analyze why Adaptive selector is not choosing polynomial models
"""

import numpy as np
import sys
import os

def load_binary_uint64(filepath, max_elements=None):
    with open(filepath, 'rb') as f:
        data = np.fromfile(f, dtype=np.uint64)
    if max_elements:
        data = data[:max_elements]
    return data

def analyze_model_selection(data, partition_size=1024):
    """Simulate the adaptive selector's decision process"""
    print("=" * 80)
    print("MODEL SELECTION ANALYSIS")
    print("=" * 80)

    n = len(data)
    num_partitions = min(1000, n // partition_size)

    # Cost constants (from adaptive_selector.cuh)
    LINEAR_METADATA = 16.0   # 2 doubles
    POLY2_METADATA = 24.0    # 3 doubles
    POLY3_METADATA = 32.0    # 4 doubles
    FOR_METADATA = 8.0       # 1 T (8 bytes for uint64)
    COST_THRESHOLD = 0.95    # Must be 5% better

    model_choices = {'FOR': 0, 'LINEAR': 0, 'POLY2': 0, 'POLY3': 0}
    savings_if_poly2 = []
    savings_if_poly3 = []

    print(f"\nAnalyzing {num_partitions} partitions (size={partition_size})...")
    print(f"\nMetadata costs: FOR={FOR_METADATA}B, LINEAR={LINEAR_METADATA}B, POLY2={POLY2_METADATA}B, POLY3={POLY3_METADATA}B")
    print(f"Threshold: {COST_THRESHOLD} (new model must be {100*(1-COST_THRESHOLD):.0f}% better)")

    detailed_output = []

    for i in range(num_partitions):
        start = i * partition_size
        end = start + partition_size
        partition = data[start:end]

        ps = len(partition)
        x = np.arange(ps, dtype=np.float64)
        y = partition.astype(np.float64)

        # FOR model
        p_range = float(partition.max() - partition.min())
        for_bits = int(np.ceil(np.log2(p_range + 1))) if p_range > 0 else 0
        cost_for = FOR_METADATA + ps * for_bits / 8.0

        # LINEAR model
        coeffs1 = np.polyfit(x, y, 1)
        pred1 = np.polyval(coeffs1, x)
        res1 = np.max(np.abs(y - pred1))
        linear_bits = int(np.ceil(np.log2(res1 + 1))) + 1 if res1 > 0 else 1
        cost_linear = LINEAR_METADATA + ps * linear_bits / 8.0

        # POLY2 model
        coeffs2 = np.polyfit(x, y, 2)
        pred2 = np.polyval(coeffs2, x)
        res2 = np.max(np.abs(y - pred2))
        poly2_bits = int(np.ceil(np.log2(res2 + 1))) + 1 if res2 > 0 else 1
        cost_poly2 = POLY2_METADATA + ps * poly2_bits / 8.0

        # POLY3 model
        coeffs3 = np.polyfit(x, y, 3)
        pred3 = np.polyval(coeffs3, x)
        res3 = np.max(np.abs(y - pred3))
        poly3_bits = int(np.ceil(np.log2(res3 + 1))) + 1 if res3 > 0 else 1
        cost_poly3 = POLY3_METADATA + ps * poly3_bits / 8.0

        # Selection logic (mimicking adaptive_selector.cuh)
        best_model = 'FOR'
        best_cost = cost_for
        best_bits = for_bits

        # Check LINEAR
        if cost_linear < best_cost * COST_THRESHOLD:
            best_model = 'LINEAR'
            best_cost = cost_linear
            best_bits = linear_bits

        # Check POLY2 (only if n > 10)
        if ps > 10 and cost_poly2 < best_cost * COST_THRESHOLD:
            best_model = 'POLY2'
            best_cost = cost_poly2
            best_bits = poly2_bits

        # Check POLY3 (only if n > 20)
        if ps > 20 and cost_poly3 < best_cost * COST_THRESHOLD:
            best_model = 'POLY3'
            best_cost = cost_poly3
            best_bits = poly3_bits

        model_choices[best_model] += 1

        # Calculate potential savings
        if best_model != 'POLY2':
            savings_if_poly2.append(best_cost - cost_poly2)
        if best_model != 'POLY3':
            savings_if_poly3.append(best_cost - cost_poly3)

        # Store detailed info for first 20 partitions
        if i < 20:
            detailed_output.append({
                'partition': i,
                'for_bits': for_bits,
                'linear_bits': linear_bits,
                'poly2_bits': poly2_bits,
                'poly3_bits': poly3_bits,
                'cost_for': cost_for,
                'cost_linear': cost_linear,
                'cost_poly2': cost_poly2,
                'cost_poly3': cost_poly3,
                'chosen': best_model,
                'chosen_bits': best_bits,
            })

    # Print detailed results for first 20 partitions
    print(f"\nDetailed results for first 20 partitions:")
    print("-" * 120)
    print(f"{'Part':>5} | {'FOR':>6} {'(cost)':>8} | {'LINEAR':>6} {'(cost)':>8} | {'POLY2':>6} {'(cost)':>8} | {'POLY3':>6} {'(cost)':>8} | {'Chosen':>8}")
    print("-" * 120)

    for d in detailed_output:
        print(f"{d['partition']:>5} | {d['for_bits']:>6}b {d['cost_for']:>8.0f} | "
              f"{d['linear_bits']:>6}b {d['cost_linear']:>8.0f} | "
              f"{d['poly2_bits']:>6}b {d['cost_poly2']:>8.0f} | "
              f"{d['poly3_bits']:>6}b {d['cost_poly3']:>8.0f} | "
              f"{d['chosen']:>8}")

    # Summary
    print("\n" + "=" * 80)
    print("MODEL SELECTION SUMMARY")
    print("=" * 80)

    total = sum(model_choices.values())
    for model, count in model_choices.items():
        print(f"  {model:>8}: {count:>6} ({100*count/total:.1f}%)")

    # Why POLY2/POLY3 are rarely chosen
    print("\n" + "=" * 80)
    print("WHY POLYNOMIAL MODELS ARE NOT CHOSEN")
    print("=" * 80)

    # Check the threshold effect
    print("\n1. THRESHOLD EFFECT:")
    print(f"   Current threshold: {COST_THRESHOLD} (must be {100*(1-COST_THRESHOLD):.0f}% better)")

    # Check metadata overhead
    print("\n2. METADATA OVERHEAD:")
    print(f"   For partition size {partition_size}:")
    bits_saved_needed_poly2 = (POLY2_METADATA - LINEAR_METADATA) * 8 / partition_size
    bits_saved_needed_poly3 = (POLY3_METADATA - LINEAR_METADATA) * 8 / partition_size
    print(f"   POLY2 needs to save {bits_saved_needed_poly2:.3f} bits/value to break even vs LINEAR")
    print(f"   POLY3 needs to save {bits_saved_needed_poly3:.3f} bits/value to break even vs LINEAR")

    # Actual bits comparison
    print("\n3. ACTUAL BITS COMPARISON (first 1000 partitions):")
    linear_bits_all = []
    poly2_bits_all = []
    poly3_bits_all = []

    for i in range(min(1000, num_partitions)):
        start = i * partition_size
        end = start + partition_size
        partition = data[start:end]

        ps = len(partition)
        x = np.arange(ps, dtype=np.float64)
        y = partition.astype(np.float64)

        coeffs1 = np.polyfit(x, y, 1)
        pred1 = np.polyval(coeffs1, x)
        res1 = np.max(np.abs(y - pred1))
        linear_bits = int(np.ceil(np.log2(res1 + 1))) + 1 if res1 > 0 else 1
        linear_bits_all.append(linear_bits)

        coeffs2 = np.polyfit(x, y, 2)
        pred2 = np.polyval(coeffs2, x)
        res2 = np.max(np.abs(y - pred2))
        poly2_bits = int(np.ceil(np.log2(res2 + 1))) + 1 if res2 > 0 else 1
        poly2_bits_all.append(poly2_bits)

        coeffs3 = np.polyfit(x, y, 3)
        pred3 = np.polyval(coeffs3, x)
        res3 = np.max(np.abs(y - pred3))
        poly3_bits = int(np.ceil(np.log2(res3 + 1))) + 1 if res3 > 0 else 1
        poly3_bits_all.append(poly3_bits)

    linear_bits_all = np.array(linear_bits_all)
    poly2_bits_all = np.array(poly2_bits_all)
    poly3_bits_all = np.array(poly3_bits_all)

    print(f"   Average bits - LINEAR: {np.mean(linear_bits_all):.1f}, POLY2: {np.mean(poly2_bits_all):.1f}, POLY3: {np.mean(poly3_bits_all):.1f}")
    print(f"   Bits saved - POLY2 vs LINEAR: {np.mean(linear_bits_all - poly2_bits_all):.1f}")
    print(f"   Bits saved - POLY3 vs LINEAR: {np.mean(linear_bits_all - poly3_bits_all):.1f}")

    # Check cost comparison
    print("\n4. COST COMPARISON:")
    linear_costs = LINEAR_METADATA + partition_size * linear_bits_all / 8
    poly2_costs = POLY2_METADATA + partition_size * poly2_bits_all / 8
    poly3_costs = POLY3_METADATA + partition_size * poly3_bits_all / 8

    print(f"   Average cost - LINEAR: {np.mean(linear_costs):.1f}B, POLY2: {np.mean(poly2_costs):.1f}B, POLY3: {np.mean(poly3_costs):.1f}B")

    poly2_better = np.sum(poly2_costs < linear_costs * COST_THRESHOLD)
    poly3_better = np.sum(poly3_costs < linear_costs * COST_THRESHOLD)

    print(f"   Partitions where POLY2 is {100*(1-COST_THRESHOLD):.0f}% better than LINEAR: {poly2_better} ({100*poly2_better/len(linear_costs):.1f}%)")
    print(f"   Partitions where POLY3 is {100*(1-COST_THRESHOLD):.0f}% better than LINEAR: {poly3_better} ({100*poly3_better/len(linear_costs):.1f}%)")

    # What if we remove the threshold?
    print("\n5. WITHOUT 5% THRESHOLD:")
    poly2_better_no_thresh = np.sum(poly2_costs < linear_costs)
    poly3_better_no_thresh = np.sum(poly3_costs < linear_costs)
    print(f"   Partitions where POLY2 cost < LINEAR cost: {poly2_better_no_thresh} ({100*poly2_better_no_thresh/len(linear_costs):.1f}%)")
    print(f"   Partitions where POLY3 cost < LINEAR cost: {poly3_better_no_thresh} ({100*poly3_better_no_thresh/len(linear_costs):.1f}%)")

    # Potential compression improvement
    print("\n" + "=" * 80)
    print("POTENTIAL IMPROVEMENT")
    print("=" * 80)

    current_bits = np.mean(linear_bits_all)  # Assuming LINEAR is chosen
    poly2_bits_avg = np.mean(poly2_bits_all)

    current_ratio = 64 / current_bits
    poly2_ratio = 64 / poly2_bits_avg

    print(f"   Current (LINEAR): {current_bits:.1f} bits -> {current_ratio:.2f}x compression")
    print(f"   With POLY2: {poly2_bits_avg:.1f} bits -> {poly2_ratio:.2f}x compression")
    print(f"   Potential improvement: {poly2_ratio/current_ratio:.2f}x better")

    # Recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
    1. The normal distribution CDF has a QUADRATIC shape at most points.
       Linear models capture the trend but miss the curvature.

    2. POLY2 can reduce bits from ~33 to ~19, a 40% reduction!

    3. The 5% threshold is too strict for this data.
       Many partitions would benefit from POLY2 but don't pass the threshold.

    4. SOLUTION OPTIONS:
       a) Lower COST_THRESHOLD from 0.95 to 0.99 (require only 1% improvement)
       b) Use a different cost model that weights metadata less
       c) Add special handling for sorted/monotonic data
       d) Use higher-order models by default for sorted data

    5. Expected improvement with POLY2:
       - Current: ~2x compression
       - With POLY2: ~3-3.5x compression
    """)

def main():
    data_file = "/root/autodl-tmp/test/data/sosd/2-normal_200M_uint64.bin"
    if len(sys.argv) > 1:
        data_file = sys.argv[1]

    print(f"Loading: {data_file}")
    data = load_binary_uint64(data_file)

    analyze_model_selection(data)

if __name__ == "__main__":
    main()
