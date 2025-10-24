#!/usr/bin/env python3

import subprocess
import json
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

# Configuration
WARMUP_RUNS = 3
MEASURE_RUNS = 10
BASE_DIR = Path("/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/crystal-opt/src")
OUTPUT_DIR = Path("/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/crystal-opt/src/ssb")

# All 13 queries
QUERIES = [
    "crystal_opt_q11", "crystal_opt_q12", "crystal_opt_q13",
    "crystal_opt_q21", "crystal_opt_q22", "crystal_opt_q23",
    "crystal_opt_q31", "crystal_opt_q32", "crystal_opt_q33", "crystal_opt_q34",
    "crystal_opt_q41", "crystal_opt_q42", "crystal_opt_q43"
]

def extract_time(output):
    """Extract execution time from query output."""
    # Try to find JSON with time_query field
    json_pattern = r'"time_query":\s*([0-9.]+)'
    match = re.search(json_pattern, output)
    if match:
        return float(match.group(1))

    # Fallback to Time Taken Total
    text_pattern = r'Time Taken Total:\s*([0-9.]+)'
    match = re.search(text_pattern, output)
    if match:
        return float(match.group(1))

    return None

def run_query(query_name):
    """Run a single query and return output."""
    try:
        result = subprocess.run(
            [f"./{query_name}"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr, result.returncode == 0
    except subprocess.TimeoutExpired:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {str(e)}", False

def calculate_stats(times):
    """Calculate statistics from a list of times."""
    if not times:
        return None

    return {
        'mean': statistics.mean(times),
        'min': min(times),
        'max': max(times),
        'stddev': statistics.stdev(times) if len(times) > 1 else 0.0,
        'median': statistics.median(times)
    }

def run_benchmark():
    """Run the complete benchmark suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = OUTPUT_DIR / f"benchmark_crystal_opt_{timestamp}.txt"
    json_file = OUTPUT_DIR / f"benchmark_crystal_opt_{timestamp}.json"

    results = {
        'timestamp': datetime.now().isoformat(),
        'warmup_runs': WARMUP_RUNS,
        'measurement_runs': MEASURE_RUNS,
        'queries': {}
    }

    # Open report file
    with open(report_file, 'w') as f:
        def log(msg):
            print(msg)
            f.write(msg + '\n')
            f.flush()

        log("=" * 80)
        log("FastLanes GPU (Crystal-Opt) SSB Benchmark Report")
        log("=" * 80)
        log(f"Timestamp: {datetime.now()}")
        log(f"Warmup Runs: {WARMUP_RUNS}")
        log(f"Measurement Runs: {MEASURE_RUNS}")
        log(f"Base Directory: {BASE_DIR}")
        log("=" * 80)
        log("")

        # Run benchmarks for each query
        for query in QUERIES:
            log("=" * 80)
            log(f"Running: {query}")
            log("=" * 80)

            # Check if executable exists
            exe_path = BASE_DIR / query
            if not exe_path.is_file():
                log(f"ERROR: {query} not found at {exe_path}")
                log("")
                continue

            # Warmup phase
            log(f"Warmup phase ({WARMUP_RUNS} runs)...")
            warmup_times = []
            for i in range(1, WARMUP_RUNS + 1):
                output, success = run_query(query)
                if success:
                    exec_time = extract_time(output)
                    if exec_time is not None:
                        warmup_times.append(exec_time)
                        log(f"  Warmup run {i}/{WARMUP_RUNS}... {exec_time:.4f}ms")
                    else:
                        log(f"  Warmup run {i}/{WARMUP_RUNS}... completed (no time found)")
                else:
                    log(f"  Warmup run {i}/{WARMUP_RUNS}... FAILED")

            # Measurement phase
            log(f"Measurement phase ({MEASURE_RUNS} runs)...")
            measure_times = []
            for i in range(1, MEASURE_RUNS + 1):
                output, success = run_query(query)
                if success:
                    exec_time = extract_time(output)
                    if exec_time is not None:
                        measure_times.append(exec_time)
                        log(f"  Run {i}/{MEASURE_RUNS}... {exec_time:.4f}ms")
                    else:
                        log(f"  Run {i}/{MEASURE_RUNS}... completed (no time found)")
                else:
                    log(f"  Run {i}/{MEASURE_RUNS}... FAILED")

            # Calculate and display statistics
            if measure_times:
                stats = calculate_stats(measure_times)
                log("")
                log(f"Statistics for {query}:")
                log(f"  Successful runs: {len(measure_times)}/{MEASURE_RUNS}")
                log(f"  Mean time:   {stats['mean']:.4f}ms")
                log(f"  Median time: {stats['median']:.4f}ms")
                log(f"  Min time:    {stats['min']:.4f}ms")
                log(f"  Max time:    {stats['max']:.4f}ms")
                log(f"  Std dev:     {stats['stddev']:.4f}ms")

                # Extract query number (e.g., crystal_opt_q11 -> Q1.1)
                query_num = query.replace('crystal_opt_q', '')
                if len(query_num) >= 2:
                    query_num = f"Q{query_num[0]}.{query_num[1:]}"
                else:
                    query_num = f"Q{query_num}"

                results['queries'][query_num] = {
                    'executable': query,
                    'successful_runs': len(measure_times),
                    'total_runs': MEASURE_RUNS,
                    'mean_ms': stats['mean'],
                    'median_ms': stats['median'],
                    'min_ms': stats['min'],
                    'max_ms': stats['max'],
                    'stddev_ms': stats['stddev'],
                    'times': measure_times
                }
            else:
                log("")
                log(f"No successful runs for {query}")

            log("")

        # Summary
        log("=" * 80)
        log("Benchmark Complete!")
        log("=" * 80)
        log(f"Report saved to: {report_file}")
        log(f"JSON data saved to: {json_file}")
        log("")

        # Summary table
        log("Summary Table:")
        log("-" * 80)
        log(f"{'Query':<10} {'Mean(ms)':>12} {'Median(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12} {'StdDev':>10}")
        log("-" * 80)

        query_keys = sorted(results['queries'].keys())
        for query in query_keys:
            q = results['queries'][query]
            log(f"{query:<10} {q['mean_ms']:>12.4f} {q['median_ms']:>12.4f} {q['min_ms']:>12.4f} {q['max_ms']:>12.4f} {q['stddev_ms']:>10.4f}")

        log("-" * 80)

        # Calculate averages
        if query_keys:
            avg_mean = statistics.mean([results['queries'][q]['mean_ms'] for q in query_keys])
            avg_median = statistics.mean([results['queries'][q]['median_ms'] for q in query_keys])
            avg_min = statistics.mean([results['queries'][q]['min_ms'] for q in query_keys])
            avg_max = statistics.mean([results['queries'][q]['max_ms'] for q in query_keys])
            avg_stddev = statistics.mean([results['queries'][q]['stddev_ms'] for q in query_keys])

            log(f"{'AVERAGE':<10} {avg_mean:>12.4f} {avg_median:>12.4f} {avg_min:>12.4f} {avg_max:>12.4f} {avg_stddev:>10.4f}")
            log("-" * 80)

    # Save JSON results
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nAll results saved in: {OUTPUT_DIR}")
    return results

if __name__ == "__main__":
    run_benchmark()
