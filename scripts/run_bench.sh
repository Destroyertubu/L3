#!/bin/bash

# Run benchmark script for L3 optimized decompression

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCH_EXEC="${BUILD_DIR}/bin/L3_bench"
RESULTS_DIR="${PROJECT_ROOT}/results"

echo "==================================="
echo "L3 Optimized Decompression Benchmark"
echo "==================================="

# Check if built
if [ ! -f "${BENCH_EXEC}" ]; then
    echo "ERROR: Benchmark executable not found at ${BENCH_EXEC}"
    echo "Please run scripts/build.sh first"
    exit 1
fi

# Create results directory
mkdir -p "${RESULTS_DIR}"

# Run benchmark
echo "Running benchmark..."
echo ""

cd "${PROJECT_ROOT}"
"${BENCH_EXEC}"

echo ""
echo "==================================="
echo "Benchmark complete!"
echo "Results saved to: ${RESULTS_DIR}/bench_log.csv"
echo "==================================="

# Display results if CSV exists
if [ -f "${RESULTS_DIR}/bench_log.csv" ]; then
    echo ""
    echo "Latest results:"
    tail -n 10 "${RESULTS_DIR}/bench_log.csv" | column -t -s,
fi
