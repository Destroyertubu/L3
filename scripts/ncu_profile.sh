#!/bin/bash

# Nsight Compute profiling script for GLECO kernels

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCH_EXEC="${BUILD_DIR}/bin/L3_bench"
RESULTS_DIR="${PROJECT_ROOT}/results"

echo "==================================="
echo "GLECO Kernel Profiling with Nsight Compute"
echo "==================================="

# Check if ncu is available
if ! command -v ncu &> /dev/null; then
    echo "WARNING: ncu (Nsight Compute) not found in PATH"
    echo "Profiling skipped. Install NVIDIA Nsight Compute to use this feature."
    exit 1
fi

# Check if built
if [ ! -f "${BENCH_EXEC}" ]; then
    echo "ERROR: Benchmark executable not found"
    echo "Please run scripts/build.sh first"
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

# Profile with key metrics
NCU_REPORT="${RESULTS_DIR}/ncu_report.txt"

echo "Profiling kernels..."
echo "This may take a few minutes..."
echo ""

ncu \
    --set full \
    --target-processes all \
    --export "${RESULTS_DIR}/ncu_profile" \
    --force-overwrite \
    "${BENCH_EXEC}" \
    > "${NCU_REPORT}" 2>&1 \
    || echo "Note: Some profiling warnings may be expected"

echo ""
echo "==================================="
echo "Profiling complete!"
echo "==================================="
echo "Report: ${NCU_REPORT}"
echo "Profile data: ${RESULTS_DIR}/ncu_profile.ncu-rep"
echo ""
echo "Open with: ncu-ui ${RESULTS_DIR}/ncu_profile.ncu-rep"
