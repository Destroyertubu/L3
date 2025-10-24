#!/bin/bash
# Quick Phase 2 performance benchmark script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GLECO Phase 2 Performance Benchmark                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if binaries exist
if [ ! -f "bin/benchmark_decode_phase2" ]; then
    echo "ERROR: Phase 2 benchmark not found. Run scripts/build_phase2.sh first."
    exit 1
fi

echo "Running Phase 2 benchmark (this may take 2-3 minutes)..."
echo ""

./bin/benchmark_decode_phase2 2>&1 | tee phase2_benchmark_output.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results saved to:"
echo "  • phase2_results.csv"
echo "  • phase2_benchmark_output.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -f "phase2_results.csv" ]; then
    echo "Performance Summary:"
    echo ""
    awk -F',' 'NR==1 {print; print "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"}
                NR>1 && $1=="Phase2" {printf "%-20s %10s %12s GB/s  %6sx speedup  %s\n", $1"_"$2"bit", $3, $5, $6, $7}' phase2_results.csv
fi
