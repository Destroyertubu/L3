#!/bin/bash
# Run baseline decompression benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$PROJECT_ROOT/bin"

cd "$PROJECT_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Running L3 Decompression Baseline Benchmarks            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$BIN_DIR/benchmark_decode_baseline" ]; then
    echo "ERROR: benchmark_decode_baseline not found"
    echo "Please run scripts/build_decode_opt.sh first"
    exit 1
fi

echo "Starting benchmark..."
echo ""

"$BIN_DIR/benchmark_decode_baseline" 2>&1 | tee baseline_benchmark_output.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results saved to:"
echo "  • baseline_results.csv"
echo "  • baseline_benchmark_output.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
