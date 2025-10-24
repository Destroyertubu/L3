#!/bin/bash
# SSB Query Test Script for FastLanesGPU with GLECO Data
# Usage: ./run_ssb_tests.sh [crystal|crystal-opt|all] [query_number]
# Examples:
#   ./run_ssb_tests.sh crystal 11        # Run crystal q11 only
#   ./run_ssb_tests.sh crystal-opt all   # Run all crystal-opt queries
#   ./run_ssb_tests.sh all 11            # Run q11 on both crystal and crystal-opt

set -e

VARIANT="${1:-all}"
QUERY="${2:-all}"

echo "======================================================================"
echo "FastLanesGPU SSB Benchmark with GLECO Data (SF=20)"
echo "Data: /root/autodl-tmp/code/data/SSB/L3/ssb_data/"
echo "Data size: ~119M rows in LINEORDER table"
echo "======================================================================"
echo ""

run_query() {
    local executable=$1
    local name=$2

    if [ ! -f "$executable" ]; then
        echo "⚠️  $executable not found, skipping..."
        return
    fi

    echo "Running $name..."
    echo "----------------------------------------------------------------------"
    $executable 2>&1 | grep -E "(Revenue|Time|query|Using device)" || $executable
    echo ""
}

run_variant() {
    local variant=$1
    local query=$2

    if [ "$query" = "all" ]; then
        queries=(11 12 13 21 22 23 31 32 33 34 41 42 43)
    else
        queries=($query)
    fi

    for q in "${queries[@]}"; do
        if [ "$variant" = "crystal" ] || [ "$variant" = "all" ]; then
            run_query "./crystal/src/crystal_q${q}" "Crystal Q${q}"
        fi

        if [ "$variant" = "crystal-opt" ] || [ "$variant" = "all" ]; then
            run_query "./crystal-opt/src/crystal_opt_q${q}" "Crystal-Opt Q${q}"
        fi
    done
}

# Main execution
if [ "$QUERY" = "all" ]; then
    echo "Running all queries for variant: $VARIANT"
    echo ""
else
    echo "Running query $QUERY for variant: $VARIANT"
    echo ""
fi

run_variant "$VARIANT" "$QUERY"

echo "======================================================================"
echo "Test completed!"
echo "======================================================================"
