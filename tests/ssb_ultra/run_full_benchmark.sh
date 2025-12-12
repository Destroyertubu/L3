#!/bin/bash
# SSB Full Benchmark - Compare all 4 strategies across 13 queries
# Generates CSV output for analysis

BIN_DIR="/root/autodl-tmp/code/L3/build/bin"
DATA_DIR="/root/autodl-tmp/test/ssb_data"
OUTPUT_DIR="/root/autodl-tmp/code/L3/tests/ssb_ultra/results"
mkdir -p $OUTPUT_DIR

STRATEGIES="decompress_first fused_query predicate_pushdown random_access"
QUERIES="q11 q12 q13 q21 q22 q23 q31 q32 q33 q34 q41 q42 q43"

# CSV header
echo "strategy,query,run,total_ms,data_load_ms,hash_build_ms,kernel_ms" > $OUTPUT_DIR/ssb_benchmark_results.csv

# Function to extract timing from output
extract_timing() {
    local output="$1"
    local total=$(echo "$output" | grep -E "^Run [0-9]+:" | tail -1 | grep -oP '[\d.]+(?= ms)')
    local data_load=$(echo "$output" | grep "data_load_ms" | tail -1 | grep -oP '[\d.]+' | head -1)
    local hash_build=$(echo "$output" | grep "hash_build_ms" | tail -1 | grep -oP '[\d.]+' | head -1)
    local kernel=$(echo "$output" | grep "kernel_ms" | tail -1 | grep -oP '[\d.]+' | head -1)

    if [ -z "$total" ]; then
        total="0"
    fi
    if [ -z "$data_load" ]; then
        data_load=$(echo "$output" | grep -E "Data Load|Decompress" | tail -1 | grep -oP '[\d.]+' | head -1)
        [ -z "$data_load" ] && data_load="0"
    fi
    if [ -z "$hash_build" ]; then
        hash_build=$(echo "$output" | grep -E "Hash Build|hash_build" | tail -1 | grep -oP '[\d.]+' | head -1)
        [ -z "$hash_build" ] && hash_build="0"
    fi
    if [ -z "$kernel" ]; then
        kernel=$(echo "$output" | grep -E "Kernel|kernel" | tail -1 | grep -oP '[\d.]+' | head -1)
        [ -z "$kernel" ] && kernel="0"
    fi

    echo "$total,$data_load,$hash_build,$kernel"
}

echo "=== SSB Full Benchmark ==="
echo "Date: $(date)"
echo "Strategies: $STRATEGIES"
echo "Queries: $QUERIES"
echo ""

for strategy in $STRATEGIES; do
    echo "=== Strategy: $strategy ==="
    for query in $QUERIES; do
        executable="$BIN_DIR/ssb_${query}_${strategy}"
        if [ -x "$executable" ]; then
            echo "Running $query ($strategy)..."
            output=$(timeout 120 $executable $DATA_DIR 2>&1)

            # Extract the last run timing (most stable)
            for run in 1 2 3; do
                run_time=$(echo "$output" | grep -E "^Run $run:" | grep -oP '[\d.]+(?= ms)')
                if [ -n "$run_time" ]; then
                    timing=$(extract_timing "$output")
                    echo "$strategy,$query,$run,$timing" >> $OUTPUT_DIR/ssb_benchmark_results.csv
                fi
            done

            # Print summary
            last_run=$(echo "$output" | grep -E "^Run 3:" | grep -oP '[\d.]+(?= ms)')
            if [ -n "$last_run" ]; then
                echo "  $query: $last_run ms"
            else
                echo "  $query: FAILED or TIMEOUT"
            fi
        else
            echo "  $query: executable not found ($executable)"
        fi
    done
    echo ""
done

echo "=== Benchmark Complete ==="
echo "Results saved to: $OUTPUT_DIR/ssb_benchmark_results.csv"

# Generate summary table
echo ""
echo "=== Summary (Run 3, ms) ==="
printf "%-8s" "Query"
for strategy in $STRATEGIES; do
    printf "%-20s" "$strategy"
done
echo ""
printf "%s\n" "$(printf '=%.0s' {1..88})"

for query in $QUERIES; do
    printf "%-8s" "$query"
    for strategy in $STRATEGIES; do
        time=$(grep "$strategy,$query,3," $OUTPUT_DIR/ssb_benchmark_results.csv 2>/dev/null | cut -d',' -f4)
        if [ -n "$time" ]; then
            printf "%-20s" "${time}"
        else
            printf "%-20s" "-"
        fi
    done
    echo ""
done
