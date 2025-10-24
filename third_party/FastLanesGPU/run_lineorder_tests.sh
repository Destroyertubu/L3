#!/bin/bash
# Test FSL-GPU benchmarks on SSB lineorder columns
# Usage: ./run_lineorder_tests.sh

DATA_DIR="/root/autodl-tmp/test/data/lineorder_columns_binary"
OUTPUT_DIR="/root/autodl-tmp/test/paint_scripts/ssb_table/FSL-GPU"
BENCHMARK_DIR="/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/benchmarks"

echo "======================================================================"
echo "FSL-GPU Benchmark on SSB Lineorder Columns"
echo "======================================================================"
echo ""
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize CSV files
DELTA_CSV="$OUTPUT_DIR/delta_results.csv"
BITPACK_CSV="$OUTPUT_DIR/bitpack_results.csv"

# Remove old results if they exist
rm -f "$DELTA_CSV" "$BITPACK_CSV"

# Get list of all bin files (lineorder columns)
cd "$DATA_DIR"
COLUMNS=($(ls -1 *.bin | sort))

echo "Found ${#COLUMNS[@]} lineorder columns to test"
echo ""

# Test each column
for col_file in "${COLUMNS[@]}"; do
    col_name=$(basename "$col_file" .bin)
    col_path="$DATA_DIR/$col_file"

    # Skip empty files
    file_size_bytes=$(stat -c%s "$col_path")
    if [ "$file_size_bytes" -lt 1000 ]; then
        echo "Skipping $col_name (file too small: $file_size_bytes bytes)"
        continue
    fi

    # Get file size
    file_size=$(du -h "$col_path" | awk '{print $1}')

    echo "======================================================================"
    echo "Testing column: $col_name (size: $file_size)"
    echo "======================================================================"

    # Run Delta benchmark
    echo "Running Delta compression..."
    "$BENCHMARK_DIR/benchmark_delta" "$col_path" "$DELTA_CSV" || echo "Delta benchmark failed for $col_name"
    echo ""

    # Run BitPack benchmark
    echo "Running BitPack compression..."
    "$BENCHMARK_DIR/benchmark_bitpack" "$col_path" "$BITPACK_CSV" || echo "BitPack benchmark failed for $col_name"
    echo ""
done

echo "======================================================================"
echo "All tests completed!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - Delta: $DELTA_CSV"
echo "  - BitPack: $BITPACK_CSV"
echo ""
