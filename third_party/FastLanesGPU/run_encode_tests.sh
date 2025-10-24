#!/bin/bash
# Test FSL-GPU compression on data files
# Usage: ./run_encode_tests.sh

DATA_DIR="/root/autodl-tmp/test/data"
OUTPUT_DIR="/root/autodl-tmp/test/paint_scripts/encode/FSL-GPU"
BENCHMARK_DIR="/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/benchmarks"

echo "======================================================================"
echo "FSL-GPU Compression Benchmark"
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

# Test only the main binary files in /root/autodl-tmp/test/data
echo "======================================================================"
echo "Testing datasets in $DATA_DIR"
echo "======================================================================"
echo ""

# Find all .bin files in the data directory (not subdirectories)
cd "$DATA_DIR"
DATA_FILES=($(ls -1 *.bin 2>/dev/null | sort))

if [ ${#DATA_FILES[@]} -eq 0 ]; then
    echo "No .bin files found in $DATA_DIR"
    exit 1
fi

echo "Found ${#DATA_FILES[@]} datasets to test"
echo ""

for data_file in "${DATA_FILES[@]}"; do
    data_path="$DATA_DIR/$data_file"

    if [ ! -f "$data_path" ]; then
        echo "Skipping $data_file (file not found)"
        continue
    fi

    # Get file size
    file_size=$(du -h "$data_path" | awk '{print $1}')
    file_size_bytes=$(stat -c%s "$data_path")

    if [ "$file_size_bytes" -lt 1000 ]; then
        echo "Skipping $data_file (file too small: $file_size_bytes bytes)"
        continue
    fi

    echo "======================================================================"
    echo "Testing: $data_file (size: $file_size)"
    echo "======================================================================"

    # Run Delta benchmark
    echo "Running Delta compression..."
    "$BENCHMARK_DIR/benchmark_delta" "$data_path" "$DELTA_CSV" 2>&1 | grep -v "ERROR:" | grep -v "free():"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Delta compression completed successfully"
    else
        echo "Delta benchmark completed with warnings for $data_file"
    fi
    echo ""

    # Run BitPack benchmark
    echo "Running BitPack compression..."
    "$BENCHMARK_DIR/benchmark_bitpack" "$data_path" "$BITPACK_CSV" 2>&1 | grep -v "ERROR:" | grep -v "free():"
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "BitPack compression completed successfully"
    else
        echo "BitPack benchmark completed with warnings for $data_file"
    fi
    echo ""
done

echo "======================================================================"
echo "All compression tests completed!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - Delta results: $DELTA_CSV"
echo "  - BitPack results: $BITPACK_CSV"
echo ""

# Display summary
if [ -f "$DELTA_CSV" ]; then
    echo "Delta Compression Results:"
    cat "$DELTA_CSV"
    echo ""
fi

if [ -f "$BITPACK_CSV" ]; then
    echo "BitPack Compression Results:"
    cat "$BITPACK_CSV"
    echo ""
fi
