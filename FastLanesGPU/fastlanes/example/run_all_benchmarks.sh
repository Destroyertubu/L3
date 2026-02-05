#!/bin/bash
# Run fastlanes_bench_delta and fastlanes_bench_bitpack on all uint32 datasets

SCRIPT_DIR="/home/xiayouyang/code/L3/third_party/tmp/FastLanesGPU-main/fastlanes/example"
BUILD_DIR="/home/xiayouyang/code/L3/third_party/tmp/FastLanesGPU-main/build"
DATA_DIR="/home/xiayouyang/code/L3/data/sosd"
REPORT_DIR="$SCRIPT_DIR/reports"

mkdir -p "$REPORT_DIR"

# uint32 datasets (sorted order)
DATASETS=(
    "1-linear:1-linear_200M_uint32.bin"
    "2-normal:2-normal_200M_uint32.bin"
    "5-books:5-books_200M_uint32.bin"
    "9-movieid:9-movieid_uint32.bin"
    "14-cosmos:14-cosmos_int32.bin"
    "18-site:18-site_250k_uint32.bin"
    "19-weight:19-weight_25k_uint32.bin"
    "20-adult:20-adult_30k_uint32.bin"
)

# Source files
DELTA_SRC="$SCRIPT_DIR/fastlanes_bench_delta.cu"
BITPACK_SRC="$SCRIPT_DIR/fastlanes_bench_bitpack.cu"

# Backup original files
cp "$DELTA_SRC" "$DELTA_SRC.bak"
cp "$BITPACK_SRC" "$BITPACK_SRC.bak"

# Output files
DELTA_CSV="$REPORT_DIR/fastlanes_delta_results.csv"
BITPACK_CSV="$REPORT_DIR/fastlanes_bitpack_results.csv"

echo "Dataset,Elements,Bitwidth,CompressionRatio,NoTranspose_GBps,Optimized_GBps,ComputePadded_GBps,Verified" > "$DELTA_CSV"
echo "Dataset,Elements,Bitwidth,CompressionRatio,Decode_GBps,Verified" > "$BITPACK_CSV"

echo "=============================================="
echo "FastLanes GPU Benchmark - All uint32 Datasets"
echo "=============================================="

for entry in "${DATASETS[@]}"; do
    name="${entry%%:*}"
    file="${entry##*:}"
    filepath="$DATA_DIR/$file"

    if [ ! -f "$filepath" ]; then
        echo "SKIP: $name - file not found: $filepath"
        continue
    fi

    echo ""
    echo "=== Testing: $name ($file) ==="

    # Update delta source file
    sed -i "s|const char\* data_file = \".*\";|const char* data_file = \"$filepath\";|g" "$DELTA_SRC"

    # Update bitpack source file
    sed -i "s|const char\* data_file = \".*\";|const char* data_file = \"$filepath\";|g" "$BITPACK_SRC"

    # Rebuild
    echo -n "  Compiling... "
    cd "$BUILD_DIR"
    make fastlanes_bench_delta fastlanes_bench_bitpack -j4 > /dev/null 2>&1
    echo "done"

    # Run delta benchmark
    echo -n "  Running delta benchmark... "
    delta_output=$("$BUILD_DIR/fastlanes/example/fastlanes_bench_delta" 2>&1)

    # Save full output for debugging
    echo "$delta_output" > "$REPORT_DIR/delta_${name}_full.log"

    # Parse delta results
    elements=$(echo "$delta_output" | grep "Processing elements:" | sed 's/.*: //')
    bitwidth=$(echo "$delta_output" | grep "Computed delta bitwidth:" | sed 's/.*: //' | sed 's/ bits//')
    ratio=$(echo "$delta_output" | grep "^-- Compression ratio:" | sed 's/.*: //' | sed 's/x//')
    no_transpose=$(echo "$delta_output" | grep "No transpose (baseline)" | awk '{print $5}')
    optimized=$(echo "$delta_output" | grep "Optimized (coalesced)" | awk '{print $5}')
    compute_padded=$(echo "$delta_output" | grep "Compute + Padded" | awk '{print $5}')

    # Check verification
    if echo "$delta_output" | grep -q "FAILED"; then
        verified="FAIL"
    else
        verified="OK"
    fi

    echo "$name,$elements,$bitwidth,$ratio,$no_transpose,$optimized,$compute_padded,$verified" >> "$DELTA_CSV"
    echo "done"
    echo "    Elements=$elements, Bits=$bitwidth, Ratio=${ratio}x, Optimized=${optimized} GB/s, Verify=$verified"

    # Run bitpack benchmark
    echo -n "  Running bitpack benchmark... "
    bitpack_output=$("$BUILD_DIR/fastlanes/example/fastlanes_bench_bitpack" 2>&1)

    # Save full output
    echo "$bitpack_output" > "$REPORT_DIR/bitpack_${name}_full.log"

    # Parse bitpack results
    bp_elements=$(echo "$bitpack_output" | grep "Processing elements:" | sed 's/.*: //')
    bp_bitwidth=$(echo "$bitpack_output" | grep "Computed bitwidth:" | sed 's/.*: //')
    bp_ratio=$(echo "$bitpack_output" | grep "Compression ratio:" | sed 's/.*: //' | sed 's/x//')
    bp_decode=$(echo "$bitpack_output" | grep "Decode throughput:" | head -1 | awk '{print $4}')

    if echo "$bitpack_output" | grep -q "successful"; then
        bp_verified="OK"
    else
        bp_verified="FAIL"
    fi

    echo "$name,$bp_elements,$bp_bitwidth,$bp_ratio,$bp_decode,$bp_verified" >> "$BITPACK_CSV"
    echo "done"
    echo "    Elements=$bp_elements, Bits=$bp_bitwidth, Ratio=${bp_ratio}x, Decode=${bp_decode} GB/s, Verify=$bp_verified"
done

# Restore original files
mv "$DELTA_SRC.bak" "$DELTA_SRC"
mv "$BITPACK_SRC.bak" "$BITPACK_SRC"

# Rebuild with original
cd "$BUILD_DIR"
make fastlanes_bench_delta fastlanes_bench_bitpack -j4 > /dev/null 2>&1

echo ""
echo "=============================================="
echo "Complete!"
echo "=============================================="
echo "Delta results:   $DELTA_CSV"
echo "Bitpack results: $BITPACK_CSV"
echo ""

# Print summary tables
echo "=== Delta Results ==="
column -t -s',' "$DELTA_CSV"
echo ""
echo "=== Bitpack Results ==="
column -t -s',' "$BITPACK_CSV"
