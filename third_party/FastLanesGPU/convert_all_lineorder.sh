#!/bin/bash
# Convert all lineorder text files to binary format

set -e

INPUT_DIR="/root/autodl-tmp/test/data/lineorder_columns"
OUTPUT_DIR="/root/autodl-tmp/test/data/lineorder_columns_binary"
CONVERTER="/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/convert_txt"

echo "======================================================================"
echo "Converting SSB Lineorder Columns to Binary Format"
echo "======================================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Get list of all txt files
cd "$INPUT_DIR"
COLUMNS=($(ls -1 *.txt | sort))

echo "Found ${#COLUMNS[@]} columns to convert"
echo ""

# Convert each file
for txt_file in "${COLUMNS[@]}"; do
    col_name=$(basename "$txt_file" .txt)
    input_path="$INPUT_DIR/$txt_file"
    output_path="$OUTPUT_DIR/${col_name}.bin"

    echo "----------------------------------------------------------------------"
    echo "Converting: $col_name"
    "$CONVERTER" "$input_path" "$output_path"
    echo ""
done

echo "======================================================================"
echo "Conversion complete! Binary files saved to:"
echo "$OUTPUT_DIR"
echo "======================================================================"
