#!/usr/bin/env python3
"""
Convert SSB lineorder text columns to binary format
"""

import os
import struct
import sys

def convert_txt_to_binary(input_file, output_file):
    """Convert text file with integers to binary uint32 format"""
    print(f"Converting {input_file} to {output_file}...")

    # Read text file
    with open(input_file, 'r') as f:
        numbers = [int(line.strip()) for line in f if line.strip()]

    print(f"  Read {len(numbers)} integers")

    # Write binary file
    with open(output_file, 'wb') as f:
        for num in numbers:
            # Write as uint32
            f.write(struct.pack('<I', num))

    # Get file sizes
    input_size = os.path.getsize(input_file)
    output_size = os.path.getsize(output_file)

    print(f"  Input size: {input_size / (1024*1024):.2f} MB")
    print(f"  Output size: {output_size / (1024*1024):.2f} MB")
    print(f"  ✓ Conversion complete\n")

def main():
    input_dir = "/root/autodl-tmp/test/data/lineorder_columns"
    output_dir = "/root/autodl-tmp/test/data/lineorder_columns_binary"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all .txt files
    txt_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.txt')])

    print(f"Found {len(txt_files)} text files to convert\n")
    print("="*70)

    for txt_file in txt_files:
        input_path = os.path.join(input_dir, txt_file)
        output_path = os.path.join(output_dir, txt_file.replace('.txt', '.bin'))
        convert_txt_to_binary(input_path, output_path)

    print("="*70)
    print(f"All files converted to {output_dir}/")

if __name__ == "__main__":
    main()
