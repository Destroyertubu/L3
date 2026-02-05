#!/usr/bin/env python3
"""
Sort LINEORDER table by lo_orderdate for RLE compression.
Saves sorted data to compressed_tilegpu directory.
"""

import numpy as np
import os
import shutil

RAW_DATA_DIR = "/home/xiayouyang/code/test/ssb_data/"
COMPRESSED_DIR = "/home/xiayouyang/code/test/ssb_data/compressed_tilegpu/"
LO_LEN = 120000000  # SF=20

# LINEORDER columns that need to be reordered
LO_COLUMNS = [
    "LINEORDER0",   # lo_orderkey
    "LINEORDER1",   # lo_linenumber
    "LINEORDER2",   # lo_custkey
    "LINEORDER3",   # lo_partkey
    "LINEORDER4",   # lo_suppkey
    "LINEORDER5",   # lo_orderdate (sort key)
    "LINEORDER6",   # lo_orderpriority
    "LINEORDER7",   # lo_shippriority
    "LINEORDER8",   # lo_quantity
    "LINEORDER9",   # lo_extendedprice
    "LINEORDER10",  # lo_ordtotalprice
    "LINEORDER11",  # lo_discount
    "LINEORDER12",  # lo_revenue
    "LINEORDER13",  # lo_supplycost
    "LINEORDER14",  # lo_tax
    "LINEORDER15",  # lo_commitdate
    "LINEORDER16",  # lo_shipmode
]

# Dimension table files to copy (not sorted, just copied)
DIM_FILES = [
    "PART0.bin", "PART2.bin", "PART3.bin", "PART4.bin",
    "PART5.bin", "PART6.bin", "PART7.bin", "PART8.bin",
    "SUPPLIER0.bin", "SUPPLIER3.bin", "SUPPLIER4.bin", "SUPPLIER5.bin",
    "CUSTOMER0.bin", "CUSTOMER3.bin", "CUSTOMER4.bin", "CUSTOMER5.bin", "CUSTOMER7.bin",
    "DDATE0.bin", "DDATE4.bin", "DDATE5.bin", "DDATE7.bin", "DDATE8.bin",
    "DDATE9.bin", "DDATE10.bin", "DDATE11.bin", "DDATE12.bin",
    "DDATE13.bin", "DDATE14.bin", "DDATE15.bin", "DDATE16.bin",
]

def main():
    os.makedirs(COMPRESSED_DIR, exist_ok=True)

    # Check if sorted data already exists
    sorted_marker = os.path.join(COMPRESSED_DIR, ".sorted_done")
    if os.path.exists(sorted_marker):
        print("Sorted data already exists. Skipping sort.")
        return

    print(f"Loading lo_orderdate (LINEORDER5) to get sort order...")
    orderdate_file = os.path.join(RAW_DATA_DIR, "LINEORDER5.bin")
    orderdate = np.fromfile(orderdate_file, dtype=np.uint32)
    print(f"  Loaded {len(orderdate):,} values")

    print("Getting sort indices...")
    sort_indices = np.argsort(orderdate, kind='stable')

    # Sort and save each LINEORDER column to compressed directory
    for col_name in LO_COLUMNS:
        src_file = os.path.join(RAW_DATA_DIR, f"{col_name}.bin")
        dst_file = os.path.join(COMPRESSED_DIR, f"{col_name}.bin")
        print(f"Processing {col_name}...")

        # Load column from raw directory
        col_data = np.fromfile(src_file, dtype=np.uint32)

        # Reorder
        col_sorted = col_data[sort_indices]

        # Save to compressed directory
        col_sorted.tofile(dst_file)
        print(f"  Saved sorted {col_name} to compressed directory")

    # Copy dimension tables to compressed directory
    print("\nCopying dimension tables...")
    for dim_file in DIM_FILES:
        src = os.path.join(RAW_DATA_DIR, dim_file)
        dst = os.path.join(COMPRESSED_DIR, dim_file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {dim_file}")

    # Create marker file
    with open(sorted_marker, 'w') as f:
        f.write("done")

    print("\nAll LINEORDER columns sorted and dimension tables copied!")

if __name__ == "__main__":
    main()
