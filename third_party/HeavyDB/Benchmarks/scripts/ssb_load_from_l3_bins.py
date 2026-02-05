#!/usr/bin/env python3

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

# Ensure local `Benchmarks/pymapd.py` shim is importable when running this script directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pymapd  # noqa: E402


def _read_schema(schema_path: str, table_name: str) -> str:
    with open(schema_path, "r") as f:
        sql = f.read()
    return sql.replace("##TAB##", table_name)


def _u32_nrows(path: str) -> int:
    size = os.path.getsize(path)
    if size % 4 != 0:
        raise RuntimeError(f"File size is not a multiple of 4 bytes: {path}")
    return size // 4


def _memmap_u32(path: str, nrows: int) -> np.memmap:
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(nrows,))


def _load_table_columnar_in_chunks(con, table_name: str, columns, chunk_rows: int) -> None:
    if not columns:
        raise ValueError("columns is empty")

    nrows = _u32_nrows(columns[0][1])
    for col_name, col_path in columns:
        n = _u32_nrows(col_path)
        if n != nrows:
            raise RuntimeError(
                f"Row count mismatch for table {table_name}: {col_path} has {n} rows, expected {nrows}"
            )

    mmaps = [(col_name, _memmap_u32(col_path, nrows)) for col_name, col_path in columns]

    start_time = time.perf_counter()
    for start in range(0, nrows, chunk_rows):
        end = min(start + chunk_rows, nrows)
        df = pd.DataFrame({name: np.asarray(arr[start:end], dtype=np.int32) for name, arr in mmaps})
        con.load_table_columnar(table_name, df, preserve_index=False)

    elapsed_s = time.perf_counter() - start_time
    print(f"[{table_name}] loaded {nrows} rows in {elapsed_s:.3f}s ({nrows / max(elapsed_s, 1e-9):.0f} rows/s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load SSB (integer-only, L3 binary columns) into HeavyDB via pymapd.load_table_columnar"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6274)
    parser.add_argument("--db", default="mapd")
    parser.add_argument("--user", default="mapd")
    parser.add_argument("--password", default="HyperInteractive")
    parser.add_argument(
        "--data-dir",
        default="/home/xiayouyang/code/test/ssb_data",
        help="Directory containing L3 SSB *.bin columns (e.g., LINEORDER5.bin, PART3.bin).",
    )
    parser.add_argument(
        "--schema-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "import_table_schemas"),
        help="Directory containing ssb_*.sql schema files with ##TAB## placeholder.",
    )
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Do not DROP TABLE IF EXISTS before CREATE TABLE.",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["dates", "supplier", "customer", "part", "lineorder"],
        help="Subset of tables to load: dates supplier customer part lineorder",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.rstrip("/") + "/"

    table_specs = {
        "dates": {
            "schema": os.path.join(args.schema_dir, "ssb_dates.sql"),
            "columns": [
                ("d_datekey", os.path.join(data_dir, "DDATE0.bin")),
                ("d_year", os.path.join(data_dir, "DDATE4.bin")),
            ],
        },
        "customer": {
            "schema": os.path.join(args.schema_dir, "ssb_customer.sql"),
            "columns": [
                ("c_custkey", os.path.join(data_dir, "CUSTOMER0.bin")),
                ("c_city", os.path.join(data_dir, "CUSTOMER3.bin")),
                ("c_nation", os.path.join(data_dir, "CUSTOMER4.bin")),
                ("c_region", os.path.join(data_dir, "CUSTOMER5.bin")),
            ],
        },
        "supplier": {
            "schema": os.path.join(args.schema_dir, "ssb_supplier.sql"),
            "columns": [
                ("s_suppkey", os.path.join(data_dir, "SUPPLIER0.bin")),
                ("s_city", os.path.join(data_dir, "SUPPLIER3.bin")),
                ("s_nation", os.path.join(data_dir, "SUPPLIER4.bin")),
                ("s_region", os.path.join(data_dir, "SUPPLIER5.bin")),
            ],
        },
        "part": {
            "schema": os.path.join(args.schema_dir, "ssb_part.sql"),
            "columns": [
                ("p_partkey", os.path.join(data_dir, "PART0.bin")),
                ("p_mfgr", os.path.join(data_dir, "PART2.bin")),
                ("p_category", os.path.join(data_dir, "PART3.bin")),
                ("p_brand1", os.path.join(data_dir, "PART4.bin")),
            ],
        },
        "lineorder": {
            "schema": os.path.join(args.schema_dir, "ssb_lineorder.sql"),
            "columns": [
                ("lo_orderdate", os.path.join(data_dir, "LINEORDER5.bin")),
                ("lo_custkey", os.path.join(data_dir, "LINEORDER2.bin")),
                ("lo_partkey", os.path.join(data_dir, "LINEORDER3.bin")),
                ("lo_suppkey", os.path.join(data_dir, "LINEORDER4.bin")),
                ("lo_quantity", os.path.join(data_dir, "LINEORDER8.bin")),
                ("lo_extendedprice", os.path.join(data_dir, "LINEORDER9.bin")),
                ("lo_discount", os.path.join(data_dir, "LINEORDER11.bin")),
                ("lo_revenue", os.path.join(data_dir, "LINEORDER12.bin")),
                ("lo_supplycost", os.path.join(data_dir, "LINEORDER13.bin")),
            ],
        },
    }

    con = pymapd.connect(
        user=args.user, password=args.password, host=args.host, port=args.port, dbname=args.db
    )
    try:
        for table_name in args.tables:
            if table_name not in table_specs:
                raise ValueError(f"Unknown table: {table_name}")
            spec = table_specs[table_name]
            schema_sql = _read_schema(spec["schema"], table_name)
            if not args.no_drop:
                con.execute(f"DROP TABLE IF EXISTS {table_name};")
            con.execute(schema_sql)
            _load_table_columnar_in_chunks(con, table_name, spec["columns"], args.chunk_rows)
    finally:
        con.close()


if __name__ == "__main__":
    main()
