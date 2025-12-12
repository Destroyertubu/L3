#!/usr/bin/env python3
"""
Comprehensive L3 Benchmark Runner for all 20 SOSD datasets.
Runs all 4 compression methods (Fixed, V1, V2, V3) on each dataset.
"""

import subprocess
import os
import struct
import time
import sys
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json

DATA_DIR = "/root/autodl-tmp/test/data/sosd"
BIN_DIR = "/root/autodl-tmp/code/L3/bin"

# All 20 datasets with their properties
DATASETS = [
    {"id": 1, "file": "1-linear_200M_uint64.bin", "name": "linear_200M", "dtype": "uint64"},
    {"id": 2, "file": "2-normal_200M_uint64.bin", "name": "normal_200M", "dtype": "uint64"},
    {"id": 3, "file": "3-poisson_87M_uint64.bin", "name": "poisson_87M", "dtype": "uint64"},
    {"id": 4, "file": "4-ml_uint64.bin", "name": "ml", "dtype": "uint64"},
    {"id": 5, "file": "5-books_200M_uint32.bin", "name": "books_200M", "dtype": "uint32"},
    {"id": 6, "file": "6-fb_200M_uint64.bin", "name": "fb_200M", "dtype": "uint64"},
    {"id": 7, "file": "7-wiki_200M_uint64.bin", "name": "wiki_200M", "dtype": "uint64"},
    {"id": 8, "file": "8-osm_cellids_800M_uint64.bin", "name": "osm_800M", "dtype": "uint64"},
    {"id": 9, "file": "9-movieid_uint32.bin", "name": "movieid", "dtype": "uint32"},
    {"id": 10, "file": "10-house_price_uint64.bin", "name": "house_price", "dtype": "uint64"},
    {"id": 11, "file": "11-planet_uint64.bin", "name": "planet", "dtype": "uint64"},
    {"id": 12, "file": "12-libio.bin", "name": "libio", "dtype": "uint64"},
    {"id": 13, "file": "13-medicare.bin", "name": "medicare", "dtype": "uint64"},
    {"id": 14, "file": "14-cosmos_int32.bin", "name": "cosmos", "dtype": "int32"},
    {"id": 15, "file": "15-polylog_10M_uint64.bin", "name": "polylog_10M", "dtype": "uint64"},
    {"id": 16, "file": "16-exp_200M_uint64.bin", "name": "exp_200M", "dtype": "uint64"},
    {"id": 17, "file": "17-poly_200M_uint64.bin", "name": "poly_200M", "dtype": "uint64"},
    {"id": 18, "file": "18-site_250k_uint32.bin", "name": "site_250k", "dtype": "uint32"},
    {"id": 19, "file": "19-weight_25k_uint32.bin", "name": "weight_25k", "dtype": "uint32"},
    {"id": 20, "file": "20-adult_30k_uint32.bin", "name": "adult_30k", "dtype": "uint32"},
]


@dataclass
class DatasetInfo:
    """Information about a dataset file."""
    path: str
    name: str
    dtype: str
    num_elements: int
    file_size_bytes: int
    has_header: bool

    @property
    def element_size(self) -> int:
        return 8 if "64" in self.dtype else 4

    @property
    def original_size_mb(self) -> float:
        return (self.num_elements * self.element_size) / (1024 * 1024)


def get_dataset_info(path: str, name: str, dtype: str) -> Optional[DatasetInfo]:
    """Get information about a dataset file."""
    if not os.path.exists(path):
        return None

    file_size = os.path.getsize(path)
    elem_size = 8 if "64" in dtype else 4

    # Check for header (8 bytes count at start)
    with open(path, 'rb') as f:
        header_count = struct.unpack('<Q', f.read(8))[0]

    data_bytes_with_header = file_size - 8
    expected_elements_with_header = data_bytes_with_header // elem_size

    if header_count == expected_elements_with_header:
        # Has header
        return DatasetInfo(
            path=path, name=name, dtype=dtype,
            num_elements=header_count,
            file_size_bytes=file_size,
            has_header=True
        )
    else:
        # No header, use full file
        return DatasetInfo(
            path=path, name=name, dtype=dtype,
            num_elements=file_size // elem_size,
            file_size_bytes=file_size,
            has_header=False
        )


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    dataset: str
    method: str
    num_elements: int
    original_mb: float
    num_partitions: int
    avg_partition_size: float
    avg_delta_bits: float
    compression_ratio: float
    decompress_gbps: float
    status: str
    error: str = ""


def parse_v3_output(output: str, dataset_name: str) -> Optional[BenchmarkResult]:
    """Parse V3 benchmark output."""
    lines = output.split('\n')
    result = BenchmarkResult(
        dataset=dataset_name, method="V3",
        num_elements=0, original_mb=0, num_partitions=0,
        avg_partition_size=0, avg_delta_bits=0,
        compression_ratio=0, decompress_gbps=0,
        status="FAIL"
    )

    for line in lines:
        if "Elements:" in line:
            result.num_elements = int(line.split(':')[1].strip())
        elif "Original size:" in line:
            result.original_mb = float(line.split(':')[1].strip().replace(" MB", ""))
        elif "Compression ratio:" in line:
            result.compression_ratio = float(line.split(':')[1].strip().replace("x", ""))
        elif "Partitions:" in line and "Avg" not in line:
            result.num_partitions = int(line.split(':')[1].strip())
        elif "Avg partition size:" in line:
            result.avg_partition_size = float(line.split(':')[1].strip())
        elif "Avg delta bits:" in line:
            result.avg_delta_bits = float(line.split(':')[1].strip())
        elif "Decompress kernel:" in line:
            parts = line.split('(')
            if len(parts) > 1:
                result.decompress_gbps = float(parts[1].replace(" GB/s)", "").strip())
        elif "Correctness:" in line:
            result.status = "PASS" if "PASS" in line else "FAIL"

    return result


def parse_pipe_output(output: str, dataset_name: str, method: str) -> Optional[BenchmarkResult]:
    """Parse pipe-separated output (V1, V2, Fixed)."""
    for line in output.split('\n'):
        if dataset_name in line and '|' in line:
            parts = line.split('|')
            if len(parts) >= 7:
                return BenchmarkResult(
                    dataset=parts[0].strip(),
                    method=method,
                    num_elements=0,  # Not provided in pipe format
                    original_mb=0,
                    num_partitions=int(parts[1]) if parts[1].isdigit() else 0,
                    avg_partition_size=float(parts[2]) if parts[2].replace('.', '').isdigit() else 0,
                    avg_delta_bits=float(parts[3]) if parts[3].replace('.', '').isdigit() else 0,
                    compression_ratio=float(parts[4]) if parts[4].replace('.', '').isdigit() else 0,
                    decompress_gbps=float(parts[5]) if parts[5].replace('.', '').isdigit() else 0,
                    status=parts[6].strip() if len(parts) > 6 else "UNKNOWN"
                )
    return None


def run_benchmark_for_dataset(ds: dict, info: DatasetInfo) -> List[BenchmarkResult]:
    """Run all benchmark methods for a single dataset."""
    results = []

    methods = [
        ("Fixed", f"{BIN_DIR}/benchmark_fixed"),
        ("V2", f"{BIN_DIR}/benchmark_v2"),
        ("V3", f"{BIN_DIR}/benchmark_v3"),
    ]

    for method_name, binary in methods:
        if not os.path.exists(binary):
            results.append(BenchmarkResult(
                dataset=info.name, method=method_name,
                num_elements=info.num_elements, original_mb=info.original_size_mb,
                num_partitions=0, avg_partition_size=0, avg_delta_bits=0,
                compression_ratio=0, decompress_gbps=0,
                status="SKIP", error="Binary not found"
            ))
            continue

        print(f"  Running {method_name}...", end=" ", flush=True)
        try:
            start = time.time()
            result = subprocess.run(
                [binary, DATA_DIR],
                capture_output=True, text=True, timeout=600
            )
            elapsed = time.time() - start

            output = result.stdout + result.stderr

            if method_name == "V3":
                # V3 outputs detailed format
                parsed = parse_v3_output(output, info.name)
            else:
                parsed = parse_pipe_output(output, info.name, method_name)

            if parsed:
                parsed.num_elements = info.num_elements
                parsed.original_mb = info.original_size_mb
                results.append(parsed)
                print(f"{parsed.compression_ratio:.2f}x, {parsed.decompress_gbps:.1f} GB/s, {parsed.status}")
            else:
                results.append(BenchmarkResult(
                    dataset=info.name, method=method_name,
                    num_elements=info.num_elements, original_mb=info.original_size_mb,
                    num_partitions=0, avg_partition_size=0, avg_delta_bits=0,
                    compression_ratio=0, decompress_gbps=0,
                    status="PARSE_FAIL", error="Could not parse output"
                ))
                print("PARSE_FAIL")

        except subprocess.TimeoutExpired:
            results.append(BenchmarkResult(
                dataset=info.name, method=method_name,
                num_elements=info.num_elements, original_mb=info.original_size_mb,
                num_partitions=0, avg_partition_size=0, avg_delta_bits=0,
                compression_ratio=0, decompress_gbps=0,
                status="TIMEOUT", error="Execution timeout"
            ))
            print("TIMEOUT")
        except Exception as e:
            results.append(BenchmarkResult(
                dataset=info.name, method=method_name,
                num_elements=info.num_elements, original_mb=info.original_size_mb,
                num_partitions=0, avg_partition_size=0, avg_delta_bits=0,
                compression_ratio=0, decompress_gbps=0,
                status="ERROR", error=str(e)
            ))
            print(f"ERROR: {e}")

    return results


def generate_markdown_report(results: List[BenchmarkResult], gpu_info: str) -> str:
    """Generate a comprehensive markdown report."""
    report = []
    report.append("# L3 Comprehensive Benchmark Report")
    report.append("")
    report.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**GPU:** {gpu_info}")
    report.append("")

    # Summary table
    report.append("## Summary Table")
    report.append("")
    report.append("| # | Dataset | Type | Elements | Size(MB) | Method | Partitions | Avg Size | Bits | Ratio | Decomp GB/s | Status |")
    report.append("|---|---------|------|----------|----------|--------|------------|----------|------|-------|-------------|--------|")

    for r in results:
        ds_info = next((d for d in DATASETS if d["name"] == r.dataset), None)
        ds_id = ds_info["id"] if ds_info else "-"
        dtype = ds_info["dtype"] if ds_info else "-"

        report.append(f"| {ds_id} | {r.dataset} | {dtype} | {r.num_elements:,} | {r.original_mb:.1f} | {r.method} | {r.num_partitions:,} | {r.avg_partition_size:.1f} | {r.avg_delta_bits:.1f} | {r.compression_ratio:.2f}x | {r.decompress_gbps:.1f} | {r.status} |")

    report.append("")

    # Method comparison
    report.append("## Method Comparison")
    report.append("")

    methods = ["Fixed", "V2", "V3"]
    for method in methods:
        method_results = [r for r in results if r.method == method and r.status == "PASS"]
        if method_results:
            avg_ratio = sum(r.compression_ratio for r in method_results) / len(method_results)
            avg_throughput = sum(r.decompress_gbps for r in method_results) / len(method_results)
            max_ratio = max(r.compression_ratio for r in method_results)
            max_throughput = max(r.decompress_gbps for r in method_results)

            report.append(f"### {method}")
            report.append(f"- **Datasets tested:** {len(method_results)}")
            report.append(f"- **Avg Compression Ratio:** {avg_ratio:.2f}x (max: {max_ratio:.2f}x)")
            report.append(f"- **Avg Decompression:** {avg_throughput:.1f} GB/s (max: {max_throughput:.1f} GB/s)")
            report.append("")

    # Dataset breakdown by type
    report.append("## Dataset Analysis")
    report.append("")

    # Synthetic datasets (1-4)
    report.append("### Synthetic Datasets (1-4)")
    synthetic = [r for r in results if any(d["id"] <= 4 and d["name"] == r.dataset for d in DATASETS)]
    if synthetic:
        report.append("| Dataset | Method | Ratio | Throughput |")
        report.append("|---------|--------|-------|------------|")
        for r in sorted(synthetic, key=lambda x: (x.dataset, x.method)):
            if r.status == "PASS":
                report.append(f"| {r.dataset} | {r.method} | {r.compression_ratio:.2f}x | {r.decompress_gbps:.1f} GB/s |")
    report.append("")

    # Real-world datasets (5-13)
    report.append("### Real-World Datasets (5-13)")
    realworld = [r for r in results if any(5 <= d["id"] <= 13 and d["name"] == r.dataset for d in DATASETS)]
    if realworld:
        report.append("| Dataset | Method | Ratio | Throughput |")
        report.append("|---------|--------|-------|------------|")
        for r in sorted(realworld, key=lambda x: (x.dataset, x.method)):
            if r.status == "PASS":
                report.append(f"| {r.dataset} | {r.method} | {r.compression_ratio:.2f}x | {r.decompress_gbps:.1f} GB/s |")
    report.append("")

    # Extended datasets (14-20)
    report.append("### Extended Datasets (14-20)")
    extended = [r for r in results if any(d["id"] >= 14 and d["name"] == r.dataset for d in DATASETS)]
    if extended:
        report.append("| Dataset | Method | Ratio | Throughput |")
        report.append("|---------|--------|-------|------------|")
        for r in sorted(extended, key=lambda x: (x.dataset, x.method)):
            if r.status == "PASS":
                report.append(f"| {r.dataset} | {r.method} | {r.compression_ratio:.2f}x | {r.decompress_gbps:.1f} GB/s |")
    report.append("")

    # Best method per dataset
    report.append("## Best Method per Dataset")
    report.append("")
    report.append("| Dataset | Best Ratio | Best Throughput |")
    report.append("|---------|------------|-----------------|")

    dataset_names = sorted(set(r.dataset for r in results))
    for ds_name in dataset_names:
        ds_results = [r for r in results if r.dataset == ds_name and r.status == "PASS"]
        if ds_results:
            best_ratio = max(ds_results, key=lambda x: x.compression_ratio)
            best_throughput = max(ds_results, key=lambda x: x.decompress_gbps)
            report.append(f"| {ds_name} | {best_ratio.method} ({best_ratio.compression_ratio:.2f}x) | {best_throughput.method} ({best_throughput.decompress_gbps:.1f} GB/s) |")

    report.append("")
    report.append("---")
    report.append("*Generated by L3 Benchmark Suite*")

    return '\n'.join(report)


def get_gpu_info() -> str:
    """Get GPU information."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Unknown GPU"


def main():
    print("=" * 60)
    print("L3 Comprehensive Benchmark - All 20 SOSD Datasets")
    print("=" * 60)

    gpu_info = get_gpu_info()
    print(f"GPU: {gpu_info}")
    print()

    all_results = []

    for ds in DATASETS:
        path = os.path.join(DATA_DIR, ds["file"])
        print(f"\n[{ds['id']}/20] {ds['name']} ({ds['dtype']})")

        info = get_dataset_info(path, ds["name"], ds["dtype"])
        if info is None:
            print(f"  SKIP: File not found: {path}")
            continue

        print(f"  Elements: {info.num_elements:,}")
        print(f"  Size: {info.original_size_mb:.2f} MB")

        results = run_benchmark_for_dataset(ds, info)
        all_results.extend(results)

    # Generate report
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)

    report = generate_markdown_report(all_results, gpu_info)

    # Save report
    report_path = "/root/autodl-tmp/code/L3/reports/L3/All/comprehensive_report.md"
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Also save JSON results
    json_path = "/root/autodl-tmp/code/L3/reports/L3/All/results.json"
    with open(json_path, 'w') as f:
        json.dump([{
            "dataset": r.dataset,
            "method": r.method,
            "num_elements": r.num_elements,
            "original_mb": r.original_mb,
            "num_partitions": r.num_partitions,
            "avg_partition_size": r.avg_partition_size,
            "avg_delta_bits": r.avg_delta_bits,
            "compression_ratio": r.compression_ratio,
            "decompress_gbps": r.decompress_gbps,
            "status": r.status,
            "error": r.error
        } for r in all_results], f, indent=2)
    print(f"JSON results saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    passed = sum(1 for r in all_results if r.status == "PASS")
    total = len(all_results)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed/Skipped: {total - passed}")


if __name__ == "__main__":
    main()
