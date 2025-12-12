# L3 Benchmarks

Performance benchmarks for L3 compression and query processing.

## Test Environment

- **GPU**: NVIDIA H20
- **CUDA**: 12.x
- **CPU**: Intel Xeon / AMD EPYC
- **Memory**: 96 GB GPU, 512 GB System
- **OS**: Linux (Ubuntu 22.04)

## SSB Benchmarks

### Scale Factor 20 Results

**Dataset**: 119,968,352 rows (LINEORDER table)

| Query | Execution Time | Candidates | Selectivity | Result Groups |
|-------|---------------|------------|-------------|---------------|
| Q1.1  | 2.83 ms      | 17,142,055 | 14.3%       | 1 |
| Q1.2  | 1.34 ms      | 1,429,167  | 1.19%       | 1 |
| Q1.3  | 1.35 ms      | 1,429,167  | 1.19%       | 1 |
| Q2.1  | 1.72 ms      | 2,734,303  | 2.28%       | 2,053 |
| Q2.2  | 1.43 ms      | 542,726    | 0.45%       | 56 |
| Q2.3  | 1.33 ms      | 69,882     | 0.058%      | 7 |
| Q3.1  | 1.44 ms      | 102,831,913| 85.7%       | 0 |
| Q3.2  | 1.92 ms      | 45,829     | 0.038%      | 5,599 |
| Q3.3  | 1.66 ms      | 2,099      | 0.0017%     | 24 |
| Q3.4  | 1.31 ms      | 1,428,491  | 1.19%       | 4 |
| Q4.1  | 2.86 ms      | 266,766    | 0.22%       | 175 |
| Q4.2  | 3.60 ms      | 9,886      | 0.0082%     | 1,249 |
| Q4.3  | 2.82 ms      | 1,906      | 0.0016%     | 965 |

**Average Query Time**: 1.97 ms

### Compression Performance

| Column | Original Size | Compressed Size | Ratio | Time |
|--------|--------------|-----------------|-------|------|
| lo_orderdate | 457.5 MB | 52.3 MB | 8.75x | 0.76s |
| lo_discount  | 457.5 MB | 45.1 MB | 10.1x | 0.58s |
| lo_quantity  | 457.5 MB | 48.2 MB | 9.49x | 0.61s |
| lo_revenue   | 457.5 MB | 71.3 MB | 6.42x | 0.64s |

**Average Compression Ratio**: 8.7x
**Compression Throughput**: 600-700 MB/s

## Random Access Performance

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| Sequential Access | 85 GB/s | N/A |
| Random Access | 50 GB/s | 85 ns |
| Predicate Eval | 65 GB/s | N/A |

## Comparison with Baselines

### vs. Uncompressed

| Metric | Uncompressed | L3 | Speedup |
|--------|-------------|-------|---------|
| Storage | 1.8 GB | 217 MB | 8.3x |
| Q1.1 Time | 45 ms | 2.83 ms | 15.9x |
| Q2.1 Time | 78 ms | 1.72 ms | 45.3x |
| Q4.3 Time | 92 ms | 2.82 ms | 32.6x |

### vs. Traditional Compression (gzip, zstd)

| Method | Ratio | Decomp Speed | Random Access |
|--------|-------|--------------|---------------|
| gzip   | 12x   | 300 MB/s     | ❌ No         |
| zstd   | 10x   | 800 MB/s     | ❌ No         |
| L3  | 8.7x  | 85 GB/s      | ✅ Yes (85ns) |

## Scalability Tests

### Data Size Scaling

| Elements | Compression | Decompression | Query (avg) |
|----------|-------------|---------------|-------------|
| 1M       | 15 ms       | 2 ms          | 0.12 ms     |
| 10M      | 145 ms      | 18 ms         | 0.58 ms     |
| 100M     | 1.4s        | 175 ms        | 1.85 ms     |
| 1B       | 14.2s       | 1.7s          | 18.5 ms     |

### Partition Size Impact

| Partition Size | Compression Ratio | Query Time | Memory |
|----------------|-------------------|------------|--------|
| 256            | 7.2x              | 3.2 ms     | 280 MB |
| 512            | 8.1x              | 2.5 ms     | 245 MB |
| 1024           | 8.7x              | 2.0 ms     | 217 MB |
| 2048           | 8.9x              | 1.9 ms     | 205 MB |
| 4096           | 9.0x              | 2.1 ms     | 198 MB |

**Recommended**: 1024-2048 elements per partition

## Running Benchmarks

```bash
# SSB benchmarks
cd tests/ssb
make run_all > ../../benchmarks/results/ssb_sf20.log

# Random access benchmarks
cd tests/ra
./test_random_access_comprehensive --max=100000000

# Compression benchmarks
cd build
./test_compression_main
```

## Result Files

- `results/ssb_sf20.log` - SSB query results
- `results/compression_stats.csv` - Compression statistics
- `results/random_access.csv` - Random access performance

## Notes

- All times are median of 5 runs
- GPU warmed up before measurements
- CUDA events used for timing
- Results may vary based on hardware

---

Last updated: October 2024
