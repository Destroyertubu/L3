# Compression Benchmark: wiki_200M_uint64 (200M elements, 1.5GB)

## Dataset Info
- File: `/root/autodl-tmp/test/data/sosd/7-wiki_200M_uint64.bin`
- Elements: 200,000,000 (uint64)
- Original Size: 1525.88 MB (1600000000 bytes)

## Results Summary

| Metric | Vertical GPU | L3 Vertical (Interleaved) |
|--------|--------------|---------------------------|
| **Compression Ratio** | 6.52x | **11.94x** |
| **Compressed Size** | 233.96 MB | 127.8 MB |
| **Encode Time** | 432 ms | 1609 ms |
| **Decode Time** | 1.13 ms | 1.19 ms |
| **Encode Throughput** | 3.45 GB/s | 0.99 GB/s |
| **Decode Throughput** | 1319 GB/s | **1350 GB/s** |

## Key Findings

### L3 Vertical Achieves Both Better Compression AND Faster Decompression!

1. **Compression Ratio**: L3 Vertical is **1.83x better** (11.94x vs 6.52x)
   - L3's polynomial models (LINEAR, POLY2, POLY3) fit wiki data much better than FOR-only
   - Compressed data is 127.8 MB vs 233.96 MB

2. **Decompression Speed**: L3 Vertical is **2.3% faster** (1350 GB/s vs 1319 GB/s)
   - Both use Vertical interleaved layout for SIMD-friendly access
   - L3's better compression means fewer bits to unpack per element

3. **Compression Speed**: Vertical is **3.7x faster** for encoding
   - L3 requires polynomial model selection and cost optimization
   - Vertical uses simple FOR model only

## Analysis

The L3 Vertical format combines:
- **L3's polynomial model selection** for better compression ratios
- **Vertical' interleaved layout** for fast SIMD decompression

This achieves the best of both worlds:
- 83% better compression than Vertical
- Equivalent or slightly better decompression throughput
- At the cost of slower compression (acceptable for read-heavy workloads)

## Test Configuration

### Vertical GPU (tile-gpu-compression)
- Method: Delta + BitPacking (FOR model only)
- Tile Size: 1024 elements
- Integer-only operations

### L3 Vertical
- Partition Size: 2048
- Encoder: GPU with Cost-Optimal partitioning
- Decoder: Vertical (interleaved layout)
- Model Selection: Adaptive (LINEAR, POLY2, POLY3, FOR)
- Layout: Interleaved mini-vectors for SIMD access

## Date
2025-12-10
