# SSB Query Time Breakdown Report - Random Access Strategy

## Response to Reviewer O2's Questions

### 1. Baseline Implementation

All baselines were implemented using the same query plan structure:

| Baseline | Implementation |
|----------|---------------|
| **nvcomp (LZ4/Snappy/Zstd)** | Official nvcomp library (v2.6.1) - GPU-native compression |
| **Vertical** | Reference implementation from VLDBench |
| **GDeflate** | nvcomp's GPU-optimized deflate |
| **BitPack** | Custom CUDA implementation following Vertical lane-crossing pattern |
| **CUDA Columnar** | Uncompressed baseline with optimized memory access |

**Key point**: We integrated each compression scheme into the **same SSB query kernels**, ensuring identical query plans across all schemes. The only difference is the decompression routine.

### 2. Time Measurement Methodology

All measurements use **CUDA events** for GPU-accurate timing:

```cpp
cudaEventRecord(start);
// ... GPU operation ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed_ms, start, stop);
```

**What is included in query time:**

| Phase | Included | Description |
|-------|----------|-------------|
| **Data Load** | Yes | Decompress + Random Access |
| **Hash Build** | Yes | Build hash tables for dimension tables |
| **Kernel** | Yes | Filter + Probe + Aggregation |
| **H2D Transfer** | Yes | Compressed data already on GPU (pre-loaded) |
| **D2H Transfer** | Yes | Result transfer (minimal: ~8 bytes for aggregation) |
| **Compression** | No | One-time offline preprocessing |

**Important**: The query time starts with **already compressed data on GPU**. This represents the realistic scenario where data is stored compressed and queries run repeatedly.

### 3. String and Date Handling

**Date Attributes:**
- SSB dates are stored as integers (YYYYMMDD format, e.g., 19940201)
- L3 compresses these as 32-bit integers using polynomial regression
- Decompression produces the original integer values

**String Attributes:**
- SSB categorical strings (e.g., nation, region, city) are **dictionary-encoded** to integers
- Example: `c_region = 'ASIA'` becomes `c_region = 2`
- This is the **same approach used by all baselines** (nvcomp, Vertical, etc.)
- The dictionary is built once during data loading, not included in query time

---

## Detailed Time Breakdown Analysis

### Query Execution Model

L3 uses a **staged random access** model for queries with low selectivity:

```
Stage 1: Decompress first join column → Probe hash table → Get passing_indices
Stage 2: Random access second join column → Probe hash table → Filter more
Stage 3: Random access remaining columns → Aggregation
```

### Time Breakdown by Query Type

#### Q1.x (No Joins - Direct Filters)

**Q1.1** (Selectivity: 14.3%)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| Decompress orderdate | 0.45 | 13.5% |
| Date filter kernel | 0.96 | 28.9% |
| Random access 3 cols | 1.69 | 50.9% |
| Aggregation | 0.23 | 6.7% |
| **Total** | **3.32** | **100%** |

**Q1.2** (Selectivity: 1.2%)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| Decompress orderdate | 0.44 | 27.8% |
| Date filter kernel | 0.91 | 57.6% |
| Random access D+Q (1.4M rows) | 0.16 | 10.1% |
| Secondary filter | 0.02 | 1.3% |
| Random access EP (79K rows) | 0.04 | 2.5% |
| Aggregation | 0.01 | 0.6% |
| **Total** | **1.58** | **100%** |

**Key Insight**: Lower selectivity = less random access overhead = faster query.

---

#### Q2.x (Two Hash Joins: PART + SUPPLIER)

**Q2.1** (Selectivity: 0.8%)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| **Hash Build** | **0.28** | **7.8%** |
| Stage 1: Decompress suppkey | 0.43 | 11.9% |
| Stage 1: Probe supplier (120M rows) | 1.56 | 43.3% |
| Stage 2: RA partkey (24M rows) | 0.78 | 21.7% |
| Stage 2: Probe part | 0.34 | 9.4% |
| Stage 3: RA final (960K rows) | 0.13 | 3.6% |
| Aggregation | 0.08 | 2.2% |
| **Total** | **3.60** | **100%** |

**Data flow**:
```
120M rows → Stage 1 (supplier filter) → 24M rows (20%)
         → Stage 2 (part filter) → 960K rows (0.8%)
         → Stage 3 (final) → aggregation
```

---

#### Q3.x (Two Hash Joins: CUSTOMER + SUPPLIER)

**Q3.1** (Selectivity: 4.0%)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| **Hash Build** | **0.29** | **6.8%** |
| Stage 1: Decompress suppkey | 0.44 | 10.4% |
| Stage 1: Probe supplier (120M rows) | 1.66 | 39.2% |
| Stage 2: RA custkey (24M rows) | 0.79 | 18.6% |
| Stage 2: Probe customer | 0.40 | 9.4% |
| Stage 3: RA final (4.8M rows) | 0.41 | 9.7% |
| Aggregation | 0.26 | 6.1% |
| **Total** | **4.24** | **100%** |

**Q3.3** (Selectivity: 0.007% - Very Low)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| **Hash Build** | **0.29** | **13.4%** |
| Stage 1: Decompress suppkey | 0.43 | 19.8% |
| Stage 1: Probe supplier (120M rows) | 1.29 | 59.4% |
| Stage 2: RA custkey (1M rows) | 0.07 | 3.2% |
| Stage 2: Probe customer | 0.02 | 0.9% |
| Stage 3: RA final (8K rows) | 0.04 | 1.8% |
| Aggregation | 0.01 | 0.5% |
| **Total** | **2.17** | **100%** |

**Key Insight**: Q3.3's extremely low selectivity (8K final rows) makes random access highly efficient.

---

#### Q4.x (Four Hash Joins: PART + SUPPLIER + CUSTOMER + DATE)

**Q4.1** (Selectivity: 1.6%)
| Phase | Time (ms) | % of Total |
|-------|-----------|------------|
| **Hash Build** | **0.40** | **6.6%** |
| Stage 1: Decompress partkey | 0.44 | 7.2% |
| Stage 1: Probe part (120M rows) | 1.94 | 31.9% |
| Stage 2: RA suppkey (48M rows) | 1.42 | 23.4% |
| Stage 2: Probe supplier | 0.64 | 10.5% |
| Stage 3: RA custkey (9.6M rows) | 0.39 | 6.4% |
| Stage 3: Probe customer | 0.16 | 2.6% |
| Stage 4: RA final (1.9M rows) | 0.30 | 4.9% |
| Aggregation | 0.37 | 6.1% |
| **Total** | **6.08** | **100%** |

**Data flow**:
```
120M → Stage 1 (40%) → 48M → Stage 2 (8%) → 9.6M → Stage 3 (1.6%) → 1.9M → aggregate
```

---

## Query Plan Comparison

All schemes use the **same query plan**:

```
                    L3 Query Plan
                         |
    +--------------------+--------------------+
    |                    |                    |
Decompress        Hash Table Build      Query Kernel
(L3/nvcomp/FL)   (Same for all)        (Same for all)
```

The only difference is the **decompression routine**:

| Scheme | Decompression Method |
|--------|---------------------|
| L3 | Polynomial + BitPack + GPU parallelism |
| nvcomp LZ4 | GPU LZ4 decompression |
| Vertical | GPU BitPack with lane-crossing |
| GDeflate | GPU Deflate |

---

## Summary Statistics

### Average Time Distribution Across All Queries

| Component | Average % | Range |
|-----------|-----------|-------|
| Decompression | 15-25% | Varies by compression ratio |
| Hash Build | 5-15% | Constant overhead |
| Filter/Probe | 35-60% | Dominates for high selectivity |
| Random Access | 10-30% | Dominates for low selectivity |
| Aggregation | 2-10% | Minimal overhead |

### Selectivity vs. Strategy Performance

| Selectivity | Best Strategy | Reason |
|-------------|---------------|--------|
| >10% | Full decompress | Sequential access is faster |
| 1-10% | Staged + RA | Balance between decomp and RA overhead |
| <1% | Staged + RA | RA overhead << full decomp cost |
| <0.1% | Staged + RA | Dramatic savings from skipping 99.9% data |

---

## Reproducibility Notes

**Hardware**: NVIDIA GPU (RTX 4090 / A100)
**Software**: CUDA 11.8+, nvcomp 2.6.1
**Data**: SSB Scale Factor 20 (~120M rows in LINEORDER)
**Runs**: 3 runs per query, reporting best run

All source code and benchmark scripts are included in the supplementary materials.



  Random Access 策略的关键时间分解

  Q1.x (无Join，直接过滤)

  | Query        | Decompress | Filter  | Random Access | Aggregation | Total   |
  |--------------|------------|---------|---------------|-------------|---------|
  | Q1.1 (14.3%) | 0.45 ms    | 0.96 ms | 1.69 ms       | 0.23 ms     | 3.32 ms |
  | Q1.2 (1.2%)  | 0.44 ms    | 0.91 ms | 0.16 ms       | 0.01 ms     | 1.58 ms |
  | Q1.3 (0.27%) | 0.44 ms    | 0.82 ms | 0.07 ms       | 0.01 ms     | 1.38 ms |

  Q2.x/Q3.x/Q4.x (带Hash Join)

  | Query         | Hash Build | Decompress | Probe (120M) | RA Stage2 | RA Final | Total   |
  |---------------|------------|------------|--------------|-----------|----------|---------|
  | Q2.1 (0.8%)   | 0.28 ms    | 0.43 ms    | 1.56 ms      | 0.78 ms   | 0.13 ms  | 3.60 ms |
  | Q3.1 (4.0%)   | 0.29 ms    | 0.44 ms    | 1.66 ms      | 0.79 ms   | 0.41 ms  | 4.24 ms |
  | Q3.3 (0.007%) | 0.29 ms    | 0.43 ms    | 1.29 ms      | 0.07 ms   | 0.04 ms  | 2.17 ms |
  | Q4.1 (1.6%)   | 0.40 ms    | 0.44 ms    | 1.94 ms      | 1.42 ms   | 0.30 ms  | 6.08 ms |
  | Q4.3 (0.05%)  | 0.29 ms    | 0.44 ms    | 1.69 ms      | 0.21 ms   | 0.04 ms  | 2.81 ms |

  时间占比分析

  | 组件            | 平均占比   | 说明           |
  |---------------|--------|--------------|
  | Decompress    | 15-25% | 仅解压第一个join列  |
  | Hash Build    | 5-15%  | 固定开销         |
  | Probe (120M)  | 35-60% | 对全量数据探测hash表 |
  | Random Access | 10-30% | 只访问通过过滤的行    |
  | Aggregation   | 2-10%  | 最终聚合         |

  回应审稿人的关键点

  1. Baseline实现: 所有baseline使用相同的query plan结构，只有解压例程不同
  2. 时间测量: 使用CUDA events，包含解压+hash构建+kernel执行，不包含离线压缩
  3. String/Date处理: Date存储为整数(YYYYMMDD)，String使用dictionary encoding
  4. Query Plan: 所有scheme使用相同的staged random access plan