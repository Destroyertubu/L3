# GLECO Architecture

## System Overview

GLECO consists of three main components:

### 1. Compression Pipeline

```
Raw Data → Partitioning → Model Fitting → Delta Encoding → Bit Packing → Compressed Data
```

**Partitioning**:
- Fixed-size or adaptive partitioning
- Default partition size: 1024 elements
- Optimized for GPU warp size (32 threads)

**Model Fitting**:
- Linear regression per partition
- Parameters: θ₀ (intercept), θ₁ (slope)
- Prediction: ŷ = θ₀ + θ₁ × i

**Delta Encoding**:
- Compute: δ = actual - predicted
- Variable-bit-width encoding
- Automatic overflow detection

### 2. Decompression Kernels

**Phase 1: Partition-Level Decompression**
- Load partition metadata
- Decode all elements in partition
- Output to shared memory

**Phase 2: Random Access Decompression**
- Binary search to locate partition
- Decode single element on-demand
- <100ns latency per access

### 3. Query Processing

**Predicate Pushdown**:
- Partition pruning using min/max bounds
- Skip partitions that don't match predicates
- Up to 90% partition reduction

**Two-Stage Query Execution**:
- Stage 1: Scan primary filter column
- Stage 2: Random access other columns
- Minimize decompression overhead

## Memory Layout

### Compressed Data Structure (SoA)

```cpp
struct CompressedDataGLECO<T> {
    // Partition metadata arrays
    int32_t* d_start_indices;       // Start index per partition
    int32_t* d_end_indices;         // End index per partition
    int32_t* d_delta_bits;          // Bits per delta
    double*  d_model_params;        // Model θ₀, θ₁ per partition
    int64_t* d_delta_array_bit_offsets;  // Bit offset in delta array

    // Compressed data
    uint32_t* delta_array;          // Bit-packed deltas
    long long* d_plain_deltas;      // Optional uncompressed deltas

    // Partition bounds (for pruning)
    T* d_partition_min_values;
    T* d_partition_max_values;
};
```

## Performance Characteristics

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| Compression | 5-10 GB/s | N/A |
| Full Decompression | 50-100 GB/s | N/A |
| Random Access | 1-5M ops/s | <100 ns |
| Predicate Eval | 40-80 GB/s | N/A |

## GPU Optimization Techniques

1. **Warp-Level Primitives**: Cooperative loading and decoding
2. **Shared Memory**: Cache partition metadata
3. **Coalesced Access**: Aligned memory reads
4. **Occupancy Tuning**: Balance registers vs shared memory
5. **Pipeline Overlap**: Hide memory latency

## Scalability

- **Elements**: Tested up to 1B elements
- **Partitions**: Scales to 100K+ partitions
- **Multi-GPU**: Ready for data parallelism
- **Compression Ratio**: 3-10x depending on data distribution
