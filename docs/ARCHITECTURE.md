# L3 Architecture Overview

This document describes the architecture and design of the L3 compression system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Core Components](#core-components)
3. [Data Structures](#data-structures)
4. [Compression Pipeline](#compression-pipeline)
5. [Decompression Pipeline](#decompression-pipeline)
6. [Random Access](#random-access)
7. [GPU Optimization](#gpu-optimization)

## System Overview

L3 (GPU-based Learned Compression) is a compression system that uses learned piecewise linear regression models to compress sorted integer sequences.

### Key Design Principles

1. **GPU-First**: All performance-critical operations run on GPU
2. **Adaptive**: Automatically selects optimal model for each partition
3. **Lossless**: Exact reconstruction of original data
4. **Fast Random Access**: Access individual elements without full decompression

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    L3 System                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Compression  │  │Decompression │  │Random Access │      │
│  │   Pipeline   │  │   Pipeline   │  │   Engine     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼───────┐    │
│  │            L3 Core Library                      │    │
│  │  - Partitioning      - Model Fitting                │    │
│  │  - Delta Encoding    - Bit Packing                  │    │
│  └────────────────────┬────────────────────────────────┘    │
│                       │                                      │
│  ┌────────────────────▼────────────────────────────────┐    │
│  │         CUDA Kernels & GPU Primitives               │    │
│  │  - Parallel Scan    - Warp Operations               │    │
│  │  - CUB Primitives   - Shared Memory Optimization    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. L3 Library (`lib/l32/`)

Main compression library with optimized SoA (Structure of Arrays) layout.

**Key Files:**
- `l32.cu` - Complete implementation (3100+ lines)

**Key Functions:**
- `compress_gpu()` - Main compression entry point
- `decompress_gpu()` - Decompression
- `random_access_gpu()` - Random access without full decompression

### 2. Benchmark Suite (`benchmarks/`)

Performance evaluation on real workloads.

**SSB (Star Schema Benchmark):**
- `baseline/` - Basic implementations
- `optimized_2push/` - With predicate pushdown optimization

### 3. Common Headers (`include/common/`)

Shared utilities for benchmarks:
- `l3_alex_index.cuh` - ALEX learned index integration
- `ssb_l3_utils.cuh` - SSB-specific utilities
- `l3_ra_utils.cuh` - Random access helpers
- `ssb_utils.h` - Data loading and verification

### 4. Legacy Library (`lib/l3_legacy/`)

Original implementation for compatibility.

## Data Structures

### CompressedData (SoA Layout)

Structure-of-Arrays layout for GPU efficiency:

```cuda
template<typename T>
struct CompressedData {
    // Metadata arrays (all device pointers)
    int32_t* d_start_indices;        // Partition start positions
    int32_t* d_end_indices;          // Partition end positions
    int32_t* d_model_types;          // Model type per partition
    double*  d_model_params;         // Model parameters (flattened)
    int32_t* d_delta_bits;           // Bits per delta
    int64_t* d_delta_array_bit_offsets; // Bit offsets in delta array
    long long* d_error_bounds;       // Error bounds

    // Compressed data
    uint32_t* delta_array;           // Bit-packed deltas
    long long* d_plain_deltas;       // Uncompressed deltas (fallback)

    // Host metadata
    size_t original_size;
    size_t num_partitions;
    // ... additional metadata
};
```

### Partition Metadata

```cuda
struct PartitionInfo {
    int32_t start_idx;               // Start position in original array
    int32_t end_idx;                 // End position
    int32_t model_type;              // 0=Constant, 1=Linear, 2=Poly2, etc.
    double model_params[4];          // Model coefficients
    int32_t delta_bits;              // Bits per delta value
    int64_t delta_array_bit_offset;  // Offset in bit-packed array
    long long error_bound;           // Maximum allowed error
};
```

### Model Types

```cuda
enum ModelType {
    MODEL_CONSTANT = 0,      // f(x) = c
    MODEL_LINEAR = 1,        // f(x) = a + b*x
    MODEL_POLYNOMIAL2 = 2,   // f(x) = a + b*x + c*x²
    MODEL_POLYNOMIAL3 = 3,   // f(x) = a + b*x + c*x² + d*x³
    MODEL_DIRECT_COPY = 4    // No model, store raw (overflow case)
};
```

## Compression Pipeline

### Phase 1: Partitioning

Divides data into variable-size partitions based on data characteristics.

```
Input: Sorted array [v₀, v₁, v₂, ..., vₙ]
             ↓
    ┌────────────────┐
    │  Partitioning  │
    └────────┬───────┘
             ↓
Partitions: [P₀] [P₁] [P₂] ... [Pₖ]
```

**Algorithm:**
1. Start with candidate partition
2. Fit model to data
3. Check if errors exceed threshold
4. If yes, split; if no, continue
5. Adaptive thresholds based on data distribution

### Phase 2: Model Fitting

For each partition, select best-fitting model.

```
For each partition Pᵢ:
  1. Try constant model:  f(x) = c
  2. Try linear model:    f(x) = a + b*x
  3. Try polynomial:      f(x) = a + b*x + c*x² + ...
  4. Select model with smallest error
```

**Model Selection:**
- Uses least-squares regression
- Compares prediction accuracy
- Fallback to direct copy if no model fits well

### Phase 3: Delta Encoding

Compute prediction errors (deltas).

```
For each value vⱼ in partition:
  predicted = model(j)
  delta = actual - predicted
```

### Phase 4: Bit Packing

Pack deltas into compact bit array.

```
Deltas: [3, -1, 2, 0, -2, ...]
           ↓ Find max |delta|
Required bits: 2 (range -2 to +3)
           ↓ Zigzag encoding
Encoded: [6, 1, 4, 0, 3, ...]  (unsigned)
           ↓ Bit pack
Output: [01101001|0000011...]  (2 bits per value)
```

## Decompression Pipeline

### Sequential Decompression

```
1. For each partition:
   ├─ Read metadata (model type, params, bit offset)
   ├─ Locate bit-packed deltas
   ├─ Unpack deltas
   ├─ Apply model: value = model(index) + delta
   └─ Write to output array

2. GPU parallelization:
   - Each warp handles one partition
   - Cooperative unpacking using warp primitives
```

### Warp-Level Optimization

```cuda
// Each warp processes one partition
__global__ void decompress_kernel(...) {
    int partition_id = blockIdx.x;
    int lane_id = threadIdx.x % 32;

    // Warp reads metadata cooperatively
    __shared__ PartitionMetadata meta;
    if (lane_id == 0) meta = load_metadata(partition_id);
    __syncwarp();

    // Each thread handles subset of values
    for (int i = lane_id; i < partition_size; i += 32) {
        uint64_t delta = unpack_delta(meta.bit_offset, i);
        double predicted = evaluate_model(meta.params, i);
        output[i] = predicted + zigzag_decode(delta);
    }
}
```

## Random Access

Accessing element at index `i` without full decompression.

### Algorithm

```
1. Binary search: Find partition P containing index i
2. Load partition metadata
3. Compute local index: local_i = i - partition_start
4. Unpack delta at local_i from bit array
5. Evaluate model: predicted = model(local_i)
6. Reconstruct: value = predicted + delta
```

### Optimization

```cuda
__device__ int64_t random_access(CompressedData* data, size_t index) {
    // 1. Find partition (binary search on d_start_indices)
    int p = binary_search_partition(data, index);

    // 2. Load metadata
    int model_type = data->d_model_types[p];
    double* params = &data->d_model_params[p * 4];
    int local_idx = index - data->d_start_indices[p];

    // 3. Compute predicted value
    double predicted = evaluate_model(model_type, params, local_idx);

    // 4. Unpack delta
    int64_t bit_offset = data->d_delta_array_bit_offsets[p];
    int delta_bits = data->d_delta_bits[p];
    uint64_t packed = unpack_bits(data->delta_array,
                                    bit_offset + local_idx * delta_bits,
                                    delta_bits);

    // 5. Reconstruct
    int64_t delta = zigzag_decode(packed);
    return (int64_t)(predicted + delta);
}
```

## GPU Optimization

### Memory Layout

**SoA (Structure of Arrays) vs AoS (Array of Structures):**

```
AoS (Poor GPU performance):
struct Partition { int start; int end; int type; ... };
Partition partitions[N];  // Bad: scattered memory access

SoA (Optimized):
int* starts;    // All starts together
int* ends;      // All ends together
int* types;     // All types together
// Good: coalesced memory access
```

### Warp-Level Primitives

Uses CUDA warp operations for efficiency:

```cuda
// Collaborative bit unpacking
__device__ uint64_t unpack_warp(uint32_t* data, int bit_offset) {
    int lane = threadIdx.x % 32;
    uint32_t word = data[(bit_offset + lane * BITS) / 32];
    // Use __shfl_sync for warp-level communication
    return cooperative_extract_bits(word, lane);
}
```

### Shared Memory Optimization

```cuda
__global__ void decompress_kernel(...) {
    __shared__ double model_params[BLOCK_SIZE][4];
    __shared__ uint32_t bit_buffer[BUFFER_SIZE];

    // Load frequently accessed data to shared memory
    // Reduces global memory traffic
}
```

### Kernel Fusion

Combines multiple operations in single kernel:

```cuda
// Bad: Multiple kernel launches
compress_kernel<<<...>>>();
cudaDeviceSynchronize();
pack_kernel<<<...>>>();

// Good: Fused operation
compress_and_pack_kernel<<<...>>>();  // 50% faster
```

## Performance Characteristics

### Compression Ratio

- **Sorted sequences**: 50-200x
- **SSB columns**: 20-80x
- **General integers**: 5-50x

### Throughput

On RTX 4090:
- **Compression**: 10-20 GB/s
- **Decompression**: 15-30 GB/s
- **Random Access**: ~1M accesses/ms

### Scalability

- Scales with GPU parallelism
- Larger partitions → better GPU utilization
- 1M+ elements → optimal performance

## See Also

- [Installation Guide](INSTALLATION.md) - Setup instructions
- [Performance Guide](PERFORMANCE.md) - Optimization tips
- [API Reference](API.md) - Function documentation
- [SSB Benchmark](SSB_BENCHMARK.md) - Benchmark details
