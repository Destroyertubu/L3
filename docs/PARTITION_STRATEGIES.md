# Partition Strategies Guide

L3 provides flexible partitioning strategies to optimize compression for different data characteristics.

## Overview

Partitioning divides data into segments before compression. The partition strategy significantly affects:
- **Compression ratio**: How much data is reduced
- **Compression speed**: Time to compress
- **Decompression speed**: Time to decompress
- **Random access performance**: Speed of accessing individual values

## Available Strategies

### 1. Fixed-Size Partitioning

Creates uniform-sized partitions.

```cpp
l3::FixedSizePartitioner partitioner(4096);  // 4K elements per partition
```

**Characteristics**:
- All partitions have the same size (except possibly the last)
- O(1) partition computation
- Predictable memory access patterns

**Best For**:
- Uniformly distributed data
- Sorted or nearly-sorted data
- When predictable performance is needed
- When partitioning speed is critical

**Example Data**:
- Sequential IDs: `[1, 2, 3, 4, 5, ...]`
- Timestamps: `[1000, 1001, 1002, ...]`
- Sorted datasets

### 2. Variable-Length Partitioning

Adaptive partitions based on data variance.

```cpp
l3::VariableLengthPartitioner partitioner(
    1024,  // base_size: base partition size
    8,     // variance_multiplier: variance analysis block size = base * multiplier
    3      // num_thresholds: number of variance buckets - 1
);
```

**Algorithm**:
1. Analyze data variance in large blocks
2. Compute variance thresholds (e.g., 25th, 50th, 75th percentiles)
3. Assign partition sizes based on variance:
   - **High variance** → Small partitions (better model fitting)
   - **Low variance** → Large partitions (faster processing)

**Characteristics**:
- Partition sizes vary from `base_size/2` to `base_size*4`
- O(n) variance analysis
- Better compression for non-uniform data

**Best For**:
- Non-uniform data distribution
- Time series with varying patterns
- Data with hot/cold regions
- When compression ratio is priority

**Example Data**:
- Stock prices: periods of stability + volatility
- Sensor data: steady states + sudden changes
- Mixed workload logs

## Choosing a Strategy

### Decision Tree

```
Is your data uniformly distributed?
├─ Yes → Use FixedSizePartitioner
│         • Simple and fast
│         • Predictable performance
│
└─ No → Consider data characteristics
    │
    ├─ Do you need maximum compression?
    │  └─ Yes → Use VariableLengthPartitioner
    │            • Better compression ratio
    │            • Adapts to data patterns
    │
    └─ Is partitioning speed critical?
       └─ Yes → Use FixedSizePartitioner
                 • O(1) partitioning
                 • No variance analysis overhead
```

### Quick Comparison

| Aspect | Fixed-Size | Variable-Length |
|--------|-----------|-----------------|
| Compression Ratio | Good (3.0-3.5x) | Better (3.5-4.5x) |
| Compression Speed | Fast | Medium |
| Partitioning Overhead | Minimal | Moderate |
| Data Adaptability | No | Yes |
| Predictability | High | Medium |
| Best Use Case | Uniform data | Non-uniform data |

## Parameter Tuning

### Fixed-Size Partitioning

**partition_size** (default: 4096)

```cpp
l3::FixedSizePartitioner p1(2048);  // Smaller partitions
l3::FixedSizePartitioner p2(4096);  // Default
l3::FixedSizePartitioner p3(8192);  // Larger partitions
```

**Guidelines**:
- **Smaller (1K-2K)**: Better compression, slower
- **Medium (4K)**: Balanced (recommended)
- **Larger (8K+)**: Faster, slightly worse compression

### Variable-Length Partitioning

**base_size** (default: 1024)
- Base for geometric progression
- Smaller → more granular adaptation
- Larger → less overhead

**variance_multiplier** (default: 8)
- Analysis block size = base_size × variance_multiplier
- Smaller → more fine-grained variance analysis
- Larger → faster analysis, less precise

**num_thresholds** (default: 3)
- Number of variance buckets - 1
- More thresholds → more partition size variations
- Fewer thresholds → simpler, faster

**Example Configurations**:

```cpp
// Aggressive compression (slower, best ratio)
l3::VariableLengthPartitioner aggressive(512, 16, 5);

// Balanced (default)
l3::VariableLengthPartitioner balanced(1024, 8, 3);

// Fast (speed priority)
l3::VariableLengthPartitioner fast(2048, 4, 2);
```

## Auto Selection

Let L3 analyze your data and choose automatically:

```cpp
auto partitioner = l3::PartitionerFactory::createAuto(
    data,
    size,
    element_size
);
```

**Analysis Process**:
1. Sample data distribution
2. Compute variance statistics
3. Estimate compression potential
4. Select optimal strategy

## Custom Strategies

Implement your own partitioning logic:

```cpp
class MyPartitioner : public l3::PartitionStrategy {
public:
    std::vector<l3::PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size) override
    {
        std::vector<l3::PartitionInfo> partitions;

        // Your custom partitioning logic
        // Example: partition by data value ranges

        return partitions;
    }

    const char* getName() const override {
        return "MyCustom";
    }

    l3::PartitionConfig getConfig() const override {
        return l3::PartitionConfig();
    }
};
```

## Performance Tips

### 1. Profile Your Data
```cpp
// Analyze variance
double variance = computeVariance(data, size);

if (variance < threshold) {
    // Use fixed-size
} else {
    // Use variable-length
}
```

### 2. Benchmark Different Configurations
```bash
# Run comparison benchmark
./bin/benchmarks/bench_compare_partitioners
```

### 3. Consider Access Patterns
- **Sequential access**: Fixed-size works well
- **Random access**: Variable-length may help
- **Range queries**: Consider data distribution

### 4. Monitor Compression Ratio
```cpp
auto partitions1 = fixed.partition(...);
auto partitions2 = variable.partition(...);

// Compare partition count and sizes
// More partitions ≠ better compression
// Optimal is data-dependent
```

## Examples

See [examples/cpp/01_partition_strategies.cpp](../examples/cpp/01_partition_strategies.cpp) for a complete working example demonstrating all strategies.

## Further Reading

- [Getting Started](GETTING_STARTED.md)
- [Development Guide](DEVELOPMENT.md)
- [API Reference](api/)
