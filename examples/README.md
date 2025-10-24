# GLECO Examples

This directory contains example code demonstrating how to use GLECO.

## Examples

1. **basic_compression.cpp** - Simple compression and decompression
2. **random_access_query.cu** - GPU random access example
3. **predicate_pushdown.cpp** - Partition pruning with predicates
4. **custom_query.cu** - Custom CUDA kernel with GLECO data

## Building Examples

```bash
cd examples
mkdir build && cd build
cmake ..
make
```

## Running Examples

```bash
# Basic compression
./basic_compression

# Random access
./random_access_query

# Predicate pushdown
./predicate_pushdown
```

## Example Descriptions

### basic_compression.cpp

Demonstrates:
- Loading data from file
- Compressing with GLECO
- Decompressing to verify correctness
- Printing compression statistics

### random_access_query.cu

Demonstrates:
- Creating compressed data on GPU
- Random access kernel
- Scatter/gather operations
- Performance measurement

### predicate_pushdown.cpp

Demonstrates:
- Computing partition bounds
- Evaluating predicates
- Partition pruning
- Optimized query execution

### custom_query.cu

Demonstrates:
- Writing custom CUDA kernels
- Using GLECO random access API
- Multi-column processing
- Result aggregation
