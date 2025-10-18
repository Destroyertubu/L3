# Development Guide

This guide is for contributors and developers working on L3.

## Project Status

**Current Phase**: Phase 1 Complete (15%)

### Completed ✅
- Core partition strategy interface
- FixedSizePartitioner implementation
- VariableLengthPartitioner skeleton
- CMake build system
- Example programs
- Documentation structure

### In Progress 🔄
- VariableLengthPartitioner GPU kernels migration
- Compression/decompression module integration

### Planned 📋
- Random access module
- Query execution module
- Python bindings
- Complete test suite

## Development Roadmap

### Phase 1: Core Interface (DONE ✅)
- [x] Directory structure
- [x] Partition strategy interface
- [x] FixedSizePartitioner
- [x] Build system
- [x] Examples

### Phase 2: Complete Partitioner (In Progress 🔄)
- [x] Interface design
- [ ] GPU kernels migration from `lib/single_file/include/l3/partitioner_impl.cuh`
  - [ ] `analyzeDataVarianceFast` kernel
  - [ ] `countPartitionsPerBlock` kernel
  - [ ] `writePartitionsOrdered` kernel
- [ ] Unit tests
- [ ] Performance tests

**Estimated**: 2-3 days

### Phase 3: Compression API (Planned 📋)
- [ ] Migrate encoder.cu
- [ ] Migrate decompression kernels
- [ ] Create unified compression API
- [ ] Integrate partition strategy selection
- [ ] Tests

**Estimated**: 1 week

### Phase 4: Additional Modules (Planned 📋)
- [ ] Random access module
- [ ] Query execution module
- [ ] Utilities migration

**Estimated**: 1-2 weeks

### Phase 5: Python & Testing (Planned 📋)
- [ ] pybind11 bindings
- [ ] Python API
- [ ] Comprehensive test suite
- [ ] Benchmarks

**Estimated**: 2 weeks

## Project Structure

```
L3/
├── include/l3/              # Public API
│   ├── partitioner.hpp      # Partition strategies
│   ├── compression.hpp      # Compression API (TODO)
│   ├── decompression.hpp    # Decompression API (TODO)
│   └── internal/            # Internal headers
│
├── src/                     # Implementation
│   ├── partitioner/         # Partitioning (DONE)
│   ├── compression/         # Compression (TODO)
│   ├── decompression/       # Decompression (TODO)
│   ├── random_access/       # Random access (TODO)
│   └── utils/               # Utilities (TODO)
│
├── examples/                # Examples (expanding)
├── tests/                   # Tests (TODO)
├── benchmarks/              # Benchmarks (TODO)
├── docs/                    # Documentation
└── python/                  # Python bindings (TODO)
```

## Building for Development

### Debug Build

```bash
mkdir build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DBUILD_TESTS=ON \
         -DBUILD_EXAMPLES=ON
make -j$(nproc)
```

### With Compiler Warnings

```bash
cmake .. -DCMAKE_CXX_FLAGS="-Wall -Wextra -Wpedantic"
```

### With CUDA Debug Info

```bash
cmake .. -DCMAKE_CUDA_FLAGS="-G -g"
```

## Contributing

### Code Style

- C++17 standard
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- 4-space indentation
- Maximum line length: 100 characters

### Adding a New Partition Strategy

1. **Create header in** `include/l3/partitioner.hpp`:

```cpp
class MyNewPartitioner : public PartitionStrategy {
public:
    MyNewPartitioner(/* parameters */);

    std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size) override;

    const char* getName() const override;
    PartitionConfig getConfig() const override;

private:
    // Private members
};
```

2. **Implement in** `src/partitioner/my_new_partitioner.cu`:

```cpp
#include "l3/partitioner.hpp"

MyNewPartitioner::MyNewPartitioner(/* params */) {
    // Initialize
}

std::vector<PartitionInfo> MyNewPartitioner::partition(...) {
    // Implementation
}

// ... other methods
```

3. **Update factory** in `src/partitioner/variable_length_partitioner.cu`:

```cpp
// Add new enum
enum Strategy {
    FIXED_SIZE,
    VARIABLE_LENGTH,
    MY_NEW_STRATEGY,  // Add this
    AUTO
};

// Update create()
case MY_NEW_STRATEGY:
    return std::make_unique<MyNewPartitioner>(...);
```

4. **Add tests** in `tests/unit/test_my_new_partitioner.cu`

5. **Add example** in `examples/cpp/`

### Testing

```bash
# Build with tests
cmake .. -DBUILD_TESTS=ON
make

# Run all tests
ctest --verbose

# Run specific test
./bin/tests/test_partitioner
```

### Benchmarking

```bash
# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON
make

# Run benchmarks
./bin/benchmarks/bench_compare_partitioners
```

## Migration Tasks

### Current Priority: Variable-Length Partitioner GPU Kernels

**Source**: `lib/single_file/include/l3/partitioner_impl.cuh` (lines 39-322)

**Target**: `src/partitioner/variable_length_partitioner.cu`

**Kernels to migrate**:
1. `analyzeDataVarianceFast` - Variance analysis
2. `countPartitionsPerBlock` - Partition counting
3. `writePartitionsOrdered` - Ordered partition writing
4. `fitPartitionsBatched_Optimized` - Model fitting

**Steps**:
1. Copy kernel implementations
2. Update includes and namespaces
3. Test correctness
4. Benchmark performance

See [MIGRATION.md](MIGRATION.md) for detailed migration status.

## Useful Commands

### Find TODOs

```bash
grep -r "TODO" src/ include/ --include="*.cpp" --include="*.cu" --include="*.hpp"
```

### Count Lines of Code

```bash
find src/ include/ -name "*.cpp" -o -name "*.cu" -o -name "*.hpp" | xargs wc -l
```

### Format Code (if using clang-format)

```bash
find src/ include/ -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

## Documentation

### Building Documentation

```bash
# TODO: Add doxygen support
```

### Documentation Standards

- All public APIs must have Doxygen comments
- Include usage examples in comments
- Document parameters, return values, and exceptions
- Add cross-references to related functions

## Getting Help

- Check [MIGRATION.md](MIGRATION.md) for migration status
- See [PARTITION_STRATEGIES.md](PARTITION_STRATEGIES.md) for design details
- Open an issue on GitHub for questions

## Quick Reference

| Task | Command |
|------|---------|
| Build | `make -j$(nproc)` |
| Test | `ctest --verbose` |
| Clean | `make clean` |
| Rebuild | `rm -rf build && mkdir build && cd build && cmake .. && make` |
| Run example | `./bin/examples/example_partition_strategies` |

## Release Checklist

Before releasing a new version:

- [ ] All tests pass
- [ ] Benchmarks run successfully
- [ ] Documentation updated
- [ ] CHANGELOG updated
- [ ] Version numbers updated
- [ ] Examples tested
- [ ] Migration guide updated (if applicable)
