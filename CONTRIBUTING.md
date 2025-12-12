# Contributing to L3

Thank you for your interest in contributing to L3! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)

## Code of Conduct

Please be respectful and constructive in all interactions with the community.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/L3_opt.git
   cd L3_opt
   ```

2. **Set up development environment**
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   make -j$(nproc)
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Process

### Before You Start

- Check existing issues and PRs to avoid duplicate work
- Open an issue to discuss major changes before implementing
- Keep changes focused and atomic

### Project Structure

```
L3_opt/
├── include/          # Public header files
├── src/
│   ├── codec/       # Compression/decompression implementation
│   ├── kernels/     # CUDA kernels
│   └── tests/       # Unit tests
├── tests/           # Integration tests (SSB, RA)
├── docs/            # Documentation
├── examples/        # Example code
└── benchmarks/      # Performance benchmarks
```

## Coding Standards

### C++ Style

- Follow C++17 standards
- Use descriptive variable names
- Add comments for complex logic
- Keep functions focused and small

**Example:**
```cpp
// Good
int computePartitionIndex(int global_idx, int partition_size) {
    return global_idx / partition_size;
}

// Avoid
int calc(int x, int y) {  // Unclear what this does
    return x / y;
}
```

### CUDA Style

- Use `__device__` for device-only functions
- Use `__global__` for kernel entry points
- Prefix shared memory variables with `s_`
- Prefix device pointers with `d_`

**Example:**
```cuda
__global__ void decompressKernel(
    const uint32_t* d_compressed_data,
    uint32_t* d_output
) {
    __shared__ uint32_t s_cache[256];
    // ...
}
```

### Naming Conventions

- Classes: `PascalCase` (e.g., `CompressedDataL3`)
- Functions: `camelCase` (e.g., `compressData`)
- Variables: `snake_case` (e.g., `partition_size`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_PARTITION_SIZE`)
- CUDA kernels: `snake_case` with `_kernel` suffix

### Documentation

- Add header comments to all public functions
- Use Doxygen-style comments for API documentation
- Include usage examples for complex features

**Example:**
```cpp
/**
 * Compress data using L3 learned compression
 *
 * @param h_data Input data vector
 * @param partition_size Elements per partition (default: 2048)
 * @param stats Optional output for compression statistics
 * @return Pointer to compressed data structure
 *
 * @example
 * std::vector<uint32_t> data = {1, 2, 3, 4, 5};
 * auto compressed = compressData(data, 1024);
 */
template<typename T>
CompressedDataL3<T>* compressData(
    const std::vector<T>& h_data,
    int partition_size = 2048,
    CompressionStats* stats = nullptr
);
```

## Testing

### Running Tests

```bash
# Build tests
cd build
make

# Run all tests
cd tests/ssb
make run_all

# Run specific test
./q11_2push 1
```

### Adding New Tests

1. Create test file in appropriate directory:
   - Unit tests: `src/tests/`
   - SSB tests: `tests/ssb/`
   - Random access tests: `tests/ra/`

2. Follow naming convention: `test_<feature>.cpp` or `test_<feature>.cu`

3. Include appropriate test assertions

### Performance Tests

- Always measure performance impact of changes
- Compare with baseline on same hardware
- Document any performance changes in PR

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
Add partition pruning for range predicates

- Implement min/max bounds computation
- Add predicate evaluation kernel
- Update query processing to use pruning
- Add tests for pruning correctness

Improves Q1.1 performance by 2.3x
```

### Pull Request Process

1. **Update your fork**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run tests**
   ```bash
   make test
   ```

3. **Push changes**
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create Pull Request**
   - Use the PR template
   - Link related issues
   - Describe changes clearly
   - Include test results
   - Add performance impact if applicable

5. **Code Review**
   - Address reviewer comments
   - Keep discussion focused and respectful
   - Update PR as needed

### PR Checklist

- [ ] Code follows project style guidelines
- [ ] Tests added for new features
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Performance impact documented
- [ ] No unnecessary dependencies added
- [ ] Commit messages are clear

## Performance Guidelines

- Profile before optimizing
- Document optimization rationale
- Compare with baselines
- Consider GPU occupancy and memory bandwidth

## Questions?

- Open an issue for questions
- Check existing documentation
- Reach out to maintainers

---

Thank you for contributing to L3! 🚀
