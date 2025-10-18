# L3 Documentation

Complete documentation for the L3 compression library.

## Quick Navigation

### Getting Started
- [Installation Guide](INSTALLATION.md) - Setup and installation instructions
- [Quick Start](#quick-start) - Get up and running in 5 minutes
- [Migration Guide](MIGRATION.md) - Migrate from old L3 project

### Understanding L3
- [Architecture Overview](ARCHITECTURE.md) - System design and components
- [Algorithm Explanation](#algorithm) - How L3 compression works
- [Performance Characteristics](#performance) - What to expect

### Using L3
- [SSB Benchmark Guide](SSB_BENCHMARK.md) - Run Star Schema Benchmarks
- [API Reference](API.md) - Function documentation
- [Examples](../examples/) - Code examples

### Advanced Topics
- [Performance Tuning](PERFORMANCE.md) - Optimization tips
- [Troubleshooting](#troubleshooting) - Common issues and solutions

## Quick Start

### 1. Verify Prerequisites

```bash
./scripts/verify.sh
```

### 2. Build

```bash
./scripts/build.sh
```

### 3. Run Test

```bash
cd build/bin/ssb/optimized
./q11_2push_opt
```

## Algorithm

L3 uses learned models to compress sorted integer sequences:

### Compression Process

```
┌─────────────┐
│ Input Data  │ Sorted integers
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Partitioning │ Split into segments
└──────┬──────┘
       │
       ▼
┌─────────────┐
│Model Fitting│ Fit linear/polynomial models
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Deltas     │ actual - predicted
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Bit Packing │ Compress deltas
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Compressed │
└─────────────┘
```

### Key Concepts

1. **Partitioning**: Data is divided into variable-size partitions
2. **Model Selection**: Each partition gets optimal model (constant, linear, polynomial)
3. **Delta Encoding**: Store differences between actual and predicted values
4. **Bit Packing**: Compress deltas using minimal bits

## Performance

Typical performance on NVIDIA RTX 4090:

| Operation | Throughput | Latency |
|-----------|-----------|---------|
| Compression | 12-18 GB/s | - |
| Decompression | 18-28 GB/s | - |
| Random Access | - | ~1 μs per element |

**Compression Ratios:**
- Sorted sequences: 50-200x
- SSB columns: 20-80x
- Real datasets: 10-100x

## Documentation Index

### Core Documentation

| Document | Description |
|----------|-------------|
| [INSTALLATION.md](INSTALLATION.md) | Complete installation guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design |
| [MIGRATION.md](MIGRATION.md) | Migration from old project |
| [API.md](API.md) | API reference |
| [PERFORMANCE.md](PERFORMANCE.md) | Performance tuning |
| [SSB_BENCHMARK.md](SSB_BENCHMARK.md) | SSB benchmark guide |

### Additional Resources

- **Main README**: [../README.md](../README.md)
- **Examples**: [../examples/](../examples/)
- **Scripts**: [../scripts/](../scripts/)

## Troubleshooting

### Build Issues

**Problem**: CMake cannot find CUDA
```bash
export CUDA_HOME=/usr/local/cuda
./scripts/build.sh
```

**Problem**: Unsupported GPU architecture
```bash
./scripts/build.sh -a 86  # For your specific GPU
```

### Runtime Issues

**Problem**: Out of memory
- Reduce dataset size
- Use GPU with more memory
- Check for memory leaks with `cuda-memcheck`

**Problem**: Slow performance
- Ensure Release build: `./scripts/build.sh -t Release`
- Check GPU utilization: `nvidia-smi`
- See [Performance Guide](PERFORMANCE.md)

### Getting Help

1. Check relevant documentation section
2. Review [Installation Guide](INSTALLATION.md#troubleshooting)
3. Search existing GitHub issues
4. Create new issue with system info

## Contributing

Documentation contributions welcome:
- Fix typos or errors
- Add examples
- Improve explanations
- Translate to other languages

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Documentation Standards

When updating documentation:
- Use clear, concise language
- Include code examples
- Add diagrams where helpful
- Keep formatting consistent
- Update table of contents

## Version History

- **v2.0** (2024) - Refactored project structure, comprehensive docs
- **v1.0** (2023) - Initial L3 implementation

## License

Documentation licensed under CC BY 4.0.
Code licensed under MIT License.

---

*Last updated: 2024*
