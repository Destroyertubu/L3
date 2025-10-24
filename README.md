# GLECO: GPU-Accelerated Learned Compression for OLAP Queries

A high-performance GPU-based data compression and query processing system using learned models for analytical workloads.

## 🎯 Overview

GLECO (GPU Learned Compression) is an optimized implementation of learned compression techniques for GPU-accelerated OLAP query processing. The system combines:

- **Learned Models**: Linear regression models to predict data values
- **Delta Encoding**: Efficient storage of prediction errors
- **Bit Packing**: Variable-bit-width encoding for compact storage
- **GPU Acceleration**: CUDA kernels for high-throughput decompression
- **Random Access**: Fast element-level decompression without full scan
- **Predicate Pushdown**: Query optimization through partition pruning

## 🏗️ Architecture

```
├── include/           # Public header files
├── src/
│   ├── codec/        # Compression/decompression codec implementation
│   ├── kernels/      # CUDA kernels
│   │   ├── compression/
│   │   ├── decompression/
│   │   └── utils/
│   ├── tests/        # Unit tests and integration tests
│   └── tools/        # Benchmark and utility tools
├── tests/
│   ├── ra/           # Random access tests
│   └── ssb/          # SSB (Star Schema Benchmark) query tests
├── benchmarks/       # Performance benchmark results
├── docs/             # Documentation
├── examples/         # Example usage code
└── scripts/          # Build and test scripts
```

## ⚙️ Requirements

- **CUDA**: 11.0 or higher
- **GPU**: NVIDIA H20 or similar (Compute Capability 9.0)
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 7.0+ or Clang with C++17 support
- **OS**: Linux (tested on Ubuntu)

## 🚀 Quick Start

### Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j8
```

### Run SSB Benchmarks

```bash
# Navigate to SSB tests directory
cd tests/ssb

# Build all SSB query tests
make

# Run all SSB queries
make run_all

# Run specific query groups
make run_q1  # Q1.x queries
make run_q2  # Q2.x queries
make run_q3  # Q3.x queries
make run_q4  # Q4.x queries
```

### Run Random Access Tests

```bash
cd tests/ra
make
./test_random_access_comprehensive --max=10000000
```

## 🧪 Testing

### Unit Tests

```bash
cd build
./test_compression_main
./test_fb_200M_phase2_bucket
```

### SSB Query Tests

```bash
cd tests/ssb
make run_all
```

Individual query tests are available:
- `q11_2push` - Q1.1: Simple aggregation with year filter
- `q21_2push` - Q2.1: Part/Supplier join with category filter
- `q31_2push` - Q3.1: Customer/Supplier region join
- `q41_2push` - Q4.1: 5-table profit analysis

## 📖 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Reference](docs/api.md)
- [Compression Format](docs/compression_format.md)
- [Query Processing](docs/query_processing.md)
- [Performance Tuning](docs/performance_tuning.md)

## 🔧 Configuration

CMake options for optimization:

```bash
cmake -DPHASE2_USE_CP_ASYNC=ON \
      -DPHASE2_CTA_BATCH=4 \
      -DPHASE2_PERSISTENT_THREADS=ON \
      ..
```

Options:
- `PHASE2_USE_CP_ASYNC`: Enable cp.async memory pipeline (default: OFF)
- `PHASE2_CTA_BATCH`: Partitions per CTA (default: 4)
- `PHASE2_PERSISTENT_THREADS`: Enable persistent kernel threads (default: OFF)
- `PHASE2_DEBUG_ROUTING`: Enable routing debugging (default: OFF)


## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on learned compression research
- SSB benchmark suite
- CUDA optimization techniques

## 📧 Contact

For questions and feedback:
- Issues: [GitHub Issues](https://github.com/yourusername/L3_opt/issues)
- Email: your.email@example.com

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: October 2025
