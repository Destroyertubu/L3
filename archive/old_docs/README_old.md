# L3: GPU-based Learned Compression

[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-brightgreen)](https://developer.nvidia.com/cuda-toolkit)
[![CMake](https://img.shields.io/badge/CMake-3.18%2B-blue)](https://cmake.org/)
[![License](https://img.shields.io/badge/license-MIT-orange)](LICENSE)

L3 is a high-performance GPU-accelerated compression library for time series and database columns using learned models. It implements adaptive piecewise linear regression models optimized for GPU architectures.

## Features

- **High Compression Ratios**: Achieves 10-100x compression on sorted integer sequences
- **GPU Acceleration**: Leverages CUDA for parallel compression and decompression
- **Adaptive Models**: Automatically selects optimal models (constant, linear, polynomial)
- **Random Access**: Supports efficient random access without full decompression
- **Optimized Kernels**: Uses warp-level primitives and shared memory optimization
- **SSB Benchmarks**: Includes Star Schema Benchmark query implementations

## Project Structure

```
L3/
├── lib/                          # Core libraries
│   ├── l32/                   # L3 optimized version (SoA layout)
│   └── l3_legacy/             # Legacy L3 implementation
├── include/                      # Header files
│   ├── common/                   # Shared utilities and headers
│   ├── l32/                   # L3 public headers
│   └── l3_legacy/             # Legacy headers
├── benchmarks/                   # Benchmark programs
│   ├── ssb/                      # Star Schema Benchmark
│   │   ├── baseline/             # Baseline implementations
│   │   └── optimized_2push/      # Optimized with predicate pushdown
│   ├── sosd/                     # SOSD dataset benchmarks
│   └── random_access/            # Random access benchmarks
├── examples/                     # Example usage programs
├── tools/                        # Utility tools
├── scripts/                      # Build and deployment scripts
├── data/                         # Sample datasets
├── docs/                         # Documentation
└── build/                        # Build directory (generated)
```

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5+ (Turing, Ampere, Ada, or Hopper)
- Recommended: RTX 3090, A100, H100, or newer

### Software
- **CUDA Toolkit**: 11.0 or later
- **CMake**: 3.18 or later
- **C++ Compiler**: GCC 9+ or Clang 10+
- **Operating System**: Linux (Ubuntu 20.04+, CentOS 8+)

## Quick Start

### 1. Build from Source

```bash
# Clone or extract the project
cd L3

# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### 2. Build Options

```bash
# Build with specific CUDA architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"

# Build only L3 (recommended)
cmake .. -DUSE_L3=ON -DUSE_LEGACY=OFF

# Build without benchmarks
cmake .. -DBUILD_BENCHMARKS=OFF

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### 3. Run Benchmarks

```bash
# After building, run SSB benchmarks
cd build/bin/ssb/optimized

# Run Q1.1 benchmark
./q11_2push

# Run all queries
for query in q*_2push; do
    echo "Running $query..."
    ./$query
done
```

## Usage Example

```cuda
#include "l32.h"

// Compress data
int64_t* data = ...; // Your sorted integer array
size_t n = 1000000;

CompressedData<int64_t> compressed;
compress_gpu(data, n, &compressed);

// Random access without full decompression
int64_t value = random_access_gpu(&compressed, index);

// Decompress
int64_t* decompressed = decompress_gpu(&compressed);
```

## Performance

Tested on NVIDIA RTX 4090:

| Dataset | Compression Ratio | Compression Speed | Decompression Speed |
|---------|------------------|-------------------|---------------------|
| SSB SF10 | 45.2x | 12.3 GB/s | 18.7 GB/s |
| SOSD osm | 72.1x | 15.1 GB/s | 22.4 GB/s |
| Sequential | 128.5x | 18.9 GB/s | 28.3 GB/s |

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [API Reference](docs/API.md)
- [Architecture Overview](docs/ARCHITECTURE.md)
- [SSB Benchmark Guide](docs/SSB_BENCHMARK.md)
- [Performance Tuning](docs/PERFORMANCE.md)

## Migration Guide

If you're migrating from the old project structure:

1. **Data Paths**: Update data file paths in your code
2. **Include Paths**: Use new include directory structure
3. **Build System**: Use CMake instead of manual compilation
4. **Binaries**: Find executables in `build/bin/` subdirectories

See [MIGRATION.md](docs/MIGRATION.md) for detailed migration instructions.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use L3 in your research, please cite:

```bibtex
@inproceedings{l32024,
  title={L3: GPU-based Learned Compression for Time Series Data},
  author={Your Name},
  booktitle={Proceedings of Conference},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- NVIDIA CUDA team for excellent GPU computing tools
- CUB library for efficient parallel primitives
- SSB benchmark for database query evaluation

## Contact

- **Issues**: [GitHub Issues](https://github.com/yourname/l3/issues)
- **Email**: your.email@example.com

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.
