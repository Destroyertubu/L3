# L3: GPU-Accelerated Learned Lossless Lightweight Compression

A high-performance GPU-based data compression and query processing system using learned models for analytical workloads.

## Overview

L3 is an optimized GPU compression system that combines machine learning models with efficient encoding techniques to achieve high compression ratios while maintaining fast decompression throughput. It is specifically designed for OLAP (Online Analytical Processing) workloads where data is compressed once and queried many times.

### Key Innovations

- **Learned Compression**: Uses linear/polynomial regression models to predict data values, storing only prediction errors
- **GPU Acceleration**: CUDA kernels achieve 50-100 GB/s decompression throughput
- **Random Access**: Access individual elements without decompressing entire dataset
- **Predicate Pushdown**: Query optimization through partition-level filtering
- **Cost-Optimal Partitioning**: Adaptive partitioning strategy based on delta-bit analysis

## Features

| Feature | Description |
|---------|-------------|
| **Learned Models** | Linear, Polynomial (degree 2/3), FOR+BitPack, Direct Copy |
| **GPU Decompression** | Multiple optimized kernels (Phase2, Warp-Opt, Specialized) |
| **Random Access** | O(log n) partition lookup + O(1) element decompression |
| **Predicate Pushdown** | Partition min/max bounds for query optimization |
| **String Compression** | Dictionary encoding with GPU acceleration |
| **SSB Benchmark** | Full support for all 13 SSB queries |

## Performance

Tested on NVIDIA H20 GPU with SSB Scale Factor 20 (119,968,352 rows):

### Query Performance

| Query Group | Avg Time | Selectivity | Speedup |
|-------------|----------|-------------|---------|
| Q1.x (Simple Aggregation) | 1.3-2.8 ms | 1.2-14.3% | ~100x |
| Q2.x (Multi-Join) | 1.3-1.7 ms | 0.06-2.3% | ~150x |
| Q3.x (Complex Join) | 1.3-1.9 ms | 0.002-85% | ~200x |
| Q4.x (5-Table Join) | 2.8-3.6 ms | 0.002-0.2% | ~250x |

### Compression Metrics

| Metric | Value |
|--------|-------|
| Compression Ratio | 3-10x (data dependent) |
| Decompression Throughput | 50-100 GB/s |
| Random Access Latency | <100 ns per element |
| Compression Throughput | 5-10 GB/s |

## Architecture

```
L3/
├── include/                    # Public API headers
│   ├── L3_codec.hpp           # Main compression/decompression API
│   ├── L3_format.hpp          # Data format definitions
│   ├── L3_random_access.hpp   # Random access API
│   ├── L3_Vertical_api.hpp    # Vertical compression API
│   └── L3_string_codec.hpp    # String compression API
├── src/
│   ├── codec/                 # Codec implementations
│   │   ├── L3_codec.cpp       # Main codec
│   │   └── L3_string_codec.cpp
│   ├── kernels/               # CUDA kernels
│   │   ├── compression/       # Encoder kernels
│   │   ├── decompression/     # Decoder kernels
│   │   └── utils/             # Utility functions
│   └── tools/                 # Test and benchmark tools
├── tests/                     # Test suites
│   ├── ra/                    # Random access tests
│   ├── ssb/                   # SSB query tests
│   └── ssb_ultra/             # Large-scale SSB tests
├── examples/                  # Usage examples
├── docs/                      # Documentation
├── benchmarks/                # Benchmark results
├── third_party/               # Dependencies (nvcomp, etc.)
└── scripts/                   # Build and test scripts
```

## Requirements

- **CUDA**: 11.0 or higher
- **GPU**: NVIDIA GPU with Compute Capability 8.0+ (recommended: H20, A100, or RTX 3090+)
- **CMake**: 3.18 or higher
- **C++ Compiler**: GCC 7.0+ or Clang with C++17 support
- **OS**: Linux (tested on Ubuntu 20.04/22.04)

## Quick Start

### Build

```bash
# Clone the repository
git clone https://github.com/yourusername/L3.git
cd L3

# Create build directory
mkdir -p build && cd build

# Configure with CMake
cmake ..

# Build the project
make -j$(nproc)
```

### Run Example

```bash
cd examples
# Compile basic example
nvcc -O3 -std=c++17 -arch=sm_80 \
    -I../include \
    basic_compression.cpp \
    -L../build -lL3_codec -lL3_kernels \
    -o basic_compression

# Run
./basic_compression
```

### Run SSB Benchmarks

```bash
cd tests/ssb

# Build SSB tests
make

# Run all queries
make run_all

# Run specific query group
make run_q1  # Q1.x queries
make run_q2  # Q2.x queries
```

## API Reference

### Basic Compression/Decompression

```cpp
#include "L3_codec.hpp"

// Compress data
std::vector<uint32_t> data = {1, 2, 3, 4, 5, ...};
CompressionStats stats;
auto* compressed = compressData(data, 2048, &stats);

// Decompress data
std::vector<uint32_t> output;
decompressData(compressed, output);

// Clean up
freeCompressedData(compressed);
```

### Configuration Options

```cpp
#include "L3_codec.hpp"

// Create custom configuration
L3Config config;
config.partition_size = 2048;
config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
config.enable_polynomial_models = false;

// Cost-optimal parameters
config.cost_analysis_block_size = 2048;
config.cost_min_partition_size = 256;
config.cost_max_partition_size = 8192;
config.cost_breakpoint_threshold = 2;
config.cost_merge_benefit_threshold = 0.05f;

// Compress with configuration
auto* compressed = compressDataWithConfig(data, config, &stats);
```

### Random Access

```cpp
#include "L3_random_access.hpp"

// Single element access (device function)
__device__ uint32_t value = randomAccessElement(compressed, index);

// Batch random access
std::vector<int> indices = {100, 500, 1000, ...};
int* d_indices;
uint32_t* d_output;
// ... allocate device memory ...

RandomAccessStats ra_stats;
randomAccessMultiple(compressed, d_indices, indices.size(),
                     d_output, nullptr, &ra_stats);
```

### Predicate Pushdown

```cpp
// Access partition bounds for query optimization
for (int p = 0; p < compressed->num_partitions; p++) {
    uint32_t min_val = compressed->d_partition_min_values[p];
    uint32_t max_val = compressed->d_partition_max_values[p];

    // Skip partition if predicate cannot be satisfied
    if (max_val < query_min || min_val > query_max) {
        continue;  // Partition pruning
    }
    // Process partition...
}
```

## Technical Details

### Compression Pipeline

```
Raw Data → Partitioning → Model Fitting → Delta Encoding → Bit Packing → Compressed
```

1. **Partitioning**: Divide data into partitions (256-8192 elements)
2. **Model Fitting**: Fit regression model to each partition
3. **Delta Encoding**: Compute prediction errors (delta = actual - predicted)
4. **Bit Packing**: Pack deltas using minimum required bits

### Model Types

| Model | Formula | Use Case |
|-------|---------|----------|
| Constant | f(x) = θ₀ | Constant data |
| Linear | f(x) = θ₀ + θ₁·x | Monotonic trends |
| Polynomial-2 | f(x) = θ₀ + θ₁·x + θ₂·x² | Quadratic patterns |
| Polynomial-3 | f(x) = θ₀ + θ₁·x + θ₂·x² + θ₃·x³ | Complex curves |
| FOR+BitPack | delta = val - base | Random-like data |

### Partitioning Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| FIXED | Fixed-size partitions | Simple, high throughput |
| COST_OPTIMAL | Delta-bit based adaptive | Default, best compression |
| VARIANCE_ADAPTIVE | Variance-based (legacy) | Backward compatibility |

## CMake Configuration Options

```bash
cmake -DPHASE2_USE_CP_ASYNC=ON \        # Enable cp.async pipeline
      -DPHASE2_CTA_BATCH=4 \            # Partitions per CTA
      -DPHASE2_PERSISTENT_THREADS=ON \  # Persistent kernel threads
      -DCUDA_ARCH=sm_80 \               # Target GPU architecture
      ..
```

| Option | Default | Description |
|--------|---------|-------------|
| `PHASE2_USE_CP_ASYNC` | OFF | Enable cp.async memory pipeline |
| `PHASE2_CTA_BATCH` | 4 | Number of partitions per CTA |
| `PHASE2_PERSISTENT_THREADS` | OFF | Enable persistent kernel threads |
| `PHASE2_DEBUG_ROUTING` | OFF | Enable routing debug output |

## Benchmarks

### SOSD Datasets (20 datasets)

The system has been tested on all 20 SOSD benchmark datasets:
- fb_200M, wiki_200M, osm_200M
- books_200M, linear_200M, normal_200M
- And 14 more...

### String Datasets

- Email addresses
- Hexadecimal strings
- English words

## Documentation

- [Getting Started](docs/getting_started.md)
- [API Reference](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [CLI Options](docs/main_cli_options.md)

## Project Structure

```
L3/
├── CMakeLists.txt          # Build configuration
├── include/                # Public headers (11 files)
├── src/
│   ├── codec/             # 2 codec implementations
│   ├── kernels/
│   │   ├── compression/   # ~10 encoder variants
│   │   ├── decompression/ # ~9 decoder variants
│   │   └── utils/         # Utility kernels
│   └── tools/             # main.cpp test harness
├── tests/                 # 40+ test files
├── examples/              # Usage examples
├── docs/                  # Documentation
├── benchmarks/            # Performance data
├── reports/               # Research reports
├── papers/                # Related papers
├── scripts/               # Build scripts
└── third_party/           # Dependencies
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{L32024,
  title={L3: GPU-Accelerated Learned Compression for OLAP Queries},
  author={},
  booktitle={},
  year={2024}
}
```

## Acknowledgments

- Based on learned compression research
- SSB (Star Schema Benchmark) suite
- NVIDIA CUDA optimization techniques
- nvcomp library for baseline comparisons

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/L3/issues)
- Email: your.email@example.com

---

**Status**: Active Development | **Version**: 1.0.0 | **Last Updated**: December 2024
