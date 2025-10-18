# L3: GPU-Accelerated Learned Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/)

L3 (Learned LEarned COmpression) is a high-performance GPU compression library that uses learned models to achieve superior compression ratios while maintaining fast decompression speeds.

## 🌟 Key Features

- **🚀 High Performance**: 40+ GB/s decompression throughput on NVIDIA A100
- **📊 Superior Compression**: 3.5-4.5x compression ratio on real-world datasets
- **🎯 Flexible Partitioning**: Choose between fixed-size or adaptive variable-length partitioning
- **🔍 Random Access**: Access compressed data without full decompression
- **💻 Query Execution**: Optimized SSB query execution on compressed data
- **🐍 Python Support**: Easy-to-use Python bindings (coming soon)

## 🎯 Partition Strategies

L3 provides **two partitioning strategies** that you can choose based on your data characteristics:

### Fixed-Size Partitioning
```cpp
l3::FixedSizePartitioner partitioner(4096);  // 4K elements per partition
auto partitions = partitioner.partition(data, size, sizeof(int64_t));
```

**Best for**:
- Uniformly distributed data
- Predictable performance requirements
- Fast partitioning is critical

### Variable-Length Partitioning
```cpp
l3::VariableLengthPartitioner partitioner(
    1024,  // base_size
    8,     // variance_multiplier
    3      // num_thresholds
);
auto partitions = partitioner.partition(data, size, sizeof(int64_t));
```

**Best for**:
- Non-uniform data distribution
- Maximum compression ratio
- Time series with varying patterns

### Auto Selection
```cpp
auto partitioner = l3::PartitionerFactory::createAuto(data, size, sizeof(int64_t));
```

## 🚀 Quick Start

### Prerequisites

- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2019+)
- NVIDIA GPU with compute capability 7.5+ (Turing, Ampere, or Hopper)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/L3.git
cd L3

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DBUILD_EXAMPLES=ON \
         -DBUILD_TESTS=ON

# Build
make -j$(nproc)

# Run examples
./bin/examples/example_partition_strategies
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TESTS` | ON | Build unit and integration tests |
| `BUILD_BENCHMARKS` | ON | Build performance benchmarks |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_PYTHON` | OFF | Build Python bindings |
| `BUILD_SHARED_LIBS` | OFF | Build shared libraries |

## 📖 Usage Examples

### Basic Compression with Fixed Partitioning

```cpp
#include <l3/l3.hpp>

int main() {
    // Your data
    std::vector<int64_t> data = loadData();

    // Configure compression
    l3::CompressionConfig config;
    config.partition_strategy = l3::PartitionerFactory::FIXED_SIZE;
    config.partition_size_hint = 4096;

    // Compress
    auto* compressed = l3::compress(data.data(), data.size(), config);

    // Decompress
    auto* decompressed = l3::decompress(compressed);

    // Use decompressed data...

    // Cleanup
    l3::free(compressed);
    l3::free(decompressed);

    return 0;
}
```

### Comparing Partition Strategies

```cpp
#include <l3/l3.hpp>
#include <l3/benchmark.hpp>

int main() {
    std::vector<int64_t> data = loadData();

    // Create different partitioners
    std::vector<std::unique_ptr<l3::PartitionStrategy>> strategies;
    strategies.push_back(std::make_unique<l3::FixedSizePartitioner>(2048));
    strategies.push_back(std::make_unique<l3::FixedSizePartitioner>(4096));
    strategies.push_back(std::make_unique<l3::VariableLengthPartitioner>(1024, 8, 3));

    // Compare performance
    auto results = l3::benchmark::comparePartitioners(
        data.data(),
        data.size(),
        strategies
    );

    // Print results
    for (const auto& result : results) {
        std::cout << result.partitioner_name << ": "
                  << result.compression_ratio << "x, "
                  << result.throughput_gbps << " GB/s\n";
    }

    return 0;
}
```

### Custom Partitioner

```cpp
class MyCustomPartitioner : public l3::PartitionStrategy {
public:
    std::vector<l3::PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size) override
    {
        // Your custom partitioning logic
        std::vector<l3::PartitionInfo> partitions;
        // ... implement ...
        return partitions;
    }

    const char* getName() const override {
        return "MyCustom";
    }

    l3::PartitionConfig getConfig() const override {
        return l3::PartitionConfig();
    }
};

// Use it
MyCustomPartitioner partitioner;
auto partitions = partitioner.partition(data, size, sizeof(int64_t));
```

## 📂 Project Structure

```
L3/
├── include/l3/          # Public API headers
│   ├── partitioner.hpp  # Partition strategy interface ⭐
│   ├── compression.hpp  # Compression API
│   └── ...
├── src/                 # Source code
│   ├── partitioner/     # Partitioner implementations ⭐
│   ├── compression/     # Compression kernels
│   ├── decompression/   # Decompression kernels
│   └── ...
├── examples/            # Example programs
├── tests/               # Unit and integration tests
├── benchmarks/          # Performance benchmarks
├── docs/                # Documentation
└── python/              # Python bindings
```

## 📊 Performance

### Compression Performance (NVIDIA A100)

| Dataset | Fixed 4K | Variable (1K,8,3) | Speedup |
|---------|----------|-------------------|---------|
| SOSD Books | 3.2x | 3.8x | 1.19x |
| SOSD OSM | 2.8x | 3.5x | 1.25x |
| SSB Lineorder | 3.5x | 4.2x | 1.20x |

### Throughput

- **Compression**: 28-32 GB/s
- **Decompression**: 40-45 GB/s
- **Random Access**: <5% overhead

## 🔬 Algorithm

L3 uses **learned model-based compression**:

1. **Partitioning**: Split data into partitions (fixed or adaptive)
2. **Model Fitting**: Fit linear/polynomial model per partition
3. **Residual Encoding**: Compute deltas (actual - predicted)
4. **Bit Packing**: Pack deltas with partition-specific bit widths

### Decompression Formula

```
decompressed_value = model(index) + delta
```

Where:
- `model(index)` = θ₀ + θ₁·x + θ₂·x² + θ₃·x³
- `delta` = signed residual from bit-packed array

## 🧪 Testing

```bash
# Build with tests
cmake .. -DBUILD_TESTS=ON
make

# Run all tests
ctest --verbose

# Run specific test
./bin/tests/test_partitioner
```

## 📈 Benchmarks

```bash
# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON
make

# Run partition strategy comparison
./bin/benchmarks/bench_compare_partitioners

# Run SSB queries
./bin/benchmarks/ssb_bench
```

## 🐍 Python Bindings (Coming Soon)

```python
import l3_compression as l3

# Compress with fixed partitioning
compressed = l3.compress(data, partition_strategy='fixed', partition_size=4096)

# Compress with variable partitioning
compressed = l3.compress(data, partition_strategy='variable', base_size=1024)

# Auto selection
compressed = l3.compress(data, partition_strategy='auto')

# Compare strategies
results = l3.benchmark.compare_partitioners(data, ['fixed', 'variable'])
l3.visualization.plot_comparison(results)
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Installation and basic usage
- **[Partition Strategies Guide](docs/PARTITION_STRATEGIES.md)** - Choosing the right strategy
- **[Environment Setup](ENVIRONMENT_SETUP.md)** - Path configuration and environment variables
- **[Development Guide](docs/DEVELOPMENT.md)** - Contributing and roadmap
- **[Migration Status](docs/MIGRATION.md)** - Current project status
- **[Architecture](docs/ARCHITECTURE.md)** - System design and internals

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Adding a Custom Partitioner

See [docs/development/adding_partitioner.md](docs/development/adding_partitioner.md) for a step-by-step guide.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use L3 in your research, please cite:

```bibtex
@software{l3_compression,
  title = {L3: GPU-Accelerated Learned Compression},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/L3}
}
```

## 🙏 Acknowledgments

- NVIDIA CUDA Toolkit
- Thrust library
- [Add other acknowledgments]

## 📧 Contact

For questions or feedback:
- Open an issue on [GitHub](https://github.com/yourusername/L3/issues)
- Email: your.email@example.com

---

Made with ❤️ using CUDA
