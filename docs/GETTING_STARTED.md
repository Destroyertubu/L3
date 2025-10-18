# Getting Started with L3

This guide will help you get started with L3 GPU compression library.

## Prerequisites

### Hardware
- NVIDIA GPU with Compute Capability 7.5 or higher
  - Turing (RTX 20xx, T4)
  - Ampere (RTX 30xx, A100, A10)
  - Hopper (H100)

### Software
- CUDA Toolkit 11.0 or later
- CMake 3.18 or later
- C++17 compatible compiler:
  - GCC 9+
  - Clang 10+
  - MSVC 2019+

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/L3.git
cd L3
```

### 2. Build

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
sudo make install
```

### 3. Build Options

```bash
# Build with examples
cmake .. -DBUILD_EXAMPLES=ON

# Build with tests
cmake .. -DBUILD_TESTS=ON

# Build with benchmarks
cmake .. -DBUILD_BENCHMARKS=ON

# Specify CUDA architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"
```

## Basic Usage

### Example 1: Fixed-Size Partitioning

```cpp
#include <l3/partitioner.hpp>
#include <vector>

int main() {
    // Your data
    std::vector<int64_t> data(100000);
    // ... fill data ...

    // Create fixed-size partitioner
    l3::FixedSizePartitioner partitioner(4096);  // 4K elements per partition

    // Partition the data
    auto partitions = partitioner.partition(
        data.data(),
        data.size(),
        sizeof(int64_t)
    );

    // Use partitions for compression...

    return 0;
}
```

### Example 2: Variable-Length Partitioning

```cpp
#include <l3/partitioner.hpp>

int main() {
    std::vector<int64_t> data(100000);
    // ... fill data ...

    // Create adaptive partitioner
    l3::VariableLengthPartitioner partitioner(
        1024,  // base_size
        8,     // variance_multiplier
        3      // num_thresholds
    );

    auto partitions = partitioner.partition(
        data.data(),
        data.size(),
        sizeof(int64_t)
    );

    return 0;
}
```

### Example 3: Factory Pattern

```cpp
#include <l3/partitioner.hpp>

int main() {
    std::vector<int64_t> data(100000);

    // Configure partitioner
    l3::PartitionConfig config;
    config.base_size = 2048;
    config.variance_multiplier = 16;
    config.num_thresholds = 5;

    // Create using factory
    auto partitioner = l3::PartitionerFactory::create(
        l3::PartitionerFactory::VARIABLE_LENGTH,
        config
    );

    auto partitions = partitioner->partition(
        data.data(),
        data.size(),
        sizeof(int64_t)
    );

    return 0;
}
```

## Running Examples

```bash
# After building with -DBUILD_EXAMPLES=ON
cd build

# Run partition strategies example
./bin/examples/example_partition_strategies
```

## Next Steps

- [Partition Strategies](PARTITION_STRATEGIES.md) - Learn how to choose the right strategy
- [Development Guide](DEVELOPMENT.md) - Contributing to L3
- [API Reference](api/) - Complete API documentation

## Troubleshooting

### CUDA not found
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Wrong CUDA architecture
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for specific architecture
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"  # For Turing
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"  # For Ampere
```

### Compilation errors
- Ensure you have C++17 support: `g++ --version` (need GCC 9+)
- Update CMake: `cmake --version` (need 3.18+)
- Update CUDA: `nvcc --version` (need 11.0+)

## Support

- GitHub Issues: https://github.com/yourusername/L3/issues
- Documentation: See `docs/` directory
