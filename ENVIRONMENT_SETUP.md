# Environment Setup Guide

This document explains how to configure paths and environment variables for the L3 project.

## Quick Start

The L3 project uses **relative paths by default**, so it works out of the box after cloning from GitHub. No configuration needed for basic usage!

```bash
git clone https://github.com/your-username/L3.git
cd L3
./build_and_run.sh
```

## Directory Structure

```
L3/
├── benchmarks/          # Benchmark programs
├── build/               # Build output (auto-generated, git-ignored)
├── cmake/               # CMake configuration files
├── docs/                # Documentation
├── examples/            # Example programs
├── include/             # Public headers
├── lib/                 # Legacy implementations
├── src/                 # New modular source code
├── test/                # Test programs and data
│   └── data/           # Test datasets (*.bin files git-ignored)
└── CMakeLists.txt       # Build configuration
```

## Environment Variables (Optional)

You can customize paths using environment variables. This is useful when:
- Running benchmarks with custom datasets
- Storing large test data in a different location
- Running on different machines with different directory structures

### L3_DATA_DIR

Specifies the directory containing test datasets.

**Default**: `../../test/data` (relative to benchmark binary location)

**Usage**:
```bash
export L3_DATA_DIR=/path/to/your/data
./build/bin/benchmark_optimized
```

**Example**:
```bash
# Use custom data directory
export L3_DATA_DIR=/mnt/datasets/compression_test
./benchmarks/codec/benchmark_optimized

# Or set it inline
L3_DATA_DIR=/data/benchmarks ./build/bin/benchmark_kernel_only
```

### L3_OUTPUT_DIR

Specifies the directory for benchmark results (CSV files).

**Default**: `.` (current directory)

**Usage**:
```bash
export L3_OUTPUT_DIR=/path/to/results
./build/bin/benchmark_optimized
```

**Example**:
```bash
# Save results to specific directory
mkdir -p results
export L3_OUTPUT_DIR=results
./build/bin/benchmark_optimized
```

## Programs That Use Environment Variables

### Benchmarks

All benchmark programs support `L3_DATA_DIR` and `L3_OUTPUT_DIR`:

1. **benchmark_kernel_only** - Kernel performance benchmark
   - Data: `L3_DATA_DIR` (default: `../../test/data`)
   - Output: `L3_OUTPUT_DIR/kernel_benchmark_results.csv`

2. **benchmark_optimized** - Full compression benchmark
   - Data: `L3_DATA_DIR` (default: `../../test/data`)
   - Output: `L3_OUTPUT_DIR/l3_optimized_results.csv`

3. **sosd_bench_demo** - SOSD dataset benchmark
   - Data: `L3_DATA_DIR` (default: `../../test/data`)

### Utilities

1. **convert_to_binary** - Convert text datasets to binary
   - Can specify directory via command line or `L3_DATA_DIR`
   - Usage: `./convert_to_binary [data_directory]`
   - Or: `L3_DATA_DIR=/data ./convert_to_binary`

## Test Data Setup

### Required for Benchmarks

If you want to run benchmarks, you need test datasets. Place them in `test/data/`:

```bash
mkdir -p test/data
cd test/data

# Download or generate your datasets
# Example structure:
test/data/
├── movieid_uint32.bin
├── linear_200M_uint32.txt
├── linear_200M_uint32_binary.bin
├── books_200M_uint32.bin
├── normal_200M_uint32.txt
├── normal_200M_uint32_binary.bin
├── fb_200M_uint64.bin
└── wiki_200M_uint64.bin
```

**Note**: Binary files (*.bin) are git-ignored to keep the repository size small.

### Dataset Formats

- **Text files** (*.txt): One integer per line
- **Binary files** (*.bin): Raw binary dump of uint32_t or uint64_t arrays

### Converting Text to Binary

Use the provided converter:

```bash
# From project root
cd lib/modular/data
make convert_to_binary

# Convert using default directory
./convert_to_binary

# Or specify custom directory
./convert_to_binary /path/to/data

# Or using environment variable
L3_DATA_DIR=/path/to/data ./convert_to_binary
```

## Running Examples (No Setup Required)

The example programs generate synthetic data and require no external datasets:

```bash
./build_and_run.sh
# or
./build/bin/example_partition_strategies
```

## Configuration File (Advanced)

For complex setups, you can create a configuration script:

```bash
# Create config.sh
cat > config.sh << 'EOF'
#!/bin/bash
# L3 Environment Configuration

export L3_DATA_DIR=/mnt/ssd/compression_datasets
export L3_OUTPUT_DIR=/home/user/results/L3

# CUDA settings
export CUDA_VISIBLE_DEVICES=0

# Build settings
export CMAKE_BUILD_TYPE=Release
export CMAKE_CUDA_ARCHITECTURES="80;86"

echo "L3 environment configured:"
echo "  Data directory: $L3_DATA_DIR"
echo "  Output directory: $L3_OUTPUT_DIR"
echo "  CUDA device: $CUDA_VISIBLE_DEVICES"
EOF

chmod +x config.sh

# Use it
source config.sh
./build_and_run.sh
```

## Docker Setup (Future)

For reproducible environments:

```dockerfile
FROM nvidia/cuda:12.0-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    git

# Clone and build L3
WORKDIR /workspace
COPY . .
RUN ./build_and_run.sh

# Set environment
ENV L3_DATA_DIR=/data
ENV L3_OUTPUT_DIR=/results

VOLUME ["/data", "/results"]
```

## Troubleshooting

### "Cannot open file" errors

**Problem**: Benchmark can't find test data

**Solution**:
```bash
# Check if data directory exists
ls test/data/

# Set correct path
export L3_DATA_DIR=/absolute/path/to/test/data
./build/bin/benchmark_optimized
```

### Relative paths not working

**Problem**: Running from wrong directory

**Solution**:
```bash
# Always run from build directory
cd build
./bin/example_partition_strategies

# Or use absolute paths
/path/to/L3/build/bin/example_partition_strategies

# Or set environment variable
export L3_DATA_DIR=/path/to/L3/test/data
```

### Permission denied

**Problem**: Can't write output files

**Solution**:
```bash
# Create output directory
mkdir -p results
chmod 755 results
export L3_OUTPUT_DIR=results
```

## GitHub Workflow

When sharing on GitHub:

1. **Clone repository**:
   ```bash
   git clone https://github.com/your-username/L3.git
   cd L3
   ```

2. **Build project**:
   ```bash
   ./build_and_run.sh
   ```

3. **Run examples** (no data needed):
   ```bash
   ./build/bin/example_partition_strategies
   ```

4. **For benchmarks**, add your own datasets:
   ```bash
   mkdir -p test/data
   # Copy your datasets
   cp /your/data/*.bin test/data/

   # Run benchmarks
   export L3_DATA_DIR=./test/data
   ./build/bin/benchmark_optimized
   ```

## Summary

✅ **Default behavior**: Works with relative paths, no configuration needed

✅ **Examples**: Run immediately after build, no data required

✅ **Benchmarks**: Need test datasets in `test/data/` or via `L3_DATA_DIR`

✅ **Portable**: Clone from GitHub and build immediately

✅ **Flexible**: Environment variables for custom paths

---

**Quick Reference**:

| What | How |
|------|-----|
| Build | `./build_and_run.sh` |
| Run example | `./build/bin/example_partition_strategies` |
| Set data dir | `export L3_DATA_DIR=/path/to/data` |
| Set output dir | `export L3_OUTPUT_DIR=/path/to/results` |
| Run benchmark | `./build/bin/benchmark_optimized` |
