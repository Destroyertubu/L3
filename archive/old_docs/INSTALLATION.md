# L3 Installation Guide

This guide covers installation and setup of L3 on various systems.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Installation Steps](#installation-steps)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

### CUDA Toolkit

L3 requires NVIDIA CUDA Toolkit 11.0 or later.

#### Ubuntu/Debian

```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# Install CUDA Toolkit
sudo apt-get install cuda-toolkit-12-3
```

#### Check Installation

```bash
nvcc --version
nvidia-smi
```

### CMake

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# Verify version (should be 3.18+)
cmake --version
```

### Build Essentials

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
```

## System Requirements

### Minimum Requirements

- **GPU**: NVIDIA GPU with Compute Capability 7.5+
  - Examples: GTX 1660, RTX 2060, Tesla T4
- **CUDA**: 11.0+
- **RAM**: 8 GB system memory
- **Disk**: 2 GB free space

### Recommended Requirements

- **GPU**: NVIDIA GPU with Compute Capability 8.0+
  - Examples: RTX 3090, A100, H100
- **CUDA**: 12.0+
- **RAM**: 16 GB+ system memory
- **Disk**: 10 GB free space (for datasets)

## Installation Steps

### Step 1: Download Source

```bash
# If you have the archive
cd /path/to/installation
tar -xzf L3.tar.gz
cd L3

# Or clone from repository
git clone https://github.com/yourname/l3.git
cd l3
```

### Step 2: Configure Build

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
  -DBUILD_BENCHMARKS=ON \
  -DBUILD_EXAMPLES=ON
```

#### Common Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | `Release` | Build type: Release, Debug, RelWithDebInfo |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86;89` | Target GPU architectures |
| `BUILD_BENCHMARKS` | `ON` | Build benchmark executables |
| `BUILD_EXAMPLES` | `ON` | Build example programs |
| `BUILD_TOOLS` | `ON` | Build utility tools |
| `USE_L3` | `ON` | Build L3 optimized version |
| `USE_LEGACY` | `OFF` | Build legacy L3 version |

#### Architecture Selection

Choose based on your GPU:

| GPU Series | Compute Capability | CMake Value |
|------------|-------------------|-------------|
| RTX 20xx (Turing) | 7.5 | `75` |
| RTX 30xx (Ampere) | 8.6 | `86` |
| RTX 40xx (Ada) | 8.9 | `89` |
| A100 | 8.0 | `80` |
| H100 | 9.0 | `90` |

```bash
# For RTX 3090 only
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"

# For multiple architectures (slower build, wider compatibility)
cmake .. -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89"
```

### Step 3: Build

```bash
# Build with all CPU cores
make -j$(nproc)

# Or specify number of parallel jobs
make -j8

# Build specific target
make l32
make q11_2push_opt
```

### Step 4: Install (Optional)

```bash
# Install to system directories (requires sudo)
sudo make install

# Or install to custom location
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/l3
make install
```

### Step 5: Set Environment Variables

Add to `~/.bashrc` or `~/.zshrc`:

```bash
# If installed to system
export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# If installed to custom location
export PATH=/opt/l3/bin:$PATH
export LD_LIBRARY_PATH=/opt/l3/lib:$LD_LIBRARY_PATH

# Reload
source ~/.bashrc
```

## Verification

### Test L3 Library

```bash
cd build

# Check if library was built
ls -lh lib/libl32.a

# Check if benchmarks were built
ls -lh bin/ssb/optimized/
```

### Run Quick Test

```bash
# Navigate to build directory
cd build/bin/ssb/optimized

# Run a simple benchmark
./q11_2push

# Expected output should show:
# - Compression statistics
# - Query execution time
# - Results verification
```

### Performance Test

```bash
# Run multiple queries to verify GPU performance
cd build/bin/ssb/optimized

for query in q11_2push q12_2push q13_2push; do
    echo "=== Testing $query ==="
    ./$query
    echo ""
done
```

## Troubleshooting

### Issue: CMake Cannot Find CUDA

**Error**: `CUDA not found` or `No CUDA toolkits found`

**Solution**:
```bash
# Set CUDA path explicitly
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reconfigure
cd build
rm -rf *
cmake ..
```

### Issue: Unsupported GPU Architecture

**Error**: `nvcc fatal: Unsupported gpu architecture 'compute_XX'`

**Solution**: Check your GPU's compute capability and update CMake configuration:

```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Use only supported architectures
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"  # Example for RTX 30xx
```

### Issue: Out of Memory During Compilation

**Error**: `virtual memory exhausted` or `c++: internal compiler error`

**Solution**: Reduce parallel compilation jobs:

```bash
# Instead of make -j$(nproc)
make -j2

# Or add swap space
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Issue: Runtime Error - CUDA Out of Memory

**Error**: `CUDA out of memory` when running benchmarks

**Solution**:
1. Check GPU memory usage: `nvidia-smi`
2. Close other GPU applications
3. Reduce dataset size for testing
4. Use a GPU with more memory

### Issue: Incorrect Results

**Solution**:
1. Ensure you're using `Release` build for correct optimizations
2. Verify CUDA architecture matches your GPU
3. Check for driver/CUDA version compatibility

```bash
# Rebuild in Release mode
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make clean
make -j$(nproc)
```

### Issue: Missing Shared Libraries

**Error**: `error while loading shared libraries: libcudart.so.XX`

**Solution**:
```bash
# Add CUDA lib path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Make permanent
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Advanced Configuration

### Custom Compiler

```bash
# Use specific C++ compiler
cmake .. -DCMAKE_CXX_COMPILER=g++-11

# Use specific CUDA compiler
cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.3/bin/nvcc
```

### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Debug with CUDA memory checker
cuda-memcheck ./build/bin/ssb/optimized/q11_2push
```

### Cross-Compilation

For deploying to different GPU architectures:

```bash
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"

# This creates a binary compatible with multiple GPU generations
```

## Next Steps

After successful installation:

1. Read [API Reference](API.md) for usage examples
2. Check [SSB Benchmark Guide](SSB_BENCHMARK.md) for running benchmarks
3. Explore [Examples](../examples/) for sample code
4. See [Performance Tuning](PERFORMANCE.md) for optimization tips

## Getting Help

If you encounter issues not covered here:

1. Check [GitHub Issues](https://github.com/yourname/l3/issues)
2. Search existing issues for solutions
3. Create a new issue with:
   - OS and version
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi`)
   - Complete error message
   - Steps to reproduce
