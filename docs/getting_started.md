# Getting Started with L3

## Installation

### Prerequisites

```bash
# Check CUDA version
nvcc --version

# Check GPU
nvidia-smi
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/yourusername/L3_opt.git
cd L3_opt

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)
```

### Build Options

```bash
# Enable optimizations
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPHASE2_USE_CP_ASYNC=ON \
      -DPHASE2_CTA_BATCH=4 \
      ..
```

## Quick Examples

### Example 1: Basic Compression

```cpp
#include "L3_codec.hpp"

// Create sample data
std::vector<uint32_t> data(1000000);
for (int i = 0; i < data.size(); i++) {
    data[i] = i * 100;  // Linear pattern
}

// Compress
CompressedDataL3<uint32_t>* compressed = compressData(data);

// Decompress
std::vector<uint32_t> decompressed;
decompressData(compressed, decompressed);

// Cleanup
freeCompressedData(compressed);
```

### Example 2: Random Access

```cpp
#include "L3_random_access.hpp"

// Compress data
CompressedDataL3<uint32_t>* compressed = compressData(data, 1024);

// Random access on GPU
uint32_t value = randomAccessGPU(compressed, 12345);
```

### Example 3: SSB Query

```bash
# Navigate to SSB tests
cd tests/ssb

# Run Q1.1
./q11_2push 1

# Run all Q1 queries
make run_q1
```

## Understanding Output

### Compression Stats

```
Compression complete (0.762 seconds)
  Original size: 457.5 MB
  Compressed size: 52.3 MB
  Compression ratio: 8.75x
  Throughput: 600.2 MB/s
```

### Query Results

```json
{
  "query": "Q1.1",
  "time_ms": 2.83,
  "num_candidates": 17142055,
  "partitions_pruned": 0,
  "result": 22674895353644
}
```

## Next Steps

- Read [Architecture Overview](architecture.md)
- Explore [API Reference](api.md)
- See [Performance Tuning](performance_tuning.md)
- Check out [SSB Benchmarks](../benchmarks/README.md)
