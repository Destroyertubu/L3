# L3: GPU-based Learned Compression - Complete Codebase Structure

## Overview
L3 is a high-performance GPU-accelerated compression library for time series and database columns using learned piecewise linear regression models. The codebase contains approximately 127 files organized into production and modular implementations.

**Project Location**: `/root/autodl-tmp/L3`
**Total Files**: 127
**Primary Languages**: CUDA (C++), C++
**Build System**: CMake 3.18+
**License**: MIT

---

## 1. DIRECTORY STRUCTURE & PURPOSES

```
L3/
├── lib/                           # Core library implementations (2 variants)
│   ├── single_file/              # Single-file production version (20 modular files)
│   │   ├── include/l3/           # Header files with modular organization
│   │   │   ├── config.cuh        # Configuration macros and CUDA setup
│   │   │   ├── data_structures.cuh # Core data structures (ModelType, CompressedData)
│   │   │   ├── device/           # GPU utility functions
│   │   │   ├── kernels/          # CUDA kernel implementations
│   │   │   └── l3gpu_impl.cuh    # Main L3GPU class implementation
│   │   └── src/                  # Implementation files
│   │       ├── main.cu           # Entry point
│   │       └── main_impl.cu      # Implementation
│   │
│   └── modular/                  # Modular development version (14 files)
│       ├── codec/                # Compression/decompression kernels
│       │   ├── encoder.cu        # Encoding with model fitting
│       │   ├── encoder_optimized.cu # Optimized encoding
│       │   ├── decoder_specialized.cu # Specialized decoding
│       │   ├── decoder_warp_opt.cu # Warp-level optimized decoding
│       │   ├── decompression_kernels.cu # Full decompression kernels
│       │   ├── l3_codec.cpp      # Unified codec interface
│       │   └── l3_codec_optimized.cpp # Optimized interface
│       ├── utils/                # Utility functions
│       │   ├── bitpack_utils.cu  # Bit packing/unpacking
│       │   ├── partition_bounds_kernel.cu # Partition boundary computation
│       │   ├── random_access_kernels.cu # Random access to compressed data
│       │   └── timers.cu         # GPU timing utilities
│       └── data/                 # Data processing
│           ├── sosd_loader.cpp   # SOSD dataset loading
│           └── convert_to_binary.cpp # Format conversion tools
│
├── include/                       # Header files (public API)
│   ├── common/                   # Shared utilities
│   │   ├── l3_predicate_pushdown.cuh # Predicate pushdown optimization
│   │   ├── l3_ra_utils.cuh       # Random access utilities
│   │   ├── l3_alex_index.cuh     # ALEX index integration
│   │   └── ssb_utils.h           # Star Schema Benchmark utilities
│   ├── modular/                  # Modular version headers
│   │   ├── l3_format.hpp         # Format specification (invariant bitstream)
│   │   ├── l3_codec.hpp          # Codec interface
│   │   ├── l3_random_access.hpp  # Random access interface
│   │   └── l3_opt.h              # Optimization utilities
│   └── single_file/              # Single-file version headers
│       └── l3.cuh                # Main unified header
│
├── benchmarks/                    # Performance benchmark suite
│   ├── codec/                    # Compression/decompression benchmarks
│   │   ├── benchmark_kernel_only.cpp # Pure kernel performance
│   │   ├── benchmark_optimized.cpp # Optimization comparisons
│   │   ├── main_bench.cpp        # End-to-end benchmarks
│   │   └── sosd_bench_demo.cpp   # SOSD dataset benchmarks
│   ├── ssb/                      # Star Schema Benchmark queries
│   │   ├── baseline/             # 13 baseline SSB queries (uncompressed)
│   │   │   └── q11.cu - q43.cu   # Query Flight 1-4 implementations
│   │   └── optimized_2push/      # Optimized implementations
│   │       ├── l32.cu            # L3 compression implementation
│   │       ├── qXX_2push.cu      # 13 queries with 2-push optimization
│   │       └── qXX_l32.cu        # 13 queries with L3 compression
│   ├── random_access/            # Random access benchmarks (planned)
│   └── sosd/                     # SOSD specialized benchmarks (planned)
│
├── docs/                         # Comprehensive documentation
│   ├── README.md                 # Documentation index
│   ├── ARCHITECTURE.md           # System architecture overview
│   ├── INSTALLATION.md           # Installation guide
│   └── MIGRATION.md              # Migration guide from old structure
│
├── scripts/                      # Build and deployment scripts
│   ├── build.sh                  # Automated build script
│   ├── verify.sh                 # Environment verification
│   └── deploy.sh                 # Deployment/packaging script
│
├── tools/                        # Utility tools (empty - for future use)
├── examples/                     # Example programs (empty - for future use)
├── data/                         # Sample datasets (if any)
└── build/                        # Build directory (generated)
    ├── lib/                      # Compiled libraries
    │   ├── libl3_single.a        # Single-file version library
    │   └── libl3_modular.a       # Modular version library
    └── bin/                      # Executable binaries
        ├── codec_benchmarks/     # Codec benchmark binaries
        ├── ssb_baseline/         # SSB baseline query binaries
        └── ssb_optimized/        # SSB optimized query binaries
```

---

## 2. ALL PYTHON FILES AND NOTEBOOKS

**Status**: No Python files or Jupyter notebooks found in this codebase.
- This is a pure C++/CUDA project
- Build system uses CMake (not setuptools/conda)
- No Python bindings are currently implemented

---

## 3. GPU COMPRESSION IMPLEMENTATIONS

### 3.1 GPU Compression (Encoding)

**Primary Files**:
- `/root/autodl-tmp/L3/lib/modular/codec/encoder.cu` (27 KB)
  - Basic encoder with linear regression model fitting
  - Key function: Model fitting using least squares
  - Delta computation and validation
  - Overflow detection for large values

- `/root/autodl-tmp/L3/lib/modular/codec/encoder_optimized.cu` (24 KB)
  - Optimized version with improved compression ratio
  - Enhanced model selection strategies
  - Warp-level parallelization

- `/root/autodl-tmp/L3/lib/single_file/include/l3/kernels/compression_kernels_impl.cuh` (449 lines)
  - Kernel implementations: `wprocessPartitionsKernel`
  - `packDeltasKernelOptimized` for bit-packing
  - `setBitOffsetsKernel` for metadata management

**Algorithm Overview**:
1. Data partitioning (variable-length partitions)
2. Model fitting per partition (models: CONSTANT, LINEAR, POLYNOMIAL2, POLYNOMIAL3, DIRECT_COPY)
3. Delta computation: actual_value - predicted_value
4. Bit-packing: Store deltas using minimal bits per partition

**Data Structure** (from `l3_format.hpp`):
```cpp
template<typename T>
struct CompressedDataL3 {
    int32_t num_partitions;
    int32_t total_values;
    int32_t* d_model_types;           // Model type per partition
    double* d_model_params;           // θ₀, θ₁, θ₂, θ₃ per partition
    int32_t* d_delta_bits;            // Bits per delta
    int64_t* d_delta_array_bit_offsets;
    uint32_t* delta_array;            // Bit-packed deltas
};
```

---

### 3.2 GPU Decompression

**Primary Files**:
- `/root/autodl-tmp/L3/lib/modular/codec/decompression_kernels.cu` (20 KB)
  - Core decompression kernel: `decompressPartitionsOptimized`
  - Warp-cooperative bit unpacking
  - Shared memory optimization (258 words buffer)
  - Register tiling for multiple elements per thread

- `/root/autodl-tmp/L3/lib/modular/codec/decoder_specialized.cu` (15 KB)
  - Specialized decoder for specific data patterns
  - Model-aware decompression

- `/root/autodl-tmp/L3/lib/modular/codec/decoder_warp_opt.cu` (16 KB)
  - Warp-level optimizations
  - Coalesced memory access patterns
  - Reduced divergence

- `/root/autodl-tmp/L3/lib/single_file/include/l3/kernels/decompression_kernels_impl.cuh` (87 lines)
  - Fast decompression kernel: `decompressFullFile_OnTheFly_Optimized_V2`
  - Support for shared memory cache and pre-unpacked deltas

**Key Functions**:
- `extractDelta()`: Extract delta from bit-packed array
- `applyDelta()`: Apply delta to predicted value
- `applyModel()`: Apply linear/polynomial model for prediction

**Decompression Pipeline**:
1. Load partition metadata
2. Compute predicted values using model
3. Extract deltas from bit-packed array
4. Apply deltas: predicted + delta = actual

---

## 4. RANDOM ACCESS FUNCTIONALITY

**Primary File**:
- `/root/autodl-tmp/L3/lib/modular/utils/random_access_kernels.cu` (21 KB)

**Key Features**:
- **Partition Lookup**: Binary search to find partition containing element
- **Direct Value Extraction**: No full decompression needed
- **Range Query Support**: Efficient range lookups on compressed data

**Key Functions**:
```cpp
// Find which partition contains an index
__device__ int findPartition(
    const CompressedDataL3<T>* compressed,
    int global_idx)

// Extract single value without full decompression
__global__ void randomAccessKernel(
    const CompressedDataL3<T>* compressed,
    const int32_t* indices,
    T* output)

// Range query kernel
__global__ void rangeQueryKernel(
    const CompressedDataL3<T>* compressed,
    T min_val, T max_val,
    int32_t* results, int32_t* result_count)
```

**Optimization Techniques**:
- Warp-level parallelization for batch queries
- Shared memory caching of partition metadata
- Vectorized bit extraction

**Supporting Utility** (`l3_ra_utils.cuh`):
- Predicate-based partition pruning
- Conservative bound checking
- Model-based prediction ranges

---

## 5. QUERY EXECUTION

### 5.1 SSB Query Implementations

**Query Files** (26 total):
- **Baseline** (13 queries): `/root/autodl-tmp/L3/benchmarks/ssb/baseline/q{11-43}.cu`
- **2-Push Optimized** (13 queries): `/root/autodl-tmp/L3/benchmarks/ssb/optimized_2push/q{XX}_2push.cu`
- **L3 Compressed** (13 queries): `/root/autodl-tmp/L3/benchmarks/ssb/optimized_2push/q{XX}_l32.cu`

**Query Flights**:
1. **Flight 1** (Q1.1-1.3): Simple aggregation queries
2. **Flight 2** (Q2.1-2.3): Two-table joins
3. **Flight 3** (Q3.1-3.4): Three-table joins
4. **Flight 4** (Q4.1-4.3): Complex multi-table queries

### 5.2 Query Optimization Techniques

**Predicate Pushdown** (`l3_predicate_pushdown.cuh`):
```cpp
// Evaluate partition-level predicates before decompression
template<typename T>
__device__ __forceinline__ bool canPartitionMatch(
    const CompressedDataL3<T>* compressed,
    int partition_idx,
    T filter_min,
    T filter_max)
```

**Algorithm**:
1. Check partition min/max bounds against filter predicate
2. Skip partitions that cannot match (pruning)
3. Decompress only partitions that may contain matching values
4. Evaluate predicates on decompressed data

**Expected Speedup**:
- Highly selective filters (<5%): 1.5-3x faster
- Range queries on sorted data: 2-5x faster
- Random data: ~5-10% overhead from bound checking

### 5.3 Codec Benchmarks

**Programs** (`/root/autodl-tmp/L3/benchmarks/codec/`):
1. `benchmark_kernel_only.cpp` - Pure kernel performance
2. `benchmark_optimized.cpp` - Optimization comparisons
3. `main_bench.cpp` - End-to-end complete flow
4. `sosd_bench_demo.cpp` - SOSD dataset evaluation

---

## 6. CONFIGURATION FILES & DATA FILES

### 6.1 Build Configuration

**CMake Files**:
- `/root/autodl-tmp/L3/CMakeLists.txt` (101 lines)
  - Main project configuration
  - CUDA architecture settings (75, 80, 86, 89 - supports Turing to Hopper)
  - Include directories and output paths
  - Build options:
    - `BUILD_BENCHMARKS` (ON)
    - `BUILD_EXAMPLES` (ON)
    - `BUILD_TOOLS` (ON)
    - `ENABLE_TESTING` (OFF)
    - `USE_SINGLE_FILE` (ON)
    - `USE_MODULAR` (ON)

- `/root/autodl-tmp/L3/lib/single_file/CMakeLists.txt`
- `/root/autodl-tmp/L3/lib/modular/CMakeLists.txt`
- `/root/autodl-tmp/L3/benchmarks/CMakeLists.txt`

### 6.2 Data Files

**Directory**: `/root/autodl-tmp/L3/data/` (empty - for future sample data)

**SOSD Dataset Support**:
Files can be loaded via `sosd_loader.cpp`:
- books (200M keys)
- fb (200M keys)
- osm (800M keys)
- wiki (200M keys)

### 6.3 Configuration & Settings Files

**Claude Settings**:
- `/.claude/settings.local.json` - IDE configuration

---

## 7. PROJECT STRUCTURE & ORGANIZATION

### 7.1 Documentation Files

**Root Level**:
- `README.md` - Main project documentation
- `PROJECT_STRUCTURE.md` - Detailed structure (Chinese)
- `START_HERE.txt` - Quick start guide
- `QUICKSTART.md` - 5-minute getting started
- `PROJECT_SUMMARY.md` - Project overview
- `PROJECT_STATUS.md` - Current status
- `DELIVERY_NOTES.md` - Delivery documentation
- `FINAL_SUMMARY.txt` - Final project summary
- `FILE_MANIFEST.txt` - File listing
- `REFACTORING_REPORT.md` - Refactoring history
- `RENAMING_REPORT.md` - Library rename details
- `LIBRARY_RENAME_REPORT.md` - Rename documentation

**Documentation Directory** (`/docs/`):
- `README.md` - Documentation index
- `ARCHITECTURE.md` - System architecture
- `INSTALLATION.md` - Installation instructions
- `MIGRATION.md` - Migration guide

**Benchmark Documentation**:
- `/benchmarks/README.md` - Benchmarks overview
- `/benchmarks/codec/README.md` - Codec benchmarks
- `/benchmarks/ssb/README.md` - SSB benchmark guide
- `/benchmarks/CONSOLIDATION_REPORT.md` - Integration report
- `/benchmarks/STRUCTURE_OVERVIEW.txt` - Structure overview

**Library Documentation**:
- `/lib/single_file/README.md` - Single-file library guide
- `/lib/single_file/REFACTORING_README.md` - Refactoring details
- `/lib/single_file/REFACTORING_SUMMARY.txt` - Summary
- `/lib/modular/README.md` - Modular library guide
- `/lib/modular/ORGANIZATION_REPORT.md` - Organization report
- `/lib/modular/STRUCTURE.txt` - Structure visualization

### 7.2 Project Statistics

| Category | Count | Lines |
|----------|-------|-------|
| **CUDA Files** | 45 | ~35,000 |
| **C++ Files** | 8 | ~6,000 |
| **Header Files** | 30+ | ~10,000 |
| **Benchmarks** | 52 | ~35,000 |
| **Documentation** | 25+ | ~2,000 |
| **Total Files** | 127 | ~90,000 |

---

## 8. KEY TECHNOLOGIES & OPTIMIZATION TECHNIQUES

### 8.1 GPU Architecture Optimizations

1. **Warp-Level Primitives**
   - Cooperative loading into shared memory
   - Warp-level reduction and scan operations
   - Shuffle operations for register communication

2. **Memory Hierarchy Exploitation**
   - Shared memory staging (258-word buffers)
   - Coalesced global memory access
   - Register tiling for temporal locality

3. **Partition Parallelism**
   - 1 block per partition
   - Work-stealing for load balancing
   - Dynamic partitioning based on data characteristics

4. **Data Layout**
   - Struct of Arrays (SoA) format
   - Optimized for GPU access patterns
   - Cache-friendly metadata layout

### 8.2 Compression Techniques

1. **Learned Model-Based Compression**
   - Piecewise linear regression
   - Polynomial models (up to degree 3)
   - Adaptive model selection

2. **Residual Encoding**
   - Delta computation (actual - predicted)
   - Bit-packing with partition-specific widths
   - Sign extension for negative residuals

3. **Format Specification**
   - Magic number: 0x474C4543 ("GLEC")
   - Version: 1.0.0
   - Invariant bitstream layout for compatibility

### 8.3 Query Optimization

1. **Predicate Pushdown**
   - Partition-level filtering before decompression
   - Conservative bound checking
   - Model-based prediction ranges

2. **2-Push Optimization**
   - Two-stage filtering strategy
   - Reduced memory access overhead
   - Improved filter efficiency

---

## 9. HEADER FILES INVENTORY

### Common Headers
- `include/common/l3_alex_index.cuh` - ALEX index integration
- `include/common/l3_predicate_pushdown.cuh` - Predicate pushdown implementation
- `include/common/l3_ra_utils.cuh` - Random access utilities
- `include/common/ssb_l3_utils.cuh` - SSB benchmark utilities
- `include/common/ssb_utils.h` - SSB data structure definitions

### Modular Headers
- `include/modular/l3_codec.hpp` - Codec interface
- `include/modular/l3_format.hpp` - Format specification (invariant bitstream)
- `include/modular/l3_opt.h` - Optimization utilities
- `include/modular/l3_random_access.hpp` - Random access interface
- `include/modular/sosd_loader.h` - SOSD dataset loader

### Single-File Headers
- `include/single_file/l3.cuh` - Main unified header

---

## 10. BUILD INSTRUCTIONS

### Basic Build
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build with Specific CUDA Architecture
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="86"
```

### Build Only Libraries (No Benchmarks)
```bash
cmake .. -DBUILD_BENCHMARKS=OFF
make
```

### Build Specific Versions
```bash
# Single-file only
cmake .. -DUSE_SINGLE_FILE=ON -DUSE_MODULAR=OFF

# Modular only
cmake .. -DUSE_SINGLE_FILE=OFF -DUSE_MODULAR=ON
```

---

## 11. PERFORMANCE CHARACTERISTICS

### Expected Performance (NVIDIA A100)

| Metric | Value |
|--------|-------|
| Encoding Throughput | 28-32 GB/s |
| Decoding Throughput | 40-45 GB/s |
| Compression Ratio | 3.5-4.5x |
| Random Access Overhead | <5% |
| Predicate Pushdown Speedup (selective) | 1.5-3x |

### SSB Query Speedup (100M rows lineorder)

| Query Type | Baseline | L3 Compressed | Speedup |
|------------|----------|---------------|---------|
| Flight 1 | 8-10ms | 4-5ms | 2.0-2.2x |
| Flight 2 | 12-16ms | 7-9ms | 1.8-2.0x |
| Flight 3 | 18-24ms | 10-13ms | 1.8-2.0x |
| Flight 4 | 24-30ms | 13-17ms | 1.8-2.0x |

---

## 12. CRITICAL INVARIANTS & DESIGN PRINCIPLES

### Bitstream Format Invariants (from `l3_format.hpp`)
- **Format Version**: 0x00010000 (v1.0.0)
- **Magic Number**: 0x474C4543 ("GLEC")
- **Partition Non-Overlapping**: `end_indices[i] <= start_indices[i+1]`
- **Delta Encoding**: Signed 2's complement, sign-extended after extraction
- **Model Parameters**: Doubles (8 bytes each), up to 4 per partition

### Compression-Decompression Invariants
- **Encoder Output** must be decodable by any compatible decoder
- **Delta Calculation**: `calculateDelta(actual, predicted)` must be inverse of `applyDelta(predicted, delta)`
- **Overflow Handling**: Unsigned values >2^53 bypass model-based compression
- **Partition Boundaries**: Strictly non-overlapping, cover entire dataset

---

## CONCLUSION

L3 is a sophisticated, production-ready GPU compression library with:
- **Two implementation variants**: Single-file (production) and Modular (development)
- **Comprehensive benchmarks**: Codec performance + SSB query execution
- **Advanced optimizations**: Predicate pushdown, learned models, warp-level parallelization
- **Professional infrastructure**: CMake build system, extensive documentation, deployment scripts
- **Proven performance**: 2-4x query speedup through compression-based optimization

The codebase demonstrates GPU software engineering best practices with clear separation of concerns, invariant-based design, and comprehensive performance evaluation.

