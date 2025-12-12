# L3 Parameter Mapping Documentation

This document maps all valid parameter combinations in `main.cpp` to their corresponding source files.

## Table of Contents
1. [Encoder Types](#1-encoder-types)
2. [Decoder Types](#2-decoder-types)
3. [Model Selection Strategies](#3-model-selection-strategies)
4. [Partitioning Strategies](#4-partitioning-strategies)
5. [Vertical Decompression Modes](#5-Vertical-decompression-modes)
6. [Valid Encoder-Decoder Pairings](#6-valid-encoder-decoder-pairings)
7. [Datasets](#7-datasets)
8. [Command Line Arguments](#8-command-line-arguments)
9. [Example Commands](#9-example-commands)

---

## 1. Encoder Types

> **IMPORTANT (v3.0)**: The `Vertical` encoder is an ALIAS for `STANDARD`. Both use `compressDataWithConfig()`
> in L3_codec.cpp. The difference is only in decoder selection. Only `GPU` and `GPU_ZEROSYNC` produce true
> Vertical vertical format.

| Encoder | CLI Flag | Description | Actual Implementation | Compatible Decoders |
|---------|----------|-------------|----------------------|---------------------|
| `STANDARD` | `--encoder STANDARD` | Default L3 encoder with horizontal layout | `compressDataWithConfig()` in `src/L3_codec.cpp` | All L3 decoders |
| `Vertical` | `--encoder Vertical` | **ALIAS for STANDARD** - same code path | `compressDataWithConfig()` in `src/L3_codec.cpp` | Vertical decoder (for compatibility) |
| `OPTIMIZED` | `--encoder OPTIMIZED` | L3 encoder with merging disabled | `compressDataWithConfig()` in `src/L3_codec.cpp` | All L3 decoders |
| `GPU` | `--encoder GPU` | True Vertical vertical layout (GPU pipeline) | `encodeVerticalGPU()` in `src/kernels/compression/encoder_Vertical_opt.cu` | Vertical only |
| `GPU_ZEROSYNC` | `--encoder GPU_ZEROSYNC` | True Vertical with zero mid-sync | `encodeVerticalGPU_ZeroSync()` in `src/kernels/compression/encoder_Vertical_opt.cu` | Vertical only |

### Encoder Type Enum (main.cpp:50-56)
```cpp
enum class EncoderType {
    STANDARD,      // Default L3 encoder
    Vertical,     // Vertical CPU encoder
    OPTIMIZED,     // High-throughput L3 encoder
    GPU,           // GPU-only pipeline
    GPU_ZEROSYNC   // Zero mid-sync GPU encoder
};
```

### Encoder File Details

#### STANDARD/OPTIMIZED Encoder
- **File**: `src/L3_codec.cpp`
- **Functions**:
  - `L3Codec::compress()` - main compression entry point
  - `L3Codec::compressPartition()` - per-partition compression
- **Data Format**: Horizontal (row-major) bitpacked layout

#### Vertical/GPU/GPU_ZEROSYNC Encoders
- **File**: `src/kernels/compression/encoder_Vertical_opt.cu`
- **Functions**:
  - `encodeAllPartitionsVertical()` - CPU Vertical encoding
  - `encodeAllPartitionsGPU()` - GPU encoding pipeline
  - `encodeAllPartitionsGPUZeroSync()` - Zero-sync GPU variant
- **Data Format**: Vertical (column-interleaved) Vertical layout

---

## 2. Decoder Types

| Decoder | CLI Flag | Description | Source File | Kernel Function |
|---------|----------|-------------|-------------|-----------------|
| `STANDARD` | `--decoder STANDARD` | Default L3 decoder with shared memory staging | `src/kernels/decompression/decompression_kernels.cu` | `launchDecompressOptimized()` |
| `Vertical` | `--decoder Vertical` | Interleaved + branchless Vertical decoder | `src/kernels/decompression/decoder_Vertical_opt.cu` | `decompressAll()` |
| `OPTIMIZED` | `--decoder OPTIMIZED` | Warp-optimized with double buffering | `src/kernels/decompression/decoder_warp_opt.cu` | `launchDecompressWarpOpt()` |
| `SPECIALIZED` | `--decoder SPECIALIZED` | Template-specialized per model type | `src/kernels/decompression/decoder_specialized.cu` | `launchDecompressSpecialized()` |
| `PHASE2` | `--decoder PHASE2` | cp.async pipeline (SM 80+) | `src/kernels/decompression/decompression_kernels_phase2.cu` | `decompressL3_Phase2()` |
| `PHASE2_BUCKET` | `--decoder PHASE2_BUCKET` | Bucket dispatch by delta_bits | `src/kernels/decompression/decompression_kernels_phase2_bucket.cu` | `decompressL3_Phase2_Bucket()` |
| `KERNELS_OPT` | `--decoder KERNELS_OPT` | 8/16-bit specialized paths | `src/kernels/decompression/decompression_kernels_opt.cu` | `decompressL3_Optimized()` |

### Decoder Type Enum (main.cpp:58-67)
```cpp
enum class DecoderType {
    STANDARD,       // Default decompression
    Vertical,      // Vertical optimized decoder
    OPTIMIZED,      // Warp-optimized decoder
    SPECIALIZED,    // Template-specialized decoder
    PHASE2,         // Phase 2 optimized decoder
    PHASE2_BUCKET,  // Phase 2 with bucket dispatch
    KERNELS_OPT     // Optimized 8/16-bit kernels
};
```

### Decoder File Details

#### STANDARD Decoder
- **File**: `src/kernels/decompression/decompression_kernels.cu`
- **Kernel**: `decompressPartitionsOptimized<T>`
- **Features**:
  - Shared memory staging (256 words buffer)
  - Warp-cooperative loading
  - Support for all model types (LINEAR, POLY2, POLY3, FOR, DIRECT_COPY)
  - Multi-word extraction for >32-bit deltas

#### Vertical Decoder
- **File**: `src/kernels/decompression/decoder_Vertical_opt.cu`
- **Kernels**:
  - `decompressInterleavedAllPartitions()` - unified interleaved kernel
  - `decompressSequentialBranchless()` - branchless sequential
- **Features**:
  - Vertical (interleaved) memory layout
  - Register buffering (20 words per thread)
  - Branchless bit extraction
  - Mini-vector (256 lanes) parallelism

#### OPTIMIZED Decoder (Warp-Opt)
- **File**: `src/kernels/decompression/decoder_warp_opt.cu`
- **Kernel**: `decompressWarpOptimized<T>`
- **Features**:
  - Double-buffered shared memory staging
  - Funnel shift for cross-word bit extraction
  - cp.async on SM 80+ (Ampere)
  - Warp-cooperative async loads

#### SPECIALIZED Decoder
- **File**: `src/kernels/decompression/decoder_specialized.cu`
- **Kernel**: `decompressPartitionsSpecialized<T, MODEL_TYPE>`
- **Features**:
  - Template-specialized per model type
  - Reduced branch divergence
  - Compile-time model type specialization

#### PHASE2 Decoder
- **File**: `src/kernels/decompression/decompression_kernels_phase2.cu`
- **Kernel**: `decompressL3_Phase2<T>`
- **Features**:
  - cp.async pipeline for memory latency hiding
  - Requires SM 80+ (Ampere or newer)
  - Producer-consumer pattern

#### PHASE2_BUCKET Decoder
- **File**: `src/kernels/decompression/decompression_kernels_phase2_bucket.cu`
- **Kernel**: `decompressL3_Phase2_Bucket<T>`
- **Features**:
  - Bucket dispatch based on delta_bits
  - Optimized kernels for common bit widths (8, 16, 24, 32)

#### KERNELS_OPT Decoder
- **File**: `src/kernels/decompression/decompression_kernels_opt.cu`
- **Kernel**: `decompressL3_Optimized<T>`
- **Features**:
  - Specialized paths for 8-bit and 16-bit deltas
  - Vectorized loads for narrow bit widths

---

## 3. Model Selection Strategies

> **Note**: Default model is `LINEAR` (not ADAPTIVE). Polynomial models (POLY2, POLY3) are beta features
> with auto-fallback logic for small partitions.

| Strategy | CLI Flag | Description | Model Types Used | Default |
|----------|----------|-------------|------------------|---------|
| `ADAPTIVE` | `--model-selection ADAPTIVE` | Auto-select best model per partition | LINEAR, POLY2, POLY3, FOR | No |
| `LINEAR` | `--model-selection LINEAR` | Fixed linear model only | LINEAR (y = a + bx) | **YES** |
| `POLY2` | `--model-selection POLY2` | Fixed quadratic model (beta) | POLYNOMIAL2 (y = a + bx + cx²) | No |
| `POLY3` | `--model-selection POLY3` | Fixed cubic model (beta) | POLYNOMIAL3 (y = a + bx + cx² + dx³) | No |
| `FOR` | `--model-selection FOR` | Fixed FOR+BitPack model only | FOR_BITPACK (base + unsigned delta) | No |

### Model Selection Enum (main.cpp:69-75)
```cpp
enum class ModelSelectionStrategy {
    ADAPTIVE,    // Auto-select best model per partition
    LINEAR,      // Force linear model
    POLY2,       // Force polynomial degree 2
    POLY3,       // Force polynomial degree 3
    FOR          // Force FOR+BitPack
};
```

### Model Type Enum (L3_format.hpp)

> **WARNING**: `MODEL_CONSTANT` and `MODEL_DIRECT_COPY` are **DEAD CODE** - defined but never used in v3.0.

```cpp
enum ModelType {
    MODEL_CONSTANT = 0,      // DEAD CODE - never selected or fitted
    MODEL_LINEAR = 1,        // y = a + bx (linear prediction)
    MODEL_POLYNOMIAL2 = 2,   // y = a + bx + cx² (quadratic)
    MODEL_POLYNOMIAL3 = 3,   // y = a + bx + cx² + dx³ (cubic)
    MODEL_FOR_BITPACK = 4,   // Frame of Reference + BitPack
    MODEL_DIRECT_COPY = 5    // DEAD CODE - never selected
};
```

### Auto-Fallback Logic for Polynomial Models

When partition size is too small for polynomial fitting:
- **POLY3**: Falls back to POLY2 if n ≤ 20, then to LINEAR if n ≤ 10
- **POLY2**: Falls back to LINEAR if n ≤ 10

### Source Files for Model Selection
- **Adaptive Selection**: `src/kernels/partitioning/partitioning_v2_gpu.cu`
  - Function: `selectBestModel()` - evaluates all models, picks lowest cost
- **Model Parameter Fitting**: `src/kernels/partitioning/partitioning_v2_gpu.cu`
  - Function: `fitLinearModel()`, `fitPolynomialModel()`, `calculateFORModel()`

---

## 4. Partitioning Strategies

> **Note**: `FIXED` is the actual default partitioning strategy. `VARIANCE_ADAPTIVE` is marked as legacy
> and may cause "partition explosion" with certain data patterns.

| Strategy | CLI Flag | Description | Source File | Status |
|----------|----------|-------------|-------------|--------|
| `FIXED` | `--partitioning FIXED` | Fixed-size partitions | `src/kernels/partitioning/partitioning.cu` | **DEFAULT** |
| `COST_OPTIMAL` | `--partitioning COST_OPTIMAL` | Cost-based with greedy merging | `src/kernels/partitioning/partitioning_v2_gpu.cu` | Recommended |
| `VARIANCE_ADAPTIVE` | `--partitioning VARIANCE_ADAPTIVE` | Variance-based adaptive sizing | `src/kernels/partitioning/partitioning.cu` | **LEGACY** (may cause explosion) |

### Partitioning Strategy Enum (main.cpp:77-81)
```cpp
enum class PartitioningStrategy {
    FIXED,             // Fixed partition size
    COST_OPTIMAL,      // Cost-optimal with merging
    VARIANCE_ADAPTIVE  // Variance-based adaptive
};
```

### Partitioning File Details

#### FIXED Partitioning
- **File**: `src/kernels/partitioning/partitioning.cu`
- **Function**: `createFixedPartitions()`
- **Behavior**: Creates equal-size partitions of `--partition-size` elements

#### COST_OPTIMAL Partitioning
- **File**: `src/kernels/partitioning/partitioning_v2_gpu.cu`
- **Functions**:
  - `partitionCostOptimal()` - main entry point
  - `greedyMergePartitions()` - merge adjacent partitions if beneficial
  - `calculatePartitionCost()` - compute compression cost
- **Behavior**: Iteratively merges partitions when combined cost is lower

#### VARIANCE_ADAPTIVE Partitioning
- **File**: `src/kernels/partitioning/partitioning.cu`
- **Function**: `createVarianceAdaptivePartitions()`
- **Behavior**: Splits partitions with high variance, merges low-variance

---

## 5. Vertical Decompression Modes

> **IMPORTANT (v3.0)**: All decompression modes route to the same kernel. The mode parameter has NO effect on which kernel is called.

| Mode | CLI Flag | Actual Kernel | Status |
|------|----------|---------------|--------|
| `SEQUENTIAL` | `--decompress-mode SEQUENTIAL` | `decompressInterleavedAllPartitions()` | DEPRECATED |
| `INTERLEAVED` | `--decompress-mode INTERLEAVED` | `decompressInterleavedAllPartitions()` | Primary |
| `BRANCHLESS` | `--decompress-mode BRANCHLESS` | `decompressInterleavedAllPartitions()` | Removed in v3.0 |
| `AUTO` | `--decompress-mode AUTO` | `decompressInterleavedAllPartitions()` | Default |

### DecompressMode Enum (decoder_Vertical_opt.cu)
```cpp
enum class DecompressMode {
    SEQUENTIAL,   // Sequential per-partition
    INTERLEAVED,  // Interleaved mini-vector
    BRANCHLESS,   // Branchless extraction
    AUTO          // Auto-select best mode
};
```

### Code Evidence (decoder_Vertical_opt.cu:1237-1312)

The `decompressAll()` function routes ALL modes to the same kernel:

```cpp
void decompressAll(..., DecompressMode mode, ...) {
    if (mode == DecompressMode::INTERLEAVED && ...) {
        decompressInterleavedAllPartitions<T><<<...>>>();  // Line 1250
    }
    else if (mode == DecompressMode::BRANCHLESS) {
        // v3.0: BRANCHLESS now uses INTERLEAVED format (sequential removed)
        decompressInterleavedAllPartitions<T><<<...>>>();  // Line 1267
    }
    else if (mode == DecompressMode::SEQUENTIAL) {
        // DEPRECATED: Sequential format removed in v3.0
        decompressInterleavedAllPartitions<T><<<...>>>();  // Line 1284
    }
    else  // AUTO mode
    {
        // v3.0: Default to INTERLEAVED (only format available)
        decompressInterleavedAllPartitions<T><<<...>>>();  // Line 1300
    }
}
```

### Dead Code (Defined but Never Called)
These kernels exist in the source file but are NOT used by `decompressAll()`:
- `decompressSequentialBranchless()` - never called
- `decompressSequentialWarpCooperative()` - never called
- `decompressInterleavedMiniVector()` - never called
- `decompressBatchAdaptive()` - never called

---

## 6. Valid Encoder-Decoder Pairings

### Compatibility Matrix

> **Note**: Vertical encoder produces L3 format (same as STANDARD), but is paired with Vertical decoder for legacy compatibility.

| Encoder | STANDARD | Vertical | OPTIMIZED | SPECIALIZED | PHASE2 | PHASE2_BUCKET | KERNELS_OPT |
|---------|:--------:|:---------:|:---------:|:-----------:|:------:|:-------------:|:-----------:|
| STANDARD | Yes | No | Yes | Yes | Yes | Yes | Yes |
| OPTIMIZED | Yes | No | Yes | Yes | Yes | Yes | Yes |
| Vertical | Yes* | Yes* | Yes* | Yes* | Yes* | Yes* | Yes* |
| GPU | No | Yes | No | No | No | No | No |
| GPU_ZEROSYNC | No | Yes | No | No | No | No | No |

*Vertical encoder produces L3 format - theoretically compatible with all L3 decoders, but paired with Vertical decoder by convention.

### Rules
```
L3 Encoders (STANDARD, OPTIMIZED, Vertical*):
  - Data Format: Horizontal (row-major) bitpacked layout
  - Compatible Decoders: STANDARD, OPTIMIZED, SPECIALIZED, PHASE2, PHASE2_BUCKET, KERNELS_OPT
  - *Vertical encoder uses same code as STANDARD

True Vertical Encoders (GPU, GPU_ZEROSYNC):
  - Data Format: Vertical (column-interleaved) Vertical layout
  - Compatible Decoders: Vertical only
```

### Validation Code (main.cpp:450-465)
```cpp
// Validate encoder-decoder compatibility
bool isValidPairing(EncoderType encoder, DecoderType decoder) {
    bool isVerticalEncoder = (encoder == EncoderType::Vertical ||
                               encoder == EncoderType::GPU ||
                               encoder == EncoderType::GPU_ZEROSYNC);
    bool isVerticalDecoder = (decoder == DecoderType::Vertical);

    // Vertical encoders require Vertical decoder
    if (isVerticalEncoder && !isVerticalDecoder) return false;
    // L3 decoders require L3 encoder
    if (!isVerticalEncoder && isVerticalDecoder) return false;

    return true;
}
```

---

## 7. Datasets

### Numeric Datasets (SOSD)

| ID | Name | Type | Size | Description |
|----|------|------|------|-------------|
| 1 | linear | uint64 | 200M | Synthetic linear sequence |
| 2 | normal | uint64 | 200M | Normal distribution |
| 3 | poisson | uint64 | 200M | Poisson distribution |
| 4 | ml | uint64 | 200M | Machine learning indices |
| 5 | books | uint32 | 200M | Book publication dates |
| 6 | fb | uint64 | 200M | Facebook user IDs |
| 7 | wiki | uint64 | 200M | Wikipedia timestamps |
| 8 | osm | uint64 | 800M | OpenStreetMap node IDs |
| 9 | movieid | uint32 | 200M | MovieLens movie IDs |
| 10 | house_price | uint32 | 200M | Housing prices |
| 11 | planet | uint64 | 200M | Astronomy data |
| 12 | libio | uint64 | 200M | Library.io timestamps |
| 13 | medicare | uint32 | 200M | Medicare claims |
| 14 | cosmos | uint64 | 200M | Cosmology simulation |
| 15 | polylog | uint64 | 200M | Polylogarithmic sequence |
| 16 | exp | uint64 | 200M | Exponential sequence |
| 17 | poly | uint64 | 200M | Polynomial sequence |
| 18 | site | uint32 | 200M | Website visit counts |
| 19 | weight | uint32 | 200M | Weight measurements |
| 20 | adult | uint32 | 200M | Adult income data |

### String Datasets

| ID | Name | Type | Description |
|----|------|------|-------------|
| 21 | email | string | Email addresses |
| 22 | hex | string | Hexadecimal strings |
| 23 | words | string | Dictionary words |

### Dataset Loading Code (main.cpp:200-300)
```cpp
std::string getDatasetPath(int dataset_id, const std::string& base_path) {
    std::map<int, std::string> paths = {
        {1, "sosd/1-linear_200M_uint64.bin"},
        {2, "sosd/2-normal_200M_uint64.bin"},
        // ... additional datasets
    };
    return base_path + "/" + paths[dataset_id];
}
```

---

## 8. Command Line Arguments

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `<dataset_id>` | int | Dataset number (1-23) |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--encoder <type>` | string | STANDARD | Encoder type |
| `--decoder <type>` | string | STANDARD | Decoder type |
| `--partition-size <n>` | int | 4096 | Partition size in elements |
| `--model-selection <strategy>` | string | ADAPTIVE | Model selection strategy |
| `--partitioning <strategy>` | string | COST_OPTIMAL | Partitioning strategy |
| `--decompress-mode <mode>` | string | INTERLEAVED | Vertical decompress mode |
| `--compare-Vertical` | flag | false | Compare with Vertical baseline |
| `--warmup <n>` | int | 3 | Warmup iterations |
| `--runs <n>` | int | 5 | Timed runs for median |
| `--verify` | flag | true | Enable correctness verification |
| `--verbose` | flag | false | Verbose output |

---

## 9. Example Commands

### Basic Usage
```bash
# Run dataset 6 (fb) with defaults
./bin/main 6

# Run dataset 7 (wiki) with custom partition size
./bin/main 7 --partition-size 2048
```

### L3 Pipeline
```bash
# L3 with warp-optimized decoder
./bin/main 6 --encoder STANDARD --decoder OPTIMIZED

# L3 with Phase 2 cp.async decoder (SM 80+)
./bin/main 6 --encoder STANDARD --decoder PHASE2

# L3 with template-specialized decoder
./bin/main 6 --encoder STANDARD --decoder SPECIALIZED
```

### Vertical Pipeline
```bash
# Vertical CPU encoder with Vertical decoder
./bin/main 6 --encoder Vertical --decoder Vertical

# GPU Vertical encoder
./bin/main 6 --encoder GPU --decoder Vertical

# Zero-sync GPU encoder
./bin/main 6 --encoder GPU_ZEROSYNC --decoder Vertical
```

### Model Selection
```bash
# Force linear model only
./bin/main 6 --model-selection LINEAR

# Force FOR+BitPack model
./bin/main 6 --model-selection FOR

# Adaptive model selection (default)
./bin/main 6 --model-selection ADAPTIVE
```

### Partitioning Strategies
```bash
# Fixed partitioning with 2048-element partitions
./bin/main 6 --partitioning FIXED --partition-size 2048

# Cost-optimal partitioning with merging
./bin/main 6 --partitioning COST_OPTIMAL

# Variance-adaptive partitioning
./bin/main 6 --partitioning VARIANCE_ADAPTIVE
```

### Benchmarking
```bash
# Full benchmark with comparison
./bin/main 6 --compare-Vertical --warmup 5 --runs 10

# Verbose output with verification
./bin/main 6 --verbose --verify
```

---

## File Structure Summary

```
src/
├── L3_codec.cpp                          # STANDARD/OPTIMIZED/Vertical* encoder (*same as STANDARD)
├── kernels/
│   ├── compression/
│   │   └── encoder_Vertical_opt.cu      # GPU/GPU_ZEROSYNC encoders (true Vertical format)
│   ├── decompression/
│   │   ├── decompression_kernels.cu      # STANDARD decoder (7 distinct decoders)
│   │   ├── decoder_Vertical_opt.cu      # Vertical decoder (all modes→same kernel in v3.0)
│   │   ├── decoder_warp_opt.cu           # OPTIMIZED decoder
│   │   ├── decoder_specialized.cu        # SPECIALIZED decoder
│   │   ├── decompression_kernels_phase2.cu    # PHASE2 decoder
│   │   ├── decompression_kernels_phase2_bucket.cu  # PHASE2_BUCKET decoder
│   │   └── decompression_kernels_opt.cu  # KERNELS_OPT decoder
│   └── partitioning/
│       ├── partitioning.cu               # FIXED (default), VARIANCE_ADAPTIVE (legacy)
│       └── partitioning_v2_gpu.cu        # COST_OPTIMAL + model selection
└── include/
    ├── L3_format.hpp                     # ModelType enum (includes 2 dead code entries)
    ├── L3.h                          # CompressedDataOpt struct
    └── bitpack_utils.cuh                 # Bit manipulation utilities
```

---

*Generated: 2025-12-09*
