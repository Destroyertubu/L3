#!/bin/bash
# Build script for GLECO decompression optimization benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
INC_DIR="$PROJECT_ROOT/include"
BIN_DIR="$PROJECT_ROOT/bin"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Building GLECO Decode Optimization Benchmarks              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Binary output: $BIN_DIR"
echo ""

# Create bin directory
mkdir -p "$BIN_DIR"

# CUDA compiler flags
NVCC_FLAGS="-std=c++17 -O3 -arch=sm_70 -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80"
NVCC_FLAGS="$NVCC_FLAGS -use_fast_math -maxrregcount=128"
NVCC_FLAGS="$NVCC_FLAGS -I$INC_DIR"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version | grep "release"
echo ""

# Build 1: Baseline benchmark
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building baseline benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvcc $NVCC_FLAGS \
    -o "$BIN_DIR/benchmark_decode_baseline" \
    "$SRC_DIR/benchmark_decode_baseline.cpp" \
    "$SRC_DIR/L3_codec.cpp" \
    "$SRC_DIR/encoder.cu" \
    "$SRC_DIR/decompression_kernels.cu" \
    "$SRC_DIR/decoder_warp_opt.cu" \
    "$SRC_DIR/partition_bounds_kernel.cu" \
    "$SRC_DIR/bitpack_utils.cu" \
    || { echo "❌ Failed to build baseline benchmark"; exit 1; }

echo "✓ Built: $BIN_DIR/benchmark_decode_baseline"
echo ""

# Build 2: Optimized kernels + comparison
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building optimized comparison benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvcc $NVCC_FLAGS \
    -o "$BIN_DIR/benchmark_decode_compare" \
    "$SRC_DIR/benchmark_decode_compare.cpp" \
    "$SRC_DIR/L3_codec.cpp" \
    "$SRC_DIR/encoder.cu" \
    "$SRC_DIR/decompression_kernels.cu" \
    "$SRC_DIR/decoder_warp_opt.cu" \
    "$SRC_DIR/decompression_kernels_opt.cu" \
    "$SRC_DIR/partition_bounds_kernel.cu" \
    "$SRC_DIR/bitpack_utils.cu" \
    || { echo "❌ Failed to build comparison benchmark"; exit 1; }

echo "✓ Built: $BIN_DIR/benchmark_decode_compare"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build complete!                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Available benchmarks:"
echo "  • $BIN_DIR/benchmark_decode_baseline"
echo "  • $BIN_DIR/benchmark_decode_compare"
echo ""
