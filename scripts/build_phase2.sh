#!/bin/bash
# Build script for GLECO Phase 2 decompression optimizations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SRC_DIR="$PROJECT_ROOT/src"
INC_DIR="$PROJECT_ROOT/include"
BIN_DIR="$PROJECT_ROOT/bin"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Building GLECO Phase 2 Optimization Suite                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Binary output: $BIN_DIR"
echo ""

# Create bin directory
mkdir -p "$BIN_DIR"

# CUDA compiler flags
# SM70: V100, SM75: Turing (T4), SM80: A100, SM86: RTX 30xx
NVCC_FLAGS="-std=c++17 -O3"
NVCC_FLAGS="$NVCC_FLAGS -gencode=arch=compute_70,code=sm_70"
NVCC_FLAGS="$NVCC_FLAGS -gencode=arch=compute_75,code=sm_75"
NVCC_FLAGS="$NVCC_FLAGS -gencode=arch=compute_80,code=sm_80"
NVCC_FLAGS="$NVCC_FLAGS -gencode=arch=compute_86,code=sm_86"
NVCC_FLAGS="$NVCC_FLAGS -use_fast_math"
NVCC_FLAGS="$NVCC_FLAGS -I$INC_DIR"

# Enable Phase 2 optimizations (F4 FIX: PERSISTENT_THREADS disabled)
NVCC_FLAGS="$NVCC_FLAGS -DPHASE2_USE_CP_ASYNC=1"
NVCC_FLAGS="$NVCC_FLAGS -DPHASE2_PERSISTENT_THREADS=0"  # Disabled: poor performance
NVCC_FLAGS="$NVCC_FLAGS -DPHASE2_VEC_LOADS=1"
NVCC_FLAGS="$NVCC_FLAGS -DPHASE2_AUTOTUNE=1"

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version | grep "release"
echo ""

# Common source files
COMMON_SRCS="$SRC_DIR/L3_codec.cpp \
    $SRC_DIR/encoder.cu \
    $SRC_DIR/decompression_kernels.cu \
    $SRC_DIR/decoder_warp_opt.cu \
    $SRC_DIR/partition_bounds_kernel.cu \
    $SRC_DIR/bitpack_utils.cu"

# Build 1: Phase 2 comprehensive benchmark
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building Phase 2 comprehensive benchmark..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvcc $NVCC_FLAGS \
    -o "$BIN_DIR/benchmark_decode_phase2" \
    "$SRC_DIR/benchmark_decode_phase2.cpp" \
    $COMMON_SRCS \
    "$SRC_DIR/decompression_kernels_opt.cu" \
    "$SRC_DIR/decompression_kernels_phase2.cu" \
    || { echo "❌ Failed to build Phase 2 benchmark"; exit 1; }

echo "✓ Built: $BIN_DIR/benchmark_decode_phase2"
echo ""

# Build 2: Auto-tuning tool
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building auto-tuning tool..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvcc $NVCC_FLAGS \
    -o "$BIN_DIR/autotune_phase2" \
    "$SRC_DIR/autotune_phase2.cpp" \
    $COMMON_SRCS \
    "$SRC_DIR/decompression_kernels_phase2.cu" \
    || { echo "❌ Failed to build auto-tuning tool"; exit 1; }

echo "✓ Built: $BIN_DIR/autotune_phase2"
echo ""

# Build 3: Quick correctness test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Building quick correctness test..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

nvcc $NVCC_FLAGS \
    -o "$BIN_DIR/test_phase2_quick" \
    "$SRC_DIR/test_phase2_quick.cpp" \
    $COMMON_SRCS \
    "$SRC_DIR/decompression_kernels_phase2.cu" \
    || { echo "❌ Failed to build quick test"; exit 1; }

echo "✓ Built: $BIN_DIR/test_phase2_quick"
echo ""

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Build Complete!                                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Available binaries:"
echo "  • $BIN_DIR/benchmark_decode_phase2"
echo "  • $BIN_DIR/autotune_phase2"
echo "  • $BIN_DIR/test_phase2_quick"
echo ""
echo "Quick start:"
echo "  1. Run correctness: ./bin/test_phase2_quick"
echo "  2. Auto-tune: ./bin/autotune_phase2"
echo "  3. Full benchmark: ./bin/benchmark_decode_phase2"
echo ""
