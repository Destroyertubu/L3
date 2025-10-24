#!/bin/bash
set -e

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
COMMON_FLAGS="-std=c++17 -O3 -I./include -I./src/opt/phase3_advanced_codec"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --expt-relaxed-constexpr -Xcompiler -fPIC"

echo "════════════════════════════════════════════════════════════════"
echo "  Building GLECO Phase 3: Delta-of-Delta Codec Test"
echo "════════════════════════════════════════════════════════════════"
echo ""

mkdir -p bin

echo "Step 1: Compiling Phase 3 test..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    src/opt/phase3_advanced_codec/test_delta2_codec.cpp \
    bin/L3_codec.o bin/encoder.o bin/partition_bounds_kernel.o \
    -o bin/test_delta2_codec \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete: bin/test_delta2_codec"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To run the test:"
echo "  ./bin/test_delta2_codec"
echo ""
