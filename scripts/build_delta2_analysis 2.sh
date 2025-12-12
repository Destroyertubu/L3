#!/bin/bash
set -e

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --expt-relaxed-constexpr -Xcompiler -fPIC"

echo "════════════════════════════════════════════════════════════════"
echo "  Building Delta-of-Delta Potential Analysis Tool"
echo "════════════════════════════════════════════════════════════════"
echo ""

mkdir -p bin

echo "Compiling analysis tool..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    src/opt/phase3_advanced_codec/analyze_delta2_potential.cpp \
    bin/L3_codec.o bin/encoder.o bin/decoder_warp_opt.o bin/partition_bounds_kernel.o \
    -o bin/analyze_delta2_potential \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete: bin/analyze_delta2_potential"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To run the analysis:"
echo "  ./bin/analyze_delta2_potential"
echo ""
