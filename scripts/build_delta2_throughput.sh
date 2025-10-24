#!/bin/bash
set -e

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --expt-relaxed-constexpr -Xcompiler -fPIC"

echo "════════════════════════════════════════════════════════════════"
echo "  Building Phase 3 Delta2 Throughput Test"
echo "════════════════════════════════════════════════════════════════"
echo ""

mkdir -p bin

echo "Compiling throughput test..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    src/opt/phase3_advanced_codec/test_delta2_throughput.cpp \
    bin/L3_codec.o bin/encoder.o bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o bin/bitpack_utils_bucket.o \
    -o bin/test_delta2_throughput \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete: bin/test_delta2_throughput"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To run the test:"
echo "  ./bin/test_delta2_throughput"
echo ""
