#!/bin/bash
set -e

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
COMMON_FLAGS="-std=c++17 -O3 -I./include -I./src/opt/phase1_aos_layout"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --expt-relaxed-constexpr -Xcompiler -fPIC"

echo "════════════════════════════════════════════════════════════════"
echo "  Building Phase 1: AoS Compact Layout"
echo "════════════════════════════════════════════════════════════════"
echo ""

mkdir -p bin

echo "Step 1: Compiling AoS encoder..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    -c src/opt/phase1_aos_layout/encoder_aos.cpp \
    -o bin/encoder_aos.o

echo "Step 2: Compiling AoS decoder kernel..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    -c src/opt/phase1_aos_layout/decoder_aos.cu \
    -o bin/decoder_aos.o

echo "Step 3: Compiling complete test program..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    src/opt/phase1_aos_layout/test_aos_complete.cpp \
    bin/encoder_aos.o bin/decoder_aos.o \
    bin/L3_codec.o bin/encoder.o bin/decoder_warp_opt.o bin/partition_bounds_kernel.o \
    -o bin/test_aos_complete \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete: bin/test_aos_complete"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To run the complete test:"
echo "  ./bin/test_aos_complete"
echo ""
