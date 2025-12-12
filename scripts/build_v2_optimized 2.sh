#!/bin/bash
# Build script for Phase 2.3 - V2: Block+LOP3+ILP Optimizations

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building V2: Block Size + LOP3 + ILP Optimizations"
echo "════════════════════════════════════════════════════════════"

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
CXX="g++"

COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

# V2 specific defines
BLOCK_SIZE_V2=${BLOCK_SIZE_V2:-512}
USE_LOP3_V2=${USE_LOP3_V2:-1}
ILP_20_21_V2=${ILP_20_21_V2:-1}

DEFINES="-DBLOCK_SIZE_V2=${BLOCK_SIZE_V2}"
DEFINES="${DEFINES} -DUSE_LOP3_V2=${USE_LOP3_V2}"
DEFINES="${DEFINES} -DILP_20_21_V2=${ILP_20_21_V2}"

echo "V2 Configuration:"
echo "  CUDA Architecture: SM ${CUDA_ARCH}"
echo "  Block size: ${BLOCK_SIZE_V2} threads"
echo "  LOP3 optimization: ${USE_LOP3_V2}"
echo "  20-21 bit ILP: ${ILP_20_21_V2}"
echo ""

mkdir -p bin

# Reuse encoder and codec objects
if [ ! -f bin/encoder.o ]; then
    echo "[1/5] Compiling encoder.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/encoder.cu -o bin/encoder.o
fi

if [ ! -f bin/L3_codec.o ]; then
    echo "[2/5] Compiling L3_codec.cpp..."
    ${CXX} ${COMMON_FLAGS} -I${CUDA_PATH}/include \
        -c src/L3_codec.cpp -o bin/L3_codec.o
fi

if [ ! -f bin/partition_bounds_kernel.o ]; then
    echo "[3/5] Compiling partition_bounds_kernel.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/partition_bounds_kernel.cu -o bin/partition_bounds_kernel.o
fi

# Compile V2 kernel
echo "[4/5] Compiling v2_block_lop3_ilp.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/decode/v2_block_lop3_ilp.cu -o bin/v2_block_lop3_ilp.o

# Compile decoder_warp_opt if needed
if [ ! -f bin/decoder_warp_opt.o ]; then
    echo "[4.5/5] Compiling decoder_warp_opt.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/decoder_warp_opt.cu -o bin/decoder_warp_opt.o
fi

# Link V2 test
echo "[5/5] Linking test_v2_optimized..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/decode/test_v2_optimized.cpp \
    bin/L3_codec.o \
    bin/encoder.o \
    bin/decoder_warp_opt.o \
    bin/v2_block_lop3_ilp.o \
    bin/partition_bounds_kernel.o \
    -o bin/test_v2_optimized \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/test_v2_optimized"
echo ""
echo "Run with:"
echo "  ./bin/test_v2_optimized"
echo ""
