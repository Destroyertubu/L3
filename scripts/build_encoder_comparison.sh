#!/bin/bash
# Build script for Encoder Comparison Test

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building Encoder Comparison Test"
echo "════════════════════════════════════════════════════════════"

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
CXX="g++"

COMMON_FLAGS="-std=c++17 -O3 -I./include -I./src/kernels/utils"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

PHASE2_USE_CP_ASYNC=${PHASE2_USE_CP_ASYNC:-0}
PHASE2_CTA_BATCH=${PHASE2_CTA_BATCH:-4}
PHASE2_PERSISTENT_THREADS=${PHASE2_PERSISTENT_THREADS:-0}
PHASE2_DEBUG_ROUTING=${PHASE2_DEBUG_ROUTING:-0}
PHASE2_DEBUG_VECTORIZATION=${PHASE2_DEBUG_VECTORIZATION:-0}

DEFINES="-DPHASE2_USE_CP_ASYNC=${PHASE2_USE_CP_ASYNC}"
DEFINES="${DEFINES} -DPHASE2_CTA_BATCH=${PHASE2_CTA_BATCH}"
DEFINES="${DEFINES} -DPHASE2_PERSISTENT_THREADS=${PHASE2_PERSISTENT_THREADS}"
DEFINES="${DEFINES} -DPHASE2_DEBUG_ROUTING=${PHASE2_DEBUG_ROUTING}"
DEFINES="${DEFINES} -DPHASE2_DEBUG_VECTORIZATION=${PHASE2_DEBUG_VECTORIZATION}"

echo "Build configuration:"
echo "  CUDA Architecture: SM ${CUDA_ARCH}"
echo "  CTA batch: ${PHASE2_CTA_BATCH}"
echo ""

mkdir -p bin

# 检查是否有必要的.o文件，如果没有则编译
if [ ! -f bin/bitpack_utils.o ]; then
    echo "[1/8] Compiling bitpack_utils.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/utils/bitpack_utils.cu -o bin/bitpack_utils.o
else
    echo "[1/8] bitpack_utils.o exists, skipping..."
fi

if [ ! -f bin/encoder.o ]; then
    echo "[2/8] Compiling encoder.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/compression/encoder.cu -o bin/encoder.o
else
    echo "[2/8] encoder.o exists, skipping..."
fi

if [ ! -f bin/encoder_variable_length.o ]; then
    echo "[3/8] Compiling encoder_variable_length.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/compression/encoder_variable_length.cu -o bin/encoder_variable_length.o
else
    echo "[3/8] encoder_variable_length.o exists, skipping..."
fi

if [ ! -f bin/decompression_kernels_phase2_bucket.o ]; then
    echo "[4/8] Compiling decompression_kernels_phase2_bucket.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/decompression/decompression_kernels_phase2_bucket.cu -o bin/decompression_kernels_phase2_bucket.o
else
    echo "[4/8] decompression_kernels_phase2_bucket.o exists, skipping..."
fi

if [ ! -f bin/decoder_warp_opt.o ]; then
    echo "[5/8] Compiling decoder_warp_opt.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/decompression/decoder_warp_opt.cu -o bin/decoder_warp_opt.o
else
    echo "[5/8] decoder_warp_opt.o exists, skipping..."
fi

if [ ! -f bin/L3_codec.o ]; then
    echo "[6/8] Compiling L3_codec.cpp..."
    ${CXX} ${COMMON_FLAGS} ${DEFINES} -I${CUDA_PATH}/include \
        -c src/codec/L3_codec.cpp -o bin/L3_codec.o
else
    echo "[6/8] L3_codec.o exists, skipping..."
fi

if [ ! -f bin/partition_bounds_kernel.o ]; then
    echo "[7/8] Compiling partition_bounds_kernel.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/kernels/utils/partition_bounds_kernel.cu -o bin/partition_bounds_kernel.o
else
    echo "[7/8] partition_bounds_kernel.o exists, skipping..."
fi

echo "[8/8] Linking test_encoder_comparison..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/tests/test_encoder_comparison.cpp \
    bin/L3_codec.o \
    bin/encoder.o \
    bin/encoder_variable_length.o \
    bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o \
    bin/bitpack_utils.o \
    -o bin/test_encoder_comparison \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/test_encoder_comparison"
echo ""
echo "Run with:"
echo "  ./bin/test_encoder_comparison /root/autodl-tmp/test/data/fb_200M_uint64.bin"
echo "  ./bin/test_encoder_comparison /root/autodl-tmp/test/data/fb_200M_uint64.bin 10000000"
echo ""
