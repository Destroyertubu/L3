#!/bin/bash
# Build script for reorganized project structure
# Uses decompression_kernels_phase2_bucket.cu for decompression

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building L3 Test (Reorganized Structure)"
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
echo "  Source directory structure: reorganized"
echo ""

mkdir -p bin

# Compile kernel utilities
echo "[1/7] Compiling bitpack_utils.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/kernels/utils/bitpack_utils.cu -o bin/bitpack_utils.o

# Compile encoder
echo "[2/7] Compiling encoder.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/kernels/compression/encoder.cu -o bin/encoder.o

# Compile Phase 2 Bucket decompression kernel
echo "[3/7] Compiling decompression_kernels_phase2_bucket.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/kernels/decompression/decompression_kernels_phase2_bucket.cu -o bin/decompression_kernels_phase2_bucket.o

# Compile decoder_warp_opt
echo "[4/7] Compiling decoder_warp_opt.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/kernels/decompression/decoder_warp_opt.cu -o bin/decoder_warp_opt.o

# Compile L3_codec
echo "[5/7] Compiling L3_codec.cpp..."
${CXX} ${COMMON_FLAGS} ${DEFINES} -I${CUDA_PATH}/include \
    -c src/codec/L3_codec.cpp -o bin/L3_codec.o

# Compile partition bounds kernel
echo "[6/7] Compiling partition_bounds_kernel.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/kernels/utils/partition_bounds_kernel.cu -o bin/partition_bounds_kernel.o

# Link test executable
echo "[7/7] Linking test_fb_200M_reorganized..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/tests/test_fb_200M_phase2_bucket.cpp \
    bin/L3_codec.o \
    bin/encoder.o \
    bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o \
    bin/bitpack_utils.o \
    -o bin/test_fb_200M_reorganized \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/test_fb_200M_reorganized"
echo ""
echo "Run with:"
echo "  ./bin/test_fb_200M_reorganized /root/autodl-tmp/test/data/fb_200M_uint64.bin"
echo ""
