#!/bin/bash
# Build script for Facebook 200M Dataset Test (Phase 2.2)

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building Facebook 200M Test (Phase 2.2 Bucket Scheduler)"
echo "════════════════════════════════════════════════════════════"

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
CXX="g++"

COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

PHASE2_USE_CP_ASYNC=${PHASE2_USE_CP_ASYNC:-0}
PHASE2_CTA_BATCH=${PHASE2_CTA_BATCH:-4}
PHASE2_PERSISTENT_THREADS=${PHASE2_PERSISTENT_THREADS:-0}
PHASE2_DEBUG_ROUTING=${PHASE2_DEBUG_ROUTING:-0}
PHASE2_DEBUG_VECTORIZATION=${PHASE2_DEBUG_VECTORIZATION:-1}

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

# Check if objects already exist, compile if needed
if [ ! -f bin/bitpack_utils_bucket.o ]; then
    echo "[1/7] Compiling bitpack_utils.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/bitpack_utils.cu -o bin/bitpack_utils_bucket.o
fi

if [ ! -f bin/encoder.o ]; then
    echo "[2/7] Compiling encoder.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/encoder.cu -o bin/encoder.o
fi

if [ ! -f bin/decompression_kernels_phase2_bucket.o ]; then
    echo "[3/7] Compiling decompression_kernels_phase2_bucket.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/decompression_kernels_phase2_bucket.cu -o bin/decompression_kernels_phase2_bucket.o
fi

if [ ! -f bin/decoder_warp_opt.o ]; then
    echo "[4/7] Compiling decoder_warp_opt.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/decoder_warp_opt.cu -o bin/decoder_warp_opt.o
fi

if [ ! -f bin/L3_codec.o ]; then
    echo "[5/7] Compiling L3_codec.cpp..."
    ${CXX} ${COMMON_FLAGS} ${DEFINES} -I${CUDA_PATH}/include \
        -c src/L3_codec.cpp -o bin/L3_codec.o
fi

if [ ! -f bin/partition_bounds_kernel.o ]; then
    echo "[6/7] Compiling partition_bounds_kernel.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
        -c src/partition_bounds_kernel.cu -o bin/partition_bounds_kernel.o
fi

echo "[7/7] Linking test_fb_200M_phase2_bucket..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/test_fb_200M_phase2_bucket.cpp \
    bin/L3_codec.o \
    bin/encoder.o \
    bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o \
    bin/bitpack_utils_bucket.o \
    -o bin/test_fb_200M_phase2_bucket \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/test_fb_200M_phase2_bucket"
echo ""
echo "Run with:"
echo "  ./bin/test_fb_200M_phase2_bucket"
echo ""
