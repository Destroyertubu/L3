#!/bin/bash
# Build script for L3 Phase 2.2 Bucket-Based Decompression

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building L3 Phase 2.2 (Bucket-Based 64-bit Support)"
echo "════════════════════════════════════════════════════════════"

# Configuration
CUDA_ARCH="90"  # H20 (Hopper)
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
CXX="g++"

COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

# Compilation options (can be overridden)
PHASE2_USE_CP_ASYNC=${PHASE2_USE_CP_ASYNC:-0}
PHASE2_CTA_BATCH=${PHASE2_CTA_BATCH:-4}
PHASE2_PERSISTENT_THREADS=${PHASE2_PERSISTENT_THREADS:-0}
PHASE2_DEBUG_ROUTING=${PHASE2_DEBUG_ROUTING:-0}
CP_STAGES=${CP_STAGES:-2}

DEFINES="-DPHASE2_USE_CP_ASYNC=${PHASE2_USE_CP_ASYNC}"
DEFINES="${DEFINES} -DPHASE2_CTA_BATCH=${PHASE2_CTA_BATCH}"
DEFINES="${DEFINES} -DPHASE2_PERSISTENT_THREADS=${PHASE2_PERSISTENT_THREADS}"
DEFINES="${DEFINES} -DPHASE2_DEBUG_ROUTING=${PHASE2_DEBUG_ROUTING}"
DEFINES="${DEFINES} -DCP_STAGES=${CP_STAGES}"

echo "Build configuration:"
echo "  CUDA Architecture: SM ${CUDA_ARCH}"
echo "  cp.async: ${PHASE2_USE_CP_ASYNC}"
echo "  CTA batch: ${PHASE2_CTA_BATCH}"
echo "  Persistent threads: ${PHASE2_PERSISTENT_THREADS}"
echo "  Debug routing: ${PHASE2_DEBUG_ROUTING}"
echo "  CP stages: ${CP_STAGES}"
echo ""

# Create bin directory
mkdir -p bin

# Compile shared objects
echo "[1/6] Compiling bitpack_utils.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/bitpack_utils.cu -o bin/bitpack_utils_bucket.o

echo "[2/6] Compiling encoder.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/encoder.cu -o bin/encoder.o

echo "[3/6] Compiling decompression_kernels_phase2_bucket.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/decompression_kernels_phase2_bucket.cu -o bin/decompression_kernels_phase2_bucket.o

echo "[4/6] Compiling decoder_warp_opt.cu (for L3_codec)..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/decoder_warp_opt.cu -o bin/decoder_warp_opt.o

echo "[5/6] Compiling L3_codec.cpp..."
${CXX} ${COMMON_FLAGS} ${DEFINES} -I${CUDA_PATH}/include \
    -c src/L3_codec.cpp -o bin/L3_codec.o

echo "[6/6] Compiling partition_bounds_kernel.cu..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    -c src/partition_bounds_kernel.cu -o bin/partition_bounds_kernel.o

echo "[7/7] Linking test_phase2_bucket..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/test_phase2_bucket.cpp \
    bin/L3_codec.o \
    bin/encoder.o \
    bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o \
    bin/bitpack_utils_bucket.o \
    -o bin/test_phase2_bucket \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/test_phase2_bucket"
echo ""
echo "Run with:"
echo "  ./bin/test_phase2_bucket"
echo ""
echo "To enable debug routing:"
echo "  PHASE2_DEBUG_ROUTING=1 bash scripts/build_phase2_bucket.sh"
echo ""
