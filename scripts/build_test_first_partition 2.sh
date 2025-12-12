#!/bin/bash
set -e
CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} --expt-relaxed-constexpr -Xcompiler -fPIC"
DEFINES="-DPHASE2_USE_CP_ASYNC=0 -DPHASE2_CTA_BATCH=4"
mkdir -p bin
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} ${DEFINES} \
    src/opt/phase2_large_partition/test_first_partition_only.cpp \
    bin/L3_codec.o bin/encoder.o bin/decoder_warp_opt.o \
    bin/decompression_kernels_phase2_bucket.o \
    bin/partition_bounds_kernel.o bin/bitpack_utils_bucket.o \
    -o bin/test_first_partition_only \
    -L${CUDA_PATH}/lib64 -lcudart
echo "✅ Build complete: bin/test_first_partition_only"
