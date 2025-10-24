#!/bin/bash
# Simple build script for benchmark_real_datasets (all-in-one compilation)

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building GLECO Random Access Benchmark (Simple Mode)"
echo "════════════════════════════════════════════════════════════"

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"

COMMON_FLAGS="-std=c++17 -O3 -I./include -I./src"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

echo "Build configuration:"
echo "  CUDA Architecture: SM ${CUDA_ARCH}"
echo "  Optimization: -O3"
echo ""

mkdir -p bin

echo "Compiling benchmark_real_datasets (all-in-one)..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    tests/benchmark_real_datasets.cu \
    src/L3_codec.cpp \
    src/sosd_loader.cpp \
    src/encoder.cu \
    src/partition_bounds_kernel.cu \
    src/random_access_kernels.cu \
    -o bin/benchmark_real_datasets \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/benchmark_real_datasets"
echo ""
echo "Usage:"
echo "  ./bin/benchmark_real_datasets <dataset_path> <output_csv>"
echo ""
