#!/bin/bash
# Build script for benchmark_real_datasets

set -e

echo "════════════════════════════════════════════════════════════"
echo "  Building GLECO Random Access Benchmark for Real Datasets"
echo "════════════════════════════════════════════════════════════"

CUDA_ARCH="90"
CUDA_PATH="/usr/local/cuda"
NVCC="${CUDA_PATH}/bin/nvcc"
CXX="g++"

COMMON_FLAGS="-std=c++17 -O3 -I./include"
CUDA_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}"
CUDA_FLAGS="${CUDA_FLAGS} --expt-relaxed-constexpr"
CUDA_FLAGS="${CUDA_FLAGS} -Xcompiler -fPIC"

echo "Build configuration:"
echo "  CUDA Architecture: SM ${CUDA_ARCH}"
echo "  Optimization: -O3"
echo ""

mkdir -p bin

# Compile dependencies (reuse if already compiled)
if [ ! -f bin/bitpack_utils_bench.o ]; then
    echo "[1/7] Compiling bitpack_utils.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/bitpack_utils.cu -o bin/bitpack_utils_bench.o
fi

if [ ! -f bin/encoder_bench.o ]; then
    echo "[2/7] Compiling encoder.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/encoder.cu -o bin/encoder_bench.o
fi

if [ ! -f bin/decoder_warp_opt_bench.o ]; then
    echo "[3/7] Compiling decoder_warp_opt.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/decoder_warp_opt.cu -o bin/decoder_warp_opt_bench.o
fi

if [ ! -f bin/partition_bounds_bench.o ]; then
    echo "[4/7] Compiling partition_bounds_kernel.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/partition_bounds_kernel.cu -o bin/partition_bounds_bench.o
fi

if [ ! -f bin/random_access_bench.o ]; then
    echo "[5/7] Compiling random_access_kernels.cu..."
    ${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
        -c src/random_access_kernels.cu -o bin/random_access_bench.o
fi

echo "[6/7] Compiling L3_codec.cpp..."
${CXX} ${COMMON_FLAGS} -I${CUDA_PATH}/include \
    -c src/L3_codec.cpp -o bin/L3_codec_bench.o

echo "[7/7] Compiling sosd_loader.cpp..."
${CXX} ${COMMON_FLAGS} -I${CUDA_PATH}/include \
    -c src/sosd_loader.cpp -o bin/sosd_loader_bench.o

echo "[8/8] Linking benchmark_real_datasets..."
${NVCC} ${COMMON_FLAGS} ${CUDA_FLAGS} \
    tests/benchmark_real_datasets.cu \
    bin/L3_codec_bench.o \
    bin/sosd_loader_bench.o \
    bin/encoder_bench.o \
    bin/decoder_warp_opt_bench.o \
    bin/partition_bounds_bench.o \
    bin/random_access_bench.o \
    bin/bitpack_utils_bench.o \
    -o bin/benchmark_real_datasets \
    -L${CUDA_PATH}/lib64 -lcudart

echo ""
echo "✅ Build complete!"
echo "   Executable: bin/benchmark_real_datasets"
echo ""
echo "Usage:"
echo "  ./bin/benchmark_real_datasets <dataset_path> <output_csv>"
echo ""
echo "Example:"
echo "  ./bin/benchmark_real_datasets /root/autodl-tmp/test/data/books_200M_uint32.bin results.csv"
echo ""
