#!/bin/bash

# Build script for GLECO optimized decompression

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "==================================="
echo "Building GLECO Optimized Project"
echo "==================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Build directory: ${BUILD_DIR}"
echo ""

# Clean old build (optional)
if [ "$1" == "clean" ]; then
    echo "Cleaning old build..."
    rm -rf "${BUILD_DIR}"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Please ensure CUDA is installed."
    exit 1
fi

echo "CUDA version:"
nvcc --version | grep "release"
echo ""

# Configure with CMake
cd "${BUILD_DIR}"
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="70;75;80;86" \
    || { echo "CMake configuration failed"; exit 1; }

echo ""
echo "Building..."
make -j$(nproc) || { echo "Build failed"; exit 1; }

echo ""
echo "==================================="
echo "Build completed successfully!"
echo "==================================="
echo "Executables:"
echo "  - ${BUILD_DIR}/bin/L3_bench"
echo "  - ${BUILD_DIR}/bin/L3_test"
echo ""
