#!/bin/bash
# Build script for SSB (Star Schema Benchmark) Query Tests

set -e  # Exit on error

echo "======================================================================="
echo "Building GLECO SSB Query Tests"
echo "======================================================================="

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Build directory
BUILD_DIR="$PROJECT_ROOT/build"
BIN_DIR="$PROJECT_ROOT/bin"

# Create build directory
if [ ! -d "$BUILD_DIR" ]; then
    echo "Creating build directory: $BUILD_DIR"
    mkdir -p "$BUILD_DIR"
fi

# Configure CMake
echo ""
echo "Configuring CMake..."
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DPHASE2_CTA_BATCH=4 \
    -DPHASE2_USE_CP_ASYNC=OFF \
    -DPHASE2_PERSISTENT_THREADS=OFF \
    -DPHASE2_DEBUG_ROUTING=OFF \
    -DPHASE2_DEBUG_VECTORIZATION=OFF

# Build
echo ""
echo "Building test_ssb_queries..."
make test_ssb_queries -j$(nproc)

# Check if binary was created
if [ -f "$BUILD_DIR/test_ssb_queries" ]; then
    echo ""
    echo "✅ Build successful!"
    echo "Binary location: $BUILD_DIR/test_ssb_queries"

    # Create symlink in project bin directory
    mkdir -p "$BIN_DIR"
    ln -sf "$BUILD_DIR/test_ssb_queries" "$BIN_DIR/test_ssb_queries"
    echo "Symlink created: $BIN_DIR/test_ssb_queries"

    echo ""
    echo "To run the tests:"
    echo "  cd $PROJECT_ROOT"
    echo "  ./bin/test_ssb_queries"
else
    echo ""
    echo "❌ Build failed! Binary not found."
    exit 1
fi

echo ""
echo "======================================================================="
echo "Build Complete!"
echo "======================================================================="
