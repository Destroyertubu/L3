#!/bin/bash
# L3 Build Script
# Usage: ./scripts/build.sh [options]

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================
BUILD_TYPE="Release"
CUDA_ARCH="auto"
NUM_JOBS=$(nproc)
BUILD_BENCHMARKS="ON"
BUILD_EXAMPLES="ON"
USE_L3="ON"
CLEAN_BUILD=0

# ============================================================================
# Colors for output
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# Helper Functions
# ============================================================================
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Detect CUDA Architecture
# ============================================================================
detect_cuda_arch() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_info "Detected GPU: $gpu_name"

        # Try to get compute capability
        if command -v nvidia-smi &> /dev/null; then
            local compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
            if [ ! -z "$compute_cap" ]; then
                print_info "Detected Compute Capability: $compute_cap"
                echo "$compute_cap"
                return
            fi
        fi
    fi

    # Default fallback
    print_warning "Could not detect GPU, using default architectures: 75;80;86"
    echo "75;80;86"
}

# ============================================================================
# Check Dependencies
# ============================================================================
check_dependencies() {
    print_header "Checking Dependencies"

    # Check CMake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake not found. Please install CMake 3.18+"
        exit 1
    fi
    local cmake_version=$(cmake --version | head -1 | awk '{print $3}')
    print_info "CMake version: $cmake_version"

    # Check NVCC
    if ! command -v nvcc &> /dev/null; then
        print_error "NVCC not found. Please install CUDA Toolkit"
        exit 1
    fi
    local cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    print_info "CUDA version: $cuda_version"

    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    else
        print_warning "nvidia-smi not found. Cannot verify GPU"
    fi

    echo ""
}

# ============================================================================
# Parse Arguments
# ============================================================================
show_help() {
    cat << EOF
L3 Build Script

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -t, --type TYPE         Build type: Release, Debug, RelWithDebInfo (default: Release)
    -a, --arch ARCH         CUDA architecture (default: auto-detect)
    -j, --jobs NUM          Number of parallel jobs (default: $(nproc))
    -c, --clean             Clean build directory before building
    --no-benchmarks         Don't build benchmarks
    --no-examples           Don't build examples
    --legacy                Build legacy L3 version

Examples:
    $0                                  # Default build
    $0 -t Debug -j 4                    # Debug build with 4 jobs
    $0 -a 86 -c                         # Clean build for RTX 30xx
    $0 --no-benchmarks --no-examples    # Build library only

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -a|--arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=1
            shift
            ;;
        --no-benchmarks)
            BUILD_BENCHMARKS="OFF"
            shift
            ;;
        --no-examples)
            BUILD_EXAMPLES="OFF"
            shift
            ;;
        --legacy)
            USE_L3="OFF"
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Build Process
# ============================================================================
print_header "L3 Build Configuration"

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Check dependencies
check_dependencies

# Auto-detect CUDA architecture if needed
if [ "$CUDA_ARCH" = "auto" ]; then
    CUDA_ARCH=$(detect_cuda_arch)
fi

# Print configuration
print_info "Build Type: $BUILD_TYPE"
print_info "CUDA Architectures: $CUDA_ARCH"
print_info "Parallel Jobs: $NUM_JOBS"
print_info "Build Benchmarks: $BUILD_BENCHMARKS"
print_info "Build Examples: $BUILD_EXAMPLES"
print_info "Use L3: $USE_L3"
echo ""

# Create/clean build directory
if [ $CLEAN_BUILD -eq 1 ] && [ -d "build" ]; then
    print_info "Cleaning build directory..."
    rm -rf build
fi

mkdir -p build
cd build

# Configure with CMake
print_header "Configuring with CMake"
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
    -DBUILD_BENCHMARKS=$BUILD_BENCHMARKS \
    -DBUILD_EXAMPLES=$BUILD_EXAMPLES \
    -DUSE_L3=$USE_L3 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed"
    exit 1
fi

echo ""

# Build
print_header "Building L3"
make -j$NUM_JOBS

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

echo ""

# Summary
print_header "Build Summary"
print_info "Build completed successfully!"
echo ""
print_info "Binaries location:"
echo "  - Libraries: $PROJECT_ROOT/build/lib/"
echo "  - Executables: $PROJECT_ROOT/build/bin/"
echo ""

if [ "$BUILD_BENCHMARKS" = "ON" ]; then
    print_info "SSB Benchmarks:"
    echo "  - Baseline: $PROJECT_ROOT/build/bin/ssb/baseline/"
    echo "  - Optimized: $PROJECT_ROOT/build/bin/ssb/optimized/"
    echo ""
fi

print_info "Quick Test:"
echo "  cd build/bin/ssb/optimized"
echo "  ./q11_2push"
echo ""

print_header "Build Complete!"
