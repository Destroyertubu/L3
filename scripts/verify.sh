#!/bin/bash
# L3 Verification Script
# Verifies project setup and readiness

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

ERRORS=0
WARNINGS=0

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

print_header "L3 Project Verification"

# ============================================================================
# Check Prerequisites
# ============================================================================
print_header "Checking Prerequisites"

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    print_success "CUDA Toolkit: $CUDA_VERSION"
else
    print_error "CUDA Toolkit not found"
    ((ERRORS++))
fi

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    print_success "CMake: $CMAKE_VERSION"
else
    print_error "CMake not found"
    ((ERRORS++))
fi

# Check GCC
if command -v g++ &> /dev/null; then
    GCC_VERSION=$(g++ --version | head -1 | awk '{print $NF}')
    print_success "G++: $GCC_VERSION"
else
    print_error "G++ not found"
    ((ERRORS++))
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    print_success "GPU: $GPU_NAME ($GPU_MEMORY)"
else
    print_error "nvidia-smi not found (GPU may not be available)"
    ((ERRORS++))
fi

echo ""

# ============================================================================
# Check Project Structure
# ============================================================================
print_header "Checking Project Structure"

check_dir() {
    if [ -d "$PROJECT_ROOT/$1" ]; then
        print_success "Directory: $1"
    else
        print_error "Missing directory: $1"
        ((ERRORS++))
    fi
}

check_file() {
    if [ -f "$PROJECT_ROOT/$1" ]; then
        print_success "File: $1"
    else
        print_error "Missing file: $1"
        ((ERRORS++))
    fi
}

check_dir "lib/l32"
check_dir "include/common"
check_dir "benchmarks/ssb/baseline"
check_dir "benchmarks/ssb/optimized_2push"
check_dir "scripts"
check_dir "docs"

check_file "CMakeLists.txt"
check_file "README.md"
check_file "lib/l32/l32.cu"
check_file "scripts/build.sh"

echo ""

# ============================================================================
# Check Source Files
# ============================================================================
print_header "Checking Source Files"

# Count source files
L3_FILES=$(find "$PROJECT_ROOT/lib/l32" -name "*.cu" | wc -l)
BASELINE_FILES=$(find "$PROJECT_ROOT/benchmarks/ssb/baseline" -name "*.cu" 2>/dev/null | wc -l)
OPTIMIZED_FILES=$(find "$PROJECT_ROOT/benchmarks/ssb/optimized_2push" -name "*.cu" 2>/dev/null | wc -l)
HEADER_FILES=$(find "$PROJECT_ROOT/include" -name "*.h" -o -name "*.hpp" -o -name "*.cuh" | wc -l)

print_success "L3 library files: $L3_FILES"
print_success "SSB baseline files: $BASELINE_FILES"
print_success "SSB optimized files: $OPTIMIZED_FILES"
print_success "Header files: $HEADER_FILES"

if [ $OPTIMIZED_FILES -lt 10 ]; then
    print_warning "Expected 13 SSB queries, found $OPTIMIZED_FILES"
    ((WARNINGS++))
fi

echo ""

# ============================================================================
# Check Build System
# ============================================================================
print_header "Checking Build System"

if [ -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    # Check CMakeLists.txt syntax
    if grep -q "cmake_minimum_required" "$PROJECT_ROOT/CMakeLists.txt"; then
        print_success "Root CMakeLists.txt is valid"
    else
        print_error "Root CMakeLists.txt may be invalid"
        ((ERRORS++))
    fi
fi

# Check subdirectory CMakeLists
if [ -f "$PROJECT_ROOT/lib/l32/CMakeLists.txt" ]; then
    print_success "L3 CMakeLists.txt exists"
else
    print_warning "L3 CMakeLists.txt missing"
    ((WARNINGS++))
fi

if [ -f "$PROJECT_ROOT/benchmarks/ssb/CMakeLists.txt" ]; then
    print_success "SSB CMakeLists.txt exists"
else
    print_warning "SSB CMakeLists.txt missing"
    ((WARNINGS++))
fi

echo ""

# ============================================================================
# Check Documentation
# ============================================================================
print_header "Checking Documentation"

check_doc() {
    if [ -f "$PROJECT_ROOT/docs/$1" ]; then
        print_success "Doc: $1"
    else
        print_warning "Missing doc: $1"
        ((WARNINGS++))
    fi
}

check_doc "INSTALLATION.md"
check_doc "MIGRATION.md"
check_doc "ARCHITECTURE.md"

echo ""

# ============================================================================
# Check Scripts
# ============================================================================
print_header "Checking Scripts"

check_script() {
    if [ -f "$PROJECT_ROOT/scripts/$1" ]; then
        if [ -x "$PROJECT_ROOT/scripts/$1" ]; then
            print_success "Script: $1 (executable)"
        else
            print_warning "Script: $1 (not executable)"
            ((WARNINGS++))
        fi
    else
        print_error "Missing script: $1"
        ((ERRORS++))
    fi
}

check_script "build.sh"
check_script "deploy.sh"
check_script "verify.sh"

echo ""

# ============================================================================
# Summary
# ============================================================================
print_header "Verification Summary"

echo ""
if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Project is ready. Next steps:"
    echo "  1. Build: ./scripts/build.sh"
    echo "  2. Test:  cd build/bin/ssb/optimized && ./q11_2push_opt"
    echo ""
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ Checks passed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Project is usable but some optional components are missing."
    echo "You can proceed with: ./scripts/build.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Verification failed with $ERRORS error(s) and $WARNINGS warning(s)${NC}"
    echo ""
    echo "Please resolve the errors above before building."
    echo ""
    exit 1
fi
