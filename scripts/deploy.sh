#!/bin/bash
# L3 Deployment Script
# Package the project for deployment to another machine

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Generate version tag
VERSION=$(date +%Y%m%d_%H%M%S)
PACKAGE_NAME="L3_${VERSION}"
OUTPUT_DIR="${PROJECT_ROOT}/../${PACKAGE_NAME}"

print_header "L3 Deployment Package Creator"

cd "$PROJECT_ROOT"

print_info "Creating deployment package: $PACKAGE_NAME"

# Create package directory
mkdir -p "$OUTPUT_DIR"

# Copy essential files
print_info "Copying source files..."
cp -r lib "$OUTPUT_DIR/"
cp -r include "$OUTPUT_DIR/"
cp -r benchmarks "$OUTPUT_DIR/"
cp -r scripts "$OUTPUT_DIR/"
cp -r docs "$OUTPUT_DIR/"

# Copy configuration files
cp CMakeLists.txt "$OUTPUT_DIR/"
cp README.md "$OUTPUT_DIR/"

# Create examples and tools directories
mkdir -p "$OUTPUT_DIR/examples"
mkdir -p "$OUTPUT_DIR/tools"
mkdir -p "$OUTPUT_DIR/data"

# Create deployment README
cat > "$OUTPUT_DIR/DEPLOY_README.md" << 'EOF'
# L3 Deployment Package

This package contains everything needed to build and run L3 on a new machine.

## Quick Start

1. **Install prerequisites:**
   ```bash
   # CUDA Toolkit 11.0+
   # CMake 3.18+
   # GCC 9+ or compatible C++ compiler
   ```

2. **Build:**
   ```bash
   cd L3_*
   ./scripts/build.sh
   ```

3. **Test:**
   ```bash
   cd build/bin/ssb/optimized
   ./q11_2push
   ```

## Full Documentation

See `docs/INSTALLATION.md` for complete installation instructions.
See `README.md` for project overview and usage.

## Package Contents

- `lib/` - Core library source code
- `include/` - Header files
- `benchmarks/` - Benchmark programs
- `scripts/` - Build and utility scripts
- `docs/` - Documentation
- `CMakeLists.txt` - Build configuration

## System Requirements

- NVIDIA GPU with Compute Capability 7.5+
- CUDA Toolkit 11.0+
- CMake 3.18+
- Linux OS (Ubuntu 20.04+ recommended)

## Support

For issues and questions, refer to the documentation in `docs/` directory.
EOF

# Create archive
print_info "Creating archive..."
cd "$(dirname "$OUTPUT_DIR")"
tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME"

# Calculate checksum
print_info "Calculating checksum..."
sha256sum "${PACKAGE_NAME}.tar.gz" > "${PACKAGE_NAME}.tar.gz.sha256"

# Cleanup temporary directory
rm -rf "$OUTPUT_DIR"

# Summary
print_header "Deployment Package Created"
echo ""
print_info "Package: ${PACKAGE_NAME}.tar.gz"
print_info "Location: $(dirname "$OUTPUT_DIR")/${PACKAGE_NAME}.tar.gz"
print_info "Size: $(du -h "$(dirname "$OUTPUT_DIR")/${PACKAGE_NAME}.tar.gz" | cut -f1)"
echo ""
print_info "Checksum saved to: ${PACKAGE_NAME}.tar.gz.sha256"
echo ""
print_info "To deploy to another machine:"
echo "  1. Copy ${PACKAGE_NAME}.tar.gz to target machine"
echo "  2. Verify checksum: sha256sum -c ${PACKAGE_NAME}.tar.gz.sha256"
echo "  3. Extract: tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  4. Build: cd ${PACKAGE_NAME} && ./scripts/build.sh"
echo ""

print_header "Complete!"
