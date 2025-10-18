#!/bin/bash

# L3 项目编译和运行脚本
# 此脚本会编译项目并运行示例程序

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  L3 GPU Compression Library${NC}"
echo -e "${BLUE}  Build and Run Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查CUDA是否可用
echo -e "${YELLOW}[1/6] Checking CUDA availability...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}Error: nvcc not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
echo ""

# 检查CMake版本
echo -e "${YELLOW}[2/6] Checking CMake availability...${NC}"
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}Error: cmake not found. Please install CMake 3.18+.${NC}"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | sed -n 's/.*version \([0-9.]*\).*/\1/p')
echo -e "${GREEN}✓ CMake found: version $CMAKE_VERSION${NC}"
echo ""

# 清理旧的构建目录
echo -e "${YELLOW}[3/6] Cleaning old build directory...${NC}"
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi
mkdir build
echo -e "${GREEN}✓ Build directory ready${NC}"
echo ""

# 配置CMake
echo -e "${YELLOW}[4/6] Configuring CMake...${NC}"
cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_EXAMPLES=ON \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DCMAKE_VERBOSE_MAKEFILE=OFF

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: CMake configuration failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ CMake configuration complete${NC}"
echo ""

# 编译项目
echo -e "${YELLOW}[5/6] Building project...${NC}"
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Build failed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# 运行示例
echo -e "${YELLOW}[6/6] Running example...${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Example: Partition Strategies${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ -f "./bin/example_partition_strategies" ]; then
    ./bin/example_partition_strategies
    EXIT_CODE=$?

    echo ""
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  ✓ All steps completed successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "Example binary location: ${BLUE}build/bin/example_partition_strategies${NC}"
        echo -e "Library location: ${BLUE}build/libl3_compression.a${NC}"
        echo ""
        echo -e "Next steps:"
        echo -e "  • Read ${BLUE}docs/GETTING_STARTED.md${NC} for more usage examples"
        echo -e "  • Check ${BLUE}docs/PARTITION_STRATEGIES.md${NC} for strategy selection guide"
        echo -e "  • Explore ${BLUE}examples/cpp/${NC} directory for more examples"
    else
        echo -e "${RED}========================================${NC}"
        echo -e "${RED}  Example execution failed${NC}"
        echo -e "${RED}========================================${NC}"
        exit $EXIT_CODE
    fi
else
    echo -e "${RED}Error: Example binary not found${NC}"
    exit 1
fi
