#!/bin/bash
# 快速编译脚本 - 不清理旧构建

cd build 2>/dev/null || mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)
