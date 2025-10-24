#!/bin/bash
#
# GLECO Phase 2.2 核心文件打包脚本
# 用于提供给其他开发者进行进一步优化
#

set -e

PACK_DIR="L3_phase2_2_core"
ARCHIVE_NAME="L3_phase2_2_core_$(date +%Y%m%d).tar.gz"

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║         GLECO Phase 2.2 核心文件打包                             ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# 创建临时目录
rm -rf ${PACK_DIR}
mkdir -p ${PACK_DIR}/{src,include,scripts,data,docs}

echo "[1/8] 复制核心实现文件..."
cp src/bitpack_utils.cuh ${PACK_DIR}/src/
cp src/decompression_kernels_phase2_bucket.cu ${PACK_DIR}/src/

echo "[2/8] 复制接口和格式定义..."
cp include/L3_codec.hpp ${PACK_DIR}/include/
cp include/L3_format.hpp ${PACK_DIR}/include/

echo "[3/8] 复制压缩端实现..."
cp src/encoder.cu ${PACK_DIR}/src/
cp src/L3_codec.cpp ${PACK_DIR}/src/

echo "[4/8] 复制测试程序..."
cp src/test_phase2_bucket.cpp ${PACK_DIR}/src/
cp src/test_fb_200M_phase2_bucket.cpp ${PACK_DIR}/src/

echo "[5/8] 复制构建脚本..."
cp scripts/build_phase2_bucket.sh ${PACK_DIR}/scripts/
cp scripts/build_fb_200M_test.sh ${PACK_DIR}/scripts/

echo "[6/8] 复制辅助文件..."
cp src/decoder_warp_opt.cu ${PACK_DIR}/src/ 2>/dev/null || echo "  (decoder_warp_opt.cu 不存在，跳过)"
cp src/partition_bounds_kernel.cu ${PACK_DIR}/src/ 2>/dev/null || echo "  (partition_bounds_kernel.cu 不存在，跳过)"
cp src/timers.cu ${PACK_DIR}/src/ 2>/dev/null || echo "  (timers.cu 不存在，跳过)"

echo "[7/8] 复制文档..."
cp README.md ${PACK_DIR}/
cp OPTIMIZATION_GUIDE.md ${PACK_DIR}/docs/

# 创建数据集说明
cat > ${PACK_DIR}/data/README.md << 'EOF'
# 测试数据集

## Facebook 200M Dataset

**文件**: `fb_200M_uint64.bin`
**大小**: 1.5 GB (200M × 8 bytes)
**类型**: uint64_t 数组（已排序）
**来源**: Facebook SOSD benchmark

### 下载方式

如果您没有此数据集，可以：

1. 使用 SOSD benchmark 生成：
   https://github.com/learnedsystems/SOSD

2. 或从原始服务器获取：
   /root/autodl-tmp/test/data/fb_200M_uint64.bin

3. 或使用合成数据测试（`test_phase2_bucket` 不需要外部数据）

### 使用方法

```bash
# 运行 FB 200M 测试
./bin/test_fb_200M_phase2_bucket

# 确保数据文件路径正确（在 test_fb_200M_phase2_bucket.cpp 中修改）
const char* filename = "data/fb_200M_uint64.bin";
```
EOF

echo "[8/8] 创建打包说明..."
cat > ${PACK_DIR}/README_PACK.txt << 'EOF'
╔═══════════════════════════════════════════════════════════════════╗
║              GLECO Phase 2.2 核心文件包                          ║
║                  用于进一步优化和调整                             ║
╚═══════════════════════════════════════════════════════════════════╝

包含内容:
═══════════════════════════════════════════════════════════════════

必需文件（6个）:
  src/bitpack_utils.cuh                       (64-bit 位提取器)
  src/decompression_kernels_phase2_bucket.cu  (桶调度器 + 核心)
  include/L3_codec.hpp                     (API 接口)
  include/L3_format.hpp                    (数据结构定义)
  src/encoder.cu                              (压缩端实现)
  src/L3_codec.cpp                         (Host 端逻辑)

测试文件（2个）:
  src/test_phase2_bucket.cpp                  (综合测试套件)
  src/test_fb_200M_phase2_bucket.cpp          (真实数据集测试)

构建脚本（2个）:
  scripts/build_phase2_bucket.sh              (构建脚本)
  scripts/build_fb_200M_test.sh               (FB 200M 构建)

文档（2个）:
  README.md                                   (项目进展总结)
  docs/OPTIMIZATION_GUIDE.md                  (优化指导)

快速开始:
═══════════════════════════════════════════════════════════════════

1. 解压文件包
   tar -xzf L3_phase2_2_core_YYYYMMDD.tar.gz
   cd L3_phase2_2_core

2. 阅读文档
   cat README.md
   cat docs/OPTIMIZATION_GUIDE.md

3. 构建测试
   bash scripts/build_phase2_bucket.sh

4. 运行验证
   ./bin/test_phase2_bucket

5. 性能基准（需要 fb_200M_uint64.bin）
   bash scripts/build_fb_200M_test.sh
   ./bin/test_fb_200M_phase2_bucket

当前性能基准:
═══════════════════════════════════════════════════════════════════

测试数据集              吞吐量 (中位数)    状态
─────────────────────────────────────────────────
1M 综合测试             141.00 GB/s      ✅ PASS
8M 综合测试             543.92 GB/s      ✅ PASS
64M 综合测试            930.94 GB/s      ✅ PASS
200M Facebook 数据集    1023.71 GB/s     ✅ PASS (peak: 1025.75)

优化目标:
═══════════════════════════════════════════════════════════════════

短期（+5-15%）:
  - 启用 cp.async 流水线
  - Stream 并发
  - 向量化写回

中期（+20-30%）:
  - 通用核特化 (19-23 bit)
  - Device-side 分桶
  - 符号扩展优化

详见: docs/OPTIMIZATION_GUIDE.md

支持:
═══════════════════════════════════════════════════════════════════

问题反馈: (您的联系方式)
文档版本: Phase 2.2
最后更新: 2025-10-21
平台要求: CUDA 12.x, SM 80+ (推荐 SM 90 / Hopper)

EOF

# 打包
echo ""
echo "正在打包..."
tar -czf ${ARCHIVE_NAME} ${PACK_DIR}/

# 统计
echo ""
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                        打包完成                                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""
echo "文件包: ${ARCHIVE_NAME}"
echo "大小: $(du -h ${ARCHIVE_NAME} | cut -f1)"
echo ""
echo "内容统计:"
find ${PACK_DIR} -type f | while read f; do
    echo "  $(basename $f) ($(wc -l < $f 2>/dev/null || echo '?') lines)"
done | sort
echo ""
echo "使用方法:"
echo "  tar -xzf ${ARCHIVE_NAME}"
echo "  cd ${PACK_DIR}"
echo "  cat README.md"
echo ""

# 可选：清理临时目录
read -p "是否删除临时目录 ${PACK_DIR}? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf ${PACK_DIR}
    echo "已清理临时目录"
fi
