#!/bin/bash
# L3项目整合脚本 - 将tests/目录整合到新结构中
# 用法: bash scripts/integrate_tests.sh [stage]
#   stage: 1=基础, 2=扩展, 3=应用, all=全部

set -e

PROJECT_ROOT="/root/autodl-tmp/test/L3_opt"
cd "$PROJECT_ROOT"

STAGE=${1:-1}

echo "════════════════════════════════════════════════════════════"
echo "  L3 Tests Integration Script"
echo "════════════════════════════════════════════════════════════"
echo "Stage: $STAGE"
echo ""

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 阶段1: 基础整合
stage1_basic() {
    log_info "=== 阶段1: 基础整合 ==="

    # 创建新目录结构
    log_info "创建目录结构..."
    mkdir -p tests/{unit,integration,performance,debug}
    mkdir -p docs/guides

    # 移动基础测试
    if [ -f "tests/test_roundtrip.cpp" ]; then
        log_info "移动单元测试..."
        mv tests/test_roundtrip.cpp tests/unit/ 2>/dev/null || true
        mv tests/test_vectors.cpp tests/unit/ 2>/dev/null || true
        log_success "单元测试已移动"
    fi

    # 移动调试工具
    if [ -f "tests/debug_model_prediction.cpp" ]; then
        log_info "移动调试工具..."
        mv tests/debug_*.cpp tests/debug/ 2>/dev/null || true
        log_success "调试工具已移动"
    fi

    # 移动性能测试
    if [ -f "tests/test_real_data_performance.cpp" ]; then
        log_info "移动性能测试..."
        mv tests/test_real_data_performance.cpp tests/performance/ 2>/dev/null || true
        mv tests/benchmark_real_datasets.cu tests/performance/ 2>/dev/null || true
        log_success "性能测试已移动"
    fi

    log_success "阶段1完成"
}

# 阶段2: 功能扩展整合
stage2_extensions() {
    log_info "=== 阶段2: 功能扩展整合 ==="

    # 创建扩展目录
    mkdir -p extensions/{random_access/{kernels,include,examples},L3_v2}

    # Random Access
    log_info "整合Random Access功能..."
    if [ -f "tests/benchmark_random_access.cu" ]; then
        cp tests/benchmark_random_access.cu extensions/random_access/examples/
        cp tests/example_random_access.cu extensions/random_access/examples/ 2>/dev/null || true
    fi

    if [ -f "tests/ssb_base/L3_random_access.cuh" ]; then
        cp tests/ssb_base/L3_random_access.cuh extensions/random_access/include/
    fi

    if [ -f "tests/ssb_new/L3_ra_utils.cuh" ]; then
        cp tests/ssb_new/L3_ra_utils.cuh extensions/random_access/include/
        cp tests/ssb_new/L3_alex_index.cuh extensions/random_access/include/ 2>/dev/null || true
        cp tests/ssb_new/L3_predicate_pushdown.cuh extensions/random_access/include/ 2>/dev/null || true
    fi

    log_success "Random Access已整合"

    # L3 v2
    log_info "整合L3 v2..."
    if [ -f "tests/L32.cu" ]; then
        cp tests/L32.cu extensions/L3_v2/
        log_success "L3 v2已整合"
    fi

    # 创建README
    cat > extensions/random_access/README.md << 'RAEOF'
# L3 Random Access Extension

随机访问功能扩展，支持在压缩数据上进行高效的随机查询。

## 功能特性
- GPU加速的随机访问
- 学习索引优化（ALEX-inspired）
- 谓词下推支持

## 目录结构
- `include/` - 头文件
- `kernels/` - CUDA内核实现
- `examples/` - 使用示例

## 使用方法
参见 `examples/` 目录中的示例代码。
RAEOF

    cat > extensions/L3_v2/README.md << 'V2EOF'
# L3 Version 2

L3压缩算法的改进版本，优化了压缩比和性能。

## 主要改进
- 结构体数组（SoA）优化
- 改进的模型拟合
- 更好的分区策略

## 文件
- `L32.cu` - 完整实现（约14万行）

## 使用注意
这是实验性版本，建议先在小数据集上测试。
V2EOF

    log_success "阶段2完成"
}

# 阶段3: 应用整合
stage3_applications() {
    log_info "=== 阶段3: 应用整合 ==="

    # 创建应用目录
    mkdir -p applications/{ssb/{baseline,optimized,random_access,include,scripts},sosd/benchmarks}

    # SSB baseline
    log_info "整合SSB基准测试..."
    if [ -d "tests/ssb_base" ]; then
        cp tests/ssb_base/q*.cu applications/ssb/baseline/ 2>/dev/null || true
    fi

    if [ -d "tests/ssb_new/baseline" ]; then
        cp tests/ssb_new/baseline/*.cu applications/ssb/baseline/ 2>/dev/null || true
    fi

    # SSB optimized
    if [ -d "tests/ssb_new/optimized_2push" ]; then
        cp -r tests/ssb_new/optimized_2push/* applications/ssb/optimized/ 2>/dev/null || true
    fi

    # SSB random access
    if [ -d "tests/ssb_ra" ]; then
        cp tests/ssb_ra/*.cu applications/ssb/random_access/ 2>/dev/null || true
    fi

    # SSB工具文件
    if [ -f "tests/ssb_new/ssb_L3_utils.cuh" ]; then
        cp tests/ssb_new/*.cuh applications/ssb/include/ 2>/dev/null || true
        cp tests/ssb_new/*.h applications/ssb/include/ 2>/dev/null || true
    fi

    # SSB脚本
    if [ -f "tests/ssb_new/generate_all_2push.py" ]; then
        cp tests/ssb_new/*.py applications/ssb/scripts/ 2>/dev/null || true
        cp tests/ssb_new/*.sh applications/ssb/scripts/ 2>/dev/null || true
        cp tests/ssb_new/Makefile applications/ssb/ 2>/dev/null || true
        chmod +x applications/ssb/scripts/*.sh 2>/dev/null || true
    fi

    log_success "SSB已整合"

    # SOSD
    log_info "整合SOSD基准测试..."
    if [ -f "tests/sosd_benchmark.cpp" ]; then
        cp tests/sosd_benchmark.cpp applications/sosd/benchmarks/
        log_success "SOSD已整合"
    fi

    # 创建SSB README
    cat > applications/ssb/README.md << 'SSBEOF'
# Star Schema Benchmark (SSB) for L3

SSB是标准的数据库分析性能基准测试。

## 版本说明
- `baseline/` - 基础实现（13个查询）
- `optimized/` - 优化版本（谓词下推）
- `random_access/` - 随机访问版本

## 查询列表
- Q1.1-Q1.3: 简单聚合
- Q2.1-Q2.3: 两表连接
- Q3.1-Q3.4: 三表连接
- Q4.1-Q4.3: 多表连接

## 运行方法
```bash
cd applications/ssb
make
./scripts/run_performance_comparison.sh
```

## 性能对比
参见 `scripts/` 目录中的性能测试脚本。
SSBEOF

    log_success "阶段3完成"
}

# 清理和整理
cleanup() {
    log_info "=== 清理工作 ==="

    # 移动工具头文件到include/
    if [ -f "tests/ssb_utils_L3.h" ]; then
        cp tests/ssb_utils_L3.h include/ 2>/dev/null || true
    fi

    # 移动内核优化到src/
    if [ -f "tests/createPartitionsFast_optimized.cuh" ]; then
        log_info "移动内核优化文件..."
        cp tests/createPartitionsFast_optimized.cuh src/kernels/utils/ 2>/dev/null || true
        cp tests/partition_optimization_patch.cu src/kernels/utils/ 2>/dev/null || true
    fi

    log_success "清理完成"
}

# 生成整合报告
generate_report() {
    log_info "=== 生成整合报告 ==="

    cat > INTEGRATION_REPORT.md << 'EOF'
# L3项目整合报告

## 执行日期
EOF
    echo "$(date)" >> INTEGRATION_REPORT.md
    cat >> INTEGRATION_REPORT.md << 'EOF'

## 整合结果

### 新目录结构
```
project_root/
├── src/                  # 核心源代码
├── extensions/          # 功能扩展
│   ├── random_access/
│   └── L3_v2/
├── applications/        # 应用示例
│   ├── ssb/
│   └── sosd/
├── tests/               # 测试
│   ├── unit/
│   ├── performance/
│   └── debug/
└── docs/                # 文档
```

### 文件统计
EOF

    echo "- 单元测试: $(find tests/unit -type f 2>/dev/null | wc -l) 个文件" >> INTEGRATION_REPORT.md
    echo "- 性能测试: $(find tests/performance -type f 2>/dev/null | wc -l) 个文件" >> INTEGRATION_REPORT.md
    echo "- 调试工具: $(find tests/debug -type f 2>/dev/null | wc -l) 个文件" >> INTEGRATION_REPORT.md
    echo "- Random Access: $(find extensions/random_access -type f 2>/dev/null | wc -l) 个文件" >> INTEGRATION_REPORT.md
    echo "- SSB查询: $(find applications/ssb -name "*.cu" 2>/dev/null | wc -l) 个文件" >> INTEGRATION_REPORT.md

    cat >> INTEGRATION_REPORT.md << 'EOF'

### 下一步
1. 验证所有文件都已正确移动
2. 更新CMakeLists.txt
3. 测试编译
4. 更新文档

### 原tests/目录
原始tests/目录已保留，可以安全删除或归档到archive/。

---
*此报告由integrate_tests.sh自动生成*
EOF

    log_success "报告已生成: INTEGRATION_REPORT.md"
}

# 主执行逻辑
main() {
    case $STAGE in
        1)
            stage1_basic
            ;;
        2)
            stage1_basic
            stage2_extensions
            ;;
        3|all)
            stage1_basic
            stage2_extensions
            stage3_applications
            cleanup
            ;;
        *)
            echo "用法: $0 [1|2|3|all]"
            echo "  1 - 仅基础整合"
            echo "  2 - 基础+扩展整合"
            echo "  3 - 完整整合"
            echo "  all - 等同于3"
            exit 1
            ;;
    esac

    generate_report

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo -e "${GREEN}✅ 整合完成！${NC}"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "查看详细报告: cat INTEGRATION_REPORT.md"
    echo "查看整合方案: cat INTEGRATION_PLAN.md"
    echo ""
}

main
