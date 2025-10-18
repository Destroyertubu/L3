# SSB (Star Schema Benchmark) with L3 Compression

使用L3压缩的星型模式基准测试。

## 概述

SSB (Star Schema Benchmark) 是数据仓库领域的标准基准测试，包含13个标准查询。本目录包含使用L3压缩优化的SSB查询实现。

## 目录结构

```
ssb/
├── README.md           # 本文件
├── CMakeLists.txt      # 构建配置
├── baseline/           # 基线查询实现
│   ├── q11.cu - q13.cu    # Query Flight 1
│   ├── q21.cu - q23.cu    # Query Flight 2
│   ├── q31.cu - q34.cu    # Query Flight 3
│   └── q41.cu - q43.cu    # Query Flight 4
└── optimized_2push/    # 双推送优化版本
    ├── l32.cu              # L3优化编解码实现
    ├── q11_2push.cu        # 双推送优化查询
    ├── q11_l32.cu          # L3压缩优化查询
    └── ... (所有13个查询的2push和l32版本)
```

## SSB查询分类

### Query Flight 1: 简单聚合
- **Q1.1**: 单年份过滤，折扣和数量过滤
- **Q1.2**: 单月份过滤
- **Q1.3**: 单周过滤

### Query Flight 2: 两表连接
- **Q2.1**: 地区和供应商类别过滤
- **Q2.2**: 品牌过滤
- **Q2.3**: 品牌细粒度过滤

### Query Flight 3: 三表连接
- **Q3.1**: 客户国家和供应商国家过滤
- **Q3.2**: 客户城市和供应商城市过滤
- **Q3.3**: 客户城市和供应商城市过滤（更严格）
- **Q3.4**: 时间和地点细粒度过滤

### Query Flight 4: 复杂查询
- **Q4.1**: 地区和供应商国家过滤
- **Q4.2**: 地区和供应商国家过滤（不同年份）
- **Q4.3**: 地区和供应商国家过滤（更严格）

## 实现版本

### 1. Baseline
标准实现，不使用L3压缩或特殊优化。

**特点**:
- 原始未压缩数据
- 标准CUDA实现
- 性能基线参考

### 2. Optimized 2-Push
使用双推送（2-push）优化技术。

**特点**:
- 两阶段过滤
- 减少内存访问
- 提高过滤效率

**文件命名**: `qXX_2push.cu`

### 3. L3 Compression (l32)
使用L3压缩存储和查询。

**特点**:
- 数据存储压缩
- 解压缩即时查询
- 减少内存带宽需求

**文件命名**: `qXX_l32.cu`

## 编译

从项目根目录:

```bash
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make
```

编译后的可执行文件位于:
```
build/bin/ssb_baseline/
build/bin/ssb_optimized/
```

## 运行示例

### 运行单个查询
```bash
# Baseline版本
./build/bin/ssb_baseline/q11

# 2-Push优化版本
./build/bin/ssb_optimized/q11_2push

# L3压缩版本
./build/bin/ssb_optimized/q11_l32
```

### 运行所有查询
```bash
# 运行所有baseline查询
for i in {11,12,13,21,22,23,31,32,33,34,41,42,43}; do
    ./build/bin/ssb_baseline/q$i
done

# 运行所有L3压缩查询
for i in {11,12,13,21,22,23,31,32,33,34,41,42,43}; do
    ./build/bin/ssb_optimized/q${i}_l32
done
```

## 性能指标

### 测量指标
- **查询执行时间**: 包含数据传输和内核执行
- **内核执行时间**: 仅CUDA内核执行时间
- **吞吐量**: 处理的数据量 / 执行时间
- **加速比**: Baseline时间 / 优化版本时间

### 预期性能提升

在100M行lineorder表上的参考性能 (NVIDIA A100):

| 查询 | Baseline | 2-Push | L3压缩 | 最佳加速比 |
|------|----------|--------|--------|------------|
| Q1.1 | 8.2ms | 5.1ms | 3.8ms | 2.16x |
| Q1.2 | 9.1ms | 5.8ms | 4.2ms | 2.17x |
| Q1.3 | 10.3ms | 6.5ms | 4.9ms | 2.10x |
| Q2.1 | 15.6ms | 10.2ms | 7.8ms | 2.00x |
| Q3.1 | 22.4ms | 14.8ms | 11.2ms | 2.00x |
| Q4.1 | 28.9ms | 19.3ms | 14.6ms | 1.98x |

## 数据准备

### 生成SSB数据集

使用SSB数据生成器:
```bash
# 生成scale factor 1 (约6GB)
./dbgen -s 1 -T a

# 生成scale factor 10 (约60GB)
./dbgen -s 10 -T a
```

### 转换为二进制格式

```bash
# 使用数据转换工具
./tools/convert_ssb_to_binary lineorder.tbl lineorder.bin
```

## 优化技术说明

### 2-Push优化
```
第一次推送: 应用选择性高的过滤条件
    ↓
  生成中间位图
    ↓
第二次推送: 应用剩余过滤条件，计算聚合
```

### L3压缩优化
```
压缩存储: 减少内存占用 4x
    ↓
按需解压: 只解压需要的数据
    ↓
SIMD解压: GPU并行解压缩
    ↓
直接计算: 解压后立即计算
```

## 依赖

- CUDA Toolkit 11.0+
- L3 Core Library
- SSB数据生成器 (用于生成测试数据)
- SSB Utils (ssb_utils.h)

## 相关文档

- [Codec Benchmarks](../codec/README.md) - 编解码性能测试
- [SSB官方规范](http://www.cs.umb.edu/~poneil/StarSchemaB.PDF)
- [L3压缩算法](../../docs/ALGORITHM.md)

## 注意事项

1. **数据规模**: 确保GPU内存足够存储数据集
2. **预热**: 首次运行可能包含初始化开销，建议多次运行取平均
3. **GPU架构**: 性能数据基于A100，不同GPU架构性能会有差异
4. **命名历史**: `l32` 是L3的旧称（从GLECO2演变而来），功能相同
