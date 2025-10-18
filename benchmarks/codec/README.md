# L3 Codec Performance Benchmarks

编解码器性能基准测试程序。

## 概述

这些benchmarks专门用于测试L3压缩算法的编解码性能，包括压缩率、吞吐量、延迟等核心指标。

## 测试程序

### 1. benchmark_kernel_only.cpp
**仅内核性能测试**

- 测试纯CUDA内核的编解码性能
- 不包含数据传输开销
- 适用于评估算法本身的效率

**测试指标**:
- 编码吞吐量 (GB/s)
- 解码吞吐量 (GB/s)
- 内核执行时间

### 2. benchmark_optimized.cpp
**优化版本性能测试**

- 测试优化后的编解码实现
- 包含各种优化技术的对比
- 评估优化效果

**测试内容**:
- Warp级优化
- 共享内存优化
- 流水线优化

### 3. main_bench.cpp
**主基准测试程序**

- 完整的端到端性能测试
- 包含数据传输、编解码全流程
- 使用合成数据集

**测试场景**:
- 不同数据规模 (1M, 10M, 100M entries)
- 不同数据分布 (均匀、正态、偏斜)
- 不同压缩配置

### 4. sosd_bench_demo.cpp
**SOSD数据集基准测试**

- 使用SOSD标准数据集
- 真实数据性能评估
- 可重现的测试结果

**SOSD数据集**:
- books (200M keys)
- fb (200M keys)
- osm (800M keys)
- wiki (200M keys)

## 编译

从项目根目录:

```bash
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make benchmark_kernel_only
make benchmark_optimized
make main_bench
make sosd_bench_demo
```

## 运行示例

### 基本性能测试
```bash
./build/bin/main_bench
```

### 仅内核测试
```bash
./build/bin/benchmark_kernel_only
```

### SOSD数据集测试
```bash
# 需要先下载SOSD数据集
./build/bin/sosd_bench_demo --dataset books
```

## 性能指标

### 压缩性能
- **压缩率**: 压缩后大小 / 原始大小
- **编码吞吐量**: 原始数据量 / 编码时间
- **解码吞吐量**: 原始数据量 / 解码时间

### 查询性能
- **随机访问延迟**: 单次查询平均时间
- **范围查询吞吐量**: 查询结果数量 / 查询时间

## 输出格式

典型输出:
```
L3 Codec Benchmark Results
==========================
Dataset Size: 100M int32
Original Size: 381.47 MB
Compressed Size: 95.37 MB
Compression Ratio: 4.00x

Encoding Time: 12.34 ms
Encoding Throughput: 30.92 GB/s

Decoding Time: 8.76 ms
Decoding Throughput: 43.54 GB/s
```

## 性能基线

在NVIDIA A100 GPU上的参考性能:

| 测试 | 编码吞吐量 | 解码吞吐量 | 压缩率 |
|------|------------|------------|--------|
| 均匀分布 | 28-32 GB/s | 40-45 GB/s | 3.5-4.0x |
| 正态分布 | 26-30 GB/s | 38-42 GB/s | 3.8-4.2x |
| SOSD books | 24-28 GB/s | 35-40 GB/s | 4.0-4.5x |

## 依赖

- CUDA Toolkit 11.0+
- L3 Core Library
- SOSD数据集 (用于sosd_bench_demo)

## 相关文档

- [SSB查询benchmarks](../ssb/README.md) - 数据库查询场景测试
- [Random Access benchmarks](../random_access/README.md) - 随机访问测试
- [L3 API文档](../../docs/API.md)
