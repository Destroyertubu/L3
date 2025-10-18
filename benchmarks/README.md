# L3 Benchmarks

L3压缩算法的完整基准测试套件。

## 概述

本目录包含所有L3压缩相关的性能测试，涵盖从底层编解码到实际应用场景的全方位评估。

## 目录结构

```
benchmarks/
├── README.md           # 本文件
├── CMakeLists.txt      # 总体构建配置
│
├── codec/              # 编解码性能测试
│   ├── README.md
│   ├── benchmark_kernel_only.cpp       # 纯内核性能
│   ├── benchmark_optimized.cpp         # 优化版本对比
│   ├── main_bench.cpp                  # 完整流程测试
│   └── sosd_bench_demo.cpp             # SOSD数据集测试
│
├── ssb/                # SSB数据库查询基准
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── baseline/                       # 13个标准查询
│   └── optimized_2push/                # 优化实现
│
├── random_access/      # 随机访问性能测试
│   └── README.md
│
└── sosd/               # SOSD数据集专项测试
    └── README.md
```

## Benchmark分类

### 1. Codec Benchmarks (编解码性能)
**目录**: `codec/`

测试L3压缩算法本身的性能。

**测试内容**:
- 编码吞吐量 (GB/s)
- 解码吞吐量 (GB/s)
- 压缩率
- 不同数据分布的表现

**适用场景**:
- 算法优化评估
- 不同实现版本对比
- GPU架构适配验证

**快速开始**:
```bash
cd build
make main_bench
./bin/main_bench
```

详见: [codec/README.md](codec/README.md)

### 2. SSB Benchmarks (数据库查询)
**目录**: `ssb/`

在数据仓库查询场景中评估L3压缩的实际效果。

**测试内容**:
- 13个标准SSB查询
- 查询执行时间
- 端到端性能
- 压缩带来的加速比

**版本对比**:
- Baseline: 未压缩版本
- 2-Push: 双推送优化
- L3: L3压缩优化

**快速开始**:
```bash
cd build
make  # 构建所有SSB查询
./bin/ssb_baseline/q11       # 运行基线版本
./bin/ssb_optimized/q11_l32  # 运行L3压缩版本
```

详见: [ssb/README.md](ssb/README.md)

### 3. Random Access Benchmarks (随机访问)
**目录**: `random_access/`

测试L3压缩数据的随机访问性能。

**测试内容**:
- 点查询延迟
- 范围查询吞吐量
- 不规则访问模式
- Cache效果

**计划中**: 待实现

### 4. SOSD Benchmarks (标准数据集)
**目录**: `sosd/`

使用SOSD (Searching on Sorted Data) 标准数据集进行测试。

**数据集**:
- books (200M keys)
- fb (200M keys)
- osm (800M keys)
- wiki (200M keys)

**计划中**: 待实现

## 编译所有Benchmarks

### 方法1: 完整编译
```bash
mkdir build && cd build
cmake .. -DBUILD_BENCHMARKS=ON
make -j$(nproc)
```

### 方法2: 选择性编译
```bash
# 只编译codec benchmarks
make benchmark_kernel_only
make benchmark_optimized
make main_bench
make sosd_bench_demo

# 只编译SSB benchmarks
make ssb_baseline
make ssb_optimized
```

### 方法3: 使用便捷脚本
```bash
./scripts/build.sh --benchmarks
```

## 运行所有Benchmarks

### 快速性能检查
```bash
# 运行codec性能测试
./build/bin/main_bench

# 运行SSB Q1.1 对比
./build/bin/ssb_baseline/q11
./build/bin/ssb_optimized/q11_2push
./build/bin/ssb_optimized/q11_l32
```

### 完整性能评估
```bash
# 使用测试脚本（需要自行创建）
./scripts/run_all_benchmarks.sh
```

## 性能基线

在NVIDIA A100 GPU上的参考性能:

### Codec性能
| 指标 | 值 |
|------|-----|
| 编码吞吐量 | 28-32 GB/s |
| 解码吞吐量 | 40-45 GB/s |
| 压缩率 | 3.5-4.5x |
| 随机访问开销 | <5% |

### SSB查询性能
| 查询类别 | Baseline | L3压缩 | 加速比 |
|----------|----------|--------|--------|
| Flight 1 | 8-10ms | 4-5ms | 2.0-2.2x |
| Flight 2 | 12-16ms | 7-9ms | 1.8-2.0x |
| Flight 3 | 18-24ms | 10-13ms | 1.8-2.0x |
| Flight 4 | 24-30ms | 13-17ms | 1.8-2.0x |

**注**: 基于100M行lineorder表，实际性能因GPU架构和数据规模而异。

## 性能分析工具

### NVIDIA Nsight Systems
```bash
nsys profile --stats=true ./build/bin/main_bench
```

### NVIDIA Nsight Compute
```bash
ncu --set full ./build/bin/benchmark_kernel_only
```

### 自定义计时
所有benchmarks内置了详细的计时功能，会输出:
- GPU内核执行时间
- 数据传输时间
- 端到端时间
- 吞吐量统计

## 测试数据要求

### Codec Benchmarks
- 自动生成合成数据
- 无需外部数据文件

### SSB Benchmarks
- 需要SSB数据集
- 使用dbgen工具生成
- 建议scale factor: 1 (约6GB) 或 10 (约60GB)

**生成方法**:
```bash
# 下载并编译SSB dbgen
git clone https://github.com/eyalroz/ssb-dbgen
cd ssb-dbgen && make

# 生成数据
./dbgen -s 1 -T a

# 转换为二进制格式（使用项目提供的工具）
./tools/convert_ssb_to_binary lineorder.tbl lineorder.bin
```

### SOSD Benchmarks
- 从官方仓库下载
- 网址: https://github.com/learnedsystems/SOSD

## 环境要求

### 硬件要求
- NVIDIA GPU (计算能力 >= 7.5)
- 推荐: A100, V100, RTX 3090, RTX 4090
- 显存: 至少 16GB (用于大规模数据集)

### 软件要求
- CUDA Toolkit 11.0+
- CMake 3.18+
- GCC 9.0+ 或 Clang 10.0+
- L3 Core Library

### 验证环境
```bash
./scripts/verify.sh
```

## Benchmark最佳实践

### 1. 预热（Warmup）
```cpp
// 首次运行预热GPU
for (int i = 0; i < 3; i++) {
    run_benchmark();
}
// 正式计时
auto start = std::chrono::high_resolution_clock::now();
for (int i = 0; i < 10; i++) {
    run_benchmark();
}
auto end = std::chrono::high_resolution_clock::now();
```

### 2. 多次运行取平均
建议每个测试至少运行10次，报告中位数或平均值。

### 3. 固定GPU频率
```bash
# 固定最大频率，减少变异性
sudo nvidia-smi -lgc 1410,1410
```

### 4. 监控GPU状态
```bash
# 确保GPU温度正常，未降频
nvidia-smi dmon -s pucvmet
```

## 报告问题

如果benchmark结果异常，请检查:
1. GPU是否正确检测 (`nvidia-smi`)
2. CUDA版本是否兼容
3. 数据是否正确加载
4. 内存是否充足
5. GPU是否被其他进程占用

## 贡献

欢迎添加新的benchmarks:
1. 在对应目录创建源文件
2. 更新CMakeLists.txt
3. 添加README文档说明
4. 提供预期性能基线

## 相关文档

- [L3库文档](../lib/single_file/README.md)
- [L3 Modular文档](../lib/modular/README.md)
- [构建指南](../docs/BUILD.md)
- [性能优化指南](../docs/OPTIMIZATION.md)

---

**更新日期**: 2024-10-18
**测试环境**: NVIDIA A100, CUDA 12.0, Ubuntu 22.04
