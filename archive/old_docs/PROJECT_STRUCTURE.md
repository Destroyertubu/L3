# L3 项目结构说明

**最后更新**: 2024-10-18

## 整体概览

```
L3/
├── lib/                              # 核心库实现
│   ├── single_file/                  # 单文件版本 (生产部署)
│   └── modular/                      # 模块化版本 (开发调试)
│
├── include/                          # 头文件
│   ├── single_file/                  # 单文件版本头文件
│   ├── modular/                      # 模块化版本头文件
│   └── common/                       # 通用头文件
│
├── benchmarks/                       # 性能测试
│   ├── codec/                        # 编解码性能测试
│   └── ssb/                          # SSB数据库查询测试
│
├── tests/                            # 单元测试
├── docs/                             # 项目文档
├── scripts/                          # 构建脚本
├── tools/                            # 辅助工具
├── examples/                         # 使用示例
└── data/                             # 测试数据
```

## 核心库详细结构

### Single-File (单文件版本)

**路径**: `lib/single_file/`

```
single_file/
├── l3.cu              # 完整L3实现 (~3,500行)
├── CMakeLists.txt     # 构建配置
└── README.md          # 详细说明 (~200行)
```

**特点**:
- ✅ 一个文件包含所有功能
- ✅ SoA数据布局，优化压缩比
- ✅ 编译快 (~30秒)
- ✅ 便于部署和集成

**适用场景**: 生产环境、快速集成、不需修改源码

### Modular (模块化版本)

**路径**: `lib/modular/`

```
modular/
├── codec/                      # 编解码核心 (7个文件)
│   ├── encoder.cu                  编码器基础版本
│   ├── encoder_optimized.cu        编码器优化版本
│   ├── decoder_specialized.cu      专用解码器
│   ├── decoder_warp_opt.cu         Warp优化解码器
│   ├── decompression_kernels.cu    解压缩内核集合
│   ├── l3_codec.cpp                编解码统一接口
│   └── l3_codec_optimized.cpp      优化接口实现
│
├── utils/                      # 工具函数 (5个文件)
│   ├── bitpack_utils.cu            位打包实现
│   ├── bitpack_utils.cuh           位打包头文件
│   ├── partition_bounds_kernel.cu  分区边界计算
│   ├── random_access_kernels.cu    随机访问内核
│   └── timers.cu                   GPU计时工具
│
├── data/                       # 数据处理 (2个文件)
│   ├── sosd_loader.cpp             SOSD数据集加载
│   └── convert_to_binary.cpp       格式转换工具
│
├── CMakeLists.txt             # 构建配置
├── README.md                  # 详细说明 (~200行)
├── ORGANIZATION_REPORT.md     # 组织重构报告
└── STRUCTURE.txt              # 目录结构可视化
```

**特点**:
- ✅ 模块化分离，易于理解
- ✅ 便于开发和调试
- ✅ 可独立修改各模块
- ✅ 14个文件，~6,300行代码

**适用场景**: 学习算法、开发修改、代码调试

## Benchmarks详细结构

**路径**: `benchmarks/`

```
benchmarks/
├── codec/                      # 编解码性能测试 (5个文件)
│   ├── benchmark_kernel_only.cpp   纯内核性能测试
│   ├── benchmark_optimized.cpp     优化版本对比
│   ├── main_bench.cpp              完整流程测试
│   ├── sosd_bench_demo.cpp         SOSD数据集测试
│   └── README.md                   测试说明
│
├── ssb/                        # SSB查询测试 (42个文件)
│   ├── baseline/                   13个标准查询
│   │   └── q11.cu - q43.cu
│   ├── optimized_2push/            27个优化版本
│   │   ├── l32.cu                  L3实现
│   │   ├── qXX_2push.cu            双推送优化 (13个)
│   │   └── qXX_l32.cu              L3压缩优化 (13个)
│   ├── CMakeLists.txt
│   └── README.md
│
├── random_access/              # 随机访问测试 (待实现)
├── sosd/                       # SOSD专项测试 (待实现)
│
├── README.md                   # Benchmarks总览
├── CMakeLists.txt              # 统一构建配置
├── CONSOLIDATION_REPORT.md     # 整合报告
└── STRUCTURE_OVERVIEW.txt      # 结构概览
```

## 头文件结构

**路径**: `include/`

```
include/
├── single_file/               # Single-File版本头文件
│   └── l3.cuh
│
├── modular/                   # Modular版本头文件
│   ├── l3_format.hpp              数据格式定义
│   ├── l3_codec.hpp               编解码接口
│   ├── bitpack_utils.cuh          位打包工具
│   └── ...
│
└── common/                    # 通用头文件
    ├── cuda_helpers.cuh           CUDA辅助函数
    ├── error_check.hpp            错误检查宏
    └── ...
```

## 文档系统

```
docs/
├── README.md                  # 文档索引
├── API.md                     # API文档
├── ALGORITHM.md               # 算法说明
├── BUILD.md                   # 构建指南
├── OPTIMIZATION.md            # 性能优化
├── FAQ.md                     # 常见问题
└── MIGRATION.md               # 迁移指南
```

## 项目配置文件

```
L3/
├── CMakeLists.txt             # 主构建配置
├── .gitignore                 # Git忽略配置
├── LICENSE                    # 许可证
├── README.md                  # 项目主文档
├── PROJECT_STATUS.md          # 项目状态报告
├── PROJECT_STRUCTURE.md       # 本文件
└── LIBRARY_RENAME_REPORT.md   # 库重命名报告
```

## 关键CMake选项

```cmake
# 库版本选择
USE_SINGLE_FILE    # 编译single-file版本 (默认: ON)
USE_MODULAR        # 编译modular版本 (默认: ON)

# 功能选项
BUILD_BENCHMARKS   # 编译benchmarks (默认: ON)
BUILD_EXAMPLES     # 编译示例 (默认: ON)
BUILD_TOOLS        # 编译工具 (默认: ON)
ENABLE_TESTING     # 启用测试 (默认: OFF)
```

## 编译输出

```
build/
├── lib/                       # 静态库
│   ├── libl3_single.a             Single-File静态库
│   └── libl3_modular.a            Modular静态库
│
└── bin/                       # 可执行文件
    ├── codec_benchmarks/          Codec性能测试
    │   ├── main_bench
    │   ├── benchmark_kernel_only
    │   ├── benchmark_optimized
    │   └── sosd_bench_demo
    │
    └── ssb_benchmarks/            SSB查询测试
        ├── ssb_baseline/          13个基线查询
        └── ssb_optimized/         27个优化查询
```

## 文件统计

| 类别 | 数量 | 说明 |
|------|------|------|
| **核心库** | 15个文件 | 1 single + 14 modular |
| **头文件** | ~20个 | single/modular/common |
| **Benchmarks** | 52个文件 | codec + ssb |
| **文档** | 18个 | READMEs + 报告 |
| **配置** | 6个 | CMakeLists.txt |

## 代码量统计

| 模块 | 行数 |
|------|------|
| Single-File | ~3,500行 |
| Modular | ~6,300行 |
| Benchmarks | ~35,000行 |
| 总计 | ~45,000行 |

## 命名规范

### 库命名
- `libl3_single.a` - Single-File静态库
- `libl3_modular.a` - Modular静态库

### CMake目标
- `l3_single` - Single-File库目标
- `l3_modular` - Modular库目标

### 可执行文件
- `benchmark_xxx` - Codec benchmarks
- `qXX`, `qXX_2push`, `qXX_l32` - SSB benchmarks

## 快速导航

### 我想...

**部署到生产环境**
→ 使用 `lib/single_file/`
→ 阅读 `lib/single_file/README.md`

**学习L3算法**
→ 使用 `lib/modular/`
→ 阅读 `lib/modular/README.md`
→ 查看 `lib/modular/codec/` 下的各个文件

**修改编码器**
→ 编辑 `lib/modular/codec/encoder.cu`
→ 重新编译并运行 `benchmarks/codec/main_bench`

**运行性能测试**
→ 编译 benchmarks: `cmake .. -DBUILD_BENCHMARKS=ON && make`
→ Codec测试: `./build/bin/codec_benchmarks/main_bench`
→ SSB测试: `./build/bin/ssb_optimized/q11_l32`

**理解项目组织**
→ 阅读本文件 (PROJECT_STRUCTURE.md)
→ 阅读 `LIBRARY_RENAME_REPORT.md`
→ 阅读 `lib/modular/ORGANIZATION_REPORT.md`

## 最近更新

### 2024-10-18
1. ✅ 库重命名: l3/l3_legacy → single_file/modular
2. ✅ Benchmark整合: 统一到/benchmarks/
3. ✅ Modular模块化组织: codec/utils/data
4. ✅ 完整文档系统创建

## 下一步

- [ ] 添加更多单元测试
- [ ] 完善API文档
- [ ] 实现random_access benchmarks
- [ ] 实现SOSD benchmarks
- [ ] 添加Python绑定

---

**维护者**: L3 Team
**项目状态**: ✅ Active Development
**最后审核**: 2024-10-18
