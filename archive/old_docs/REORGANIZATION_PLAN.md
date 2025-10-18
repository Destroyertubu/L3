# L3 GPU压缩项目 - 完整重构方案

## 📋 项目现状分析

### 发现的核心功能

1. **GPU压缩/解压缩**
   - 学习型压缩（模型: CONSTANT, LINEAR, POLYNOMIAL2/3, DIRECT_COPY）
   - 基于残差的bit-packing
   - 多种优化版本（warp优化、specialized解码器）

2. **两种分块策略**
   - ✅ **固定大小分块** (`createFixedSizePartitions`)
   - ✅ **变长自适应分块** (`GPUVariableLengthPartitionerV6`)

3. **随机访问**
   - 无需全解压的单值访问
   - 分区级别的查找优化

4. **查询执行**
   - SSB基准测试（13个查询）
   - 三种实现：baseline, 2-push优化, L3压缩

### 代码重复问题

1. **双实现系统**
   - `lib/single_file/` - 单文件实现 (701行 l3gpu_impl.cuh)
   - `lib/modular/` - 模块化实现
   - ❌ 两套代码维护困难

2. **文档混乱**
   - 根目录17个文档文件
   - 缺少清晰的入口文档

3. **分块策略分离**
   - 固定分块在 `l3_codec.cpp`
   - 变长分块在 `single_file/partitioner_impl.cuh`
   - ❌ 用户无法灵活选择

---

## 🎯 重构目标

### 核心目标
1. **统一分块接口** - 让用户能够选择分块策略
2. **消除代码重复** - 保留一套最优实现
3. **清晰的模块划分** - 按功能组织代码
4. **易用的API** - C++和Python双接口
5. **完善的文档** - 从入门到高级

### 设计原则
- **Strategy Pattern** - 分块策略可插拔
- **SOLID** - 单一职责、开闭原则
- **DRY** - 不要重复自己
- **KISS** - 保持简单

---

## 🏗️ 新项目结构

```
L3/
├── README.md                          # 项目主文档
├── CHANGELOG.md                       # 版本更新日志
├── LICENSE                            # MIT许可证
├── CMakeLists.txt                     # 主构建文件
├── setup.py                          # Python包配置
├── requirements.txt                   # Python依赖
├── .gitignore
│
├── include/                          # 公共头文件（API）
│   ├── l3/
│   │   ├── l3.hpp                   # 主头文件
│   │   ├── compression.hpp          # 压缩API
│   │   ├── decompression.hpp        # 解压缩API
│   │   ├── random_access.hpp        # 随机访问API
│   │   ├── query.hpp                # 查询执行API
│   │   │
│   │   ├── partitioner.hpp          # 分块策略接口 ⭐核心
│   │   │   └── PartitionStrategy (抽象基类)
│   │   │       ├── FixedSizePartitioner (固定分块)
│   │   │       └── VariableLengthPartitioner (变长分块)
│   │   │
│   │   ├── config.hpp               # 配置结构
│   │   └── types.hpp                # 数据类型定义
│   │
│   └── l3/                          # 内部实现头文件
│       └── internal/
│           ├── format.hpp           # 格式规范
│           ├── kernels.cuh          # CUDA kernels声明
│           └── utils.cuh            # 工具函数
│
├── src/                             # 源代码实现
│   ├── core/                        # 核心组件
│   │   ├── format.cpp              # 格式实现
│   │   ├── config.cpp              # 配置管理
│   │   └── CMakeLists.txt
│   │
│   ├── partitioner/                 # 分块策略实现 ⭐核心
│   │   ├── partitioner_base.cpp    # 基类实现
│   │   ├── fixed_size_partitioner.cu      # 固定分块
│   │   ├── variable_length_partitioner.cu # 变长分块
│   │   ├── partition_kernels.cu    # GPU kernels
│   │   └── CMakeLists.txt
│   │
│   ├── compression/                 # 压缩模块
│   │   ├── encoder.cu              # 编码器（基础版）
│   │   ├── encoder_optimized.cu    # 编码器（优化版）
│   │   ├── model_fitting.cu        # 模型拟合kernels
│   │   ├── bitpacking.cu           # Bit-packing kernels
│   │   ├── compression_api.cpp     # API实现
│   │   └── CMakeLists.txt
│   │
│   ├── decompression/              # 解压缩模块
│   │   ├── decoder.cu              # 解码器（基础版）
│   │   ├── decoder_warp_opt.cu     # Warp优化解码器
│   │   ├── decoder_specialized.cu  # 专用解码器
│   │   ├── decompression_api.cpp   # API实现
│   │   └── CMakeLists.txt
│   │
│   ├── random_access/              # 随机访问模块
│   │   ├── ra_kernels.cu           # 随机访问kernels
│   │   ├── ra_api.cpp              # API实现
│   │   └── CMakeLists.txt
│   │
│   ├── query/                      # 查询执行模块
│   │   ├── optimizer/              # 查询优化器
│   │   │   ├── predicate_pushdown.cu
│   │   │   └── partition_pruning.cu
│   │   ├── operators/              # 查询算子
│   │   │   ├── scan.cu
│   │   │   ├── filter.cu
│   │   │   ├── aggregate.cu
│   │   │   └── join.cu
│   │   ├── query_api.cpp           # API实现
│   │   └── CMakeLists.txt
│   │
│   └── utils/                      # 工具函数
│       ├── gpu_utils.cu            # GPU工具
│       ├── bitpack_utils.cu        # Bit-packing工具
│       ├── timers.cu               # 计时器
│       └── CMakeLists.txt
│
├── python/                         # Python绑定
│   ├── l3_compression/
│   │   ├── __init__.py
│   │   ├── compression.py          # 压缩接口
│   │   ├── decompression.py        # 解压缩接口
│   │   ├── partitioner.py          # 分块策略接口 ⭐
│   │   ├── random_access.py        # 随机访问接口
│   │   ├── query.py                # 查询接口
│   │   └── _bindings.cpp           # pybind11绑定
│   │
│   ├── visualization/              # 可视化工具
│   │   ├── __init__.py
│   │   ├── painter.py              # 图表生成
│   │   ├── heatmap.py              # 热力图
│   │   └── performance.py          # 性能分析
│   │
│   └── setup.py
│
├── benchmarks/                     # 性能测试
│   ├── compression/               # 压缩性能
│   │   ├── bench_fixed_partition.cpp
│   │   ├── bench_variable_partition.cpp
│   │   ├── bench_compare_partitioners.cpp  ⭐
│   │   └── bench_sosd.cpp
│   │
│   ├── decompression/             # 解压缩性能
│   │   └── bench_decompression.cpp
│   │
│   ├── random_access/             # 随机访问性能
│   │   └── bench_random_access.cpp
│   │
│   ├── ssb/                       # SSB查询测试
│   │   ├── baseline/              # 无压缩基准
│   │   ├── l3_fixed/              # 固定分块版本
│   │   ├── l3_variable/           # 变长分块版本
│   │   └── optimized/             # 2-push优化版本
│   │
│   ├── scripts/
│   │   ├── run_all_benchmarks.sh
│   │   ├── compare_partitioners.sh  ⭐
│   │   └── analyze_results.py
│   │
│   └── CMakeLists.txt
│
├── tests/                         # 单元测试
│   ├── unit/
│   │   ├── test_fixed_partitioner.cu      ⭐
│   │   ├── test_variable_partitioner.cu   ⭐
│   │   ├── test_compression.cu
│   │   ├── test_decompression.cu
│   │   ├── test_random_access.cu
│   │   └── test_query.cu
│   │
│   ├── integration/
│   │   ├── test_end_to_end_fixed.cu
│   │   ├── test_end_to_end_variable.cu
│   │   └── test_ssb_queries.cu
│   │
│   └── CMakeLists.txt
│
├── examples/                      # 使用示例
│   ├── cpp/
│   │   ├── 01_basic_compression.cpp
│   │   ├── 02_choose_partitioner.cpp      ⭐核心示例
│   │   ├── 03_custom_config.cpp
│   │   ├── 04_random_access.cpp
│   │   └── 05_ssb_query.cpp
│   │
│   ├── python/
│   │   ├── 01_basic_usage.py
│   │   ├── 02_partition_strategies.py     ⭐核心示例
│   │   ├── 03_benchmark_comparison.py
│   │   ├── 04_visualization.ipynb
│   │   └── 05_ssb_queries.py
│   │
│   └── README.md
│
├── data/                          # 数据文件
│   ├── samples/                   # 示例数据
│   └── README.md
│
├── tools/                         # 辅助工具
│   ├── data_generator.py         # 数据生成器
│   ├── format_converter.py       # 格式转换
│   ├── profiler.py               # 性能分析
│   └── partition_tuner.py        # 分块参数调优 ⭐
│
├── docs/                          # 文档
│   ├── README.md                 # 文档索引
│   ├── getting_started.md        # 快速开始
│   │
│   ├── user_guide/              # 用户指南
│   │   ├── installation.md
│   │   ├── basic_usage.md
│   │   ├── partition_strategies.md       ⭐重点
│   │   ├── performance_tuning.md
│   │   └── advanced_features.md
│   │
│   ├── api_reference/           # API参考
│   │   ├── cpp/
│   │   │   ├── compression.md
│   │   │   ├── partitioner.md            ⭐重点
│   │   ��   ├── decompression.md
│   │   │   ├── random_access.md
│   │   │   └── query.md
│   │   └── python/
│   │       └── ... (同上)
│   │
│   ├── architecture/            # 架构文档
│   │   ├── overview.md
│   │   ├── format_specification.md
│   │   ├── partition_strategies.md       ⭐重点
│   │   ├── compression_pipeline.md
│   │   └── query_optimization.md
│   │
│   ├── performance/             # 性能文档
│   │   ├── benchmarks.md
│   │   ├── partition_comparison.md       ⭐重点
│   │   └── tuning_guide.md
│   │
│   └── development/             # 开发文档
│       ├── build.md
│       ├── testing.md
│       ├── contributing.md
│       └── adding_partitioner.md         ⭐重点
│
├── scripts/                      # 构建和部署脚本
│   ├── build.sh
│   ├── test.sh
│   ├── install.sh
│   └── deploy.sh
│
└── archive/                      # 归档
    ├── old_docs/                # 旧文档（17个文件）
    └── deprecated/              # 废弃代码
        └── single_file/         # 旧的单文件实现
```

---

## ⭐ 核心设计：统一的分块策略接口

### 1. 抽象基类设计

```cpp
// include/l3/partitioner.hpp

namespace l3 {

/**
 * 分块策略抽象接口
 *
 * 用户可以选择不同的分块策略：
 * - FixedSizePartitioner: 固定大小分块
 * - VariableLengthPartitioner: 变长自适应分块
 * - 或自定义分块策略
 */
class PartitionStrategy {
public:
    virtual ~PartitionStrategy() = default;

    /**
     * 对数据进行分块
     *
     * @param data 输入数据
     * @param size 数据大小
     * @return PartitionInfo vector包含每个分区的[start, end]
     */
    virtual std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) = 0;

    /**
     * 获取策略名称
     */
    virtual const char* getName() const = 0;

    /**
     * 获取策略配置
     */
    virtual PartitionConfig getConfig() const = 0;
};

/**
 * 固定大小分块策略
 *
 * 特点：
 * - 简单、可预测
 * - 固定的partition_size
 * - 适合数据均匀分布的场景
 */
class FixedSizePartitioner : public PartitionStrategy {
public:
    explicit FixedSizePartitioner(int partition_size = 4096);

    std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) override;

    const char* getName() const override {
        return "FixedSize";
    }

    PartitionConfig getConfig() const override;

private:
    int partition_size_;
};

/**
 * 变长自适应分块策略
 *
 * 特点：
 * - 基于方差的自适应分块
 * - 高方差区域 → 小分区（更精细压缩）
 * - 低方差区域 → 大分区（更快处理）
 * - 适合数据分布不均的场景
 *
 * 算法：
 * 1. 分析数据方差分布
 * 2. 根据方差阈值划分bucket
 * 3. 为每个bucket分配不同的分区大小
 */
class VariableLengthPartitioner : public PartitionStrategy {
public:
    /**
     * 构造函数
     *
     * @param base_size 基础分区大小（默认1024）
     * @param variance_multiplier 方差块大小倍数（默认8）
     * @param num_thresholds 方差阈值数量（默认3）
     */
    explicit VariableLengthPartitioner(
        int base_size = 1024,
        int variance_multiplier = 8,
        int num_thresholds = 3
    );

    std::vector<PartitionInfo> partition(
        const void* data,
        size_t size,
        size_t element_size
    ) override;

    const char* getName() const override {
        return "VariableLength";
    }

    PartitionConfig getConfig() const override;

private:
    int base_size_;
    int variance_multiplier_;
    int num_thresholds_;
};

/**
 * 分块策略工厂
 */
class PartitionerFactory {
public:
    enum Strategy {
        FIXED_SIZE,
        VARIABLE_LENGTH,
        AUTO  // 自动选择最优策略
    };

    static std::unique_ptr<PartitionStrategy> create(
        Strategy strategy,
        const PartitionConfig& config = PartitionConfig()
    );

    static std::unique_ptr<PartitionStrategy> createAuto(
        const void* data,
        size_t size,
        size_t element_size
    );
};

} // namespace l3
```

### 2. 压缩API集成分块策略

```cpp
// include/l3/compression.hpp

namespace l3 {

/**
 * 压缩配置
 */
struct CompressionConfig {
    // 分块策略选择
    PartitionerFactory::Strategy partition_strategy = PartitionerFactory::AUTO;

    // 分块参数
    int partition_size_hint = 4096;        // 固定分块大小或基础大小
    int variance_multiplier = 8;           // 变长分块：方差块倍数
    int num_thresholds = 3;                // 变长分块：阈值数量

    // 压缩参数
    int max_delta_bits = 32;
    double error_bound_factor = 1.0;

    // 性能选项
    bool use_optimized_encoder = true;
    bool enable_predicate_pushdown = true;

    CompressionConfig() = default;
};

/**
 * 压缩API - 简单版本（自动选择策略）
 */
template<typename T>
CompressedData<T>* compress(
    const T* data,
    size_t size,
    const CompressionConfig& config = CompressionConfig()
);

/**
 * 压缩API - 高级版本（指定分块策略）
 */
template<typename T>
CompressedData<T>* compressWithPartitioner(
    const T* data,
    size_t size,
    PartitionStrategy* partitioner,
    const CompressionConfig& config = CompressionConfig()
);

} // namespace l3
```

### 3. 使用示例

#### C++ 示例

```cpp
// examples/cpp/02_choose_partitioner.cpp

#include <l3/l3.hpp>
#include <vector>
#include <iostream>

int main() {
    // 准备数据
    std::vector<int64_t> data = generateTestData(1000000);

    // ========== 方式1: 自动选择策略 ==========
    l3::CompressionConfig config_auto;
    config_auto.partition_strategy = l3::PartitionerFactory::AUTO;

    auto* compressed_auto = l3::compress(
        data.data(),
        data.size(),
        config_auto
    );

    std::cout << "Auto strategy: "
              << compressed_auto->getCompressionRatio() << "x\n";

    // ========== 方式2: 固定大小分块 ==========
    l3::CompressionConfig config_fixed;
    config_fixed.partition_strategy = l3::PartitionerFactory::FIXED_SIZE;
    config_fixed.partition_size_hint = 4096;

    auto* compressed_fixed = l3::compress(
        data.data(),
        data.size(),
        config_fixed
    );

    std::cout << "Fixed-size (4096): "
              << compressed_fixed->getCompressionRatio() << "x\n";

    // ========== 方式3: 变长自适应分块 ==========
    l3::CompressionConfig config_variable;
    config_variable.partition_strategy = l3::PartitionerFactory::VARIABLE_LENGTH;
    config_variable.partition_size_hint = 1024;    // base_size
    config_variable.variance_multiplier = 8;
    config_variable.num_thresholds = 3;

    auto* compressed_variable = l3::compress(
        data.data(),
        data.size(),
        config_variable
    );

    std::cout << "Variable-length (1024,8,3): "
              << compressed_variable->getCompressionRatio() << "x\n";

    // ========== 方式4: 自定义分块策略对象 ==========
    auto partitioner = l3::PartitionerFactory::create(
        l3::PartitionerFactory::VARIABLE_LENGTH,
        {.base_size = 2048, .variance_multiplier = 16, .num_thresholds = 5}
    );

    auto* compressed_custom = l3::compressWithPartitioner(
        data.data(),
        data.size(),
        partitioner.get(),
        config_auto
    );

    std::cout << "Custom variable-length (2048,16,5): "
              << compressed_custom->getCompressionRatio() << "x\n";

    // ========== 对比不同策略 ==========
    l3::benchmark::comparePartitioners(data.data(), data.size(), {
        l3::PartitionerFactory::FIXED_SIZE,
        l3::PartitionerFactory::VARIABLE_LENGTH
    });

    // 清理
    delete compressed_auto;
    delete compressed_fixed;
    delete compressed_variable;
    delete compressed_custom;

    return 0;
}
```

#### Python 示例

```python
# examples/python/02_partition_strategies.py

import l3_compression as l3
import numpy as np

# 准备数据
data = np.random.randint(0, 1000000, size=1000000, dtype=np.int64)

# ========== 方式1: 自动选择策略 ==========
config_auto = l3.CompressionConfig(partition_strategy='auto')
compressed_auto = l3.compress(data, config_auto)
print(f"Auto strategy: {compressed_auto.compression_ratio}x")

# ========== 方式2: 固定大小分块 ==========
config_fixed = l3.CompressionConfig(
    partition_strategy='fixed',
    partition_size=4096
)
compressed_fixed = l3.compress(data, config_fixed)
print(f"Fixed-size (4096): {compressed_fixed.compression_ratio}x")

# ========== 方式3: 变长自适应分块 ==========
config_variable = l3.CompressionConfig(
    partition_strategy='variable',
    base_size=1024,
    variance_multiplier=8,
    num_thresholds=3
)
compressed_variable = l3.compress(data, config_variable)
print(f"Variable-length (1024,8,3): {compressed_variable.compression_ratio}x")

# ========== 方式4: 使用分块器对象 ==========
partitioner = l3.VariableLengthPartitioner(
    base_size=2048,
    variance_multiplier=16,
    num_thresholds=5
)
compressed_custom = l3.compress_with_partitioner(data, partitioner)
print(f"Custom variable-length: {compressed_custom.compression_ratio}x")

# ========== 对比不同策略 ==========
results = l3.benchmark.compare_partitioners(data, [
    ('Fixed 2048', l3.FixedSizePartitioner(2048)),
    ('Fixed 4096', l3.FixedSizePartitioner(4096)),
    ('Variable (1024,8,3)', l3.VariableLengthPartitioner(1024, 8, 3)),
    ('Variable (2048,16,5)', l3.VariableLengthPartitioner(2048, 16, 5))
])

# 可视化对比结果
l3.visualization.plot_partitioner_comparison(results)
```

---

## 🔄 文件映射和迁移计划

### Phase 1: 创建新目录结构
```bash
# 创建核心目录
mkdir -p include/l3/{,internal}
mkdir -p src/{core,partitioner,compression,decompression,random_access,query,utils}
mkdir -p python/l3_compression python/visualization
mkdir -p benchmarks/{compression,decompression,random_access,ssb}
mkdir -p tests/{unit,integration}
mkdir -p examples/{cpp,python}
mkdir -p tools
mkdir -p docs/{user_guide,api_reference,architecture,performance,development}
mkdir -p archive/{old_docs,deprecated}
```

### Phase 2: 迁移分块相关代码

| 源文件 | 目标文件 | 操作 |
|-------|---------|------|
| `lib/modular/codec/l3_codec.cpp` (createFixedSizePartitions) | `src/partitioner/fixed_size_partitioner.cu` | 提取+重构 |
| `lib/single_file/include/l3/partitioner_impl.cuh` | `src/partitioner/variable_length_partitioner.cu` | 移动+重构 |
| `lib/single_file/include/l3/kernels/partition_kernels_impl.cuh` | `src/partitioner/partition_kernels.cu` | 移动+重构 |
| 新建 | `include/l3/partitioner.hpp` | 创建接口 |
| 新建 | `src/partitioner/partitioner_base.cpp` | 创建基类 |

### Phase 3: 迁移编解码代码

| 源文件 | 目标文件 | 操作 |
|-------|---------|------|
| `lib/modular/codec/encoder.cu` | `src/compression/encoder.cu` | 移动+整合 |
| `lib/modular/codec/encoder_optimized.cu` | `src/compression/encoder_optimized.cu` | 移动+整合 |
| `lib/modular/codec/decompression_kernels.cu` | `src/decompression/decoder.cu` | 移动+重命名 |
| `lib/modular/codec/decoder_warp_opt.cu` | `src/decompression/decoder_warp_opt.cu` | 移动 |
| `lib/modular/codec/decoder_specialized.cu` | `src/decompression/decoder_specialized.cu` | 移动 |
| `lib/modular/codec/l3_codec.cpp` | `src/compression/compression_api.cpp` | 提取+重构 |
| ~~`lib/single_file/`~~ | `archive/deprecated/single_file/` | 归档 |

### Phase 4: 迁移其他模块

| 源文件 | 目标文件 | 操作 |
|-------|---------|------|
| `lib/modular/utils/random_access_kernels.cu` | `src/random_access/ra_kernels.cu` | 移动 |
| `lib/modular/utils/bitpack_utils.cu` | `src/utils/bitpack_utils.cu` | 移动 |
| `lib/modular/utils/timers.cu` | `src/utils/timers.cu` | 移动 |
| `include/modular/l3_format.hpp` | `include/l3/internal/format.hpp` | 移动 |
| `include/common/*.cuh` | `include/l3/internal/` | 移动+整理 |

### Phase 5: 迁移测试和示例

| 源文件 | 目标文件 | 操作 |
|-------|---------|------|
| `benchmarks/codec/*.cpp` | `benchmarks/compression/` | 移动+重构 |
| `benchmarks/ssb/baseline/*.cu` | `benchmarks/ssb/baseline/` | 保持 |
| `benchmarks/ssb/optimized_2push/*.cu` | `benchmarks/ssb/{l3_fixed,l3_variable,optimized}/` | 分类移动 |
| 新建 | `benchmarks/compression/bench_compare_partitioners.cpp` | 创建对比测试 |
| 新建 | `tests/unit/test_*_partitioner.cu` | 创建单元测试 |
| 新建 | `examples/cpp/02_choose_partitioner.cpp` | 创建示例 |

### Phase 6: 文档整理

| 源文件 | 目标文件 | 操作 |
|-------|---------|------|
| 根目录17个文档 | `archive/old_docs/` | 归档 |
| 新建 | `README.md` | 创建主文档 |
| 新建 | `docs/getting_started.md` | 快速入门 |
| 新建 | `docs/user_guide/partition_strategies.md` | 分块策略指南 |
| 新建 | `docs/api_reference/cpp/partitioner.md` | API文档 |
| 新建 | `docs/performance/partition_comparison.md` | 性能对比 |

---

## 📊 代码统计

### 当前项目
- CUDA文件: 74个, ~26K行
- C++文件: 14个, ~3K行
- **总计**: ~29K行代码

### 重复代码估算
- `lib/single_file/` vs `lib/modular/`: ~40%重复
- 删除后预计: ~18K行核心代码

### 新增代码估算
- 分块策略接口: ~500行
- Python绑定: ~1000行
- 测试代码: ~2000行
- 文档: ~5000行
- **预计总量**: ~26K行（优化后）

---

## ⚙️ 构建系统

### CMake 结构
```cmake
# CMakeLists.txt (根目录)
project(L3_Compression LANGUAGES CXX CUDA)

option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON "Build Python bindings" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# 子模块
add_subdirectory(src/core)
add_subdirectory(src/partitioner)      # 分块策略
add_subdirectory(src/compression)
add_subdirectory(src/decompression)
add_subdirectory(src/random_access)
add_subdirectory(src/query)
add_subdirectory(src/utils)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# 库目标
add_library(l3_compression STATIC
    $<TARGET_OBJECTS:l3_core>
    $<TARGET_OBJECTS:l3_partitioner>
    $<TARGET_OBJECTS:l3_compression_impl>
    $<TARGET_OBJECTS:l3_decompression_impl>
    $<TARGET_OBJECTS:l3_random_access>
    $<TARGET_OBJECTS:l3_utils>
)

target_include_directories(l3_compression
    PUBLIC include
    PRIVATE src
)
```

---

## 🧪 测试策略

### 单元测试
```cpp
// tests/unit/test_fixed_partitioner.cu
TEST(FixedSizePartitioner, BasicPartitioning) {
    FixedSizePartitioner p(1024);
    std::vector<int> data(10000);
    auto partitions = p.partition(data.data(), data.size(), sizeof(int));

    EXPECT_EQ(partitions.size(), 10);
    EXPECT_EQ(partitions[0].start, 0);
    EXPECT_EQ(partitions[0].end, 1024);
}

// tests/unit/test_variable_partitioner.cu
TEST(VariableLengthPartitioner, AdaptivePartitioning) {
    VariableLengthPartitioner p(1024, 8, 3);
    std::vector<int> data = generateVariableData(10000);
    auto partitions = p.partition(data.data(), data.size(), sizeof(int));

    // 验证分区大小不同
    std::set<int> sizes;
    for (const auto& part : partitions) {
        sizes.insert(part.end - part.start);
    }
    EXPECT_GT(sizes.size(), 1);  // 至少有2种不同大小
}
```

### 集成测试
```cpp
// tests/integration/test_end_to_end_comparison.cu
TEST(Integration, ComparePartitioners) {
    std::vector<int64_t> data = loadSOSDDataset("books");

    // 固定分块
    auto config_fixed = createConfig(PartitionerFactory::FIXED_SIZE);
    auto* c1 = compress(data.data(), data.size(), config_fixed);

    // 变长分块
    auto config_var = createConfig(PartitionerFactory::VARIABLE_LENGTH);
    auto* c2 = compress(data.data(), data.size(), config_var);

    // 验证压缩率
    EXPECT_GT(c1->getCompressionRatio(), 1.0);
    EXPECT_GT(c2->getCompressionRatio(), 1.0);

    // 验证解压正确性
    auto* d1 = decompress(c1);
    auto* d2 = decompress(c2);
    EXPECT_EQ(memcmp(d1, data.data(), data.size() * sizeof(int64_t)), 0);
    EXPECT_EQ(memcmp(d2, data.data(), data.size() * sizeof(int64_t)), 0);
}
```

---

## 📈 性能测试

### 对比测试框架
```cpp
// benchmarks/compression/bench_compare_partitioners.cpp

struct BenchmarkResult {
    std::string partitioner_name;
    double compression_ratio;
    double compression_time_ms;
    double decompression_time_ms;
    double throughput_gbps;
};

std::vector<BenchmarkResult> benchmarkPartitioners(
    const void* data,
    size_t size,
    const std::vector<PartitionStrategy*>& strategies
) {
    std::vector<BenchmarkResult> results;

    for (auto* strategy : strategies) {
        BenchmarkResult result;
        result.partitioner_name = strategy->getName();

        // 测试压缩
        auto start = std::chrono::high_resolution_clock::now();
        auto* compressed = compressWithPartitioner(data, size, strategy);
        auto end = std::chrono::high_resolution_clock::now();

        result.compression_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        result.compression_ratio = compressed->getCompressionRatio();

        // 测试解压缩
        start = std::chrono::high_resolution_clock::now();
        auto* decompressed = decompress(compressed);
        end = std::chrono::high_resolution_clock::now();

        result.decompression_time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        result.throughput_gbps =
            (size * sizeof(int64_t) / 1e9) / (result.decompression_time_ms / 1000.0);

        results.push_back(result);

        delete compressed;
        delete decompressed;
    }

    return results;
}

int main() {
    // 加载测试数据
    auto data = loadSOSDDataset("books");

    // 创建不同的分块策略
    std::vector<PartitionStrategy*> strategies = {
        new FixedSizePartitioner(1024),
        new FixedSizePartitioner(2048),
        new FixedSizePartitioner(4096),
        new VariableLengthPartitioner(1024, 8, 3),
        new VariableLengthPartitioner(1024, 16, 5),
        new VariableLengthPartitioner(2048, 8, 3)
    };

    // 运行测试
    auto results = benchmarkPartitioners(data.data(), data.size(), strategies);

    // 输出结果
    printBenchmarkTable(results);
    exportToCSV(results, "partition_comparison.csv");

    // 清理
    for (auto* s : strategies) delete s;

    return 0;
}
```

---

## 📚 文档计划

### 核心文档
1. **README.md** - 项目概述、快速开始
2. **docs/user_guide/partition_strategies.md** - 分块策略详解
3. **docs/api_reference/cpp/partitioner.md** - API参考
4. **docs/performance/partition_comparison.md** - 性能对比
5. **docs/development/adding_partitioner.md** - 如何添加新策略

### 示例代码
- 每个示例都有详细注释
- C++和Python版本对应
- 从简单到复杂的学习路径

---

## ⏱️ 执行时间表

### Week 1: 基础重构
- Day 1-2: 创建新目录结构
- Day 3-4: 实现分块策略接口
- Day 5-7: 迁移和重构分块代码

### Week 2: 模块迁移
- Day 1-2: 迁移压缩/解压缩代码
- Day 3-4: 迁移随机访问和工具代码
- Day 5-7: 更新CMake构建系统

### Week 3: 测试和示例
- Day 1-3: 编写单元测试
- Day 4-5: 编写集成测试
- Day 6-7: 创建示例代码

### Week 4: 文档和优化
- Day 1-3: 编写核心文档
- Day 4-5: 性能测试和优化
- Day 6-7: 最终验证和发布

---

## ✅ 验收标准

### 功能完整性
- [x] 支持固定大小分块
- [x] 支持变长自适应分块
- [x] 用户可以轻松选择分块策略
- [x] API简洁易用
- [x] Python绑定完整

### 代码质量
- [x] 消除重复代码
- [x] 清晰的模块划分
- [x] 完善的注释
- [x] 统一的代码风格

### 测试覆盖
- [x] 单元测试覆盖率 > 80%
- [x] 集成测试覆盖主要场景
- [x] 性能测试对比不同策略

### 文档完善
- [x] API文档完整
- [x] 用户指南清晰
- [x] 示例代码丰富
- [x] 性能数据详实

---

## 🎉 预期成果

### 用户体验提升
1. **灵活性**: 用户可以根据数据特点选择最优分块策略
2. **简单性**: 一行代码切换策略
3. **性能**: 自动选择或手动优化都能获得好性能
4. **可扩展**: 轻松添加新的分块策略

### 代码质量提升
1. **可维护性**: 消除重复代码，单一实现
2. **可读性**: 清晰的模块划分和文档
3. **可测试性**: 完善的测试框架
4. **可扩展性**: 策略模式支持新算法

### 性能数据
- 固定分块: 适合均匀数据，速度快
- 变长分块: 适合不均匀数据，压缩率高
- 自动选择: 根据数据特征自动优化

---

## 🚀 下一步行动

是否批准此重构方案？如果同意，我将立即开始执行：

1. 创建新目录结构
2. 实现分块策略接口
3. 迁移和重构代码
4. 编写测试和示例
5. 完善文档

请确认是否开始执行！
