# L3 V10 可调参数说明

## 1. 压缩阶段参数 (VerticalConfig)

这些参数在数据加载和压缩时设置，影响压缩率和解压效率。

### 1.1 分区大小参数

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `partition_size_hint` | 4096 | 256-8192 | 目标分区大小（需要256对齐） |
| `cost_min_partition_size` | 256 | ≥256 | 最小分区大小（Cost-optimal模式） |
| `cost_max_partition_size` | 8192 | ≥min | 最大分区大小（Cost-optimal模式） |
| `cost_target_partition_size` | 2048 | min-max | Cost-optimal目标分区大小 |

**调优建议**:
- 较小的分区（1024-2048）：更好的压缩率（分区内delta范围更小）
- 较大的分区（4096-8192）：更低的元数据开销，但可能压缩率下降

### 1.2 分区策略参数

| 参数 | 默认值 | 选项 | 说明 |
|------|--------|------|------|
| `partitioning_strategy` | FIXED | FIXED, VARIANCE_ADAPTIVE, COST_OPTIMAL | 分区策略 |
| `cost_breakpoint_threshold` | 2 | 1-5 | delta-bits变化触发断点的阈值 |
| `cost_merge_benefit_threshold` | 0.05 | 0.01-0.20 | 合并分区的最小收益阈值（5%） |
| `cost_max_merge_rounds` | 4 | 1-10 | 最大合并迭代次数 |
| `cost_enable_merging` | true | true/false | 是否启用基于成本的合并 |

**策略说明**:
- `FIXED`: 固定大小分区，最简单，性能最稳定
- `VARIANCE_ADAPTIVE`: 基于方差自适应分区大小
- `COST_OPTIMAL`: 基于压缩成本优化分区边界，压缩率最好但可能略慢

### 1.3 模型选择参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enable_adaptive_selection` | true | 是否启用自适应模型选择 |
| `fixed_model_type` | MODEL_LINEAR(1) | 固定模型类型（当adaptive=false时） |

**模型类型**:
- `MODEL_LINEAR (1)`: 线性模型 y = a*x + b
- `MODEL_POLYNOMIAL2 (2)`: 二次多项式 y = a*x² + b*x + c
- `MODEL_POLYNOMIAL3 (3)`: 三次多项式
- `MODEL_FOR_BITPACK (4)`: FOR (Frame of Reference) 纯位压缩

**当前SSB数据**: 100%使用FOR_BITPACK，因为数据无明显线性趋势

### 1.4 其他压缩参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_delta_bits` | 64 | 最大delta位宽 |
| `error_bound_factor` | 1.0 | 误差边界因子 |
| `enable_branchless_unpack` | true | 使用无分支位提取 |
| `register_buffer_size` | 4 | 预取到寄存器的字数 |

---

## 2. 查询阶段参数 (V10 Kernel)

这些参数在查询内核中设置，影响GPU执行效率。

### 2.1 并行度参数

| 参数 | 当前值 | 可选范围 | 说明 |
|------|--------|----------|------|
| `V10_BLOCK_THREADS` | 32 | 32, 64, 128, 256 | 每个block的线程数 |
| `V10_PARTITION_SIZE` | 1024 | 256, 512, 1024, 2048 | V10 tile大小 |
| `V10_MV_SIZE` | 256 | 256 (固定) | Mini-vector大小（不建议修改） |
| `parallelism_factor` | 4 | 1, 2, 4, 8 | 每个L3分区的block数 |

**当前配置**:
- 4x并行度: 每个L3分区4个block，每个处理256个值
- 每block 32线程 = 1个warp

### 2.2 可能的调优方向

```cpp
// 方案A: 保持当前配置（推荐）
constexpr int V10_BLOCK_THREADS = 32;   // 1 warp
constexpr int V10_PARTITION_SIZE = 1024; // 4个mini-vectors
// 4x parallelism: 234K blocks

// 方案B: 增加block内线程
constexpr int V10_BLOCK_THREADS = 64;   // 2 warps
constexpr int V10_PARTITION_SIZE = 2048; // 8个mini-vectors
// 可能减少block调度开销，但增加寄存器压力

// 方案C: 8x并行度
constexpr int V10_PARTITION_SIZE = 2048;
// 8 blocks per partition, each handles 256 values
// 更高并行度，但更多block可能导致调度开销
```

---

## 3. 数据加载参数

在 `ssb_data_loader.hpp` 中的 `loadAndCompress()` 函数：

```cpp
void loadAndCompress(const std::string& data_dir, int partition_size = 4096) {
    VerticalConfig config = VerticalConfig::costOptimal();
    config.partition_size_hint = partition_size;
    config.enable_interleaved = true;
    // ...
}
```

### 3.1 可调参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `partition_size` | 4096 | 传入loadAndCompress的分区大小 |

**注意**: 当前SSB使用2048分区大小压缩，产生29291个分区。

---

## 4. 推荐调优实验

### 实验1: 分区大小对比

```cpp
// 在 ssb_data_loader.hpp 中修改
data.loadAndCompress(path, 1024);  // 小分区，更好压缩
data.loadAndCompress(path, 2048);  // 当前默认
data.loadAndCompress(path, 4096);  // 大分区
```

预期:
- 1024: 压缩率+5-10%, 性能可能-5%（更多元数据）
- 4096: 压缩率-5%, 性能可能+5%（更少元数据）

### 实验2: 并行度对比

```cpp
// 在 q11_fused_v10.cu 中修改
// 方案1: 当前 4x
int num_blocks = num_partitions * 4;

// 方案2: 2x
int num_blocks = num_partitions * 2;

// 方案3: 8x (需要修改PARTITION_SIZE为2048)
int num_blocks = num_partitions * 8;
```

### 实验3: Block线程数对比

```cpp
// 32 threads (1 warp) - 当前
constexpr int V10_BLOCK_THREADS = 32;

// 64 threads (2 warps)
constexpr int V10_BLOCK_THREADS = 64;
// 需要修改kernel以利用额外warp

// 128 threads (4 warps)
constexpr int V10_BLOCK_THREADS = 128;
```

### 实验4: 分区策略对比

```cpp
// FIXED (当前)
config.partitioning_strategy = PartitioningStrategy::FIXED;

// COST_OPTIMAL (可能更好压缩)
config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
```

---

## 5. 当前V10配置参数汇总

| 层级 | 参数 | 当前值 | 影响 |
|------|------|--------|------|
| **压缩** | partition_size | 2048 | 压缩率/元数据开销 |
| **压缩** | partitioning_strategy | FIXED | 分区边界选择 |
| **压缩** | enable_adaptive_selection | true | 模型选择（SSB全为FOR） |
| **查询** | V10_BLOCK_THREADS | 32 | GPU占用率 |
| **查询** | V10_PARTITION_SIZE | 1024 | 每个tile的值数 |
| **查询** | parallelism | 4x | 每L3分区的block数 |
| **查询** | values_per_thread | 8 | 每线程处理的值数 |

---

## 6. 性能调优建议

1. **当前配置已经高度优化**: 0.37ms (Q1.1) 比Vertical快32%

2. **如果需要更好压缩率**:
   - 尝试 `partition_size = 1024`
   - 启用 `COST_OPTIMAL` 分区策略

3. **如果需要更快性能**:
   - 尝试 `8x parallelism`
   - 但可能受限于内存带宽

4. **不建议修改**:
   - `V10_MV_SIZE = 256` (Vertical格式要求)
   - `values_per_thread = 8` (与mini-vector布局绑定)
