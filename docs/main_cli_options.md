# L3/L3 Main 程序命令行参数参考

本文档列出了 `./bin/main` 程序的所有命令行参数及其可选值。

## 用法

```bash
./bin/main [options] dataset_id
```

---

## 参数列表

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `--partition-strategy` | `FIXED`, `COST_OPTIMAL`, `VARIANCE_ADAPTIVE` | `COST_OPTIMAL` | 分区策略 |
| `--partition-size` | 正整数 (如 1024, 2048, 4096) | `4096` | 分区大小 |
| `--encoder` | `STANDARD`, `Vertical`, `OPTIMIZED`, `GPU`, `GPU_ZEROSYNC` | `STANDARD` | 编码器类型 |
| `--decoder` | `STANDARD`, `Vertical`, `OPTIMIZED`, `SPECIALIZED`, `PHASE2`, `PHASE2_BUCKET`, `KERNELS_OPT` | `STANDARD` | 解码器类型 |
| `--model-selection` | `ADAPTIVE`, `LINEAR`, `POLY2`, `POLY3`, `FOR` | `ADAPTIVE` | 模型选择策略 |
| `--Vertical-mode` | `SEQUENTIAL`, `INTERLEAVED`, `BRANCHLESS`, `AUTO` | `AUTO` | Vertical解压模式 |
| `--max-delta-bits` | 正整数 (1-64) | `32` | 最大delta位宽 |
| `--random-access-samples` | 正整数 | `10000` | 随机访问测试样本数 |
| `--output-csv` | 文件路径 | (无) | 输出CSV结果文件 |
| `--polynomial` | (无参数，开关) | 关闭 | 在ADAPTIVE模式启用POLY2/POLY3 |
| `--all` | (无参数，开关) | 关闭 | 运行所有数据集(1-23) |
| `--compare-decoders` | (无参数，开关) | 关闭 | 比较所有解码器性能 |
| `--compare-Vertical` | (无参数，开关) | 关闭 | 比较Vertical顺序vs交错 |
| `--help`, `-h` | (无参数，开关) | - | 显示帮助信息 |
| `dataset_id` | `1`-`23` 或省略 | - | 数据集ID |

---

## 详细说明

### 分区策略 (`--partition-strategy`)

| 值 | 说明 |
|----|------|
| `FIXED` | 固定大小分区，最简单 |
| `COST_OPTIMAL` | 成本驱动分区，推荐默认值 |
| `VARIANCE_ADAPTIVE` | 方差驱动分区 (legacy) |

### 编码器类型 (`--encoder`)

| 值 | 说明 | 兼容解码器 |
|----|------|------------|
| `STANDARD` | 标准L3编码器 | L3解码器 |
| `OPTIMIZED` | GPU优化L3编码器 | L3解码器 |
| `Vertical` | Vertical编码器 (CPU元数据) | **仅Vertical** |
| `GPU` | GPU全流程编码器 (动态分配) | **仅Vertical** |
| `GPU_ZEROSYNC` | GPU全流程编码器 (预分配) | **仅Vertical** |

### 解码器类型 (`--decoder`)

| 值 | 说明 | 兼容编码器 |
|----|------|------------|
| `STANDARD` | 标准L3解码器 | L3编码器 |
| `OPTIMIZED` | Warp优化解码器 | L3编码器 |
| `SPECIALIZED` | 运行时位宽适配解码器 | L3编码器 |
| `PHASE2` | cp.async流水线解码器 | L3编码器 |
| `PHASE2_BUCKET` | 分桶调度解码器 | L3编码器 |
| `KERNELS_OPT` | 8/16位特化解码器 | L3编码器 |
| `Vertical` | Vertical解码器 | **仅Vertical编码器** |

### 模型选择策略 (`--model-selection`)

| 值 | 说明 |
|----|------|
| `ADAPTIVE` | 每分区自动选择最优模型 |
| `LINEAR` | 固定使用线性模型 (y = a + bx) |
| `POLY2` | 固定使用二次多项式 |
| `POLY3` | 固定使用三次多项式 |
| `FOR` | 固定使用FOR+BitPack |

### Vertical解压模式 (`--Vertical-mode`)

| 值 | 说明 |
|----|------|
| `AUTO` | 自动选择最优路径 |
| `SEQUENTIAL` | 顺序解压 (已废弃，回退到INTERLEAVED) |
| `INTERLEAVED` | Mini-vector交错解压 |
| `BRANCHLESS` | 无分支解压 (回退到INTERLEAVED) |

### 数据集ID (`dataset_id`)

| ID | 数据集 | 类型 |
|----|--------|------|
| 1 | linear_200M_uint64 | uint64 |
| 2 | normal_200M_uint64 | uint64 |
| 3 | poisson_87M_uint64 | uint64 |
| 4 | ml_uint64 | uint64 |
| 5 | books_200M_uint32 | uint32 |
| 6 | fb_200M_uint64 | uint64 |
| 7 | wiki_200M_uint64 | uint64 |
| 8 | osm_cellids_800M_uint64 | uint64 |
| 9 | movieid_uint32 | uint32 |
| 10 | house_price_uint64 | uint64 |
| 11 | planet_uint64 | uint64 |
| 12 | libio | uint64 |
| 13 | medicare | uint64 |
| 14 | cosmos_int32 | int32 |
| 15 | polylog_10M_uint64 | uint64 |
| 16 | exp_200M_uint64 | uint64 |
| 17 | poly_200M_uint64 | uint64 |
| 18 | site_250k_uint32 | uint32 |
| 19 | weight_25k_uint32 | uint32 |
| 20 | adult_30k_uint32 | uint32 |
| 21 | email (字符串) | string |
| 22 | hex (字符串) | string |
| 23 | words (字符串) | string |

---

## 编码器-解码器配对规则

**强制配对** (2025-12-08):

| 编码器类型 | 允许的解码器 |
|------------|-------------|
| STANDARD, OPTIMIZED | STANDARD, OPTIMIZED, SPECIALIZED, PHASE2, PHASE2_BUCKET, KERNELS_OPT |
| Vertical, GPU, GPU_ZEROSYNC | **仅 Vertical** |

无效组合将在启动时报错退出。

---

## 示例

```bash
# 测试所有数据集，使用默认设置
./bin/main --all

# 测试单个数据集，使用GPU编码器
./bin/main 1 --encoder GPU --decoder Vertical

# 比较所有L3解码器性能
./bin/main 5 --encoder STANDARD --compare-decoders

# 使用线性模型，输出到CSV
./bin/main 2 --model-selection LINEAR --output-csv results.csv

# 比较Vertical解压模式
./bin/main 6 --encoder Vertical --decoder Vertical --compare-Vertical

# 使用固定分区策略，2048分区大小
./bin/main 8 --partition-strategy FIXED --partition-size 2048
```

---

*文档生成日期: 2025-12-08*
