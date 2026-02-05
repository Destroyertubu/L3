# Planner (Cascading Decompression) SSB

本目录实现论文 **Planer.pdf**（cascaded compression + compression planner）中描述的 **cascading decompression** 基线，并对齐
`ssb/guide.txt` 中的 13 个 SSB Query 逻辑与 query plan（join 顺序 / 过滤常量 / Q43 的聚合键等）。

与 SIGMOD'22 论文 *Tile-based Lightweight Integer Compression in GPU* 中的对比口径一致：
- **不支持 bit-packing**（仅使用 byte-aligned NS、FOR、DELTA、RLE 等）
- **解压采用 cascading decompression**：每一层方案单独 kernel pass，解压完成后再跑 query kernel（不做 inline decode）

## Build

```bash
cmake -S "Planner" -B "Planner/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "Planner/build" -j
```

## Run

```bash
"Planner/build/ssb_planner" --query 0 --runs 3 --data_dir "/home/xiayouyang/code/test/ssb_data/"
```

参数：
- `--query/-q`: 11,12,13,21,22,23,31,32,33,34,41,42,43（0 = 全部）
- `--runs/-r`: 每个 query 重复次数
- `--data_dir`: L3 生成的 SSB 二进制列目录（默认同上）

输出格式与现有实现一致（JSON 行）：
`time_h2d`（压缩后列上传 GPU） + `time_metadata`（cascading decompression） + `time_ht_build` + `time_kernel` + `time_d2h`。

