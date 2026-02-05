#!/bin/bash
DATA_DIR="/root/autodl-tmp/code/L3/data/sosd"
OUTPUT="/root/autodl-tmp/code/L3/H20/SOSD/EDR/Third_Party/tilegpu_results.csv"

mkdir -p "$(dirname "$OUTPUT")"

# 清除旧结果
rm -f "$OUTPUT"

# 数据集列表（不含13-medicare，改用切分后的版本）
DATASETS=(
  "1-linear_200M_uint32.bin"
  "2-normal_200M_uint32.bin"
  "3-poisson_87M_uint64.bin"
  "4-ml_uint64.bin"
  "5-books_200M_uint32.bin"
  "6-fb_200M_uint64.bin"
  "7-wiki_200M_uint64.bin"
  "8-osm_cellids_800M_uint64.bin"
  "9-movieid_uint32.bin"
  "10-house_price_uint64.bin"
  "11-planet_uint64.bin"
  "12-libio.bin"
  "13-medicare_part1.bin"
  "13-medicare_part2.bin"
  "13-medicare_part3.bin"
  "13-medicare_part4.bin"
  "14-cosmos_int32.bin"
  "15-polylog_10M_uint64.bin"
  "16-exp_200M_uint64.bin"
  "17-poly_200M_uint64.bin"
  "18-site_250k_uint32.bin"
  "19-weight_25k_uint32.bin"
  "20-adult_30k_uint32.bin"
)

echo "Running TileGPU benchmark on ${#DATASETS[@]} datasets..."

for dataset in "${DATASETS[@]}"; do
  filepath="$DATA_DIR/$dataset"
  if [ -f "$filepath" ]; then
    echo "Processing: $dataset"
    ./benchmark_tilegpu -o "$OUTPUT" -f "$filepath" -a all -n 1 -w 1 -g 1
    echo "  Saved to $OUTPUT"
  else
    echo "Skipping (not found): $dataset"
  fi
done

echo "All results saved to $OUTPUT"
wc -l "$OUTPUT"
