#!/bin/bash
DATA_DIR="/root/autodl-tmp/code/test/data/sosd"
OUTPUT="reports/nvcomp_results.csv"

mkdir -p "$(dirname "$OUTPUT")"

./benchmark_nvcomp -o "$OUTPUT" \
  -f "$DATA_DIR/1-linear_200M_uint32.bin" \
  -f "$DATA_DIR/2-normal_200M_uint32.bin" \
  -f "$DATA_DIR/3-poisson_87M_uint64.bin" \
  -f "$DATA_DIR/4-ml_uint64.bin" \
  -f "$DATA_DIR/5-books_200M_uint32.bin" \
  -f "$DATA_DIR/6-fb_200M_uint64.bin" \
  -f "$DATA_DIR/7-wiki_200M_uint64.bin" \
  -f "$DATA_DIR/8-osm_cellids_800M_uint64.bin" \
  -f "$DATA_DIR/9-movieid_uint32.bin" \
  -f "$DATA_DIR/10-house_price_uint64.bin" \
  -f "$DATA_DIR/11-planet_uint64.bin" \
  -f "$DATA_DIR/12-libio.bin" \
  -f "$DATA_DIR/13-medicare.bin" \
  -f "$DATA_DIR/14-cosmos_int32.bin" \
  -f "$DATA_DIR/15-polylog_10M_uint64.bin" \
  -f "$DATA_DIR/16-exp_200M_uint64.bin" \
  -f "$DATA_DIR/17-poly_200M_uint64.bin" \
  -f "$DATA_DIR/18-site_250k_uint32.bin" \
  -f "$DATA_DIR/19-weight_25k_uint32.bin" \
  -f "$DATA_DIR/20-adult_30k_uint32.bin" \
  -a all -n 1 -w 1 -g 1

echo "Results saved to $OUTPUT"
wc -l "$OUTPUT"
