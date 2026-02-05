#!/bin/bash

BENCH=/root/autodl-tmp/code/L3/third_party/tmp/FastLanesGPU-main/fastlanes/example/fastlanes_bench_bitpack
DATA_DIR=/root/autodl-tmp/code/L3/data/sosd
OUT_DIR=/root/autodl-tmp/code/L3/H20/SOSD/EDR/Third_Party/FLS-GPU

mkdir -p $OUT_DIR

echo "=== FastLanes FOR + BitPack Benchmark on SOSD datasets ===" | tee $OUT_DIR/summary.txt
echo "Start: $(date)" | tee -a $OUT_DIR/summary.txt
echo "" | tee -a $OUT_DIR/summary.txt

# 定义数据集: 编号 文件名 类型(32/64)
declare -a DATASETS=(
    "1 1-linear_200M_uint32.bin 32"
    "2 2-normal_200M_uint32.bin 32"
    "3 3-poisson_87M_uint64.bin 64"
    "4 4-ml_uint64.bin 64"
    "5 5-books_200M_uint32.bin 32"
    "6 6-fb_200M_uint64.bin 64"
    "7 7-wiki_200M_uint64.bin 64"
    "8 8-osm_cellids_800M_uint64.bin 64"
    "9 9-movieid_uint32.bin 32"
    "10 10-house_price_uint64.bin 64"
    "11 11-planet_uint64.bin 64"
    "12 12-libio.bin 64"
    "13 13-medicare.bin 64"
    "14 14-cosmos_int32.bin 32"
    "15 15-polylog_10M_uint64.bin 64"
    "16 16-exp_200M_uint64.bin 64"
    "17 17-poly_200M_uint64.bin 64"
    "18 18-site_250k_uint32.bin 32"
    "19 19-weight_25k_uint32.bin 32"
    "20 20-adult_30k_uint32.bin 32"
)

# 结果表头
echo "Dataset,Name,Type,BP_Bitwidth,BP_Ratio,BP_DecTime_ms,BP_DecThroughput_GBs,FOR_Bitwidth,FOR_Ratio,FOR_DecTime_ms,FOR_DecThroughput_GBs" > $OUT_DIR/results.csv

for entry in "${DATASETS[@]}"; do
    read -r idx filename dtype <<< "$entry"
    filepath="$DATA_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo "[$idx] File not found: $filepath, skipping..." | tee -a $OUT_DIR/summary.txt
        continue
    fi

    # 提取数据集名称
    name=$(echo $filename | sed 's/.*-\(.*\)\.bin/\1/' | sed 's/_.*//g')

    echo "----------------------------------------" | tee -a $OUT_DIR/summary.txt
    echo "[$idx] Running: $filename ($dtype-bit)" | tee -a $OUT_DIR/summary.txt

    outfile="$OUT_DIR/${idx}_${name}.log"

    if [ "$dtype" == "64" ]; then
        $BENCH --file "$filepath" --uint64 > "$outfile" 2>&1
    else
        $BENCH --file "$filepath" > "$outfile" 2>&1
    fi

    # 提取结果
    bp_bw=$(grep "BitPack bitwidth:" "$outfile" | head -1 | awk '{print $NF}')
    bp_ratio=$(grep "\[BitPack\]" -A 20 "$outfile" | grep "Compression ratio:" | head -1 | awk '{print $NF}' | tr -d 'x')
    bp_time=$(grep "\[BitPack\] Decode:" -A 5 "$outfile" | grep "Decode time:" | awk '{print $NF}' | tr -d 'ms')
    bp_tp=$(grep "\[BitPack\] Decode:" -A 5 "$outfile" | grep "Decode throughput:" | awk '{print $NF}' | tr -d 'GB/s')

    for_bw=$(grep "FOR uniform bitwidth:" "$outfile" | awk '{print $NF}')
    for_ratio=$(grep "\[FOR\] Encode:" -A 10 "$outfile" | grep "Compression ratio:" | awk '{print $NF}' | tr -d 'x')
    for_time=$(grep "\[FOR\] Decode:" -A 5 "$outfile" | grep "Decode time:" | awk '{print $NF}' | tr -d 'ms')
    for_tp=$(grep "\[FOR\] Decode:" -A 5 "$outfile" | grep "Decode throughput:" | awk '{print $NF}' | tr -d 'GB/s')

    echo "  BitPack: bw=$bp_bw, ratio=${bp_ratio}x, decode=${bp_tp} GB/s" | tee -a $OUT_DIR/summary.txt
    echo "  FOR:     bw=$for_bw, ratio=${for_ratio}x, decode=${for_tp} GB/s" | tee -a $OUT_DIR/summary.txt

    echo "$idx,$name,$dtype,$bp_bw,$bp_ratio,$bp_time,$bp_tp,$for_bw,$for_ratio,$for_time,$for_tp" >> $OUT_DIR/results.csv
done

echo "" | tee -a $OUT_DIR/summary.txt
echo "=== All Complete: $(date) ===" | tee -a $OUT_DIR/summary.txt
echo "Results saved to: $OUT_DIR/results.csv" | tee -a $OUT_DIR/summary.txt
