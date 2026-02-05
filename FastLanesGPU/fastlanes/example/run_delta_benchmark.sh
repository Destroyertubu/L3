#!/bin/bash

BENCH=/root/autodl-tmp/code/L3/third_party/tmp/FastLanesGPU-main/fastlanes/example/fastlanes_bench_delta
DATA_DIR=/root/autodl-tmp/code/L3/data/sosd
OUT_DIR=/root/autodl-tmp/code/L3/H20/SOSD/EDR/Third_Party/FLS-GPU

mkdir -p $OUT_DIR

echo "=== FastLanes Delta+FOR+BitPack Benchmark (Per-Block COMPUTE+PADDED) ===" | tee $OUT_DIR/delta_summary.txt
echo "Decode kernel: delta_decode_per_block_compute_padded (u32/u64)" | tee -a $OUT_DIR/delta_summary.txt
echo "Start: $(date)" | tee -a $OUT_DIR/delta_summary.txt
echo "" | tee -a $OUT_DIR/delta_summary.txt

# 定义数据集
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

echo "Dataset,Name,Type,PerBlock_Ratio,DecTime_ms,Throughput_GBs" > $OUT_DIR/delta_results.csv

for entry in "${DATASETS[@]}"; do
    read -r idx filename dtype <<< "$entry"
    filepath="$DATA_DIR/$filename"

    if [ ! -f "$filepath" ]; then
        echo "[$idx] File not found: $filepath, skipping..." | tee -a $OUT_DIR/delta_summary.txt
        continue
    fi

    name=$(echo $filename | sed 's/.*-\(.*\)\.bin/\1/' | sed 's/_.*//g')

    echo "[$idx] Running: $filename ($dtype-bit)" | tee -a $OUT_DIR/delta_summary.txt

    outfile="$OUT_DIR/delta_${idx}_${name}.log"

    if [ "$dtype" == "64" ]; then
        $BENCH --file "$filepath" --uint64 > "$outfile" 2>&1
    else
        $BENCH --file "$filepath" > "$outfile" 2>&1
    fi

    # 提取 Per-Block COMPUTE+PADDED 结果
    padded_line=$(grep "Per-Block, COMPUTE+PADDED" -A 1 "$outfile" | grep "Decode time:")
    dec_time=$(echo "$padded_line" | sed 's/.*Decode time: \([0-9.]*\) ms.*/\1/')
    throughput=$(echo "$padded_line" | sed 's/.*Throughput: \([0-9.]*\) GB\/s.*/\1/')

    # 提取压缩比
    ratio=$(grep "Per-Block.*Compression:" "$outfile" | head -1 | sed 's/.*Compression: \([0-9.]*\)x.*/\1/')

    echo "  Ratio: ${ratio}x, DecTime: ${dec_time}ms, Throughput: ${throughput} GB/s" | tee -a $OUT_DIR/delta_summary.txt

    echo "$idx,$name,$dtype,$ratio,$dec_time,$throughput" >> $OUT_DIR/delta_results.csv
done

echo "" | tee -a $OUT_DIR/delta_summary.txt
echo "=== All Complete: $(date) ===" | tee -a $OUT_DIR/delta_summary.txt
