#!/bin/bash
# FastLanes Benchmark Runner - Robust Version

DATA_DIR="/root/autodl-tmp/code/L3/data/sosd"
OUTPUT_DIR="/root/autodl-tmp/code/L3/H20/SOSD/EDR/Third_Party/FLS-GPU"
BENCH_DIR="/root/autodl-tmp/code/L3/third_party/tmp/FastLanesGPU-main/fastlanes/example"

mkdir -p "$OUTPUT_DIR"

BITPACK_CSV="$OUTPUT_DIR/fastlanes_bitpack_results.csv"
DELTA_CSV="$OUTPUT_DIR/fastlanes_delta_results.csv"

rm -f "$BITPACK_CSV" "$DELTA_CSV"

echo "framework,algorithm,dataset,data_size_bytes,compressed_size_bytes,compression_ratio,encode_time_ms,decode_time_ms,encode_throughput_gbps,decode_throughput_gbps,verified" > "$BITPACK_CSV"
echo "framework,algorithm,dataset,data_size_bytes,compressed_size_bytes,compression_ratio,encode_time_ms,decode_time_ms,encode_throughput_gbps,decode_throughput_gbps,verified" > "$DELTA_CSV"

DATASETS_32=(
    "1-linear_200M_uint32.bin"
    "2-normal_200M_uint32.bin"
    "5-books_200M_uint32.bin"
    "9-movieid_uint32.bin"
    "14-cosmos_int32.bin"
    "18-site_250k_uint32.bin"
    "19-weight_25k_uint32.bin"
    "20-adult_30k_uint32.bin"
)

DATASETS_64=(
    "3-poisson_87M_uint64.bin"
    "4-ml_uint64.bin"
    "6-fb_200M_uint64.bin"
    "7-wiki_200M_uint64.bin"
    "8-osm_cellids_800M_uint64.bin"
    "10-house_price_uint64.bin"
    "11-planet_uint64.bin"
    "12-libio.bin"
    "13-medicare_part1.bin"
    "13-medicare_part2.bin"
    "13-medicare_part3.bin"
    "13-medicare_part4.bin"
    "15-polylog_10M_uint64.bin"
    "16-exp_200M_uint64.bin"
    "17-poly_200M_uint64.bin"
)

cd "$BENCH_DIR"

echo "=========================================="
echo "Running FastLanes Benchmarks"
echo "=========================================="

echo ""
echo "=== Processing uint32 datasets ==="
for dataset in "${DATASETS_32[@]}"; do
    filepath="$DATA_DIR/$dataset"
    [ ! -f "$filepath" ] && echo "Skipping: $dataset" && continue

    echo "Processing: $dataset"

    # BitPack
    output=$(./fastlanes_bench_bitpack --file "$filepath" 2>&1)
    file_size=$(echo "$output" | grep "File size:" | awk '{print $4}')
    comp_ratio=$(echo "$output" | grep "Compression ratio:" | awk '{print $4}' | tr -d 'x')
    enc_time=$(echo "$output" | grep "Encode time:" | awk '{print $4}')
    enc_tp=$(echo "$output" | grep "Encode throughput:" | awk '{print $4}')
    dec_time=$(echo "$output" | grep "Decode time:" | awk '{print $4}')
    dec_tp=$(echo "$output" | grep "Decode throughput:" | awk '{print $4}')
    comp_size=$(python3 -c "print(int($file_size / $comp_ratio))" 2>/dev/null || echo "0")
    verified="true"
    echo "$output" | grep -q "FAILED" && verified="false"
    echo "FastLanes,BitPack-uint32,$dataset,$file_size,$comp_size,$comp_ratio,$enc_time,$dec_time,$enc_tp,$dec_tp,$verified" >> "$BITPACK_CSV"

    # Delta
    output=$(./fastlanes_bench_delta --file "$filepath" 2>&1)
    orig_mb=$(echo "$output" | grep "Original size:" | head -1 | awk '{print $3}')
    orig_size=$(python3 -c "print(int($orig_mb * 1024 * 1024))" 2>/dev/null || echo "0")

    pb_line=$(echo "$output" | grep "\[Per-Block\] Compressed data:")
    total_mb=$(echo "$pb_line" | sed 's/.*Total: \([0-9.]*\) MB.*/\1/')
    comp_size=$(python3 -c "print(int($total_mb * 1024 * 1024))" 2>/dev/null || echo "0")
    comp_ratio=$(echo "$pb_line" | sed 's/.*Ratio: \([0-9.]*\)x.*/\1/')

    enc_time=$(echo "$output" | grep "\[Per-Block\] Encode time:" | awk '{print $5}')
    enc_tp=$(echo "$output" | grep "\[Per-Block\] Encode throughput:" | awk '{print $5}')

    dec_line=$(echo "$output" | grep "Per-Block BW, OPTIMIZED transpose" -A1 | tail -1)
    dec_time=$(echo "$dec_line" | sed 's/.*Decode time: \([0-9.]*\) ms.*/\1/')
    dec_tp=$(echo "$dec_line" | sed 's/.*Throughput: \([0-9.]*\) GB\/s.*/\1/')

    verified="true"
    echo "$output" | grep "Per-Block BW, OPTIMIZED" -A2 | grep -q "FAILED" && verified="false"
    echo "FastLanes,Delta-PerBlock-Opt-uint32,$dataset,$orig_size,$comp_size,$comp_ratio,$enc_time,$dec_time,$enc_tp,$dec_tp,$verified" >> "$DELTA_CSV"

    echo "  Done."
done

echo ""
echo "=== Processing uint64 datasets ==="
for dataset in "${DATASETS_64[@]}"; do
    filepath="$DATA_DIR/$dataset"
    [ ! -f "$filepath" ] && echo "Skipping: $dataset" && continue

    echo "Processing: $dataset"

    # BitPack
    output=$(./fastlanes_bench_bitpack --file "$filepath" --uint64 2>&1)
    file_size=$(echo "$output" | grep "File size:" | awk '{print $4}')
    comp_ratio=$(echo "$output" | grep "Compression ratio:" | awk '{print $4}' | tr -d 'x')
    enc_time=$(echo "$output" | grep "Encode time:" | awk '{print $4}')
    enc_tp=$(echo "$output" | grep "Encode throughput:" | awk '{print $4}')
    dec_time=$(echo "$output" | grep "Decode time:" | awk '{print $4}')
    dec_tp=$(echo "$output" | grep "Decode throughput:" | awk '{print $4}')
    comp_size=$(python3 -c "print(int($file_size / $comp_ratio))" 2>/dev/null || echo "0")
    verified="true"
    echo "$output" | grep -q "FAILED" && verified="false"
    echo "FastLanes,BitPack-uint64,$dataset,$file_size,$comp_size,$comp_ratio,$enc_time,$dec_time,$enc_tp,$dec_tp,$verified" >> "$BITPACK_CSV"

    # Delta uint64
    output=$(./fastlanes_bench_delta --file "$filepath" --uint64 2>&1)
    orig_mb=$(echo "$output" | grep "Original size:" | head -1 | awk '{print $3}')
    orig_size=$(python3 -c "print(int($orig_mb * 1024 * 1024))" 2>/dev/null || echo "0")

    comp_mb=$(echo "$output" | grep "Compressed data size:" | awk '{print $4}')
    comp_size=$(python3 -c "print(int($comp_mb * 1024 * 1024))" 2>/dev/null || echo "0")
    comp_ratio=$(echo "$output" | grep "Compression ratio:" | head -1 | awk '{print $3}' | tr -d 'x')

    enc_time=$(echo "$output" | grep "Encode time:" | head -1 | awk '{print $4}')
    enc_tp=$(echo "$output" | grep "Encode throughput:" | head -1 | awk '{print $4}')

    dec_section=$(echo "$output" | grep -A5 "NO inverse transpose (uint64)")
    dec_time=$(echo "$dec_section" | grep "Decode time (avg" | sed 's/.*): \([0-9.]*\) ms/\1/')
    dec_tp=$(echo "$dec_section" | grep "Decode throughput:" | head -1 | awk '{print $4}')

    verified="true"
    echo "$dec_section" | grep -q "FAILED" && verified="false"
    echo "FastLanes,Delta-NoTranspose-uint64,$dataset,$orig_size,$comp_size,$comp_ratio,$enc_time,$dec_time,$enc_tp,$dec_tp,$verified" >> "$DELTA_CSV"

    echo "  Done."
done

echo ""
echo "=========================================="
echo "Results saved to:"
echo "  $BITPACK_CSV"
echo "  $DELTA_CSV"
wc -l "$BITPACK_CSV" "$DELTA_CSV"

echo ""
echo "=== Sample Results ==="
echo "BitPack:"
head -5 "$BITPACK_CSV"
echo ""
echo "Delta:"
head -5 "$DELTA_CSV"
