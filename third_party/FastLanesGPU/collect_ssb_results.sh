#!/bin/bash
# Collect SSB Query Performance Results
# This script runs all SSB queries and extracts timing information

set -e

OUTPUT_FILE="ssb_results.json"
TEMP_FILE=$(mktemp)

echo "======================================================================"
echo "Collecting SSB Query Performance Data"
echo "======================================================================"
echo ""

# Initialize JSON output
echo "{" > $OUTPUT_FILE
echo '  "timestamp": "'$(date -Iseconds)'",' >> $OUTPUT_FILE
echo '  "gpu": "Tesla V100-PCIE-32GB",' >> $OUTPUT_FILE
echo '  "scale_factor": 20,' >> $OUTPUT_FILE
echo '  "data_rows": 119968352,' >> $OUTPUT_FILE
echo '  "results": {' >> $OUTPUT_FILE

first_variant=true

run_variant() {
    local variant=$1
    local variant_name=$2

    if [ "$first_variant" = false ]; then
        echo "    ," >> $OUTPUT_FILE
    fi
    first_variant=false

    echo "    \"$variant_name\": {" >> $OUTPUT_FILE

    queries=(11 12 13 21 22 23 31 32 33 34 41 42 43)
    first_query=true

    for q in "${queries[@]}"; do
        # Handle crystal-opt naming (uses underscore)
        if [ "$variant" = "crystal-opt" ]; then
            executable="./$variant/src/crystal_opt_q${q}"
        else
            executable="./$variant/src/${variant}_q${q}"
        fi

        if [ ! -f "$executable" ]; then
            echo "⚠️  $executable not found, skipping..."
            continue
        fi

        echo "Running $variant_name Q${q}..."

        # Run query and extract timing info
        $executable 2>&1 > $TEMP_FILE || {
            echo "  Error running Q${q}"
            continue
        }

        # Extract query time from JSON output
        query_time=$(grep -o '"time_query":[0-9.]*' $TEMP_FILE | cut -d':' -f2 | head -1)

        # Extract total time
        total_time=$(grep -o 'Time Taken Total: [0-9.]*' $TEMP_FILE | awk '{print $4}' | head -1)

        # Extract revenue/result if available
        revenue=$(grep -o 'Revenue: [0-9]*' $TEMP_FILE | awk '{print $2}' | head -1)

        if [ -n "$query_time" ]; then
            if [ "$first_query" = false ]; then
                echo "," >> $OUTPUT_FILE
            fi
            first_query=false

            echo -n "      \"q${q}\": {" >> $OUTPUT_FILE
            echo -n "\"query_time\": $query_time" >> $OUTPUT_FILE

            if [ -n "$total_time" ]; then
                echo -n ", \"total_time\": $total_time" >> $OUTPUT_FILE
            fi

            if [ -n "$revenue" ]; then
                echo -n ", \"revenue\": $revenue" >> $OUTPUT_FILE
            fi

            echo -n "}" >> $OUTPUT_FILE

            echo "  ✓ Q${q}: ${query_time}s"
        else
            echo "  ⚠️  Q${q}: No timing data found"
        fi
    done

    echo "" >> $OUTPUT_FILE
    echo -n "    }" >> $OUTPUT_FILE
}

# Run all variants
run_variant "crystal" "crystal"
run_variant "crystal-opt" "crystal_opt"

# Close JSON
echo "" >> $OUTPUT_FILE
echo "  }" >> $OUTPUT_FILE
echo "}" >> $OUTPUT_FILE

# Clean up
rm -f $TEMP_FILE

echo ""
echo "======================================================================"
echo "Data collection completed!"
echo "Results saved to: $OUTPUT_FILE"
echo "======================================================================"

# Display summary
echo ""
echo "Summary:"
python3 -c "
import json
with open('$OUTPUT_FILE', 'r') as f:
    data = json.load(f)

for variant in data['results']:
    print(f'\n{variant}:')
    queries = data['results'][variant]
    for q in sorted(queries.keys(), key=lambda x: int(x[1:])):
        qdata = queries[q]
        print(f'  {q}: {qdata[\"query_time\"]}s')
"
