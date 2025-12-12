# Quick Start: Running FastLanesGPU with L3 Data

## Data Location

L3 SSB data is located at:
```
/root/autodl-tmp/code/data/SSB/L3/ssb_data/
```

Scale Factor: **SF=20** (119M rows in LINEORDER table)

## Running Queries

### Using the Test Script

```bash
# Run a single query
./run_ssb_tests.sh crystal 11

# Run all queries for a variant
./run_ssb_tests.sh crystal-opt all

# Run all variants for a query
./run_ssb_tests.sh all 21

# Run everything
./run_ssb_tests.sh all all
```

### Running Directly

```bash
# Crystal version
./crystal/src/crystal_q11

# Crystal-Opt version (optimized)
./crystal-opt/src/crystal_opt_q11

# FastLanes version
./fastlanes/src/fls_q11
```

## Available Queries

All SSB queries are available: **q11, q12, q13, q21, q22, q23, q31, q32, q33, q34, q41, q42, q43**

## Example Output

```
Using device 0: Tesla V100-PCIE-32GB
** LOADED DATA **
LO_LEN 119968352
** LOADED DATA TO GPU **
Revenue: 22660807639355
Time Taken Total: 2.95903
{"query":11,"time_query":2.73203}
```

## Performance (Tesla V100)

- **Q11 Crystal**: ~2.7 seconds
- **Q11 Crystal-Opt**: ~1.6 seconds
- **Data Loading**: ~0.2-0.3 seconds

## Query Descriptions

- **Q11-Q13**: Selection queries (lo_orderdate, lo_discount, lo_quantity filters)
- **Q21-Q23**: Aggregation with joins (LINEORDER + PART + SUPPLIER + DATE)
- **Q31-Q34**: Complex joins (LINEORDER + CUSTOMER + SUPPLIER + DATE)
- **Q41-Q43**: Profit queries (all tables joined)

## Rebuilding (if needed)

```bash
make clean
make -j8
```

## Troubleshooting

If you encounter file not found errors:
1. Check that data files exist: `ls /root/autodl-tmp/code/data/SSB/L3/ssb_data/*.bin`
2. Verify permissions: `chmod 644 /root/autodl-tmp/code/data/SSB/L3/ssb_data/*.bin`
3. Rebuild the project: `make clean && make -j8`

For detailed information, see `L3_DATA_INTEGRATION.md`
