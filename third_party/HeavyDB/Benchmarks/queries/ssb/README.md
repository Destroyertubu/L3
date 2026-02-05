# SSB (Star Schema Benchmark) for HeavyDB/OmniSci

This directory contains the 13 SSB queries used in our L3 comparison setup.
All predicates assume the same **integer encoding** as the L3-generated SSB binary data
in `/home/xiayouyang/code/test/ssb_data/` (e.g., `s_region=1` for AMERICA, `c_nation=24` for UNITED STATES).

## Load data (from L3 *.bin columns)

Use `Benchmarks/scripts/ssb_load_from_l3_bins.py` to create tables and ingest from `*.bin` files:

```bash
python3 "HeavyDB/Benchmarks/scripts/ssb_load_from_l3_bins.py" \
  --host localhost --port 6274 --db mapd --user mapd --password HyperInteractive \
  --data-dir "/home/xiayouyang/code/test/ssb_data" \
  --chunk-rows 1000000
```

Tables created: `dates`, `supplier`, `customer`, `part`, `lineorder`.

## Run benchmark

Run the built-in benchmark runner against these query files:

```bash
python3 "HeavyDB/Benchmarks/run_benchmark.py" \
  --user mapd --passwd HyperInteractive --server localhost --port 6274 --name mapd \
  --table lineorder --label SSB --queries-dir "HeavyDB/Benchmarks/queries/ssb" \
  --iterations 6 --destination output
```

