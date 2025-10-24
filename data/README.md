# Data Directory

This directory is for storing SSB (Star Schema Benchmark) data files.

## Data Generation Required

The GLECO project requires SSB benchmark data to run tests. The data must be generated using the SSB data generator tool.

### Quick Start

1. **Generate SSB data** using the official SSB dbgen tool:
   - Repository: https://github.com/eyalroz/ssb-dbgen
   - Recommended scale factor: SF=20 (120M rows)

2. **Convert to binary format**: Convert the generated `.tbl` files to columnar binary format (`.bin` files)

3. **Place data files** in: `/root/autodl-tmp/test/ssb_data/`
   - Or update the path in `tests/ssb/ssb_utils.h` (line 29)

### Expected Data Structure

```
ssb_data/
├── LINEORDER0.bin ... LINEORDER16.bin  (17 columns)
├── PART0.bin ... PART8.bin             (9 columns)
├── SUPPLIER0.bin ... SUPPLIER6.bin     (7 columns)
├── CUSTOMER0.bin ... CUSTOMER7.bin     (8 columns)
└── DDATE0.bin ... DDATE14.bin          (15 columns)
```

### Scale Factors

- **SF=1**: ~6M rows, ~600 MB (quick testing)
- **SF=10**: ~60M rows, ~6 GB (medium testing)
- **SF=20**: ~120M rows, ~12 GB (recommended, used in benchmarks)
- **SF=100**: ~600M rows, ~60 GB (large-scale)

---

**Note**: Data files are not included in the repository due to size. You must generate them yourself using the SSB data generator.

For detailed instructions, see: https://github.com/eyalroz/ssb-dbgen
