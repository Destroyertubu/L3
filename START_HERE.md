# 🚀 Start Here

Welcome to L3 - GPU-Accelerated Learned Compression Library!

## Quick Navigation

### For Users

1. **[README.md](README.md)** - Project overview and features (5 min)
2. **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** - Installation and basic usage (10 min)
3. **[docs/PARTITION_STRATEGIES.md](docs/PARTITION_STRATEGIES.md)** - Choosing partition strategies (15 min)
4. **[examples/](examples/)** - Working code examples

### For Developers

1. **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** - Development guide and roadmap
2. **[docs/MIGRATION.md](docs/MIGRATION.md)** - Migration status
3. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture

## What is L3?

L3 is a high-performance GPU compression library that uses **learned models** to achieve:
- **40+ GB/s** decompression throughput
- **3.5-4.5x** compression ratio
- **Flexible partitioning** strategies

## Key Feature: Partition Strategies

L3 lets you choose how to partition your data:

```cpp
// Fixed-size partitions (simple, fast)
l3::FixedSizePartitioner fixed(4096);

// Variable-length partitions (adaptive, better compression)
l3::VariableLengthPartitioner adaptive(1024, 8, 3);

// Auto-select best strategy
auto partitioner = l3::PartitionerFactory::createAuto(data, size);
```

## Quick Start

```bash
# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=ON
make -j$(nproc)

# Run example
./bin/examples/example_partition_strategies
```

## Project Status

**Phase 1 Complete**: Core partition interface implemented ✅
**Progress**: 15% - See [docs/MIGRATION.md](docs/MIGRATION.md) for details

## Documentation Structure

```
├── README.md                      # Project overview
├── START_HERE.md                  # This file
└── docs/
    ├── GETTING_STARTED.md         # Installation & usage
    ├── PARTITION_STRATEGIES.md    # Strategy guide
    ├── DEVELOPMENT.md             # Developer guide
    ├── MIGRATION.md               # Migration status
    └── ARCHITECTURE.md            # System design
```

## Need Help?

- **Getting Started**: [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
- **Environment Setup**: [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)
- **Examples**: [examples/cpp/](examples/cpp/)
- **Issues**: GitHub Issues
- **Development**: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)

---

**Recommended First Steps**:
1. Read [README.md](README.md)
2. Try [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)
3. Run examples in [examples/](examples/)
