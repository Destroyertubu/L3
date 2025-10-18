# L3 Quick Start Guide

Get up and running with L3 in 5 minutes!

## Prerequisites Check

Run the verification script:

```bash
./scripts/verify.sh
```

If all checks pass ✓, proceed to build!

## Build (One Command)

```bash
./scripts/build.sh
```

This will:
- Auto-detect your GPU
- Configure CMake
- Compile all components
- Takes 2-5 minutes

## Test

```bash
cd build/bin/ssb/optimized
./q11_2push_opt
```

Expected output:
```
Loading data...
Compressing...
Running query...
Execution Time: XX.XX ms
Compression Ratio: XX.Xx
Results verified ✓
```

## What's Next?

### Run More Benchmarks

```bash
# Run all SSB queries
for q in q1{1,2,3}_2push_opt q2{1,2,3}_2push_opt; do
    ./$q
done
```

### Explore Documentation

```
docs/
├── INSTALLATION.md     # Detailed setup
├── ARCHITECTURE.md     # How it works
├── MIGRATION.md        # Migrate from old project
└── README.md          # Doc index
```

### Customize Build

```bash
# Debug build
./scripts/build.sh -t Debug

# Specific GPU architecture
./scripts/build.sh -a 86  # RTX 30xx

# Parallel build with 8 jobs
./scripts/build.sh -j 8

# Clean rebuild
./scripts/build.sh -c
```

## Deploy to Another Machine

### On Source Machine

```bash
./scripts/deploy.sh
```

This creates `L3_YYYYMMDD_HHMMSS.tar.gz`

### On Target Machine

```bash
# Copy the archive
scp L3_*.tar.gz target:/path/

# On target machine
tar -xzf L3_*.tar.gz
cd L3_*
./scripts/build.sh
```

## Common Issues

### CUDA Not Found
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
./scripts/build.sh
```

### Wrong GPU Architecture
```bash
# Check your GPU
nvidia-smi --query-gpu=compute_cap --format=csv

# Build for your GPU
./scripts/build.sh -a 86  # Use your compute capability
```

### Build Errors
```bash
# Clean and rebuild
rm -rf build
./scripts/build.sh -c
```

## Project Structure

```
L3/
├── build/              # Build outputs (generated)
│   ├── bin/           # Executables
│   │   └── ssb/       # SSB benchmarks
│   └── lib/           # Libraries
├── docs/              # Documentation
├── scripts/           # Helper scripts
│   ├── build.sh       # Main build script
│   ├── deploy.sh      # Package for deployment
│   └── verify.sh      # Verify setup
└── [source files...]
```

## Getting Help

1. **Documentation**: Read `docs/` directory
2. **Troubleshooting**: See `docs/INSTALLATION.md#troubleshooting`
3. **Architecture**: See `docs/ARCHITECTURE.md`
4. **Migration**: See `docs/MIGRATION.md`

## Performance Tips

1. **Use Release Build** (default)
   ```bash
   ./scripts/build.sh -t Release
   ```

2. **Match GPU Architecture**
   ```bash
   ./scripts/build.sh -a $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.')
   ```

3. **Check GPU Utilization**
   ```bash
   nvidia-smi dmon  # Monitor during execution
   ```

## Next Steps

- [ ] Build project
- [ ] Run test benchmark
- [ ] Review documentation
- [ ] Try custom queries
- [ ] Deploy to production machine

## Support

For issues:
1. Check `docs/INSTALLATION.md`
2. Review verification output
3. Search existing issues
4. Create detailed bug report

---

**Ready to build?**

```bash
./scripts/build.sh
```

**Questions?**

Read the full documentation in `docs/` or see `README.md`!
