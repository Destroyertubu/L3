# Migration Guide: From Old L3 to L3 Refactored

This guide helps you migrate from the old unstructured L3 project to the new engineered L3 project.

## Overview of Changes

### Old Structure
```
L3/
├── tests/
│   ├── l32.cu
│   ├── l32_optimized.cu
│   ├── ssb_base/
│   ├── ssb_new/
│   └── ssb_ra/
├── src/
├── include/
└── scripts/
```

### New Structure
```
L3/
├── lib/
│   ├── l32/          # Main compression library
│   └── l3_legacy/    # Legacy implementations
├── include/
│   ├── common/          # Shared headers
│   ├── l32/          # L3 headers
│   └── l3_legacy/    # Legacy headers
├── benchmarks/
│   └── ssb/
│       ├── baseline/
│       └── optimized_2push/
├── docs/                # Comprehensive documentation
└── scripts/             # Build and deployment scripts
```

## Key Changes

### 1. File Organization

| Old Location | New Location | Notes |
|--------------|--------------|-------|
| `tests/l32.cu` | `lib/l32/l32.cu` | Main library |
| `tests/l32_optimized.cu` | ❌ Removed | Duplicate file |
| `tests/ssb_new/baseline/*.cu` | `benchmarks/ssb/baseline/*.cu` | Renamed |
| `tests/ssb_new/optimized_2push/*.cu` | `benchmarks/ssb/optimized_2push/*.cu` | Organized |
| `tests/ssb_base/*` | ❌ Archived | Legacy code |
| `src/*.cu` | `lib/l3_legacy/*.cu` | Legacy library |

### 2. Build System

#### Old (Manual Compilation)
```bash
nvcc -o q11 tests/ssb_new/optimized_2push/q11_2push.cu \
     -I./tests/ssb_new -I./include \
     -arch=sm_86 -O3
```

#### New (CMake)
```bash
cd L3
mkdir build && cd build
cmake ..
make q11_2push_opt
```

### 3. Include Paths

#### Old Code
```cuda
#include "../../l3_alex_index.cuh"
#include "../ssb_l3_utils.cuh"
```

#### New Code
```cuda
#include "l3_alex_index.cuh"
#include "ssb_l3_utils.cuh"
```

All headers are now in standardized include directories.

## Migration Steps

### Step 1: Backup Old Project

```bash
cd /path/to/old/L3
tar -czf L3_backup_$(date +%Y%m%d).tar.gz .
mv L3_backup_*.tar.gz ~/backups/
```

### Step 2: Download New Project

```bash
# Extract new refactored project
tar -xzf L3_*.tar.gz
cd L3_*
```

### Step 3: Migrate Custom Code

If you have custom modifications:

#### For SSB Queries

1. Copy your modified query files:
```bash
cp /path/to/old/L3/tests/ssb_new/optimized_2push/my_custom_query.cu \
   L3/benchmarks/ssb/optimized_2push/
```

2. Update include paths in the file:
```cuda
// Old
#include "../../l3_alex_index.cuh"

// New
#include "l3_alex_index.cuh"
```

3. Rebuild:
```bash
cd build
cmake ..
make
```

#### For Library Modifications

1. If you modified `l32.cu`:
```bash
cp /path/to/old/L3/tests/l32.cu \
   L3/lib/l32/l32.cu
```

2. Rebuild library:
```bash
cd build
make l32
```

### Step 4: Migrate Data Files

```bash
# Copy data files to new location
cp -r /path/to/old/L3/data/* L3/data/

# Or create symbolic link
ln -s /path/to/data L3/data
```

### Step 5: Update Scripts

If you have custom scripts that reference old paths:

```bash
# Old script
./tests/ssb_new/optimized_2push/q11_2push

# New script
./build/bin/ssb/optimized/q11_2push_opt
```

## Code Migration Examples

### Example 1: SSB Query

#### Old Version (`tests/ssb_new/optimized_2push/q11_2push.cu`)
```cuda
#include <cuda_runtime.h>
#include "../../l3_alex_index.cuh"
#include "../../ssb_l3_utils.cuh"
#include "../ssb_utils.h"

// ... code ...
```

#### New Version (`benchmarks/ssb/optimized_2push/q11_2push.cu`)
```cuda
#include <cuda_runtime.h>
#include "l3_alex_index.cuh"
#include "ssb_l3_utils.cuh"
#include "ssb_utils.h"

// ... code ... (no other changes needed)
```

### Example 2: Build Configuration

#### Old Makefile
```makefile
NVCC = nvcc
ARCH = -arch=sm_86
INC = -I./tests/ssb_new -I./include

q11: tests/ssb_new/optimized_2push/q11_2push.cu
    $(NVCC) $(ARCH) $(INC) -o $@ $<
```

#### New CMakeLists.txt
```cmake
# Already configured!
# Just run: make q11_2push_opt
```

## Compatibility Notes

### What Stays the Same

✅ Algorithm logic - no changes
✅ Data formats - fully compatible
✅ CUDA kernels - identical functionality
✅ Performance - same or better
✅ API - mostly unchanged

### What Changed

⚠️ File paths - use new directory structure
⚠️ Build system - use CMake instead of manual compilation
⚠️ Include paths - simplified and standardized
⚠️ Binary names - may have `_opt` or `_baseline` suffix
⚠️ Output locations - in `build/bin/` subdirectories

## Troubleshooting Migration Issues

### Issue: Cannot Find Header Files

**Old error:**
```
fatal error: l3_alex_index.cuh: No such file or directory
```

**Solution:**
Headers are now in `include/common/`. CMake handles this automatically.
If compiling manually, add:
```bash
-I./include/common -I./include/l32
```

### Issue: Binaries Not Where Expected

**Old location:**
```
tests/ssb_new/optimized_2push/q11_2push
```

**New location:**
```
build/bin/ssb/optimized/q11_2push_opt
```

**Solution:** Update your scripts to use new paths.

### Issue: Duplicate Symbol Errors

This happens if you link both L3 and legacy versions.

**Solution:** Choose one version in CMakeLists.txt:
```cmake
cmake .. -DUSE_L3=ON -DUSE_LEGACY=OFF
```

### Issue: Data Files Not Found

Queries may look for data in old locations.

**Solution:** Either:
1. Copy data to new location:
```bash
cp -r old/data/* L3/data/
```

2. Or use symbolic link:
```bash
ln -s /absolute/path/to/data L3/data
```

3. Or set environment variable:
```bash
export DATA_PATH=/path/to/data
```

## Performance Validation

After migration, validate that performance matches old version:

```bash
# Run benchmark suite
cd build/bin/ssb/optimized

# Test each query
for query in q1{1,2,3}_2push_opt; do
    echo "Testing $query..."
    ./$query | grep "Execution Time"
done

# Compare with old results
```

Expected: Similar or better performance due to optimized build system.

## Rollback Procedure

If you need to revert to old version:

```bash
# Extract backup
cd /path/to/project
tar -xzf ~/backups/L3_backup_*.tar.gz

# Rebuild old version
nvcc -o q11 tests/ssb_new/optimized_2push/q11_2push.cu \
     -I./tests/ssb_new -I./include -arch=sm_86
```

## Support

For migration issues:

1. Check [Installation Guide](INSTALLATION.md)
2. Review [API Documentation](API.md)
3. See [Troubleshooting](INSTALLATION.md#troubleshooting)
4. Open an issue on GitHub

## Checklist

- [ ] Backed up old project
- [ ] Extracted new project
- [ ] Installed dependencies (CUDA, CMake)
- [ ] Built project successfully
- [ ] Ran test benchmark
- [ ] Migrated custom code (if any)
- [ ] Migrated data files
- [ ] Updated scripts
- [ ] Validated performance
- [ ] Documented changes

## Next Steps

After successful migration:

1. Read [README.md](../README.md) for project overview
2. Review [Architecture](ARCHITECTURE.md) to understand new structure
3. Check [Performance Guide](PERFORMANCE.md) for optimization tips
4. Explore [Examples](../examples/) for usage patterns
