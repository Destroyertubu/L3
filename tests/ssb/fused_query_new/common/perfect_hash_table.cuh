/**
 * @file perfect_hash_table.cuh
 * @brief Perfect Hash Table (PHT) implementation for SSB queries
 *
 * Adapted from FSL-GPU/Crystal-opt join.cuh
 * PHT_1: Keys only (existence check)
 * PHT_2: Key-value pairs (read as uint64_t for atomic read)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Hash Function (from crystal-opt)
// ============================================================================

#define HASH(X, Y, Z) ((X - Z) % Y)  // key, table_size, min_key

// ============================================================================
// PHT_1: Probe for Existence (Keys Only)
// ============================================================================

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K (&items)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(items[ITEM], ht_len, keys_min);
            K slot = ht[hash];
            if (slot != 0) {
                selection_flags[ITEM] = 1;
            } else {
                selection_flags[ITEM] = 0;
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(
    int tid,
    K (&items)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(items[ITEM], ht_len, keys_min);
                K slot = ht[hash];
                if (slot != 0) {
                    selection_flags[ITEM] = 1;
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K (&items)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, items, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, items, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

// Convenience overload with keys_min = 0
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(
    K (&items)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items) {
    BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, selection_flags, ht, ht_len, 0, num_items);
}

// ============================================================================
// PHT_2: Probe and Get Value (Key-Value Pairs)
// Layout: [key0, val0, key1, val1, ...] - interleaved
// Read as uint64_t for atomic access
// ============================================================================

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    V (&res)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(keys[ITEM], ht_len, keys_min);
            // Read key-value pair atomically as uint64_t
            uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
            if (slot != 0) {
                res[ITEM] = (slot >> 32);  // Value is in upper 32 bits
            } else {
                selection_flags[ITEM] = 0;
            }
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    V (&res)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(keys[ITEM], ht_len, keys_min);
                uint64_t slot = *reinterpret_cast<uint64_t*>(&ht[hash << 1]);
                if (slot != 0) {
                    res[ITEM] = (slot >> 32);
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K (&keys)[ITEMS_PER_THREAD],
    V (&res)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

// Convenience overload with keys_min = 0
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(
    K (&keys)[ITEMS_PER_THREAD],
    V (&res)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    int num_items) {
    BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
        keys, res, selection_flags, ht, ht_len, 0, num_items);
}

// ============================================================================
// PHT Build Operations (Block-level)
// ============================================================================

// Build PHT_1: Insert keys only (with selection)
template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(keys[ITEM], ht_len, keys_min);
            K old = atomicCAS(&ht[hash], 0, keys[ITEM]);
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(keys[ITEM], ht_len, keys_min);
                K old = atomicCAS(&ht[hash], 0, keys[ITEM]);
            }
        }
    }
}

template<typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(
    K (&keys)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

// Build PHT_2: Insert key-value pairs (with selection)
template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    V (&vals)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            int hash = HASH(keys[ITEM], ht_len, keys_min);
            K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
            ht[(hash << 1) + 1] = vals[ITEM];
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(
    int tid,
    K (&keys)[ITEMS_PER_THREAD],
    V (&vals)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                int hash = HASH(keys[ITEM], ht_len, keys_min);
                K old = atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
                ht[(hash << 1) + 1] = vals[ITEM];
            }
        }
    }
}

template<typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(
    K (&keys)[ITEMS_PER_THREAD],
    V (&vals)[ITEMS_PER_THREAD],
    int (&selection_flags)[ITEMS_PER_THREAD],
    K* ht,
    int ht_len,
    K keys_min,
    int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, vals, selection_flags, ht, ht_len, keys_min);
    } else {
        BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(
            threadIdx.x, keys, vals, selection_flags, ht, ht_len, keys_min, num_items);
    }
}

// ============================================================================
// Host-side PHT Build Kernels
// ============================================================================

// Build PHT_1 from dimension table with filter
template<typename K>
__global__ void build_pht_1_kernel(
    const K* __restrict__ keys,
    const int* __restrict__ filter_col,
    int filter_val,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        if (filter_col[tid] == filter_val) {
            K key = keys[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash], 0, key);
        }
    }
}

// Build PHT_1 from dimension table with range filter
template<typename K>
__global__ void build_pht_1_range_kernel(
    const K* __restrict__ keys,
    const int* __restrict__ filter_col,
    int filter_min,
    int filter_max,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        int val = filter_col[tid];
        if (val >= filter_min && val <= filter_max) {
            K key = keys[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash], 0, key);
        }
    }
}

// Build PHT_2 from dimension table with filter
template<typename K, typename V>
__global__ void build_pht_2_kernel(
    const K* __restrict__ keys,
    const V* __restrict__ vals,
    const int* __restrict__ filter_col,
    int filter_val,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        if (filter_col[tid] == filter_val) {
            K key = keys[tid];
            V val = vals[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash << 1], 0, key);
            ht[(hash << 1) + 1] = val;
        }
    }
}

// Build PHT_2 from dimension table with range filter (returns values)
template<typename K, typename V>
__global__ void build_pht_2_range_kernel(
    const K* __restrict__ keys,
    const V* __restrict__ vals,
    const int* __restrict__ filter_col,
    int filter_min,
    int filter_max,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        int fval = filter_col[tid];
        if (fval >= filter_min && fval <= filter_max) {
            K key = keys[tid];
            V val = vals[tid];
            int hash = HASH(key, ht_len, keys_min);
            atomicCAS(&ht[hash << 1], 0, key);
            ht[(hash << 1) + 1] = val;
        }
    }
}

// Build PHT_2 from dimension table (no filter, returns value column)
template<typename K, typename V>
__global__ void build_pht_2_all_kernel(
    const K* __restrict__ keys,
    const V* __restrict__ vals,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        K key = keys[tid];
        V val = vals[tid];
        int hash = HASH(key, ht_len, keys_min);
        atomicCAS(&ht[hash << 1], 0, key);
        ht[(hash << 1) + 1] = val;
    }
}

// Build PHT_1 from dimension table (no filter)
template<typename K>
__global__ void build_pht_1_all_kernel(
    const K* __restrict__ keys,
    int num_rows,
    K* ht,
    int ht_len,
    K keys_min) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_rows) {
        K key = keys[tid];
        int hash = HASH(key, ht_len, keys_min);
        atomicCAS(&ht[hash], 0, key);
    }
}
