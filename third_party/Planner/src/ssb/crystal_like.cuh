// A minimal subset of Crystal-style block primitives (load/pred/join/reduce).
// Used by the Planner baseline to run the same SSB query plans as in ssb/guide.txt.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// ----------------------------
// Load
// ----------------------------

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(const unsigned int tid, const T* block_itr, T (&items)[ITEMS_PER_THREAD]) {
    const T* thread_itr = block_itr + tid;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
    }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockLoadDirect(const unsigned int tid, const T* block_itr, T (&items)[ITEMS_PER_THREAD], int num_items) {
    const T* thread_itr = block_itr + tid;
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            items[ITEM] = thread_itr[ITEM * BLOCK_THREADS];
        }
    }
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(const T* inp, T (&items)[ITEMS_PER_THREAD], int num_items) {
    const T* block_itr = inp;
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items);
    } else {
        BlockLoadDirect<T, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, num_items);
    }
}

// ----------------------------
// Predicates
// ----------------------------

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = 1;
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredDirect(int tid, const T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = select_op(items[ITEM]);
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(int tid,
                                                const T (&items)[ITEMS_PER_THREAD],
                                                SelectOp select_op,
                                                int (&selection_flags)[ITEMS_PER_THREAD],
                                                int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = select_op(items[ITEM]);
        }
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPred(const T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(int tid,
                                                   const T (&items)[ITEMS_PER_THREAD],
                                                   SelectOp select_op,
                                                   int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(int tid,
                                                   const T (&items)[ITEMS_PER_THREAD],
                                                   SelectOp select_op,
                                                   int (&selection_flags)[ITEMS_PER_THREAD],
                                                   int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
        }
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAnd(const T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredAndDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(int tid,
                                                  const T (&items)[ITEMS_PER_THREAD],
                                                  SelectOp select_op,
                                                  int (&selection_flags)[ITEMS_PER_THREAD]) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(int tid,
                                                  const T (&items)[ITEMS_PER_THREAD],
                                                  SelectOp select_op,
                                                  int (&selection_flags)[ITEMS_PER_THREAD],
                                                  int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
        }
    }
}

template <typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredOr(const T (&items)[ITEMS_PER_THREAD], SelectOp select_op, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    if ((BLOCK_THREADS * ITEMS_PER_THREAD) == num_items) {
        BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredOrDirect<T, SelectOp, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

template <typename T>
struct LessThan {
    T compare;
    __device__ __forceinline__ explicit LessThan(T c) : compare(c) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a < compare; }
};

template <typename T>
struct GreaterThan {
    T compare;
    __device__ __forceinline__ explicit GreaterThan(T c) : compare(c) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a > compare; }
};

template <typename T>
struct GreaterThanEq {
    T compare;
    __device__ __forceinline__ explicit GreaterThanEq(T c) : compare(c) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a >= compare; }
};

template <typename T>
struct LessThanEq {
    T compare;
    __device__ __forceinline__ explicit LessThanEq(T c) : compare(c) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a <= compare; }
};

template <typename T>
struct Eq {
    T compare;
    __device__ __forceinline__ explicit Eq(T c) : compare(c) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a == compare; }
};

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredEQ(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, Eq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredOrEQ(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredOr<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, Eq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredGTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, GreaterThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndGTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, GreaterThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndLTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredGT(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, GreaterThan<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLT(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThan<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndLT(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThan<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndGT(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, GreaterThan<T>(compare), selection_flags, num_items);
}

// ----------------------------
// Reduce
// ----------------------------

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T value, T* shared) {
    static_assert(BLOCK_THREADS == 32, "BlockSum assumes 1 warp (32 threads).");
    const int lane = static_cast<int>(threadIdx.x) & 31;
    shared[lane] = value;
    __syncwarp();

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        shared[lane] += __shfl_down_sync(0xffffffff, shared[lane], offset);
    }
    return static_cast<T>(shared[0]);
}

// ----------------------------
// Perfect Hash Table (PHT) build/probe
// ----------------------------

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         int* ht,
                                                         int ht_len,
                                                         int key_min,
                                                         int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) {
            if (selection_flags[ITEM]) {
                const int slot = static_cast<int>(keys[ITEM] - key_min);
                if (slot >= 0 && slot < ht_len) {
                    ht[slot] = static_cast<int>(keys[ITEM]);
                }
            }
        }
    }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         int* ht,
                                                         int ht_len,
                                                         int num_items) {
    BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht, ht_len, 0, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                         V (&vals)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         int* ht,
                                                         int ht_len,
                                                         int key_min,
                                                         int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) {
            if (selection_flags[ITEM]) {
                const int slot = static_cast<int>(keys[ITEM] - key_min);
                if (slot >= 0 && slot < ht_len) {
                    ht[slot * 2]     = static_cast<int>(keys[ITEM]);
                    ht[slot * 2 + 1] = static_cast<int>(vals[ITEM]);
                }
            }
        }
    }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                         V (&vals)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         int* ht,
                                                         int ht_len,
                                                         int num_items) {
    BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht, ht_len, 0, num_items);
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockProbeAndPHT_1(K (&keys)[ITEMS_PER_THREAD], int (&selection_flags)[ITEMS_PER_THREAD], const int* ht, int ht_len, int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) {
            if (selection_flags[ITEM]) {
                const int slot = static_cast<int>(keys[ITEM]);
                selection_flags[ITEM] = (slot >= 0 && slot < ht_len && ht[slot] != 0);
            }
        }
    }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                   V (&vals)[ITEMS_PER_THREAD],
                                                   int (&selection_flags)[ITEMS_PER_THREAD],
                                                   const int* ht,
                                                   int ht_len,
                                                   int key_min,
                                                   int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if ((threadIdx.x + (BLOCK_THREADS * ITEM)) < num_items) {
            if (selection_flags[ITEM]) {
                const int slot = static_cast<int>(keys[ITEM] - key_min);
                if (slot >= 0 && slot < ht_len) {
                    const int k = ht[slot * 2];
                    selection_flags[ITEM] = (k != 0);
                    vals[ITEM] = ht[slot * 2 + 1];
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                   V (&vals)[ITEMS_PER_THREAD],
                                                   int (&selection_flags)[ITEMS_PER_THREAD],
                                                   const int* ht,
                                                   int ht_len,
                                                   int num_items) {
    BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht, ht_len, 0, num_items);
}

