// A minimal subset of Crystal-style block primitives (load/pred/join/reduce)
// to implement the same SSB query plans as in the ssb-fslgpu survey.

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
BlockPredAndGTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, GreaterThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredLTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThanEq<T>(compare), selection_flags, num_items);
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockPredAndLTE(const T (&items)[ITEMS_PER_THREAD], T compare, int (&selection_flags)[ITEMS_PER_THREAD], int num_items) {
    BlockPredAnd<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(items, LessThanEq<T>(compare), selection_flags, num_items);
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

// ----------------------------
// Join (Perfect Hash Table)
// ----------------------------

#define HASH(X, Y, Z) (((X) - (Z)) % (Y))

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_1(int tid,
                                                         K (&items)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         const K* ht,
                                                         int ht_len,
                                                         K keys_min,
                                                         int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                const int hash = HASH(items[ITEM], ht_len, keys_min);
                const K   slot = ht[hash];
                selection_flags[ITEM] = (slot != 0) ? 1 : 0;
            }
        }
    }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_1(K (&items)[ITEMS_PER_THREAD],
                                                   int (&selection_flags)[ITEMS_PER_THREAD],
                                                   const K* ht,
                                                   int ht_len,
                                                   K keys_min,
                                                   int num_items) {
    BlockProbeDirectAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, items, selection_flags, ht, ht_len, keys_min, num_items);
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockProbeAndPHT_1(K (&items)[ITEMS_PER_THREAD], int (&selection_flags)[ITEMS_PER_THREAD], const K* ht, int ht_len, int num_items) {
    BlockProbeAndPHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(items, selection_flags, ht, ht_len, 0, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeDirectAndPHT_2(int tid,
                                                         K (&keys)[ITEMS_PER_THREAD],
                                                         V (&res)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         const K* ht,
                                                         int ht_len,
                                                         K keys_min,
                                                         int num_items) {
#pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * BLOCK_THREADS) < num_items) {
            if (selection_flags[ITEM]) {
                const int hash = HASH(keys[ITEM], ht_len, keys_min);
                const uint64_t slot = *reinterpret_cast<const uint64_t*>(&ht[hash << 1]);
                if (slot != 0) {
                    res[ITEM] = static_cast<V>(slot >> 32);
                } else {
                    selection_flags[ITEM] = 0;
                }
            }
        }
    }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                   V (&res)[ITEMS_PER_THREAD],
                                                   int (&selection_flags)[ITEMS_PER_THREAD],
                                                   const K* ht,
                                                   int ht_len,
                                                   K keys_min,
                                                   int num_items) {
    BlockProbeDirectAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, res, selection_flags, ht, ht_len, keys_min, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void
BlockProbeAndPHT_2(K (&keys)[ITEMS_PER_THREAD], V (&res)[ITEMS_PER_THREAD], int (&selection_flags)[ITEMS_PER_THREAD], const K* ht, int ht_len, int num_items) {
    BlockProbeAndPHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, res, selection_flags, ht, ht_len, 0, num_items);
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_1(int tid,
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
                const int hash = HASH(keys[ITEM], ht_len, keys_min);
                atomicCAS(&ht[hash], 0, keys[ITEM]);
            }
        }
    }
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         K* ht,
                                                         int ht_len,
                                                         K keys_min,
                                                         int num_items) {
    BlockBuildDirectSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, selection_flags, ht, ht_len, keys_min, num_items);
}

template <typename K, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_1(K (&keys)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         K* ht,
                                                         int ht_len,
                                                         int num_items) {
    BlockBuildSelectivePHT_1<K, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, selection_flags, ht, ht_len, 0, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildDirectSelectivePHT_2(int tid,
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
                const int hash = HASH(keys[ITEM], ht_len, keys_min);
                atomicCAS(&ht[hash << 1], 0, keys[ITEM]);
                ht[(hash << 1) + 1] = static_cast<K>(vals[ITEM]);
            }
        }
    }
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                         V (&vals)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         K* ht,
                                                         int ht_len,
                                                         K keys_min,
                                                         int num_items) {
    BlockBuildDirectSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(threadIdx.x, keys, vals, selection_flags, ht, ht_len, keys_min, num_items);
}

template <typename K, typename V, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockBuildSelectivePHT_2(K (&keys)[ITEMS_PER_THREAD],
                                                         V (&vals)[ITEMS_PER_THREAD],
                                                         int (&selection_flags)[ITEMS_PER_THREAD],
                                                         K* ht,
                                                         int ht_len,
                                                         int num_items) {
    BlockBuildSelectivePHT_2<K, V, BLOCK_THREADS, ITEMS_PER_THREAD>(keys, vals, selection_flags, ht, ht_len, 0, num_items);
}

// ----------------------------
// Reduce
// ----------------------------

template <typename T>
__device__ __forceinline__ T WarpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T item, T* shared) {
    __syncthreads();

    T         val       = item;
    const int warp_size = 32;
    const int lane      = threadIdx.x % warp_size;
    const int wid       = threadIdx.x / warp_size;

    val = WarpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < (BLOCK_THREADS / warp_size)) ? shared[lane] : 0;
    if (wid == 0) {
        val = WarpReduceSum(val);
    }
    return val;
}

