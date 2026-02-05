/**
 * @file crystal_ssb.cuh
 * @brief Crystal-opt style block operations for SSB queries
 *
 * Adapted from FSL-GPU/Crystal-opt for L3 fused decompression kernels.
 * Contains: InitFlags, BlockPred*, BlockSum, IsTerm
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Configuration (matches FSL-GPU/Crystal-opt)
// ============================================================================

// Use constexpr instead of macros to avoid template parameter name conflicts
constexpr int BLOCK_THREADS = 128;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;  // 512

// ============================================================================
// Flag Initialization
// ============================================================================

template<int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(int (&selection_flags)[_ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = 1;
    }
}

// ============================================================================
// Early Termination Check
// ============================================================================

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ bool IsTerm(int (&selection_flags)[_ITEMS_PER_THREAD]) {
    int count = 0;
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        count += selection_flags[ITEM];
    }
    return count == 0;
}

// ============================================================================
// Predicate Functors
// ============================================================================

template<typename T>
struct LessThan {
    T compare;
    __device__ __forceinline__ LessThan(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T &a) const { return (a < compare); }
};

template<typename T>
struct GreaterThan {
    T compare;
    __device__ __forceinline__ GreaterThan(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T &a) const { return (a > compare); }
};

template<typename T>
struct LessThanEq {
    T compare;
    __device__ __forceinline__ LessThanEq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T &a) const { return (a <= compare); }
};

template<typename T>
struct GreaterThanEq {
    T compare;
    __device__ __forceinline__ GreaterThanEq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T &a) const { return (a >= compare); }
};

template<typename T>
struct Eq {
    T compare;
    __device__ __forceinline__ Eq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T &a) const { return (a == compare); }
};

// ============================================================================
// Block Predicate Operations (Direct)
// ============================================================================

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = select_op(items[ITEM]);
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * _BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = select_op(items[ITEM]);
        }
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPred(
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    if ((_BLOCK_THREADS * _ITEMS_PER_THREAD) == num_items) {
        BlockPredDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

// ============================================================================
// Block Predicate AND Operations
// ============================================================================

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * _BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
        }
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAnd(
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    if ((_BLOCK_THREADS * _ITEMS_PER_THREAD) == num_items) {
        BlockPredAndDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredAndDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

// ============================================================================
// Block Predicate OR Operations
// ============================================================================

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrDirect(
    int tid,
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * _BLOCK_THREADS) < num_items) {
            selection_flags[ITEM] = selection_flags[ITEM] || select_op(items[ITEM]);
        }
    }
}

template<typename T, typename SelectOp, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOr(
    T (&items)[_ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[_ITEMS_PER_THREAD],
    int num_items) {
    if ((_BLOCK_THREADS * _ITEMS_PER_THREAD) == num_items) {
        BlockPredOrDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags);
    } else {
        BlockPredOrDirect<T, SelectOp, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, items, select_op, selection_flags, num_items);
    }
}

// ============================================================================
// Convenience Predicate Functions
// ============================================================================

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLT(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    LessThan<T> select_op(compare);
    BlockPred<T, LessThan<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLT(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    LessThan<T> select_op(compare);
    BlockPredAnd<T, LessThan<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGT(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    GreaterThan<T> select_op(compare);
    BlockPred<T, GreaterThan<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGT(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    GreaterThan<T> select_op(compare);
    BlockPredAnd<T, GreaterThan<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLTE(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    LessThanEq<T> select_op(compare);
    BlockPred<T, LessThanEq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTE(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    LessThanEq<T> select_op(compare);
    BlockPredAnd<T, LessThanEq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGTE(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    GreaterThanEq<T> select_op(compare);
    BlockPred<T, GreaterThanEq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGTE(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    GreaterThanEq<T> select_op(compare);
    BlockPredAnd<T, GreaterThanEq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredEQ(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    Eq<T> select_op(compare);
    BlockPred<T, Eq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndEQ(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    Eq<T> select_op(compare);
    BlockPredAnd<T, Eq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredOrEQ(
    T (&items)[_ITEMS_PER_THREAD], T compare,
    int (&selection_flags)[_ITEMS_PER_THREAD], int num_items) {
    Eq<T> select_op(compare);
    BlockPredOr<T, Eq<T>, _BLOCK_THREADS, _ITEMS_PER_THREAD>(items, select_op, selection_flags, num_items);
}

// ============================================================================
// Block Sum (Warp Shuffle Based Reduction)
// ============================================================================

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T item, T* shared) {
    __syncthreads();

    T val = item;
    const int warp_size = 32;
    int lane = threadIdx.x % warp_size;
    int wid = threadIdx.x / warp_size;

    // Calculate sum across warp
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Store sum in buffer
    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    // Load the sums into the first warp
    val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : 0;

    // Calculate sum of sums
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T (&items)[_ITEMS_PER_THREAD], T* shared) {
    T thread_sum = 0;

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        thread_sum += items[ITEM];
    }

    return BlockSum<T, _BLOCK_THREADS, _ITEMS_PER_THREAD>(thread_sum, shared);
}

// ============================================================================
// Block Load (for uncompressed data - used for dimension tables)
// ============================================================================

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T (&items)[_ITEMS_PER_THREAD]) {
    T* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        items[ITEM] = thread_itr[ITEM * _BLOCK_THREADS];
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T (&items)[_ITEMS_PER_THREAD],
    int num_items) {
    T* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (tid + (ITEM * _BLOCK_THREADS) < num_items) {
            items[ITEM] = thread_itr[ITEM * _BLOCK_THREADS];
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockLoad(
    T* inp,
    T (&items)[_ITEMS_PER_THREAD],
    int num_items) {
    T* block_itr = inp;

    if ((_BLOCK_THREADS * _ITEMS_PER_THREAD) == num_items) {
        BlockLoadDirect<T, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, block_itr, items);
    } else {
        BlockLoadDirect<T, _BLOCK_THREADS, _ITEMS_PER_THREAD>(threadIdx.x, block_itr, items, num_items);
    }
}

// ============================================================================
// Block Predicated Load (for uncompressed data)
// ============================================================================

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T (&items)[_ITEMS_PER_THREAD],
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    T* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            items[ITEM] = thread_itr[ITEM * _BLOCK_THREADS];
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoadDirect(
    const unsigned int tid,
    T* block_itr,
    T (&items)[_ITEMS_PER_THREAD],
    int num_items,
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    T* thread_itr = block_itr + tid;

    #pragma unroll
    for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++) {
        if (selection_flags[ITEM]) {
            if (tid + (ITEM * _BLOCK_THREADS) < num_items) {
                items[ITEM] = thread_itr[ITEM * _BLOCK_THREADS];
            }
        }
    }
}

template<typename T, int _BLOCK_THREADS, int _ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoad(
    T* inp,
    T (&items)[_ITEMS_PER_THREAD],
    int num_items,
    int (&selection_flags)[_ITEMS_PER_THREAD]) {
    T* block_itr = inp;

    if ((_BLOCK_THREADS * _ITEMS_PER_THREAD) == num_items) {
        BlockPredLoadDirect<T, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            threadIdx.x, block_itr, items, selection_flags);
    } else {
        BlockPredLoadDirect<T, _BLOCK_THREADS, _ITEMS_PER_THREAD>(
            threadIdx.x, block_itr, items, num_items, selection_flags);
    }
}
