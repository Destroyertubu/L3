/**
 * @file crystal_fls.cuh
 * @brief Crystal-style predicates and reduction for Vertical L3 queries
 *
 * Based on Crystal-opt predicates with modifications for L3 FLS integration.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_fls {

// ============================================================================
// Selection Flag Initialization
// ============================================================================

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(int (&selection_flags)[ITEMS_PER_THREAD]) {
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        selection_flags[ITEM] = 1;
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void InitFlags(
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx = threadIdx.x + ITEM * BLOCK_THREADS;
        selection_flags[ITEM] = (idx < num_items) ? 1 : 0;
    }
}

// ============================================================================
// Comparison Operators
// ============================================================================

template<typename T>
struct LessThan {
    T compare;
    __device__ __forceinline__ LessThan(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a < compare; }
};

template<typename T>
struct GreaterThan {
    T compare;
    __device__ __forceinline__ GreaterThan(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a > compare; }
};

template<typename T>
struct LessThanEq {
    T compare;
    __device__ __forceinline__ LessThanEq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a <= compare; }
};

template<typename T>
struct GreaterThanEq {
    T compare;
    __device__ __forceinline__ GreaterThanEq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a >= compare; }
};

template<typename T>
struct Eq {
    T compare;
    __device__ __forceinline__ Eq(T compare) : compare(compare) {}
    __device__ __forceinline__ bool operator()(const T& a) const { return a == compare; }
};

// ============================================================================
// Block Predicate Functions
// ============================================================================

// Core AND predicate - applies operation and ANDs with existing flags
template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAnd(
    T (&items)[ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (threadIdx.x + ITEM * BLOCK_THREADS < num_items) {
            selection_flags[ITEM] = selection_flags[ITEM] && select_op(items[ITEM]);
        }
    }
}

// Core predicate - replaces existing flags
template<typename T, typename SelectOp, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPred(
    T (&items)[ITEMS_PER_THREAD],
    SelectOp select_op,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        if (threadIdx.x + ITEM * BLOCK_THREADS < num_items) {
            selection_flags[ITEM] = select_op(items[ITEM]);
        }
    }
}

// ============================================================================
// Convenience Wrappers
// ============================================================================

// Greater Than (>)
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGT(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    GreaterThan<T> select_op(compare);
    BlockPred<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGT(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    GreaterThan<T> select_op(compare);
    BlockPredAnd<T, GreaterThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

// Less Than (<)
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLT(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    LessThan<T> select_op(compare);
    BlockPred<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLT(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    LessThan<T> select_op(compare);
    BlockPredAnd<T, LessThan<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

// Greater Than or Equal (>=)
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredGTE(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    GreaterThanEq<T> select_op(compare);
    BlockPred<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndGTE(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    GreaterThanEq<T> select_op(compare);
    BlockPredAnd<T, GreaterThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

// Less Than or Equal (<=)
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLTE(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    LessThanEq<T> select_op(compare);
    BlockPred<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndLTE(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    LessThanEq<T> select_op(compare);
    BlockPredAnd<T, LessThanEq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

// Equal (==)
template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredEQ(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    Eq<T> select_op(compare);
    BlockPred<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredAndEQ(
    T (&items)[ITEMS_PER_THREAD],
    T compare,
    int (&selection_flags)[ITEMS_PER_THREAD],
    int num_items)
{
    Eq<T> select_op(compare);
    BlockPredAnd<T, Eq<T>, BLOCK_THREADS, ITEMS_PER_THREAD>(
        items, select_op, selection_flags, num_items);
}

// ============================================================================
// Block Reduction
// ============================================================================

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T item, T* shared) {
    __syncthreads();

    T val = item;
    const int warp_size = 32;
    int lane = threadIdx.x % warp_size;
    int wid = threadIdx.x / warp_size;

    // Warp reduction using shuffle
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Store warp sum
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Final reduction in first warp
    val = (threadIdx.x < blockDim.x / warp_size) ? shared[lane] : 0;

    if (wid == 0) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
    }

    return val;
}

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ T BlockSum(T (&items)[ITEMS_PER_THREAD], T* shared) {
    T thread_sum = 0;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        thread_sum += items[ITEM];
    }

    return BlockSum<T, BLOCK_THREADS, ITEMS_PER_THREAD>(thread_sum, shared);
}

// ============================================================================
// Early Termination Check
// ============================================================================

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ bool IsTerm(int (&selection_flags)[ITEMS_PER_THREAD]) {
    int has_active = 0;

    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        has_active |= selection_flags[ITEM];
    }

    // Use ballot to check across warp
    unsigned int warp_active = __ballot_sync(0xFFFFFFFF, has_active);
    return (warp_active == 0);
}

// ============================================================================
// Predicate Load (only load where selection flag is set)
// ============================================================================

template<typename T, int BLOCK_THREADS, int ITEMS_PER_THREAD>
__device__ __forceinline__ void BlockPredLoad(
    const T* __restrict__ data,
    T (&items)[ITEMS_PER_THREAD],
    int num_items,
    int (&selection_flags)[ITEMS_PER_THREAD])
{
    #pragma unroll
    for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++) {
        int idx = threadIdx.x + ITEM * BLOCK_THREADS;
        if (selection_flags[ITEM] && idx < num_items) {
            items[ITEM] = data[idx];
        }
    }
}

}  // namespace l3_fls
