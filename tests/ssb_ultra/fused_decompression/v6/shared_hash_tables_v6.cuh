/**
 * @file shared_hash_tables_v6.cuh
 * @brief V6 Shared Memory Hash Tables
 *
 * Key optimization: Load small dimension tables (Date) into shared memory
 * to eliminate global memory latency for hash probes.
 *
 * Date table: 2556 rows × 8 bytes = ~20KB (fits in 48KB shared memory)
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_v6 {

// Date table constants
constexpr int DATE_HT_SIZE = 2556;       // Number of dates
constexpr int DATE_KEY_MIN = 19920101;   // Minimum datekey
constexpr int DATE_KEY_MAX = 19981230;   // Maximum datekey

// ============================================================================
// Shared Memory Date Hash Table Structure
// ============================================================================

/**
 * @brief Date hash table entry (datekey -> year)
 */
struct DateHTEntry {
    uint32_t datekey;
    uint32_t year;
};

// ============================================================================
// Load Date Hash Table to Shared Memory
// ============================================================================

/**
 * @brief Cooperative load of date hash table to shared memory
 *
 * Call this at kernel start with all threads participating.
 * Uses direct indexing (datekey - DATE_KEY_MIN) as hash.
 *
 * @param d_datekey  Global memory datekey array
 * @param d_year     Global memory year array
 * @param num_dates  Number of dates (should be DATE_HT_SIZE)
 * @param ht_shared  Shared memory array (must be DateHTEntry[DATE_HT_SIZE])
 */
__device__ __forceinline__
void loadDateHTToShared(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_dates,
    DateHTEntry* ht_shared)
{
    // Cooperative load - each thread loads multiple entries
    for (int i = threadIdx.x; i < num_dates; i += blockDim.x) {
        uint32_t key = d_datekey[i];
        uint32_t year = d_year[i];

        // Direct slot = key - min_key (perfect hash for dates)
        int slot = key - DATE_KEY_MIN;
        if (slot >= 0 && slot < DATE_HT_SIZE) {
            ht_shared[slot].datekey = key;
            ht_shared[slot].year = year;
        }
    }
    __syncthreads();
}

/**
 * @brief Initialize shared memory date HT to empty
 */
__device__ __forceinline__
void initDateHTShared(DateHTEntry* ht_shared) {
    for (int i = threadIdx.x; i < DATE_HT_SIZE; i += blockDim.x) {
        ht_shared[i].datekey = 0xFFFFFFFF;  // Empty marker
        ht_shared[i].year = 0;
    }
    __syncthreads();
}

// ============================================================================
// Probe Date Hash Table from Shared Memory
// ============================================================================

/**
 * @brief Probe date hash table (single key)
 *
 * @param ht_shared  Shared memory date hash table
 * @param datekey    Date key to probe
 * @param year       Output year (valid only if returns true)
 * @return true if found
 */
__device__ __forceinline__
bool probeDateHTShared(
    const DateHTEntry* ht_shared,
    uint32_t datekey,
    uint32_t& year)
{
    int slot = datekey - DATE_KEY_MIN;
    if (slot >= 0 && slot < DATE_HT_SIZE) {
        DateHTEntry entry = ht_shared[slot];
        if (entry.datekey == datekey) {
            year = entry.year;
            return true;
        }
    }
    return false;
}

/**
 * @brief Batch probe date hash table (8 keys)
 *
 * Zero global memory latency - all from shared memory!
 *
 * @param ht_shared  Shared memory date hash table
 * @param datekeys   8 date keys to probe
 * @param flags      Selection flags (in/out)
 * @param years      Output years
 */
__device__ __forceinline__
void probeDateHTShared8(
    const DateHTEntry* ht_shared,
    const uint32_t datekeys[8],
    int flags[8],
    uint32_t years[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = datekeys[v] - DATE_KEY_MIN;
            if (slot >= 0 && slot < DATE_HT_SIZE) {
                DateHTEntry entry = ht_shared[slot];
                if (entry.datekey == datekeys[v]) {
                    years[v] = entry.year;
                } else {
                    flags[v] = 0;  // Not found
                }
            } else {
                flags[v] = 0;  // Out of range
            }
        }
    }
}

// ============================================================================
// Filtered Date Hash Table (Year Range)
// ============================================================================

/**
 * @brief Load date hash table with year range filter
 *
 * Only loads dates within [year_min, year_max] range.
 */
__device__ __forceinline__
void loadDateHTToSharedFiltered(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_dates,
    uint32_t year_min,
    uint32_t year_max,
    DateHTEntry* ht_shared)
{
    // Initialize to empty first
    for (int i = threadIdx.x; i < DATE_HT_SIZE; i += blockDim.x) {
        ht_shared[i].datekey = 0xFFFFFFFF;
        ht_shared[i].year = 0;
    }
    __syncthreads();

    // Load only dates in year range
    for (int i = threadIdx.x; i < num_dates; i += blockDim.x) {
        uint32_t year = d_year[i];
        if (year >= year_min && year <= year_max) {
            uint32_t key = d_datekey[i];
            int slot = key - DATE_KEY_MIN;
            if (slot >= 0 && slot < DATE_HT_SIZE) {
                ht_shared[slot].datekey = key;
                ht_shared[slot].year = year;
            }
        }
    }
    __syncthreads();
}

// ============================================================================
// Alternative: Packed Key-Year Format
// ============================================================================

/**
 * @brief Load date hash table in packed format (saves shared memory)
 *
 * Uses uint64_t = (year << 32) | datekey format.
 * Reduces shared memory from 20KB to 20KB (same, but single array).
 */
__device__ __forceinline__
void loadDateHTPacked(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_dates,
    uint64_t* ht_packed)
{
    for (int i = threadIdx.x; i < DATE_HT_SIZE; i += blockDim.x) {
        ht_packed[i] = 0xFFFFFFFFFFFFFFFFULL;  // Empty
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_dates; i += blockDim.x) {
        uint32_t key = d_datekey[i];
        uint32_t year = d_year[i];
        int slot = key - DATE_KEY_MIN;
        if (slot >= 0 && slot < DATE_HT_SIZE) {
            ht_packed[slot] = (static_cast<uint64_t>(year) << 32) | key;
        }
    }
    __syncthreads();
}

__device__ __forceinline__
bool probeDateHTPacked(
    const uint64_t* ht_packed,
    uint32_t datekey,
    uint32_t& year)
{
    int slot = datekey - DATE_KEY_MIN;
    if (slot >= 0 && slot < DATE_HT_SIZE) {
        uint64_t packed = ht_packed[slot];
        uint32_t stored_key = static_cast<uint32_t>(packed);
        if (stored_key == datekey) {
            year = static_cast<uint32_t>(packed >> 32);
            return true;
        }
    }
    return false;
}

__device__ __forceinline__
void probeDateHTPacked8(
    const uint64_t* ht_packed,
    const uint32_t datekeys[8],
    int flags[8],
    uint32_t years[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = datekeys[v] - DATE_KEY_MIN;
            if (slot >= 0 && slot < DATE_HT_SIZE) {
                uint64_t packed = ht_packed[slot];
                uint32_t stored_key = static_cast<uint32_t>(packed);
                if (stored_key == datekeys[v]) {
                    years[v] = static_cast<uint32_t>(packed >> 32);
                } else {
                    flags[v] = 0;
                }
            } else {
                flags[v] = 0;
            }
        }
    }
}

} // namespace l3_v6
