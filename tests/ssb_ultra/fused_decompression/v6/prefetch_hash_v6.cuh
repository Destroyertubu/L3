/**
 * @file prefetch_hash_v6.cuh
 * @brief V6 Hash Table Prefetch Helpers
 *
 * Key innovation: Apply the same prefetch-to-registers strategy used for
 * decompression to hash table probing. This hides memory latency by
 * issuing all 8 hash table lookups in parallel.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_v6 {

constexpr uint32_t HT_EMPTY = 0xFFFFFFFF;

// ============================================================================
// Fast Hash Functions
// ============================================================================

__device__ __forceinline__
uint32_t hash_fast(uint32_t key, int table_size) {
    return ((key ^ (key >> 16)) & 0x7FFFFFFF) % table_size;
}

__device__ __forceinline__
uint32_t hash_murmur3(uint32_t key, int table_size) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key % table_size;
}

// ============================================================================
// Batch Prefetch Hash Probe - Keys Only (8 values)
// ============================================================================

/**
 * @brief Prefetch 8 hash table entries and check for key existence
 *
 * Strategy:
 *   1. Compute all 8 hash slots upfront
 *   2. Issue all 8 __ldg() loads in parallel (prefetch)
 *   3. Check results from registers (zero latency)
 *
 * @param ht_keys     Hash table keys array
 * @param ht_size     Hash table size
 * @param keys        8 keys to probe
 * @param flags       Selection flags (in/out)
 */
__device__ __forceinline__
void prefetchProbe8Exists(
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    const uint32_t keys[8],
    int flags[8])
{
    // Step 1: Compute hash slots
    int slots[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        slots[v] = flags[v] ? hash_fast(keys[v], ht_size) : 0;
    }

    // Step 2: PREFETCH all 8 entries to registers
    uint32_t entries[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        entries[v] = flags[v] ? __ldg(&ht_keys[slots[v]]) : HT_EMPTY;
    }

    // Step 3: Check matches (registers only - zero latency)
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            if (entries[v] != keys[v]) {
                // Miss on fast hash - try MurmurHash3
                int slot2 = hash_murmur3(keys[v], ht_size);
                uint32_t entry2 = __ldg(&ht_keys[slot2]);
                if (entry2 != keys[v]) {
                    // Linear probe up to 4 slots
                    bool found = false;
                    #pragma unroll
                    for (int p = 1; p <= 4; p++) {
                        int ps = (slot2 + p) % ht_size;
                        uint32_t pe = __ldg(&ht_keys[ps]);
                        if (pe == keys[v]) { found = true; break; }
                        if (pe == HT_EMPTY) break;
                    }
                    if (!found) flags[v] = 0;
                }
            }
        }
    }
}

// ============================================================================
// Batch Prefetch Hash Probe - With Values (8 values)
// ============================================================================

/**
 * @brief Prefetch 8 hash table entries and retrieve values
 *
 * @param ht_keys     Hash table keys array
 * @param ht_values   Hash table values array
 * @param ht_size     Hash table size
 * @param keys        8 keys to probe
 * @param flags       Selection flags (in/out)
 * @param values      Output values (only valid if flags[v] remains 1)
 */
__device__ __forceinline__
void prefetchProbe8Get(
    const uint32_t* __restrict__ ht_keys,
    const uint32_t* __restrict__ ht_values,
    int ht_size,
    const uint32_t keys[8],
    int flags[8],
    uint32_t values[8])
{
    // Step 1: Compute hash slots
    int slots[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        slots[v] = flags[v] ? hash_fast(keys[v], ht_size) : 0;
    }

    // Step 2: PREFETCH all 8 key-value pairs to registers
    uint32_t entry_keys[8];
    uint32_t entry_vals[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            entry_keys[v] = __ldg(&ht_keys[slots[v]]);
            entry_vals[v] = __ldg(&ht_values[slots[v]]);
        } else {
            entry_keys[v] = HT_EMPTY;
            entry_vals[v] = 0;
        }
    }

    // Step 3: Check matches and extract values
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            if (entry_keys[v] == keys[v]) {
                values[v] = entry_vals[v];
            } else {
                // Miss on fast hash - try MurmurHash3
                int slot2 = hash_murmur3(keys[v], ht_size);
                uint32_t key2 = __ldg(&ht_keys[slot2]);
                if (key2 == keys[v]) {
                    values[v] = __ldg(&ht_values[slot2]);
                } else {
                    // Linear probe up to 4 slots
                    bool found = false;
                    #pragma unroll
                    for (int p = 1; p <= 4; p++) {
                        int ps = (slot2 + p) % ht_size;
                        uint32_t pk = __ldg(&ht_keys[ps]);
                        if (pk == keys[v]) {
                            values[v] = __ldg(&ht_values[ps]);
                            found = true;
                            break;
                        }
                        if (pk == HT_EMPTY) break;
                    }
                    if (!found) flags[v] = 0;
                }
            }
        }
    }
}

// ============================================================================
// Interleaved Prefetch - Decompress + Hash Probe Overlap
// ============================================================================

/**
 * @brief Issue hash prefetch for 8 keys (non-blocking)
 *
 * This function computes slots and prefetches entries but does NOT wait
 * for results. The caller should do other work (e.g., decompress next column)
 * then call checkPrefetchResults() to get the actual values.
 */
__device__ __forceinline__
void issuePrefetch8(
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    const uint32_t keys[8],
    const int flags[8],
    int slots[8],
    uint32_t prefetched[8])
{
    // Compute slots and issue loads
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        slots[v] = flags[v] ? hash_fast(keys[v], ht_size) : 0;
    }

    // Prefetch (these loads will be in-flight while caller does other work)
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        prefetched[v] = flags[v] ? __ldg(&ht_keys[slots[v]]) : HT_EMPTY;
    }
}

/**
 * @brief Check prefetch results after issuing issuePrefetch8()
 */
__device__ __forceinline__
void checkPrefetchExists(
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    const uint32_t keys[8],
    int flags[8],
    const uint32_t prefetched[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v] && prefetched[v] != keys[v]) {
            // Fast hash missed - fallback to MurmurHash3
            int slot2 = hash_murmur3(keys[v], ht_size);
            uint32_t entry2 = __ldg(&ht_keys[slot2]);
            if (entry2 != keys[v]) {
                bool found = false;
                #pragma unroll
                for (int p = 1; p <= 4; p++) {
                    int ps = (slot2 + p) % ht_size;
                    uint32_t pe = __ldg(&ht_keys[ps]);
                    if (pe == keys[v]) { found = true; break; }
                    if (pe == HT_EMPTY) break;
                }
                if (!found) flags[v] = 0;
            }
        }
    }
}

} // namespace l3_v6
