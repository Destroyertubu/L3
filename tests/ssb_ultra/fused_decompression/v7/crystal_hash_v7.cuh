/**
 * @file crystal_hash_v7.cuh
 * @brief Crystal-compatible hash table probe functions (Vertical style)
 *
 * Key insight: Vertical uses PERFECT HASH with SINGLE memory access.
 * This is why Vertical Q2.1 achieves 0.89ms while L3 V5 takes 4.38ms.
 *
 * Vertical hash strategy:
 *   #define HASH(X,Y,Z) ((X-Z) % Y)  -- direct slot, no collision handling
 *   slot = ht[hash];  -- ONE memory read per probe
 *
 * L3 V5 hash strategy (old):
 *   slot = hash_murmur3(key, ht_size);
 *   for (probe = 0; probe < 32; ++probe) {...}  -- UP TO 32 reads!
 *
 * This file implements Vertical-compatible single-probe hash tables.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_crystal {

// ============================================================================
// Crystal-Style Hash Macro (Vertical compatible)
// ============================================================================

/**
 * Perfect hash function from Vertical crystal/join.cuh
 * @param key   The key to hash
 * @param len   Hash table size
 * @param min   Minimum key value (for offset)
 * @return Direct slot index
 */
#define CRYSTAL_HASH(key, len, min) (((key) - (min)) % (len))

// ============================================================================
// Single-Probe Existence Check (Keys-Only Table)
// ============================================================================

/**
 * @brief Probe hash table for key existence (single memory access)
 *
 * Table structure: ht[slot] = key (or 0 if empty)
 * Exactly matches Vertical BlockProbeAndPHT_1
 *
 * @param ht      Hash table (keys only)
 * @param ht_len  Hash table size
 * @param key     Key to probe
 * @param key_min Minimum key value for offset (0 for most tables)
 * @return true if key exists in table
 */
__device__ __forceinline__
bool probe_exists_crystal(
    const uint32_t* __restrict__ ht,
    int ht_len,
    uint32_t key,
    uint32_t key_min)
{
    int slot = CRYSTAL_HASH(key, ht_len, key_min);
    return __ldg(&ht[slot]) != 0;  // ONE memory read!
}

/**
 * @brief Batch probe 8 keys for existence
 *
 * @param ht       Hash table (keys only)
 * @param ht_len   Hash table size
 * @param keys     8 keys to probe
 * @param key_min  Minimum key value for offset
 * @param flags    Selection flags (in/out) - cleared if not found
 */
__device__ __forceinline__
void probe_exists_crystal_8(
    const uint32_t* __restrict__ ht,
    int ht_len,
    const uint32_t keys[8],
    uint32_t key_min,
    int flags[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = CRYSTAL_HASH(keys[v], ht_len, key_min);
            if (__ldg(&ht[slot]) == 0) {
                flags[v] = 0;
            }
        }
    }
}

// ============================================================================
// Single-Probe with Value Retrieval (Key-Value Table)
// ============================================================================

/**
 * @brief Probe hash table and get value (single memory access)
 *
 * Table structure: interleaved [key0, val0, key1, val1, ...]
 * ht[slot*2] = key, ht[slot*2+1] = value
 *
 * Exactly matches Vertical BlockProbeAndPHT_2
 *
 * @param ht      Hash table (interleaved key-value pairs)
 * @param ht_len  Hash table size (number of KV pairs)
 * @param key     Key to probe
 * @param key_min Minimum key value for offset
 * @param value   Output value (only valid if returns true)
 * @return true if key found
 */
__device__ __forceinline__
bool probe_get_crystal(
    const uint32_t* __restrict__ ht,
    int ht_len,
    uint32_t key,
    uint32_t key_min,
    uint32_t& value)
{
    int slot = CRYSTAL_HASH(key, ht_len, key_min);

    // Read key-value pair as single 64-bit load (Vertical style)
    uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht[slot << 1]);

    if (kv != 0) {
        // Value is in upper 32 bits
        value = static_cast<uint32_t>(kv >> 32);
        return true;
    }
    return false;
}

/**
 * @brief Batch probe 8 keys and get values
 *
 * @param ht       Hash table (interleaved key-value pairs)
 * @param ht_len   Hash table size
 * @param keys     8 keys to probe
 * @param key_min  Minimum key value for offset
 * @param flags    Selection flags (in/out) - cleared if not found
 * @param values   Output values (only valid where flags[v] remains 1)
 */
__device__ __forceinline__
void probe_get_crystal_8(
    const uint32_t* __restrict__ ht,
    int ht_len,
    const uint32_t keys[8],
    uint32_t key_min,
    int flags[8],
    uint32_t values[8])
{
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = CRYSTAL_HASH(keys[v], ht_len, key_min);
            uint64_t kv = *reinterpret_cast<const uint64_t*>(&ht[slot << 1]);

            if (kv != 0) {
                values[v] = static_cast<uint32_t>(kv >> 32);
            } else {
                flags[v] = 0;
            }
        }
    }
}

// ============================================================================
// Prefetch + Probe (Batch 8 keys with latency hiding)
// ============================================================================

/**
 * @brief Prefetch 8 hash slots, then check results
 *
 * This overlaps memory latency by issuing all loads first, then checking.
 */
__device__ __forceinline__
void prefetch_probe_exists_8(
    const uint32_t* __restrict__ ht,
    int ht_len,
    const uint32_t keys[8],
    uint32_t key_min,
    int flags[8])
{
    // Phase 1: Compute slots and prefetch
    uint32_t entries[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = CRYSTAL_HASH(keys[v], ht_len, key_min);
            entries[v] = __ldg(&ht[slot]);
        } else {
            entries[v] = 0;
        }
    }

    // Phase 2: Check results (from registers)
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v] && entries[v] == 0) {
            flags[v] = 0;
        }
    }
}

/**
 * @brief Prefetch 8 hash slots with values, then check results
 */
__device__ __forceinline__
void prefetch_probe_get_8(
    const uint32_t* __restrict__ ht,
    int ht_len,
    const uint32_t keys[8],
    uint32_t key_min,
    int flags[8],
    uint32_t values[8])
{
    // Phase 1: Compute slots and prefetch key-value pairs
    uint64_t entries[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            int slot = CRYSTAL_HASH(keys[v], ht_len, key_min);
            entries[v] = *reinterpret_cast<const uint64_t*>(&ht[slot << 1]);
        } else {
            entries[v] = 0;
        }
    }

    // Phase 2: Check results and extract values (from registers)
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        if (flags[v]) {
            if (entries[v] != 0) {
                values[v] = static_cast<uint32_t>(entries[v] >> 32);
            } else {
                flags[v] = 0;
            }
        }
    }
}

// ============================================================================
// Hash Table Constants
// ============================================================================

// Date table range (for direct indexing)
constexpr uint32_t DATE_KEY_MIN = 19920101;
constexpr uint32_t DATE_KEY_MAX = 19981230;
constexpr int DATE_HT_LEN = DATE_KEY_MAX - DATE_KEY_MIN + 1;  // 60,130

}  // namespace l3_crystal
