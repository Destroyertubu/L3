/**
 * @file fused_hash_table.cuh
 * @brief Hash Table Helpers for True Fused SSB Queries
 *
 * This header provides device functions for hash table operations used in
 * Q2.x, Q3.x, and Q4.x queries with true fused decompression.
 *
 * Hash tables are built from dimension tables (which are small and uncompressed),
 * then used to probe during the fused query kernel.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace l3_fused {

// ============================================================================
// Constants
// ============================================================================

constexpr uint32_t HT_EMPTY = 0xFFFFFFFF;

// NOTE: D_LEN, P_LEN, S_LEN, C_LEN are defined in ssb_data_loader.hpp
// Use those constants from the ssb namespace when building hash tables

// Aggregation dimensions
constexpr int NUM_YEARS = 7;      // 1992-1998
constexpr int NUM_BRANDS = 1000;
constexpr int NUM_CATEGORIES = 25;
constexpr int NUM_NATIONS = 25;
constexpr int NUM_CITIES = 250;

// ============================================================================
// Murmur3 Hash Function
// ============================================================================

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
// Fast XOR Hash (3 ALU ops - for two-level probing)
// ============================================================================

/**
 * @brief Ultra-fast XOR hash for first-level probing
 *
 * Only 3 ALU operations vs ~12 for MurmurHash3.
 * Achieves 60-70% hit rate on typical SSB data.
 */
__device__ __forceinline__
uint32_t hash_fast(uint32_t key, int table_size) {
    return ((key ^ (key >> 16)) & 0x7FFFFFFF) % table_size;
}

// ============================================================================
// Two-Level Hash Probing (Optimized)
// ============================================================================

/**
 * @brief Two-level probe for key existence
 *
 * Level 1: Fast XOR hash (3 ops) - hits 60-70% of lookups
 * Level 2: MurmurHash3 + linear probing (fallback)
 *
 * Saves 25-35% hash computation compared to MurmurHash3 only.
 */
__device__ __forceinline__
bool ht_probe_exists_fast(
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    uint32_t key)
{
    // Level 1: Fast XOR hash (quick check only, no early exit on empty)
    int slot = hash_fast(key, ht_size);
    uint32_t stored = ht_keys[slot];
    if (stored == key) return true;

    // Level 2: MurmurHash3 + linear probing (always check here)
    slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < 32; ++probe) {
        int s = (slot + probe) % ht_size;
        stored = ht_keys[s];
        if (stored == key) return true;
        if (stored == HT_EMPTY) return false;
    }
    return false;
}

/**
 * @brief Two-level probe with value retrieval
 */
__device__ __forceinline__
bool ht_probe_get_fast(
    const uint32_t* __restrict__ ht_keys,
    const uint32_t* __restrict__ ht_values,
    int ht_size,
    uint32_t key,
    uint32_t& value)
{
    // Level 1: Fast XOR hash (quick check only, no early exit on empty)
    int slot = hash_fast(key, ht_size);
    uint32_t stored = ht_keys[slot];
    if (stored == key) {
        value = ht_values[slot];
        return true;
    }

    // Level 2: MurmurHash3 + linear probing (always check here)
    slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < 32; ++probe) {
        int s = (slot + probe) % ht_size;
        stored = ht_keys[s];
        if (stored == key) {
            value = ht_values[s];
            return true;
        }
        if (stored == HT_EMPTY) return false;
    }
    return false;
}

// ============================================================================
// Warp-Parallel Multi-Table Probing (for Q3.x/Q4.x)
// ============================================================================

/**
 * @brief Hash table info for warp-parallel probing
 */
struct HashTableInfo {
    const uint32_t* keys;
    const uint32_t* values;
    int size;
};

/**
 * @brief Warp-parallel probe of 3 hash tables (for Q3.x queries)
 *
 * Divides 32 lanes into 3 groups:
 *   Group 0 (lanes 0-10):  Date table
 *   Group 1 (lanes 11-21): Customer table
 *   Group 2 (lanes 22-31): Supplier table
 *
 * Each group does parallel linear probing, then uses __ballot_sync
 * and __shfl_sync to broadcast found values.
 */
__device__ __forceinline__
bool warpProbe3Tables(
    uint32_t key_date, uint32_t key_cust, uint32_t key_supp,
    const HashTableInfo& ht_date,
    const HashTableInfo& ht_cust,
    const HashTableInfo& ht_supp,
    uint32_t& val_date, uint32_t& val_cust, uint32_t& val_supp)
{
    int lane = threadIdx.x & 31;

    // Assign lanes to groups (roughly 11, 11, 10)
    int group;
    int lane_in_group;
    uint32_t my_key;
    const uint32_t* ht_keys;
    const uint32_t* ht_vals;
    int ht_size;

    if (lane < 11) {
        group = 0;
        lane_in_group = lane;
        my_key = key_date;
        ht_keys = ht_date.keys;
        ht_vals = ht_date.values;
        ht_size = ht_date.size;
    } else if (lane < 22) {
        group = 1;
        lane_in_group = lane - 11;
        my_key = key_cust;
        ht_keys = ht_cust.keys;
        ht_vals = ht_cust.values;
        ht_size = ht_cust.size;
    } else {
        group = 2;
        lane_in_group = lane - 22;
        my_key = key_supp;
        ht_keys = ht_supp.keys;
        ht_vals = ht_supp.values;
        ht_size = ht_supp.size;
    }

    // Each lane probes one slot
    int base_slot = hash_fast(my_key, ht_size);
    int probe_slot = (base_slot + lane_in_group) % ht_size;
    uint32_t stored = ht_keys[probe_slot];
    bool found = (stored == my_key);
    uint32_t my_val = found ? ht_vals[probe_slot] : 0;

    // Ballot to check results
    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);

    // Check each group
    bool g0_found = (ballot & 0x000007FF) != 0;  // Lanes 0-10
    bool g1_found = (ballot & 0x003FF800) != 0;  // Lanes 11-21
    bool g2_found = (ballot & 0xFFC00000) != 0;  // Lanes 22-31

    // If any group failed first try, do extended probing
    if (!g0_found || !g1_found || !g2_found) {
        base_slot = hash_murmur3(my_key, ht_size);
        for (int round = 0; round < 3 && !(g0_found && g1_found && g2_found); ++round) {
            probe_slot = (base_slot + round * 11 + lane_in_group) % ht_size;
            if (probe_slot < ht_size) {
                stored = ht_keys[probe_slot];
                if (stored == my_key) {
                    found = true;
                    my_val = ht_vals[probe_slot];
                }
            }
            ballot = __ballot_sync(0xFFFFFFFF, found);
            g0_found = (ballot & 0x000007FF) != 0;
            g1_found = (ballot & 0x003FF800) != 0;
            g2_found = (ballot & 0xFFC00000) != 0;
        }
    }

    // Broadcast values via shuffle
    int src0 = __ffs(ballot & 0x000007FF) - 1;
    int src1 = __ffs((ballot >> 11) & 0x7FF) - 1 + 11;
    int src2 = __ffs((ballot >> 22) & 0x3FF) - 1 + 22;

    val_date = g0_found ? __shfl_sync(0xFFFFFFFF, my_val, src0 >= 0 ? src0 : 0) : 0;
    val_cust = g1_found ? __shfl_sync(0xFFFFFFFF, my_val, src1 >= 11 ? src1 : 11) : 0;
    val_supp = g2_found ? __shfl_sync(0xFFFFFFFF, my_val, src2 >= 22 ? src2 : 22) : 0;

    return g0_found && g1_found && g2_found;
}

/**
 * @brief Warp-parallel probe of 4 hash tables (for Q4.x queries)
 *
 * Divides 32 lanes into 4 groups of 8:
 *   Group 0 (lanes 0-7):   Date table
 *   Group 1 (lanes 8-15):  Part table
 *   Group 2 (lanes 16-23): Supplier table
 *   Group 3 (lanes 24-31): Customer table
 */
__device__ __forceinline__
bool warpProbe4Tables(
    uint32_t key_date, uint32_t key_part, uint32_t key_supp, uint32_t key_cust,
    const HashTableInfo& ht_date,
    const HashTableInfo& ht_part,
    const HashTableInfo& ht_supp,
    const HashTableInfo& ht_cust,
    uint32_t& val_date, uint32_t& val_part, uint32_t& val_supp, uint32_t& val_cust)
{
    int lane = threadIdx.x & 31;
    int group = lane >> 3;           // 0-3
    int lane_in_group = lane & 7;    // 0-7

    // Select table based on group
    uint32_t my_key;
    const uint32_t* ht_keys;
    const uint32_t* ht_vals;
    int ht_size;

    switch (group) {
        case 0:
            my_key = key_date;
            ht_keys = ht_date.keys;
            ht_vals = ht_date.values;
            ht_size = ht_date.size;
            break;
        case 1:
            my_key = key_part;
            ht_keys = ht_part.keys;
            ht_vals = ht_part.values;
            ht_size = ht_part.size;
            break;
        case 2:
            my_key = key_supp;
            ht_keys = ht_supp.keys;
            ht_vals = ht_supp.values;
            ht_size = ht_supp.size;
            break;
        default:  // 3
            my_key = key_cust;
            ht_keys = ht_cust.keys;
            ht_vals = ht_cust.values;
            ht_size = ht_cust.size;
            break;
    }

    // Parallel probe
    int base_slot = hash_fast(my_key, ht_size);
    int probe_slot = (base_slot + lane_in_group) % ht_size;
    uint32_t stored = ht_keys[probe_slot];
    bool found = (stored == my_key);
    uint32_t my_val = found ? ht_vals[probe_slot] : 0;

    // Ballot to check results
    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);

    bool g0_found = (ballot & 0x000000FF) != 0;  // Lanes 0-7
    bool g1_found = (ballot & 0x0000FF00) != 0;  // Lanes 8-15
    bool g2_found = (ballot & 0x00FF0000) != 0;  // Lanes 16-23
    bool g3_found = (ballot & 0xFF000000) != 0;  // Lanes 24-31

    // Extended probing if needed
    if (!g0_found || !g1_found || !g2_found || !g3_found) {
        base_slot = hash_murmur3(my_key, ht_size);
        for (int round = 0; round < 4; ++round) {
            probe_slot = (base_slot + round * 8 + lane_in_group) % ht_size;
            stored = ht_keys[probe_slot];
            if (stored == my_key) {
                found = true;
                my_val = ht_vals[probe_slot];
            }
            ballot = __ballot_sync(0xFFFFFFFF, found);
        }
        g0_found = (ballot & 0x000000FF) != 0;
        g1_found = (ballot & 0x0000FF00) != 0;
        g2_found = (ballot & 0x00FF0000) != 0;
        g3_found = (ballot & 0xFF000000) != 0;
    }

    // Broadcast found values
    int src0 = __ffs(ballot & 0x000000FF) - 1;
    int src1 = 8 + __ffs((ballot >> 8) & 0xFF) - 1;
    int src2 = 16 + __ffs((ballot >> 16) & 0xFF) - 1;
    int src3 = 24 + __ffs((ballot >> 24) & 0xFF) - 1;

    val_date = g0_found ? __shfl_sync(0xFFFFFFFF, my_val, src0 >= 0 ? src0 : 0) : 0;
    val_part = g1_found ? __shfl_sync(0xFFFFFFFF, my_val, src1 >= 8 ? src1 : 8) : 0;
    val_supp = g2_found ? __shfl_sync(0xFFFFFFFF, my_val, src2 >= 16 ? src2 : 16) : 0;
    val_cust = g3_found ? __shfl_sync(0xFFFFFFFF, my_val, src3 >= 24 ? src3 : 24) : 0;

    return g0_found && g1_found && g2_found && g3_found;
}

// ============================================================================
// Hash Table Probe Functions (Device)
// ============================================================================

/**
 * @brief Probe hash table for key existence only (no value)
 * @return true if key found
 */
__device__ __forceinline__
bool ht_probe_exists(
    const uint32_t* __restrict__ ht_keys,
    int ht_size,
    uint32_t key)
{
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        if (ht_keys[s] == key) return true;
        if (ht_keys[s] == HT_EMPTY) return false;
    }
    return false;
}

/**
 * @brief Probe hash table and get value
 * @return true if found, value is set
 */
__device__ __forceinline__
bool ht_probe_get(
    const uint32_t* __restrict__ ht_keys,
    const uint32_t* __restrict__ ht_values,
    int ht_size,
    uint32_t key,
    uint32_t& value)
{
    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        if (ht_keys[s] == key) {
            value = ht_values[s];
            return true;
        }
        if (ht_keys[s] == HT_EMPTY) return false;
    }
    return false;
}

// ============================================================================
// Hash Table Build Kernels (Host-launched)
// ============================================================================

/**
 * @brief Build date hash table (datekey -> year)
 */
__global__ void build_date_ht_kernel(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_rows,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t key = d_datekey[idx];
    uint32_t value = d_year[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build part hash table with category filter (partkey -> brand1)
 */
__global__ void build_part_ht_filtered_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_category,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t filter_category,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_category[idx] != filter_category) return;

    uint32_t key = p_partkey[idx];
    uint32_t value = p_brand1[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build part hash table with brand range filter (partkey -> brand1)
 */
__global__ void build_part_ht_brand_range_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t brand_min,
    uint32_t brand_max,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t brand = p_brand1[idx];
    if (brand < brand_min || brand > brand_max) return;

    uint32_t key = p_partkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = brand;
            return;
        }
    }
}

/**
 * @brief Build part hash table with exact brand filter (partkey only)
 */
__global__ void build_part_ht_brand_exact_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t target_brand,
    uint32_t* __restrict__ ht_keys,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_brand1[idx] != target_brand) return;

    uint32_t key = p_partkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) return;
    }
}

/**
 * @brief Build part hash table with mfgr filter (keys only)
 * For Q4.1 which filters MFGR#1 or MFGR#2
 */
__global__ void build_part_ht_mfgr_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_mfgr,
    int num_rows,
    uint32_t mfgr1,
    uint32_t mfgr2,
    uint32_t* __restrict__ ht_keys,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;

    uint32_t key = p_partkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) return;
    }
}

/**
 * @brief Build part hash table with mfgr filter (partkey -> category)
 * For Q4.2 which needs category grouping
 */
__global__ void build_part_ht_mfgr_category_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_mfgr,
    const uint32_t* __restrict__ p_category,
    int num_rows,
    uint32_t mfgr1,
    uint32_t mfgr2,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;

    uint32_t key = p_partkey[idx];
    uint32_t value = p_category[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build part hash table with category filter (partkey -> brand)
 * For Q4.3 which needs brand grouping
 */
__global__ void build_part_ht_category_brand_kernel(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_category,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t filter_category,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_category[idx] != filter_category) return;

    uint32_t key = p_partkey[idx];
    uint32_t value = p_brand1[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build supplier hash table with nation filter (suppkey -> city)
 * For Q4.3 which needs city grouping
 */
__global__ void build_supplier_ht_nation_city_kernel(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_nation,
    const uint32_t* __restrict__ s_city,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_nation[idx] != filter_nation) return;

    uint32_t key = s_suppkey[idx];
    uint32_t value = s_city[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build supplier hash table with region filter (keys only)
 */
__global__ void build_supplier_ht_region_kernel(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_region,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht_keys,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_region[idx] != filter_region) return;

    uint32_t key = s_suppkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) return;
    }
}

/**
 * @brief Build supplier hash table with region filter (suppkey -> nation)
 * For Q3.1 which needs nation grouping
 */
__global__ void build_supplier_ht_region_nation_kernel(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_region,
    const uint32_t* __restrict__ s_nation,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_region[idx] != filter_region) return;

    uint32_t key = s_suppkey[idx];
    uint32_t value = s_nation[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build customer hash table with region filter (custkey -> nation)
 */
__global__ void build_customer_ht_region_kernel(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_region,
    const uint32_t* __restrict__ c_nation,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_region[idx] != filter_region) return;

    uint32_t key = c_custkey[idx];
    uint32_t value = c_nation[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build customer hash table with nation filter (custkey -> city)
 */
__global__ void build_customer_ht_nation_kernel(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_nation,
    const uint32_t* __restrict__ c_city,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_nation[idx] != filter_nation) return;

    uint32_t key = c_custkey[idx];
    uint32_t value = c_city[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build supplier hash table with nation filter (suppkey -> city)
 */
__global__ void build_supplier_ht_nation_kernel(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_nation,
    const uint32_t* __restrict__ s_city,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_nation[idx] != filter_nation) return;

    uint32_t key = s_suppkey[idx];
    uint32_t value = s_city[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = value;
            return;
        }
    }
}

/**
 * @brief Build customer hash table with city filter (custkey -> city)
 * For Q3.3/Q3.4 which filter by specific cities
 */
__global__ void build_customer_ht_city_filter_kernel(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_city,
    int num_rows,
    uint32_t city1,
    uint32_t city2,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t city = c_city[idx];
    if (city != city1 && city != city2) return;

    uint32_t key = c_custkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = city;
            return;
        }
    }
}

/**
 * @brief Build supplier hash table with city filter (suppkey -> city)
 */
__global__ void build_supplier_ht_city_filter_kernel(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_city,
    int num_rows,
    uint32_t city1,
    uint32_t city2,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t city = s_city[idx];
    if (city != city1 && city != city2) return;

    uint32_t key = s_suppkey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = city;
            return;
        }
    }
}

/**
 * @brief Build date hash table with year range filter (datekey only)
 */
__global__ void build_date_ht_year_range_kernel(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_rows,
    uint32_t year_min,
    uint32_t year_max,
    uint32_t* __restrict__ ht_keys,
    uint32_t* __restrict__ ht_values,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t year = d_year[idx];
    if (year < year_min || year > year_max) return;

    uint32_t key = d_datekey[idx];

    int slot = hash_murmur3(key, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, key);
        if (old == HT_EMPTY || old == key) {
            ht_values[s] = year;
            return;
        }
    }
}

/**
 * @brief Build date hash table with month filter (keys only)
 */
__global__ void build_date_ht_month_kernel(
    const uint32_t* __restrict__ d_datekey,
    int num_rows,
    uint32_t date_min,
    uint32_t date_max,
    uint32_t* __restrict__ ht_keys,
    int ht_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t datekey = d_datekey[idx];
    if (datekey < date_min || datekey > date_max) return;

    int slot = hash_murmur3(datekey, ht_size);
    for (int probe = 0; probe < ht_size; ++probe) {
        int s = (slot + probe) % ht_size;
        uint32_t old = atomicCAS(&ht_keys[s], HT_EMPTY, datekey);
        if (old == HT_EMPTY || old == datekey) return;
    }
}

// ============================================================================
// Hash Table Allocation Helpers
// ============================================================================

struct HashTable {
    uint32_t* d_keys;
    uint32_t* d_values;  // May be nullptr for keys-only tables
    int size;

    void allocate(int table_len, bool with_values = true) {
        size = table_len * 2;  // 50% load factor
        cudaMalloc(&d_keys, size * sizeof(uint32_t));
        cudaMemset(d_keys, 0xFF, size * sizeof(uint32_t));  // HT_EMPTY
        if (with_values) {
            cudaMalloc(&d_values, size * sizeof(uint32_t));
        } else {
            d_values = nullptr;
        }
    }

    void free() {
        if (d_keys) cudaFree(d_keys);
        if (d_values) cudaFree(d_values);
        d_keys = nullptr;
        d_values = nullptr;
    }
};

} // namespace l3_fused
