/**
 * @file parallel_hash_probe.cuh
 * @brief Warp-Level Parallel Hash Probing for SSB Queries
 *
 * This file implements warp-cooperative hash probing where 32 lanes
 * simultaneously probe multiple hash tables and exchange results via shuffle.
 *
 * Key optimizations:
 * - 4-table parallel probing for Q4.x queries (PART, SUPPLIER, CUSTOMER, DATE)
 * - 3-table parallel probing for Q3.x queries (SUPPLIER, CUSTOMER, DATE)
 * - Linear probing within each group of lanes
 * - Result exchange via __shfl_sync()
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "ssb_common.cuh"

namespace ssb {

// ============================================================================
// Warp-Level Parallel Hash Probing
// ============================================================================

/**
 * @brief Warp-parallel probe of 4 hash tables simultaneously
 *
 * Divides 32 lanes into 4 groups of 8 lanes each:
 * - Lanes 0-7:   probe hash table 0 (e.g., PART)
 * - Lanes 8-15:  probe hash table 1 (e.g., SUPPLIER)
 * - Lanes 16-23: probe hash table 2 (e.g., CUSTOMER)
 * - Lanes 24-31: probe hash table 3 (e.g., DATE)
 *
 * Each group does parallel linear probing (8 slots checked simultaneously)
 * Results are exchanged via __shfl_sync()
 *
 * @return true if all 4 keys were found in their respective tables
 */
__device__ __forceinline__ bool warpParallelProbe4Tables(
    uint32_t key0, uint32_t key1, uint32_t key2, uint32_t key3,
    const uint32_t* __restrict__ ht0_keys, const uint32_t* __restrict__ ht0_vals, int ht0_size,
    const uint32_t* __restrict__ ht1_keys, const uint32_t* __restrict__ ht1_vals, int ht1_size,
    const uint32_t* __restrict__ ht2_keys, const uint32_t* __restrict__ ht2_vals, int ht2_size,
    const uint32_t* __restrict__ ht3_keys, const uint32_t* __restrict__ ht3_vals, int ht3_size,
    uint32_t& out_val0, uint32_t& out_val1, uint32_t& out_val2, uint32_t& out_val3)
{
    const int lane = threadIdx.x & 31;
    const int group = lane >> 3;        // 0-3 for 4 groups
    const int lane_in_group = lane & 7; // 0-7 within each group

    // Select key and hash table based on group
    uint32_t my_key;
    const uint32_t* ht_keys;
    const uint32_t* ht_vals;
    int ht_size;

    if (group == 0) {
        my_key = key0; ht_keys = ht0_keys; ht_vals = ht0_vals; ht_size = ht0_size;
    } else if (group == 1) {
        my_key = key1; ht_keys = ht1_keys; ht_vals = ht1_vals; ht_size = ht1_size;
    } else if (group == 2) {
        my_key = key2; ht_keys = ht2_keys; ht_vals = ht2_vals; ht_size = ht2_size;
    } else {
        my_key = key3; ht_keys = ht3_keys; ht_vals = ht3_vals; ht_size = ht3_size;
    }

    // Compute base slot using fast hash
    int base_slot = hash_fast(my_key, ht_size);

    // Each lane in group probes a different slot (parallel linear probing)
    int probe_slot = (base_slot + lane_in_group) % ht_size;
    uint32_t stored_key = ht_keys[probe_slot];
    bool found = (stored_key == my_key);
    uint32_t my_val = found ? ht_vals[probe_slot] : 0;

    // Ballot to find which lanes found their key
    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);

    // Check if each group found at least one match
    bool group0_found = (ballot & 0x000000FF) != 0;
    bool group1_found = (ballot & 0x0000FF00) != 0;
    bool group2_found = (ballot & 0x00FF0000) != 0;
    bool group3_found = (ballot & 0xFF000000) != 0;

    // If initial probe failed for any group, try extended probing
    if (!group0_found || !group1_found || !group2_found || !group3_found) {
        // Fall back to full MurmurHash + more probes
        base_slot = hash_murmur3(my_key, ht_size);

        // Try 24 more slots (3 rounds of 8)
        for (int round = 0; round < 3 && !((ballot >> (group * 8)) & 0xFF); round++) {
            probe_slot = (base_slot + round * 8 + lane_in_group) % ht_size;
            stored_key = ht_keys[probe_slot];
            found = (stored_key == my_key);
            if (found) my_val = ht_vals[probe_slot];
            ballot = __ballot_sync(0xFFFFFFFF, found);
        }

        group0_found = (ballot & 0x000000FF) != 0;
        group1_found = (ballot & 0x0000FF00) != 0;
        group2_found = (ballot & 0x00FF0000) != 0;
        group3_found = (ballot & 0xFF000000) != 0;
    }

    // Exchange results via shuffle - get value from the lane that found the key
    // For each group, find the first lane with a match and broadcast its value
    int src_lane0 = __ffs(ballot & 0x000000FF) - 1;
    int src_lane1 = 8 + __ffs((ballot >> 8) & 0xFF) - 1;
    int src_lane2 = 16 + __ffs((ballot >> 16) & 0xFF) - 1;
    int src_lane3 = 24 + __ffs((ballot >> 24) & 0xFF) - 1;

    out_val0 = group0_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane0) : 0;
    out_val1 = group1_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane1) : 0;
    out_val2 = group2_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane2) : 0;
    out_val3 = group3_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane3) : 0;

    return group0_found && group1_found && group2_found && group3_found;
}

/**
 * @brief Warp-parallel probe of 3 hash tables simultaneously
 *
 * Divides 32 lanes into 3 groups:
 * - Lanes 0-10:  probe hash table 0 (11 lanes)
 * - Lanes 11-21: probe hash table 1 (11 lanes)
 * - Lanes 22-31: probe hash table 2 (10 lanes)
 *
 * @return true if all 3 keys were found
 */
__device__ __forceinline__ bool warpParallelProbe3Tables(
    uint32_t key0, uint32_t key1, uint32_t key2,
    const uint32_t* __restrict__ ht0_keys, const uint32_t* __restrict__ ht0_vals, int ht0_size,
    const uint32_t* __restrict__ ht1_keys, const uint32_t* __restrict__ ht1_vals, int ht1_size,
    const uint32_t* __restrict__ ht2_keys, const uint32_t* __restrict__ ht2_vals, int ht2_size,
    uint32_t& out_val0, uint32_t& out_val1, uint32_t& out_val2)
{
    const int lane = threadIdx.x & 31;

    // Determine group (0, 1, or 2) and lane within group
    int group, lane_in_group;
    if (lane < 11) {
        group = 0;
        lane_in_group = lane;
    } else if (lane < 22) {
        group = 1;
        lane_in_group = lane - 11;
    } else {
        group = 2;
        lane_in_group = lane - 22;
    }

    // Select key and hash table
    uint32_t my_key;
    const uint32_t* ht_keys;
    const uint32_t* ht_vals;
    int ht_size;

    if (group == 0) {
        my_key = key0; ht_keys = ht0_keys; ht_vals = ht0_vals; ht_size = ht0_size;
    } else if (group == 1) {
        my_key = key1; ht_keys = ht1_keys; ht_vals = ht1_vals; ht_size = ht1_size;
    } else {
        my_key = key2; ht_keys = ht2_keys; ht_vals = ht2_vals; ht_size = ht2_size;
    }

    // Fast hash probe
    int base_slot = hash_fast(my_key, ht_size);
    int probe_slot = (base_slot + lane_in_group) % ht_size;
    uint32_t stored_key = ht_keys[probe_slot];
    bool found = (stored_key == my_key);
    uint32_t my_val = found ? ht_vals[probe_slot] : 0;

    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);

    // Group masks
    const unsigned mask0 = 0x000007FF;  // lanes 0-10
    const unsigned mask1 = 0x003FF800;  // lanes 11-21
    const unsigned mask2 = 0xFFC00000;  // lanes 22-31

    bool group0_found = (ballot & mask0) != 0;
    bool group1_found = (ballot & mask1) != 0;
    bool group2_found = (ballot & mask2) != 0;

    // Extended probing if needed
    if (!group0_found || !group1_found || !group2_found) {
        base_slot = hash_murmur3(my_key, ht_size);

        for (int round = 0; round < 3; round++) {
            unsigned group_mask = (group == 0) ? mask0 : (group == 1) ? mask1 : mask2;
            if ((ballot & group_mask) != 0) continue;

            probe_slot = (base_slot + round * 11 + lane_in_group) % ht_size;
            stored_key = ht_keys[probe_slot];
            found = (stored_key == my_key);
            if (found) my_val = ht_vals[probe_slot];
            ballot = __ballot_sync(0xFFFFFFFF, found);
        }

        group0_found = (ballot & mask0) != 0;
        group1_found = (ballot & mask1) != 0;
        group2_found = (ballot & mask2) != 0;
    }

    // Shuffle results
    int src_lane0 = __ffs(ballot & mask0) - 1;
    int src_lane1 = __ffs(ballot & mask1) - 1;
    int src_lane2 = __ffs(ballot & mask2) - 1;

    out_val0 = group0_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane0) : 0;
    out_val1 = group1_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane1) : 0;
    out_val2 = group2_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane2) : 0;

    return group0_found && group1_found && group2_found;
}

/**
 * @brief Warp-parallel probe of 2 hash tables simultaneously
 *
 * Divides 32 lanes into 2 groups of 16 lanes each.
 */
__device__ __forceinline__ bool warpParallelProbe2Tables(
    uint32_t key0, uint32_t key1,
    const uint32_t* __restrict__ ht0_keys, const uint32_t* __restrict__ ht0_vals, int ht0_size,
    const uint32_t* __restrict__ ht1_keys, const uint32_t* __restrict__ ht1_vals, int ht1_size,
    uint32_t& out_val0, uint32_t& out_val1)
{
    const int lane = threadIdx.x & 31;
    const int group = lane >> 4;          // 0 or 1
    const int lane_in_group = lane & 15;  // 0-15

    uint32_t my_key = (group == 0) ? key0 : key1;
    const uint32_t* ht_keys = (group == 0) ? ht0_keys : ht1_keys;
    const uint32_t* ht_vals = (group == 0) ? ht0_vals : ht1_vals;
    int ht_size = (group == 0) ? ht0_size : ht1_size;

    int base_slot = hash_fast(my_key, ht_size);
    int probe_slot = (base_slot + lane_in_group) % ht_size;
    uint32_t stored_key = ht_keys[probe_slot];
    bool found = (stored_key == my_key);
    uint32_t my_val = found ? ht_vals[probe_slot] : 0;

    unsigned ballot = __ballot_sync(0xFFFFFFFF, found);

    bool group0_found = (ballot & 0x0000FFFF) != 0;
    bool group1_found = (ballot & 0xFFFF0000) != 0;

    if (!group0_found || !group1_found) {
        base_slot = hash_murmur3(my_key, ht_size);
        for (int round = 0; round < 2; round++) {
            unsigned group_mask = (group == 0) ? 0x0000FFFF : 0xFFFF0000;
            if ((ballot & group_mask) != 0) continue;

            probe_slot = (base_slot + round * 16 + lane_in_group) % ht_size;
            stored_key = ht_keys[probe_slot];
            found = (stored_key == my_key);
            if (found) my_val = ht_vals[probe_slot];
            ballot = __ballot_sync(0xFFFFFFFF, found);
        }
        group0_found = (ballot & 0x0000FFFF) != 0;
        group1_found = (ballot & 0xFFFF0000) != 0;
    }

    int src_lane0 = __ffs(ballot & 0x0000FFFF) - 1;
    int src_lane1 = 16 + __ffs((ballot >> 16) & 0xFFFF) - 1;

    out_val0 = group0_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane0) : 0;
    out_val1 = group1_found ? __shfl_sync(0xFFFFFFFF, my_val, src_lane1) : 0;

    return group0_found && group1_found;
}

// ============================================================================
// Specialized Versions for Q4.x Queries (with filter conditions)
// ============================================================================

/**
 * @brief Q4.1/Q4.2/Q4.3 parallel probe with year filter
 *
 * For Q4.x queries that need:
 * - PART: filter by mfgr/category, return category
 * - SUPPLIER: filter by region/nation, return nation/city
 * - CUSTOMER: filter by region/nation, return nation/city
 * - DATE: filter by year range, return year
 */
__device__ __forceinline__ bool warpParallelProbe4TablesQ4(
    uint32_t partkey, uint32_t suppkey, uint32_t custkey, uint32_t datekey,
    const uint32_t* __restrict__ ht_p_keys, const uint32_t* __restrict__ ht_p_vals, int ht_p_size,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_vals, int ht_s_size,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_vals, int ht_c_size,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    uint32_t& out_p_attr, uint32_t& out_s_attr, uint32_t& out_c_attr, uint32_t& out_year)
{
    // First do the parallel probe
    bool all_found = warpParallelProbe4Tables(
        partkey, suppkey, custkey, datekey,
        ht_p_keys, ht_p_vals, ht_p_size,
        ht_s_keys, ht_s_vals, ht_s_size,
        ht_c_keys, ht_c_vals, ht_c_size,
        ht_d_keys, ht_d_vals, ht_d_size,
        out_p_attr, out_s_attr, out_c_attr, out_year);

    if (!all_found) return false;

    // Apply year filter
    if (out_year < year_min || out_year > year_max) return false;

    return true;
}

/**
 * @brief Q3.x parallel probe with year filter
 */
__device__ __forceinline__ bool warpParallelProbe3TablesQ3(
    uint32_t suppkey, uint32_t custkey, uint32_t datekey,
    const uint32_t* __restrict__ ht_s_keys, const uint32_t* __restrict__ ht_s_vals, int ht_s_size,
    const uint32_t* __restrict__ ht_c_keys, const uint32_t* __restrict__ ht_c_vals, int ht_c_size,
    const uint32_t* __restrict__ ht_d_keys, const uint32_t* __restrict__ ht_d_vals, int ht_d_size,
    uint32_t year_min, uint32_t year_max,
    uint32_t& out_s_attr, uint32_t& out_c_attr, uint32_t& out_year)
{
    bool all_found = warpParallelProbe3Tables(
        suppkey, custkey, datekey,
        ht_s_keys, ht_s_vals, ht_s_size,
        ht_c_keys, ht_c_vals, ht_c_size,
        ht_d_keys, ht_d_vals, ht_d_size,
        out_s_attr, out_c_attr, out_year);

    if (!all_found) return false;
    if (out_year < year_min || out_year > year_max) return false;

    return true;
}

}  // namespace ssb
