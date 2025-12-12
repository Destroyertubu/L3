/**
 * @file crystal_hash_build_v7.cuh
 * @brief Crystal-compatible hash table build kernels (Vertical style)
 *
 * These kernels build hash tables using direct slot assignment,
 * matching Vertical' BlockBuildSelectivePHT_1 and BlockBuildSelectivePHT_2.
 *
 * Key differences from L3 V5:
 * - No collision resolution (direct slot assignment)
 * - Table size = expected entries (not 2x for load factor)
 * - Date table uses range-based direct indexing
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "crystal_hash_v7.cuh"

namespace l3_crystal {

// ============================================================================
// Date Hash Table Build (datekey -> year)
// ============================================================================

/**
 * @brief Build date hash table with direct indexing
 *
 * Uses slot = (datekey - 19920101) for direct O(1) lookup.
 * No collisions possible because each datekey maps to unique slot.
 *
 * Hash table structure: ht[slot*2] = datekey, ht[slot*2+1] = year
 *
 * @param d_datekey  Input datekey array
 * @param d_year     Input year array
 * @param num_rows   Number of date rows (D_LEN)
 * @param ht         Output hash table (size: DATE_HT_LEN * 2 * sizeof(uint32_t))
 * @param ht_len     Hash table length (should be DATE_HT_LEN = 60130)
 */
__global__ void build_date_ht_crystal(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_rows,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t key = d_datekey[idx];
    uint32_t val = d_year[idx];

    // Direct slot = datekey offset from minimum
    int slot = CRYSTAL_HASH(key, ht_len, DATE_KEY_MIN);

    // Direct write - no atomicCAS needed (unique slots)
    ht[slot << 1] = key;
    ht[(slot << 1) + 1] = val;
}

// ============================================================================
// Supplier Hash Table Build (filtered by region)
// ============================================================================

/**
 * @brief Build supplier hash table filtered by region (keys only)
 *
 * For Q2.1: s_region = 'AMERICA' (filter_region = 1)
 *
 * Hash table structure: ht[slot] = suppkey (or 0 if empty)
 *
 * @param s_suppkey     Input suppkey array
 * @param s_region      Input region array
 * @param num_rows      Number of supplier rows (S_LEN)
 * @param filter_region Region to filter (e.g., 1 for AMERICA)
 * @param ht            Output hash table (size: S_LEN * sizeof(uint32_t))
 * @param ht_len        Hash table length (should be S_LEN)
 */
__global__ void build_supplier_ht_region_crystal(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_region,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_region[idx] != filter_region) return;

    uint32_t key = s_suppkey[idx];
    int slot = key % ht_len;  // suppkey starts from 1

    // atomicCAS to handle potential collisions (rare for unique suppkeys)
    atomicCAS(&ht[slot], 0, key);
}

/**
 * @brief Build supplier hash table with nation value (suppkey -> nation)
 *
 * For Q3.1: s_region = 'ASIA', need s_nation for grouping
 *
 * Hash table structure: ht[slot*2] = suppkey, ht[slot*2+1] = nation
 */
__global__ void build_supplier_ht_region_nation_crystal(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_region,
    const uint32_t* __restrict__ s_nation,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_region[idx] != filter_region) return;

    uint32_t key = s_suppkey[idx];
    uint32_t val = s_nation[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

// ============================================================================
// Part Hash Table Build (filtered by category)
// ============================================================================

/**
 * @brief Build part hash table filtered by category (partkey -> brand1)
 *
 * For Q2.1: p_category = 'MFGR#12' (filter_category = 12)
 *
 * Hash table structure: ht[slot*2] = partkey, ht[slot*2+1] = brand1
 *
 * @param p_partkey       Input partkey array
 * @param p_category      Input category array
 * @param p_brand1        Input brand1 array
 * @param num_rows        Number of part rows (P_LEN)
 * @param filter_category Category to filter
 * @param ht              Output hash table (size: P_LEN * 2 * sizeof(uint32_t))
 * @param ht_len          Hash table length (should be P_LEN)
 */
__global__ void build_part_ht_category_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_category,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t filter_category,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_category[idx] != filter_category) return;

    uint32_t key = p_partkey[idx];
    uint32_t val = p_brand1[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

/**
 * @brief Build part hash table filtered by brand range (partkey -> brand1)
 *
 * For Q2.2: p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228' (brand 260-267)
 *
 * Hash table structure: ht[slot*2] = partkey, ht[slot*2+1] = brand1
 */
__global__ void build_part_ht_brand_range_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t brand_min,
    uint32_t brand_max,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t brand = p_brand1[idx];
    if (brand < brand_min || brand > brand_max) return;

    uint32_t key = p_partkey[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = brand;
    }
}

/**
 * @brief Build part hash table filtered by exact brand (partkey -> brand1)
 *
 * For Q2.3: p_brand1 = 'MFGR#2239' (brand = 260)
 *
 * Hash table structure: ht[slot*2] = partkey, ht[slot*2+1] = brand1
 */
__global__ void build_part_ht_brand_exact_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t target_brand,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_brand1[idx] != target_brand) return;

    uint32_t key = p_partkey[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = target_brand;
    }
}

/**
 * @brief Build part hash table with mfgr filter (keys only)
 *
 * For Q4.1: p_mfgr in ('MFGR#1', 'MFGR#2')
 */
__global__ void build_part_ht_mfgr_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_mfgr,
    int num_rows,
    uint32_t mfgr1,
    uint32_t mfgr2,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;

    uint32_t key = p_partkey[idx];
    int slot = key % ht_len;

    atomicCAS(&ht[slot], 0, key);
}

// ============================================================================
// Customer Hash Table Build (filtered by region)
// ============================================================================

/**
 * @brief Build customer hash table filtered by region (custkey -> nation)
 *
 * For Q3.1: c_region = 'ASIA'
 *
 * Hash table structure: ht[slot*2] = custkey, ht[slot*2+1] = nation
 */
__global__ void build_customer_ht_region_crystal(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_region,
    const uint32_t* __restrict__ c_nation,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_region[idx] != filter_region) return;

    uint32_t key = c_custkey[idx];
    uint32_t val = c_nation[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

/**
 * @brief Build customer hash table filtered by nation (custkey -> city)
 *
 * For Q3.2/Q3.3: c_nation = 'UNITED STATES' or 'CHINA'
 */
__global__ void build_customer_ht_nation_crystal(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_nation,
    const uint32_t* __restrict__ c_city,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_nation[idx] != filter_nation) return;

    uint32_t key = c_custkey[idx];
    uint32_t val = c_city[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

/**
 * @brief Build customer hash table filtered by city (custkey -> city)
 *
 * For Q3.3/Q3.4: c_city IN ('UNITED KI1', 'UNITED KI5')
 */
__global__ void build_customer_ht_city_crystal(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_city,
    int num_rows,
    uint32_t city1,
    uint32_t city2,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t city = c_city[idx];
    if (city != city1 && city != city2) return;

    uint32_t key = c_custkey[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = city;
    }
}

/**
 * @brief Build supplier hash table filtered by nation (suppkey -> city)
 *
 * For Q3.2: s_nation = 'UNITED STATES'
 */
__global__ void build_supplier_ht_nation_city_crystal(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_nation,
    const uint32_t* __restrict__ s_city,
    int num_rows,
    uint32_t filter_nation,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (s_nation[idx] != filter_nation) return;

    uint32_t key = s_suppkey[idx];
    uint32_t val = s_city[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

/**
 * @brief Build supplier hash table filtered by city (suppkey -> city)
 *
 * For Q3.3/Q3.4: s_city IN ('UNITED KI1', 'UNITED KI5')
 */
__global__ void build_supplier_ht_city_crystal(
    const uint32_t* __restrict__ s_suppkey,
    const uint32_t* __restrict__ s_city,
    int num_rows,
    uint32_t city1,
    uint32_t city2,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t city = s_city[idx];
    if (city != city1 && city != city2) return;

    uint32_t key = s_suppkey[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = city;
    }
}

/**
 * @brief Build date hash table for specific yearmonth (datekey -> year)
 *
 * For Q3.4: d_yearmonth = 'Dec1997'
 */
__global__ void build_date_ht_yearmonth_crystal(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_rows,
    uint32_t yearmonth_start,
    uint32_t yearmonth_end,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t key = d_datekey[idx];
    if (key < yearmonth_start || key > yearmonth_end) return;

    uint32_t val = d_year[idx];
    int slot = CRYSTAL_HASH(key, ht_len, DATE_KEY_MIN);

    ht[slot << 1] = key;
    ht[(slot << 1) + 1] = val;
}

/**
 * @brief Build date hash table for year range (datekey -> year)
 *
 * For Q4.2/Q4.3: d_year IN (1997, 1998)
 */
__global__ void build_date_ht_year_range_crystal(
    const uint32_t* __restrict__ d_datekey,
    const uint32_t* __restrict__ d_year,
    int num_rows,
    uint32_t year_min,
    uint32_t year_max,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t year = d_year[idx];
    if (year < year_min || year > year_max) return;

    uint32_t key = d_datekey[idx];
    int slot = CRYSTAL_HASH(key, ht_len, DATE_KEY_MIN);

    ht[slot << 1] = key;
    ht[(slot << 1) + 1] = year;
}

/**
 * @brief Build customer hash table filtered by region (keys only)
 *
 * For Q4.2: c_region = 'AMERICA'
 */
__global__ void build_customer_ht_region_keys_crystal(
    const uint32_t* __restrict__ c_custkey,
    const uint32_t* __restrict__ c_region,
    int num_rows,
    uint32_t filter_region,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (c_region[idx] != filter_region) return;

    uint32_t key = c_custkey[idx];
    int slot = key % ht_len;

    atomicCAS(&ht[slot], 0, key);
}

/**
 * @brief Build part hash table filtered by mfgr (partkey -> category)
 *
 * For Q4.2: p_mfgr IN ('MFGR#1', 'MFGR#2')
 */
__global__ void build_part_ht_mfgr_category_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_mfgr,
    const uint32_t* __restrict__ p_category,
    int num_rows,
    uint32_t mfgr1,
    uint32_t mfgr2,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    uint32_t mfgr = p_mfgr[idx];
    if (mfgr != mfgr1 && mfgr != mfgr2) return;

    uint32_t key = p_partkey[idx];
    uint32_t val = p_category[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

/**
 * @brief Build part hash table filtered by category (partkey -> brand1)
 *
 * For Q4.3: p_category = 'MFGR#14' (category = 4)
 */
__global__ void build_part_ht_category_brand_crystal(
    const uint32_t* __restrict__ p_partkey,
    const uint32_t* __restrict__ p_category,
    const uint32_t* __restrict__ p_brand1,
    int num_rows,
    uint32_t filter_category,
    uint32_t* __restrict__ ht,
    int ht_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;

    if (p_category[idx] != filter_category) return;

    uint32_t key = p_partkey[idx];
    uint32_t val = p_brand1[idx];
    int slot = key % ht_len;

    uint32_t old = atomicCAS(&ht[slot << 1], 0, key);
    if (old == 0 || old == key) {
        ht[(slot << 1) + 1] = val;
    }
}

// ============================================================================
// Hash Table Allocation Helper
// ============================================================================

/**
 * @brief Crystal-style hash table (matches Vertical layout)
 */
struct CrystalHashTable {
    uint32_t* d_data;   // GPU memory (interleaved [key,val] or keys-only)
    int len;            // Number of slots
    bool has_values;    // true = key-value pairs, false = keys only

    void allocate(int table_len, bool with_values = true) {
        len = table_len;
        has_values = with_values;
        int num_elements = with_values ? (len * 2) : len;
        cudaMalloc(&d_data, num_elements * sizeof(uint32_t));
        cudaMemset(d_data, 0, num_elements * sizeof(uint32_t));
    }

    void free() {
        if (d_data) {
            cudaFree(d_data);
            d_data = nullptr;
        }
    }
};

}  // namespace l3_crystal
