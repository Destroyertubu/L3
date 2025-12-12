/**
 * @file ssb_data_loader.hpp
 * @brief SSB Data Loader for L3 Compressed Queries
 *
 * Loads SSB binary data files and provides interfaces for:
 * - Raw data loading (for baseline comparison)
 * - L3 compression of LINEORDER columns
 * - GPU memory management
 */

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

// L3 compression headers
#include "L3_format.hpp"
#include "L3_codec.hpp"
#include "L3_Vertical_api.hpp"
#include "L3_Vertical_format.hpp"

namespace ssb {

// ============================================================================
// Table Size Constants (SF=10)
// ============================================================================

constexpr size_t LO_LEN = 59986214;   // LINEORDER rows
constexpr size_t P_LEN = 800000;      // PART rows
constexpr size_t S_LEN = 20000;       // SUPPLIER rows
constexpr size_t C_LEN = 300000;      // CUSTOMER rows
constexpr size_t D_LEN = 2557;        // DATE rows

// ============================================================================
// Column File Names
// ============================================================================

// LINEORDER columns
constexpr const char* LO_ORDERKEY      = "LINEORDER0.bin";
constexpr const char* LO_LINENUMBER    = "LINEORDER1.bin";
constexpr const char* LO_CUSTKEY       = "LINEORDER2.bin";
constexpr const char* LO_PARTKEY       = "LINEORDER3.bin";
constexpr const char* LO_SUPPKEY       = "LINEORDER4.bin";
constexpr const char* LO_ORDERDATE     = "LINEORDER5.bin";
constexpr const char* LO_ORDERPRIORITY = "LINEORDER6.bin";
constexpr const char* LO_SHIPPRIORITY  = "LINEORDER7.bin";
constexpr const char* LO_QUANTITY      = "LINEORDER8.bin";
constexpr const char* LO_EXTENDEDPRICE = "LINEORDER9.bin";
constexpr const char* LO_ORDTOTALPRICE = "LINEORDER10.bin";
constexpr const char* LO_DISCOUNT      = "LINEORDER11.bin";
constexpr const char* LO_REVENUE       = "LINEORDER12.bin";
constexpr const char* LO_SUPPLYCOST    = "LINEORDER13.bin";
constexpr const char* LO_TAX           = "LINEORDER14.bin";
constexpr const char* LO_COMMITDATE    = "LINEORDER15.bin";
constexpr const char* LO_SHIPMODE      = "LINEORDER16.bin";

// PART columns
constexpr const char* P_PARTKEY   = "PART0.bin";
constexpr const char* P_MFGR      = "PART2.bin";
constexpr const char* P_CATEGORY  = "PART3.bin";
constexpr const char* P_BRAND1    = "PART4.bin";
constexpr const char* P_COLOR     = "PART5.bin";
constexpr const char* P_TYPE      = "PART6.bin";
constexpr const char* P_SIZE      = "PART7.bin";
constexpr const char* P_CONTAINER = "PART8.bin";

// SUPPLIER columns
constexpr const char* S_SUPPKEY = "SUPPLIER0.bin";
constexpr const char* S_CITY    = "SUPPLIER3.bin";
constexpr const char* S_NATION  = "SUPPLIER4.bin";
constexpr const char* S_REGION  = "SUPPLIER5.bin";

// CUSTOMER columns
constexpr const char* C_CUSTKEY    = "CUSTOMER0.bin";
constexpr const char* C_CITY       = "CUSTOMER3.bin";
constexpr const char* C_NATION     = "CUSTOMER4.bin";
constexpr const char* C_REGION     = "CUSTOMER5.bin";
constexpr const char* C_MKTSEGMENT = "CUSTOMER7.bin";

// DATE columns
constexpr const char* D_DATEKEY        = "DDATE0.bin";
constexpr const char* D_YEAR           = "DDATE4.bin";
constexpr const char* D_YEARMONTHNUM   = "DDATE5.bin";
constexpr const char* D_DAYNUMINWEEK   = "DDATE7.bin";
constexpr const char* D_DAYNUMINMONTH  = "DDATE8.bin";
constexpr const char* D_DAYNUMINYEAR   = "DDATE9.bin";
constexpr const char* D_MONTHNUMINYEAR = "DDATE10.bin";
constexpr const char* D_WEEKNUMINYEAR  = "DDATE11.bin";

// ============================================================================
// Data Loading Utilities
// ============================================================================

/**
 * @brief Load a binary column file from disk
 */
template<typename T>
std::vector<T> loadColumnFromFile(const std::string& filepath, size_t expected_rows = 0) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(T);

    if (expected_rows > 0 && num_elements != expected_rows) {
        std::cerr << "Warning: " << filepath << " has " << num_elements
                  << " elements, expected " << expected_rows << std::endl;
    }

    file.seekg(0, std::ios::beg);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return data;
}

/**
 * @brief Load column and transfer to GPU
 */
template<typename T>
T* loadColumnToGPU(const std::string& filepath, size_t expected_rows = 0) {
    auto host_data = loadColumnFromFile<T>(filepath, expected_rows);

    T* d_data;
    cudaMalloc(&d_data, host_data.size() * sizeof(T));
    cudaMemcpy(d_data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);

    return d_data;
}

// ============================================================================
// Raw SSB Data Structure (Uncompressed)
// ============================================================================

struct SSBDataRaw {
    // LINEORDER columns on GPU
    uint32_t* d_lo_orderkey;
    uint32_t* d_lo_linenumber;
    uint32_t* d_lo_custkey;
    uint32_t* d_lo_partkey;
    uint32_t* d_lo_suppkey;
    uint32_t* d_lo_orderdate;
    uint32_t* d_lo_orderpriority;
    uint32_t* d_lo_shippriority;
    uint32_t* d_lo_quantity;
    uint32_t* d_lo_extendedprice;
    uint32_t* d_lo_ordtotalprice;
    uint32_t* d_lo_discount;
    uint32_t* d_lo_revenue;
    uint32_t* d_lo_supplycost;
    uint32_t* d_lo_tax;
    uint32_t* d_lo_commitdate;
    uint32_t* d_lo_shipmode;

    // PART columns on GPU
    uint32_t* d_p_partkey;
    uint32_t* d_p_mfgr;
    uint32_t* d_p_category;
    uint32_t* d_p_brand1;
    uint32_t* d_p_color;
    uint32_t* d_p_type;
    uint32_t* d_p_size;
    uint32_t* d_p_container;

    // SUPPLIER columns on GPU
    uint32_t* d_s_suppkey;
    uint32_t* d_s_city;
    uint32_t* d_s_nation;
    uint32_t* d_s_region;

    // CUSTOMER columns on GPU
    uint32_t* d_c_custkey;
    uint32_t* d_c_city;
    uint32_t* d_c_nation;
    uint32_t* d_c_region;
    uint32_t* d_c_mktsegment;

    // DATE columns on GPU
    uint32_t* d_d_datekey;
    uint32_t* d_d_year;
    uint32_t* d_d_yearmonthnum;

    bool is_loaded = false;

    /**
     * @brief Load all SSB tables from data directory
     */
    void load(const std::string& data_dir) {
        std::cout << "Loading SSB data from " << data_dir << std::endl;

        // Load LINEORDER (largest table)
        std::cout << "  Loading LINEORDER (" << LO_LEN << " rows)..." << std::endl;
        d_lo_orderkey      = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_ORDERKEY, LO_LEN);
        d_lo_linenumber    = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_LINENUMBER, LO_LEN);
        d_lo_custkey       = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_CUSTKEY, LO_LEN);
        d_lo_partkey       = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_PARTKEY, LO_LEN);
        d_lo_suppkey       = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_SUPPKEY, LO_LEN);
        d_lo_orderdate     = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_ORDERDATE, LO_LEN);
        d_lo_orderpriority = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_ORDERPRIORITY, LO_LEN);
        d_lo_shippriority  = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_SHIPPRIORITY, LO_LEN);
        d_lo_quantity      = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_QUANTITY, LO_LEN);
        d_lo_extendedprice = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_EXTENDEDPRICE, LO_LEN);
        d_lo_ordtotalprice = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_ORDTOTALPRICE, LO_LEN);
        d_lo_discount      = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_DISCOUNT, LO_LEN);
        d_lo_revenue       = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_REVENUE, LO_LEN);
        d_lo_supplycost    = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_SUPPLYCOST, LO_LEN);
        d_lo_tax           = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_TAX, LO_LEN);
        d_lo_commitdate    = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_COMMITDATE, LO_LEN);
        d_lo_shipmode      = loadColumnToGPU<uint32_t>(data_dir + "/" + LO_SHIPMODE, LO_LEN);

        // Load PART
        std::cout << "  Loading PART (" << P_LEN << " rows)..." << std::endl;
        d_p_partkey   = loadColumnToGPU<uint32_t>(data_dir + "/" + P_PARTKEY, P_LEN);
        d_p_mfgr      = loadColumnToGPU<uint32_t>(data_dir + "/" + P_MFGR, P_LEN);
        d_p_category  = loadColumnToGPU<uint32_t>(data_dir + "/" + P_CATEGORY, P_LEN);
        d_p_brand1    = loadColumnToGPU<uint32_t>(data_dir + "/" + P_BRAND1, P_LEN);
        d_p_color     = loadColumnToGPU<uint32_t>(data_dir + "/" + P_COLOR, P_LEN);
        d_p_type      = loadColumnToGPU<uint32_t>(data_dir + "/" + P_TYPE, P_LEN);
        d_p_size      = loadColumnToGPU<uint32_t>(data_dir + "/" + P_SIZE, P_LEN);
        d_p_container = loadColumnToGPU<uint32_t>(data_dir + "/" + P_CONTAINER, P_LEN);

        // Load SUPPLIER
        std::cout << "  Loading SUPPLIER (" << S_LEN << " rows)..." << std::endl;
        d_s_suppkey = loadColumnToGPU<uint32_t>(data_dir + "/" + S_SUPPKEY, S_LEN);
        d_s_city    = loadColumnToGPU<uint32_t>(data_dir + "/" + S_CITY, S_LEN);
        d_s_nation  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_NATION, S_LEN);
        d_s_region  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_REGION, S_LEN);

        // Load CUSTOMER
        std::cout << "  Loading CUSTOMER (" << C_LEN << " rows)..." << std::endl;
        d_c_custkey    = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CUSTKEY, C_LEN);
        d_c_city       = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CITY, C_LEN);
        d_c_nation     = loadColumnToGPU<uint32_t>(data_dir + "/" + C_NATION, C_LEN);
        d_c_region     = loadColumnToGPU<uint32_t>(data_dir + "/" + C_REGION, C_LEN);
        d_c_mktsegment = loadColumnToGPU<uint32_t>(data_dir + "/" + C_MKTSEGMENT, C_LEN);

        // Load DATE
        std::cout << "  Loading DATE (" << D_LEN << " rows)..." << std::endl;
        d_d_datekey      = loadColumnToGPU<uint32_t>(data_dir + "/" + D_DATEKEY, D_LEN);
        d_d_year         = loadColumnToGPU<uint32_t>(data_dir + "/" + D_YEAR, D_LEN);
        d_d_yearmonthnum = loadColumnToGPU<uint32_t>(data_dir + "/" + D_YEARMONTHNUM, D_LEN);

        is_loaded = true;
        std::cout << "SSB data loaded successfully." << std::endl;
    }

    /**
     * @brief Free all GPU memory
     */
    void free() {
        if (!is_loaded) return;

        // LINEORDER
        cudaFree(d_lo_orderkey);
        cudaFree(d_lo_linenumber);
        cudaFree(d_lo_custkey);
        cudaFree(d_lo_partkey);
        cudaFree(d_lo_suppkey);
        cudaFree(d_lo_orderdate);
        cudaFree(d_lo_orderpriority);
        cudaFree(d_lo_shippriority);
        cudaFree(d_lo_quantity);
        cudaFree(d_lo_extendedprice);
        cudaFree(d_lo_ordtotalprice);
        cudaFree(d_lo_discount);
        cudaFree(d_lo_revenue);
        cudaFree(d_lo_supplycost);
        cudaFree(d_lo_tax);
        cudaFree(d_lo_commitdate);
        cudaFree(d_lo_shipmode);

        // PART
        cudaFree(d_p_partkey);
        cudaFree(d_p_mfgr);
        cudaFree(d_p_category);
        cudaFree(d_p_brand1);
        cudaFree(d_p_color);
        cudaFree(d_p_type);
        cudaFree(d_p_size);
        cudaFree(d_p_container);

        // SUPPLIER
        cudaFree(d_s_suppkey);
        cudaFree(d_s_city);
        cudaFree(d_s_nation);
        cudaFree(d_s_region);

        // CUSTOMER
        cudaFree(d_c_custkey);
        cudaFree(d_c_city);
        cudaFree(d_c_nation);
        cudaFree(d_c_region);
        cudaFree(d_c_mktsegment);

        // DATE
        cudaFree(d_d_datekey);
        cudaFree(d_d_year);
        cudaFree(d_d_yearmonthnum);

        is_loaded = false;
    }

    /**
     * @brief Get total GPU memory usage in bytes
     */
    size_t getGPUMemoryUsage() const {
        if (!is_loaded) return 0;

        size_t total = 0;
        total += LO_LEN * 17 * sizeof(uint32_t);  // LINEORDER
        total += P_LEN * 8 * sizeof(uint32_t);    // PART
        total += S_LEN * 4 * sizeof(uint32_t);    // SUPPLIER
        total += C_LEN * 5 * sizeof(uint32_t);    // CUSTOMER
        total += D_LEN * 3 * sizeof(uint32_t);    // DATE

        return total;
    }
};

// ============================================================================
// Compressed SSB Data Structure (L3 Compressed)
// ============================================================================

struct SSBDataCompressed {
    // Compressed LINEORDER columns (using L3 L3 format)
    CompressedDataL3<uint32_t>* lo_orderdate;
    CompressedDataL3<uint32_t>* lo_quantity;
    CompressedDataL3<uint32_t>* lo_extendedprice;
    CompressedDataL3<uint32_t>* lo_discount;
    CompressedDataL3<uint32_t>* lo_revenue;
    CompressedDataL3<uint32_t>* lo_supplycost;
    CompressedDataL3<uint32_t>* lo_custkey;
    CompressedDataL3<uint32_t>* lo_partkey;
    CompressedDataL3<uint32_t>* lo_suppkey;

    // Dimension tables (kept uncompressed for hash table building)
    // These are copied from SSBDataRaw
    uint32_t* d_p_partkey;
    uint32_t* d_p_mfgr;
    uint32_t* d_p_category;
    uint32_t* d_p_brand1;

    uint32_t* d_s_suppkey;
    uint32_t* d_s_city;
    uint32_t* d_s_nation;
    uint32_t* d_s_region;

    uint32_t* d_c_custkey;
    uint32_t* d_c_city;
    uint32_t* d_c_nation;
    uint32_t* d_c_region;

    uint32_t* d_d_datekey;
    uint32_t* d_d_year;

    bool is_loaded = false;

    /**
     * @brief Load and compress SSB data
     */
    void loadAndCompress(const std::string& data_dir, L3Config config = L3Config::costOptimal(2048)) {
        std::cout << "Loading and compressing SSB data from " << data_dir << std::endl;

        // Load and compress key LINEORDER columns
        std::cout << "  Compressing LINEORDER columns..." << std::endl;

        auto compress_column = [&](const char* filename, const char* name) -> CompressedDataL3<uint32_t>* {
            std::cout << "    - " << name << "..." << std::flush;
            auto host_data = loadColumnFromFile<uint32_t>(data_dir + "/" + filename, LO_LEN);

            CompressionStats stats;
            CompressedDataL3<uint32_t>* compressed = compressDataWithConfig(host_data, config, &stats);

            double ratio = stats.compression_ratio;
            std::cout << " done (ratio: " << ratio << "x)" << std::endl;
            return compressed;
        };

        lo_orderdate     = compress_column(LO_ORDERDATE, "lo_orderdate");
        lo_quantity      = compress_column(LO_QUANTITY, "lo_quantity");
        lo_extendedprice = compress_column(LO_EXTENDEDPRICE, "lo_extendedprice");
        lo_discount      = compress_column(LO_DISCOUNT, "lo_discount");
        lo_revenue       = compress_column(LO_REVENUE, "lo_revenue");
        lo_supplycost    = compress_column(LO_SUPPLYCOST, "lo_supplycost");
        lo_custkey       = compress_column(LO_CUSTKEY, "lo_custkey");
        lo_partkey       = compress_column(LO_PARTKEY, "lo_partkey");
        lo_suppkey       = compress_column(LO_SUPPKEY, "lo_suppkey");

        // Load dimension tables (uncompressed for hash table building)
        std::cout << "  Loading dimension tables (uncompressed)..." << std::endl;
        d_p_partkey  = loadColumnToGPU<uint32_t>(data_dir + "/" + P_PARTKEY, P_LEN);
        d_p_mfgr     = loadColumnToGPU<uint32_t>(data_dir + "/" + P_MFGR, P_LEN);
        d_p_category = loadColumnToGPU<uint32_t>(data_dir + "/" + P_CATEGORY, P_LEN);
        d_p_brand1   = loadColumnToGPU<uint32_t>(data_dir + "/" + P_BRAND1, P_LEN);

        d_s_suppkey = loadColumnToGPU<uint32_t>(data_dir + "/" + S_SUPPKEY, S_LEN);
        d_s_city    = loadColumnToGPU<uint32_t>(data_dir + "/" + S_CITY, S_LEN);
        d_s_nation  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_NATION, S_LEN);
        d_s_region  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_REGION, S_LEN);

        d_c_custkey = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CUSTKEY, C_LEN);
        d_c_city    = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CITY, C_LEN);
        d_c_nation  = loadColumnToGPU<uint32_t>(data_dir + "/" + C_NATION, C_LEN);
        d_c_region  = loadColumnToGPU<uint32_t>(data_dir + "/" + C_REGION, C_LEN);

        d_d_datekey = loadColumnToGPU<uint32_t>(data_dir + "/" + D_DATEKEY, D_LEN);
        d_d_year    = loadColumnToGPU<uint32_t>(data_dir + "/" + D_YEAR, D_LEN);

        is_loaded = true;
        std::cout << "SSB compressed data loaded successfully." << std::endl;
    }

    /**
     * @brief Free all resources
     */
    void free() {
        if (!is_loaded) return;

        // Free compressed columns
        freeCompressedData(lo_orderdate);
        freeCompressedData(lo_quantity);
        freeCompressedData(lo_extendedprice);
        freeCompressedData(lo_discount);
        freeCompressedData(lo_revenue);
        freeCompressedData(lo_supplycost);
        freeCompressedData(lo_custkey);
        freeCompressedData(lo_partkey);
        freeCompressedData(lo_suppkey);

        // Free dimension table GPU memory
        cudaFree(d_p_partkey);
        cudaFree(d_p_mfgr);
        cudaFree(d_p_category);
        cudaFree(d_p_brand1);

        cudaFree(d_s_suppkey);
        cudaFree(d_s_city);
        cudaFree(d_s_nation);
        cudaFree(d_s_region);

        cudaFree(d_c_custkey);
        cudaFree(d_c_city);
        cudaFree(d_c_nation);
        cudaFree(d_c_region);

        cudaFree(d_d_datekey);
        cudaFree(d_d_year);

        is_loaded = false;
    }

    /**
     * @brief Get total compressed size in bytes
     */
    size_t getCompressedSize() const {
        if (!is_loaded) return 0;

        size_t total = 0;
        total += lo_orderdate->delta_array_words * sizeof(uint32_t);
        total += lo_quantity->delta_array_words * sizeof(uint32_t);
        total += lo_extendedprice->delta_array_words * sizeof(uint32_t);
        total += lo_discount->delta_array_words * sizeof(uint32_t);
        total += lo_revenue->delta_array_words * sizeof(uint32_t);
        total += lo_supplycost->delta_array_words * sizeof(uint32_t);
        total += lo_custkey->delta_array_words * sizeof(uint32_t);
        total += lo_partkey->delta_array_words * sizeof(uint32_t);
        total += lo_suppkey->delta_array_words * sizeof(uint32_t);

        // Add dimension tables
        total += P_LEN * 4 * sizeof(uint32_t);
        total += S_LEN * 4 * sizeof(uint32_t);
        total += C_LEN * 4 * sizeof(uint32_t);
        total += D_LEN * 2 * sizeof(uint32_t);

        return total;
    }

    /**
     * @brief Get compression ratio
     */
    double getCompressionRatio() const {
        size_t original = LO_LEN * 9 * sizeof(uint32_t);  // 9 compressed columns
        size_t compressed = lo_orderdate->delta_array_words * sizeof(uint32_t) +
                           lo_quantity->delta_array_words * sizeof(uint32_t) +
                           lo_extendedprice->delta_array_words * sizeof(uint32_t) +
                           lo_discount->delta_array_words * sizeof(uint32_t) +
                           lo_revenue->delta_array_words * sizeof(uint32_t) +
                           lo_supplycost->delta_array_words * sizeof(uint32_t) +
                           lo_custkey->delta_array_words * sizeof(uint32_t) +
                           lo_partkey->delta_array_words * sizeof(uint32_t) +
                           lo_suppkey->delta_array_words * sizeof(uint32_t);
        return (double)original / compressed;
    }
};

// ============================================================================
// Vertical Compressed SSB Data Structure (Interleaved Format)
// ============================================================================

struct SSBDataCompressedVertical {
    // Compressed LINEORDER columns (using L3 Vertical format)
    CompressedDataVertical<uint32_t> lo_orderdate;
    CompressedDataVertical<uint32_t> lo_quantity;
    CompressedDataVertical<uint32_t> lo_extendedprice;
    CompressedDataVertical<uint32_t> lo_discount;
    CompressedDataVertical<uint32_t> lo_revenue;
    CompressedDataVertical<uint32_t> lo_supplycost;
    CompressedDataVertical<uint32_t> lo_custkey;
    CompressedDataVertical<uint32_t> lo_partkey;
    CompressedDataVertical<uint32_t> lo_suppkey;

    // Dimension tables (kept uncompressed for hash table building)
    uint32_t* d_p_partkey;
    uint32_t* d_p_mfgr;
    uint32_t* d_p_category;
    uint32_t* d_p_brand1;

    uint32_t* d_s_suppkey;
    uint32_t* d_s_city;
    uint32_t* d_s_nation;
    uint32_t* d_s_region;

    uint32_t* d_c_custkey;
    uint32_t* d_c_city;
    uint32_t* d_c_nation;
    uint32_t* d_c_region;

    uint32_t* d_d_datekey;
    uint32_t* d_d_year;

    bool is_loaded = false;

    /**
     * @brief Load and compress SSB data using Vertical encoder
     */
    void loadAndCompress(const std::string& data_dir, int partition_size = 4096) {
        std::cout << "Loading and compressing SSB data with Vertical from " << data_dir << std::endl;

        VerticalConfig config = VerticalConfig::costOptimal();
        config.partition_size_hint = partition_size;
        config.enable_interleaved = true;
        // Ensure Cost-Optimal partitioning is enabled
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_target_partition_size = partition_size;
        config.cost_min_partition_size = 256;
        config.cost_max_partition_size = partition_size * 2;

        std::cout << "  Partitioning strategy: COST_OPTIMAL" << std::endl;
        std::cout << "  Target partition size: " << partition_size << std::endl;

        auto compress_column = [&](const char* filename, const char* name) -> CompressedDataVertical<uint32_t> {
            std::cout << "    - " << name << "..." << std::flush;
            auto host_data = loadColumnFromFile<uint32_t>(data_dir + "/" + filename, LO_LEN);

            // Use Vertical encoder
            CompressedDataVertical<uint32_t> compressed =
                Vertical_encoder::encodeVerticalGPU<uint32_t>(host_data, partition_size, config, 0);

            double ratio = (double)(host_data.size() * sizeof(uint32_t)) /
                          (compressed.interleaved_delta_words * sizeof(uint32_t));
            std::cout << " done (ratio: " << ratio << "x, partitions: "
                      << compressed.num_partitions << ")" << std::endl;
            return compressed;
        };

        // Compress LINEORDER columns
        std::cout << "  Compressing LINEORDER columns with Vertical encoder..." << std::endl;
        lo_orderdate     = compress_column(LO_ORDERDATE, "lo_orderdate");
        lo_quantity      = compress_column(LO_QUANTITY, "lo_quantity");
        lo_extendedprice = compress_column(LO_EXTENDEDPRICE, "lo_extendedprice");
        lo_discount      = compress_column(LO_DISCOUNT, "lo_discount");
        lo_revenue       = compress_column(LO_REVENUE, "lo_revenue");
        lo_supplycost    = compress_column(LO_SUPPLYCOST, "lo_supplycost");
        lo_custkey       = compress_column(LO_CUSTKEY, "lo_custkey");
        lo_partkey       = compress_column(LO_PARTKEY, "lo_partkey");
        lo_suppkey       = compress_column(LO_SUPPKEY, "lo_suppkey");

        // Load dimension tables (uncompressed for hash table building)
        std::cout << "  Loading dimension tables (uncompressed)..." << std::endl;
        d_p_partkey  = loadColumnToGPU<uint32_t>(data_dir + "/" + P_PARTKEY, P_LEN);
        d_p_mfgr     = loadColumnToGPU<uint32_t>(data_dir + "/" + P_MFGR, P_LEN);
        d_p_category = loadColumnToGPU<uint32_t>(data_dir + "/" + P_CATEGORY, P_LEN);
        d_p_brand1   = loadColumnToGPU<uint32_t>(data_dir + "/" + P_BRAND1, P_LEN);

        d_s_suppkey = loadColumnToGPU<uint32_t>(data_dir + "/" + S_SUPPKEY, S_LEN);
        d_s_city    = loadColumnToGPU<uint32_t>(data_dir + "/" + S_CITY, S_LEN);
        d_s_nation  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_NATION, S_LEN);
        d_s_region  = loadColumnToGPU<uint32_t>(data_dir + "/" + S_REGION, S_LEN);

        d_c_custkey = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CUSTKEY, C_LEN);
        d_c_city    = loadColumnToGPU<uint32_t>(data_dir + "/" + C_CITY, C_LEN);
        d_c_nation  = loadColumnToGPU<uint32_t>(data_dir + "/" + C_NATION, C_LEN);
        d_c_region  = loadColumnToGPU<uint32_t>(data_dir + "/" + C_REGION, C_LEN);

        d_d_datekey = loadColumnToGPU<uint32_t>(data_dir + "/" + D_DATEKEY, D_LEN);
        d_d_year    = loadColumnToGPU<uint32_t>(data_dir + "/" + D_YEAR, D_LEN);

        is_loaded = true;
        std::cout << "SSB Vertical compressed data loaded successfully." << std::endl;
    }

    /**
     * @brief Free all resources
     */
    void free() {
        if (!is_loaded) return;

        // Free compressed columns
        Vertical_encoder::freeCompressedData(lo_orderdate);
        Vertical_encoder::freeCompressedData(lo_quantity);
        Vertical_encoder::freeCompressedData(lo_extendedprice);
        Vertical_encoder::freeCompressedData(lo_discount);
        Vertical_encoder::freeCompressedData(lo_revenue);
        Vertical_encoder::freeCompressedData(lo_supplycost);
        Vertical_encoder::freeCompressedData(lo_custkey);
        Vertical_encoder::freeCompressedData(lo_partkey);
        Vertical_encoder::freeCompressedData(lo_suppkey);

        // Free dimension table GPU memory
        cudaFree(d_p_partkey);
        cudaFree(d_p_mfgr);
        cudaFree(d_p_category);
        cudaFree(d_p_brand1);

        cudaFree(d_s_suppkey);
        cudaFree(d_s_city);
        cudaFree(d_s_nation);
        cudaFree(d_s_region);

        cudaFree(d_c_custkey);
        cudaFree(d_c_city);
        cudaFree(d_c_nation);
        cudaFree(d_c_region);

        cudaFree(d_d_datekey);
        cudaFree(d_d_year);

        is_loaded = false;
    }

    /**
     * @brief Get total compressed size in bytes
     */
    size_t getCompressedSize() const {
        if (!is_loaded) return 0;

        size_t total = 0;
        total += lo_orderdate.interleaved_delta_words * sizeof(uint32_t);
        total += lo_quantity.interleaved_delta_words * sizeof(uint32_t);
        total += lo_extendedprice.interleaved_delta_words * sizeof(uint32_t);
        total += lo_discount.interleaved_delta_words * sizeof(uint32_t);
        total += lo_revenue.interleaved_delta_words * sizeof(uint32_t);
        total += lo_supplycost.interleaved_delta_words * sizeof(uint32_t);
        total += lo_custkey.interleaved_delta_words * sizeof(uint32_t);
        total += lo_partkey.interleaved_delta_words * sizeof(uint32_t);
        total += lo_suppkey.interleaved_delta_words * sizeof(uint32_t);

        // Add dimension tables
        total += P_LEN * 4 * sizeof(uint32_t);
        total += S_LEN * 4 * sizeof(uint32_t);
        total += C_LEN * 4 * sizeof(uint32_t);
        total += D_LEN * 2 * sizeof(uint32_t);

        return total;
    }

    /**
     * @brief Get compression ratio
     */
    double getCompressionRatio() const {
        size_t original = LO_LEN * 9 * sizeof(uint32_t);  // 9 compressed columns
        size_t compressed = lo_orderdate.interleaved_delta_words * sizeof(uint32_t) +
                           lo_quantity.interleaved_delta_words * sizeof(uint32_t) +
                           lo_extendedprice.interleaved_delta_words * sizeof(uint32_t) +
                           lo_discount.interleaved_delta_words * sizeof(uint32_t) +
                           lo_revenue.interleaved_delta_words * sizeof(uint32_t) +
                           lo_supplycost.interleaved_delta_words * sizeof(uint32_t) +
                           lo_custkey.interleaved_delta_words * sizeof(uint32_t) +
                           lo_partkey.interleaved_delta_words * sizeof(uint32_t) +
                           lo_suppkey.interleaved_delta_words * sizeof(uint32_t);
        return (double)original / compressed;
    }
};

// ============================================================================
// Query Result Structures
// ============================================================================

struct Q1Result {
    unsigned long long revenue;
};

struct Q2Result {
    std::vector<uint32_t> d_year;
    std::vector<uint32_t> p_brand1;
    std::vector<unsigned long long> revenue;
};

struct Q3Result {
    std::vector<uint32_t> c_nation;
    std::vector<uint32_t> s_nation;
    std::vector<uint32_t> d_year;
    std::vector<unsigned long long> revenue;
};

struct Q4Result {
    std::vector<uint32_t> d_year;
    std::vector<uint32_t> c_nation;
    std::vector<uint32_t> p_category;
    std::vector<long long> profit;  // Can be negative
};

// ============================================================================
// Timing Utilities
// ============================================================================

struct QueryTiming {
    float data_load_ms;
    float hash_build_ms;
    float kernel_ms;
    float total_ms;

    void print(const std::string& query_name) const {
        std::cout << query_name << " timing:" << std::endl;
        std::cout << "  Data load:   " << data_load_ms << " ms" << std::endl;
        std::cout << "  Hash build:  " << hash_build_ms << " ms" << std::endl;
        std::cout << "  Kernel:      " << kernel_ms << " ms" << std::endl;
        std::cout << "  Total:       " << total_ms << " ms" << std::endl;
    }
};

/**
 * @brief CUDA event-based timer
 */
class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_, stream);
    }

    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_, stream);
        cudaEventSynchronize(stop_);
    }

    float elapsed_ms() const {
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

}  // namespace ssb
