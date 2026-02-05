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
// Table Size Constants (SF=20)
// ============================================================================

constexpr size_t LO_LEN = 120000000;  // LINEORDER rows (SF=20)
constexpr size_t P_LEN = 1000000;     // PART rows
constexpr size_t S_LEN = 40000;       // SUPPLIER rows
constexpr size_t C_LEN = 600000;      // CUSTOMER rows
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
     * @brief Convert variable-length model params to fixed pid*4 layout
     *
     * The encoder may use variable-length params (e.g., FOR_BITPACK only has 1 param).
     * The decoder and cache save/load assume fixed [num_partitions * 4] layout.
     * This function converts variable layout to fixed layout.
     */
    static void normalizeParamsToFixed(CompressedDataVertical<uint32_t>& col) {
        if (!col.use_variable_params || col.d_param_offsets == nullptr) return;

        int np = col.num_partitions;

        // Read variable params, offsets, model_types, delta_bits from GPU
        std::vector<double> h_var_params(col.total_param_count);
        std::vector<int64_t> h_offsets(np);
        std::vector<int32_t> h_model_types(np);
        std::vector<int32_t> h_delta_bits(np);

        cudaMemcpy(h_var_params.data(), col.d_model_params,
                   col.total_param_count * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_offsets.data(), col.d_param_offsets,
                   np * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_model_types.data(), col.d_model_types,
                   np * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_delta_bits.data(), col.d_delta_bits,
                   np * sizeof(int32_t), cudaMemcpyDeviceToHost);

        // Build fixed layout: pid * 4
        std::vector<double> h_fixed_params(np * 4, 0.0);
        for (int pid = 0; pid < np; pid++) {
            int64_t base = h_offsets[pid];
            int count = getParamCount(h_model_types[pid], h_delta_bits[pid]);
            for (int i = 0; i < count && i < 4; i++) {
                h_fixed_params[pid * 4 + i] = h_var_params[base + i];
            }
        }

        // Reallocate GPU buffer with fixed layout
        cudaFree(col.d_model_params);
        cudaMalloc(&col.d_model_params, np * 4 * sizeof(double));
        cudaMemcpy(col.d_model_params, h_fixed_params.data(),
                   np * 4 * sizeof(double), cudaMemcpyHostToDevice);

        // Free offsets and update flags
        cudaFree(col.d_param_offsets);
        col.d_param_offsets = nullptr;
        col.use_variable_params = false;
        col.total_param_count = np * 4;
    }

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
        // Use partition_size as minimum to ensure decoder compatibility (V11 needs 1024)
        config.cost_min_partition_size = partition_size;
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

        // Convert variable params to fixed pid*4 layout for decoder/cache compatibility
        normalizeParamsToFixed(lo_orderdate);
        normalizeParamsToFixed(lo_quantity);
        normalizeParamsToFixed(lo_extendedprice);
        normalizeParamsToFixed(lo_discount);
        normalizeParamsToFixed(lo_revenue);
        normalizeParamsToFixed(lo_supplycost);
        normalizeParamsToFixed(lo_custkey);
        normalizeParamsToFixed(lo_partkey);
        normalizeParamsToFixed(lo_suppkey);
    }

    /**
     * @brief Measure H2D transfer time for compressed data + dimension tables
     *
     * This copies compressed lineorder data back to CPU, then re-transfers
     * and times the H2D operation for fair comparison with FSL-GPU.
     */
    float measureH2DTime() {
        if (!is_loaded) return 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // 1. Copy compressed lineorder deltas to CPU
        size_t lo_bytes = 0;
        std::vector<std::vector<uint32_t>> h_lo_deltas;

        auto copy_deltas_to_host = [&](const CompressedDataVertical<uint32_t>& col) {
            std::vector<uint32_t> h_data(col.interleaved_delta_words);
            cudaMemcpy(h_data.data(), col.d_interleaved_deltas,
                       col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            lo_bytes += col.interleaved_delta_words * sizeof(uint32_t);
            h_lo_deltas.push_back(std::move(h_data));
        };

        copy_deltas_to_host(lo_orderdate);
        copy_deltas_to_host(lo_quantity);
        copy_deltas_to_host(lo_extendedprice);
        copy_deltas_to_host(lo_discount);
        copy_deltas_to_host(lo_revenue);
        copy_deltas_to_host(lo_supplycost);
        copy_deltas_to_host(lo_custkey);
        copy_deltas_to_host(lo_partkey);
        copy_deltas_to_host(lo_suppkey);

        // 2. Copy dimension tables from GPU to CPU for fair H2D timing
        size_t dim_bytes = 0;

        // PART table (4 columns)
        std::vector<uint32_t> h_p_partkey(P_LEN), h_p_mfgr(P_LEN), h_p_category(P_LEN), h_p_brand1(P_LEN);
        cudaMemcpy(h_p_partkey.data(), d_p_partkey, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_mfgr.data(), d_p_mfgr, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_category.data(), d_p_category, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_brand1.data(), d_p_brand1, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes += P_LEN * 4 * sizeof(uint32_t);

        // SUPPLIER table (4 columns)
        std::vector<uint32_t> h_s_suppkey(S_LEN), h_s_city(S_LEN), h_s_nation(S_LEN), h_s_region(S_LEN);
        cudaMemcpy(h_s_suppkey.data(), d_s_suppkey, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_city.data(), d_s_city, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_nation.data(), d_s_nation, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_region.data(), d_s_region, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes += S_LEN * 4 * sizeof(uint32_t);

        // CUSTOMER table (4 columns)
        std::vector<uint32_t> h_c_custkey(C_LEN), h_c_city(C_LEN), h_c_nation(C_LEN), h_c_region(C_LEN);
        cudaMemcpy(h_c_custkey.data(), d_c_custkey, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_city.data(), d_c_city, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_nation.data(), d_c_nation, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_region.data(), d_c_region, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes += C_LEN * 4 * sizeof(uint32_t);

        // DATE table (2 columns)
        std::vector<uint32_t> h_d_datekey(D_LEN), h_d_year(D_LEN);
        cudaMemcpy(h_d_datekey.data(), d_d_datekey, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d_year.data(), d_d_year, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes += D_LEN * 2 * sizeof(uint32_t);

        // 3. Allocate temporary GPU memory for H2D benchmark
        uint32_t* d_temp;
        cudaMalloc(&d_temp, lo_bytes + dim_bytes);

        // 4. Time the H2D transfer (compressed LO + dimension tables)
        cudaEventRecord(start, 0);

        // Transfer all lineorder compressed data
        size_t offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }

        // Transfer dimension tables
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_partkey.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_mfgr.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_category.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_brand1.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);

        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_suppkey.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_city.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_nation.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_region.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);

        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_custkey.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_city.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_nation.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_region.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);

        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_datekey.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_year.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time_h2d;
        cudaEventElapsedTime(&time_h2d, start, stop);

        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  Compressed LO H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Dimension tables H2D: " << (dim_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total H2D: " << ((lo_bytes + dim_bytes) / 1024.0 / 1024.0) << " MB in " << time_h2d << " ms" << std::endl;

        return time_h2d;
    }

    /**
     * @brief Measure H2D time for Q1x queries (Q11, Q12, Q13)
     * LO columns: orderdate, quantity, extendedprice, discount
     * DIM columns: none (Q1x only uses lineorder)
     */
    float measureH2DTimeQ1() {
        if (!is_loaded) return 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t lo_bytes = 0;
        std::vector<std::vector<uint32_t>> h_lo_deltas;

        auto copy_deltas_to_host = [&](const CompressedDataVertical<uint32_t>& col) {
            std::vector<uint32_t> h_data(col.interleaved_delta_words);
            cudaMemcpy(h_data.data(), col.d_interleaved_deltas,
                       col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            lo_bytes += col.interleaved_delta_words * sizeof(uint32_t);
            h_lo_deltas.push_back(std::move(h_data));
        };

        // Q1x uses: orderdate, quantity, extendedprice, discount
        copy_deltas_to_host(lo_orderdate);
        copy_deltas_to_host(lo_quantity);
        copy_deltas_to_host(lo_extendedprice);
        copy_deltas_to_host(lo_discount);

        // Q1x has no dimension table joins
        size_t dim_bytes = 0;

        uint32_t* d_temp;
        cudaMalloc(&d_temp, lo_bytes);

        // Warmup: do a full transfer first to stabilize PCIe link
        size_t offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        cudaDeviceSynchronize();

        // Now measure the actual H2D time
        cudaEventRecord(start, 0);
        offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time_h2d;
        cudaEventElapsedTime(&time_h2d, start, stop);

        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  Compressed LO H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB in " << time_h2d << " ms" << std::endl;

        return time_h2d;
    }

    /**
     * @brief Measure H2D time for Q2x queries (Q21, Q22, Q23)
     * LO columns: orderdate, partkey, suppkey, revenue
     * DIM columns: d_datekey, d_year, p_partkey, p_brand1, p_category, s_suppkey, s_region
     */
    float measureH2DTimeQ2() {
        if (!is_loaded) return 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t lo_bytes = 0;
        std::vector<std::vector<uint32_t>> h_lo_deltas;

        auto copy_deltas_to_host = [&](const CompressedDataVertical<uint32_t>& col) {
            std::vector<uint32_t> h_data(col.interleaved_delta_words);
            cudaMemcpy(h_data.data(), col.d_interleaved_deltas,
                       col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            lo_bytes += col.interleaved_delta_words * sizeof(uint32_t);
            h_lo_deltas.push_back(std::move(h_data));
        };

        // Q2x uses: orderdate, partkey, suppkey, revenue
        copy_deltas_to_host(lo_orderdate);
        copy_deltas_to_host(lo_partkey);
        copy_deltas_to_host(lo_suppkey);
        copy_deltas_to_host(lo_revenue);

        // Q2x dimension tables: DATE(2), PART(3), SUPPLIER(2)
        size_t dim_bytes = 0;
        std::vector<uint32_t> h_d_datekey(D_LEN), h_d_year(D_LEN);
        std::vector<uint32_t> h_p_partkey(P_LEN), h_p_brand1(P_LEN), h_p_category(P_LEN);
        std::vector<uint32_t> h_s_suppkey(S_LEN), h_s_region(S_LEN);

        cudaMemcpy(h_d_datekey.data(), d_d_datekey, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d_year.data(), d_d_year, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_partkey.data(), d_p_partkey, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_brand1.data(), d_p_brand1, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_category.data(), d_p_category, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_suppkey.data(), d_s_suppkey, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_region.data(), d_s_region, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes = D_LEN * 2 * sizeof(uint32_t) + P_LEN * 3 * sizeof(uint32_t) + S_LEN * 2 * sizeof(uint32_t);

        uint32_t* d_temp;
        cudaMalloc(&d_temp, lo_bytes + dim_bytes);

        // Warmup: do a full transfer first to stabilize PCIe link
        size_t offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        cudaDeviceSynchronize();

        // Now measure the actual H2D time
        cudaEventRecord(start, 0);
        offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        // Dimension tables
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_datekey.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_year.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_partkey.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_brand1.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_category.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_suppkey.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_region.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time_h2d;
        cudaEventElapsedTime(&time_h2d, start, stop);

        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  Compressed LO H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Dimension tables H2D: " << (dim_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total H2D: " << ((lo_bytes + dim_bytes) / 1024.0 / 1024.0) << " MB in " << time_h2d << " ms" << std::endl;

        return time_h2d;
    }

    /**
     * @brief Measure H2D time for Q3x queries (Q31, Q32, Q33, Q34)
     * LO columns: orderdate, custkey, suppkey, revenue
     * DIM columns: d_datekey, d_year, s_suppkey, s_city, s_nation, s_region, c_custkey, c_city, c_nation, c_region
     */
    float measureH2DTimeQ3() {
        if (!is_loaded) return 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t lo_bytes = 0;
        std::vector<std::vector<uint32_t>> h_lo_deltas;

        auto copy_deltas_to_host = [&](const CompressedDataVertical<uint32_t>& col) {
            std::vector<uint32_t> h_data(col.interleaved_delta_words);
            cudaMemcpy(h_data.data(), col.d_interleaved_deltas,
                       col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            lo_bytes += col.interleaved_delta_words * sizeof(uint32_t);
            h_lo_deltas.push_back(std::move(h_data));
        };

        // Q3x uses: orderdate, custkey, suppkey, revenue
        copy_deltas_to_host(lo_orderdate);
        copy_deltas_to_host(lo_custkey);
        copy_deltas_to_host(lo_suppkey);
        copy_deltas_to_host(lo_revenue);

        // Q3x dimension tables: DATE(2), SUPPLIER(4), CUSTOMER(4)
        size_t dim_bytes = 0;
        std::vector<uint32_t> h_d_datekey(D_LEN), h_d_year(D_LEN);
        std::vector<uint32_t> h_s_suppkey(S_LEN), h_s_city(S_LEN), h_s_nation(S_LEN), h_s_region(S_LEN);
        std::vector<uint32_t> h_c_custkey(C_LEN), h_c_city(C_LEN), h_c_nation(C_LEN), h_c_region(C_LEN);

        cudaMemcpy(h_d_datekey.data(), d_d_datekey, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d_year.data(), d_d_year, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_suppkey.data(), d_s_suppkey, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_city.data(), d_s_city, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_nation.data(), d_s_nation, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_region.data(), d_s_region, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_custkey.data(), d_c_custkey, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_city.data(), d_c_city, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_nation.data(), d_c_nation, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_region.data(), d_c_region, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes = D_LEN * 2 * sizeof(uint32_t) + S_LEN * 4 * sizeof(uint32_t) + C_LEN * 4 * sizeof(uint32_t);

        uint32_t* d_temp;
        cudaMalloc(&d_temp, lo_bytes + dim_bytes);

        // Warmup: do a full transfer first to stabilize PCIe link
        size_t offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        cudaDeviceSynchronize();

        // Now measure the actual H2D time
        cudaEventRecord(start, 0);
        offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        // Dimension tables
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_datekey.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_year.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_suppkey.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_city.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_nation.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_region.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_custkey.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_city.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_nation.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_region.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time_h2d;
        cudaEventElapsedTime(&time_h2d, start, stop);

        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  Compressed LO H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Dimension tables H2D: " << (dim_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total H2D: " << ((lo_bytes + dim_bytes) / 1024.0 / 1024.0) << " MB in " << time_h2d << " ms" << std::endl;

        return time_h2d;
    }

    /**
     * @brief Measure H2D time for Q4x queries (Q41, Q42, Q43)
     * LO columns: orderdate, custkey, suppkey, partkey, revenue, supplycost
     * DIM columns: d_datekey, d_year, p_partkey, p_mfgr, p_category, p_brand1, s_suppkey, s_region, s_nation, s_city, c_custkey, c_region, c_nation, c_city
     */
    float measureH2DTimeQ4() {
        if (!is_loaded) return 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        size_t lo_bytes = 0;
        std::vector<std::vector<uint32_t>> h_lo_deltas;

        auto copy_deltas_to_host = [&](const CompressedDataVertical<uint32_t>& col) {
            std::vector<uint32_t> h_data(col.interleaved_delta_words);
            cudaMemcpy(h_data.data(), col.d_interleaved_deltas,
                       col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            lo_bytes += col.interleaved_delta_words * sizeof(uint32_t);
            h_lo_deltas.push_back(std::move(h_data));
        };

        // Q4x uses: orderdate, custkey, suppkey, partkey, revenue, supplycost
        copy_deltas_to_host(lo_orderdate);
        copy_deltas_to_host(lo_custkey);
        copy_deltas_to_host(lo_suppkey);
        copy_deltas_to_host(lo_partkey);
        copy_deltas_to_host(lo_revenue);
        copy_deltas_to_host(lo_supplycost);

        // Q4x dimension tables: DATE(2), PART(4), SUPPLIER(4), CUSTOMER(4)
        size_t dim_bytes = 0;
        std::vector<uint32_t> h_d_datekey(D_LEN), h_d_year(D_LEN);
        std::vector<uint32_t> h_p_partkey(P_LEN), h_p_mfgr(P_LEN), h_p_category(P_LEN), h_p_brand1(P_LEN);
        std::vector<uint32_t> h_s_suppkey(S_LEN), h_s_region(S_LEN), h_s_nation(S_LEN), h_s_city(S_LEN);
        std::vector<uint32_t> h_c_custkey(C_LEN), h_c_region(C_LEN), h_c_nation(C_LEN), h_c_city(C_LEN);

        cudaMemcpy(h_d_datekey.data(), d_d_datekey, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d_year.data(), d_d_year, D_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_partkey.data(), d_p_partkey, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_mfgr.data(), d_p_mfgr, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_category.data(), d_p_category, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_p_brand1.data(), d_p_brand1, P_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_suppkey.data(), d_s_suppkey, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_region.data(), d_s_region, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_nation.data(), d_s_nation, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s_city.data(), d_s_city, S_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_custkey.data(), d_c_custkey, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_region.data(), d_c_region, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_nation.data(), d_c_nation, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_c_city.data(), d_c_city, C_LEN * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        dim_bytes = D_LEN * 2 * sizeof(uint32_t) + P_LEN * 4 * sizeof(uint32_t) + S_LEN * 4 * sizeof(uint32_t) + C_LEN * 4 * sizeof(uint32_t);

        uint32_t* d_temp;
        cudaMalloc(&d_temp, lo_bytes + dim_bytes);

        // Warmup: do a full transfer first to stabilize PCIe link
        size_t offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        cudaDeviceSynchronize();

        // Now measure the actual H2D time
        cudaEventRecord(start, 0);
        offset = 0;
        for (const auto& h_data : h_lo_deltas) {
            cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_data.data(),
                       h_data.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
            offset += h_data.size() * sizeof(uint32_t);
        }
        // Dimension tables
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_datekey.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_d_year.data(), D_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += D_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_partkey.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_mfgr.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_category.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_p_brand1.data(), P_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += P_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_suppkey.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_region.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_nation.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_s_city.data(), S_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += S_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_custkey.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_region.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_nation.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);
        offset += C_LEN * sizeof(uint32_t);
        cudaMemcpy(d_temp + offset / sizeof(uint32_t), h_c_city.data(), C_LEN * sizeof(uint32_t), cudaMemcpyHostToDevice);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float time_h2d;
        cudaEventElapsedTime(&time_h2d, start, stop);

        cudaFree(d_temp);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        std::cout << "  Compressed LO H2D: " << (lo_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Dimension tables H2D: " << (dim_bytes / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "  Total H2D: " << ((lo_bytes + dim_bytes) / 1024.0 / 1024.0) << " MB in " << time_h2d << " ms" << std::endl;

        return time_h2d;
    }

    /**
     * @brief Load and compress SSB data using V3 encoder (PolyCost)
     *
     * V3 encoder provides adaptive model selection with cost-optimal partitioning:
     * - MODEL_CONSTANT, LINEAR, POLYNOMIAL2, POLYNOMIAL3, FOR_BITPACK
     * - Better compression ratio for diverse data patterns
     */
    void loadAndCompressV3(const std::string& data_dir, int partition_size = 4096) {
        std::cout << "Loading and compressing SSB data with V3 (PolyCost) from " << data_dir << std::endl;

        VerticalConfig config = VerticalConfig::costOptimal();
        config.partition_size_hint = partition_size;
        config.enable_interleaved = true;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_min_partition_size = 256;
        config.cost_max_partition_size = partition_size * 2;

        std::cout << "  Partitioning strategy: COST_OPTIMAL (V3 PolyCost)" << std::endl;
        std::cout << "  Target partition size: " << partition_size << std::endl;

        auto compress_column = [&](const char* filename, const char* name) -> CompressedDataVertical<uint32_t> {
            std::cout << "    - " << name << "..." << std::flush;
            auto host_data = loadColumnFromFile<uint32_t>(data_dir + "/" + filename, LO_LEN);

            // Use V3 encoder (PolyCost) with adaptive model selection
            CompressedDataVertical<uint32_t> compressed =
                Vertical_encoder::encodeVerticalGPU_PolyCost<uint32_t>(host_data, partition_size, config, 0);

            double ratio = (double)(host_data.size() * sizeof(uint32_t)) /
                          (compressed.interleaved_delta_words * sizeof(uint32_t));
            std::cout << " done (ratio: " << ratio << "x, partitions: "
                      << compressed.num_partitions << ")" << std::endl;
            return compressed;
        };

        // Compress LINEORDER columns with V3 encoder
        std::cout << "  Compressing LINEORDER columns with V3 encoder..." << std::endl;
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
        std::cout << "SSB V3 compressed data loaded successfully." << std::endl;
    }

    /**
     * @brief Load and compress SSB data using V13 encoder
     *
     * V13 encoder uses cost-optimal partitioning with:
     * - min_partition_size = 1024 (aligned with decoder tile size)
     * - Full polynomial model support
     */
    void loadAndCompressV13(const std::string& data_dir, int partition_size = 1024) {
        std::cout << "Loading and compressing SSB data with V13 (min_partition=1024) from " << data_dir << std::endl;

        VerticalConfig config = VerticalConfig::costOptimal();
        config.partition_size_hint = partition_size;
        config.enable_interleaved = true;
        config.partitioning_strategy = PartitioningStrategy::COST_OPTIMAL;
        config.cost_min_partition_size = 1024;  // V13: min_partition_size = 1024
        config.cost_max_partition_size = partition_size * 2;

        std::cout << "  Partitioning strategy: COST_OPTIMAL (V13)" << std::endl;
        std::cout << "  Target partition size: " << partition_size << std::endl;
        std::cout << "  Min partition size: 1024" << std::endl;

        auto compress_column = [&](const char* filename, const char* name) -> CompressedDataVertical<uint32_t> {
            std::cout << "    - " << name << "..." << std::flush;
            auto host_data = loadColumnFromFile<uint32_t>(data_dir + "/" + filename, LO_LEN);

            // Use V3 encoder (PolyCost) with min_partition_size = 1024
            CompressedDataVertical<uint32_t> compressed =
                Vertical_encoder::encodeVerticalGPU_PolyCost<uint32_t>(host_data, partition_size, config, 0);

            double ratio = (double)(host_data.size() * sizeof(uint32_t)) /
                          (compressed.interleaved_delta_words * sizeof(uint32_t));
            std::cout << " done (ratio: " << ratio << "x, partitions: "
                      << compressed.num_partitions << ")" << std::endl;
            return compressed;
        };

        // Compress LINEORDER columns with V13 encoder
        std::cout << "  Compressing LINEORDER columns with V13 encoder..." << std::endl;
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
        std::cout << "SSB V13 compressed data loaded successfully." << std::endl;
    }

    /**
     * @brief Save compressed data to cache directory
     */
    void saveToCache(const std::string& cache_dir) {
        if (!is_loaded) return;

        // Create cache directory if it doesn't exist
        std::string mkdir_cmd = "mkdir -p " + cache_dir;
        system(mkdir_cmd.c_str());

        auto save_column = [&](const CompressedDataVertical<uint32_t>& col, const char* name) {
            std::string filepath = cache_dir + "/" + name + ".bin";
            std::ofstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to save: " << filepath << std::endl;
                return;
            }

            // Save metadata
            file.write(reinterpret_cast<const char*>(&col.total_values), sizeof(col.total_values));
            file.write(reinterpret_cast<const char*>(&col.num_partitions), sizeof(col.num_partitions));
            file.write(reinterpret_cast<const char*>(&col.interleaved_delta_words), sizeof(col.interleaved_delta_words));

            // Save delta array from GPU
            std::vector<uint32_t> h_deltas(col.interleaved_delta_words);
            cudaMemcpy(h_deltas.data(), col.d_interleaved_deltas, col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyDeviceToHost);
            file.write(reinterpret_cast<const char*>(h_deltas.data()), h_deltas.size() * sizeof(uint32_t));

            // Save partition metadata arrays
            std::vector<int32_t> h_start(col.num_partitions), h_end(col.num_partitions);
            std::vector<int32_t> h_delta_bits(col.num_partitions), h_model_types(col.num_partitions);
            std::vector<int32_t> h_num_mvs(col.num_partitions);
            std::vector<int64_t> h_offsets(col.num_partitions);
            std::vector<double> h_params(col.num_partitions * 4);

            cudaMemcpy(h_start.data(), col.d_start_indices, col.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_end.data(), col.d_end_indices, col.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_delta_bits.data(), col.d_delta_bits, col.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_offsets.data(), col.d_interleaved_offsets, col.num_partitions * sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_model_types.data(), col.d_model_types, col.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_params.data(), col.d_model_params, col.num_partitions * 4 * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_num_mvs.data(), col.d_num_mini_vectors, col.num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

            file.write(reinterpret_cast<const char*>(h_start.data()), h_start.size() * sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(h_end.data()), h_end.size() * sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(h_delta_bits.data()), h_delta_bits.size() * sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(h_offsets.data()), h_offsets.size() * sizeof(int64_t));
            file.write(reinterpret_cast<const char*>(h_model_types.data()), h_model_types.size() * sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(h_params.data()), h_params.size() * sizeof(double));
            file.write(reinterpret_cast<const char*>(h_num_mvs.data()), h_num_mvs.size() * sizeof(int32_t));

            file.close();
        };

        save_column(lo_orderdate, "lo_orderdate");
        save_column(lo_quantity, "lo_quantity");
        save_column(lo_extendedprice, "lo_extendedprice");
        save_column(lo_discount, "lo_discount");
        save_column(lo_revenue, "lo_revenue");
        save_column(lo_supplycost, "lo_supplycost");
        save_column(lo_custkey, "lo_custkey");
        save_column(lo_partkey, "lo_partkey");
        save_column(lo_suppkey, "lo_suppkey");

        std::cout << "Compressed data saved to: " << cache_dir << std::endl;
    }

    /**
     * @brief Load compressed data from cache directory
     * @return true if cache exists and loaded successfully
     */
    bool loadFromCache(const std::string& cache_dir, const std::string& data_dir) {
        std::string check_file = cache_dir + "/lo_orderdate.bin";
        std::ifstream check(check_file);
        if (!check.good()) {
            return false;  // Cache doesn't exist
        }
        check.close();

        std::cout << "Loading compressed data from cache: " << cache_dir << std::endl;

        auto load_column = [&](CompressedDataVertical<uint32_t>& col, const char* name) -> bool {
            std::string filepath = cache_dir + "/" + name + ".bin";
            std::ifstream file(filepath, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to load: " << filepath << std::endl;
                return false;
            }

            // Load metadata
            file.read(reinterpret_cast<char*>(&col.total_values), sizeof(col.total_values));
            file.read(reinterpret_cast<char*>(&col.num_partitions), sizeof(col.num_partitions));
            file.read(reinterpret_cast<char*>(&col.interleaved_delta_words), sizeof(col.interleaved_delta_words));

            // Load delta array to GPU
            std::vector<uint32_t> h_deltas(col.interleaved_delta_words);
            file.read(reinterpret_cast<char*>(h_deltas.data()), h_deltas.size() * sizeof(uint32_t));
            cudaMalloc(&col.d_interleaved_deltas, col.interleaved_delta_words * sizeof(uint32_t));
            cudaMemcpy(col.d_interleaved_deltas, h_deltas.data(), col.interleaved_delta_words * sizeof(uint32_t), cudaMemcpyHostToDevice);

            // Load partition metadata arrays
            std::vector<int32_t> h_start(col.num_partitions), h_end(col.num_partitions);
            std::vector<int32_t> h_delta_bits(col.num_partitions), h_model_types(col.num_partitions);
            std::vector<int32_t> h_num_mvs(col.num_partitions);
            std::vector<int64_t> h_offsets(col.num_partitions);
            std::vector<double> h_params(col.num_partitions * 4);

            file.read(reinterpret_cast<char*>(h_start.data()), h_start.size() * sizeof(int32_t));
            file.read(reinterpret_cast<char*>(h_end.data()), h_end.size() * sizeof(int32_t));
            file.read(reinterpret_cast<char*>(h_delta_bits.data()), h_delta_bits.size() * sizeof(int32_t));
            file.read(reinterpret_cast<char*>(h_offsets.data()), h_offsets.size() * sizeof(int64_t));
            file.read(reinterpret_cast<char*>(h_model_types.data()), h_model_types.size() * sizeof(int32_t));
            file.read(reinterpret_cast<char*>(h_params.data()), h_params.size() * sizeof(double));
            file.read(reinterpret_cast<char*>(h_num_mvs.data()), h_num_mvs.size() * sizeof(int32_t));

            cudaMalloc(&col.d_start_indices, col.num_partitions * sizeof(int32_t));
            cudaMalloc(&col.d_end_indices, col.num_partitions * sizeof(int32_t));
            cudaMalloc(&col.d_delta_bits, col.num_partitions * sizeof(int32_t));
            cudaMalloc(&col.d_interleaved_offsets, col.num_partitions * sizeof(int64_t));
            cudaMalloc(&col.d_model_types, col.num_partitions * sizeof(int32_t));
            cudaMalloc(&col.d_model_params, col.num_partitions * 4 * sizeof(double));
            cudaMalloc(&col.d_num_mini_vectors, col.num_partitions * sizeof(int32_t));

            cudaMemcpy(col.d_start_indices, h_start.data(), col.num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_end_indices, h_end.data(), col.num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_delta_bits, h_delta_bits.data(), col.num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_interleaved_offsets, h_offsets.data(), col.num_partitions * sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_model_types, h_model_types.data(), col.num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_model_params, h_params.data(), col.num_partitions * 4 * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(col.d_num_mini_vectors, h_num_mvs.data(), col.num_partitions * sizeof(int32_t), cudaMemcpyHostToDevice);

            file.close();
            std::cout << "    - " << name << "... loaded" << std::endl;
            return true;
        };

        if (!load_column(lo_orderdate, "lo_orderdate")) return false;
        if (!load_column(lo_quantity, "lo_quantity")) return false;
        if (!load_column(lo_extendedprice, "lo_extendedprice")) return false;
        if (!load_column(lo_discount, "lo_discount")) return false;
        if (!load_column(lo_revenue, "lo_revenue")) return false;
        if (!load_column(lo_supplycost, "lo_supplycost")) return false;
        if (!load_column(lo_custkey, "lo_custkey")) return false;
        if (!load_column(lo_partkey, "lo_partkey")) return false;
        if (!load_column(lo_suppkey, "lo_suppkey")) return false;

        // Load dimension tables (uncompressed)
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
        std::cout << "Compressed data loaded from cache successfully." << std::endl;
        return true;
    }

    /**
     * @brief Smart load: try cache first, compress if not found
     */
    void loadOrCompress(const std::string& data_dir, const std::string& cache_dir, int partition_size = 4096) {
        if (!loadFromCache(cache_dir, data_dir)) {
            loadAndCompress(data_dir, partition_size);
            saveToCache(cache_dir);
        }
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
