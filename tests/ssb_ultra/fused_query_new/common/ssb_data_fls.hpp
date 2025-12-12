/**
 * @file ssb_data_fls.hpp
 * @brief SSB Data Loader for Vertical-style FOR+BitPack encoding
 *
 * This file provides data loading and encoding that matches Vertical-GPU:
 * - Fixed 1024-element tiles (32 threads x 32 items)
 * - FOR (Frame of Reference) encoding: delta = value - min
 * - BitPack with single bitwidth per column
 * - Memory layout: tile N at offset N * bitwidth * 32
 *
 * Usage:
 *   SSBDataFLS data;
 *   data.loadAndEncode("/path/to/ssb_data");
 *   // Use data.lo_orderdate, etc. in kernels
 *   data.free();
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <iostream>

#include "fls_constants.cuh"

namespace l3_fls {

// ============================================================================
// SSB Table Sizes (SF=20)
// ============================================================================

constexpr size_t SSB_LO_LEN = 119968352;  // LINEORDER rows
constexpr size_t SSB_P_LEN = 1000000;     // PART rows
constexpr size_t SSB_S_LEN = 40000;       // SUPPLIER rows
constexpr size_t SSB_C_LEN = 600000;      // CUSTOMER rows
constexpr size_t SSB_D_LEN = 2557;        // DATE rows

// ============================================================================
// Column File Names
// ============================================================================

// LINEORDER columns
constexpr const char* LO_ORDERKEY_FILE      = "LINEORDER0.bin";
constexpr const char* LO_LINENUMBER_FILE    = "LINEORDER1.bin";
constexpr const char* LO_CUSTKEY_FILE       = "LINEORDER2.bin";
constexpr const char* LO_PARTKEY_FILE       = "LINEORDER3.bin";
constexpr const char* LO_SUPPKEY_FILE       = "LINEORDER4.bin";
constexpr const char* LO_ORDERDATE_FILE     = "LINEORDER5.bin";
constexpr const char* LO_ORDERPRIORITY_FILE = "LINEORDER6.bin";
constexpr const char* LO_SHIPPRIORITY_FILE  = "LINEORDER7.bin";
constexpr const char* LO_QUANTITY_FILE      = "LINEORDER8.bin";
constexpr const char* LO_EXTENDEDPRICE_FILE = "LINEORDER9.bin";
constexpr const char* LO_ORDTOTALPRICE_FILE = "LINEORDER10.bin";
constexpr const char* LO_DISCOUNT_FILE      = "LINEORDER11.bin";
constexpr const char* LO_REVENUE_FILE       = "LINEORDER12.bin";
constexpr const char* LO_SUPPLYCOST_FILE    = "LINEORDER13.bin";
constexpr const char* LO_TAX_FILE           = "LINEORDER14.bin";
constexpr const char* LO_COMMITDATE_FILE    = "LINEORDER15.bin";
constexpr const char* LO_SHIPMODE_FILE      = "LINEORDER16.bin";

// PART columns
constexpr const char* P_PARTKEY_FILE   = "PART0.bin";
constexpr const char* P_MFGR_FILE      = "PART2.bin";
constexpr const char* P_CATEGORY_FILE  = "PART3.bin";
constexpr const char* P_BRAND1_FILE    = "PART4.bin";

// SUPPLIER columns
constexpr const char* S_SUPPKEY_FILE = "SUPPLIER0.bin";
constexpr const char* S_CITY_FILE    = "SUPPLIER3.bin";
constexpr const char* S_NATION_FILE  = "SUPPLIER4.bin";
constexpr const char* S_REGION_FILE  = "SUPPLIER5.bin";

// CUSTOMER columns
constexpr const char* C_CUSTKEY_FILE = "CUSTOMER0.bin";
constexpr const char* C_CITY_FILE    = "CUSTOMER3.bin";
constexpr const char* C_NATION_FILE  = "CUSTOMER4.bin";
constexpr const char* C_REGION_FILE  = "CUSTOMER5.bin";

// DATE columns
constexpr const char* D_DATEKEY_FILE      = "DDATE0.bin";
constexpr const char* D_YEAR_FILE         = "DDATE4.bin";
constexpr const char* D_YEARMONTHNUM_FILE = "DDATE5.bin";

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Calculate bit width needed for a value range
 */
inline int calcBitWidth(uint32_t max_delta) {
    if (max_delta == 0) return 0;
    int bits = 0;
    while (max_delta > 0) {
        max_delta >>= 1;
        bits++;
    }
    // Round up to common bit widths for efficient unpacking
    if (bits <= 4) return 4;
    if (bits <= 8) return 8;
    if (bits <= 12) return 12;
    if (bits <= 16) return 16;
    if (bits <= 20) return 20;
    if (bits <= 24) return 24;
    return 32;
}

/**
 * @brief Load binary column from file
 */
inline std::vector<int32_t> loadColumn(const std::string& filepath, size_t expected_rows = 0) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(int32_t);

    if (expected_rows > 0 && num_elements != expected_rows) {
        std::cerr << "Warning: " << filepath << " has " << num_elements
                  << " elements, expected " << expected_rows << std::endl;
    }

    file.seekg(0, std::ios::beg);
    std::vector<int32_t> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();

    return data;
}

// ============================================================================
// FOR+BitPack Column Encoding
// ============================================================================

/**
 * @brief Encoded column data for GPU (Vertical-style)
 *
 * Memory layout matches Vertical-GPU:
 * - Tile N at offset: N * bitwidth * 32 (in uint32_t words)
 * - Each tile contains 1024 elements
 * - 32 threads, each unpacks 32 values
 */
struct EncodedColumnFLS {
    // Device pointers
    uint32_t* d_encoded;      // Bit-packed delta array on GPU
    int64_t   encoded_words;  // Total words in encoded array

    // Encoding metadata
    int32_t   min_value;      // FOR base (minimum value)
    int32_t   max_value;      // Maximum value (for reference)
    uint8_t   bitwidth;       // Bits per delta
    int64_t   num_values;     // Total values
    int64_t   num_tiles;      // Number of 1024-element tiles

    // Host copy of encoded data (for debugging)
    std::vector<uint32_t> h_encoded;

    EncodedColumnFLS()
        : d_encoded(nullptr), encoded_words(0),
          min_value(0), max_value(0), bitwidth(0),
          num_values(0), num_tiles(0) {}

    /**
     * @brief Encode a column using FOR+BitPack
     *
     * @param data Raw column data
     * @param name Column name (for logging)
     */
    void encode(const std::vector<int32_t>& data, const char* name = "column") {
        num_values = data.size();
        num_tiles = (num_values + FLS_TILE_SIZE - 1) / FLS_TILE_SIZE;

        // Find min/max
        min_value = *std::min_element(data.begin(), data.end());
        max_value = *std::max_element(data.begin(), data.end());

        // Calculate bit width
        uint32_t max_delta = static_cast<uint32_t>(max_value - min_value);
        bitwidth = static_cast<uint8_t>(calcBitWidth(max_delta));

        // Calculate encoded size
        // Each tile: 1024 values * bitwidth bits = bitwidth * 32 words
        encoded_words = num_tiles * bitwidth * 32;

        // Allocate and encode
        h_encoded.resize(encoded_words, 0);

        // Encode each tile
        for (int64_t tile = 0; tile < num_tiles; tile++) {
            int64_t tile_start = tile * FLS_TILE_SIZE;
            int64_t tile_offset = tile * bitwidth * 32;  // Word offset

            // Pack values into tile
            for (int t = 0; t < 32; t++) {  // 32 threads
                for (int v = 0; v < 32; v++) {  // 32 values per thread
                    int64_t global_idx = tile_start + v * 32 + t;

                    // Get value (or 0 if past end)
                    int32_t value = 0;
                    if (global_idx < num_values) {
                        value = data[global_idx];
                    }

                    // Compute delta
                    uint32_t delta = static_cast<uint32_t>(value - min_value);

                    // Pack delta into encoded array
                    // Bit position: thread t, value v => bit = v * bitwidth within thread's words
                    // Word position: tile_offset + t + (v * bitwidth / 32) * 32
                    // But Vertical uses: each register for thread t contains bits for values at strided positions

                    // Vertical packing: thread t's bits are at words [tile_offset + t, tile_offset + t + 32, ...]
                    // For value v of thread t, bits start at bit position (v * bitwidth) within thread's bit stream

                    int64_t bit_idx = static_cast<int64_t>(v) * bitwidth;
                    int64_t word_in_thread = bit_idx / 32;
                    int bit_in_word = bit_idx % 32;

                    // Thread t's word w is at position: tile_offset + w * 32 + t
                    int64_t word_idx = tile_offset + word_in_thread * 32 + t;

                    if (word_idx < encoded_words) {
                        // Insert delta bits
                        if (bit_in_word + bitwidth <= 32) {
                            // Fits in one word
                            h_encoded[word_idx] |= (delta << bit_in_word);
                        } else {
                            // Spans two words
                            int bits_in_first = 32 - bit_in_word;
                            h_encoded[word_idx] |= (delta << bit_in_word);

                            int64_t next_word_idx = tile_offset + (word_in_thread + 1) * 32 + t;
                            if (next_word_idx < encoded_words) {
                                h_encoded[next_word_idx] |= (delta >> bits_in_first);
                            }
                        }
                    }
                }
            }
        }

        // Copy to GPU
        cudaMalloc(&d_encoded, encoded_words * sizeof(uint32_t));
        cudaMemcpy(d_encoded, h_encoded.data(), encoded_words * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);

        std::cout << "  " << name << ": min=" << min_value << ", max=" << max_value
                  << ", bw=" << (int)bitwidth << ", tiles=" << num_tiles
                  << ", size=" << (encoded_words * 4 / 1024.0 / 1024.0) << " MB" << std::endl;
    }

    /**
     * @brief Free GPU memory
     */
    void free() {
        if (d_encoded) {
            cudaFree(d_encoded);
            d_encoded = nullptr;
        }
        h_encoded.clear();
    }
};

// ============================================================================
// SSB Data Structure (Vertical-style)
// ============================================================================

/**
 * @brief SSB dataset encoded in Vertical-style FOR+BitPack format
 */
struct SSBDataFLS {
    // LINEORDER columns (encoded)
    EncodedColumnFLS lo_orderdate;
    EncodedColumnFLS lo_quantity;
    EncodedColumnFLS lo_discount;
    EncodedColumnFLS lo_extendedprice;
    EncodedColumnFLS lo_revenue;
    EncodedColumnFLS lo_supplycost;
    EncodedColumnFLS lo_custkey;
    EncodedColumnFLS lo_partkey;
    EncodedColumnFLS lo_suppkey;

    // Dimension tables (uncompressed - small, used for hash joins)
    int32_t* d_p_partkey;
    int32_t* d_p_mfgr;
    int32_t* d_p_category;
    int32_t* d_p_brand1;

    int32_t* d_s_suppkey;
    int32_t* d_s_city;
    int32_t* d_s_nation;
    int32_t* d_s_region;

    int32_t* d_c_custkey;
    int32_t* d_c_city;
    int32_t* d_c_nation;
    int32_t* d_c_region;

    int32_t* d_d_datekey;
    int32_t* d_d_year;
    int32_t* d_d_yearmonthnum;

    // Metadata
    int64_t n_tup_lineorder;
    int64_t n_tiles;
    bool is_loaded;

    SSBDataFLS()
        : d_p_partkey(nullptr), d_p_mfgr(nullptr), d_p_category(nullptr), d_p_brand1(nullptr),
          d_s_suppkey(nullptr), d_s_city(nullptr), d_s_nation(nullptr), d_s_region(nullptr),
          d_c_custkey(nullptr), d_c_city(nullptr), d_c_nation(nullptr), d_c_region(nullptr),
          d_d_datekey(nullptr), d_d_year(nullptr), d_d_yearmonthnum(nullptr),
          n_tup_lineorder(0), n_tiles(0), is_loaded(false) {}

    /**
     * @brief Load and encode SSB data
     */
    void loadAndEncode(const std::string& data_dir) {
        std::cout << "Loading and encoding SSB data from " << data_dir << std::endl;
        std::cout << "Encoding format: FOR+BitPack, tile size=" << FLS_TILE_SIZE << std::endl;

        // Load and encode LINEORDER columns
        std::cout << "\nEncoding LINEORDER columns:" << std::endl;

        auto load_and_encode = [&](const char* filename, EncodedColumnFLS& col, const char* name) {
            auto data = loadColumn(data_dir + "/" + filename, SSB_LO_LEN);
            col.encode(data, name);
        };

        load_and_encode(LO_ORDERDATE_FILE, lo_orderdate, "lo_orderdate");
        load_and_encode(LO_QUANTITY_FILE, lo_quantity, "lo_quantity");
        load_and_encode(LO_DISCOUNT_FILE, lo_discount, "lo_discount");
        load_and_encode(LO_EXTENDEDPRICE_FILE, lo_extendedprice, "lo_extendedprice");
        load_and_encode(LO_REVENUE_FILE, lo_revenue, "lo_revenue");
        load_and_encode(LO_SUPPLYCOST_FILE, lo_supplycost, "lo_supplycost");
        load_and_encode(LO_CUSTKEY_FILE, lo_custkey, "lo_custkey");
        load_and_encode(LO_PARTKEY_FILE, lo_partkey, "lo_partkey");
        load_and_encode(LO_SUPPKEY_FILE, lo_suppkey, "lo_suppkey");

        n_tup_lineorder = lo_orderdate.num_values;
        n_tiles = lo_orderdate.num_tiles;

        // Load dimension tables (uncompressed)
        std::cout << "\nLoading dimension tables (uncompressed):" << std::endl;

        auto load_dim = [&](const char* filename, int32_t*& d_ptr, size_t rows, const char* name) {
            auto data = loadColumn(data_dir + "/" + filename, rows);
            cudaMalloc(&d_ptr, data.size() * sizeof(int32_t));
            cudaMemcpy(d_ptr, data.data(), data.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            std::cout << "  " << name << ": " << data.size() << " rows" << std::endl;
        };

        load_dim(P_PARTKEY_FILE, d_p_partkey, SSB_P_LEN, "p_partkey");
        load_dim(P_MFGR_FILE, d_p_mfgr, SSB_P_LEN, "p_mfgr");
        load_dim(P_CATEGORY_FILE, d_p_category, SSB_P_LEN, "p_category");
        load_dim(P_BRAND1_FILE, d_p_brand1, SSB_P_LEN, "p_brand1");

        load_dim(S_SUPPKEY_FILE, d_s_suppkey, SSB_S_LEN, "s_suppkey");
        load_dim(S_CITY_FILE, d_s_city, SSB_S_LEN, "s_city");
        load_dim(S_NATION_FILE, d_s_nation, SSB_S_LEN, "s_nation");
        load_dim(S_REGION_FILE, d_s_region, SSB_S_LEN, "s_region");

        load_dim(C_CUSTKEY_FILE, d_c_custkey, SSB_C_LEN, "c_custkey");
        load_dim(C_CITY_FILE, d_c_city, SSB_C_LEN, "c_city");
        load_dim(C_NATION_FILE, d_c_nation, SSB_C_LEN, "c_nation");
        load_dim(C_REGION_FILE, d_c_region, SSB_C_LEN, "c_region");

        load_dim(D_DATEKEY_FILE, d_d_datekey, SSB_D_LEN, "d_datekey");
        load_dim(D_YEAR_FILE, d_d_year, SSB_D_LEN, "d_year");
        load_dim(D_YEARMONTHNUM_FILE, d_d_yearmonthnum, SSB_D_LEN, "d_yearmonthnum");

        is_loaded = true;

        std::cout << "\nSSB data loaded: " << n_tup_lineorder << " lineorder rows, "
                  << n_tiles << " tiles" << std::endl;
    }

    /**
     * @brief Free all GPU memory
     */
    void free() {
        if (!is_loaded) return;

        // Free encoded columns
        lo_orderdate.free();
        lo_quantity.free();
        lo_discount.free();
        lo_extendedprice.free();
        lo_revenue.free();
        lo_supplycost.free();
        lo_custkey.free();
        lo_partkey.free();
        lo_suppkey.free();

        // Free dimension tables
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
        cudaFree(d_d_yearmonthnum);

        is_loaded = false;
    }

    /**
     * @brief Get total compressed size in bytes
     */
    size_t getCompressedSize() const {
        if (!is_loaded) return 0;

        size_t total = 0;
        total += lo_orderdate.encoded_words * sizeof(uint32_t);
        total += lo_quantity.encoded_words * sizeof(uint32_t);
        total += lo_discount.encoded_words * sizeof(uint32_t);
        total += lo_extendedprice.encoded_words * sizeof(uint32_t);
        total += lo_revenue.encoded_words * sizeof(uint32_t);
        total += lo_supplycost.encoded_words * sizeof(uint32_t);
        total += lo_custkey.encoded_words * sizeof(uint32_t);
        total += lo_partkey.encoded_words * sizeof(uint32_t);
        total += lo_suppkey.encoded_words * sizeof(uint32_t);

        return total;
    }

    /**
     * @brief Get compression ratio
     */
    double getCompressionRatio() const {
        size_t original = n_tup_lineorder * 9 * sizeof(int32_t);  // 9 columns
        size_t compressed = getCompressedSize();
        return compressed > 0 ? (double)original / compressed : 1.0;
    }

    /**
     * @brief Print encoding statistics
     */
    void printStats() const {
        std::cout << "\n=== SSB FLS Encoding Statistics ===" << std::endl;
        std::cout << "Total rows: " << n_tup_lineorder << std::endl;
        std::cout << "Total tiles: " << n_tiles << std::endl;
        std::cout << "Tile size: " << FLS_TILE_SIZE << " elements" << std::endl;
        std::cout << "\nColumn bitwidths:" << std::endl;
        std::cout << "  lo_orderdate:     " << (int)lo_orderdate.bitwidth << " bits" << std::endl;
        std::cout << "  lo_quantity:      " << (int)lo_quantity.bitwidth << " bits" << std::endl;
        std::cout << "  lo_discount:      " << (int)lo_discount.bitwidth << " bits" << std::endl;
        std::cout << "  lo_extendedprice: " << (int)lo_extendedprice.bitwidth << " bits" << std::endl;
        std::cout << "  lo_revenue:       " << (int)lo_revenue.bitwidth << " bits" << std::endl;
        std::cout << "  lo_supplycost:    " << (int)lo_supplycost.bitwidth << " bits" << std::endl;
        std::cout << "  lo_custkey:       " << (int)lo_custkey.bitwidth << " bits" << std::endl;
        std::cout << "  lo_partkey:       " << (int)lo_partkey.bitwidth << " bits" << std::endl;
        std::cout << "  lo_suppkey:       " << (int)lo_suppkey.bitwidth << " bits" << std::endl;
        std::cout << "\nCompressed size: " << (getCompressedSize() / 1024.0 / 1024.0) << " MB" << std::endl;
        std::cout << "Compression ratio: " << getCompressionRatio() << "x" << std::endl;
    }
};

// ============================================================================
// Query Timing Utilities
// ============================================================================

struct FLSQueryTiming {
    float encode_ms;
    float kernel_ms;
    float total_ms;

    void print(const char* query_name) const {
        std::cout << query_name << " timing:" << std::endl;
        std::cout << "  Kernel:  " << kernel_ms << " ms" << std::endl;
        std::cout << "  Total:   " << total_ms << " ms" << std::endl;
    }
};

}  // namespace l3_fls
