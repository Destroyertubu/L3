/**
 * @file ssb_data_new.hpp
 * @brief SSB Data Loader for Crystal-opt Style Queries
 *
 * Simplified data loading for fused_query_new implementation.
 * Reuses existing SSB data loader from common/ssb_data_loader.hpp
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

namespace ssb_new {

// ============================================================================
// Table Size Constants (SF=20)
// ============================================================================

constexpr size_t LO_LEN = 119968352;  // LINEORDER rows
constexpr size_t P_LEN = 1000000;     // PART rows
constexpr size_t S_LEN = 40000;       // SUPPLIER rows
constexpr size_t C_LEN = 600000;      // CUSTOMER rows
constexpr size_t D_LEN = 2557;        // DATE rows

// Min key values for PHT hash (key - min_key) % size
constexpr int D_MIN_KEY = 19920101;   // Date min key
constexpr int P_MIN_KEY = 1;          // Part min key
constexpr int S_MIN_KEY = 1;          // Supplier min key
constexpr int C_MIN_KEY = 1;          // Customer min key

// ============================================================================
// Column File Names
// ============================================================================

// LINEORDER columns
constexpr const char* LO_ORDERDATE     = "LINEORDER5.bin";
constexpr const char* LO_QUANTITY      = "LINEORDER8.bin";
constexpr const char* LO_EXTENDEDPRICE = "LINEORDER9.bin";
constexpr const char* LO_DISCOUNT      = "LINEORDER11.bin";
constexpr const char* LO_REVENUE       = "LINEORDER12.bin";
constexpr const char* LO_SUPPLYCOST    = "LINEORDER13.bin";
constexpr const char* LO_CUSTKEY       = "LINEORDER2.bin";
constexpr const char* LO_PARTKEY       = "LINEORDER3.bin";
constexpr const char* LO_SUPPKEY       = "LINEORDER4.bin";

// PART columns
constexpr const char* P_PARTKEY  = "PART0.bin";
constexpr const char* P_MFGR     = "PART2.bin";
constexpr const char* P_CATEGORY = "PART3.bin";
constexpr const char* P_BRAND1   = "PART4.bin";

// SUPPLIER columns
constexpr const char* S_SUPPKEY = "SUPPLIER0.bin";
constexpr const char* S_CITY    = "SUPPLIER3.bin";
constexpr const char* S_NATION  = "SUPPLIER4.bin";
constexpr const char* S_REGION  = "SUPPLIER5.bin";

// CUSTOMER columns
constexpr const char* C_CUSTKEY = "CUSTOMER0.bin";
constexpr const char* C_CITY    = "CUSTOMER3.bin";
constexpr const char* C_NATION  = "CUSTOMER4.bin";
constexpr const char* C_REGION  = "CUSTOMER5.bin";

// DATE columns
constexpr const char* D_DATEKEY = "DDATE0.bin";
constexpr const char* D_YEAR    = "DDATE4.bin";

// ============================================================================
// Data Loading Utilities
// ============================================================================

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

template<typename T>
T* loadColumnToGPU(const std::string& filepath, size_t expected_rows = 0) {
    auto host_data = loadColumnFromFile<T>(filepath, expected_rows);

    T* d_data;
    cudaMalloc(&d_data, host_data.size() * sizeof(T));
    cudaMemcpy(d_data, host_data.data(), host_data.size() * sizeof(T), cudaMemcpyHostToDevice);

    return d_data;
}

// ============================================================================
// L3 Compressed Column Structure
// ============================================================================

struct L3CompressedColumn {
    // GPU arrays
    uint32_t* d_delta_array;
    int32_t* d_model_types;
    double* d_model_params;
    int32_t* d_delta_bits;
    int64_t* d_bit_offsets;
    int32_t* d_start_indices;
    int32_t* d_end_indices;

    int num_partitions;
    size_t total_elements;

    void free() {
        if (d_delta_array) cudaFree(d_delta_array);
        if (d_model_types) cudaFree(d_model_types);
        if (d_model_params) cudaFree(d_model_params);
        if (d_delta_bits) cudaFree(d_delta_bits);
        if (d_bit_offsets) cudaFree(d_bit_offsets);
        if (d_start_indices) cudaFree(d_start_indices);
        if (d_end_indices) cudaFree(d_end_indices);
    }
};

// Convert from CompressedDataL3 to L3CompressedColumn
inline L3CompressedColumn wrapCompressedData(CompressedDataL3<uint32_t>* L3) {
    L3CompressedColumn col;
    col.d_delta_array = L3->delta_array;
    col.d_model_types = L3->d_model_types;
    col.d_model_params = L3->d_model_params;
    col.d_delta_bits = L3->d_delta_bits;
    col.d_bit_offsets = L3->d_delta_array_bit_offsets;
    col.d_start_indices = L3->d_start_indices;
    col.d_end_indices = L3->d_end_indices;
    col.num_partitions = L3->num_partitions;
    col.total_elements = L3->total_values;
    return col;
}

// ============================================================================
// SSB Data Structure
// ============================================================================

struct SSBData {
    // L3 Compressed LINEORDER columns
    L3CompressedColumn lo_orderdate;
    L3CompressedColumn lo_quantity;
    L3CompressedColumn lo_extendedprice;
    L3CompressedColumn lo_discount;
    L3CompressedColumn lo_revenue;
    L3CompressedColumn lo_supplycost;
    L3CompressedColumn lo_custkey;
    L3CompressedColumn lo_partkey;
    L3CompressedColumn lo_suppkey;

    // Uncompressed L3 data (for proper cleanup)
    CompressedDataL3<uint32_t>* L3_orderdate = nullptr;
    CompressedDataL3<uint32_t>* L3_quantity = nullptr;
    CompressedDataL3<uint32_t>* L3_extendedprice = nullptr;
    CompressedDataL3<uint32_t>* L3_discount = nullptr;
    CompressedDataL3<uint32_t>* L3_revenue = nullptr;
    CompressedDataL3<uint32_t>* L3_supplycost = nullptr;
    CompressedDataL3<uint32_t>* L3_custkey = nullptr;
    CompressedDataL3<uint32_t>* L3_partkey = nullptr;
    CompressedDataL3<uint32_t>* L3_suppkey = nullptr;

    // Dimension tables (uncompressed, for hash table building)
    int* d_p_partkey;
    int* d_p_mfgr;
    int* d_p_category;
    int* d_p_brand1;

    int* d_s_suppkey;
    int* d_s_city;
    int* d_s_nation;
    int* d_s_region;

    int* d_c_custkey;
    int* d_c_city;
    int* d_c_nation;
    int* d_c_region;

    int* d_d_datekey;
    int* d_d_year;

    bool is_loaded = false;

    void load(const std::string& data_dir, int partition_size = 2048) {
        std::cout << "Loading SSB data with L3 compression (partition_size=" << partition_size << ")..." << std::endl;

        L3Config config = L3Config::fixedPartitioning(partition_size);

        auto compress_column = [&](const char* filename, const char* name) -> CompressedDataL3<uint32_t>* {
            std::cout << "  Compressing " << name << "..." << std::flush;
            auto host_data = loadColumnFromFile<uint32_t>(data_dir + "/" + filename, LO_LEN);

            CompressionStats stats;
            CompressedDataL3<uint32_t>* compressed = compressDataWithConfig(host_data, config, &stats);

            std::cout << " done (ratio: " << stats.compression_ratio << "x, partitions: "
                      << compressed->num_partitions << ")" << std::endl;
            return compressed;
        };

        // Compress LINEORDER columns
        L3_orderdate     = compress_column(LO_ORDERDATE, "lo_orderdate");
        L3_quantity      = compress_column(LO_QUANTITY, "lo_quantity");
        L3_extendedprice = compress_column(LO_EXTENDEDPRICE, "lo_extendedprice");
        L3_discount      = compress_column(LO_DISCOUNT, "lo_discount");
        L3_revenue       = compress_column(LO_REVENUE, "lo_revenue");
        L3_supplycost    = compress_column(LO_SUPPLYCOST, "lo_supplycost");
        L3_custkey       = compress_column(LO_CUSTKEY, "lo_custkey");
        L3_partkey       = compress_column(LO_PARTKEY, "lo_partkey");
        L3_suppkey       = compress_column(LO_SUPPKEY, "lo_suppkey");

        // Wrap for easy access
        lo_orderdate     = wrapCompressedData(L3_orderdate);
        lo_quantity      = wrapCompressedData(L3_quantity);
        lo_extendedprice = wrapCompressedData(L3_extendedprice);
        lo_discount      = wrapCompressedData(L3_discount);
        lo_revenue       = wrapCompressedData(L3_revenue);
        lo_supplycost    = wrapCompressedData(L3_supplycost);
        lo_custkey       = wrapCompressedData(L3_custkey);
        lo_partkey       = wrapCompressedData(L3_partkey);
        lo_suppkey       = wrapCompressedData(L3_suppkey);

        // Load dimension tables (cast to int* for PHT compatibility)
        std::cout << "  Loading dimension tables..." << std::endl;
        d_p_partkey  = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + P_PARTKEY, P_LEN));
        d_p_mfgr     = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + P_MFGR, P_LEN));
        d_p_category = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + P_CATEGORY, P_LEN));
        d_p_brand1   = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + P_BRAND1, P_LEN));

        d_s_suppkey = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + S_SUPPKEY, S_LEN));
        d_s_city    = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + S_CITY, S_LEN));
        d_s_nation  = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + S_NATION, S_LEN));
        d_s_region  = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + S_REGION, S_LEN));

        d_c_custkey = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + C_CUSTKEY, C_LEN));
        d_c_city    = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + C_CITY, C_LEN));
        d_c_nation  = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + C_NATION, C_LEN));
        d_c_region  = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + C_REGION, C_LEN));

        d_d_datekey = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + D_DATEKEY, D_LEN));
        d_d_year    = reinterpret_cast<int*>(loadColumnToGPU<uint32_t>(data_dir + "/" + D_YEAR, D_LEN));

        is_loaded = true;
        std::cout << "SSB data loaded successfully. Partitions: " << lo_orderdate.num_partitions << std::endl;
    }

    void free() {
        if (!is_loaded) return;

        // Free compressed columns via L3 cleanup
        freeCompressedData(L3_orderdate);
        freeCompressedData(L3_quantity);
        freeCompressedData(L3_extendedprice);
        freeCompressedData(L3_discount);
        freeCompressedData(L3_revenue);
        freeCompressedData(L3_supplycost);
        freeCompressedData(L3_custkey);
        freeCompressedData(L3_partkey);
        freeCompressedData(L3_suppkey);

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

        is_loaded = false;
    }

    int numPartitions() const {
        return lo_orderdate.num_partitions;
    }
};

// ============================================================================
// PHT Helper Structures
// ============================================================================

struct PHT {
    int* data;    // PHT_1: keys only, PHT_2: [key0, val0, key1, val1, ...]
    int size;
    int min_key;
    bool has_values;

    void allocate(int table_size, bool with_values, int key_min = 0) {
        size = table_size;
        min_key = key_min;
        has_values = with_values;
        int alloc_size = with_values ? (table_size * 2) : table_size;
        cudaMalloc(&data, alloc_size * sizeof(int));
        cudaMemset(data, 0, alloc_size * sizeof(int));
    }

    void free() {
        if (data) cudaFree(data);
        data = nullptr;
    }
};

// ============================================================================
// Timing Utilities
// ============================================================================

struct QueryTiming {
    float hash_build_ms = 0;
    float kernel_ms = 0;
    float total_ms = 0;

    void print(const std::string& query_name) const {
        std::cout << query_name << " timing:" << std::endl;
        std::cout << "  Hash build: " << hash_build_ms << " ms" << std::endl;
        std::cout << "  Kernel:     " << kernel_ms << " ms" << std::endl;
        std::cout << "  Total:      " << total_ms << " ms" << std::endl;
    }
};

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

}  // namespace ssb_new
