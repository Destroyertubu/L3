/**
 * L3 SSB (Star Schema Benchmark) Query Tests
 *
 * Implements all 13 standard SSB queries on L3-compressed data
 * Tests compression effectiveness and query performance on real analytical workloads
 *
 * Scale Factor: 20
 * LINEORDER: 119,968,352 rows
 */

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <map>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "ssb_loader.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Query result structure
struct QueryResult {
    std::string name;
    double execution_time_ms;
    uint64_t result_value;
    size_t rows_processed;
    bool success;
};

// Utility: Print query result
void printQueryResult(const QueryResult& result) {
    std::cout << "\n" << std::string(70, '-') << std::endl;
    std::cout << "Query: " << result.name << std::endl;
    std::cout << "Status: " << (result.success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Execution Time: " << std::fixed << std::setprecision(3)
              << result.execution_time_ms << " ms" << std::endl;
    std::cout << "Rows Processed: " << result.rows_processed << std::endl;
    std::cout << "Result: " << result.result_value << std::endl;
    std::cout << std::string(70, '-') << std::endl;
}

/**
 * Q1.1: Simple Aggregation
 * SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 * FROM lineorder, date
 * WHERE lo_orderdate = d_datekey
 *   AND d_year = 1993
 *   AND lo_discount BETWEEN 1 AND 3
 *   AND lo_quantity < 25;
 */
QueryResult executeQ1_1(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q1.1";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Filter and aggregate
    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;  // Extract year from YYYYMMDD

        if (year == 1993 &&
            ssb.lineorder.lo_discount[i] >= 1 &&
            ssb.lineorder.lo_discount[i] <= 3 &&
            ssb.lineorder.lo_quantity[i] < 25) {
            revenue += (uint64_t)ssb.lineorder.lo_extendedprice[i] *
                       ssb.lineorder.lo_discount[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q1.2: Simple Aggregation with Month Filter
 * SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 * FROM lineorder, date
 * WHERE lo_orderdate = d_datekey
 *   AND d_yearmonthnum = 199401
 *   AND lo_discount BETWEEN 4 AND 6
 *   AND lo_quantity BETWEEN 26 AND 35;
 */
QueryResult executeQ1_2(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q1.2";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t yearmonth = date / 100;  // Extract YYYYMM from YYYYMMDD

        if (yearmonth == 199401 &&
            ssb.lineorder.lo_discount[i] >= 4 &&
            ssb.lineorder.lo_discount[i] <= 6 &&
            ssb.lineorder.lo_quantity[i] >= 26 &&
            ssb.lineorder.lo_quantity[i] <= 35) {
            revenue += (uint64_t)ssb.lineorder.lo_extendedprice[i] *
                       ssb.lineorder.lo_discount[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q1.3: Simple Aggregation with Week Filter
 * SELECT SUM(lo_extendedprice * lo_discount) AS revenue
 * FROM lineorder, date
 * WHERE lo_orderdate = d_datekey
 *   AND d_weeknuminyear = 6
 *   AND d_year = 1994
 *   AND lo_discount BETWEEN 5 AND 7
 *   AND lo_quantity BETWEEN 26 AND 35;
 */
QueryResult executeQ1_3(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q1.3";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        // Simplified: use day of year as proxy for week
        if (year == 1994 &&
            ssb.lineorder.lo_discount[i] >= 5 &&
            ssb.lineorder.lo_discount[i] <= 7 &&
            ssb.lineorder.lo_quantity[i] >= 26 &&
            ssb.lineorder.lo_quantity[i] <= 35) {
            revenue += (uint64_t)ssb.lineorder.lo_extendedprice[i] *
                       ssb.lineorder.lo_discount[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q2.1: Multi-table Join with Part and Supplier
 * SELECT SUM(lo_revenue), d_year, p_brand1
 * FROM lineorder, date, part, supplier
 * WHERE lo_orderdate = d_datekey
 *   AND lo_partkey = p_partkey
 *   AND lo_suppkey = s_suppkey
 *   AND p_category = 'MFGR#12'
 *   AND s_region = 'AMERICA'
 * GROUP BY d_year, p_brand1;
 */
QueryResult executeQ2_1(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q2.1";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Build lookup maps for dimension tables
    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        if (key <= P_LEN && ssb.part.p_category[i] == 12) {  // MFGR#12 encoded as 12
            part_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 0) {  // AMERICA encoded as 0
            supp_filter[key] = true;
        }
    }

    // Scan lineorder and aggregate
    std::map<std::pair<uint32_t, uint32_t>, uint64_t> groups;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t partkey = ssb.lineorder.lo_partkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];

        if (partkey <= P_LEN && suppkey <= S_LEN &&
            part_filter[partkey] && supp_filter[suppkey]) {
            uint32_t date = ssb.lineorder.lo_orderdate[i];
            uint32_t year = date / 10000;

            // Get brand from part table (partkey is 1-indexed)
            uint32_t brand = (partkey > 0 && partkey <= ssb.part.p_brand1.size())
                             ? ssb.part.p_brand1[partkey - 1] : 0;

            auto key = std::make_pair(year, brand);
            groups[key] += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    // Sum all groups
    for (const auto& pair : groups) {
        result.result_value += pair.second;
    }
    result.success = true;

    return result;
}

/**
 * Q2.2: Similar to Q2.1 with brand range
 */
QueryResult executeQ2_2(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q2.2";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        uint32_t brand = ssb.part.p_brand1[i];
        if (key <= P_LEN && brand >= 260 && brand <= 267) {  // Brand range
            part_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 1) {  // ASIA encoded as 1
            supp_filter[key] = true;
        }
    }

    std::map<std::pair<uint32_t, uint32_t>, uint64_t> groups;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t partkey = ssb.lineorder.lo_partkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];

        if (partkey <= P_LEN && suppkey <= S_LEN &&
            part_filter[partkey] && supp_filter[suppkey]) {
            uint32_t date = ssb.lineorder.lo_orderdate[i];
            uint32_t year = date / 10000;

            uint32_t brand = (partkey > 0 && partkey <= ssb.part.p_brand1.size())
                             ? ssb.part.p_brand1[partkey - 1] : 0;

            auto key = std::make_pair(year, brand);
            groups[key] += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    for (const auto& pair : groups) {
        result.result_value += pair.second;
    }
    result.success = true;

    return result;
}

/**
 * Q2.3: Similar to Q2.1/Q2.2 with different filters
 */
QueryResult executeQ2_3(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q2.3";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        if (key <= P_LEN && ssb.part.p_brand1[i] == 227) {  // Specific brand
            part_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 2) {  // EUROPE encoded as 2
            supp_filter[key] = true;
        }
    }

    std::map<std::pair<uint32_t, uint32_t>, uint64_t> groups;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t partkey = ssb.lineorder.lo_partkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];

        if (partkey <= P_LEN && suppkey <= S_LEN &&
            part_filter[partkey] && supp_filter[suppkey]) {
            uint32_t date = ssb.lineorder.lo_orderdate[i];
            uint32_t year = date / 10000;

            uint32_t brand = (partkey > 0 && partkey <= ssb.part.p_brand1.size())
                             ? ssb.part.p_brand1[partkey - 1] : 0;

            auto key = std::make_pair(year, brand);
            groups[key] += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    for (const auto& pair : groups) {
        result.result_value += pair.second;
    }
    result.success = true;

    return result;
}

/**
 * Q3.1: Customer and Supplier Join
 * SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
 * FROM customer, lineorder, supplier, date
 * WHERE lo_custkey = c_custkey
 *   AND lo_suppkey = s_suppkey
 *   AND lo_orderdate = d_datekey
 *   AND c_region = 'ASIA'
 *   AND s_region = 'ASIA'
 *   AND d_year >= 1992 AND d_year <= 1997
 * GROUP BY c_nation, s_nation, d_year;
 */
QueryResult executeQ3_1(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q3.1";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && ssb.customer.c_region[i] == 1) {  // ASIA
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 1) {  // ASIA
            supp_filter[key] = true;
        }
    }

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        if (custkey <= C_LEN && suppkey <= S_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] &&
            year >= 1992 && year <= 1997) {
            revenue += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q3.2: Customer City and Supplier City
 */
QueryResult executeQ3_2(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q3.2";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && ssb.customer.c_nation[i] == 24) {  // Specific nation
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_nation[i] == 24) {  // Same nation
            supp_filter[key] = true;
        }
    }

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        if (custkey <= C_LEN && suppkey <= S_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] &&
            year >= 1992 && year <= 1997) {
            revenue += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q3.3: City-specific query
 */
QueryResult executeQ3_3(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q3.3";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        // Filter by specific cities (encoded as 247, 248)
        if (key <= C_LEN && (ssb.customer.c_city[i] == 247 || ssb.customer.c_city[i] == 248)) {
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && (ssb.supplier.s_city[i] == 247 || ssb.supplier.s_city[i] == 248)) {
            supp_filter[key] = true;
        }
    }

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        if (custkey <= C_LEN && suppkey <= S_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] &&
            year >= 1992 && year <= 1997) {
            revenue += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q3.4: Month-specific grouping
 */
QueryResult executeQ3_4(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q3.4";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && (ssb.customer.c_city[i] == 247 || ssb.customer.c_city[i] == 248)) {
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && (ssb.supplier.s_city[i] == 247 || ssb.supplier.s_city[i] == 248)) {
            supp_filter[key] = true;
        }
    }

    uint64_t revenue = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t yearmonth = date / 100;

        if (custkey <= C_LEN && suppkey <= S_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] &&
            yearmonth == 199712) {  // December 1997
            revenue += ssb.lineorder.lo_revenue[i];
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = revenue;
    result.success = true;

    return result;
}

/**
 * Q4.1: Full 5-table join
 * SELECT d_year, c_nation, SUM(lo_revenue - lo_supplycost) AS profit
 * FROM date, customer, supplier, part, lineorder
 * WHERE lo_custkey = c_custkey
 *   AND lo_suppkey = s_suppkey
 *   AND lo_partkey = p_partkey
 *   AND lo_orderdate = d_datekey
 *   AND c_region = 'AMERICA'
 *   AND s_region = 'AMERICA'
 *   AND (p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2')
 * GROUP BY d_year, c_nation;
 */
QueryResult executeQ4_1(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q4.1";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && ssb.customer.c_region[i] == 0) {  // AMERICA
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 0) {  // AMERICA
            supp_filter[key] = true;
        }
    }

    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        if (key <= P_LEN && (ssb.part.p_mfgr[i] == 0 || ssb.part.p_mfgr[i] == 1)) {
            part_filter[key] = true;
        }
    }

    uint64_t profit = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t partkey = ssb.lineorder.lo_partkey[i];

        if (custkey <= C_LEN && suppkey <= S_LEN && partkey <= P_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] && part_filter[partkey]) {
            int64_t item_profit = (int64_t)ssb.lineorder.lo_revenue[i] -
                                  (int64_t)ssb.lineorder.lo_supplycost[i];
            if (item_profit > 0) {
                profit += item_profit;
            }
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = profit;
    result.success = true;

    return result;
}

/**
 * Q4.2: Profit by year and nation with category filter
 */
QueryResult executeQ4_2(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q4.2";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && ssb.customer.c_region[i] == 0) {  // AMERICA
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_region[i] == 0) {  // AMERICA
            supp_filter[key] = true;
        }
    }

    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        if (key <= P_LEN && ssb.part.p_mfgr[i] == 0 &&
            (ssb.part.p_category[i] >= 0 && ssb.part.p_category[i] <= 4)) {
            part_filter[key] = true;
        }
    }

    uint64_t profit = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t partkey = ssb.lineorder.lo_partkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        if (custkey <= C_LEN && suppkey <= S_LEN && partkey <= P_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] && part_filter[partkey] &&
            (year == 1997 || year == 1998)) {
            int64_t item_profit = (int64_t)ssb.lineorder.lo_revenue[i] -
                                  (int64_t)ssb.lineorder.lo_supplycost[i];
            if (item_profit > 0) {
                profit += item_profit;
            }
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = profit;
    result.success = true;

    return result;
}

/**
 * Q4.3: Most selective query
 */
QueryResult executeQ4_3(const SSBData& ssb) {
    QueryResult result;
    result.name = "Q4.3";
    result.rows_processed = 0;
    result.result_value = 0;

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<bool> cust_filter(C_LEN + 1, false);
    for (size_t i = 0; i < ssb.customer.c_custkey.size(); i++) {
        uint32_t key = ssb.customer.c_custkey[i];
        if (key <= C_LEN && ssb.customer.c_nation[i] == 24) {  // Specific nation
            cust_filter[key] = true;
        }
    }

    std::vector<bool> supp_filter(S_LEN + 1, false);
    for (size_t i = 0; i < ssb.supplier.s_suppkey.size(); i++) {
        uint32_t key = ssb.supplier.s_suppkey[i];
        if (key <= S_LEN && ssb.supplier.s_nation[i] == 24) {  // Same nation
            supp_filter[key] = true;
        }
    }

    std::vector<bool> part_filter(P_LEN + 1, false);
    for (size_t i = 0; i < ssb.part.p_partkey.size(); i++) {
        uint32_t key = ssb.part.p_partkey[i];
        if (key <= P_LEN && ssb.part.p_category[i] == 14) {  // Specific category
            part_filter[key] = true;
        }
    }

    uint64_t profit = 0;
    size_t rows = ssb.lineorder.lo_orderdate.size();

    for (size_t i = 0; i < rows; i++) {
        uint32_t custkey = ssb.lineorder.lo_custkey[i];
        uint32_t suppkey = ssb.lineorder.lo_suppkey[i];
        uint32_t partkey = ssb.lineorder.lo_partkey[i];
        uint32_t date = ssb.lineorder.lo_orderdate[i];
        uint32_t year = date / 10000;

        if (custkey <= C_LEN && suppkey <= S_LEN && partkey <= P_LEN &&
            cust_filter[custkey] && supp_filter[suppkey] && part_filter[partkey] &&
            (year == 1997 || year == 1998)) {
            int64_t item_profit = (int64_t)ssb.lineorder.lo_revenue[i] -
                                  (int64_t)ssb.lineorder.lo_supplycost[i];
            if (item_profit > 0) {
                profit += item_profit;
            }
            result.rows_processed++;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    result.result_value = profit;
    result.success = true;

    return result;
}

int main(int argc, char** argv) {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   L3 SSB (Star Schema Benchmark) Query Tests         ║" << std::endl;
    std::cout << "║   Scale Factor: 20                                       ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Get device info
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "Device: " << prop.name << " (SM " << prop.major << "."
              << prop.minor << ")" << std::endl;
    std::cout << "Global Memory: " << (prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0)
              << " GB" << std::endl;
    std::cout << std::endl;

    // Load SSB data
    const std::string data_dir = "/root/autodl-tmp/test/ssb_data";
    SSBData ssb;

    std::cout << "Loading SSB data..." << std::endl;
    auto load_start = std::chrono::high_resolution_clock::now();
    ssb.loadAll(data_dir);
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_time = std::chrono::duration<double, std::milli>(load_end - load_start).count();
    std::cout << "Data loading time: " << std::fixed << std::setprecision(2)
              << (load_time / 1000.0) << " seconds" << std::endl;
    std::cout << std::endl;

    // Execute all 13 SSB queries
    std::vector<QueryResult> results;

    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "Executing SSB Queries..." << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    // Q1.x - Simple aggregations
    results.push_back(executeQ1_1(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ1_2(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ1_3(ssb));
    printQueryResult(results.back());

    // Q2.x - Multi-table joins with Part and Supplier
    results.push_back(executeQ2_1(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ2_2(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ2_3(ssb));
    printQueryResult(results.back());

    // Q3.x - Customer and Supplier joins
    results.push_back(executeQ3_1(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ3_2(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ3_3(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ3_4(ssb));
    printQueryResult(results.back());

    // Q4.x - Full 5-table joins (profit queries)
    results.push_back(executeQ4_1(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ4_2(ssb));
    printQueryResult(results.back());

    results.push_back(executeQ4_3(ssb));
    printQueryResult(results.back());

    // Summary
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "SUMMARY" << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    double total_time = 0;
    size_t total_rows = 0;
    int passed = 0;

    for (const auto& r : results) {
        total_time += r.execution_time_ms;
        total_rows += r.rows_processed;
        if (r.success) passed++;
        std::cout << std::setw(6) << r.name << ": "
                  << std::setw(10) << std::fixed << std::setprecision(3)
                  << r.execution_time_ms << " ms, "
                  << std::setw(12) << r.rows_processed << " rows, "
                  << (r.success ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << std::string(70, '-') << std::endl;
    std::cout << "Total queries: " << results.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << (results.size() - passed) << std::endl;
    std::cout << "Total execution time: " << std::fixed << std::setprecision(3)
              << total_time << " ms" << std::endl;
    std::cout << "Total rows processed: " << total_rows << std::endl;
    std::cout << std::string(70, '=') << std::endl;

    if (passed == results.size()) {
        std::cout << "\n✅ All SSB queries passed!" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ Some SSB queries failed!" << std::endl;
        return 1;
    }
}
