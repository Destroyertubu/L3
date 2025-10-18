#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

#define SF 20

// Data directory
#ifndef SSB_DATA_DIR
#define SSB_DATA_DIR "./ssb_data/"
#endif

#define BASE_PATH SSB_DATA_DIR

#if SF == 1
#define DATA_DIR BASE_PATH
#define LO_LEN 6001171
#define P_LEN 200000
#define S_LEN 2000
#define C_LEN 30000
#define D_LEN 2556
#elif SF == 10
#define DATA_DIR BASE_PATH
#define LO_LEN 59986214
#define P_LEN 800000
#define S_LEN 20000
#define C_LEN 300000
#define D_LEN 2556
#else // SF == 20
#define DATA_DIR BASE_PATH
#define LO_LEN 119968352
#define P_LEN 1400000
#define S_LEN 40000
#define C_LEN 600000
#define D_LEN 2556
#endif

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA Error: " << cudaGetErrorString(err) \
                 << " at " << __FILE__ << ":" << __LINE__ << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int index_of(string* arr, int len, string val) {
    for (int i=0; i<len; i++)
        if (arr[i] == val)
            return i;
    return -1;
}

// Column name lookup
string lookup(string col_name) {
    string lineorder[] = { "lo_orderkey", "lo_linenumber", "lo_custkey", "lo_partkey", "lo_suppkey", "lo_orderdate", "lo_orderpriority", "lo_shippriority", "lo_quantity", "lo_extendedprice", "lo_ordtotalprice", "lo_discount", "lo_revenue", "lo_supplycost", "lo_tax", "lo_commitdate", "lo_shipmode"};
    string part[] = {"p_partkey", "p_name", "p_mfgr", "p_category", "p_brand1", "p_color", "p_type", "p_size", "p_container"};
    string supplier[] = {"s_suppkey", "s_name", "s_address", "s_city", "s_nation", "s_region", "s_phone"};
    string customer[] = {"c_custkey", "c_name", "c_address", "c_city", "c_nation", "c_region", "c_phone", "c_mktsegment"};
    string date[] = {"d_datekey", "d_date", "d_dayofweek", "d_month", "d_year", "d_yearmonthnum", "d_yearmonth", "d_daynuminweek", "d_daynuminmonth", "d_daynuminyear", "d_sellingseason", "d_lastdayinweekfl", "d_lastdayinmonthfl", "d_holidayfl", "d_weekdayfl"};

    if (col_name[0] == 'l') {
        int index = index_of(lineorder, 17, col_name);
        return "LINEORDER" + to_string(index);
    } else if (col_name[0] == 's') {
        int index = index_of(supplier, 7, col_name);
        return "SUPPLIER" + to_string(index);
    } else if (col_name[0] == 'c') {
        int index = index_of(customer, 8, col_name);
        return "CUSTOMER" + to_string(index);
    } else if (col_name[0] == 'p') {
        int index = index_of(part, 9, col_name);
        return "PART" + to_string(index);
    } else if (col_name[0] == 'd') {
        int index = index_of(date, 15, col_name);
        return "DDATE" + to_string(index);
    } else {
        cout << "Unknown column " << col_name << endl;
        exit(1);
    }
    return "";
}

// Load column into vector (for optimized L3)
template<typename T>
vector<T> loadColumn(string col_name, int num_entries) {
    vector<T> data(num_entries);
    string filename = string(DATA_DIR) + lookup(col_name) + ".bin";

    ifstream colData (filename.c_str(), ios::in | ios::binary);
    if (!colData) {
        cout << "Error: Cannot open file " << filename << endl;
        return vector<T>();
    }

    colData.read((char*)data.data(), num_entries * sizeof(T));
    colData.close();

    return data;
}
