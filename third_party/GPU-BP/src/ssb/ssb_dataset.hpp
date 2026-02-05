#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace ssb {

// SF=20 (matches the previous SSB setup we used in ssb-fslgpu).
constexpr std::size_t LO_LEN = 120000000;
constexpr std::size_t P_LEN  = 1000000;
constexpr std::size_t S_LEN  = 40000;
constexpr std::size_t C_LEN  = 600000;
constexpr std::size_t D_LEN  = 2557;

inline const char* default_data_dir() {
    return "/root/autodl-tmp/test/ssb_data/";
}

struct HostColumnView {
    const void*     data = nullptr;
    std::size_t     size = 0;
    std::size_t     elem_size = 0;
};

template <typename T>
struct HostColumn {
    std::unique_ptr<T[]> data;
    std::size_t          size = 0;

    T*       get() { return data.get(); }
    const T* get() const { return data.get(); }
};

inline int index_of(const char* const* arr, int len, const std::string& val) {
    for (int i = 0; i < len; ++i) {
        if (val == arr[i]) {
            return i;
        }
    }
    return -1;
}

inline std::string file_name_for_column(const std::string& col_name) {
    static const char* const lineorder[] = {
        "lo_orderkey",
        "lo_linenumber",
        "lo_custkey",
        "lo_partkey",
        "lo_suppkey",
        "lo_orderdate",
        "lo_orderpriority",
        "lo_shippriority",
        "lo_quantity",
        "lo_extendedprice",
        "lo_ordtotalprice",
        "lo_discount",
        "lo_revenue",
        "lo_supplycost",
        "lo_tax",
        "lo_commitdate",
        "lo_shipmode",
    };
    static const char* const part[] = {
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_category",
        "p_brand1",
        "p_color",
        "p_type",
        "p_size",
        "p_container",
    };
    static const char* const supplier[] = {
        "s_suppkey",
        "s_name",
        "s_address",
        "s_city",
        "s_nation",
        "s_region",
        "s_phone",
    };
    static const char* const customer[] = {
        "c_custkey",
        "c_name",
        "c_address",
        "c_city",
        "c_nation",
        "c_region",
        "c_phone",
        "c_mktsegment",
    };
    static const char* const date[] = {
        "d_datekey",
        "d_date",
        "d_dayofweek",
        "d_month",
        "d_year",
        "d_yearmonthnum",
        "d_yearmonth",
        "d_daynuminweek",
        "d_daynuminmonth",
        "d_daynuminyear",
        "d_sellingseason",
        "d_lastdayinweekfl",
        "d_lastdayinmonthfl",
        "d_holidayfl",
        "d_weekdayfl",
    };

    if (col_name.empty()) {
        throw std::runtime_error("empty column name");
    }

    const char prefix = col_name[0];
    if (prefix == 'l') {
        const int idx = index_of(lineorder, 17, col_name);
        if (idx < 0) {
            throw std::runtime_error("unknown column: " + col_name);
        }
        return "LINEORDER" + std::to_string(idx) + ".bin";
    }
    if (prefix == 'p') {
        const int idx = index_of(part, 9, col_name);
        if (idx < 0) {
            throw std::runtime_error("unknown column: " + col_name);
        }
        return "PART" + std::to_string(idx) + ".bin";
    }
    if (prefix == 's') {
        const int idx = index_of(supplier, 7, col_name);
        if (idx < 0) {
            throw std::runtime_error("unknown column: " + col_name);
        }
        return "SUPPLIER" + std::to_string(idx) + ".bin";
    }
    if (prefix == 'c') {
        const int idx = index_of(customer, 8, col_name);
        if (idx < 0) {
            throw std::runtime_error("unknown column: " + col_name);
        }
        return "CUSTOMER" + std::to_string(idx) + ".bin";
    }
    if (prefix == 'd') {
        const int idx = index_of(date, 15, col_name);
        if (idx < 0) {
            throw std::runtime_error("unknown column: " + col_name);
        }
        return "DDATE" + std::to_string(idx) + ".bin";
    }

    throw std::runtime_error("unknown column prefix: " + col_name);
}

template <typename T>
HostColumn<T> load_column(const std::string& data_dir, const std::string& col_name, std::size_t num_entries) {
    HostColumn<T> col;
    col.size = num_entries;
    col.data.reset(new T[num_entries]); // uninitialized for POD types

    const std::string filename = data_dir + file_name_for_column(col_name);
    std::ifstream     in(filename.c_str(), std::ios::in | std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open: " + filename);
    }
    in.read(reinterpret_cast<char*>(col.data.get()), static_cast<std::streamsize>(num_entries * sizeof(T)));
    if (!in) {
        throw std::runtime_error("failed to read: " + filename);
    }
    return col;
}

} // namespace ssb

