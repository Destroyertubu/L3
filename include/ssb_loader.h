/**
 * SSB (Star Schema Benchmark) Data Loader for GLECO Testing
 *
 * Loads SSB data generated at Scale Factor 20:
 * - LINEORDER: 119,968,352 rows (17 columns)
 * - PART: 1,400,000 rows (8 columns)
 * - SUPPLIER: 40,000 rows (4 columns)
 * - CUSTOMER: 600,000 rows (5 columns)
 * - DATE: 2,556 rows (11 columns)
 *
 * All columns are stored as uint32_t binary arrays.
 */

#ifndef SSB_LOADER_H
#define SSB_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstdlib>

// SSB Table sizes (SF=20)
constexpr size_t LO_LEN = 119968352;  // LINEORDER rows
constexpr size_t P_LEN = 1400000;     // PART rows
constexpr size_t S_LEN = 40000;       // SUPPLIER rows
constexpr size_t C_LEN = 600000;      // CUSTOMER rows
constexpr size_t D_LEN = 2556;        // DATE rows

/**
 * Load a binary column file
 * @param filename Path to the .bin file
 * @param expected_size Expected number of elements (0 = no check)
 * @return Vector containing the loaded data
 */
inline std::vector<uint32_t> loadColumn(const std::string& filename, size_t expected_size = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "ERROR: Failed to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(uint32_t);

    if (expected_size > 0 && num_elements != expected_size) {
        std::cerr << "WARNING: Expected " << expected_size << " elements, got "
                  << num_elements << " in " << filename << std::endl;
    }

    std::vector<uint32_t> data(num_elements);
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(uint32_t));

    if (!file) {
        std::cerr << "ERROR: Failed to read data from " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    return data;
}

// LINEORDER table column indices
struct LineOrderColumns {
    std::vector<uint32_t> lo_orderkey;        // LINEORDER0
    std::vector<uint32_t> lo_linenumber;      // LINEORDER1
    std::vector<uint32_t> lo_custkey;         // LINEORDER2
    std::vector<uint32_t> lo_partkey;         // LINEORDER3
    std::vector<uint32_t> lo_suppkey;         // LINEORDER4
    std::vector<uint32_t> lo_orderdate;       // LINEORDER5
    std::vector<uint32_t> lo_orderpriority;   // LINEORDER6
    std::vector<uint32_t> lo_shippriority;    // LINEORDER7
    std::vector<uint32_t> lo_quantity;        // LINEORDER8
    std::vector<uint32_t> lo_extendedprice;   // LINEORDER9
    std::vector<uint32_t> lo_ordtotalprice;   // LINEORDER10
    std::vector<uint32_t> lo_discount;        // LINEORDER11
    std::vector<uint32_t> lo_revenue;         // LINEORDER12
    std::vector<uint32_t> lo_supplycost;      // LINEORDER13
    std::vector<uint32_t> lo_tax;             // LINEORDER14
    std::vector<uint32_t> lo_commitdate;      // LINEORDER15
    std::vector<uint32_t> lo_shipmode;        // LINEORDER16

    void load(const std::string& data_dir) {
        std::cout << "Loading LINEORDER table..." << std::endl;
        lo_orderkey = loadColumn(data_dir + "/LINEORDER0.bin", LO_LEN);
        lo_linenumber = loadColumn(data_dir + "/LINEORDER1.bin", LO_LEN);
        lo_custkey = loadColumn(data_dir + "/LINEORDER2.bin", LO_LEN);
        lo_partkey = loadColumn(data_dir + "/LINEORDER3.bin", LO_LEN);
        lo_suppkey = loadColumn(data_dir + "/LINEORDER4.bin", LO_LEN);
        lo_orderdate = loadColumn(data_dir + "/LINEORDER5.bin", LO_LEN);
        lo_orderpriority = loadColumn(data_dir + "/LINEORDER6.bin", LO_LEN);
        lo_shippriority = loadColumn(data_dir + "/LINEORDER7.bin", LO_LEN);
        lo_quantity = loadColumn(data_dir + "/LINEORDER8.bin", LO_LEN);
        lo_extendedprice = loadColumn(data_dir + "/LINEORDER9.bin", LO_LEN);
        lo_ordtotalprice = loadColumn(data_dir + "/LINEORDER10.bin", LO_LEN);
        lo_discount = loadColumn(data_dir + "/LINEORDER11.bin", LO_LEN);
        lo_revenue = loadColumn(data_dir + "/LINEORDER12.bin", LO_LEN);
        lo_supplycost = loadColumn(data_dir + "/LINEORDER13.bin", LO_LEN);
        lo_tax = loadColumn(data_dir + "/LINEORDER14.bin", LO_LEN);
        lo_commitdate = loadColumn(data_dir + "/LINEORDER15.bin", LO_LEN);
        lo_shipmode = loadColumn(data_dir + "/LINEORDER16.bin", LO_LEN);
        std::cout << "  Loaded " << LO_LEN << " rows" << std::endl;
    }
};

// PART table columns
struct PartColumns {
    std::vector<uint32_t> p_partkey;      // PART0
    std::vector<uint32_t> p_mfgr;         // PART2
    std::vector<uint32_t> p_category;     // PART3
    std::vector<uint32_t> p_brand1;       // PART4
    std::vector<uint32_t> p_color;        // PART5
    std::vector<uint32_t> p_type;         // PART6
    std::vector<uint32_t> p_size;         // PART7
    std::vector<uint32_t> p_container;    // PART8

    void load(const std::string& data_dir) {
        std::cout << "Loading PART table..." << std::endl;
        p_partkey = loadColumn(data_dir + "/PART0.bin", P_LEN);
        p_mfgr = loadColumn(data_dir + "/PART2.bin", P_LEN);
        p_category = loadColumn(data_dir + "/PART3.bin", P_LEN);
        p_brand1 = loadColumn(data_dir + "/PART4.bin", P_LEN);
        p_color = loadColumn(data_dir + "/PART5.bin", P_LEN);
        p_type = loadColumn(data_dir + "/PART6.bin", P_LEN);
        p_size = loadColumn(data_dir + "/PART7.bin", P_LEN);
        p_container = loadColumn(data_dir + "/PART8.bin", P_LEN);
        std::cout << "  Loaded " << P_LEN << " rows" << std::endl;
    }
};

// SUPPLIER table columns
struct SupplierColumns {
    std::vector<uint32_t> s_suppkey;      // SUPPLIER0
    std::vector<uint32_t> s_city;         // SUPPLIER3
    std::vector<uint32_t> s_nation;       // SUPPLIER4
    std::vector<uint32_t> s_region;       // SUPPLIER5

    void load(const std::string& data_dir) {
        std::cout << "Loading SUPPLIER table..." << std::endl;
        s_suppkey = loadColumn(data_dir + "/SUPPLIER0.bin", S_LEN);
        s_city = loadColumn(data_dir + "/SUPPLIER3.bin", S_LEN);
        s_nation = loadColumn(data_dir + "/SUPPLIER4.bin", S_LEN);
        s_region = loadColumn(data_dir + "/SUPPLIER5.bin", S_LEN);
        std::cout << "  Loaded " << S_LEN << " rows" << std::endl;
    }
};

// CUSTOMER table columns
struct CustomerColumns {
    std::vector<uint32_t> c_custkey;      // CUSTOMER0
    std::vector<uint32_t> c_city;         // CUSTOMER3
    std::vector<uint32_t> c_nation;       // CUSTOMER4
    std::vector<uint32_t> c_region;       // CUSTOMER5
    std::vector<uint32_t> c_mktsegment;   // CUSTOMER7

    void load(const std::string& data_dir) {
        std::cout << "Loading CUSTOMER table..." << std::endl;
        c_custkey = loadColumn(data_dir + "/CUSTOMER0.bin", C_LEN);
        c_city = loadColumn(data_dir + "/CUSTOMER3.bin", C_LEN);
        c_nation = loadColumn(data_dir + "/CUSTOMER4.bin", C_LEN);
        c_region = loadColumn(data_dir + "/CUSTOMER5.bin", C_LEN);
        c_mktsegment = loadColumn(data_dir + "/CUSTOMER7.bin", C_LEN);
        std::cout << "  Loaded " << C_LEN << " rows" << std::endl;
    }
};

// DATE table columns
struct DateColumns {
    std::vector<uint32_t> d_datekey;          // DDATE0
    std::vector<uint32_t> d_year;             // DDATE4
    std::vector<uint32_t> d_yearmonthnum;     // DDATE5
    std::vector<uint32_t> d_daynuminweek;     // DDATE7
    std::vector<uint32_t> d_daynuminmonth;    // DDATE8
    std::vector<uint32_t> d_daynuminyear;     // DDATE9
    std::vector<uint32_t> d_sellingseason;    // DDATE10
    std::vector<uint32_t> d_lastdayinweekfl;  // DDATE11
    std::vector<uint32_t> d_lastdayinmonthfl; // DDATE12
    std::vector<uint32_t> d_holidayfl;        // DDATE13
    std::vector<uint32_t> d_weekdayfl;        // DDATE14

    void load(const std::string& data_dir) {
        std::cout << "Loading DATE table..." << std::endl;
        d_datekey = loadColumn(data_dir + "/DDATE0.bin", D_LEN);
        d_year = loadColumn(data_dir + "/DDATE4.bin", D_LEN);
        d_yearmonthnum = loadColumn(data_dir + "/DDATE5.bin", D_LEN);
        d_daynuminweek = loadColumn(data_dir + "/DDATE7.bin", D_LEN);
        d_daynuminmonth = loadColumn(data_dir + "/DDATE8.bin", D_LEN);
        d_daynuminyear = loadColumn(data_dir + "/DDATE9.bin", D_LEN);
        d_sellingseason = loadColumn(data_dir + "/DDATE10.bin", D_LEN);
        d_lastdayinweekfl = loadColumn(data_dir + "/DDATE11.bin", D_LEN);
        d_lastdayinmonthfl = loadColumn(data_dir + "/DDATE12.bin", D_LEN);
        d_holidayfl = loadColumn(data_dir + "/DDATE13.bin", D_LEN);
        d_weekdayfl = loadColumn(data_dir + "/DDATE14.bin", D_LEN);
        std::cout << "  Loaded " << D_LEN << " rows" << std::endl;
    }
};

// Main SSB data structure
struct SSBData {
    LineOrderColumns lineorder;
    PartColumns part;
    SupplierColumns supplier;
    CustomerColumns customer;
    DateColumns date;

    void loadAll(const std::string& data_dir) {
        std::cout << "Loading SSB data from: " << data_dir << std::endl;
        std::cout << std::string(70, '=') << std::endl;
        lineorder.load(data_dir);
        part.load(data_dir);
        supplier.load(data_dir);
        customer.load(data_dir);
        date.load(data_dir);
        std::cout << std::string(70, '=') << std::endl;
        std::cout << "All SSB tables loaded successfully!" << std::endl;
    }
};

#endif // SSB_LOADER_H
