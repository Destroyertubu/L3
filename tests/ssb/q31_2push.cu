// SSB Query 3.1 with L32 Compression - RANDOM ACCESS VERSION (_2push version)
// Query: SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue
//        FROM customer, lineorder, supplier, date
//        WHERE lo_custkey = c_custkey
//              AND lo_suppkey = s_suppkey
//              AND lo_orderdate = d_datekey
//              AND c_region = 'ASIA' (encoded as 2)
//              AND s_region = 'ASIA' (encoded as 2)
//              AND d_year >= 1992 AND d_year <= 1997
//        GROUP BY c_nation, s_nation, d_year
//        ORDER BY d_year ASC, revenue DESC


#include "L3_codec.hpp"
#include "L3_predicate_pushdown.cuh"
#include "ssb_L3_utils.cuh"

#include "ssb_utils.h"
#include <iostream>
#include <chrono>
#include <map>

using namespace std;

struct CustomerEntry {
    uint32_t c_nation;
    uint32_t c_region;
};

struct SupplierEntry {
    uint32_t s_nation;
    uint32_t s_region;
};

struct DateEntry {
    uint32_t d_year;
};

struct Candidate {
    int global_idx;
};

struct FinalCandidate {
    int global_idx;
    uint32_t c_nation;
    uint32_t s_nation;
};

struct GroupKey {
    uint32_t c_nation;
    uint32_t s_nation;
    uint32_t d_year;

    bool operator<(const GroupKey& other) const {
        if (d_year != other.d_year) return d_year < other.d_year;
        if (c_nation != other.c_nation) return c_nation < other.c_nation;
        return s_nation < other.s_nation;
    }
};

// Stage 1: Filter lo_orderdate with predicate pushdown (d_year >= 1992 AND d_year <= 1997)
// This corresponds to orderdate range [19920101, 19971231]
// NOTE: Using predicate pushdown to prune partitions that don't overlap with this range!

// Stage 2: For orderdate candidates, random access custkey/suppkey, filter, then access revenue
template<typename T>
__global__ void ssb_q31_stage2_filter_and_aggregate(
    const CompressedDataL3<T>* c_custkey,
    const CompressedDataL3<T>* c_suppkey,
    const CompressedDataL3<T>* c_orderdate,
    const CompressedDataL3<T>* c_revenue,
    const CustomerEntry* d_customer_ht,
    const SupplierEntry* d_supplier_ht,
    const DateEntry* d_date_ht,
    const int* d_candidate_indices,
    int num_candidates,
    unsigned long long* d_group_results,
    uint32_t* d_group_cnations,
    uint32_t* d_group_snations,
    uint32_t* d_group_years,
    int max_groups) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;

    int global_idx = d_candidate_indices[idx];

    // Random access lo_custkey
    T custkey = randomAccessDecompress<T>(c_custkey, global_idx);
    if (custkey == 0 || custkey > 300000) return;

    CustomerEntry cust = d_customer_ht[custkey];
    if (cust.c_region != 2) return; // c_region = 'ASIA'

    // Random access lo_suppkey
    T suppkey = randomAccessDecompress<T>(c_suppkey, global_idx);
    if (suppkey == 0 || suppkey > 20000) return;

    SupplierEntry supp = d_supplier_ht[suppkey];
    if (supp.s_region != 2) return; // s_region = 'ASIA'

    // Random access lo_orderdate for d_year (orderdate already filtered by pushdown)
    T orderdate = randomAccessDecompress<T>(c_orderdate, global_idx);
    if (orderdate >= 20000000) return;

    uint32_t d_year = d_date_ht[orderdate].d_year;
    if (d_year == 0 || d_year < 1992 || d_year > 1997) return;

    // Random access lo_revenue
    T revenue = randomAccessDecompress<T>(c_revenue, global_idx);

    // Hash-based GROUP BY (c_nation, s_nation, d_year)
    uint64_t combined_key = ((uint64_t)cust.c_nation << 32) | ((uint64_t)supp.s_nation << 16) | d_year;
    uint32_t hash = (uint32_t)(combined_key % max_groups);

    for (int probe = 0; probe < 64; probe++) {
        int gidx = (hash + probe * probe) % max_groups;

        if (d_group_cnations[gidx] == cust.c_nation &&
            d_group_snations[gidx] == supp.s_nation &&
            d_group_years[gidx] == d_year) {
            atomicAdd(&d_group_results[gidx], (unsigned long long)revenue);
            return;
        }

        if (d_group_cnations[gidx] == 0) {
            unsigned int old = atomicCAS(&d_group_cnations[gidx], 0, cust.c_nation);
            if (old == 0) {
                d_group_snations[gidx] = supp.s_nation;
                d_group_years[gidx] = d_year;
                atomicAdd(&d_group_results[gidx], (unsigned long long)revenue);
                return;
            } else if (old == cust.c_nation &&
                       d_group_snations[gidx] == supp.s_nation &&
                       d_group_years[gidx] == d_year) {
                atomicAdd(&d_group_results[gidx], (unsigned long long)revenue);
                return;
            }
        }
    }
}

int main(int argc, char** argv) {
    cout << "========================================================================" << endl;
    cout << "  SSB Query 3.1 with L32 Compression - RANDOM ACCESS" << endl;
    cout << "========================================================================" << endl;
    cout << endl;

    cout << "Query: SELECT c_nation, s_nation, d_year, SUM(lo_revenue) AS revenue" << endl;
    cout << "       FROM customer, lineorder, supplier, date" << endl;
    cout << "       WHERE lo_custkey = c_custkey" << endl;
    cout << "             AND lo_suppkey = s_suppkey" << endl;
    cout << "             AND lo_orderdate = d_datekey" << endl;
    cout << "             AND c_region = 'ASIA'" << endl;
    cout << "             AND s_region = 'ASIA'" << endl;
    cout << "             AND d_year >= 1992 AND d_year <= 1997" << endl;
    cout << "       GROUP BY c_nation, s_nation, d_year" << endl;
    cout << endl;

    int num_trials = 5;
    if (argc > 1) num_trials = atoi(argv[1]);

    cout << "Loading LINEORDER columns..." << endl;
    auto load_start = chrono::high_resolution_clock::now();

    vector<uint32_t> lo_orderdate = loadColumn<uint32_t>("lo_orderdate", LO_LEN);
    vector<uint32_t> lo_custkey = loadColumn<uint32_t>("lo_custkey", LO_LEN);
    vector<uint32_t> lo_suppkey = loadColumn<uint32_t>("lo_suppkey", LO_LEN);
    vector<uint32_t> lo_revenue = loadColumn<uint32_t>("lo_revenue", LO_LEN);

    cout << "Loading dimension tables..." << endl;
    vector<uint32_t> c_custkey = loadColumn<uint32_t>("c_custkey", 300000);
    vector<uint32_t> c_nation = loadColumn<uint32_t>("c_nation", 300000);
    vector<uint32_t> c_region = loadColumn<uint32_t>("c_region", 300000);

    vector<uint32_t> s_suppkey = loadColumn<uint32_t>("s_suppkey", 20000);
    vector<uint32_t> s_nation = loadColumn<uint32_t>("s_nation", 20000);
    vector<uint32_t> s_region = loadColumn<uint32_t>("s_region", 20000);

    vector<uint32_t> d_datekey = loadColumn<uint32_t>("d_datekey", 2556);
    vector<uint32_t> d_year = loadColumn<uint32_t>("d_year", 2556);

    auto load_end = chrono::high_resolution_clock::now();
    double load_time = chrono::duration<double>(load_end - load_start).count();
    cout << "✓ Loaded all columns in " << load_time << " seconds" << endl;
    cout << endl;

    cout << "Building hash tables..." << endl;
    vector<CustomerEntry> customer_ht(300001);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        customer_ht[c_custkey[i]].c_nation = c_nation[i];
        customer_ht[c_custkey[i]].c_region = c_region[i];
    }

    vector<SupplierEntry> supplier_ht(20001);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        supplier_ht[s_suppkey[i]].s_nation = s_nation[i];
        supplier_ht[s_suppkey[i]].s_region = s_region[i];
    }

    vector<DateEntry> date_ht(20000000);
    for (size_t i = 0; i < d_datekey.size(); i++) {
        date_ht[d_datekey[i]].d_year = d_year[i];
    }

    cout << "✓ Hash tables built" << endl;
    cout << endl;

    cout << "Compressing LINEORDER columns with L32..." << endl;
    
    
    auto compress_start = chrono::high_resolution_clock::now();

    CompressedDataL3<uint32_t>* c_orderdate = compressData(lo_orderdate, 1024);
    CompressedDataL3<uint32_t>* c_custkey_c = compressData(lo_custkey, 1024);
    CompressedDataL3<uint32_t>* c_suppkey_c = compressData(lo_suppkey, 1024);
    CompressedDataL3<uint32_t>* c_revenue = compressData(lo_revenue, 1024);

    auto compress_end = chrono::high_resolution_clock::now();
    double compress_time = chrono::duration<double>(compress_end - compress_start).count();
    cout << "✓ Compression complete (" << compress_time << " seconds)" << endl;
    cout << endl;

    CustomerEntry* d_customer_ht;
    SupplierEntry* d_supplier_ht;
    DateEntry* d_date_ht;

    CUDA_CHECK(cudaMalloc(&d_customer_ht, customer_ht.size() * sizeof(CustomerEntry)));
    CUDA_CHECK(cudaMalloc(&d_supplier_ht, supplier_ht.size() * sizeof(SupplierEntry)));
    CUDA_CHECK(cudaMalloc(&d_date_ht, date_ht.size() * sizeof(DateEntry)));

    CUDA_CHECK(cudaMemcpy(d_customer_ht, customer_ht.data(), customer_ht.size() * sizeof(CustomerEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_supplier_ht, supplier_ht.data(), supplier_ht.size() * sizeof(SupplierEntry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_date_ht, date_ht.data(), date_ht.size() * sizeof(DateEntry), cudaMemcpyHostToDevice));

    // Allocate for orderdate filtering with predicate pushdown
    const int MAX_CANDIDATES = 50000000;
    int* d_candidate_indices;
    int* d_num_candidates;
    unsigned long long* d_partitions_pruned;

    CUDA_CHECK(cudaMalloc(&d_candidate_indices, MAX_CANDIDATES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_candidates, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_partitions_pruned, sizeof(unsigned long long)));

    const int MAX_GROUPS = 50000;
    unsigned long long* d_group_results;
    uint32_t* d_group_cnations;
    uint32_t* d_group_snations;
    uint32_t* d_group_years;

    CUDA_CHECK(cudaMalloc(&d_group_results, MAX_GROUPS * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_group_cnations, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_snations, MAX_GROUPS * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_group_years, MAX_GROUPS * sizeof(uint32_t)));

    cout << "✓ Using predicate pushdown on lo_orderdate (d_year 1992-1997)" << endl;
    cout << endl;

    cout << "** Running Predicate Pushdown Query " << num_trials << " times **" << endl;
    cout << endl;

    for (int t = 0; t < num_trials; t++) {
        CUDA_CHECK(cudaMemset(d_num_candidates, 0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_partitions_pruned, 0, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_results, 0, MAX_GROUPS * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_group_cnations, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_snations, 0, MAX_GROUPS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_group_years, 0, MAX_GROUPS * sizeof(uint32_t)));

        auto query_start = chrono::high_resolution_clock::now();

        // Stage 1: Filter lo_orderdate with predicate pushdown
        // d_year >= 1992 AND d_year <= 1997 → orderdate in [19920101, 19980101)
        stage1_filter_with_predicate_pushdown<uint32_t><<<c_orderdate->num_partitions, 256>>>(
            c_orderdate->d_self,
            LO_LEN,
            (uint32_t)19920101,  // filter_min
            (uint32_t)19980101,  // filter_max (exclusive, so use 19980101 for <= 19971231)
            d_candidate_indices,
            d_num_candidates,
            d_partitions_pruned
        );

        CUDA_CHECK(cudaDeviceSynchronize());

        int h_num_candidates = 0;
        unsigned long long h_partitions_pruned = 0;
        CUDA_CHECK(cudaMemcpy(&h_num_candidates, d_num_candidates, sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_partitions_pruned, d_partitions_pruned, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        if (t == 0) {
            cout << "  Stage 1 results: " << h_num_candidates << " candidates, "
                 << h_partitions_pruned << " / " << c_orderdate->num_partitions << " partitions pruned" << endl;
        }

        // Stage 2: For orderdate candidates, access custkey/suppkey, filter, and aggregate
        if (h_num_candidates > 0 && h_num_candidates < MAX_CANDIDATES) {
            int blocks = (h_num_candidates + 255) / 256;
            ssb_q31_stage2_filter_and_aggregate<uint32_t><<<blocks, 256>>>(
                c_custkey_c->d_self,
                c_suppkey_c->d_self,
                c_orderdate->d_self,
                c_revenue->d_self,
                d_customer_ht,
                d_supplier_ht,
                d_date_ht,
                d_candidate_indices,
                h_num_candidates,
                d_group_results,
                d_group_cnations,
                d_group_snations,
                d_group_years,
                MAX_GROUPS
            );
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        auto query_end = chrono::high_resolution_clock::now();
        double query_time = chrono::duration<double, milli>(query_end - query_start).count();

        vector<unsigned long long> h_group_results(MAX_GROUPS);
        vector<uint32_t> h_group_cnations(MAX_GROUPS);
        vector<uint32_t> h_group_snations(MAX_GROUPS);
        vector<uint32_t> h_group_years(MAX_GROUPS);

        CUDA_CHECK(cudaMemcpy(h_group_results.data(), d_group_results, MAX_GROUPS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_cnations.data(), d_group_cnations, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_snations.data(), d_group_snations, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_group_years.data(), d_group_years, MAX_GROUPS * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        map<GroupKey, unsigned long long> results_map;
        for (int i = 0; i < MAX_GROUPS; i++) {
            if (h_group_cnations[i] != 0) {
                GroupKey key{h_group_cnations[i], h_group_snations[i], h_group_years[i]};
                results_map[key] += h_group_results[i];
            }
        }

        double selectivity = (double)h_num_candidates / LO_LEN * 100.0;
        cout << "{\"query\":31,\"optimization\":\"predicate_pushdown\",\"time_query\":" << query_time
             << ",\"num_groups\":" << results_map.size()
             << ",\"candidates\":" << h_num_candidates
             << ",\"partitions_pruned\":" << h_partitions_pruned
             << ",\"total_partitions\":" << c_orderdate->num_partitions
             << ",\"selectivity\":" << selectivity << "}" << endl;

        if (t == 0) {
            cout << "Sample results (c_nation, s_nation, d_year, revenue):" << endl;
            int count = 0;
            for (const auto& [key, revenue] : results_map) {
                cout << "  " << key.c_nation << ", " << key.s_nation << ", " << key.d_year << ": " << revenue << endl;
                if (++count >= 10) break;
            }
        }
    }

    cout << endl;

    CUDA_CHECK(cudaFree(d_customer_ht));
    CUDA_CHECK(cudaFree(d_supplier_ht));
    CUDA_CHECK(cudaFree(d_date_ht));
    CUDA_CHECK(cudaFree(d_candidate_indices));
    CUDA_CHECK(cudaFree(d_num_candidates));
    CUDA_CHECK(cudaFree(d_partitions_pruned));
    CUDA_CHECK(cudaFree(d_group_results));
    CUDA_CHECK(cudaFree(d_group_cnations));
    CUDA_CHECK(cudaFree(d_group_snations));
    CUDA_CHECK(cudaFree(d_group_years));

    freeCompressedData(c_orderdate);
    freeCompressedData(c_custkey_c);
    freeCompressedData(c_suppkey_c);
    freeCompressedData(c_revenue);

    cout << "========================================================================" << endl;

    return 0;
}
