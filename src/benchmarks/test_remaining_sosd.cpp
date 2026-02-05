/**
 * Quick test for SOSD datasets 13-20
 */

#include "benchmark_cpu_leco.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>

class Timer {
public:
    void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
    double elapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_time_).count();
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time_;
};

template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> data(file_size / sizeof(T));
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    return data;
}

template<typename T>
std::vector<T> loadSOSDBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open: " + filename);
    uint64_t num_elements = 0;
    file.read(reinterpret_cast<char*>(&num_elements), sizeof(uint64_t));
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    return data;
}

template<typename T>
bool verifyRandomAccess(const std::vector<T>& original,
                        const leco::LeCoCompressedBlock<T>& compressed, int samples = 1000) {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> dist(0, original.size() - 1);
    for (int i = 0; i < samples; i++) {
        size_t idx = dist(gen);
        if (leco::lecoDecodeAt(compressed, idx) != original[idx]) return false;
    }
    return true;
}

template<typename T>
bool runTest(const std::string& name, const std::vector<T>& data, size_t max_elem = 100000) {
    std::vector<T> test_data = data;
    if (max_elem > 0 && test_data.size() > max_elem) {
        test_data.resize(max_elem);
    }

    std::cout << "\n=== " << name << " ===" << std::endl;
    std::cout << "Elements: " << test_data.size() << std::endl;

    Timer timer;
    leco::LeCoConfig config;

    timer.start();
    auto compressed = leco::lecoEncode(test_data, config);
    double enc_time = timer.elapsedMs();

    timer.start();
    auto decoded = leco::lecoDecode(compressed);
    double dec_time = timer.elapsedMs();

    bool roundtrip_ok = (decoded.size() == test_data.size());
    if (roundtrip_ok) {
        for (size_t i = 0; i < test_data.size(); i++) {
            if (decoded[i] != test_data[i]) {
                std::cerr << "Mismatch at " << i << ": " << decoded[i] << " vs " << test_data[i] << std::endl;
                roundtrip_ok = false;
                break;
            }
        }
    }

    bool random_ok = verifyRandomAccess(test_data, compressed);

    std::cout << "Compression: " << std::fixed << std::setprecision(2)
              << compressed.compression_ratio << "x" << std::endl;
    std::cout << "Encode: " << enc_time << " ms, Decode: " << dec_time << " ms" << std::endl;
    std::cout << "Roundtrip: " << (roundtrip_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Random Access: " << (random_ok ? "PASSED" : "FAILED") << std::endl;

    return roundtrip_ok && random_ok;
}

int main(int argc, char* argv[]) {
    size_t max_elem = 100000;  // Default 100k for fast testing
    if (argc > 1) max_elem = std::stoull(argv[1]);

    std::cout << "Testing SOSD datasets 13-20 (max " << max_elem << " elements)" << std::endl;

    int passed = 0, failed = 0;

    // Dataset 13: medicare
    try {
        auto data = loadSOSDBinaryFile<uint64_t>("data/sosd/13-medicare.bin");
        if (runTest("13-medicare_uint64", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "13-medicare: " << e.what() << std::endl; failed++; }

    // Dataset 14: cosmos (no header)
    try {
        auto data = loadBinaryFile<uint32_t>("data/sosd/14-cosmos_int32.bin");
        if (runTest("14-cosmos_int32", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "14-cosmos: " << e.what() << std::endl; failed++; }

    // Dataset 15: polylog
    try {
        auto data = loadSOSDBinaryFile<uint64_t>("data/sosd/15-polylog_10M_uint64.bin");
        if (runTest("15-polylog_10M_uint64", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "15-polylog: " << e.what() << std::endl; failed++; }

    // Dataset 16: exp
    try {
        auto data = loadSOSDBinaryFile<uint64_t>("data/sosd/16-exp_200M_uint64.bin");
        if (runTest("16-exp_200M_uint64", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "16-exp: " << e.what() << std::endl; failed++; }

    // Dataset 17: poly
    try {
        auto data = loadSOSDBinaryFile<uint64_t>("data/sosd/17-poly_200M_uint64.bin");
        if (runTest("17-poly_200M_uint64", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "17-poly: " << e.what() << std::endl; failed++; }

    // Dataset 18: site
    try {
        auto data = loadSOSDBinaryFile<uint32_t>("data/sosd/18-site_250k_uint32.bin");
        if (runTest("18-site_250k_uint32", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "18-site: " << e.what() << std::endl; failed++; }

    // Dataset 19: weight
    try {
        auto data = loadSOSDBinaryFile<uint32_t>("data/sosd/19-weight_25k_uint32.bin");
        if (runTest("19-weight_25k_uint32", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "19-weight: " << e.what() << std::endl; failed++; }

    // Dataset 20: adult
    try {
        auto data = loadSOSDBinaryFile<uint32_t>("data/sosd/20-adult_30k_uint32.bin");
        if (runTest("20-adult_30k_uint32", data, max_elem)) passed++; else failed++;
    } catch (const std::exception& e) { std::cout << "20-adult: " << e.what() << std::endl; failed++; }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " PASSED, " << failed << " FAILED" << std::endl;
    std::cout << "========================================" << std::endl;

    return failed == 0 ? 0 : 1;
}
