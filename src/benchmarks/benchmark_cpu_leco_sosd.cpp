/**
 * LeCo CPU Implementation - SOSD Dataset Benchmark Only
 * Tests compression/decompression on SOSD datasets 1-20
 */

#include "benchmark_cpu_leco.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>

// Timer Utility
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

// Load binary file (no header)
template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t num_elements = file_size / sizeof(T);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    return data;
}

// Load SOSD binary file (8-byte header with element count)
template<typename T>
std::vector<T> loadSOSDBinaryFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    uint64_t num_elements = 0;
    file.read(reinterpret_cast<char*>(&num_elements), sizeof(uint64_t));
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    file.close();
    return data;
}

// Verification
template<typename T>
bool verifyRoundtrip(const std::vector<T>& original,
                     const leco::LeCoCompressedBlock<T>& compressed) {
    std::vector<T> decoded = leco::lecoDecode(compressed);
    if (decoded.size() != original.size()) {
        std::cerr << "Size mismatch: " << decoded.size() << " vs " << original.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < original.size(); i++) {
        if (decoded[i] != original[i]) {
            std::cerr << "Mismatch at index " << i << ": " << decoded[i] << " vs " << original[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
bool verifyRandomAccess(const std::vector<T>& original,
                        const leco::LeCoCompressedBlock<T>& compressed,
                        int num_samples = 1000) {
    std::mt19937_64 gen(42);
    std::uniform_int_distribution<size_t> dist(0, original.size() - 1);
    for (int i = 0; i < num_samples; i++) {
        size_t idx = dist(gen);
        T decoded = leco::lecoDecodeAt(compressed, static_cast<int32_t>(idx));
        if (decoded != original[idx]) {
            std::cerr << "Random access mismatch at index " << idx << std::endl;
            return false;
        }
    }
    return true;
}

// Benchmark Function (reduced trials for large datasets)
template<typename T>
leco::LeCoStats runBenchmark(const std::string& name, const std::vector<T>& data,
                              const leco::LeCoConfig& config) {
    leco::LeCoStats stats = {};
    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << name << std::endl;
    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Size: " << (data.size() * sizeof(T) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    Timer timer;

    // Single warmup
    std::cout << "Warming up..." << std::endl;
    auto warmup_compressed = leco::lecoEncode(data, config);

    // Single encode trial
    std::cout << "Encoding..." << std::endl;
    timer.start();
    auto compressed = leco::lecoEncode(data, config);
    double encode_time = timer.elapsedMs();
    std::cout << "  Encode time: " << encode_time << " ms" << std::endl;

    // Single decode trial
    std::cout << "Decoding..." << std::endl;
    timer.start();
    auto decoded = leco::lecoDecode(compressed);
    double decode_time = timer.elapsedMs();
    std::cout << "  Decode time: " << decode_time << " ms" << std::endl;

    // Verification
    std::cout << "Verifying..." << std::endl;
    bool roundtrip_ok = true;
    if (decoded.size() != data.size()) {
        roundtrip_ok = false;
        std::cerr << "Size mismatch!" << std::endl;
    } else {
        for (size_t i = 0; i < data.size(); i++) {
            if (decoded[i] != data[i]) {
                roundtrip_ok = false;
                std::cerr << "Mismatch at index " << i << ": " << decoded[i] << " vs " << data[i] << std::endl;
                break;
            }
        }
    }

    bool random_ok = verifyRandomAccess(data, compressed);

    // Calculate stats
    stats.compression_ratio = compressed.compression_ratio;
    stats.encode_time_ms = encode_time;
    stats.decode_time_ms = decode_time;
    stats.original_bytes = compressed.original_bytes;
    stats.compressed_bytes = compressed.compressed_bytes;
    stats.num_segments = compressed.num_segments;
    stats.avg_segment_length = static_cast<double>(data.size()) / compressed.num_segments;

    double total_bits = 0;
    int count = 0;
    for (const auto& seg : compressed.segments) {
        if (seg.delta_bits < 200) {
            total_bits += seg.delta_bits;
            count++;
        }
    }
    stats.avg_delta_bits = count > 0 ? total_bits / count : 0;

    // Print results
    std::cout << "\n--- Results ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Compression Ratio: " << stats.compression_ratio << "x" << std::endl;
    std::cout << "Compressed Size: " << (stats.compressed_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "Num Segments: " << stats.num_segments << std::endl;
    std::cout << "Avg Segment Length: " << stats.avg_segment_length << std::endl;
    std::cout << "Avg Delta Bits: " << stats.avg_delta_bits << std::endl;
    std::cout << "Encode Time: " << stats.encode_time_ms << " ms" << std::endl;
    std::cout << "Decode Time: " << stats.decode_time_ms << " ms" << std::endl;

    double data_size_gb = data.size() * sizeof(T) / 1e9;
    std::cout << "Encode Throughput: " << (data_size_gb / (stats.encode_time_ms / 1000.0)) << " GB/s" << std::endl;
    std::cout << "Decode Throughput: " << (data_size_gb / (stats.decode_time_ms / 1000.0)) << " GB/s" << std::endl;

    std::cout << "Roundtrip Verification: " << (roundtrip_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Random Access Verification: " << (random_ok ? "PASSED" : "FAILED") << std::endl;

    if (!roundtrip_ok || !random_ok) {
        throw std::runtime_error("Verification failed for " + name);
    }

    return stats;
}

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  LeCo CPU - SOSD Dataset Benchmark (1-20)" << std::endl;
    std::cout << "==================================================" << std::endl;

    // Optional: limit data size for faster testing
    // Set to 0 for no limit (full datasets)
    size_t max_elements = 0;  // No limit by default
    if (argc > 1) {
        max_elements = std::stoull(argv[1]);
        std::cout << "Limiting to " << max_elements << " elements per dataset" << std::endl;
    }

    leco::LeCoConfig config;
    config.overhead = 13;
    config.cost_decline_threshold = 0.0001;

    struct DatasetInfo {
        std::string name;
        std::string path;
        bool is_uint64;
        bool has_header;
    };

    std::vector<DatasetInfo> sosd_datasets = {
        {"01-linear_200M_uint64", "data/sosd/1-linear_200M_uint64.bin", true, false},
        {"02-normal_200M_uint64", "data/sosd/2-normal_200M_uint64.bin", true, false},
        {"03-poisson_87M_uint64", "data/sosd/3-poisson_87M_uint64.bin", true, true},
        {"04-ml_uint64", "data/sosd/4-ml_uint64.bin", true, true},
        {"05-books_200M_uint32", "data/sosd/5-books_200M_uint32.bin", false, true},
        {"06-fb_200M_uint64", "data/sosd/6-fb_200M_uint64.bin", true, true},
        {"07-wiki_200M_uint64", "data/sosd/7-wiki_200M_uint64.bin", true, true},
        {"08-osm_cellids_800M_uint64", "data/sosd/8-osm_cellids_800M_uint64.bin", true, true},
        {"09-movieid_uint32", "data/sosd/9-movieid_uint32.bin", false, true},
        {"10-house_price_uint64", "data/sosd/10-house_price_uint64.bin", true, true},
        {"11-planet_uint64", "data/sosd/11-planet_uint64.bin", true, true},
        {"12-libio_uint64", "data/sosd/12-libio.bin", true, true},
        {"13-medicare_uint64", "data/sosd/13-medicare.bin", true, true},
        {"14-cosmos_int32", "data/sosd/14-cosmos_int32.bin", false, false},
        {"15-polylog_10M_uint64", "data/sosd/15-polylog_10M_uint64.bin", true, true},
        {"16-exp_200M_uint64", "data/sosd/16-exp_200M_uint64.bin", true, true},
        {"17-poly_200M_uint64", "data/sosd/17-poly_200M_uint64.bin", true, true},
        {"18-site_250k_uint32", "data/sosd/18-site_250k_uint32.bin", false, true},
        {"19-weight_25k_uint32", "data/sosd/19-weight_25k_uint32.bin", false, true},
        {"20-adult_30k_uint32", "data/sosd/20-adult_30k_uint32.bin", false, true},
    };

    int passed = 0, failed = 0;
    std::vector<std::pair<std::string, leco::LeCoStats>> all_results;

    for (const auto& info : sosd_datasets) {
        try {
            leco::LeCoStats stats;
            if (info.is_uint64) {
                std::vector<uint64_t> data;
                if (info.has_header) {
                    data = loadSOSDBinaryFile<uint64_t>(info.path);
                } else {
                    data = loadBinaryFile<uint64_t>(info.path);
                }
                if (max_elements > 0 && data.size() > max_elements) {
                    data.resize(max_elements);
                }
                stats = runBenchmark(info.name, data, config);
            } else {
                std::vector<uint32_t> data;
                if (info.has_header) {
                    data = loadSOSDBinaryFile<uint32_t>(info.path);
                } else {
                    data = loadBinaryFile<uint32_t>(info.path);
                }
                if (max_elements > 0 && data.size() > max_elements) {
                    data.resize(max_elements);
                }
                stats = runBenchmark(info.name, data, config);
            }
            all_results.push_back({info.name, stats});
            passed++;
        } catch (const std::exception& e) {
            std::cout << "\n*** FAILED " << info.name << ": " << e.what() << " ***\n" << std::endl;
            failed++;
        }
    }

    // Print summary table
    std::cout << "\n\n" << std::string(120, '=') << std::endl;
    std::cout << "                                    SUMMARY TABLE" << std::endl;
    std::cout << std::string(120, '=') << std::endl;
    std::cout << std::left << std::setw(28) << "Dataset"
              << std::right << std::setw(14) << "Elements"
              << std::setw(10) << "Ratio"
              << std::setw(10) << "Segments"
              << std::setw(10) << "AvgBits"
              << std::setw(14) << "EncodeMS"
              << std::setw(14) << "DecodeMS"
              << std::setw(12) << "EncGB/s"
              << std::setw(12) << "DecGB/s"
              << std::endl;
    std::cout << std::string(120, '-') << std::endl;

    for (const auto& [name, stats] : all_results) {
        int64_t elements = stats.original_bytes / 8;  // Approximate
        if (name.find("uint32") != std::string::npos || name.find("int32") != std::string::npos) {
            elements = stats.original_bytes / 4;
        }
        double enc_throughput = (stats.original_bytes / 1e9) / (stats.encode_time_ms / 1000.0);
        double dec_throughput = (stats.original_bytes / 1e9) / (stats.decode_time_ms / 1000.0);

        std::cout << std::left << std::setw(28) << name
                  << std::right << std::setw(14) << elements
                  << std::setw(9) << std::fixed << std::setprecision(2) << stats.compression_ratio << "x"
                  << std::setw(10) << stats.num_segments
                  << std::setw(10) << std::fixed << std::setprecision(2) << stats.avg_delta_bits
                  << std::setw(14) << std::fixed << std::setprecision(1) << stats.encode_time_ms
                  << std::setw(14) << std::fixed << std::setprecision(1) << stats.decode_time_ms
                  << std::setw(12) << std::fixed << std::setprecision(3) << enc_throughput
                  << std::setw(12) << std::fixed << std::setprecision(3) << dec_throughput
                  << std::endl;
    }

    std::cout << std::string(120, '=') << std::endl;
    std::cout << "\nTests: " << passed << " PASSED, " << failed << " FAILED" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    return (failed == 0) ? 0 : 1;
}
