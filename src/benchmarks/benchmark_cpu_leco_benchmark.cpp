/**
 * LeCo CPU Implementation Benchmark Driver
 * Tests compression/decompression with various datasets
 */

#include "benchmark_cpu_leco.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cassert>
#include <cstring>

// ============================================================================
// Timer Utility
// ============================================================================

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

// ============================================================================
// Data Generation
// ============================================================================

template<typename T>
std::vector<T> generateLinearData(size_t count, T base, T slope) {
    std::vector<T> data(count);
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<T>(base + slope * i);
    }
    return data;
}

template<typename T>
std::vector<T> generateNormalData(size_t count, T mean, T stddev, unsigned seed = 42) {
    std::vector<T> data(count);
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> dist(static_cast<double>(mean),
                                           static_cast<double>(stddev));
    for (size_t i = 0; i < count; i++) {
        data[i] = static_cast<T>(std::max(0.0, dist(gen)));
    }
    std::sort(data.begin(), data.end());  // Sort for better compression
    return data;
}

template<typename T>
std::vector<T> generateUniformData(size_t count, T min_val, T max_val, unsigned seed = 42) {
    std::vector<T> data(count);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<T> dist(min_val, max_val);
    for (size_t i = 0; i < count; i++) {
        data[i] = dist(gen);
    }
    std::sort(data.begin(), data.end());  // Sort for better compression
    return data;
}

template<typename T>
std::vector<T> generateNoisyLinearData(size_t count, T base, T slope, T noise_range,
                                        unsigned seed = 42) {
    std::vector<T> data(count);
    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int64_t> noise_dist(-static_cast<int64_t>(noise_range),
                                                       static_cast<int64_t>(noise_range));
    for (size_t i = 0; i < count; i++) {
        T linear_val = static_cast<T>(base + slope * i);
        int64_t noise = noise_dist(gen);
        data[i] = static_cast<T>(static_cast<int64_t>(linear_val) + noise);
    }
    return data;
}

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

    // Read 8-byte header containing element count
    uint64_t num_elements = 0;
    file.read(reinterpret_cast<char*>(&num_elements), sizeof(uint64_t));

    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    file.close();

    return data;
}

// ============================================================================
// Verification
// ============================================================================

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
            std::cerr << "Mismatch at index " << i << ": "
                      << decoded[i] << " vs " << original[i] << std::endl;
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
            std::cerr << "Random access mismatch at index " << idx << ": "
                      << decoded << " vs " << original[idx] << std::endl;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Benchmark Function
// ============================================================================

template<typename T>
leco::LeCoStats runBenchmark(const std::string& name, const std::vector<T>& data,
                              const leco::LeCoConfig& config = leco::LeCoConfig(),
                              int warmup_runs = 3, int bench_runs = 10) {
    leco::LeCoStats stats = {};
    std::cout << "\n========================================" << std::endl;
    std::cout << "Dataset: " << name << std::endl;
    std::cout << "Elements: " << data.size() << std::endl;
    std::cout << "Size: " << (data.size() * sizeof(T) / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "========================================" << std::endl;

    Timer timer;

    // Warmup
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < warmup_runs; i++) {
        auto compressed = leco::lecoEncode(data, config);
        auto decoded = leco::lecoDecode(compressed);
        (void)decoded;
    }

    // Encoding benchmark
    std::cout << "Running " << bench_runs << " encode trials..." << std::endl;
    double total_encode_time = 0;
    leco::LeCoCompressedBlock<T> final_compressed;

    for (int i = 0; i < bench_runs; i++) {
        timer.start();
        auto compressed = leco::lecoEncode(data, config);
        double elapsed = timer.elapsedMs();
        total_encode_time += elapsed;
        if (i == bench_runs - 1) {
            final_compressed = std::move(compressed);
        }
        std::cout << "  Encode trial " << (i + 1) << ": " << elapsed << " ms" << std::endl;
    }

    // Decoding benchmark
    std::cout << "Running " << bench_runs << " decode trials..." << std::endl;
    double total_decode_time = 0;

    for (int i = 0; i < bench_runs; i++) {
        timer.start();
        auto decoded = leco::lecoDecode(final_compressed);
        double elapsed = timer.elapsedMs();
        total_decode_time += elapsed;
        std::cout << "  Decode trial " << (i + 1) << ": " << elapsed << " ms" << std::endl;
    }

    // Verification
    std::cout << "Verifying..." << std::endl;
    bool roundtrip_ok = verifyRoundtrip(data, final_compressed);
    bool random_ok = verifyRandomAccess(data, final_compressed);

    // Calculate stats
    stats.compression_ratio = final_compressed.compression_ratio;
    stats.encode_time_ms = total_encode_time / bench_runs;
    stats.decode_time_ms = total_decode_time / bench_runs;
    stats.original_bytes = final_compressed.original_bytes;
    stats.compressed_bytes = final_compressed.compressed_bytes;
    stats.num_segments = final_compressed.num_segments;
    stats.avg_segment_length = static_cast<double>(data.size()) / final_compressed.num_segments;

    // Calculate average delta bits
    double total_bits = 0;
    int count = 0;
    for (const auto& seg : final_compressed.segments) {
        if (seg.delta_bits < 200) {  // Exclude special markers
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
    std::cout << "Avg Encode Time: " << stats.encode_time_ms << " ms" << std::endl;
    std::cout << "Avg Decode Time: " << stats.decode_time_ms << " ms" << std::endl;

    double data_size_gb = data.size() * sizeof(T) / 1e9;
    std::cout << "Encode Throughput: " << (data_size_gb / (stats.encode_time_ms / 1000.0))
              << " GB/s" << std::endl;
    std::cout << "Decode Throughput: " << (data_size_gb / (stats.decode_time_ms / 1000.0))
              << " GB/s" << std::endl;

    std::cout << "Roundtrip Verification: " << (roundtrip_ok ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Random Access Verification: " << (random_ok ? "PASSED" : "FAILED") << std::endl;

    return stats;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "==================================================" << std::endl;
    std::cout << "  LeCo CPU Implementation Benchmark" << std::endl;
    std::cout << "  Cost-Optimal Encoder (SIGMOD'24)" << std::endl;
    std::cout << "==================================================" << std::endl;

    leco::LeCoConfig config;
    config.overhead = 13;
    config.cost_decline_threshold = 0.0001;

    // ========================================
    // Synthetic Data Tests
    // ========================================

    std::cout << "\n\n*** SYNTHETIC DATA TESTS ***\n" << std::endl;

    // Small test
    {
        std::cout << "\n=== Small Linear Test (1000 elements) ===" << std::endl;
        auto data = generateLinearData<uint32_t>(1000, 100, 5);
        runBenchmark("linear_1K", data, config, 1, 3);
    }

    // Linear data - 1M uint32
    {
        auto data = generateLinearData<uint32_t>(1000000, 1000000, 3);
        runBenchmark("linear_1M_uint32", data, config);
    }

    // Noisy linear data - 1M uint32
    {
        auto data = generateNoisyLinearData<uint32_t>(1000000, 1000000, 3, 100);
        runBenchmark("noisy_linear_1M_uint32", data, config);
    }

    // Normal distribution - 1M uint32
    {
        auto data = generateNormalData<uint32_t>(1000000, 1000000, 10000);
        runBenchmark("normal_1M_uint32", data, config);
    }

    // Uniform distribution - 1M uint32
    {
        auto data = generateUniformData<uint32_t>(1000000, 0, 10000000);
        runBenchmark("uniform_1M_uint32", data, config);
    }

    // Linear data - 1M uint64
    {
        auto data = generateLinearData<uint64_t>(1000000, 1000000000ULL, 7);
        runBenchmark("linear_1M_uint64", data, config);
    }

    // ========================================
    // Larger Scale Tests
    // ========================================

    std::cout << "\n\n*** LARGER SCALE TESTS ***\n" << std::endl;

    // Linear data - 10M uint32
    {
        auto data = generateLinearData<uint32_t>(10000000, 1000000, 3);
        runBenchmark("linear_10M_uint32", data, config);
    }

    // Noisy linear data - 10M uint32
    {
        auto data = generateNoisyLinearData<uint32_t>(10000000, 1000000, 3, 100);
        runBenchmark("noisy_linear_10M_uint32", data, config);
    }

    // ========================================
    // SOSD Dataset Tests (1-20)
    // ========================================

    std::cout << "\n\n*** SOSD DATASET TESTS (1-20) ***\n" << std::endl;

    struct DatasetInfo {
        std::string name;
        std::string path;
        bool is_uint64;
        bool has_header;  // SOSD files have 8-byte header
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
                stats = runBenchmark(info.name, data, config, 1, 3);
            } else {
                std::vector<uint32_t> data;
                if (info.has_header) {
                    data = loadSOSDBinaryFile<uint32_t>(info.path);
                } else {
                    data = loadBinaryFile<uint32_t>(info.path);
                }
                stats = runBenchmark(info.name, data, config, 1, 3);
            }
            all_results.push_back({info.name, stats});
            passed++;
        } catch (const std::exception& e) {
            std::cout << "FAILED " << info.name << ": " << e.what() << std::endl;
            failed++;
        }
    }

    // Print summary table
    std::cout << "\n\n==================================================" << std::endl;
    std::cout << "                 SUMMARY TABLE" << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << std::left << std::setw(30) << "Dataset"
              << std::right << std::setw(12) << "Elements"
              << std::setw(12) << "Ratio"
              << std::setw(12) << "Segments"
              << std::setw(12) << "AvgBits"
              << std::setw(14) << "EncodeMS"
              << std::setw(14) << "DecodeMS"
              << std::endl;
    std::cout << std::string(106, '-') << std::endl;

    for (const auto& [name, stats] : all_results) {
        std::cout << std::left << std::setw(30) << name
                  << std::right << std::setw(12) << (stats.original_bytes / (stats.compression_ratio > 0 ? (stats.original_bytes / stats.compression_ratio / 4) : 1))
                  << std::setw(12) << std::fixed << std::setprecision(2) << stats.compression_ratio << "x"
                  << std::setw(12) << stats.num_segments
                  << std::setw(12) << std::fixed << std::setprecision(2) << stats.avg_delta_bits
                  << std::setw(14) << std::fixed << std::setprecision(2) << stats.encode_time_ms
                  << std::setw(14) << std::fixed << std::setprecision(2) << stats.decode_time_ms
                  << std::endl;
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << "Tests: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "==================================================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
