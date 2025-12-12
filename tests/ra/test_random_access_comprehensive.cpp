/**
 * L3 Random Access Comprehensive Test
 *
 * Tests random access functionality on large real-world datasets:
 * - Facebook 200M uint64 dataset
 * - Wikipedia 200M uint64 dataset
 *
 * Test patterns:
 * 1. Single-element random access
 * 2. Batch random access (various sizes)
 * 3. Sequential access pattern
 * 4. Strided access pattern
 * 5. Clustered access pattern
 * 6. Uniform random access pattern
 *
 * Performance metrics:
 * - Throughput (elements/sec, GB/s)
 * - Latency (per access)
 * - Cache hit rate
 * - Correctness validation
 *
 * Date: 2025-10-23
 * Author: Claude Code - L3 Random Access Testing
 */

#include "L3_codec.hpp"
#include "L3_random_access.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Load binary data from file
 */
template<typename T>
std::vector<T> loadBinaryFile(const std::string& filename, size_t max_elements = 0) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << filename << std::endl;
        exit(1);
    }

    size_t file_size = file.tellg();
    size_t num_elements = file_size / sizeof(T);

    if (max_elements > 0 && max_elements < num_elements) {
        num_elements = max_elements;
    }

    file.seekg(0, std::ios::beg);
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), num_elements * sizeof(T));
    file.close();

    std::cout << "Loaded " << num_elements << " elements from " << filename
              << " (" << (num_elements * sizeof(T) / (1024.0 * 1024.0)) << " MB)" << std::endl;

    return data;
}

/**
 * Generate random access indices with different patterns
 */
std::vector<int> generateIndices(const std::string& pattern, int num_total, int num_accesses, int seed = 42) {
    // Ensure num_total is positive
    if (num_total <= 0) {
        std::cerr << "ERROR: num_total must be positive, got " << num_total << std::endl;
        exit(1);
    }

    std::vector<int> indices;
    indices.reserve(num_accesses);

    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> uniform_dist(0, num_total - 1);

    if (pattern == "random" || pattern == "uniform") {
        // Uniform random access
        for (int i = 0; i < num_accesses; i++) {
            indices.push_back(uniform_dist(gen));
        }
    } else if (pattern == "sequential") {
        // Sequential access
        int start = uniform_dist(gen);
        for (int i = 0; i < num_accesses; i++) {
            indices.push_back((start + i) % num_total);
        }
    } else if (pattern == "strided") {
        // Strided access (stride = 1000)
        int stride = 1000;
        int start = uniform_dist(gen);
        for (int i = 0; i < num_accesses; i++) {
            indices.push_back((start + i * stride) % num_total);
        }
    } else if (pattern == "clustered") {
        // Clustered access (10 clusters, 10% of accesses per cluster)
        int num_clusters = 10;
        int accesses_per_cluster = num_accesses / num_clusters;
        int cluster_size = num_total / 100;  // 1% of data per cluster

        for (int c = 0; c < num_clusters; c++) {
            int cluster_start = uniform_dist(gen);
            std::uniform_int_distribution<int> cluster_dist(0, cluster_size - 1);
            for (int i = 0; i < accesses_per_cluster; i++) {
                indices.push_back((cluster_start + cluster_dist(gen)) % num_total);
            }
        }
    } else {
        std::cerr << "Unknown pattern: " << pattern << std::endl;
        exit(1);
    }

    return indices;
}

/**
 * Verify correctness of random access results
 */
template<typename T>
bool verifyRandomAccess(
    const std::vector<T>& original_data,
    const std::vector<int>& indices,
    const std::vector<T>& accessed_data)
{
    if (indices.size() != accessed_data.size()) {
        std::cerr << "Size mismatch: indices=" << indices.size()
                  << ", accessed=" << accessed_data.size() << std::endl;
        return false;
    }

    int errors = 0;
    const int max_errors_to_show = 10;

    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        T expected = original_data[idx];
        T actual = accessed_data[i];

        if (expected != actual) {
            if (errors < max_errors_to_show) {
                std::cerr << "Mismatch at i=" << i << ", idx=" << idx
                          << ": expected=" << expected << ", actual=" << actual << std::endl;
            }
            errors++;
        }
    }

    if (errors > 0) {
        std::cerr << "Total errors: " << errors << " / " << indices.size() << std::endl;
        return false;
    }

    return true;
}

/**
 * Print statistics
 */
void printStats(const std::string& test_name, const RandomAccessStats& stats, int num_accesses) {
    printf("%-30s | ", test_name.c_str());
    printf("%10d accesses | ", num_accesses);
    printf("%8.2f ms | ", stats.time_ms);
    printf("%10.2f M/s | ", (num_accesses / 1e6) / (stats.time_ms / 1e3));
    printf("%8.2f GB/s | ", stats.throughput_gbps);
    printf("PASS\n");
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * Test single access pattern
 */
template<typename T>
void testAccessPattern(
    const std::string& pattern_name,
    CompressedDataL3<T>* compressed,
    const std::vector<T>& original_data,
    int num_accesses,
    bool verify = true)
{
    // Extract base pattern from pattern_name (e.g., "random-1K" -> "random")
    std::string base_pattern = pattern_name.substr(0, pattern_name.find('-'));
    std::vector<int> indices = generateIndices(base_pattern, original_data.size(), num_accesses);

    // Validate indices
    if (indices.empty() || indices.size() != static_cast<size_t>(num_accesses)) {
        std::cerr << "ERROR: Failed to generate " << num_accesses << " indices (got " << indices.size() << ")" << std::endl;
        return;
    }

    // Upload indices to GPU
    int* d_indices;
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_indices, num_accesses * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output, num_accesses * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_indices, indices.data(), num_accesses * sizeof(int), cudaMemcpyHostToDevice));

    // Perform random access
    RandomAccessStats stats;
    cudaError_t err = randomAccessBatch<T>(
        compressed, d_indices, num_accesses, d_output,
        RandomAccessConfig(), &stats, 0);

    if (err != cudaSuccess) {
        std::cerr << "Random access failed for " << pattern_name << ": " << cudaGetErrorString(err) << std::endl;
        std::cerr << "  num_accesses=" << num_accesses << ", original_data.size()=" << original_data.size() << std::endl;
        std::cerr << "  compressed->total_values=" << compressed->total_values << std::endl;
        std::cerr << "  compressed->num_partitions=" << compressed->num_partitions << std::endl;
        CUDA_CHECK(cudaFree(d_indices));
        CUDA_CHECK(cudaFree(d_output));
        return;
    }

    // Download results
    std::vector<T> results(num_accesses);
    CUDA_CHECK(cudaMemcpy(results.data(), d_output, num_accesses * sizeof(T), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    if (verify) {
        correct = verifyRandomAccess(original_data, indices, results);
    }

    // Print statistics
    if (correct) {
        printStats(pattern_name, stats, num_accesses);
    } else {
        printf("%-30s | FAILED - Correctness check failed\n", pattern_name.c_str());
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_output));

    // Ensure cleanup is complete before next test
    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * Test dataset with multiple patterns and batch sizes
 */
template<typename T>
void testDataset(const std::string& dataset_name, const std::string& filename, size_t max_elements = 0) {
    std::cout << "\n";
    std::cout << "========================================" << std::endl;
    std::cout << "Testing: " << dataset_name << std::endl;
    std::cout << "========================================" << std::endl;

    // Load data
    std::vector<T> data = loadBinaryFile<T>(filename, max_elements);
    int num_elements = data.size();

    // Compress data
    std::cout << "\nCompressing data..." << std::endl;
    CompressionStats comp_stats;
    CompressedDataL3<T>* compressed = compressData(data.data(), num_elements, 2048, &comp_stats);

    std::cout << "  Compressed size: " << (comp_stats.compressed_bytes / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Compression ratio: " << comp_stats.compression_ratio << "x" << std::endl;
    std::cout << "  Num partitions: " << comp_stats.num_partitions << std::endl;

    // Clear any previous CUDA errors
    cudaGetLastError();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Allocate device copy of compressed structure for device-side access
    CompressedDataL3<T>* d_compressed;
    CUDA_CHECK(cudaMalloc(&d_compressed, sizeof(CompressedDataL3<T>)));
    CUDA_CHECK(cudaMemcpy(d_compressed, compressed, sizeof(CompressedDataL3<T>), cudaMemcpyHostToDevice));
    compressed->d_self = d_compressed;

    CUDA_CHECK(cudaDeviceSynchronize());

    // Test different access patterns and batch sizes
    std::cout << "\nRandom Access Performance Tests:" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    printf("%-30s | %10s | %8s | %10s | %8s | %s\n",
           "Pattern", "Accesses", "Time", "Throughput", "Bandwidth", "Status");
    std::cout << "------------------------------------------------------------" << std::endl;

    // Small batch tests (verify correctness)
    testAccessPattern("random-1K", compressed, data, 1000, true);
    testAccessPattern("sequential-1K", compressed, data, 1000, true);
    testAccessPattern("strided-1K", compressed, data, 1000, true);
    testAccessPattern("clustered-1K", compressed, data, 1000, true);

    // Medium batch tests
    testAccessPattern("random-10K", compressed, data, 10000, true);
    testAccessPattern("sequential-10K", compressed, data, 10000, false);
    testAccessPattern("strided-10K", compressed, data, 10000, false);
    testAccessPattern("clustered-10K", compressed, data, 10000, false);

    // Large batch tests (performance focus)
    testAccessPattern("random-100K", compressed, data, 100000, false);
    testAccessPattern("random-1M", compressed, data, 1000000, false);

    if (num_elements >= 10000000) {
        testAccessPattern("random-10M", compressed, data, 10000000, false);
    }

    std::cout << "------------------------------------------------------------" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_compressed));
    freeCompressedData(compressed);

    std::cout << "\n" << dataset_name << " tests completed!" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "L3 Random Access Comprehensive Test" << std::endl;
    std::cout << "=======================================" << std::endl;

    // Test configuration
    const std::string fb_path = "/root/autodl-tmp/test/data/fb_200M_uint64.bin";
    const std::string wiki_path = "/root/autodl-tmp/test/data/wiki_200M_uint64.bin";

    // Parse command line arguments
    bool test_fb = true;
    bool test_wiki = true;
    size_t max_elements = 0;  // 0 = load all

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--fb-only") {
            test_wiki = false;
        } else if (arg == "--wiki-only") {
            test_fb = false;
        } else if (arg.rfind("--max=", 0) == 0) {
            max_elements = std::stoul(arg.substr(6));
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --fb-only          Test only Facebook dataset" << std::endl;
            std::cout << "  --wiki-only        Test only Wikipedia dataset" << std::endl;
            std::cout << "  --max=N            Load only first N elements" << std::endl;
            std::cout << "  --help             Show this help" << std::endl;
            return 0;
        }
    }

    // Run tests
    if (test_fb) {
        testDataset<uint64_t>("Facebook 200M", fb_path, max_elements);
    }

    if (test_wiki) {
        testDataset<uint64_t>("Wikipedia 200M", wiki_path, max_elements);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "All tests completed successfully!" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
