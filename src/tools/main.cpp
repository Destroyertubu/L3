/**
 * L3 Unified Test Program
 *
 * Comprehensive testing tool for L3 compression system.
 * Supports configurable partitioning strategies, encoder/decoder types,
 * all SOSD datasets (1-20), and string datasets (21-23).
 *
 * Usage:
 *   ./main [options] dataset_id
 *
 * Options:
 *   --partition-strategy [FIXED|COST_OPTIMAL|VARIANCE_ADAPTIVE]
 *   --partition-size N       (default: 4096)
 *   --encoder [STANDARD|Vertical|OPTIMIZED|GPU|GPU_ZEROSYNC]
 *   --decoder [STANDARD|Vertical|OPTIMIZED]
 *   --max-delta-bits N       (default: 32)
 *   --random-access-samples N (default: 10000)
 *   --all                   Run all datasets (1-23)
 *   --output-csv FILE       Output results to CSV file
 *
 * Datasets:
 *   1-20: Numeric SOSD datasets
 *   21-23: String datasets (email, hex, words)
 *
 * Examples:
 *   ./main --all --partition-strategy COST_OPTIMAL
 *   ./main 1 --encoder Vertical --decoder Vertical
 *   ./main 5 --partition-size 2048 --output-csv results.csv
 *   ./main 21  # Test email string dataset
 *
 * Author: Auto-generated test harness
 * Date: 2025-12-08 (updated with string support)
 */

#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <unordered_map>
#include <random>
#include <cuda_runtime.h>

// L3 headers
#include "L3_codec.hpp"
#include "L3_format.hpp"
#include "L3_random_access.hpp"
#include "L3.h"

// Vertical headers
#include "L3_Vertical_format.hpp"
#include "L3_Vertical_api.hpp"

// String compression headers
#include "L3_string_codec.hpp"
#include "L3_string_format.hpp"

// SOSD loader
#include "sosd_loader.h"

// ============================================================================
// Configuration Structures
// ============================================================================

enum class EncoderType {
    STANDARD,
    Vertical,
    OPTIMIZED,
    GPU,              // GPU-only pipeline with dynamic allocation
    GPU_ZEROSYNC      // GPU-only pipeline with zero mid-pipeline synchronization
};

enum class DecoderType {
    STANDARD,           // Standard decoder (decompression_kernels.cu)
    Vertical,          // Vertical with branchless+interleaved (decoder_Vertical_opt.cu)
    OPTIMIZED,          // Warp-optimized (decoder_warp_opt.cu)
    SPECIALIZED,        // Bit-width specialized templates (decoder_specialized.cu)
    PHASE2,             // Phase2 with cp.async pipeline (decompression_kernels_phase2.cu)
    PHASE2_BUCKET,      // Bucket-based dispatch (decompression_kernels_phase2_bucket.cu)
    KERNELS_OPT         // 8/16-bit optimized (decompression_kernels_opt.cu)
};

enum class ModelSelectionStrategy {
    ADAPTIVE,           // Adaptive: each partition selects optimal model (LINEAR/POLY2/POLY3/FOR)
    LINEAR_ONLY,        // Fixed: only use linear model (y = a + bx)
    POLY2_ONLY,         // Fixed: only use quadratic polynomial (y = a + bx + cx^2)
    POLY3_ONLY,         // Fixed: only use cubic polynomial (y = a + bx + cx^2 + dx^3)
    FOR_ONLY            // Fixed: only use FOR+BitPacking
};

// ============================================================================
// Encoder-Decoder Pairing Helper Functions
// ============================================================================

inline bool isL3Encoder(EncoderType type) {
    return type == EncoderType::STANDARD || type == EncoderType::OPTIMIZED;
}

inline bool isVerticalEncoder(EncoderType type) {
    return type == EncoderType::Vertical || type == EncoderType::GPU || type == EncoderType::GPU_ZEROSYNC;
}

inline bool isL3Decoder(DecoderType type) {
    return type != DecoderType::Vertical;
}

struct TestConfig {
    PartitioningStrategy partition_strategy = PartitioningStrategy::COST_OPTIMAL;
    int partition_size = 4096;
    EncoderType encoder_type = EncoderType::STANDARD;
    DecoderType decoder_type = DecoderType::STANDARD;
    ModelSelectionStrategy model_strategy = ModelSelectionStrategy::ADAPTIVE;  // Model selection strategy
    bool enable_polynomial_models = false;  // Enable POLY2/POLY3 in ADAPTIVE mode
    int max_delta_bits = 32;
    int random_access_samples = 10000;
    bool run_all_datasets = false;
    bool compare_decoders = false;  // Run all decoders and compare performance
    bool compare_Vertical = false; // Compare Vertical sequential vs interleaved
    DecompressMode Vertical_mode = DecompressMode::AUTO;  // Vertical decompression mode
    std::string dataset_id = "";
    std::string output_csv = "";
    std::string data_dir = "/root/autodl-tmp/test/data/sosd";
};

// ============================================================================
// Dataset Information
// ============================================================================

struct DatasetInfo {
    std::string filename;
    std::string name;
    bool is_uint64;
    size_t expected_elements;
    bool is_string;  // New: flag for string datasets

    DatasetInfo() : filename(""), name(""), is_uint64(false), expected_elements(0), is_string(false) {}
    DatasetInfo(std::string f, std::string n, bool u64, size_t e, bool str = false)
        : filename(f), name(n), is_uint64(u64), expected_elements(e), is_string(str) {}
};

const std::unordered_map<std::string, DatasetInfo> DATASETS = {
    {"1", {"1-linear_200M_uint64.bin", "linear", true, 200000000}},
    {"2", {"2-normal_200M_uint64.bin", "normal", true, 200000000}},
    {"3", {"3-poisson_87M_uint64.bin", "poisson", true, 87000000}},
    {"4", {"4-ml_uint64.bin", "ml", true, 0}},  // Size determined at runtime
    {"5", {"5-books_200M_uint32.bin", "books", false, 200000000}},
    {"6", {"6-fb_200M_uint64.bin", "fb", true, 200000000}},
    {"7", {"7-wiki_200M_uint64.bin", "wiki", true, 200000000}},
    {"8", {"8-osm_cellids_800M_uint64.bin", "osm", true, 800000000}},
    {"9", {"9-movieid_uint32.bin", "movieid", false, 0}},  // Size determined at runtime
    {"10", {"10-house_price_uint64.bin", "house_price", true, 0}},
    {"11", {"11-planet_uint64.bin", "planet", true, 200000000}},
    {"12", {"12-libio.bin", "libio", true, 200000000}},
    {"13", {"13-medicare.bin", "medicare", true, 0}},
    {"14", {"14-cosmos_int32.bin", "cosmos", false, 0}},
    {"15", {"15-polylog_10M_uint64.bin", "polylog", true, 10000000}},
    {"16", {"16-exp_200M_uint64.bin", "exp", true, 200000000}},
    {"17", {"17-poly_200M_uint64.bin", "poly", true, 200000000}},
    {"18", {"18-site_250k_uint32.bin", "site", false, 250000}},
    {"19", {"19-weight_25k_uint32.bin", "weight", false, 25000}},
    {"20", {"20-adult_30k_uint32.bin", "adult", false, 30000}},
    // String datasets (21-23)
    {"21", {"strings/email_leco_30k.txt", "email", false, 30000, true}},
    {"22", {"strings/hex.txt", "hex", false, 0, true}},
    {"23", {"strings/words.txt", "words", false, 0, true}}
};

// ============================================================================
// Utility Functions
// ============================================================================

// CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Print usage information
void printUsage(const char* program_name) {
    std::cout << "L3 Unified Test Program\n\n";
    std::cout << "Usage: " << program_name << " [options] dataset_id\n\n";
    std::cout << "Options:\n";
    std::cout << "  --partition-strategy [FIXED|COST_OPTIMAL|VARIANCE_ADAPTIVE]\n";
    std::cout << "  --partition-size N       (default: 4096)\n";
    std::cout << "  --encoder [STANDARD|Vertical|OPTIMIZED|GPU|GPU_ZEROSYNC]\n";
    std::cout << "                           (GPU: GPU-only pipeline, GPU_ZEROSYNC: zero mid-sync)\n";
    std::cout << "  --decoder [STANDARD|Vertical|OPTIMIZED|SPECIALIZED|PHASE2|PHASE2_BUCKET|KERNELS_OPT]\n";
    std::cout << "  --model-selection [ADAPTIVE|LINEAR|POLY2|POLY3|FOR]\n";
    std::cout << "                           (default: ADAPTIVE, controls model selection per partition)\n";
    std::cout << "  --polynomial             Enable POLY2/POLY3 model selection in ADAPTIVE mode\n";
    std::cout << "  --Vertical-mode [SEQUENTIAL|INTERLEAVED|BRANCHLESS|AUTO]\n";
    std::cout << "                           (default: AUTO, only for Vertical decoder)\n";
    std::cout << "  --max-delta-bits N       (default: 32)\n";
    std::cout << "  --random-access-samples N (default: 10000)\n";
    std::cout << "  --all                   Run all datasets (1-23, including strings)\n";
    std::cout << "  --compare-decoders      Compare all decoder types on each dataset\n";
    std::cout << "  --compare-Vertical     Compare Vertical sequential vs interleaved\n";
    std::cout << "  --output-csv FILE       Output results to CSV file\n";
    std::cout << "  --help                  Show this help message\n\n";
    std::cout << "Datasets:\n";
    std::cout << "  all: Test all 23 datasets (1-20 numeric, 21-23 string)\n";
    std::cout << "  1-20: Numeric datasets (SOSD)\n";
    std::cout << "  21-23: String datasets (email, hex, words)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --all --partition-strategy COST_OPTIMAL\n";
    std::cout << "  " << program_name << " 1 --encoder Vertical --decoder Vertical\n";
    std::cout << "  " << program_name << " 5 --partition-size 2048 --output-csv results.csv\n";
    std::cout << "  " << program_name << " 2 --encoder Vertical --compare-Vertical\n";
    std::cout << "  " << program_name << " 21  # Test email string dataset\n";
}

// Parse command line arguments
TestConfig parseArguments(int argc, char* argv[]) {
    TestConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            exit(0);
        } else if (arg == "--all") {
            config.run_all_datasets = true;
        } else if (arg == "--compare-decoders") {
            config.compare_decoders = true;
        } else if (arg == "--compare-Vertical") {
            config.compare_Vertical = true;
        } else if (arg == "--polynomial") {
            config.enable_polynomial_models = true;
        } else if (arg == "--model-selection") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --model-selection requires an argument\n";
                exit(1);
            }
            std::string model = argv[++i];
            if (model == "ADAPTIVE") {
                config.model_strategy = ModelSelectionStrategy::ADAPTIVE;
            } else if (model == "LINEAR") {
                config.model_strategy = ModelSelectionStrategy::LINEAR_ONLY;
            } else if (model == "POLY2") {
                config.model_strategy = ModelSelectionStrategy::POLY2_ONLY;
            } else if (model == "POLY3") {
                config.model_strategy = ModelSelectionStrategy::POLY3_ONLY;
            } else if (model == "FOR") {
                config.model_strategy = ModelSelectionStrategy::FOR_ONLY;
            } else {
                std::cerr << "Error: Invalid model selection: " << model << "\n";
                exit(1);
            }
        } else if (arg == "--Vertical-mode") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --Vertical-mode requires an argument\n";
                exit(1);
            }
            std::string mode = argv[++i];
            if (mode == "SEQUENTIAL") {
                config.Vertical_mode = DecompressMode::SEQUENTIAL;
            } else if (mode == "INTERLEAVED") {
                config.Vertical_mode = DecompressMode::INTERLEAVED;
            } else if (mode == "BRANCHLESS") {
                config.Vertical_mode = DecompressMode::BRANCHLESS;
            } else if (mode == "AUTO") {
                config.Vertical_mode = DecompressMode::AUTO;
            } else {
                std::cerr << "Error: Invalid Vertical mode: " << mode << "\n";
                exit(1);
            }
        } else if (arg == "--partition-strategy") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --partition-strategy requires an argument\n";
                exit(1);
            }
            std::string strategy = argv[++i];
            if (strategy == "FIXED") {
                config.partition_strategy = PartitioningStrategy::FIXED;
            } else if (strategy == "COST_OPTIMAL") {
                config.partition_strategy = PartitioningStrategy::COST_OPTIMAL;
            } else if (strategy == "VARIANCE_ADAPTIVE") {
                config.partition_strategy = PartitioningStrategy::VARIANCE_ADAPTIVE;
            } else {
                std::cerr << "Error: Invalid partition strategy: " << strategy << "\n";
                exit(1);
            }
        } else if (arg == "--partition-size") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --partition-size requires an argument\n";
                exit(1);
            }
            config.partition_size = std::atoi(argv[++i]);
        } else if (arg == "--encoder") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --encoder requires an argument\n";
                exit(1);
            }
            std::string encoder = argv[++i];
            if (encoder == "STANDARD") {
                config.encoder_type = EncoderType::STANDARD;
            } else if (encoder == "Vertical") {
                config.encoder_type = EncoderType::Vertical;
            } else if (encoder == "OPTIMIZED") {
                config.encoder_type = EncoderType::OPTIMIZED;
            } else if (encoder == "GPU") {
                config.encoder_type = EncoderType::GPU;
            } else if (encoder == "GPU_ZEROSYNC") {
                config.encoder_type = EncoderType::GPU_ZEROSYNC;
            } else {
                std::cerr << "Error: Invalid encoder type: " << encoder << "\n";
                exit(1);
            }
        } else if (arg == "--decoder") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --decoder requires an argument\n";
                exit(1);
            }
            std::string decoder = argv[++i];
            if (decoder == "STANDARD") {
                config.decoder_type = DecoderType::STANDARD;
            } else if (decoder == "Vertical") {
                config.decoder_type = DecoderType::Vertical;
            } else if (decoder == "OPTIMIZED") {
                config.decoder_type = DecoderType::OPTIMIZED;
            } else if (decoder == "SPECIALIZED") {
                config.decoder_type = DecoderType::SPECIALIZED;
            } else if (decoder == "PHASE2") {
                config.decoder_type = DecoderType::PHASE2;
            } else if (decoder == "PHASE2_BUCKET") {
                config.decoder_type = DecoderType::PHASE2_BUCKET;
            } else if (decoder == "KERNELS_OPT") {
                config.decoder_type = DecoderType::KERNELS_OPT;
            } else {
                std::cerr << "Error: Invalid decoder type: " << decoder << "\n";
                exit(1);
            }
        } else if (arg == "--max-delta-bits") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --max-delta-bits requires an argument\n";
                exit(1);
            }
            config.max_delta_bits = std::atoi(argv[++i]);
        } else if (arg == "--random-access-samples") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --random-access-samples requires an argument\n";
                exit(1);
            }
            config.random_access_samples = std::atoi(argv[++i]);
        } else if (arg == "--output-csv") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --output-csv requires an argument\n";
                exit(1);
            }
            config.output_csv = argv[++i];
        } else if (arg.substr(0, 2) != "--") {
            // Dataset ID
            if (!config.dataset_id.empty()) {
                std::cerr << "Error: Multiple dataset IDs specified\n";
                exit(1);
            }
            config.dataset_id = arg;
        } else {
            std::cerr << "Error: Unknown option: " << arg << "\n";
            exit(1);
        }
    }

    // Validation
    if (config.run_all_datasets && !config.dataset_id.empty()) {
        std::cerr << "Error: Cannot specify both --all and a dataset ID\n";
        exit(1);
    }

    if (!config.run_all_datasets && config.dataset_id.empty()) {
        std::cerr << "Error: Must specify either --all or a dataset ID\n";
        exit(1);
    }

    if (!config.run_all_datasets && DATASETS.find(config.dataset_id) == DATASETS.end()) {
        std::cerr << "Error: Invalid dataset ID: " << config.dataset_id << "\n";
        exit(1);
    }

    // Validate encoder-decoder pairing
    if (isVerticalEncoder(config.encoder_type) && isL3Decoder(config.decoder_type)) {
        std::cerr << "Error: Vertical encoders (Vertical/GPU/GPU_ZEROSYNC) must use Vertical decoder.\n";
        std::cerr << "       Use --decoder Vertical with --encoder "
                  << (config.encoder_type == EncoderType::Vertical ? "Vertical" :
                      config.encoder_type == EncoderType::GPU ? "GPU" : "GPU_ZEROSYNC") << "\n";
        exit(1);
    }
    if (isL3Encoder(config.encoder_type) && config.decoder_type == DecoderType::Vertical) {
        std::cerr << "Error: L3 encoders (STANDARD/OPTIMIZED) cannot use Vertical decoder.\n";
        std::cerr << "       Use --decoder STANDARD (or other L3 decoders) with --encoder "
                  << (config.encoder_type == EncoderType::STANDARD ? "STANDARD" : "OPTIMIZED") << "\n";
        exit(1);
    }

    return config;
}

// Load dataset
template<typename T>
std::vector<T> loadDataset(const DatasetInfo& info, const std::string& data_dir) {
    std::string filepath = data_dir + "/" + info.filename;
    std::vector<T> data;

    bool success;
    if (info.name == "movieid") {
        // Text file
        success = SOSDLoader::loadText<T>(filepath, data);
    } else {
        // Binary file
        success = SOSDLoader::loadBinary<T>(filepath, data);
    }

    if (!success) {
        throw std::runtime_error("Failed to load dataset: " + filepath);
    }

    return data;
}

// Load string dataset
std::vector<std::string> loadStringDataset(const DatasetInfo& info, const std::string& data_dir) {
    std::string filepath = data_dir + "/" + info.filename;
    std::vector<std::string> strings;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open string dataset: " + filepath);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            strings.push_back(line);
        }
    }

    file.close();
    return strings;
}

// Test string compression and decompression
bool testStringCompressionDecompression(
    const std::vector<std::string>& strings,
    const TestConfig& config,
    StringCompressionStats& comp_stats,
    StringDecompressionStats& decomp_stats)
{
    try {
        // Configure string compression
        StringCompressionConfig str_config;
        str_config.mode = STRING_MODE_SUBSET_SHIFT;
        str_config.partition_size = config.partition_size;
        str_config.use_common_prefix = true;

        // Compress strings
        CompressedStringData* compressed = compressStrings(strings, str_config, &comp_stats);

        if (!compressed || compressed->num_partitions == 0) {
            std::cerr << "String compression failed\n";
            return false;
        }

        // Decompress strings
        std::vector<std::string> decompressed;
        int count = decompressStrings(compressed, str_config, decompressed, &decomp_stats);

        if (count != static_cast<int>(strings.size())) {
            std::cerr << "String decompression size mismatch: expected " << strings.size()
                      << ", got " << count << "\n";
            freeCompressedStringData(compressed);
            return false;
        }

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < strings.size(); ++i) {
            if (strings[i] != decompressed[i]) {
                if (correct) {  // Only print first error
                    std::cerr << "String verification failed at index " << i
                              << ": expected \"" << strings[i].substr(0, 20)
                              << "\", got \"" << decompressed[i].substr(0, 20) << "\"\n";
                }
                correct = false;
            }
        }

        // Copy stats
        comp_stats.compression_ratio = compressed->compression_ratio;

        freeCompressedStringData(compressed);
        return correct;

    } catch (const std::exception& e) {
        std::cerr << "String compression error: " << e.what() << "\n";
        return false;
    }
}

// Run string dataset test
void runStringDatasetTest(const DatasetInfo& info, const TestConfig& config, std::ofstream* csv_file) {
    std::cout << "\n========================================\n";
    std::cout << "Testing String Dataset: " << info.name << "\n";
    std::cout << "========================================\n";

    try {
        // Load string dataset
        std::cout << "Loading string dataset...\n";
        auto load_start = std::chrono::high_resolution_clock::now();
        std::vector<std::string> strings = loadStringDataset(info, config.data_dir);
        auto load_end = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_end - load_start).count() / 1000.0;

        std::cout << "Loaded " << strings.size() << " strings in " << load_time << " seconds\n";

        // Calculate total bytes
        size_t total_bytes = 0;
        for (const auto& s : strings) {
            total_bytes += s.size();
        }
        std::cout << "Total size: " << (total_bytes / 1024.0 / 1024.0) << " MB\n";

        // Test compression/decompression
        std::cout << "Testing string compression/decompression...\n";
        StringCompressionStats comp_stats = {};
        StringDecompressionStats decomp_stats = {};
        bool comp_success = testStringCompressionDecompression(strings, config, comp_stats, decomp_stats);

        if (comp_success) {
            std::cout << "✓ String Compression/Decompression: PASSED\n";
            std::cout << "  Compression Ratio: " << std::fixed << std::setprecision(2)
                      << comp_stats.compression_ratio << "x\n";
            std::cout << "  Compression Time: " << comp_stats.compression_time_ms << " ms\n";
            std::cout << "  Decompression Time: " << decomp_stats.decompression_time_ms << " ms\n";
        } else {
            std::cout << "✗ String Compression/Decompression: FAILED\n";
        }

        // Output to CSV if requested
        if (csv_file && csv_file->is_open()) {
            *csv_file << info.name << ","
                      << "string" << ","
                      << strings.size() << ","
                      << comp_stats.num_partitions << ","
                      << comp_stats.compression_ratio << ","
                      << comp_success << ","
                      << comp_stats.compression_time_ms << ","
                      << decomp_stats.decompression_throughput_gbps << ","
                      << "N/A" << ","  // random_access_success
                      << "N/A" << ","  // random_access_throughput
                      << "N/A" << ","  // cache_hits
                      << "N/A" << "\n"; // cache_misses
        }

    } catch (const std::exception& e) {
        std::cerr << "Error testing string dataset " << info.name << ": " << e.what() << "\n";
    }
}

// ============================================================================
// Decoder Dispatch Functions
// ============================================================================

/**
 * Helper function to compute average delta bits from compressed data
 */
template<typename T>
int computeAvgDeltaBits(CompressedDataL3<T>* compressed) {
    if (!compressed || compressed->num_partitions == 0) return 32;

    std::vector<int32_t> h_delta_bits(compressed->num_partitions);
    cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
               compressed->num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int64_t total = 0;
    for (int i = 0; i < compressed->num_partitions; i++) {
        total += h_delta_bits[i];
    }
    return static_cast<int>(total / compressed->num_partitions);
}

/**
 * Dispatch decompression to the selected decoder type
 *
 * @param compressed   Compressed data
 * @param output       Pre-allocated output buffer (host memory)
 * @param decoder_type Which decoder to use
 * @param stats        Optional stats to record timing
 * @return Number of decompressed elements
 */
template<typename T>
int dispatchDecoder(
    CompressedDataL3<T>* compressed,
    std::vector<T>& output,
    DecoderType decoder_type,
    DecompressionStats* stats)
{
    if (!compressed || compressed->num_partitions == 0) {
        return 0;
    }

    size_t n = output.size();

    // Allocate device output buffer
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(T)));

    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Compute average delta bits for decoders that need routing hints
    int avg_delta_bits = computeAvgDeltaBits<T>(compressed);

    // Start timing
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    // Dispatch to the appropriate decoder (L3 decoders only)
    switch (decoder_type) {
        case DecoderType::STANDARD:
            // Standard decoder (decompression_kernels.cu)
            launchDecompressOptimized<T>(compressed, d_output, 0);
            break;

        case DecoderType::Vertical:
            // Vertical decoder cannot be used with L3-encoded data
            // Validation should catch this at argument parsing time
            std::cerr << "Error: Vertical decoder cannot be used with L3 encoders.\n";
            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            return 0;

        case DecoderType::OPTIMIZED:
            // Warp-optimized decoder (decoder_warp_opt.cu)
            launchDecompressWarpOpt<T>(compressed, d_output, 0);
            break;

        case DecoderType::SPECIALIZED:
            // Bit-width specialized decoder (decoder_specialized.cu)
            launchDecompressSpecialized<T>(compressed, d_output, avg_delta_bits, 0);
            break;

        case DecoderType::PHASE2:
            // Phase2 decoder with cp.async pipeline (decompression_kernels_phase2.cu)
            decompressL3_Phase2<T>(
                compressed->d_start_indices,
                compressed->d_end_indices,
                compressed->d_model_types,
                compressed->d_model_params,
                compressed->d_delta_bits,
                compressed->d_delta_array_bit_offsets,
                compressed->delta_array,
                compressed->num_partitions,
                static_cast<int>(n),
                d_output,
                avg_delta_bits,
                false  // use_persistent
            );
            break;

        case DecoderType::PHASE2_BUCKET: {
            // Phase2 bucket-based decoder (decompression_kernels_phase2_bucket.cu)
            // Convert int32 delta_bits to uint8
            std::vector<int32_t> h_delta_bits(compressed->num_partitions);
            cudaMemcpy(h_delta_bits.data(), compressed->d_delta_bits,
                       compressed->num_partitions * sizeof(int32_t), cudaMemcpyDeviceToHost);

            std::vector<uint8_t> h_delta_bits_u8(compressed->num_partitions);
            for (int i = 0; i < compressed->num_partitions; i++) {
                h_delta_bits_u8[i] = static_cast<uint8_t>(h_delta_bits[i]);
            }

            uint8_t* d_delta_bits_u8;
            CUDA_CHECK(cudaMalloc(&d_delta_bits_u8, compressed->num_partitions * sizeof(uint8_t)));
            CUDA_CHECK(cudaMemcpy(d_delta_bits_u8, h_delta_bits_u8.data(),
                                  compressed->num_partitions * sizeof(uint8_t), cudaMemcpyHostToDevice));

            decompressL3_Phase2_Bucket<T>(
                compressed->d_start_indices,
                compressed->d_end_indices,
                compressed->d_model_types,
                compressed->d_model_params,
                d_delta_bits_u8,
                compressed->d_delta_array_bit_offsets,
                compressed->delta_array,
                compressed->num_partitions,
                static_cast<int>(n),
                d_output
            );

            CUDA_CHECK(cudaFree(d_delta_bits_u8));
            break;
        }

        case DecoderType::KERNELS_OPT:
            // 8/16-bit optimized decoder (decompression_kernels_opt.cu)
            decompressL3_Optimized<T>(
                compressed->d_start_indices,
                compressed->d_end_indices,
                compressed->d_model_types,
                compressed->d_model_params,
                compressed->d_delta_bits,
                compressed->d_delta_array_bit_offsets,
                compressed->delta_array,
                compressed->num_partitions,
                static_cast<int>(n),
                d_output,
                avg_delta_bits
            );
            break;
    }

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    // Record stats if requested
    if (stats) {
        stats->decompression_time_ms = elapsed_ms;
        stats->decompression_throughput_gbps = (n * sizeof(T) / 1e9) / (elapsed_ms / 1000.0);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<int>(n);
}

/**
 * Helper function to get decoder type name
 */
const char* getDecoderTypeName(DecoderType type) {
    switch (type) {
        case DecoderType::STANDARD: return "STANDARD";
        case DecoderType::Vertical: return "Vertical";
        case DecoderType::OPTIMIZED: return "OPTIMIZED";
        case DecoderType::SPECIALIZED: return "SPECIALIZED";
        case DecoderType::PHASE2: return "PHASE2";
        case DecoderType::PHASE2_BUCKET: return "PHASE2_BUCKET";
        case DecoderType::KERNELS_OPT: return "KERNELS_OPT";
        default: return "UNKNOWN";
    }
}

/**
 * Dispatch decompression for Vertical-encoded data (Vertical decoder only)
 *
 * This function handles CompressedDataVertical from GPU/GPU_ZEROSYNC/Vertical encoders.
 * Only Vertical decoder is supported - validation is done at argument parsing time.
 *
 * @param compressed     Vertical compressed data
 * @param output         Pre-allocated output buffer (host memory)
 * @param Vertical_mode Decompression mode for Vertical decoder
 * @param stats          Optional stats to record timing
 * @return Number of decompressed elements
 */
template<typename T>
int dispatchDecoderVertical(
    const CompressedDataVertical<T>& compressed,
    std::vector<T>& output,
    DecompressMode Vertical_mode,
    DecompressionStats* stats)
{
    if (compressed.num_partitions == 0) {
        return 0;
    }

    size_t n = output.size();

    // Allocate device output buffer
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(T)));

    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Start timing
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    // Use Vertical decoder directly
    Vertical_decoder::decompressAll<T>(compressed, d_output, Vertical_mode, 0);

    // Stop timing
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(output.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    // Record stats if requested
    if (stats) {
        stats->decompression_time_ms = elapsed_ms;
        stats->decompression_throughput_gbps = (n * sizeof(T) / 1e9) / (elapsed_ms / 1000.0);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return static_cast<int>(n);
}

/**
 * Compare all decoder types on a single compressed dataset
 *
 * @param compressed   Pre-compressed data
 * @param original     Original data for verification
 * @param dataset_name Name of dataset for output
 */
template<typename T>
void compareDecoders(
    CompressedDataL3<T>* compressed,
    const std::vector<T>& original,
    const std::string& dataset_name)
{
    const size_t n = original.size();
    const double data_size_gb = n * sizeof(T) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Decoder Performance Comparison\n";
    std::cout << "Dataset: " << dataset_name << "\n";
    std::cout << "Elements: " << n << "\n";
    std::cout << "Partitions: " << compressed->num_partitions << "\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(18) << "Decoder"
              << std::right << std::setw(12) << "Time (ms)"
              << std::setw(14) << "Throughput"
              << std::setw(10) << "Correct"
              << "\n";
    std::cout << std::string(54, '-') << "\n";

    // List of L3 decoder types to test (Vertical excluded - requires Vertical encoder)
    std::vector<DecoderType> decoders = {
        DecoderType::STANDARD,
        DecoderType::OPTIMIZED,
        DecoderType::SPECIALIZED,
        DecoderType::PHASE2,
        DecoderType::PHASE2_BUCKET,
        DecoderType::KERNELS_OPT
    };

    // Best results tracking
    double best_throughput = 0;
    DecoderType best_decoder = DecoderType::STANDARD;

    for (DecoderType decoder_type : decoders) {
        std::vector<T> decompressed(n);
        DecompressionStats stats = {};

        // Warmup run
        dispatchDecoder<T>(compressed, decompressed, decoder_type, nullptr);

        // Timed runs (take median of 3)
        std::vector<double> times;
        for (int run = 0; run < 3; run++) {
            std::fill(decompressed.begin(), decompressed.end(), T(0));
            dispatchDecoder<T>(compressed, decompressed, decoder_type, &stats);
            times.push_back(stats.decompression_time_ms);
        }
        std::sort(times.begin(), times.end());
        double median_time = times[1];  // Median of 3
        double throughput = data_size_gb / (median_time / 1000.0);

        // Verify correctness
        bool correct = true;
        for (size_t i = 0; i < n; i++) {
            if (decompressed[i] != original[i]) {
                correct = false;
                break;
            }
        }

        // Track best
        if (correct && throughput > best_throughput) {
            best_throughput = throughput;
            best_decoder = decoder_type;
        }

        std::cout << std::left << std::setw(18) << getDecoderTypeName(decoder_type)
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << median_time
                  << std::setprecision(1)
                  << std::setw(12) << throughput << " GB/s"
                  << std::setw(10) << (correct ? "PASS" : "FAIL")
                  << "\n";
    }

    std::cout << std::string(54, '-') << "\n";
    std::cout << "Best: " << getDecoderTypeName(best_decoder)
              << " (" << std::fixed << std::setprecision(1) << best_throughput << " GB/s)\n";
    std::cout << "========================================\n";
}

/**
 * Compare Vertical sequential vs interleaved decompression
 *
 * This function:
 * 1. Compresses data using Vertical encoder (creates both sequential and interleaved format)
 * 2. Decompresses using SEQUENTIAL mode (branchless per-partition)
 * 3. Decompresses using INTERLEAVED mode (mini-vector optimized)
 * 4. Compares throughput and verifies correctness for both
 *
 * @param data          Original data
 * @param dataset_name  Name of dataset for output
 * @param partition_size Partition size to use
 * @param model_strategy Model selection strategy
 * @param encoder_type  Encoder type to use (Vertical, GPU, GPU_ZEROSYNC)
 * @param partition_strategy Partitioning strategy (FIXED, COST_OPTIMAL, VARIANCE_ADAPTIVE)
 */
template<typename T>
void compareVertical(
    const std::vector<T>& data,
    const std::string& dataset_name,
    int partition_size = 4096,
    ModelSelectionStrategy model_strategy = ModelSelectionStrategy::ADAPTIVE,
    EncoderType encoder_type = EncoderType::Vertical,
    PartitioningStrategy partition_strategy = PartitioningStrategy::FIXED)
{
    const size_t n = data.size();
    const double data_size_gb = n * sizeof(T) / (1024.0 * 1024.0 * 1024.0);

    // Get model selection name for display
    const char* model_name = "ADAPTIVE";
    switch (model_strategy) {
        case ModelSelectionStrategy::ADAPTIVE: model_name = "ADAPTIVE"; break;
        case ModelSelectionStrategy::LINEAR_ONLY: model_name = "LINEAR"; break;
        case ModelSelectionStrategy::POLY2_ONLY: model_name = "POLY2"; break;
        case ModelSelectionStrategy::POLY3_ONLY: model_name = "POLY3"; break;
        case ModelSelectionStrategy::FOR_ONLY: model_name = "FOR"; break;
    }

    // Get encoder name for display
    const char* encoder_name = "Vertical";
    switch (encoder_type) {
        case EncoderType::STANDARD: encoder_name = "STANDARD"; break;
        case EncoderType::Vertical: encoder_name = "Vertical"; break;
        case EncoderType::OPTIMIZED: encoder_name = "OPTIMIZED"; break;
        case EncoderType::GPU: encoder_name = "GPU"; break;
        case EncoderType::GPU_ZEROSYNC: encoder_name = "GPU_ZEROSYNC"; break;
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Vertical Sequential vs Interleaved Comparison\n";
    std::cout << "Dataset: " << dataset_name << "\n";
    std::cout << "Elements: " << n << "\n";
    std::cout << "Partition size: " << partition_size << "\n";
    std::cout << "Model selection: " << model_name << "\n";
    std::cout << "Encoder: " << encoder_name << "\n";
    std::cout << "Mini-vector size: " << MINI_VECTOR_SIZE << "\n";
    std::cout << "========================================\n";

    // Configure Vertical encoder (v3.0: Interleaved-only format)
    VerticalConfig fl_config;
    fl_config.partition_size_hint = partition_size;
    fl_config.enable_interleaved = true;
    fl_config.enable_branchless_unpack = true;

    // Configure model selection strategy
    switch (model_strategy) {
        case ModelSelectionStrategy::ADAPTIVE:
            fl_config.enable_adaptive_selection = true;
            break;
        case ModelSelectionStrategy::LINEAR_ONLY:
            fl_config.enable_adaptive_selection = false;
            fl_config.fixed_model_type = MODEL_LINEAR;
            break;
        case ModelSelectionStrategy::POLY2_ONLY:
            fl_config.enable_adaptive_selection = false;
            fl_config.fixed_model_type = MODEL_POLYNOMIAL2;
            break;
        case ModelSelectionStrategy::POLY3_ONLY:
            fl_config.enable_adaptive_selection = false;
            fl_config.fixed_model_type = MODEL_POLYNOMIAL3;
            break;
        case ModelSelectionStrategy::FOR_ONLY:
            fl_config.enable_adaptive_selection = false;
            fl_config.fixed_model_type = MODEL_FOR_BITPACK;
            break;
    }

    // Configure partitioning strategy
    fl_config.partitioning_strategy = partition_strategy;

    // Create partitions (only needed for CPU/Vertical encoder)
    std::vector<PartitionInfo> partitions;
    int num_partitions = 0;

    if (encoder_type == EncoderType::Vertical || encoder_type == EncoderType::STANDARD ||
        encoder_type == EncoderType::OPTIMIZED) {
        partitions = Vertical_encoder::createFixedPartitions<T>(static_cast<int>(n), partition_size);
        num_partitions = partitions.size();
    } else {
        // GPU encoders compute partitions internally
        num_partitions = (n + partition_size - 1) / partition_size;
    }

    std::cout << "Number of partitions: " << num_partitions << "\n\n";

    // Compress using selected encoder
    std::cout << "--- Compression (" << encoder_name << ") ---\n";

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CompressedDataVertical<T> compressed;
    auto compress_start = std::chrono::high_resolution_clock::now();

    switch (encoder_type) {
        case EncoderType::Vertical:
        case EncoderType::STANDARD:
        case EncoderType::OPTIMIZED:
            compressed = Vertical_encoder::encodeVertical<T>(data, partitions, fl_config);
            break;
        case EncoderType::GPU:
            compressed = Vertical_encoder::encodeVerticalGPU<T>(data, partition_size, fl_config);
            break;
        case EncoderType::GPU_ZEROSYNC:
            compressed = Vertical_encoder::encodeVerticalGPU_ZeroSync<T>(data, partition_size, fl_config);
            break;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto compress_end = std::chrono::high_resolution_clock::now();
    double compress_time_ms = std::chrono::duration<double, std::milli>(
        compress_end - compress_start).count();

    // v3.0: Only interleaved format, no sequential format
    double int_size_mb = static_cast<double>(compressed.interleaved_delta_words * sizeof(uint32_t))
                         / (1024.0 * 1024.0);
    double original_size_mb = static_cast<double>(n * sizeof(T)) / (1024.0 * 1024.0);
    double compression_ratio = original_size_mb / int_size_mb;

    std::cout << "Compression ratio: " << std::fixed << std::setprecision(2)
              << compression_ratio << "x\n";
    std::cout << "Compressed size (interleaved): " << int_size_mb << " MB\n";
    std::cout << "Compression time: " << compress_time_ms << " ms\n";
    std::cout << "Total partitions: " << num_partitions << "\n\n";

    // Allocate device output buffer
    T* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(T)));

    const int WARMUP_RUNS = 3;
    const int BENCHMARK_RUNS = 5;

    // ========== BRANCHLESS (Sequential) Decompression ==========
    std::cout << "--- BRANCHLESS (Sequential) Decompression ---\n";

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; i++) {
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::BRANCHLESS);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Benchmark
    std::vector<float> seq_times(BENCHMARK_RUNS);
    for (int run = 0; run < BENCHMARK_RUNS; run++) {
        CUDA_CHECK(cudaEventRecord(start));
        Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::BRANCHLESS);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&seq_times[run], start, stop));
    }

    // Verify correctness
    std::vector<T> decoded_seq(n);
    CUDA_CHECK(cudaMemcpy(decoded_seq.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

    bool seq_correct = true;
    for (size_t i = 0; i < n; i++) {
        if (data[i] != decoded_seq[i]) {
            seq_correct = false;
            break;
        }
    }

    std::sort(seq_times.begin(), seq_times.end());
    float seq_median = seq_times[BENCHMARK_RUNS / 2];
    double seq_throughput = data_size_gb / (seq_median / 1000.0);

    std::cout << "Time (median): " << std::fixed << std::setprecision(3) << seq_median << " ms\n";
    std::cout << "Throughput: " << std::setprecision(1) << seq_throughput << " GB/s\n";
    std::cout << "Correctness: " << (seq_correct ? "PASS" : "FAIL") << "\n\n";

    // ========== INTERLEAVED Decompression ==========
    std::cout << "--- INTERLEAVED (Mini-Vector) Decompression ---\n";

    if (compressed.d_interleaved_deltas == nullptr || compressed.total_interleaved_partitions == 0) {
        std::cout << "No interleaved data available (partitions too small or disabled)\n\n";
    } else {
        // Clear output
        CUDA_CHECK(cudaMemset(d_output, 0, n * sizeof(T)));

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; i++) {
            Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::INTERLEAVED);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Benchmark
        std::vector<float> int_times(BENCHMARK_RUNS);
        for (int run = 0; run < BENCHMARK_RUNS; run++) {
            CUDA_CHECK(cudaEventRecord(start));
            Vertical_decoder::decompressAll<T>(compressed, d_output, DecompressMode::INTERLEAVED);
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&int_times[run], start, stop));
        }

        // Verify correctness
        std::vector<T> decoded_int(n);
        CUDA_CHECK(cudaMemcpy(decoded_int.data(), d_output, n * sizeof(T), cudaMemcpyDeviceToHost));

        bool int_correct = true;
        for (size_t i = 0; i < n; i++) {
            if (data[i] != decoded_int[i]) {
                int_correct = false;
                break;
            }
        }

        std::sort(int_times.begin(), int_times.end());
        float int_median = int_times[BENCHMARK_RUNS / 2];
        double int_throughput = data_size_gb / (int_median / 1000.0);

        std::cout << "Time (median): " << std::fixed << std::setprecision(3) << int_median << " ms\n";
        std::cout << "Throughput: " << std::setprecision(1) << int_throughput << " GB/s\n";
        std::cout << "Correctness: " << (int_correct ? "PASS" : "FAIL") << "\n\n";

        // ========== Summary ==========
        std::cout << "--- Summary ---\n";
        double speedup = seq_median / int_median;
        std::cout << "BRANCHLESS (Sequential): " << std::setprecision(1) << seq_throughput
                  << " GB/s (" << (seq_correct ? "PASS" : "FAIL") << ")\n";
        std::cout << "INTERLEAVED (Mini-Vector): " << int_throughput
                  << " GB/s (" << (int_correct ? "PASS" : "FAIL") << ")\n";
        std::cout << "Speedup (Interleaved/Sequential): " << std::setprecision(2)
                  << speedup << "x\n";

        if (speedup > 1.0) {
            std::cout << "Result: INTERLEAVED is " << std::setprecision(0)
                      << ((speedup - 1) * 100) << "% FASTER\n";
        } else if (speedup < 1.0) {
            std::cout << "Result: BRANCHLESS is " << std::setprecision(0)
                      << ((1.0/speedup - 1) * 100) << "% FASTER\n";
        } else {
            std::cout << "Result: Both modes perform equally\n";
        }
    }

    std::cout << "========================================\n";

    // Cleanup
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    Vertical_encoder::freeCompressedData(compressed);
}

// ============================================================================
// Test Functions
// ============================================================================

// Test compression and decompression
template<typename T>
bool testCompressionDecompression(const std::vector<T>& data,
                                  const TestConfig& config,
                                  CompressionStats& comp_stats,
                                  DecompressionStats& decomp_stats) {
    try {
        // Check if using GPU encoder (GPU or GPU_ZEROSYNC)
        bool use_gpu_encoder = (config.encoder_type == EncoderType::GPU ||
                                config.encoder_type == EncoderType::GPU_ZEROSYNC);

        if (use_gpu_encoder) {
            // ========== GPU Encoder Path ==========
            // Configure Vertical encoder (v3.0: Interleaved-only format)
            VerticalConfig fl_config;
            fl_config.partition_size_hint = config.partition_size;
            fl_config.enable_interleaved = true;
            fl_config.enable_branchless_unpack = true;

            // Configure model selection strategy
            switch (config.model_strategy) {
                case ModelSelectionStrategy::ADAPTIVE:
                    fl_config.enable_adaptive_selection = true;
                    break;
                case ModelSelectionStrategy::LINEAR_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_LINEAR;
                    break;
                case ModelSelectionStrategy::POLY2_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_POLYNOMIAL2;
                    break;
                case ModelSelectionStrategy::POLY3_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_POLYNOMIAL3;
                    break;
                case ModelSelectionStrategy::FOR_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_FOR_BITPACK;
                    break;
            }

            // Configure partitioning strategy for GPU encoder
            fl_config.partitioning_strategy = config.partition_strategy;

            // Compress using GPU encoder
            auto start_time = std::chrono::high_resolution_clock::now();
            CompressedDataVertical<T> compressed;

            if (config.encoder_type == EncoderType::GPU) {
                compressed = Vertical_encoder::encodeVerticalGPU<T>(data, config.partition_size, fl_config);
            } else {
                compressed = Vertical_encoder::encodeVerticalGPU_ZeroSync<T>(data, config.partition_size, fl_config);
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            // Use kernel timing from encoder (excludes H2D transfer and memory allocation)
            comp_stats.compression_time_ms = compressed.kernel_time_ms;

            if (compressed.num_partitions == 0) {
                std::cerr << "GPU Compression failed\n";
                return false;
            }

            // Calculate compression stats
            comp_stats.num_partitions = compressed.num_partitions;
            double original_size = data.size() * sizeof(T);
            // Calculate full compressed size including partition metadata and delta array
            // v3.0: Only interleaved format
            int64_t metadata_size = compressed.num_partitions * (
                sizeof(int32_t) +     // start_indices
                sizeof(int32_t) +     // end_indices
                sizeof(int32_t) +     // model_types
                sizeof(double) * 4 +  // model_params (4 doubles per partition)
                sizeof(int32_t) +     // delta_bits
                sizeof(int32_t) +     // num_mini_vectors
                sizeof(int32_t) +     // tail_sizes
                sizeof(int64_t)       // interleaved_offsets
            );
            int64_t delta_array_size = compressed.interleaved_delta_words * sizeof(uint32_t);
            double compressed_size = metadata_size + delta_array_size;
            comp_stats.compression_ratio = original_size / compressed_size;

            // Decompress using Vertical decoder
            std::vector<T> decompressed(data.size());
            int decompressed_count = dispatchDecoderVertical<T>(
                compressed, decompressed, config.Vertical_mode, &decomp_stats);

            if (decompressed_count != static_cast<int>(data.size())) {
                std::cerr << "Decompression size mismatch\n";
                Vertical_encoder::freeCompressedData(compressed);
                return false;
            }

            // Verify correctness
            bool correct = true;
            for (size_t i = 0; i < data.size(); ++i) {
                if (data[i] != decompressed[i]) {
                    if (correct) {  // Only print first error
                        std::cerr << "Verification failed at index " << i
                                  << ": expected " << data[i] << ", got " << decompressed[i] << "\n";
                    }
                    correct = false;
                }
            }

            // Free compressed data
            Vertical_encoder::freeCompressedData(compressed);

            return correct;
        } else {
            // ========== L3 Encoder Path (STANDARD, Vertical, OPTIMIZED) ==========
            // Create L3 config based on test config
            L3Config L3_config;
            L3_config.partition_size = config.partition_size;
            L3_config.max_delta_bits = config.max_delta_bits;
            L3_config.partitioning_strategy = config.partition_strategy;

            // Configure based on partitioning strategy
            switch (config.partition_strategy) {
                case PartitioningStrategy::FIXED:
                    L3_config.use_adaptive_partitioning = false;
                    break;
                case PartitioningStrategy::COST_OPTIMAL:
                    L3_config.cost_target_partition_size = config.partition_size;
                    L3_config.cost_enable_merging = true;
                    L3_config.enable_polynomial_models = config.enable_polynomial_models;
                    break;
                case PartitioningStrategy::VARIANCE_ADAPTIVE:
                    L3_config.use_adaptive_partitioning = true;
                    break;
            }

            // Apply encoder type configuration
            switch (config.encoder_type) {
                case EncoderType::STANDARD:
                    // Default configuration
                    break;
                case EncoderType::Vertical:
                    // Vertical uses same codec, but could set model_type if available
                    // Currently not implemented as separate encoder
                    break;
                case EncoderType::OPTIMIZED:
                    // Use high throughput settings
                    L3_config.cost_enable_merging = false;  // Skip merging for speed
                    break;
                default:
                    break;
            }

            // Compress data with full configuration
            auto start_time = std::chrono::high_resolution_clock::now();
            CompressedDataL3<T>* compressed = compressDataWithConfig(data, L3_config, &comp_stats);
            auto end_time = std::chrono::high_resolution_clock::now();
            comp_stats.compression_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                end_time - start_time).count() / 1000.0;

            if (!compressed) {
                std::cerr << "Compression failed\n";
                return false;
            }

            // Decompress data using selected decoder
            std::vector<T> decompressed(data.size());
            int decompressed_count = dispatchDecoder<T>(compressed, decompressed, config.decoder_type, &decomp_stats);

            if (decompressed_count != static_cast<int>(data.size())) {
                std::cerr << "Decompression size mismatch\n";
                freeCompressedData(compressed);
                return false;
            }

            // Verify correctness
            bool correct = true;
            for (size_t i = 0; i < data.size(); ++i) {
                if (data[i] != decompressed[i]) {
                    if (correct) {  // Only print first error
                        std::cerr << "Verification failed at index " << i
                                  << ": expected " << data[i] << ", got " << decompressed[i] << "\n";
                    }
                    correct = false;
                }
            }

            // Free compressed data
            freeCompressedData(compressed);

            return correct;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during compression/decompression: " << e.what() << "\n";
        return false;
    }
}

// Test random access
template<typename T>
bool testRandomAccess(const std::vector<T>& data,
                      const TestConfig& config,
                      RandomAccessStats& ra_stats) {
    try {
        // Check if using GPU encoder (Vertical format)
        bool use_Vertical = (config.encoder_type == EncoderType::GPU ||
                              config.encoder_type == EncoderType::GPU_ZEROSYNC);

        if (use_Vertical) {
            // ========== Vertical Random Access Path ==========
            // Configure Vertical encoder
            VerticalConfig fl_config;
            fl_config.partition_size_hint = config.partition_size;
            fl_config.enable_interleaved = true;
            fl_config.enable_branchless_unpack = true;

            // Configure model selection
            switch (config.model_strategy) {
                case ModelSelectionStrategy::ADAPTIVE:
                    fl_config.enable_adaptive_selection = true;
                    break;
                case ModelSelectionStrategy::LINEAR_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_LINEAR;
                    break;
                case ModelSelectionStrategy::POLY2_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_POLYNOMIAL2;
                    break;
                case ModelSelectionStrategy::POLY3_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_POLYNOMIAL3;
                    break;
                case ModelSelectionStrategy::FOR_ONLY:
                    fl_config.enable_adaptive_selection = false;
                    fl_config.fixed_model_type = MODEL_FOR_BITPACK;
                    break;
            }

            // Configure partitioning strategy
            fl_config.partitioning_strategy = config.partition_strategy;

            // Compress using GPU encoder
            CompressedDataVertical<T> compressed;
            if (config.encoder_type == EncoderType::GPU) {
                compressed = Vertical_encoder::encodeVerticalGPU<T>(data, config.partition_size, fl_config);
            } else {
                compressed = Vertical_encoder::encodeVerticalGPU_ZeroSync<T>(data, config.partition_size, fl_config);
            }

            if (compressed.num_partitions == 0) {
                std::cerr << "Vertical compression failed for random access test\n";
                return false;
            }

            // Generate random indices
            std::vector<int> indices(config.random_access_samples);
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            std::uniform_int_distribution<int> dist(0, data.size() - 1);
            for (int i = 0; i < config.random_access_samples; ++i) {
                indices[i] = dist(rng);
            }

            // Allocate device memory
            int* d_indices;
            T* d_output;
            CUDA_CHECK(cudaMalloc(&d_indices, indices.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_output, indices.size() * sizeof(T)));
            CUDA_CHECK(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // Create timing events
            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            // Time the Vertical random access
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));

            Vertical_decoder::decompressIndices<T>(compressed, d_indices, indices.size(), d_output, 0);

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            ra_stats.time_ms = elapsed_ms;

            // Copy results and verify
            std::vector<T> results(indices.size());
            CUDA_CHECK(cudaMemcpy(results.data(), d_output, results.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost));

            bool correct = true;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (results[i] != data[indices[i]]) {
                    if (correct) {  // Only print first error
                        std::cerr << "Vertical random access verification failed at sample " << i
                                  << ": expected " << data[indices[i]] << ", got " << results[i] << "\n";
                    }
                    correct = false;
                }
            }

            // Calculate throughput
            size_t bytes_accessed = indices.size() * sizeof(T);
            ra_stats.throughput_gbps = (bytes_accessed / (1024.0 * 1024.0 * 1024.0)) /
                                       (ra_stats.time_ms / 1000.0);
            ra_stats.cache_hits = 0;    // Not tracked for Vertical
            ra_stats.cache_misses = 0;  // Not tracked for Vertical

            // Cleanup
            CUDA_CHECK(cudaFree(d_indices));
            CUDA_CHECK(cudaFree(d_output));
            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
            Vertical_encoder::freeCompressedData(compressed);

            return correct;
        } else {
            // ========== L3 Random Access Path (optimized with branchless extraction) ==========
            // Compress data first
            CompressedDataL3<T>* compressed = compressData(data, config.partition_size);
            if (!compressed) {
                std::cerr << "Compression failed for random access test\n";
                return false;
            }

            // Generate random indices
            std::vector<int> indices(config.random_access_samples);
            std::mt19937 rng(42);  // Fixed seed for reproducibility
            std::uniform_int_distribution<int> dist(0, data.size() - 1);
            for (int i = 0; i < config.random_access_samples; ++i) {
                indices[i] = dist(rng);
            }

            // Allocate device memory for indices and output
            int* d_indices;
            T* d_output;
            CUDA_CHECK(cudaMalloc(&d_indices, indices.size() * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_output, indices.size() * sizeof(T)));

            CUDA_CHECK(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int),
                                  cudaMemcpyHostToDevice));

            // Perform random access using optimized kernel (Vertical-style direct pointers)
            // This matches Vertical kernel structure: direct pointers, branchless extraction
            cudaError_t err = randomAccessOptimized(compressed, d_indices, indices.size(),
                                                    d_output, &ra_stats);

            if (err != cudaSuccess) {
                std::cerr << "Random access failed: " << cudaGetErrorString(err) << "\n";
                CUDA_CHECK(cudaFree(d_indices));
                CUDA_CHECK(cudaFree(d_output));
                freeCompressedData(compressed);
                return false;
            }

            // ra_stats.time_ms is already set by randomAccessOptimized using CUDA events

            // Copy results back
            std::vector<T> results(indices.size());
            CUDA_CHECK(cudaMemcpy(results.data(), d_output, results.size() * sizeof(T),
                                  cudaMemcpyDeviceToHost));

            // Verify correctness
            bool correct = true;
            for (size_t i = 0; i < indices.size(); ++i) {
                if (results[i] != data[indices[i]]) {
                    if (correct) {  // Only print first error
                        std::cerr << "Random access verification failed at sample " << i
                                  << ": expected " << data[indices[i]] << ", got " << results[i] << "\n";
                    }
                    correct = false;
                }
            }

            // Throughput is already calculated by randomAccessBatch

            // Free memory
            CUDA_CHECK(cudaFree(d_indices));
            CUDA_CHECK(cudaFree(d_output));
            freeCompressedData(compressed);

            return correct;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception during random access: " << e.what() << "\n";
        return false;
    }
}

// Run complete test suite for a dataset
template<typename T>
void runDatasetTest(const DatasetInfo& info, const TestConfig& config,
                    std::ofstream* csv_file = nullptr) {
    std::cout << "\n========================================\n";
    std::cout << "Testing Dataset: " << info.name << " (" << info.filename << ")\n";
    std::cout << "========================================\n";

    try {
        // Load dataset
        std::cout << "Loading dataset...\n";
        auto load_start = std::chrono::high_resolution_clock::now();
        std::vector<T> data = loadDataset<T>(info, config.data_dir);
        auto load_end = std::chrono::high_resolution_clock::now();
        double load_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            load_end - load_start).count() / 1000.0;

        std::cout << "Loaded " << data.size() << " elements in " << load_time << " seconds\n";

        // Test compression/decompression
        std::cout << "Testing compression/decompression...\n";
        CompressionStats comp_stats = {};
        DecompressionStats decomp_stats = {};
        bool comp_success = testCompressionDecompression(data, config, comp_stats, decomp_stats);

        if (comp_success) {
            std::cout << "✓ Compression/Decompression: PASSED\n";
            std::cout << "  Compression Ratio: " << std::fixed << std::setprecision(2)
                      << comp_stats.compression_ratio << "x\n";
            std::cout << "  Compression Time: " << comp_stats.compression_time_ms << " ms\n";
            std::cout << "  Decompression Time: " << decomp_stats.decompression_time_ms << " ms\n";
            std::cout << "  Decompression Throughput: " << std::fixed << std::setprecision(1)
                      << decomp_stats.decompression_throughput_gbps << " GB/s\n";
        } else {
            std::cout << "✗ Compression/Decompression: FAILED\n";
        }

        // Compare all decoder types if requested
        if (config.compare_decoders) {
            std::cout << "Running decoder comparison...\n";

            // Create L3 config for compression
            L3Config L3_config;
            L3_config.partition_size = config.partition_size;
            L3_config.max_delta_bits = config.max_delta_bits;
            L3_config.partitioning_strategy = config.partition_strategy;
            L3_config.enable_polynomial_models = config.enable_polynomial_models;

            // Compress data
            CompressedDataL3<T>* compressed = compressDataWithConfig(data, L3_config, nullptr);
            if (compressed) {
                compareDecoders<T>(compressed, data, info.name);
                freeCompressedData(compressed);
            } else {
                std::cerr << "Failed to compress data for decoder comparison\n";
            }
        }

        // Compare Vertical sequential vs interleaved if requested
        if (config.compare_Vertical) {
            std::cout << "Running Vertical comparison (Sequential vs Interleaved)...\n";
            compareVertical<T>(data, info.name, config.partition_size, config.model_strategy, config.encoder_type, config.partition_strategy);
        }

        // Test random access
        std::cout << "Testing random access...\n";
        RandomAccessStats ra_stats = {};
        bool ra_success = testRandomAccess(data, config, ra_stats);

        if (ra_success) {
            std::cout << "✓ Random Access: PASSED\n";
            std::cout << "  Samples: " << config.random_access_samples << "\n";
            std::cout << "  Time: " << ra_stats.time_ms << " ms\n";
            std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                      << ra_stats.throughput_gbps << " GB/s\n";
            std::cout << "  Cache Hit Rate: " << std::fixed << std::setprecision(1)
                      << (static_cast<double>(ra_stats.cache_hits) /
                          (ra_stats.cache_hits + ra_stats.cache_misses) * 100.0) << "%\n";
        } else {
            std::cout << "✗ Random Access: FAILED\n";
        }

        // Output to CSV if requested
        if (csv_file && csv_file->is_open()) {
            *csv_file << info.name << ","
                      << (info.is_uint64 ? "uint64" : "uint32") << ","
                      << data.size() << ","
                      << comp_stats.num_partitions << ","
                      << comp_stats.compression_ratio << ","
                      << comp_success << ","
                      << comp_stats.compression_time_ms << ","
                      << decomp_stats.decompression_throughput_gbps << ","
                      << ra_success << ","
                      << ra_stats.throughput_gbps << ","
                      << ra_stats.cache_hits << ","
                      << ra_stats.cache_misses << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error testing dataset " << info.name << ": " << e.what() << "\n";

        if (csv_file && csv_file->is_open()) {
            *csv_file << info.name << ","
                      << (info.is_uint64 ? "uint64" : "uint32") << ","
                      << "0" << ","
                      << "ERROR" << ","
                      << "ERROR" << ","
                      << "false" << ","
                      << "ERROR" << ","
                      << "ERROR" << ","
                      << "false" << ","
                      << "ERROR" << ","
                      << "ERROR" << ","
                      << "ERROR" << "\n";
        }
    }
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    // Parse command line arguments
    TestConfig config = parseArguments(argc, argv);

    // Open CSV output file if requested
    std::ofstream csv_file;
    if (!config.output_csv.empty()) {
        csv_file.open(config.output_csv);
        if (!csv_file.is_open()) {
            std::cerr << "Error: Cannot open output CSV file: " << config.output_csv << "\n";
            return 1;
        }

        // Write CSV header
        csv_file << "dataset,data_type,num_elements,num_partitions,compression_ratio,"
                 << "compression_success,compression_time_ms,decompression_throughput_gbps,"
                 << "random_access_success,random_access_throughput_gbps,cache_hits,cache_misses\n";
    }

    std::cout << "L3 Unified Test Program\n";
    std::cout << "==========================\n";
    std::cout << "Partition Strategy: ";
    switch (config.partition_strategy) {
        case PartitioningStrategy::FIXED: std::cout << "FIXED"; break;
        case PartitioningStrategy::COST_OPTIMAL: std::cout << "COST_OPTIMAL"; break;
        case PartitioningStrategy::VARIANCE_ADAPTIVE: std::cout << "VARIANCE_ADAPTIVE"; break;
    }
    std::cout << "\n";
    std::cout << "Partition Size: " << config.partition_size << "\n";
    std::cout << "Encoder: ";
    switch (config.encoder_type) {
        case EncoderType::STANDARD: std::cout << "STANDARD"; break;
        case EncoderType::Vertical: std::cout << "Vertical"; break;
        case EncoderType::OPTIMIZED: std::cout << "OPTIMIZED"; break;
        case EncoderType::GPU: std::cout << "GPU"; break;
        case EncoderType::GPU_ZEROSYNC: std::cout << "GPU_ZEROSYNC"; break;
    }
    std::cout << "\n";
    std::cout << "Decoder: ";
    switch (config.decoder_type) {
        case DecoderType::STANDARD: std::cout << "STANDARD"; break;
        case DecoderType::Vertical: std::cout << "Vertical"; break;
        case DecoderType::OPTIMIZED: std::cout << "OPTIMIZED"; break;
        case DecoderType::SPECIALIZED: std::cout << "SPECIALIZED"; break;
        case DecoderType::PHASE2: std::cout << "PHASE2"; break;
        case DecoderType::PHASE2_BUCKET: std::cout << "PHASE2_BUCKET"; break;
        case DecoderType::KERNELS_OPT: std::cout << "KERNELS_OPT"; break;
    }
    std::cout << "\n";
    std::cout << "Model Selection: ";
    switch (config.model_strategy) {
        case ModelSelectionStrategy::ADAPTIVE: std::cout << "ADAPTIVE"; break;
        case ModelSelectionStrategy::LINEAR_ONLY: std::cout << "LINEAR"; break;
        case ModelSelectionStrategy::POLY2_ONLY: std::cout << "POLY2"; break;
        case ModelSelectionStrategy::POLY3_ONLY: std::cout << "POLY3"; break;
        case ModelSelectionStrategy::FOR_ONLY: std::cout << "FOR"; break;
    }
    std::cout << "\n";
    std::cout << "Max Delta Bits: " << config.max_delta_bits << "\n";
    std::cout << "Random Access Samples: " << config.random_access_samples << "\n";

    if (config.run_all_datasets) {
        std::cout << "Testing: ALL DATASETS (1-23, including strings)\n";
    } else {
        std::cout << "Testing: Dataset " << config.dataset_id << "\n";
    }

    if (!config.output_csv.empty()) {
        std::cout << "Output CSV: " << config.output_csv << "\n";
    }

    std::cout << "\n";

    // Run tests
    if (config.run_all_datasets) {
        // Test all datasets (1-23, including string datasets)
        for (int i = 1; i <= 23; ++i) {
            std::string dataset_id = std::to_string(i);
            auto it = DATASETS.find(dataset_id);
            if (it != DATASETS.end()) {
                const DatasetInfo& info = it->second;
                if (info.is_string) {
                    runStringDatasetTest(info, config, &csv_file);
                } else if (info.is_uint64) {
                    runDatasetTest<uint64_t>(info, config, &csv_file);
                } else {
                    runDatasetTest<uint32_t>(info, config, &csv_file);
                }
            }
        }
    } else {
        // Test single dataset
        auto it = DATASETS.find(config.dataset_id);
        if (it != DATASETS.end()) {
            const DatasetInfo& info = it->second;
            if (info.is_string) {
                runStringDatasetTest(info, config, &csv_file);
            } else if (info.is_uint64) {
                runDatasetTest<uint64_t>(info, config, &csv_file);
            } else {
                runDatasetTest<uint32_t>(info, config, &csv_file);
            }
        }
    }

    // Close CSV file
    if (csv_file.is_open()) {
        csv_file.close();
        std::cout << "\nResults saved to: " << config.output_csv << "\n";
    }

    std::cout << "\nTest program completed.\n";
    return 0;
}

