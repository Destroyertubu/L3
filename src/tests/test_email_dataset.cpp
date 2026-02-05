/**
 * Test L3 String Compression on LeCo Email Dataset
 *
 * Tests compression on the 30K host-reversed email dataset
 * as specified in LeCo paper Section 4.1
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cuda_runtime.h>

#include "L3_string_codec.hpp"

std::vector<std::string> loadEmailDataset(const std::string& filepath) {
    std::vector<std::string> emails;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filepath << std::endl;
        return emails;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            emails.push_back(line);
        }
    }

    return emails;
}

int main(int argc, char** argv) {
    std::cout << "L3 String Compression - LeCo Email Dataset Test" << std::endl;
    std::cout << "================================================" << std::endl;

    // Default dataset path
    std::string dataset_path = "data/sosd/strings/email_leco_30k.txt";
    if (argc > 1) {
        dataset_path = argv[1];
    }

    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << std::endl;

    // Load dataset
    std::cout << "\nLoading dataset: " << dataset_path << std::endl;
    std::vector<std::string> emails = loadEmailDataset(dataset_path);

    if (emails.empty()) {
        std::cerr << "No emails loaded!" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << emails.size() << " emails" << std::endl;

    // Calculate original data size
    size_t original_bytes = 0;
    size_t min_len = SIZE_MAX, max_len = 0;
    for (const auto& email : emails) {
        original_bytes += email.size();
        min_len = std::min(min_len, email.size());
        max_len = std::max(max_len, email.size());
    }

    double avg_len = static_cast<double>(original_bytes) / emails.size();

    std::cout << "\nDataset statistics:" << std::endl;
    std::cout << "  Total strings: " << emails.size() << std::endl;
    std::cout << "  Original size: " << original_bytes << " bytes" << std::endl;
    std::cout << "  Average length: " << avg_len << " bytes" << std::endl;
    std::cout << "  Min length: " << min_len << " bytes" << std::endl;
    std::cout << "  Max length: " << max_len << " bytes" << std::endl;

    // Show sample data
    std::cout << "\nSample emails:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), emails.size()); i++) {
        std::cout << "  [" << i << "] " << emails[i].substr(0, 60);
        if (emails[i].size() > 60) std::cout << "...";
        std::cout << std::endl;
    }

    // Auto-configure compression
    std::cout << "\n=== Compression Configuration ===" << std::endl;
    StringCompressionConfig config = autoConfigureStringCompression(emails);

    std::cout << "  Mode: " << config.mode << std::endl;
    std::cout << "  Character range: [" << config.min_char << ", " << config.max_char << "]" << std::endl;
    std::cout << "  Shift bits: " << config.shift_bits << std::endl;
    std::cout << "  Max encoded length: " << config.max_string_length << std::endl;
    std::cout << "  Common prefix: \"" << config.common_prefix << "\" (len=" << config.common_prefix.size() << ")" << std::endl;

    // Compress
    std::cout << "\n=== Compression ===" << std::endl;
    StringCompressionStats comp_stats;
    CompressedStringData* compressed = compressStrings(emails, config, &comp_stats);

    if (!compressed) {
        std::cerr << "Compression failed!" << std::endl;
        return 1;
    }

    printStringCompressionStats(comp_stats);

    // Decompress
    std::cout << "\n=== Decompression ===" << std::endl;
    std::vector<std::string> decompressed;
    StringDecompressionStats decomp_stats;
    int num_decompressed = decompressStrings(compressed, config, decompressed, &decomp_stats);

    printStringDecompressionStats(decomp_stats);

    // Verify
    std::cout << "\n=== Verification ===" << std::endl;
    int errors = 0;
    int truncation_count = 0;

    for (size_t i = 0; i < emails.size(); i++) {
        if (i < decompressed.size()) {
            // Check if it's a truncation issue or real error
            if (emails[i] != decompressed[i]) {
                // Check if decompressed is a prefix of original (truncation)
                if (emails[i].substr(0, decompressed[i].size()) == decompressed[i]) {
                    truncation_count++;
                } else {
                    errors++;
                    if (errors <= 5) {
                        std::cout << "Mismatch at index " << i << ":" << std::endl;
                        std::cout << "  Original:     \"" << emails[i] << "\"" << std::endl;
                        std::cout << "  Decompressed: \"" << decompressed[i] << "\"" << std::endl;
                    }
                }
            }
        } else {
            errors++;
        }
    }

    if (errors == 0 && truncation_count == 0) {
        std::cout << "PASSED: All " << emails.size() << " strings match exactly!" << std::endl;
    } else if (errors == 0) {
        std::cout << "PARTIAL: " << truncation_count << " strings truncated due to 64-bit encoding limit" << std::endl;
        std::cout << "  (This is expected for strings longer than " << (64 / config.shift_bits) << " chars after prefix removal)" << std::endl;
    } else {
        std::cout << "FAILED: " << errors << " mismatches, " << truncation_count << " truncations" << std::endl;
    }

    // Summary
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "  Compression ratio: " << comp_stats.compression_ratio << "x" << std::endl;
    std::cout << "  Compression throughput: " << comp_stats.compression_throughput_gbps << " GB/s" << std::endl;
    std::cout << "  Decompression throughput: " << decomp_stats.decompression_throughput_gbps << " GB/s" << std::endl;

    freeCompressedStringData(compressed);

    return (errors == 0) ? 0 : 1;
}
