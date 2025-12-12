/**
 * L3 String Compression Test
 *
 * Tests for GPU-accelerated string compression using learned models.
 * Verifies:
 * - String-to-integer encoding/decoding
 * - Compression correctness (lossless)
 * - Performance benchmarks
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#include "L3_string_codec.hpp"

// ============================================================================
// Test Data Generators
// ============================================================================

/**
 * Generate email-like strings (short version for 64-bit encoding)
 * Format: user@dom (max 9 chars for 7-bit charset)
 */
std::vector<std::string> generateEmailStrings(int count) {
    std::vector<std::string> emails;
    emails.reserve(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> name_len_dist(2, 4);
    std::uniform_int_distribution<int> char_dist(0, 25);

    // Short domains to fit in 64 bits
    const std::vector<std::string> domains = {"@ab", "@cd", "@ef"};

    for (int i = 0; i < count; i++) {
        std::string email;
        int name_len = name_len_dist(rng);

        // Generate short username
        for (int j = 0; j < name_len; j++) {
            email += static_cast<char>('a' + char_dist(rng));
        }

        email += domains[i % domains.size()];
        emails.push_back(email);
    }

    // Sort for better compression (sequential correlation)
    std::sort(emails.begin(), emails.end());

    return emails;
}

/**
 * Generate hex strings (10 chars max for 6-bit charset = 60 bits)
 */
std::vector<std::string> generateHexStrings(int count, int length = 10) {
    std::vector<std::string> hexes;
    hexes.reserve(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> hex_dist(0, 15);

    const char hex_chars[] = "0123456789abcdef";

    for (int i = 0; i < count; i++) {
        std::string hex;
        for (int j = 0; j < length; j++) {
            hex += hex_chars[hex_dist(rng)];
        }
        hexes.push_back(hex);
    }

    std::sort(hexes.begin(), hexes.end());

    return hexes;
}

/**
 * Generate word-like strings (dictionary words)
 */
std::vector<std::string> generateWordStrings(int count) {
    std::vector<std::string> words;
    words.reserve(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> len_dist(3, 10);
    std::uniform_int_distribution<int> char_dist(0, 25);

    for (int i = 0; i < count; i++) {
        std::string word;
        int len = len_dist(rng);
        for (int j = 0; j < len; j++) {
            word += static_cast<char>('a' + char_dist(rng));
        }
        words.push_back(word);
    }

    std::sort(words.begin(), words.end());

    return words;
}

/**
 * Generate URL-like strings with long common prefix
 * Variable part max 9 chars for 7-bit charset
 */
std::vector<std::string> generateUrlStrings(int count) {
    std::vector<std::string> urls;
    urls.reserve(count);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> path_len_dist(5, 8);
    std::uniform_int_distribution<int> char_dist(0, 35);

    // Long common prefix
    const std::string prefix = "https://www.example.com/api/v1/";

    const char path_chars[] = "abcdefghijklmnopqrstuvwxyz0123456789";

    for (int i = 0; i < count; i++) {
        std::string url = prefix;
        int path_len = path_len_dist(rng);
        for (int j = 0; j < path_len; j++) {
            url += path_chars[char_dist(rng) % 36];
        }
        urls.push_back(url);
    }

    std::sort(urls.begin(), urls.end());

    return urls;
}

// ============================================================================
// Test Functions
// ============================================================================

/**
 * Test CPU string encoding/decoding
 */
bool testCpuStringEncoding() {
    std::cout << "\n=== Testing CPU String Encoding/Decoding ===" << std::endl;

    bool all_passed = true;

    // Test 1: ASCII 256-base encoding
    {
        std::string test = "hello";
        uint64_t encoded = stringToUint64_ASCII256(test);
        std::string decoded = uint64ToString_ASCII256(encoded, 5);

        bool passed = (test == decoded);
        std::cout << "ASCII256 encoding: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Original: " << test << ", Decoded: " << decoded << std::endl;
        }
        all_passed &= passed;
    }

    // Test 2: Subset shift encoding (lowercase only)
    {
        std::string test = "hello";
        int min_char = 'a';
        int shift_bits = 5;  // 2^5 = 32, enough for 26 letters

        uint64_t encoded = stringToUint64_Shift(test, min_char, shift_bits);
        std::string decoded = uint64ToString_Shift(encoded, 5, min_char, shift_bits);

        bool passed = (test == decoded);
        std::cout << "Subset shift encoding: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Original: " << test << ", Decoded: " << decoded << std::endl;
        }
        all_passed &= passed;
    }

    // Test 3: 128-bit encoding for longer strings
    {
        std::string test = "helloworldtest";  // 14 chars
        int min_char = 'a';
        int shift_bits = 5;

        uint128_gpu encoded = stringToUint128_Shift(test, min_char, shift_bits);
        std::string decoded = uint128ToString_Shift(encoded, 14, min_char, shift_bits);

        bool passed = (test == decoded);
        std::cout << "128-bit encoding: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Original: " << test << ", Decoded: " << decoded << std::endl;
        }
        all_passed &= passed;
    }

    // Test 6: 256-bit encoding for very long strings
    {
        std::string test = "thisisaverylongstringfortesting";  // 31 chars
        int min_char = 'a';
        int shift_bits = 5;

        uint256_gpu encoded = stringToUint256_Shift(test, min_char, shift_bits);
        std::string decoded = uint256ToString_Shift(encoded, 31, min_char, shift_bits);

        bool passed = (test == decoded);
        std::cout << "256-bit encoding: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Original: " << test << ", Decoded: " << decoded << std::endl;
        }
        all_passed &= passed;
    }

    // Test 7: Automatic encoding type selection
    {
        // Test cases: (max_length, shift_bits) -> expected_type
        struct TypeSelectionTest {
            int32_t max_len;
            int32_t shift_bits;
            StringEncodingType expected;
        };

        std::vector<TypeSelectionTest> tests = {
            {10, 5, STRING_ENCODING_64},   // 10 chars @ 5-bit = 50 bits -> 64-bit
            {12, 5, STRING_ENCODING_64},   // 12 chars @ 5-bit = 60 bits -> 64-bit
            {13, 5, STRING_ENCODING_128},  // 13 chars @ 5-bit = 65 bits -> 128-bit
            {20, 5, STRING_ENCODING_128},  // 20 chars @ 5-bit = 100 bits -> 128-bit
            {25, 5, STRING_ENCODING_128},  // 25 chars @ 5-bit = 125 bits -> 128-bit
            {26, 5, STRING_ENCODING_256},  // 26 chars @ 5-bit = 130 bits -> 256-bit
            {40, 5, STRING_ENCODING_256},  // 40 chars @ 5-bit = 200 bits -> 256-bit
            {51, 5, STRING_ENCODING_256},  // 51 chars @ 5-bit = 255 bits -> 256-bit
            {8, 8, STRING_ENCODING_64},    // 8 chars @ 8-bit = 64 bits -> 64-bit
            {9, 8, STRING_ENCODING_128},   // 9 chars @ 8-bit = 72 bits -> 128-bit
            {16, 8, STRING_ENCODING_128},  // 16 chars @ 8-bit = 128 bits -> 128-bit
            {17, 8, STRING_ENCODING_256},  // 17 chars @ 8-bit = 136 bits -> 256-bit
        };

        bool all_type_passed = true;
        for (const auto& t : tests) {
            StringEncodingType actual = selectEncodingType(t.max_len, t.shift_bits);
            if (actual != t.expected) {
                all_type_passed = false;
                std::cout << "  Type selection failed: len=" << t.max_len
                          << ", shift=" << t.shift_bits
                          << ", expected=" << getEncodingTypeName(t.expected)
                          << ", got=" << getEncodingTypeName(actual) << std::endl;
            }
        }

        std::cout << "Encoding type selection: " << (all_type_passed ? "PASSED" : "FAILED") << std::endl;
        all_passed &= all_type_passed;
    }

    // Test 8: autoConfigureWithEncodingType
    {
        std::vector<std::string> short_strings = {"abc", "def", "ghi"};
        auto [config1, type1] = autoConfigureWithEncodingType(short_strings);
        bool passed1 = (type1 == STRING_ENCODING_64);

        std::vector<std::string> medium_strings = {"abcdefghijklmnopqrs", "bcdefghijklmnopqrst"};  // 19 chars
        auto [config2, type2] = autoConfigureWithEncodingType(medium_strings);
        bool passed2 = (type2 == STRING_ENCODING_128);

        std::vector<std::string> long_strings = {"abcdefghijklmnopqrstuvwxyzabcdef", "bcdefghijklmnopqrstuvwxyzabcdefg"};  // 32 chars
        auto [config3, type3] = autoConfigureWithEncodingType(long_strings);
        bool passed3 = (type3 == STRING_ENCODING_256);

        bool all_auto_passed = passed1 && passed2 && passed3;
        std::cout << "Auto configure with encoding type: " << (all_auto_passed ? "PASSED" : "FAILED") << std::endl;
        if (!all_auto_passed) {
            std::cout << "  Short strings (3 chars): " << getEncodingTypeName(type1) << std::endl;
            std::cout << "  Medium strings (19 chars): " << getEncodingTypeName(type2) << std::endl;
            std::cout << "  Long strings (32 chars): " << getEncodingTypeName(type3) << std::endl;
        }
        all_passed &= all_auto_passed;
    }

    // Test 4: Common prefix extraction
    {
        std::vector<std::string> strings = {
            "https://www.example.com/page1",
            "https://www.example.com/page2",
            "https://www.example.com/about"
        };

        std::string prefix = findCommonPrefix(strings);
        bool passed = (prefix == "https://www.example.com/");
        std::cout << "Common prefix extraction: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Expected: https://www.example.com/, Got: " << prefix << std::endl;
        }
        all_passed &= passed;
    }

    // Test 5: Character set analysis
    {
        std::vector<std::string> strings = {"abc", "def", "xyz"};
        int32_t min_char, max_char;
        analyzeCharacterSet(strings, min_char, max_char);

        bool passed = (min_char == 'a' && max_char == 'z');
        std::cout << "Character set analysis: " << (passed ? "PASSED" : "FAILED") << std::endl;
        if (!passed) {
            std::cout << "  Expected: [a,z], Got: [" << (char)min_char << "," << (char)max_char << "]" << std::endl;
        }
        all_passed &= passed;
    }

    return all_passed;
}

/**
 * Test GPU compression and decompression
 */
bool testGpuCompression(const std::vector<std::string>& test_strings, const std::string& test_name) {
    std::cout << "\n=== Testing GPU Compression: " << test_name << " ===" << std::endl;
    std::cout << "Number of strings: " << test_strings.size() << std::endl;

    if (test_strings.empty()) {
        std::cout << "Empty input, skipping..." << std::endl;
        return true;
    }

    // Show sample strings
    std::cout << "Sample strings:" << std::endl;
    for (size_t i = 0; i < std::min(size_t(3), test_strings.size()); i++) {
        std::cout << "  [" << i << "] " << test_strings[i].substr(0, 50);
        if (test_strings[i].size() > 50) std::cout << "...";
        std::cout << std::endl;
    }

    // Auto-configure compression
    StringCompressionConfig config = autoConfigureStringCompression(test_strings);
    std::cout << "\nCompression config:" << std::endl;
    std::cout << "  Mode: " << config.mode << std::endl;
    std::cout << "  Character range: [" << config.min_char << ", " << config.max_char << "]" << std::endl;
    std::cout << "  Shift bits: " << config.shift_bits << std::endl;
    std::cout << "  Max string length: " << config.max_string_length << std::endl;
    std::cout << "  Common prefix: \"" << config.common_prefix << "\" (len=" << config.common_prefix.size() << ")" << std::endl;

    // Compress
    StringCompressionStats comp_stats;
    CompressedStringData* compressed = compressStrings(test_strings, config, &comp_stats);

    if (!compressed) {
        std::cout << "Compression failed!" << std::endl;
        return false;
    }

    printStringCompressionStats(comp_stats);

    // Decompress
    std::vector<std::string> decompressed;
    StringDecompressionStats decomp_stats;
    int num_decompressed = decompressStrings(compressed, config, decompressed, &decomp_stats);

    if (num_decompressed != static_cast<int>(test_strings.size())) {
        std::cout << "Decompression returned wrong count: " << num_decompressed
                  << " vs " << test_strings.size() << std::endl;
        freeCompressedStringData(compressed);
        return false;
    }

    printStringDecompressionStats(decomp_stats);

    // Verify correctness
    int errors = 0;
    for (size_t i = 0; i < test_strings.size(); i++) {
        if (test_strings[i] != decompressed[i]) {
            errors++;
            if (errors <= 5) {
                std::cout << "Mismatch at index " << i << ":" << std::endl;
                std::cout << "  Original:     \"" << test_strings[i] << "\"" << std::endl;
                std::cout << "  Decompressed: \"" << decompressed[i] << "\"" << std::endl;
            }
        }
    }

    bool passed = (errors == 0);
    std::cout << "\nVerification: " << (passed ? "PASSED" : "FAILED");
    if (errors > 0) {
        std::cout << " (" << errors << " mismatches)";
    }
    std::cout << std::endl;

    freeCompressedStringData(compressed);

    return passed;
}

/**
 * Benchmark string compression
 */
void benchmarkCompression(int num_strings) {
    std::cout << "\n=== String Compression Benchmark ===" << std::endl;
    std::cout << "Number of strings: " << num_strings << std::endl;

    struct BenchCase {
        std::string name;
        std::vector<std::string> data;
    };

    std::vector<BenchCase> cases = {
        {"Email addresses", generateEmailStrings(num_strings)},
        {"Hex strings (32 chars)", generateHexStrings(num_strings, 32)},
        {"Word strings", generateWordStrings(num_strings)},
        {"URL strings", generateUrlStrings(num_strings)}
    };

    std::cout << "\n" << std::setw(25) << "Dataset"
              << std::setw(12) << "Ratio"
              << std::setw(15) << "Comp (GB/s)"
              << std::setw(15) << "Decomp (GB/s)"
              << std::setw(12) << "Delta bits"
              << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    for (auto& tc : cases) {
        StringCompressionConfig config = autoConfigureStringCompression(tc.data);
        StringCompressionStats comp_stats;
        StringDecompressionStats decomp_stats;

        // Warm up
        CompressedStringData* warm = compressStrings(tc.data, config, nullptr);
        freeCompressedStringData(warm);

        // Timed compression
        CompressedStringData* compressed = compressStrings(tc.data, config, &comp_stats);

        // Timed decompression
        std::vector<std::string> output;
        decompressStrings(compressed, config, output, &decomp_stats);

        std::cout << std::setw(25) << tc.name
                  << std::setw(12) << std::fixed << std::setprecision(2) << comp_stats.compression_ratio
                  << std::setw(15) << std::fixed << std::setprecision(2) << comp_stats.compression_throughput_gbps
                  << std::setw(15) << std::fixed << std::setprecision(2) << decomp_stats.decompression_throughput_gbps
                  << std::setw(12) << comp_stats.avg_delta_bits
                  << std::endl;

        freeCompressedStringData(compressed);
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "L3 String Compression Test Suite" << std::endl;
    std::cout << "=================================" << std::endl;

    // Check CUDA device
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Using GPU: " << props.name << std::endl;

    bool all_passed = true;

    // Test 1: CPU encoding/decoding
    all_passed &= testCpuStringEncoding();

    // Test 2: GPU compression with different string types
    int test_size = 10000;
    if (argc > 1) {
        test_size = std::atoi(argv[1]);
    }

    all_passed &= testGpuCompression(generateEmailStrings(test_size), "Email strings");
    all_passed &= testGpuCompression(generateHexStrings(test_size, 10), "Hex strings (10 chars)");
    all_passed &= testGpuCompression(generateWordStrings(test_size), "Word strings");
    all_passed &= testGpuCompression(generateUrlStrings(test_size), "URL strings");

    // Benchmark
    if (test_size >= 10000) {
        benchmarkCompression(test_size);
    }

    // Summary
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "All tests " << (all_passed ? "PASSED" : "FAILED") << std::endl;

    return all_passed ? 0 : 1;
}
